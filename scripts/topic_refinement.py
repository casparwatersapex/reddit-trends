from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


def call_ollama(model: str, prompt: str, timeout_s: int) -> dict[str, object]:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("response", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def call_openai(model: str, prompt: str, api_key: str, timeout_s: int) -> dict[str, object]:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}


def compute_topic_embeddings(
    embeddings: np.ndarray, topics: np.ndarray, topic_ids: list[int]
) -> np.ndarray:
    topic_vectors = []
    for topic_id in topic_ids:
        topic_mask = topics == topic_id
        topic_vectors.append(embeddings[topic_mask].mean(axis=0))
    return np.vstack(topic_vectors)


def compute_top_n_assignments(
    embeddings: np.ndarray, topic_embeddings: np.ndarray, topic_ids: list[int], top_n: int
) -> tuple[list[list[int]], list[list[float]]]:
    sims = cosine_similarity(embeddings, topic_embeddings)
    order = np.argsort(-sims, axis=1)[:, :top_n]
    top_topics: list[list[int]] = []
    top_scores: list[list[float]] = []
    for row_idx, indices in enumerate(order):
        top_topics.append([topic_ids[i] for i in indices])
        top_scores.append([float(sims[row_idx, i]) for i in indices])
    return top_topics, top_scores


def select_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_id: int,
    sample_size: int,
    max_chars: int,
) -> list[str]:
    topic_mask = df["topic"] == topic_id
    topic_embeddings = embeddings[topic_mask]
    if len(topic_embeddings) == 0:
        return []
    centroid = topic_embeddings.mean(axis=0, keepdims=True)
    sims = cosine_similarity(topic_embeddings, centroid).reshape(-1)
    order = np.argsort(-sims)[:sample_size]
    examples = df.loc[topic_mask, "text"].iloc[order].tolist()
    return [ex[:max_chars] for ex in examples]


def build_split_prompt(topic_id: int, examples: list[str]) -> str:
    joined = "\n---\n".join(examples)
    return (
        "You are reviewing a topic cluster from Reddit posts.\n"
        f"Topic id: {topic_id}\n"
        "Decide whether this topic should be split into two themes.\n"
        "Return JSON with keys: action ('keep' or 'split'), reason.\n"
        "Examples:\n"
        f"{joined}\n"
    )


def build_merge_prompt(
    topic_a: int,
    examples_a: list[str],
    topic_b: int,
    examples_b: list[str],
) -> str:
    joined_a = "\n---\n".join(examples_a)
    joined_b = "\n---\n".join(examples_b)
    return (
        "You are comparing two topic clusters from Reddit posts.\n"
        f"Topic A: {topic_a}\n"
        f"{joined_a}\n"
        "Topic B:\n"
        f"{joined_b}\n"
        "Return JSON with keys: merge (true/false), reason.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven topic refinement.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("clusters_path", help="Parquet with post_id/topic assignments.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--report-path", default="data/topic_refinement.csv")
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument("--local-model", default="llama3.1:8b")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-chars-per-example", type=int, default=400)
    parser.add_argument("--split-min-size", type=int, default=800)
    parser.add_argument("--split-limit", type=int, default=20)
    parser.add_argument("--merge-similarity", type=float, default=0.9)
    parser.add_argument("--merge-limit", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    openai_key = os.environ.get("OPENAI_API_KEY")

    embeddings_df = pd.read_parquet(args.embeddings_path)
    clusters_df = pd.read_parquet(args.clusters_path)
    merged = clusters_df.merge(embeddings_df, on="post_id", how="inner")
    embeddings = np.vstack(merged["embedding"].values)
    topics = merged["topic"].to_numpy()

    topic_counts = merged.loc[merged["topic"] != -1, "topic"].value_counts()
    candidate_splits = [
        int(topic_id)
        for topic_id, count in topic_counts.items()
        if int(count) >= args.split_min_size
    ][: args.split_limit]

    report_rows: list[dict[str, object]] = []
    next_topic_id = int(topic_counts.index.max()) + 1 if len(topic_counts) else 0

    for topic_id in candidate_splits:
        examples = select_examples(
            merged,
            embeddings,
            topic_id,
            args.sample_size,
            args.max_chars_per_example,
        )
        if not examples:
            continue
        prompt = build_split_prompt(topic_id, examples)
        if args.backend == "local":
            result = call_ollama(args.local_model, prompt, args.timeout_s)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI refinement.")
            result = call_openai(args.openai_model, prompt, openai_key, args.timeout_s)
        action = str(result.get("action", "keep")).lower()
        report_rows.append(
            {
                "action": "split_check",
                "topic": topic_id,
                "decision": action,
                "reason": result.get("reason", ""),
            }
        )
        if action != "split":
            continue
        topic_mask = topics == topic_id
        if topic_mask.sum() < 4:
            continue
        kmeans = KMeans(n_clusters=2, random_state=7, n_init="auto")
        split_labels = kmeans.fit_predict(embeddings[topic_mask])
        for split_id in [0, 1]:
            split_mask = split_labels == split_id
            topics[np.where(topic_mask)[0][split_mask]] = next_topic_id
            report_rows.append(
                {
                    "action": "split_apply",
                    "topic": topic_id,
                    "new_topic": next_topic_id,
                }
            )
            next_topic_id += 1

    merged["topic"] = topics
    topic_ids = sorted([t for t in set(topics) if t != -1])
    topic_embeddings = compute_topic_embeddings(embeddings, topics, topic_ids)

    similarity = cosine_similarity(topic_embeddings)
    candidate_pairs: list[tuple[int, int, float]] = []
    for i, topic_a in enumerate(topic_ids):
        for j in range(i + 1, len(topic_ids)):
            topic_b = topic_ids[j]
            sim = float(similarity[i, j])
            if sim >= args.merge_similarity:
                candidate_pairs.append((topic_a, topic_b, sim))

    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    candidate_pairs = candidate_pairs[: args.merge_limit]

    parent = {topic_id: topic_id for topic_id in topic_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for topic_a, topic_b, sim in candidate_pairs:
        if find(topic_a) == find(topic_b):
            continue
        examples_a = select_examples(
            merged,
            embeddings,
            topic_a,
            args.sample_size,
            args.max_chars_per_example,
        )
        examples_b = select_examples(
            merged,
            embeddings,
            topic_b,
            args.sample_size,
            args.max_chars_per_example,
        )
        if not examples_a or not examples_b:
            continue
        prompt = build_merge_prompt(topic_a, examples_a, topic_b, examples_b)
        if args.backend == "local":
            result = call_ollama(args.local_model, prompt, args.timeout_s)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI refinement.")
            result = call_openai(args.openai_model, prompt, openai_key, args.timeout_s)
        decision = bool(result.get("merge", False))
        report_rows.append(
            {
                "action": "merge_check",
                "topic_a": topic_a,
                "topic_b": topic_b,
                "similarity": sim,
                "decision": decision,
                "reason": result.get("reason", ""),
            }
        )
        if decision:
            union(topic_a, topic_b)

    if candidate_pairs:
        merged["topic"] = merged["topic"].map(lambda t: find(int(t)) if t != -1 else -1)
        non_noise = merged.loc[merged["topic"] != -1, "topic"]
        remapped, uniques = pd.factorize(non_noise)
        merged.loc[merged["topic"] != -1, "topic"] = remapped
        merged["topic"] = merged["topic"].astype(int)
        report_rows.append({"action": "merge_apply", "notes": "Applied union-find merges."})

    topics = merged["topic"].to_numpy()
    topic_ids = sorted([t for t in set(topics) if t != -1])
    if topic_ids:
        topic_embeddings = compute_topic_embeddings(embeddings, topics, topic_ids)
        top_topics, top_scores = compute_top_n_assignments(
            embeddings, topic_embeddings, topic_ids, args.top_n
        )
    else:
        top_topics = [[] for _ in range(len(merged))]
        top_scores = [[] for _ in range(len(merged))]

    out_df = merged[["post_id", "topic"]].copy()
    out_df["top_n_topics"] = top_topics
    out_df["top_n_scores"] = top_scores
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"Wrote refined clusters: {output_path}")
    print(f"Wrote refinement report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
