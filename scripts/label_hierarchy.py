from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import AgglomerativeClustering
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


def build_prompt(label: str, examples: list[str], level: str) -> str:
    joined = "\n---\n".join(examples)
    return (
        f"You are labeling a {level} topic group.\n"
        f"Existing label/keywords: {label}\n"
        "Provide a concise label (5-8 words) and a 1-2 sentence summary.\n"
        "Examples:\n"
        f"{joined}\n"
        "Return JSON with keys: label, summary."
    )


def call_ollama(model: str, prompt: str, timeout_s: int) -> dict[str, str]:
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
        return {"label": text[:120], "summary": text}


def call_openai(model: str, prompt: str, api_key: str, timeout_s: int) -> dict[str, str]:
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
        return {"label": content[:120], "summary": content}


def select_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_ids: list[int],
    sample_size: int,
    max_chars: int,
) -> list[str]:
    topic_mask = df["topic"].isin(topic_ids)
    topic_embeddings = embeddings[topic_mask]
    if len(topic_embeddings) == 0:
        return []
    centroid = topic_embeddings.mean(axis=0, keepdims=True)
    sims = cosine_similarity(topic_embeddings, centroid).reshape(-1)
    order = np.argsort(-sims)[:sample_size]
    examples = df.loc[topic_mask, "text"].iloc[order].tolist()
    return [ex[:max_chars] for ex in examples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and label L1/L2 hierarchy.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("clusters_path", help="Parquet with post_id/topic assignments.")
    parser.add_argument("--output-path", default="data/hierarchy_labels.csv")
    parser.add_argument("--l1-count", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-chars-per-example", type=int, default=400)
    parser.add_argument("--l1-label-limit", type=int, default=10)
    parser.add_argument("--l2-label-limit", type=int, default=20)
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument("--local-model", default="llama3.1:8b")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--timeout-s", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    openai_key = os.environ.get("OPENAI_API_KEY")

    embeddings_df = pd.read_parquet(args.embeddings_path)
    clusters_df = pd.read_parquet(args.clusters_path)
    merged = clusters_df.merge(embeddings_df, on="post_id", how="inner")
    embeddings = np.vstack(merged["embedding"].values)

    topic_counts = (
        merged.loc[merged["topic"] != -1, "topic"].value_counts().sort_values(ascending=False)
    )
    topic_ids = topic_counts.index.tolist()
    topic_embeddings = []
    for topic_id in topic_ids:
        topic_embeddings.append(embeddings[merged["topic"] == topic_id].mean(axis=0))
    topic_embeddings = np.vstack(topic_embeddings)

    l1_model = AgglomerativeClustering(n_clusters=args.l1_count)
    l1_labels = l1_model.fit_predict(topic_embeddings)

    topic_to_l1 = {
        int(topic_id): int(l1_id) for topic_id, l1_id in zip(topic_ids, l1_labels, strict=False)
    }
    l1_sizes = (
        merged.loc[merged["topic"] != -1, "topic"]
        .map(topic_to_l1)
        .value_counts()
        .sort_values(ascending=False)
    )
    top_l1_ids = l1_sizes.index.tolist()[: args.l1_label_limit]
    top_l2_topics = topic_ids[: args.l2_label_limit]

    rows: list[dict[str, object]] = []
    for l1_id in top_l1_ids:
        l1_topics = [t for t in top_l2_topics if topic_to_l1.get(t) == l1_id]
        if not l1_topics:
            continue
        examples = select_examples(
            merged,
            embeddings,
            l1_topics,
            sample_size=args.sample_size,
            max_chars=args.max_chars_per_example,
        )
        prompt = build_prompt(f"L1 {l1_id}", examples, level="L1")
        if args.backend == "local":
            result = call_ollama(args.local_model, prompt, args.timeout_s)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI labeling.")
            result = call_openai(args.openai_model, prompt, openai_key, args.timeout_s)
        rows.append(
            {
                "level": "L1",
                "l1_id": int(l1_id),
                "l2_topic": None,
                "size": int(l1_sizes.get(l1_id, 0)),
                "label": result.get("label", ""),
                "summary": result.get("summary", ""),
            }
        )

    for topic_id in top_l2_topics:
        l1_id = topic_to_l1.get(topic_id)
        examples = select_examples(
            merged,
            embeddings,
            [topic_id],
            sample_size=args.sample_size,
            max_chars=args.max_chars_per_example,
        )
        prompt = build_prompt(f"Topic {topic_id}", examples, level="L2")
        if args.backend == "local":
            result = call_ollama(args.local_model, prompt, args.timeout_s)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI labeling.")
            result = call_openai(args.openai_model, prompt, openai_key, args.timeout_s)
        rows.append(
            {
                "level": "L2",
                "l1_id": int(l1_id) if l1_id is not None else None,
                "l2_topic": int(topic_id),
                "size": int(topic_counts.get(topic_id, 0)),
                "label": result.get("label", ""),
                "summary": result.get("summary", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote hierarchy labels: {output_path} (rows={len(out_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
