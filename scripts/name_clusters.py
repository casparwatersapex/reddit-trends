from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


def parse_sample_sizes(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_prompt(topic_id: int, examples: list[str]) -> str:
    joined = "\n---\n".join(examples)
    return (
        "You are labeling a topic cluster from Reddit posts.\n"
        f"Topic id: {topic_id}\n"
        "Provide a concise topic label (5-8 words) and a 1-2 sentence summary.\n"
        "Examples:\n"
        f"{joined}\n"
        "Return JSON with keys: label, summary."
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Name clusters with local/OpenAI.")
    parser.add_argument("embeddings_path", help="Parquet with post_id/text/embedding.")
    parser.add_argument("clusters_path", help="Parquet with post_id/topic assignments.")
    parser.add_argument(
        "--output-path",
        default="data/cluster_labels.csv",
        help="CSV output path.",
    )
    parser.add_argument("--sample-sizes", default="10,30,50")
    parser.add_argument(
        "--backends",
        default="local,openai",
        help="Comma-separated list: local,openai",
    )
    parser.add_argument("--local-model", default="llama3.1:8b")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--topic-limit", type=int, default=None)
    parser.add_argument("--max-chars-per-example", type=int, default=600)
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=None,
        help="Limit to the most frequent N topics (excluding -1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    openai_key = os.environ.get("OPENAI_API_KEY")

    embeddings_df = pd.read_parquet(args.embeddings_path)
    clusters_df = pd.read_parquet(args.clusters_path)
    merged = clusters_df.merge(embeddings_df, on="post_id", how="inner")

    embeddings = np.vstack(merged["embedding"].values)
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    topic_counts = (
        merged.loc[merged["topic"] != -1, "topic"].value_counts().sort_values(ascending=False)
    )
    topics = topic_counts.index.tolist()
    if args.top_n_topics:
        topics = topics[: args.top_n_topics]
    if args.topic_limit:
        topics = topics[: args.topic_limit]

    rows: list[dict[str, object]] = []
    total_tasks = len(topics) * len(sample_sizes) * len(backends)
    tasks_done = 0
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, int, str]] = set()
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        for _, row in existing.iterrows():
            seen.add((int(row["topic"]), int(row["sample_size"]), str(row["backend"])))
    for topic_id in topics:
        for sample_size in sample_sizes:
            examples = select_examples(
                merged,
                embeddings,
                topic_id,
                sample_size,
                max_chars=args.max_chars_per_example,
            )
            if not examples:
                continue
            prompt = build_prompt(topic_id, examples)
            for backend in backends:
                if (topic_id, sample_size, backend) in seen:
                    tasks_done += 1
                    print(
                        f"Skipped {tasks_done}/{total_tasks} "
                        f"(topic {topic_id}, n={sample_size}, {backend})"
                    )
                    continue
                if backend == "local":
                    result = call_ollama(args.local_model, prompt, args.timeout_s)
                elif backend == "openai":
                    if not openai_key:
                        raise ValueError("OPENAI_API_KEY is required for OpenAI labeling.")
                    result = call_openai(args.openai_model, prompt, openai_key, args.timeout_s)
                else:
                    raise ValueError(f"Unknown backend: {backend}")

                row = {
                    "topic": topic_id,
                    "sample_size": sample_size,
                    "backend": backend,
                    "model": args.local_model if backend == "local" else args.openai_model,
                    "label": result.get("label", ""),
                    "summary": result.get("summary", ""),
                }
                rows.append(row)
                pd.DataFrame([row]).to_csv(
                    output_path, mode="a", header=not output_path.exists(), index=False
                )
                tasks_done += 1
                print(
                    f"Completed {tasks_done}/{total_tasks} "
                    f"(topic {topic_id}, n={sample_size}, {backend})"
                )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        print(out_df.to_string(index=False))
    print(f"Wrote labels: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
