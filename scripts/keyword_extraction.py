from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests


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


def clean_label(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("```"):
        cleaned = cleaned.strip("`").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned.strip("'\" ")


def parse_keywords(payload: object) -> list[str]:
    if isinstance(payload, dict):
        keywords = payload.get("keywords")
        if isinstance(keywords, list):
            return [str(k).strip() for k in keywords if str(k).strip()]
    if isinstance(payload, str):
        parts = [p.strip() for p in payload.split(",") if p.strip()]
        return parts[:12]
    return []


def build_prompt(
    l2_label: str,
    l3_label: str,
    examples: list[str],
) -> str:
    joined = "\n---\n".join(examples)
    return (
        "You are creating Google search keyword ideas from Reddit posts.\n"
        f"L2 theme: {l2_label}\n"
        f"L3 theme: {l3_label}\n"
        "Return JSON with key: keywords (list of 8-12 phrases).\n"
        "Examples:\n"
        f"{joined}\n"
    )


def select_examples(df: pd.DataFrame, sample_size: int, max_chars: int) -> list[str]:
    if df.empty:
        return []
    view = df.sort_values("score", ascending=False).head(sample_size)
    examples = []
    for _, row in view.iterrows():
        title = str(row.get("title", "")).strip()
        body = str(row.get("body", "")).strip()
        text = " - ".join([part for part in (title, body) if part])
        if not text:
            continue
        examples.append(text[:max_chars])
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract keyword ideas from L2/L3 topics.")
    parser.add_argument("canonical_path", help="Canonical parquet with posts and scores.")
    parser.add_argument("l3_clusters_path", help="Parquet with L3 topic assignments.")
    parser.add_argument("--output-path", default="data/keywords_l3.csv")
    parser.add_argument("--l3-threshold", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-chars-per-example", type=int, default=400)
    parser.add_argument("--l2-labels-path", default=None)
    parser.add_argument("--l3-labels-path", default=None)
    parser.add_argument("--backend", choices=["local", "openai"], default="openai")
    parser.add_argument("--local-model", default="llama3.1:8b")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--timeout-s", type=int, default=120)
    return parser.parse_args()


def load_labels(path: Path | None) -> dict[int, str]:
    if not path or not path.exists():
        return {}
    df = pd.read_csv(path)
    labels: dict[int, str] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("topic")):
            label = clean_label(str(row.get("label", "")))
            if label:
                labels[int(row["topic"])] = label
    return labels


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    openai_key = os.environ.get("OPENAI_API_KEY")

    canonical_df = pd.read_parquet(args.canonical_path)
    l3_df = pd.read_parquet(args.l3_clusters_path)
    merged = l3_df.merge(canonical_df, on="post_id", how="inner")
    l3_labels = load_labels(Path(args.l3_labels_path)) if args.l3_labels_path else {}
    l2_labels = load_labels(Path(args.l2_labels_path)) if args.l2_labels_path else {}

    l3_counts = merged.loc[merged["topic"] != -1, "topic"].value_counts()
    rows: list[dict[str, object]] = []
    for l3_topic, count in l3_counts.items():
        if int(count) < args.l3_threshold:
            continue
        subset = merged[merged["topic"] == l3_topic]
        l2_topic = subset["parent_topic"].mode().iloc[0]
        examples = select_examples(subset, args.sample_size, args.max_chars_per_example)
        if not examples:
            continue
        l2_label = l2_labels.get(int(l2_topic), f"L2 {int(l2_topic)}")
        l3_label = l3_labels.get(int(l3_topic), f"L3 {int(l3_topic)}")
        prompt = build_prompt(l2_label, l3_label, examples)
        if args.backend == "local":
            result = call_ollama(args.local_model, prompt, args.timeout_s)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI keywords.")
            result = call_openai(args.openai_model, prompt, openai_key, args.timeout_s)
        keywords = parse_keywords(result)
        rows.append(
            {
                "l2_topic": int(l2_topic),
                "l3_topic": int(l3_topic),
                "l2_label": l2_label,
                "l3_label": l3_label,
                "keywords": "; ".join(keywords),
                "sample_size": args.sample_size,
                "post_count": int(count),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote keyword ideas: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
