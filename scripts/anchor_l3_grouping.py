from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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


def call_openai_chat(model: str, prompt: str, api_key: str, timeout_s: int) -> dict[str, object]:
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


def call_openai_embeddings(model: str, inputs: list[str], api_key: str) -> list[list[float]]:
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": inputs},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def clean_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("```"):
        cleaned = cleaned.strip("`").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned.strip("'\" ")


def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("```"):
        cleaned = cleaned.strip("`").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_anchors(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        anchors = payload.get("anchors")
        if isinstance(anchors, list):
            return anchors
        raw = payload.get("raw")
        if isinstance(raw, str):
            try:
                parsed = json.loads(_extract_json_block(raw))
                if isinstance(parsed, dict) and isinstance(parsed.get("anchors"), list):
                    return parsed["anchors"]
            except json.JSONDecodeError:
                return []
    if isinstance(payload, str):
        try:
            parsed = json.loads(_extract_json_block(payload))
            if isinstance(parsed, dict) and isinstance(parsed.get("anchors"), list):
                return parsed["anchors"]
        except json.JSONDecodeError:
            return []
    return []


def build_anchor_prompt(l2_label: str, examples: list[str], max_anchors: int) -> str:
    joined = "\n---\n".join(examples)
    return (
        "You are defining subtopics for a large Reddit topic cluster.\n"
        f"Parent L2 label: {l2_label}\n"
        f"Create {max_anchors} or fewer L3 anchors that capture distinct intents.\n"
        "Return JSON with key: anchors (list of objects with keys label, keywords).\n"
        "Each keywords value should be a list of 3-6 search phrases.\n"
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
    parser = argparse.ArgumentParser(description="LLM-guided L3 anchors + keyword grouping.")
    parser.add_argument("canonical_path", help="Canonical parquet with posts.")
    parser.add_argument("embeddings_path", help="Parquet with post embeddings.")
    parser.add_argument("clusters_path", help="Parquet with L2 post_id/topic assignments.")
    parser.add_argument("--output-clusters", required=True, help="Output L3 cluster parquet.")
    parser.add_argument("--output-anchors", default="data/l3_anchors.csv")
    parser.add_argument("--output-keywords", default="data/keywords_l3_anchor.csv")
    parser.add_argument("--output-labels", default="data/cluster_labels_l3_anchor.csv")
    parser.add_argument("--l2-threshold", type=int, default=500)
    parser.add_argument("--l2-limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-chars-per-example", type=int, default=400)
    parser.add_argument("--anchors-per-l2", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    parser.add_argument("--backend", choices=["local", "openai"], default="openai")
    parser.add_argument("--local-model", default="llama3.1:8b")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--l2-labels-path", default=None)
    return parser.parse_args()


def load_labels(path: Path | None) -> dict[int, str]:
    if not path or not path.exists():
        return {}
    df = pd.read_csv(path)
    labels: dict[int, str] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("topic")):
            label = clean_text(str(row.get("label", "")))
            if label:
                labels[int(row["topic"])] = label
    return labels


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    openai_key = os.environ.get("OPENAI_API_KEY")

    canonical_df = pd.read_parquet(args.canonical_path)
    embeddings_df = pd.read_parquet(args.embeddings_path)
    clusters_df = pd.read_parquet(args.clusters_path)

    merged = clusters_df.merge(embeddings_df, on="post_id", how="inner")
    merged = merged.merge(canonical_df, on="post_id", how="left")
    merged["score"] = pd.to_numeric(merged.get("score"), errors="coerce").fillna(0)

    l2_labels = load_labels(Path(args.l2_labels_path)) if args.l2_labels_path else {}
    l2_counts = merged.loc[merged["topic"] != -1, "topic"].value_counts()
    l2_topics = [int(t) for t, c in l2_counts.items() if int(c) >= args.l2_threshold]
    if args.l2_limit:
        l2_topics = l2_topics[: args.l2_limit]

    anchors_rows: list[dict[str, object]] = []
    keywords_rows: list[dict[str, object]] = []
    labels_rows: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    next_l3_id = 0

    for l2_topic in l2_topics:
        l2_subset = merged[merged["topic"] == l2_topic].copy()
        l2_label = l2_labels.get(l2_topic, f"L2 {l2_topic}")
        examples = select_examples(l2_subset, args.sample_size, args.max_chars_per_example)
        if not examples:
            continue
        prompt = build_anchor_prompt(l2_label, examples, args.anchors_per_l2)
        if args.backend == "local":
            payload = call_ollama(args.local_model, prompt, args.timeout_s)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI anchors.")
            payload = call_openai_chat(args.openai_model, prompt, openai_key, args.timeout_s)
        anchors = parse_anchors(payload)
        if not anchors:
            continue

        anchor_defs: list[tuple[int, str, list[str], str]] = []
        anchor_texts: list[str] = []
        for anchor in anchors:
            label = clean_text(str(anchor.get("label", "")))
            keywords = anchor.get("keywords", [])
            if not label:
                continue
            if not isinstance(keywords, list):
                keywords = []
            keywords = [clean_text(str(k)) for k in keywords if str(k).strip()]
            anchor_text = f"{label}. Keywords: {', '.join(keywords)}"
            anchor_defs.append((next_l3_id, label, keywords, anchor_text))
            anchor_texts.append(anchor_text)
            anchors_rows.append(
                {
                    "l2_topic": l2_topic,
                    "l3_topic": next_l3_id,
                    "l2_label": l2_label,
                    "l3_label": label,
                    "keywords": "; ".join(keywords),
                }
            )
            keywords_rows.append(
                {
                    "l2_topic": l2_topic,
                    "l3_topic": next_l3_id,
                    "l2_label": l2_label,
                    "l3_label": label,
                    "keywords": "; ".join(keywords),
                    "sample_size": args.sample_size,
                    "post_count": int(len(l2_subset)),
                }
            )
            labels_rows.append(
                {
                    "topic": next_l3_id,
                    "sample_size": args.sample_size,
                    "backend": args.backend,
                    "model": args.local_model if args.backend == "local" else args.openai_model,
                    "label": label,
                    "summary": "; ".join(keywords),
                }
            )
            next_l3_id += 1

        if not anchor_texts:
            continue
        if args.backend == "local":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            anchor_embeddings = model.encode(anchor_texts, normalize_embeddings=True)
        else:
            if not openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            anchor_embeddings = np.array(
                call_openai_embeddings(args.embedding_model, anchor_texts, openai_key)
            )

        post_embeddings = np.vstack(l2_subset["embedding"].values)
        sims = cosine_similarity(post_embeddings, anchor_embeddings)
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(best_idx)), best_idx]
        for row_idx, (anchor_idx, score) in enumerate(zip(best_idx, best_scores, strict=False)):
            l3_topic = anchor_defs[int(anchor_idx)][0]
            if float(score) < args.min_similarity:
                l3_topic = -1
            assignments.append(
                {
                    "post_id": l2_subset.iloc[row_idx]["post_id"],
                    "topic": l3_topic,
                    "parent_topic": l2_topic,
                    "top_n_topics": [a[0] for a in anchor_defs],
                    "top_n_scores": [float(s) for s in sims[row_idx]],
                }
            )

    clusters_out = pd.DataFrame(assignments)
    clusters_path = Path(args.output_clusters)
    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_out.to_parquet(clusters_path, index=False)

    anchors_out = pd.DataFrame(anchors_rows)
    anchors_path = Path(args.output_anchors)
    anchors_path.parent.mkdir(parents=True, exist_ok=True)
    anchors_out.to_csv(anchors_path, index=False)

    keywords_out = pd.DataFrame(keywords_rows)
    keywords_path = Path(args.output_keywords)
    keywords_path.parent.mkdir(parents=True, exist_ok=True)
    keywords_out.to_csv(keywords_path, index=False)

    labels_out = pd.DataFrame(labels_rows)
    labels_path = Path(args.output_labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_out.to_csv(labels_path, index=False)

    print(f"Wrote anchor L3 clusters: {clusters_path}")
    print(f"Wrote anchor definitions: {anchors_path}")
    print(f"Wrote anchor keywords: {keywords_path}")
    print(f"Wrote anchor labels: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
