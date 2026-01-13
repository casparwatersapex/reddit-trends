from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


def build_text(df: pd.DataFrame, title_col: str, body_col: str) -> pd.Series:
    title = df[title_col].fillna("").astype(str)
    body = df[body_col].fillna("").astype(str)
    return title + "\n\n" + body


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


def embed_local(text: list[str], model_name: str, batch_size: int) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(text, batch_size=batch_size, show_progress_bar=True).tolist()


def embed_openai(
    text: list[str],
    model_name: str,
    api_key: str,
    batch_size: int,
    timeout_s: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(text), batch_size):
        batch = text[i : i + batch_size]
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "input": batch},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings.extend([item["embedding"] for item in payload["data"]])
    return embeddings


def append_embeddings_stream(writer: pq.ParquetWriter, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    writer.write_table(table.cast(writer.schema, safe=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute embeddings for canonical data.")
    parser.add_argument("input_path", help="Path to canonical Parquet input.")
    parser.add_argument("--output-path", required=True, help="Parquet output path.")
    parser.add_argument("--id-col", default="post_id")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--body-col", default="body")
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument(
        "--stream-write",
        action="store_true",
        help="Write embeddings incrementally to avoid large memory use.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    df = pd.read_parquet(args.input_path)
    text = build_text(df, args.title_col, args.body_col).tolist()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    start = time.time()
    total_batches = math.ceil(len(text) / args.batch_size)
    batches_done = 0
    if args.backend == "local" and not args.stream_write:
        embeddings = embed_local(text, args.model, args.batch_size)
        out_df = pd.DataFrame(
            {
                "post_id": df[args.id_col],
                "text": text,
                "embedding": embeddings,
            }
        )
        out_df.to_parquet(output_path, index=False)
    else:
        writer: pq.ParquetWriter | None = None
        for i in range(0, len(text), args.batch_size):
            batch_start = time.time()
            batch_text = text[i : i + args.batch_size]
            batch_ids = df[args.id_col].iloc[i : i + args.batch_size].tolist()
            if args.backend == "local":
                batch_embeddings = embed_local(batch_text, args.model, args.batch_size)
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
                batch_embeddings = embed_openai(
                    batch_text,
                    model_name=args.model,
                    api_key=api_key,
                    batch_size=args.batch_size,
                    timeout_s=args.timeout_s,
                )
            rows = [
                {"post_id": pid, "text": txt, "embedding": emb}
                for pid, txt, emb in zip(batch_ids, batch_text, batch_embeddings, strict=False)
            ]
            if writer is None:
                table = pa.Table.from_pylist(rows)
                schema = pa.schema([field.with_nullable(True) for field in table.schema])
                writer = pq.ParquetWriter(output_path, schema)
            append_embeddings_stream(writer, rows)
            batches_done += 1
            batch_elapsed = max(time.time() - batch_start, 1e-6)
            overall_elapsed = max(time.time() - start, 1e-6)
            avg_batch = overall_elapsed / batches_done
            eta = avg_batch * (total_batches - batches_done)
            print(
                f"Batch {batches_done}/{total_batches} "
                f"({batch_elapsed:.1f}s, avg {avg_batch:.1f}s, ETA {eta / 60:.1f}m)"
            )
        if writer is not None:
            writer.close()
    elapsed = time.time() - start
    print(f"Wrote embeddings: {output_path} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
