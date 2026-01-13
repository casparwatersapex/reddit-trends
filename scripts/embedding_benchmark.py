from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests


def build_text(df: pd.DataFrame, title_col: str, body_col: str) -> pd.Series:
    title = df[title_col].fillna("").astype(str)
    body = df[body_col].fillna("").astype(str)
    return title + "\n\n" + body


def estimate_tokens(text: pd.Series) -> float:
    # Simple heuristic: 1 token ~= 4 characters in English.
    return float(text.str.len().sum()) / 4.0


def sample_frame(df: pd.DataFrame, sample_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_rows:
        return df
    return df.sample(n=sample_rows, random_state=seed)


def measure_local_rows_per_sec(text: pd.Series, model_name: str) -> tuple[float | None, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None, "sentence-transformers not installed"

    model = SentenceTransformer(model_name)
    start = time.time()
    model.encode(text.tolist(), show_progress_bar=False)
    elapsed = max(time.time() - start, 1e-6)
    return len(text) / elapsed, None


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


def measure_openai_tokens_per_sec(
    text: pd.Series,
    model_name: str,
    api_key: str,
    batch_size: int,
    timeout_s: int,
) -> tuple[float, float]:
    total_tokens = 0.0
    start = time.time()
    for i in range(0, len(text), batch_size):
        batch = text.iloc[i : i + batch_size].tolist()
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "input": batch},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        batch_tokens = usage.get("total_tokens")
        if batch_tokens is None:
            batch_tokens = estimate_tokens(pd.Series(batch))
        total_tokens += float(batch_tokens)
    elapsed = max(time.time() - start, 1e-6)
    return total_tokens / elapsed, total_tokens


def build_benchmark_rows(
    df: pd.DataFrame,
    title_col: str,
    body_col: str,
    sample_rows: int,
    seed: int,
    local_model: str,
    local_rows_per_sec: float | None,
    openai_model: str,
    openai_tokens_per_sec: float,
    openai_cost_per_million: float,
    openai_live: bool,
    openai_api_key: str | None,
    openai_batch_size: int,
    openai_timeout_s: int,
) -> list[dict[str, object]]:
    sample_df = sample_frame(df, sample_rows, seed)
    text = build_text(sample_df, title_col, body_col)
    sample_tokens = estimate_tokens(text)
    total_tokens = sample_tokens * (len(df) / max(len(sample_df), 1))

    rows: list[dict[str, object]] = []

    local_notes = []
    measured_rows_per_sec = None
    if local_rows_per_sec is None:
        measured_rows_per_sec, err = measure_local_rows_per_sec(text, local_model)
        if err:
            local_notes.append(err)
    rows_per_sec = local_rows_per_sec or measured_rows_per_sec
    if rows_per_sec:
        est_hours = (len(df) / rows_per_sec) / 3600.0
        local_notes.append("estimated from sample run")
    else:
        est_hours = None
        local_notes.append("set --local-rows-per-sec to estimate time")

    rows.append(
        {
            "option": "local",
            "model": local_model,
            "rows_total": len(df),
            "sample_rows": len(sample_df),
            "est_total_tokens": round(total_tokens),
            "est_time_hours": None if est_hours is None else round(est_hours, 2),
            "est_cost_usd": 0.0,
            "notes": "; ".join(local_notes),
        }
    )

    openai_notes = []
    tokens_per_sec = openai_tokens_per_sec
    if openai_live:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for live OpenAI sampling.")
        tokens_per_sec, sample_token_count = measure_openai_tokens_per_sec(
            text=text,
            model_name=openai_model,
            api_key=openai_api_key,
            batch_size=openai_batch_size,
            timeout_s=openai_timeout_s,
        )
        openai_notes.append(
            f"sampled {int(sample_token_count)} tokens at {tokens_per_sec:.0f} tokens/sec"
        )
    else:
        openai_notes.append(
            f"assumes {openai_tokens_per_sec:.0f} tokens/sec and "
            f"${openai_cost_per_million:.2f}/1M tokens"
        )

    openai_hours = (total_tokens / max(tokens_per_sec, 1.0)) / 3600.0
    openai_cost = (total_tokens / 1_000_000.0) * openai_cost_per_million
    rows.append(
        {
            "option": "openai",
            "model": openai_model,
            "rows_total": len(df),
            "sample_rows": len(sample_df),
            "est_total_tokens": round(total_tokens),
            "est_time_hours": round(openai_hours, 2),
            "est_cost_usd": round(openai_cost, 2),
            "notes": "; ".join(openai_notes),
        }
    )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding options with estimated time and cost."
    )
    parser.add_argument("input_path", help="Path to canonical Parquet input.")
    parser.add_argument(
        "--output-path",
        default="data/embedding_benchmark.csv",
        help="CSV output path for the benchmark table.",
    )
    parser.add_argument("--title-col", default="title", help="Title column name.")
    parser.add_argument("--body-col", default="body", help="Body column name.")
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--local-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local model name for timing runs.",
    )
    parser.add_argument(
        "--local-rows-per-sec",
        type=float,
        default=None,
        help="Override local rows/sec to estimate time.",
    )
    parser.add_argument(
        "--openai-model", default="text-embedding-3-small", help="OpenAI model name."
    )
    parser.add_argument(
        "--openai-tokens-per-sec",
        type=float,
        default=4000.0,
        help="Assumed OpenAI throughput for time estimates.",
    )
    parser.add_argument(
        "--openai-cost-per-million",
        type=float,
        default=0.02,
        help="Assumed OpenAI cost per 1M tokens (USD).",
    )
    parser.add_argument(
        "--openai-live",
        action="store_true",
        help="Run a live OpenAI sample to estimate throughput.",
    )
    parser.add_argument("--openai-batch-size", type=int, default=128)
    parser.add_argument("--openai-timeout-s", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(Path(".env"))
    df = pd.read_parquet(args.input_path)
    rows = build_benchmark_rows(
        df=df,
        title_col=args.title_col,
        body_col=args.body_col,
        sample_rows=args.sample_rows,
        seed=args.seed,
        local_model=args.local_model,
        local_rows_per_sec=args.local_rows_per_sec,
        openai_model=args.openai_model,
        openai_tokens_per_sec=args.openai_tokens_per_sec,
        openai_cost_per_million=args.openai_cost_per_million,
        openai_live=args.openai_live,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_batch_size=args.openai_batch_size,
        openai_timeout_s=args.openai_timeout_s,
    )
    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(out_df.to_string(index=False))
    print(f"Wrote benchmark table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
