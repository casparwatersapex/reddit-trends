from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare clustering outputs.")
    parser.add_argument("left_path", help="Parquet output from run_clustering.py.")
    parser.add_argument("right_path", help="Parquet output from run_clustering.py.")
    parser.add_argument("--output-path", default="data/clustering_compare.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left = pd.read_parquet(args.left_path)
    right = pd.read_parquet(args.right_path)
    merged = left.merge(right, on="post_id", suffixes=("_left", "_right"))

    left_topics = merged["topic_left"].astype(int)
    right_topics = merged["topic_right"].astype(int)

    mask = (left_topics != -1) & (right_topics != -1)
    ari = adjusted_rand_score(left_topics[mask], right_topics[mask])
    nmi = normalized_mutual_info_score(left_topics[mask], right_topics[mask])

    topn_jaccard = []
    top1_overlap = []
    for left_list, right_list in zip(
        merged["top_n_topics_left"], merged["top_n_topics_right"], strict=False
    ):
        left_set = set(left_list)
        right_set = set(right_list)
        topn_jaccard.append(jaccard(left_set, right_set))
        top1_overlap.append(int(left_list[0] in right_set))

    summary = pd.DataFrame(
        [
            {
                "rows_compared": len(merged),
                "ari_no_neg1": round(ari, 4),
                "nmi_no_neg1": round(nmi, 4),
                "avg_topn_jaccard": round(float(pd.Series(topn_jaccard).mean()), 4),
                "top1_in_other_topn_pct": round(float(pd.Series(top1_overlap).mean()) * 100, 2),
            }
        ]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote comparison: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
