from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from client_portal.analysis.label_review import extract_llm_value

DEFAULT_CANONICAL = Path("data/gardeninguk_canonical.parquet")
DEFAULT_CLUSTER_PATH = Path("data/clusters_openai_refined.parquet")
DEFAULT_HIERARCHY_PATH = Path("data/hierarchy_openai_refined_openai.csv")
DEFAULT_L2_LABELS_PATH = Path("data/cluster_labels_openai_refined.csv")
DEFAULT_L3_CLUSTER_PATH = Path("data/clusters_openai_l3_anchor.parquet")
DEFAULT_L3_LABELS_PATH = Path("data/cluster_labels_openai_l3_anchor.csv")
CLUSTER_PATHS = {
    "local": Path("data/clusters_local_v2.parquet"),
    "openai": DEFAULT_CLUSTER_PATH,
}
HIERARCHY_PATHS = {
    ("local", "local"): Path("data/hierarchy_local_local.csv"),
    ("local", "openai"): Path("data/hierarchy_local_openai.csv"),
    ("openai", "local"): Path("data/hierarchy_openai_local.csv"),
    ("openai", "openai"): DEFAULT_HIERARCHY_PATH,
}


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_permalink(permalink: str) -> str:
    if not isinstance(permalink, str):
        return ""
    if permalink.startswith("http://") or permalink.startswith("https://"):
        return permalink
    return f"https://www.reddit.com{permalink}"


def make_excerpt(title: str, body: str, limit: int = 280) -> str:
    parts = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(body, str) and body.strip():
        parts.append(body.strip())
    text = " - ".join(parts)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def prepare_posts(
    canonical_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    l2_to_l1: dict[int, int],
) -> pd.DataFrame:
    cols = ["post_id", "title", "body", "score", "subreddit", "permalink", "date"]
    posts = canonical_df[[c for c in cols if c in canonical_df.columns]].copy()
    posts = posts.merge(clusters_df[["post_id", "topic"]], on="post_id", how="inner")
    posts["l1_id"] = posts["topic"].map(l2_to_l1)
    posts = posts.loc[posts["l1_id"].notna()].copy()
    posts["score"] = pd.to_numeric(posts["score"], errors="coerce").fillna(0)
    posts["permalink"] = posts["permalink"].map(build_permalink)
    posts["excerpt"] = posts.apply(
        lambda row: make_excerpt(row.get("title", ""), row.get("body", "")), axis=1
    )
    return posts


def prepare_l3_posts(
    canonical_df: pd.DataFrame,
    l3_df: pd.DataFrame,
    l2_to_l1: dict[int, int],
) -> pd.DataFrame:
    cols = ["post_id", "title", "body", "score", "subreddit", "permalink", "date"]
    posts = canonical_df[[c for c in cols if c in canonical_df.columns]].copy()
    posts = posts.merge(l3_df[["post_id", "topic", "parent_topic"]], on="post_id", how="inner")
    posts["l2_id"] = posts["parent_topic"]
    posts["l1_id"] = posts["l2_id"].map(l2_to_l1)
    posts = posts.loc[posts["l1_id"].notna()].copy()
    posts["score"] = pd.to_numeric(posts["score"], errors="coerce").fillna(0)
    posts["permalink"] = posts["permalink"].map(build_permalink)
    posts["excerpt"] = posts.apply(
        lambda row: make_excerpt(row.get("title", ""), row.get("body", "")), axis=1
    )
    return posts


def render_posts(posts: pd.DataFrame, limit: int = 20) -> None:
    if posts.empty:
        st.caption("No posts available for this group.")
        return
    view = (
        posts.sort_values("score", ascending=False)
        .head(limit)
        .loc[:, ["score", "subreddit", "excerpt", "permalink"]]
    )
    st.dataframe(view, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Label Review", layout="wide")
    st.title("Label Review")
    st.markdown("Browse L1 and L2 labels with top posts by Reddit score.")

    with st.sidebar:
        embedding_source = st.selectbox("Embedding source", options=["local", "openai"])
        labeling_backend = st.selectbox("Labeling backend", options=["local", "openai"])
        canonical_path = st.text_input("Canonical data path", str(DEFAULT_CANONICAL))
        cluster_path_input = st.text_input(
            "Clusters path (optional)",
            str(CLUSTER_PATHS[embedding_source]),
        )
        hierarchy_path_input = st.text_input(
            "Hierarchy labels path (optional)",
            str(HIERARCHY_PATHS[(embedding_source, labeling_backend)]),
        )
        l2_labels_path = st.text_input(
            "L2 labels path (optional)",
            str(DEFAULT_L2_LABELS_PATH),
        )
        show_l2_labels = st.checkbox("Show L2 labels from CSV", value=True)
        l3_clusters_path = st.text_input(
            "L3 clusters path (optional)",
            str(DEFAULT_L3_CLUSTER_PATH),
        )
        l3_labels_path = st.text_input(
            "L3 labels path (optional)",
            str(DEFAULT_L3_LABELS_PATH),
        )
        show_l3 = st.checkbox("Show L3 topics", value=True)
        show_l3_labels = st.checkbox("Show L3 labels from CSV", value=True)
        st.caption("Uses top-20 posts by Reddit score for each group.")

    canonical_path = Path(canonical_path)
    clusters_path = (
        Path(cluster_path_input) if cluster_path_input else CLUSTER_PATHS[embedding_source]
    )
    hierarchy_path = (
        Path(hierarchy_path_input)
        if hierarchy_path_input
        else HIERARCHY_PATHS[(embedding_source, labeling_backend)]
    )
    l2_labels_path = Path(l2_labels_path) if l2_labels_path else None
    l3_clusters_path = Path(l3_clusters_path) if l3_clusters_path else None
    l3_labels_path = Path(l3_labels_path) if l3_labels_path else None

    required = [canonical_path, clusters_path, hierarchy_path]
    if show_l3 and l3_clusters_path:
        required.append(l3_clusters_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        st.error(f"Missing required data files: {', '.join(str(p) for p in missing)}")
        st.stop()

    canonical_df = load_parquet(canonical_path)
    clusters_df = load_parquet(clusters_path)
    hierarchy_df = load_csv(hierarchy_path)
    l2_labels_df = None
    if show_l2_labels and l2_labels_path and l2_labels_path.exists():
        l2_labels_df = load_csv(l2_labels_path)
    l3_clusters_df = None
    if show_l3 and l3_clusters_path and l3_clusters_path.exists():
        l3_clusters_df = load_parquet(l3_clusters_path)
    l3_labels_df = None
    if show_l3_labels and l3_labels_path and l3_labels_path.exists():
        l3_labels_df = load_csv(l3_labels_path)

    hierarchy_df["label"] = (
        hierarchy_df["label"].fillna("").map(lambda val: extract_llm_value(val, "label"))
    )
    hierarchy_df["summary"] = (
        hierarchy_df["summary"].fillna("").map(lambda val: extract_llm_value(val, "summary"))
    )
    l1_df = hierarchy_df[hierarchy_df["level"] == "L1"].copy()
    l2_df = hierarchy_df[hierarchy_df["level"] == "L2"].copy()
    l2_label_map: dict[int, dict[str, str]] = {}
    if l2_labels_df is not None and not l2_labels_df.empty:
        filtered = l2_labels_df.copy()
        if "backend" in filtered.columns:
            filtered = filtered[filtered["backend"] == "openai"]
        if "sample_size" in filtered.columns:
            filtered = filtered[filtered["sample_size"] == 20]
        filtered["label"] = (
            filtered["label"].fillna("").map(lambda val: extract_llm_value(val, "label"))
        )
        filtered["summary"] = (
            filtered["summary"].fillna("").map(lambda val: extract_llm_value(val, "summary"))
        )
        for _, row in filtered.iterrows():
            if pd.notna(row.get("topic")):
                l2_label_map[int(row["topic"])] = {
                    "label": str(row.get("label", "")),
                    "summary": str(row.get("summary", "")),
                }
    l3_label_map: dict[int, dict[str, str]] = {}
    if l3_labels_df is not None and not l3_labels_df.empty:
        filtered = l3_labels_df.copy()
        if "backend" in filtered.columns:
            filtered = filtered[filtered["backend"] == "openai"]
        if "sample_size" in filtered.columns:
            filtered = filtered.sort_values("sample_size", ascending=False)
        filtered["label"] = (
            filtered["label"].fillna("").map(lambda val: extract_llm_value(val, "label"))
        )
        filtered["summary"] = (
            filtered["summary"].fillna("").map(lambda val: extract_llm_value(val, "summary"))
        )
        for _, row in filtered.iterrows():
            if pd.notna(row.get("topic")):
                topic_id = int(row["topic"])
                if topic_id in l3_label_map:
                    continue
                l3_label_map[topic_id] = {
                    "label": str(row.get("label", "")),
                    "summary": str(row.get("summary", "")),
                }

    l2_to_l1 = {
        int(row["l2_topic"]): int(row["l1_id"])
        for _, row in l2_df.iterrows()
        if pd.notna(row["l2_topic"]) and pd.notna(row["l1_id"])
    }
    posts = prepare_posts(canonical_df, clusters_df, l2_to_l1)
    l3_posts = None
    l3_counts: dict[int, int] = {}
    if l3_clusters_df is not None:
        l3_counts = (
            l3_clusters_df.loc[l3_clusters_df["topic"] != -1, "topic"].value_counts().to_dict()
        )
        l3_posts = prepare_l3_posts(canonical_df, l3_clusters_df, l2_to_l1)

    st.subheader("L1 Groups")
    l1_df = l1_df.sort_values("size", ascending=False)
    for _, l1_row in l1_df.iterrows():
        l1_id = int(l1_row["l1_id"])
        label = l1_row["label"] or f"L1 {l1_id}"
        summary = l1_row["summary"]
        size = int(l1_row["size"]) if pd.notna(l1_row["size"]) else 0
        expander_label = f"L1 {l1_id}: {label} (posts={size})"
        with st.expander(expander_label, expanded=False):
            if summary:
                st.caption(summary)
            l1_posts = posts[posts["l1_id"] == l1_id]
            show_posts = st.checkbox(
                "Show top posts by score",
                value=False,
                key=f"l1_posts_{l1_id}",
            )
            if show_posts:
                render_posts(l1_posts)

            children = l2_df[l2_df["l1_id"] == l1_id].sort_values("size", ascending=False)
            if children.empty:
                st.caption("No labeled L2 topics for this group.")
                continue

            st.markdown("**L2 Topics**")
            for _, l2_row in children.iterrows():
                topic_id = int(l2_row["l2_topic"])
                override = l2_label_map.get(topic_id, {})
                l2_label = override.get("label") or l2_row["label"] or f"Topic {topic_id}"
                l2_summary = override.get("summary") or l2_row["summary"]
                l2_size = int(l2_row["size"]) if pd.notna(l2_row["size"]) else 0
                with st.expander(
                    f"Topic {topic_id}: {l2_label} (posts={l2_size})",
                    expanded=False,
                ):
                    if l2_summary:
                        st.caption(l2_summary)
                    topic_posts = posts[posts["topic"] == topic_id]
                    render_posts(topic_posts)
                    if not show_l3 or l3_posts is None:
                        continue
                    l3_children = l3_posts[l3_posts["l2_id"] == topic_id]
                    if l3_children.empty:
                        st.caption("No L3 topics available for this L2.")
                        continue
                    st.markdown("**L3 Topics**")
                    for l3_topic in l3_children["topic"].dropna().astype(int).unique().tolist():
                        l3_override = l3_label_map.get(l3_topic, {})
                        l3_label = l3_override.get("label") or f"L3 {l3_topic}"
                        l3_summary = l3_override.get("summary", "")
                        l3_size = int(l3_counts.get(l3_topic, 0))
                        with st.expander(
                            f"L3 {l3_topic}: {l3_label} (posts={l3_size})",
                            expanded=False,
                        ):
                            if l3_summary:
                                st.caption(l3_summary)
                            l3_topic_posts = l3_children[l3_children["topic"] == l3_topic]
                            render_posts(l3_topic_posts)


if __name__ == "__main__":
    main()
