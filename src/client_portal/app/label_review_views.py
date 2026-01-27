from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from client_portal.analysis.label_review import extract_llm_value
from client_portal.analysis.metrics import coerce_dates, compute_topic_growth


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
    available = [
        col for col in ["score", "subreddit", "excerpt", "permalink"] if col in posts.columns
    ]
    if not available:
        st.caption("Post preview columns are unavailable for this view.")
        return
    view = posts.sort_values("score", ascending=False).head(limit).loc[:, available]
    st.dataframe(view, use_container_width=True, hide_index=True)


def render_labels_view(
    *,
    canonical_path: Path,
    clusters_path: Path,
    hierarchy_path: Path,
    l2_labels_path: Path | None,
    l3_clusters_path: Path | None,
    l3_labels_path: Path | None,
    show_l2_labels: bool,
    show_l3: bool,
    show_l3_labels: bool,
) -> None:
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


def render_growth_view(
    *,
    canonical_path: Path,
    l3_clusters_path: Path,
    l3_labels_path: Path | None,
    growth_limit: int,
) -> None:
    clusters_path = Path(l3_clusters_path)
    if not clusters_path.exists():
        st.warning(f"Missing L3 clusters file: {clusters_path}")
        st.stop()
    if not canonical_path.exists():
        st.warning(f"Missing canonical data file: {canonical_path}")
        st.stop()

    l3_df = load_parquet(clusters_path)
    if "post_id" not in l3_df.columns or "topic" not in l3_df.columns:
        st.warning("L3 clusters file must include post_id and topic columns.")
        st.stop()

    canonical_df = load_parquet(canonical_path)
    merged = l3_df.merge(canonical_df, on="post_id", how="inner")
    if "date" not in merged.columns:
        st.warning("Canonical data must include date.")
        st.stop()
    merged["date"] = coerce_dates(merged["date"])
    merged = merged.dropna(subset=["date"])
    merged = merged[merged["topic"] != -1]
    if "score" in merged.columns:
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0)
    if "permalink" in merged.columns:
        merged["permalink"] = merged["permalink"].map(build_permalink)
    merged["excerpt"] = merged.apply(
        lambda row: make_excerpt(row.get("title", ""), row.get("body", "")),
        axis=1,
    )

    label_map: dict[int, str] = {}
    if l3_labels_path and l3_labels_path.exists():
        labels_df = load_csv(l3_labels_path)
        for _, row in labels_df.iterrows():
            if pd.notna(row.get("topic")):
                label_map[int(row["topic"])] = str(row.get("label", ""))

    for window in (90, 180, 360):
        growth = compute_topic_growth(merged, "topic", "date", window)
        if label_map:
            growth["label"] = growth["topic"].map(label_map)
        growth = growth.sort_values("growth_abs", ascending=False).head(growth_limit)
        st.markdown(f"**{window}-day growth**")
        st.dataframe(growth, use_container_width=True)

        options = []
        for _, row in growth.iterrows():
            topic_id = int(row["topic"])
            label = row.get("label") or f"L3 {topic_id}"
            options.append((topic_id, str(label)))
        if not options:
            continue
        selection = st.selectbox(
            f"Top posts for {window}-day growth",
            options=options,
            format_func=lambda item: f"L3 {item[0]}: {item[1]}",
            key=f"growth_select_{window}",
        )
        selected_topic = selection[0]
        selected_posts = merged[merged["topic"] == selected_topic]
        render_posts(selected_posts)
