from __future__ import annotations

from pathlib import Path

import streamlit as st

from client_portal.app.label_review_views import render_growth_view, render_labels_view

DEFAULT_CANONICAL = Path("data/gardeninguk_canonical.parquet")
DEFAULT_CLUSTER_PATH = Path("data/clusters_openai_refined.parquet")
DEFAULT_HIERARCHY_PATH = Path("data/hierarchy_openai_refined_openai.csv")
DEFAULT_L2_LABELS_PATH = Path("data/cluster_labels_openai_refined.csv")
DEFAULT_L3_CLUSTER_PATH = Path("data/clusters_openai_l3_anchor.parquet")
DEFAULT_L3_LABELS_PATH = Path("data/cluster_labels_openai_l3_anchor.csv")
DEFAULT_L3_GROWTH_LIMIT = 200
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


def main() -> None:
    st.set_page_config(page_title="Label Review", layout="wide")
    st.title("Label Review")
    st.markdown("Browse L1/L2/L3 labels and growth tables.")

    labels_tab, growth_tab = st.tabs(["Labels", "L3 Growth"])

    with labels_tab:
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

        render_labels_view(
            canonical_path=canonical_path,
            clusters_path=clusters_path,
            hierarchy_path=hierarchy_path,
            l2_labels_path=l2_labels_path,
            l3_clusters_path=l3_clusters_path,
            l3_labels_path=l3_labels_path,
            show_l2_labels=show_l2_labels,
            show_l3=show_l3,
            show_l3_labels=show_l3_labels,
        )

    with growth_tab:
        st.subheader("L3 Growth")
        l3_clusters_path = st.text_input(
            "L3 clusters path",
            str(DEFAULT_L3_CLUSTER_PATH),
        )
        l3_labels_path = st.text_input(
            "L3 labels path",
            str(DEFAULT_L3_LABELS_PATH),
        )
        growth_limit = st.number_input(
            "Max rows per window",
            min_value=50,
            max_value=1000,
            value=DEFAULT_L3_GROWTH_LIMIT,
            step=50,
        )
        labels_path = Path(l3_labels_path) if l3_labels_path else None
        render_growth_view(
            canonical_path=DEFAULT_CANONICAL,
            l3_clusters_path=Path(l3_clusters_path),
            l3_labels_path=labels_path,
            growth_limit=int(growth_limit),
        )


if __name__ == "__main__":
    main()
