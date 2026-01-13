from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from client_portal.analysis.charts import value_over_time
from client_portal.analysis.metrics import basic_summary, coerce_dates, compute_topic_growth
from client_portal.pipeline.run import run_pipeline
from client_portal.reporting.pptx import build_pptx

st.set_page_config(page_title="Client Analytics Portal", layout="wide")
st.title("Client Analytics Portal (Template)")

st.markdown(
    "Upload a CSV or Parquet file, pick a client config, and generate charts + a PowerPoint report."
)

DEFAULT_L3_CLUSTER_PATH = Path("data/clusters_openai_l3_anchor.parquet")
DEFAULT_L3_LABELS_PATH = Path("data/cluster_labels_openai_l3_anchor.csv")

client_cfg = st.selectbox(
    "Client config",
    options=[
        "clients/example_client/config.yml",
    ],
)

uploaded = st.file_uploader("Upload data (.csv or .parquet)", type=["csv", "parquet", "pq"])
if uploaded:
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())

        try:
            df = run_pipeline(input_path=input_path, client_config_path=client_cfg)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

        st.success("Pipeline succeeded.")
        summary_tab, charts_tab, l3_tab = st.tabs(["Summary", "Charts", "L3 Growth"])

        with summary_tab:
            st.subheader("Summary")
            st.dataframe(basic_summary(df), use_container_width=True)

        with charts_tab:
            st.subheader("Charts")
            fig = value_over_time(df)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Export")
            out_path = Path(tmp) / "client_report.pptx"
            pptx_path = build_pptx(out_path, figures=[fig], title="Client Report")
            st.download_button(
                "Download PowerPoint report",
                data=pptx_path.read_bytes(),
                file_name="client_report.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

        with l3_tab:
            st.subheader("L3 Growth")
            l3_clusters_path = st.text_input(
                "L3 clusters path",
                str(DEFAULT_L3_CLUSTER_PATH),
            )
            l3_labels_path = st.text_input(
                "L3 labels path",
                str(DEFAULT_L3_LABELS_PATH),
            )
            if "post_id" not in df.columns or "date" not in df.columns:
                st.warning("Loaded data must include post_id and date columns.")
                st.stop()
            clusters_path = Path(l3_clusters_path)
            labels_path = Path(l3_labels_path)
            if not clusters_path.exists():
                st.warning(f"Missing L3 clusters file: {clusters_path}")
                st.stop()

            l3_df = pd.read_parquet(clusters_path)
            merged = l3_df.merge(df, on="post_id", how="inner")
            merged = merged[merged["topic"] != -1].copy()
            merged["date"] = coerce_dates(merged["date"])
            merged = merged.dropna(subset=["date"])

            label_map = {}
            if labels_path.exists():
                labels_df = pd.read_csv(labels_path)
                for _, row in labels_df.iterrows():
                    if pd.notna(row.get("topic")):
                        label_map[int(row["topic"])] = str(row.get("label", ""))

            for window in (90, 180, 360):
                growth = compute_topic_growth(merged, "topic", "date", window)
                if label_map:
                    growth["label"] = growth["topic"].map(label_map)
                growth = growth.sort_values("growth_abs", ascending=False)
                st.markdown(f"**{window}-day growth**")
                st.dataframe(growth, use_container_width=True)
