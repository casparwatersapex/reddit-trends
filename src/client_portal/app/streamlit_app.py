from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from client_portal.analysis.charts import value_over_time
from client_portal.analysis.metrics import basic_summary
from client_portal.pipeline.run import run_pipeline
from client_portal.reporting.pptx import build_pptx


st.set_page_config(page_title="Client Analytics Portal", layout="wide")
st.title("Client Analytics Portal (Template)")

st.markdown(
    "Upload a CSV or Parquet file, pick a client config, and generate charts + a PowerPoint report."
)

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
        st.subheader("Summary")
        st.dataframe(basic_summary(df), use_container_width=True)

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
