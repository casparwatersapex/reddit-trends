from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from client_portal.app.label_review_views import render_growth_view, render_labels_view
from client_portal.projects import (
    PROJECTS_ROOT,
    create_project,
    list_projects,
    missing_outputs,
    project_paths,
    save_project,
)

DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_LABEL_MODEL = "gpt-4o-mini"
DEFAULT_ANCHOR_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_MIN_SIMILARITY = 0.2
DEFAULT_L2_THRESHOLD = 500
DEFAULT_LABEL_SAMPLE_SIZE = 20


def list_client_configs() -> list[str]:
    configs = sorted(Path("clients").glob("*/config.yml"))
    return [str(path) for path in configs]


def save_uploads(uploaded_files: list, raw_dir: Path) -> list[dict[str, object]]:
    saved = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for file in uploaded_files:
        dest = raw_dir / file.name
        dest.write_bytes(file.getbuffer())
        saved.append(
            {
                "filename": file.name,
                "path": str(dest),
                "size_bytes": dest.stat().st_size,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return saved


def combine_jsonl_files(raw_dir: Path, output_path: Path) -> list[Path]:
    sources = sorted(raw_dir.glob("*.jsonl"))
    if not sources:
        sources = sorted(raw_dir.glob("*.json"))
    if not sources:
        raise ValueError("No JSONL files found in the project raw folder.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as writer:
        for idx, src in enumerate(sources):
            with src.open("rb") as reader:
                if idx > 0:
                    writer.write(b"\n")
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(chunk)
    return sources


def run_command(label: str, args: list[str]) -> tuple[int, str]:
    result = subprocess.run(args, capture_output=True, text=True)
    output = "\n".join(
        [
            f"$ {' '.join(args)}",
            result.stdout.strip(),
            result.stderr.strip(),
        ]
    ).strip()
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed.\n{output}")
    return result.returncode, output


def main() -> None:
    st.set_page_config(page_title="Project Manager", layout="wide")
    st.title("Project Manager")
    st.markdown(
        "Create named projects, upload multiple subreddit JSONL files, and run the OpenAI-based "
        "embedding + labeling pipeline with preset settings."
    )

    with st.sidebar:
        st.subheader("Create Project")
        with st.form("create_project"):
            project_name = st.text_input("Project name")
            config_options = list_client_configs()
            config_path = st.selectbox(
                "Client config",
                options=config_options if config_options else ["clients/example_client/config.yml"],
            )
            created = st.form_submit_button("Create project")

        if created:
            if not project_name.strip():
                st.error("Project name is required.")
            else:
                meta = create_project(project_name.strip(), config_path)
                st.success(f"Created project: {meta['name']}")
                st.session_state["selected_project"] = meta["slug"]

        projects = list_projects()
        options = {p.get("slug"): p for p in projects}
        if options:
            selected_slug = st.selectbox(
                "Select project",
                options=list(options.keys()),
                format_func=lambda slug: options[slug].get("name", slug),
            )
            st.session_state["selected_project"] = selected_slug
        else:
            st.caption("No projects available yet.")

    if not projects:
        st.info("No projects yet. Create one from the sidebar to begin.")
        return

    project_meta = options[st.session_state["selected_project"]]
    paths = project_paths(project_meta["slug"], PROJECTS_ROOT)

    st.subheader("Project Overview")
    st.markdown(
        f"**Name:** {project_meta.get('name')}  \n"
        f"**Slug:** {project_meta.get('slug')}  \n"
        f"**Created:** {project_meta.get('created_at')}  \n"
        f"**Config:** {project_meta.get('config_path')}"
    )

    upload_tab, pipeline_tab, review_tab, growth_tab = st.tabs(
        ["Uploads", "Run Pipeline", "Label Review", "L3 Growth"]
    )

    with upload_tab:
        st.subheader("Upload subreddit JSONL files")
        uploads = st.file_uploader(
            "Upload one or more JSONL files",
            type=["jsonl", "json"],
            accept_multiple_files=True,
        )
        if st.button("Save uploads"):
            if not uploads:
                st.warning("Select at least one JSONL file.")
            else:
                saved = save_uploads(uploads, paths.raw_dir)
                project_meta.setdefault("inputs", []).extend(saved)
                save_project(project_meta, PROJECTS_ROOT)
                st.success(f"Saved {len(saved)} files to {paths.raw_dir}.")

        raw_files = sorted(paths.raw_dir.glob("*.jsonl"))
        if raw_files:
            st.caption("Files in raw folder:")
            st.dataframe(
                [
                    {"filename": p.name, "size_mb": round(p.stat().st_size / 1e6, 2)}
                    for p in raw_files
                ],
                use_container_width=True,
                hide_index=True,
            )

    with pipeline_tab:
        st.subheader("OpenAI pipeline (preset)")
        st.markdown(
            "- Export canonical data\n"
            "- OpenAI embeddings\n"
            "- k-NN graph clustering (Leiden fallback)\n"
            "- LLM refinement (split/merge)\n"
            "- L2 labels + L1/L2 hierarchy (OpenAI)\n"
            "- L3 anchors + labels (OpenAI)\n"
        )

        resume = st.checkbox("Resume pipeline (skip completed outputs)", value=True)

        if st.button("Run pipeline"):
            try:
                sources = combine_jsonl_files(paths.raw_dir, paths.combined_jsonl)
                project_meta["inputs"] = [
                    {
                        "filename": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sources
                ]
                save_project(project_meta, PROJECTS_ROOT)
                st.success(f"Combined {len(sources)} files into {paths.combined_jsonl}.")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            steps = [
                (
                    "Export canonical",
                    [
                        sys.executable,
                        "scripts/export_canonical.py",
                        str(paths.combined_jsonl),
                        project_meta.get("config_path", "clients/example_client/config.yml"),
                        "--output-path",
                        str(paths.canonical_parquet),
                    ],
                    [paths.canonical_parquet],
                ),
                (
                    "Compute OpenAI embeddings",
                    [
                        sys.executable,
                        "scripts/compute_embeddings.py",
                        str(paths.canonical_parquet),
                        "--output-path",
                        str(paths.embeddings_parquet),
                        "--backend",
                        "openai",
                        "--model",
                        DEFAULT_OPENAI_EMBED_MODEL,
                        "--batch-size",
                        "128",
                        "--stream-write",
                    ],
                    [paths.embeddings_parquet],
                ),
                (
                    "Graph clustering (Leiden)",
                    [
                        sys.executable,
                        "scripts/graph_clustering.py",
                        str(paths.embeddings_parquet),
                        "--output-path",
                        str(paths.graph_clusters_parquet),
                        "--n-neighbors",
                        "30",
                        "--top-n",
                        "3",
                        "--method",
                        "leiden",
                    ],
                    [paths.graph_clusters_parquet],
                ),
                (
                    "Topic refinement (OpenAI)",
                    [
                        sys.executable,
                        "scripts/topic_refinement.py",
                        str(paths.embeddings_parquet),
                        str(paths.graph_clusters_parquet),
                        "--output-path",
                        str(paths.refined_clusters_parquet),
                        "--report-path",
                        str(paths.refinement_report_csv),
                        "--backend",
                        "openai",
                        "--openai-model",
                        DEFAULT_OPENAI_LABEL_MODEL,
                        "--sample-size",
                        str(DEFAULT_LABEL_SAMPLE_SIZE),
                        "--top-n",
                        "3",
                    ],
                    [paths.refined_clusters_parquet, paths.refinement_report_csv],
                ),
                (
                    "Name L2 clusters (OpenAI)",
                    [
                        sys.executable,
                        "scripts/name_clusters.py",
                        str(paths.embeddings_parquet),
                        str(paths.refined_clusters_parquet),
                        "--output-path",
                        str(paths.l2_labels_csv),
                        "--sample-sizes",
                        str(DEFAULT_LABEL_SAMPLE_SIZE),
                        "--backends",
                        "openai",
                        "--openai-model",
                        DEFAULT_OPENAI_LABEL_MODEL,
                    ],
                    [paths.l2_labels_csv],
                ),
                (
                    "Label hierarchy (OpenAI)",
                    [
                        sys.executable,
                        "scripts/label_hierarchy.py",
                        str(paths.embeddings_parquet),
                        str(paths.refined_clusters_parquet),
                        "--output-path",
                        str(paths.hierarchy_csv),
                        "--backend",
                        "openai",
                        "--openai-model",
                        DEFAULT_OPENAI_LABEL_MODEL,
                    ],
                    [paths.hierarchy_csv],
                ),
                (
                    "L3 anchors + labels (OpenAI)",
                    [
                        sys.executable,
                        "scripts/anchor_l3_grouping.py",
                        str(paths.canonical_parquet),
                        str(paths.embeddings_parquet),
                        str(paths.refined_clusters_parquet),
                        "--output-clusters",
                        str(paths.l3_clusters_parquet),
                        "--output-anchors",
                        str(paths.l3_anchors_csv),
                        "--output-keywords",
                        str(paths.l3_keywords_csv),
                        "--output-labels",
                        str(paths.l3_labels_csv),
                        "--l2-threshold",
                        str(DEFAULT_L2_THRESHOLD),
                        "--sample-size",
                        str(DEFAULT_LABEL_SAMPLE_SIZE),
                        "--min-similarity",
                        str(DEFAULT_MIN_SIMILARITY),
                        "--backend",
                        "openai",
                        "--openai-model",
                        DEFAULT_OPENAI_LABEL_MODEL,
                        "--embedding-model",
                        DEFAULT_ANCHOR_EMBED_MODEL,
                        "--l2-labels-path",
                        str(paths.l2_labels_csv),
                    ],
                    [
                        paths.l3_clusters_parquet,
                        paths.l3_anchors_csv,
                        paths.l3_keywords_csv,
                        paths.l3_labels_csv,
                    ],
                ),
            ]

            logs = []
            for label, args, outputs in steps:
                if resume and not missing_outputs(outputs):
                    logs.append(f"{label}\nSkipped (outputs already exist).")
                    continue
                with st.status(label, expanded=False) as status:
                    try:
                        _, output = run_command(label, args)
                        logs.append(f"{label}\n{output}")
                        status.update(label=f"{label} (done)", state="complete")
                    except RuntimeError as exc:
                        status.update(label=f"{label} (failed)", state="error")
                        st.error(str(exc))
                        st.stop()

            project_meta["last_run_at"] = datetime.now(timezone.utc).isoformat()
            save_project(project_meta, PROJECTS_ROOT)
            st.success("Pipeline complete.")
            st.text_area("Pipeline log", value="\n\n".join(logs), height=300)

    with review_tab:
        st.subheader("Label review (project outputs)")
        render_labels_view(
            canonical_path=paths.canonical_parquet,
            clusters_path=paths.refined_clusters_parquet,
            hierarchy_path=paths.hierarchy_csv,
            l2_labels_path=paths.l2_labels_csv,
            l3_clusters_path=paths.l3_clusters_parquet,
            l3_labels_path=paths.l3_labels_csv,
            show_l2_labels=True,
            show_l3=True,
            show_l3_labels=True,
        )

    with growth_tab:
        st.subheader("L3 Growth (project outputs)")
        render_growth_view(
            canonical_path=paths.canonical_parquet,
            l3_clusters_path=paths.l3_clusters_parquet,
            l3_labels_path=paths.l3_labels_csv,
            growth_limit=200,
        )


if __name__ == "__main__":
    main()
