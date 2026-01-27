from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECTS_ROOT = Path("projects")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_dir: Path
    outputs_dir: Path
    combined_jsonl: Path
    canonical_parquet: Path
    embeddings_parquet: Path
    graph_clusters_parquet: Path
    refined_clusters_parquet: Path
    refinement_report_csv: Path
    l2_labels_csv: Path
    hierarchy_csv: Path
    l3_clusters_parquet: Path
    l3_labels_csv: Path
    l3_anchors_csv: Path
    l3_keywords_csv: Path


def slugify(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned or "project"


def _project_root(slug: str, root: Path) -> Path:
    return root / slug


def project_paths(slug: str, root: Path = PROJECTS_ROOT) -> ProjectPaths:
    root_dir = _project_root(slug, root)
    raw_dir = root_dir / "raw"
    outputs_dir = root_dir / "outputs"
    return ProjectPaths(
        root=root_dir,
        raw_dir=raw_dir,
        outputs_dir=outputs_dir,
        combined_jsonl=raw_dir / "combined.jsonl",
        canonical_parquet=outputs_dir / "canonical.parquet",
        embeddings_parquet=outputs_dir / "embeddings_openai.parquet",
        graph_clusters_parquet=outputs_dir / "clusters_openai_graph.parquet",
        refined_clusters_parquet=outputs_dir / "clusters_openai_refined.parquet",
        refinement_report_csv=outputs_dir / "topic_refinement.csv",
        l2_labels_csv=outputs_dir / "cluster_labels_openai_refined.csv",
        hierarchy_csv=outputs_dir / "hierarchy_openai_refined_openai.csv",
        l3_clusters_parquet=outputs_dir / "clusters_openai_l3_anchor.parquet",
        l3_labels_csv=outputs_dir / "cluster_labels_openai_l3_anchor.csv",
        l3_anchors_csv=outputs_dir / "l3_anchors.csv",
        l3_keywords_csv=outputs_dir / "keywords_l3_anchor.csv",
    )


def _meta_path(slug: str, root: Path) -> Path:
    return _project_root(slug, root) / "meta.json"


def load_project(slug: str, root: Path = PROJECTS_ROOT) -> dict:
    meta_path = _meta_path(slug, root)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing project metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_project(meta: dict, root: Path = PROJECTS_ROOT) -> None:
    slug = meta.get("slug")
    if not slug:
        raise ValueError("Project metadata must include slug.")
    meta_path = _meta_path(str(slug), root)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_projects(root: Path = PROJECTS_ROOT) -> list[dict]:
    if not root.exists():
        return []
    projects = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        meta_path = path / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        projects.append(meta)
    projects.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return projects


def create_project(name: str, config_path: str, root: Path = PROJECTS_ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    base_slug = slugify(name)
    slug = base_slug
    counter = 2
    while _project_root(slug, root).exists():
        slug = f"{base_slug}-{counter}"
        counter += 1
    paths = project_paths(slug, root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name.strip(),
        "slug": slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "inputs": [],
        "outputs": {key: str(value) for key, value in paths.__dict__.items()},
    }
    save_project(meta, root)
    return meta


def missing_outputs(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]
