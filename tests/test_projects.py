from pathlib import Path

from client_portal.projects import (
    create_project,
    list_projects,
    load_project,
    missing_outputs,
    project_paths,
    slugify,
)


def test_slugify() -> None:
    assert slugify("My Project") == "my-project"
    assert slugify("  ") == "project"
    assert slugify("Garden_Insights 2024!") == "garden-insights-2024"


def test_create_and_list_projects(tmp_path) -> None:
    meta = create_project(
        "Test Project",
        "clients/example_client/config.yml",
        root=tmp_path,
    )
    assert meta["slug"] == "test-project"
    projects = list_projects(tmp_path)
    assert len(projects) == 1
    loaded = load_project(meta["slug"], root=tmp_path)
    assert loaded["name"] == "Test Project"
    paths = project_paths(meta["slug"], root=tmp_path)
    assert paths.raw_dir.exists()
    assert paths.outputs_dir.exists()


def test_missing_outputs(tmp_path: Path) -> None:
    existing = tmp_path / "exists.txt"
    existing.write_text("ok", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    result = missing_outputs([existing, missing])
    assert result == [missing]
