from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .application import WorkflowService
from .db import bootstrap_schema, connect_db, transaction
from .repositories import PairRepository, ProjectRepository, RevisionRepository, SectionRepository, SlideRepository
from .ui.workflow_window import WorkflowWindow


def build_app_shell(db_path: Path, schema_path: Path) -> WorkflowWindow:
    conn = connect_db(db_path)
    bootstrap_schema(conn, schema_path)
    project_repository = ProjectRepository(conn)
    slide_repository = SlideRepository(conn)
    section_repository = SectionRepository(conn)
    with transaction(conn):
        project_id = project_repository.ensure_default_project(Path(__file__).resolve().parents[1] / "workspace_example")
    revision_repository = RevisionRepository(conn, project_repository.get_workspace_root(project_id))
    pair_repository = PairRepository(conn)
    workflow_service = WorkflowService(
        conn=conn,
        project_repository=project_repository,
        slide_repository=slide_repository,
        section_repository=section_repository,
        revision_repository=revision_repository,
        pair_repository=pair_repository,
        project_id=project_id,
    )
    return WorkflowWindow(workflow_service=workflow_service)


def main() -> int:
    app = QApplication(sys.argv)
    root = Path(__file__).resolve().parents[1]
    db_path = root / "workspace_example" / "db" / "project.sqlite"
    schema_path = root / "schema" / "histology_project_v1.sql"
    shell = build_app_shell(db_path=db_path, schema_path=schema_path)
    shell.resize(1280, 900)
    shell.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
