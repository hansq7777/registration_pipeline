from __future__ import annotations

import sqlite3
from pathlib import Path


class ProjectRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_default_project(self, workspace_root: Path) -> str:
        project_id = "default_histology_project"
        self.conn.execute(
            """
            INSERT OR IGNORE INTO projects (
                project_id, project_name, created_at, updated_at,
                nissl_root, gallyas_root, workspace_root,
                default_review_profile, default_cyclegan_profile, default_registration_profile
            ) VALUES (?, ?, datetime('now'), datetime('now'), '', '', ?, 'review_mask', 'cyclegan_train', 'registration_fullres')
            """,
            (project_id, "Histology GUI MVP", str(workspace_root)),
        )
        return project_id

    def get_workspace_root(self, project_id: str) -> Path:
        row = self.conn.execute("SELECT workspace_root FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        return Path(row["workspace_root"])
