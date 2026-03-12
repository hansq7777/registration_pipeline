from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np


class RevisionRepository:
    def __init__(self, conn: sqlite3.Connection, workspace_root: Path) -> None:
        self.conn = conn
        self.workspace_root = workspace_root

    def get_latest_revision_id(self, section_uid: str) -> str | None:
        row = self.conn.execute(
            "SELECT revision_id FROM revisions WHERE section_uid = ? ORDER BY timestamp DESC LIMIT 1",
            (section_uid,),
        ).fetchone()
        return str(row["revision_id"]) if row is not None else None

    def count_revisions(self, section_uid: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM revisions WHERE section_uid = ?",
            (section_uid,),
        ).fetchone()
        return int(row["n"]) if row is not None else 0

    def create_mask_revision(
        self,
        *,
        section_uid: str,
        tissue_mask: np.ndarray,
        artifact_mask: np.ndarray,
        mirror_enabled: bool,
        bbox_overview: dict,
        notes: str = "",
        author: str = "gui_user",
    ) -> str:
        revision_id = uuid.uuid4().hex
        base_revision_id = self.get_latest_revision_id(section_uid)
        revision_dir = self.workspace_root / "revisions" / section_uid
        revision_dir.mkdir(parents=True, exist_ok=True)
        mask_path = revision_dir / f"{revision_id}.npz"
        np.savez_compressed(
            mask_path,
            tissue_mask=tissue_mask.astype(np.uint8),
            artifact_mask=artifact_mask.astype(np.uint8),
        )
        delta = {
            "mirror_enabled": bool(mirror_enabled),
            "bbox_overview": bbox_overview,
            "mask_npz_path": str(mask_path),
            "notes": notes,
        }
        self.conn.execute(
            """
            INSERT INTO revisions (
                revision_id, section_uid, revision_type, author, timestamp, base_revision_id, delta_json, note
            ) VALUES (?, ?, 'mask_revision', ?, datetime('now'), ?, ?, ?)
            """,
            (
                revision_id,
                section_uid,
                author,
                base_revision_id,
                json.dumps(delta),
                notes,
            ),
        )
        return revision_id
