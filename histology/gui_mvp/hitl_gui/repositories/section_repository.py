from __future__ import annotations

import json
import sqlite3
from typing import Optional


class SectionRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def upsert_proposal(
        self,
        *,
        project_id: str,
        slide_id: str,
        section_uid: str,
        stain: str,
        sample_id: str,
        section_id: int,
        proposal_rank: int,
        bbox_overview: dict,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO sections (
                section_uid, project_id, slide_id, stain, sample_id, section_id,
                proposal_rank, proposal_method, proposal_bbox_overview_json, proposal_bbox_level0_json,
                proposal_qc_flags_json, crop_profile, crop_bbox_level0_json,
                crop_canvas_w, crop_canvas_h, crop_level,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'gui_manual', ?, '{}', '{}', 'review_mask', '{}', 0, 0, 0, datetime('now'), datetime('now'))
            ON CONFLICT(section_uid) DO UPDATE SET
                proposal_bbox_overview_json=excluded.proposal_bbox_overview_json,
                proposal_rank=excluded.proposal_rank,
                updated_at=datetime('now')
            """,
            (
                section_uid,
                project_id,
                slide_id,
                stain,
                sample_id,
                section_id,
                proposal_rank,
                json.dumps(bbox_overview),
            ),
        )

    def update_review_state(
        self,
        *,
        section_uid: str,
        mirror_state: str,
        review_status: str,
        manual_mask_version: int,
        notes: str = "",
    ) -> None:
        self.conn.execute(
            """
            UPDATE sections
            SET mirror_state = ?,
                review_status = ?,
                manual_mask_version = ?,
                notes = ?,
                updated_at = datetime('now')
            WHERE section_uid = ?
            """,
            (mirror_state, review_status, manual_mask_version, notes, section_uid),
        )

    def get_manual_mask_version(self, section_uid: str) -> int:
        row = self.conn.execute(
            "SELECT manual_mask_version FROM sections WHERE section_uid = ?",
            (section_uid,),
        ).fetchone()
        return int(row["manual_mask_version"]) if row is not None else 0

    def get_notes(self, section_uid: str) -> str:
        row = self.conn.execute("SELECT notes FROM sections WHERE section_uid = ?", (section_uid,)).fetchone()
        return str(row["notes"]) if row is not None else ""

    def get_latest_revision_id(self, section_uid: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT revision_id FROM revisions WHERE section_uid = ? ORDER BY timestamp DESC LIMIT 1",
            (section_uid,),
        ).fetchone()
        return str(row["revision_id"]) if row is not None else None

    def get_section_state(self, section_uid: str) -> dict:
        row = self.conn.execute(
            """
            SELECT mirror_state, review_status, manual_mask_version, notes
            FROM sections
            WHERE section_uid = ?
            """,
            (section_uid,),
        ).fetchone()
        if row is None:
            return {
                "mirror_state": "original",
                "review_status": "proposed",
                "manual_mask_version": 0,
                "notes": "",
            }
        return {
            "mirror_state": str(row["mirror_state"]),
            "review_status": str(row["review_status"]),
            "manual_mask_version": int(row["manual_mask_version"]),
            "notes": str(row["notes"]),
        }

    def delete_section(self, section_uid: str) -> None:
        self.conn.execute("DELETE FROM sections WHERE section_uid = ?", (section_uid,))
