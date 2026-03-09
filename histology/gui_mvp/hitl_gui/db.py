from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def bootstrap_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[sqlite3.Row]:
    cur = conn.execute(query, params)
    return cur.fetchall()


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        conn.execute("BEGIN")
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def ensure_default_project(conn: sqlite3.Connection, workspace_root: Path) -> str:
    project_id = "default_histology_project"
    conn.execute(
        """
        INSERT OR IGNORE INTO projects (
            project_id, project_name, created_at, updated_at,
            nissl_root, gallyas_root, workspace_root,
            default_review_profile, default_cyclegan_profile, default_registration_profile
        ) VALUES (?, ?, datetime('now'), datetime('now'), '', '', ?, 'review_mask', 'cyclegan_train', 'registration_fullres')
        """,
        (project_id, "Histology GUI MVP", str(workspace_root)),
    )
    conn.commit()
    return project_id


def upsert_slide_record(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    slide_id: str,
    stain: str,
    source_path: str,
    source_name: str,
) -> None:
    conn.execute(
        """
        INSERT INTO slides (
            slide_id, project_id, stain, sample_group, source_path, source_name,
            readable, import_status, created_at, updated_at
        ) VALUES (?, ?, ?, '', ?, ?, 1, 'imported', datetime('now'), datetime('now'))
        ON CONFLICT(slide_id) DO UPDATE SET
            stain=excluded.stain,
            source_path=excluded.source_path,
            source_name=excluded.source_name,
            updated_at=datetime('now')
        """,
        (slide_id, project_id, stain, source_path, source_name),
    )
    conn.commit()


def upsert_section_proposal(
    conn: sqlite3.Connection,
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
    conn.execute(
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
    conn.commit()
