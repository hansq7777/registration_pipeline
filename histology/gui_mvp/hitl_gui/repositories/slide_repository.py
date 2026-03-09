from __future__ import annotations

import sqlite3


class SlideRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def upsert_slide(
        self,
        *,
        project_id: str,
        slide_id: str,
        stain: str,
        source_path: str,
        source_name: str,
        readable: bool = True,
        level_count: int = 0,
        width_level0: int = 0,
        height_level0: int = 0,
        focal_metadata_json: str = "{}",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO slides (
                slide_id, project_id, stain, sample_group, source_path, source_name,
                readable, import_status, level_count, width_level0, height_level0, focal_metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, '', ?, ?, ?, 'imported', ?, ?, ?, ?, datetime('now'), datetime('now'))
            ON CONFLICT(slide_id) DO UPDATE SET
                stain=excluded.stain,
                source_path=excluded.source_path,
                source_name=excluded.source_name,
                readable=excluded.readable,
                level_count=excluded.level_count,
                width_level0=excluded.width_level0,
                height_level0=excluded.height_level0,
                focal_metadata_json=excluded.focal_metadata_json,
                updated_at=datetime('now')
            """,
            (
                slide_id,
                project_id,
                stain,
                source_path,
                source_name,
                1 if readable else 0,
                level_count,
                width_level0,
                height_level0,
                focal_metadata_json,
            ),
        )
