from __future__ import annotations

import sqlite3


class PairRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def list_pairs_for_sample(self, sample_id: str):
        return self.conn.execute(
            "SELECT * FROM pairs WHERE sample_id = ? ORDER BY updated_at DESC",
            (sample_id,),
        ).fetchall()
