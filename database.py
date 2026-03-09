import json
import uuid
import pickle
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    context TEXT,
    tenant_id TEXT,
    metadata TEXT,
    embedding BLOB,
    created_at TEXT NOT NULL,
    resolved INTEGER DEFAULT 0,
    resolved_at TEXT,
    resolved_by TEXT,
    resolution_notes TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    tenant_id TEXT,
    topic TEXT,
    question_ids TEXT,
    representative_questions TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_tenant ON questions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_questions_resolved ON questions(resolved);
CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at);
"""


class Database:
    """Async SQLite database for question tracking."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Config.db_path())
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.executescript(SCHEMA)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def save_question( self, question: str, context: str | None = None, tenant_id: str | None = None, 
                            metadata: dict | None = None, embedding: list[float] | None = None) -> str:
        """Save a question. Returns its ID."""
        qid = str(uuid.uuid4())
        await self._conn.execute(
            """INSERT INTO questions
               (id, question, context, tenant_id, metadata, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                qid,
                question,
                context,
                tenant_id,
                json.dumps(metadata) if metadata else None,
                pickle.dumps(embedding) if embedding else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._conn.commit()
        return qid

    async def get_question(self, question_id: str) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        )
        row = await cursor.fetchone()
        return self._to_dict(row) if row else None

    async def get_unresolved(self, tenant_id: str | None = None, limit: int | None = None) -> list[dict]:
        """Get unresolved questions, optionally filtered by tenant."""
        query = "SELECT * FROM questions WHERE resolved = 0"
        params: list = []
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await self._conn.execute(query, params)
        return [self._to_dict(row) for row in await cursor.fetchall()]

    async def mark_resolved(self, question_id: str, resolved_by: str, notes: str | None = None) -> bool:
        cursor = await self._conn.execute(
            """UPDATE questions
               SET resolved = 1, resolved_at = ?, resolved_by = ?, resolution_notes = ?
               WHERE id = ?""",
            (datetime.now(timezone.utc).isoformat(), resolved_by, notes, question_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def update_embedding(self, question_id: str, embedding: list[float]) -> bool:
        cursor = await self._conn.execute(
            "UPDATE questions SET embedding = ? WHERE id = ?",
            (pickle.dumps(embedding), question_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def save_cluster(self, topic: str, question_ids: list[str], representative_questions: list[str], tenant_id: str | None = None,) -> str:
        cid = str(uuid.uuid4())
        await self._conn.execute(
            """INSERT INTO clusters
               (id, tenant_id, topic, question_ids, representative_questions, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                cid,
                tenant_id,
                topic,
                json.dumps(question_ids),
                json.dumps(representative_questions),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._conn.commit()
        return cid

    async def get_stats(self, tenant_id: str | None = None) -> dict:
        base = "FROM questions"
        where = " WHERE tenant_id = ?" if tenant_id else ""
        params = [tenant_id] if tenant_id else []

        total = await self._scalar(f"SELECT COUNT(*) {base}{where}", params)
        resolved = await self._scalar(
            f"SELECT COUNT(*) {base}{where}{' AND' if tenant_id else ' WHERE'} resolved = 1",
            params,
        )
        recent = await self._scalar(
            f"SELECT COUNT(*) {base}{where}{' AND' if tenant_id else ' WHERE'} created_at >= datetime('now', '-7 days')",
            params,
        )
        return {
            "total_questions": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "recent_7_days": recent,
            "resolution_rate": round(resolved / total * 100, 2) if total else 0,
        }

    async def _scalar(self, query: str, params: list) -> int:
        cursor = await self._conn.execute(query, params)
        return (await cursor.fetchone())[0]

    @staticmethod
    def _to_dict(row) -> dict:
        return {
            "id": row[0],
            "question": row[1],
            "context": row[2],
            "tenant_id": row[3],
            "metadata": json.loads(row[4]) if row[4] else None,
            "embedding": pickle.loads(row[5]) if row[5] else None,
            "created_at": row[6],
            "resolved": bool(row[7]),
            "resolved_at": row[8],
            "resolved_by": row[9],
            "resolution_notes": row[10],
        }
