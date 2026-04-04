"""Substrata state registry — SQLite-backed tracking for processed files, embeddings, and runs."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from loguru import logger


@dataclass
class RunLogger:
    llm_calls: int = 0
    errors: int = 0


class StateDB:
    """SQLite-backed state registry for Substrata."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                processed_at TEXT,
                wiki_entry_path TEXT,
                status TEXT DEFAULT 'completed'
            );
            CREATE TABLE IF NOT EXISTS embeddings_log (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                chunk_count INTEGER,
                embedded_at TEXT
            );
            CREATE TABLE IF NOT EXISTS run_log (
                id INTEGER PRIMARY KEY,
                run_id TEXT,
                run_type TEXT,
                started_at TEXT,
                completed_at TEXT,
                llm_calls_made INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()

    def is_file_processed(self, file_path: str, current_hash: str) -> bool:
        """Return True if file_path exists in DB and hash matches."""
        row = self._conn.execute(
            "SELECT file_hash FROM processed_files WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return row is not None and row["file_hash"] == current_hash

    def mark_processed(self, file_path: str, file_hash: str, wiki_entry_path: str) -> None:
        """Insert or update processed file record."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO processed_files (file_path, file_hash, processed_at, wiki_entry_path, status)
            VALUES (?, ?, ?, ?, 'completed')
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash=excluded.file_hash,
                processed_at=excluded.processed_at,
                wiki_entry_path=excluded.wiki_entry_path,
                status=excluded.status
            """,
            (file_path, file_hash, now, wiki_entry_path),
        )
        self._conn.commit()

    def get_all_processed(self) -> list[dict]:
        """Return all processed file records as dicts."""
        rows = self._conn.execute("SELECT * FROM processed_files").fetchall()
        return [dict(row) for row in rows]

    def log_embedding(self, doc_id: str, chunk_count: int) -> None:
        """Log an embedding operation."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO embeddings_log (doc_id, chunk_count, embedded_at) VALUES (?, ?, ?)",
            (doc_id, chunk_count, now),
        )
        self._conn.commit()

    @contextmanager
    def log_run(self, run_type: str) -> Generator[RunLogger, None, None]:
        """Context manager that records a run in run_log."""
        import uuid
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        run_logger = RunLogger()
        try:
            yield run_logger
        finally:
            completed_at = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """
                INSERT INTO run_log (run_id, run_type, started_at, completed_at, llm_calls_made, errors)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, run_type, started_at, completed_at, run_logger.llm_calls, run_logger.errors),
            )
            self._conn.commit()
            logger.info(
                "Run completed: type={} llm_calls={} errors={}",
                run_type, run_logger.llm_calls, run_logger.errors,
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
