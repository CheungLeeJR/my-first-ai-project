from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import DB_PATH


class Database:
    """Small SQLite wrapper used by the local single-user application."""

    def __init__(self, path: str | Path = DB_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        # SQLite foreign-key enforcement is connection-local.
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents(
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_size INTEGER,
                    page_count INTEGER,
                    chunk_count INTEGER,
                    ocr_pages INTEGER,
                    uploaded_at TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks(
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    file_name TEXT,
                    page_number INTEGER,
                    content TEXT,
                    embedding BLOB,
                    dimension INTEGER,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    citations_json TEXT,
                    metrics_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS usage(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    latency_ms INTEGER,
                    retrieved_chunks INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def clear_all(self) -> None:
        with self.connect() as con:
            for table in ("chunks", "documents", "messages", "usage"):
                con.execute(f"DELETE FROM {table}")
