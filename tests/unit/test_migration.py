from __future__ import annotations

import typing as t
from datetime import datetime

import duckdb
from session_buddy.memory.migration import (
    get_schema_version,
    migrate_v1_to_v2,
)
from session_buddy.memory.schema_v2 import SCHEMA_V2_SQL


def _create_v1_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR PRIMARY KEY,
            content TEXT NOT NULL,
            embedding FLOAT[384],
            project VARCHAR,
            timestamp TIMESTAMP,
            metadata JSON
        );
        CREATE TABLE IF NOT EXISTS reflections (
            id VARCHAR PRIMARY KEY,
            content TEXT NOT NULL,
            embedding FLOAT[384],
            tags VARCHAR[],
            timestamp TIMESTAMP,
            metadata JSON
        );
        """
    )


def test_migrate_v1_to_v2_success(tmp_path: t.Any, monkeypatch: t.Any) -> None:
    # Prepare DB with v1 schema and data
    db_path = tmp_path / "test_migration.duckdb"
    conn = duckdb.connect(str(db_path), config={"allow_unsigned_extensions": True})
    _create_v1_schema(conn)

    # Insert sample v1 data
    for i in range(3):
        conn.execute(
            "INSERT INTO conversations (id, content, embedding, project, timestamp, metadata) VALUES (?, ?, NULL, ?, CURRENT_TIMESTAMP, NULL)",
            [f"c{i}", f"content {i}", "proj"],
        )
    for i in range(2):
        conn.execute(
            "INSERT INTO reflections (id, content, embedding, tags, timestamp, metadata) VALUES (?, ?, NULL, NULL, CURRENT_TIMESTAMP, NULL)",
            [f"r{i}", f"reflection {i}"],
        )

    conn.close()

    # Dry run preview
    preview = migrate_v1_to_v2(db_path=db_path, dry_run=True)
    assert preview.success is True
    assert preview.stats
    assert preview.stats.get("would_migrate") == 3

    # Execute migration
    result = migrate_v1_to_v2(db_path=db_path)
    assert result.success is True

    # Verify version and counts
    assert get_schema_version(db_path) == "v2"

    conn = duckdb.connect(str(db_path), config={"allow_unsigned_extensions": True})
    # Ensure v2 schema exists and records migrated
    conn.execute(SCHEMA_V2_SQL)
    v2 = conn.execute("SELECT COUNT(*) FROM conversations_v2").fetchone()[0]
    assert int(v2) >= 3
    conn.close()
