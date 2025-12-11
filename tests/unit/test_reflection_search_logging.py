from __future__ import annotations

import typing as t

import duckdb
import pytest
from session_buddy.reflection_tools import ReflectionDatabase


@pytest.mark.asyncio
async def test_search_by_file_logs_access(monkeypatch: t.Any) -> None:
    # Use file-based DB to avoid in-memory locking in tests
    import tempfile
    from pathlib import Path

    tmpdb = Path(tempfile.gettempdir()) / "search_logging.duckdb"
    try:
        tmpdb.unlink()
    except FileNotFoundError:
        pass
    db = ReflectionDatabase(str(tmpdb))
    await db.initialize()

    # Insert a conversation mentioning a file path
    conn = db._get_conn()
    conn.execute(
        "INSERT INTO conversations (id, content, embedding, project, timestamp, metadata) VALUES ('c1','see src/app.py', NULL, 'proj', NOW(), NULL)"
    )

    # Capture access log writes
    calls: list[tuple[str, str]] = []

    def _mock_log(mid: str, access_type: str = "search") -> None:
        calls.append((mid, access_type))

    import session_buddy.memory.persistence as persistence_mod

    monkeypatch.setattr(persistence_mod, "log_memory_access", _mock_log)

    results = await db.search_by_file("src/app.py", limit=5, project=None)
    assert len(results) >= 1
    assert any(mid == "c1" and at == "search" for (mid, at) in calls)
