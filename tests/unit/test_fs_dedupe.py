from __future__ import annotations

import sqlite3
import typing as t
from pathlib import Path

from session_buddy.app_monitor import IDEFileHandler, ProjectActivityMonitor


def test_persistent_dedupe(tmp_path: t.Any) -> None:
    # Prepare monitor with temp db path
    mon = ProjectActivityMonitor(project_paths=[str(tmp_path)])
    mon.db_path = str(tmp_path / "activity.db")
    handler = IDEFileHandler(mon)

    # Ensure table exists
    handler._ensure_recent_table()

    test_file = tmp_path / "demo.py"
    test_file.write_text("print('hi')\n", encoding="utf-8")

    # First call should not dedupe
    assert handler._recently_processed_persisted(str(test_file)) is False

    # Second call within TTL should dedupe
    assert handler._recently_processed_persisted(str(test_file)) is True

    # Expire TTL by manual DB update: set last_extracted to 0
    with sqlite3.connect(mon.db_path) as conn:
        conn.execute(
            "UPDATE recent_extractions SET last_extracted=? WHERE file_path=?",
            (0, str(test_file)),
        )
        conn.commit()

    # Now should not dedupe
    assert handler._recently_processed_persisted(str(test_file)) is False
