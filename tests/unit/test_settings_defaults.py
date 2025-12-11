from __future__ import annotations

from session_buddy.settings import get_settings


def test_settings_defaults_present() -> None:
    s = get_settings(reload=True)
    assert s.filesystem_dedupe_ttl_seconds >= 60
    assert s.filesystem_max_file_size_bytes >= 10000
    assert isinstance(s.filesystem_ignore_dirs, list)
    assert s.llm_extraction_timeout >= 1
    assert s.llm_extraction_retries >= 0
