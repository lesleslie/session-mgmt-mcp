"""Tests for the ACB dependency injection configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from acb.depends import depends

from session_mgmt_mcp.core import SessionLifecycleManager
from session_mgmt_mcp.di import CLAUDE_DIR_KEY, LOGS_DIR_KEY, configure, reset
from session_mgmt_mcp.server_core import SessionPermissionsManager
from session_mgmt_mcp.utils.logging import SessionLogger


def test_configure_registers_singletons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """configure() should register shared instances for core services."""
    original_home = Path.home()
    monkeypatch.setenv("HOME", str(tmp_path))

    configure(force=True)

    logger = depends.get_sync(SessionLogger)
    assert isinstance(logger, SessionLogger)
    assert logger.log_dir == tmp_path / ".claude" / "logs"
    logger2 = depends.get_sync(SessionLogger)
    assert logger2 is logger

    permissions = depends.get_sync(SessionPermissionsManager)
    assert permissions.permissions_file.parent == tmp_path / ".claude" / "sessions"
    permissions2 = depends.get_sync(SessionPermissionsManager)
    assert permissions2 is permissions

    lifecycle = depends.get_sync(SessionLifecycleManager)
    lifecycle2 = depends.get_sync(SessionLifecycleManager)
    assert lifecycle2 is lifecycle

    claude_dir = depends.get_sync(CLAUDE_DIR_KEY)
    logs_dir = depends.get_sync(LOGS_DIR_KEY)
    assert claude_dir == tmp_path / ".claude"
    assert logs_dir == tmp_path / ".claude" / "logs"

    monkeypatch.setenv("HOME", str(original_home))
    reset()


def test_reset_restores_default_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """reset() should replace overrides with freshly configured defaults."""
    original_home = Path.home()
    monkeypatch.setenv("HOME", str(tmp_path))
    configure(force=True)

    custom_logs = tmp_path / "custom" / "logs"
    custom_logger = SessionLogger(custom_logs)
    depends.set(SessionLogger, custom_logger)
    assert depends.get_sync(SessionLogger) is custom_logger

    reset()

    restored_logger = depends.get_sync(SessionLogger)
    assert restored_logger is not custom_logger
    assert restored_logger.log_dir == tmp_path / ".claude" / "logs"

    monkeypatch.setenv("HOME", str(original_home))
    reset()
