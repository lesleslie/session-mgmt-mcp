"""Tests for the DI-backed instance manager helpers."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from acb.depends import depends

from session_mgmt_mcp.di import configure, reset as reset_di
from session_mgmt_mcp.utils import instance_managers


@pytest.fixture(autouse=True)
def _reset_di_after() -> None:
    """Ensure DI state is reset after each test."""
    yield
    reset_di()
    instance_managers.reset_instances()


@pytest.mark.asyncio
async def test_get_app_monitor_registers_singleton(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """App monitor is created once and cached through the DI container."""
    module = types.ModuleType("session_mgmt_mcp.app_monitor")
    module.__spec__ = types.SimpleNamespace(name="session_mgmt_mcp.app_monitor")  # type: ignore[attr-defined]

    class DummyMonitor:
        def __init__(self, data_dir: str, project_paths: list[str]) -> None:
            self.data_dir = data_dir
            self.project_paths = project_paths
            self.started = False

        async def start_monitoring(self, project_paths: list[str] | None = None) -> None:
            self.started = True

    module.ApplicationMonitor = DummyMonitor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "session_mgmt_mcp.app_monitor", module)

    # Monkeypatch HOME first, then reset and configure
    monkeypatch.setenv("HOME", str(tmp_path))
    os.chdir(tmp_path)
    from session_mgmt_mcp.server_core import SessionPermissionsManager
    SessionPermissionsManager.reset_singleton()
    configure(force=True)
    monitor = await instance_managers.get_app_monitor()
    assert isinstance(monitor, DummyMonitor)
    assert depends.get_sync(module.ApplicationMonitor) is monitor  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_get_llm_manager_uses_di_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """LLM manager is provided from DI and preserved between calls."""
    module = types.ModuleType("session_mgmt_mcp.llm_providers")
    module.__spec__ = types.SimpleNamespace(name="session_mgmt_mcp.llm_providers")  # type: ignore[attr-defined]

    class DummyLLMManager:
        def __init__(self, config: str | None = None) -> None:
            self.config = config

    module.LLMManager = DummyLLMManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "session_mgmt_mcp.llm_providers", module)

    # Monkeypatch HOME first, then reset and configure
    monkeypatch.setenv("HOME", str(tmp_path))
    from session_mgmt_mcp.server_core import SessionPermissionsManager
    SessionPermissionsManager.reset_singleton()
    configure(force=True)

    first = await instance_managers.get_llm_manager()
    second = await instance_managers.get_llm_manager()

    assert isinstance(first, DummyLLMManager)
    assert first is second
    assert depends.get_sync(module.LLMManager) is first  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_serverless_manager_uses_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Serverless manager resolves through DI and respects config loading."""
    module = types.ModuleType("session_mgmt_mcp.serverless_mode")
    module.__spec__ = types.SimpleNamespace(name="session_mgmt_mcp.serverless_mode")  # type: ignore[attr-defined]

    class DummyStorage:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

    class DummyConfigManager:
        called = False

        @staticmethod
        def load_config(path: str | None) -> dict[str, Any]:
            DummyConfigManager.called = True
            return {"path": path or "memory"}

        @staticmethod
        def create_storage_backend(config: dict[str, Any]) -> DummyStorage:
            return DummyStorage(config)

    class DummyServerlessManager:
        def __init__(self, backend: DummyStorage) -> None:
            self.backend = backend

        async def create_session(
            self, user_id: str, project_id: str, session_data: dict[str, Any] | None, ttl_hours: int
        ) -> str:
            return "session-id"

    module.ServerlessConfigManager = DummyConfigManager  # type: ignore[attr-defined]
    module.ServerlessSessionManager = DummyServerlessManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "session_mgmt_mcp.serverless_mode", module)

    # Monkeypatch HOME first, then reset and configure
    monkeypatch.setenv("HOME", str(tmp_path))
    from session_mgmt_mcp.server_core import SessionPermissionsManager
    SessionPermissionsManager.reset_singleton()
    configure(force=True)

    manager = await instance_managers.get_serverless_manager()
    assert isinstance(manager, DummyServerlessManager)
    assert DummyConfigManager.called is True
    assert manager.backend.config["path"] == "memory"
    assert depends.get_sync(module.ServerlessSessionManager) is manager  # type: ignore[arg-type]
    assert depends.get_sync(module.ServerlessSessionManager) is manager  # type: ignore[arg-type]
