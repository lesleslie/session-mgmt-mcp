"""Test fixtures for server.py and server_core.py testing.

Week 8 Day 2 - Phase 2: Mock fixtures for MCP server testing.
Provides isolated FastMCP server instances and mock tool registration.
"""

from __future__ import annotations

import typing as t
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

if t.TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
def mock_fastmcp_server() -> Mock:
    """Create a mock FastMCP server for testing tool registration.

    Returns:
        Mock FastMCP server with tool/resource registration capabilities.

    """
    server = Mock()
    server.tool = Mock(return_value=lambda f: f)  # Decorator passthrough
    server.resource = Mock(return_value=lambda f: f)
    server.prompt = Mock(return_value=lambda f: f)
    server.run = AsyncMock()
    return server


@pytest.fixture
def mock_session_paths(tmp_path: Path) -> Mock:
    """Create a mock SessionPaths with temporary directories.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Mock SessionPaths instance with test directories.

    """
    paths = Mock()
    paths.claude_dir = tmp_path / ".claude"
    paths.logs_dir = paths.claude_dir / "logs"
    paths.data_dir = paths.claude_dir / "data"
    paths.sessions_dir = paths.claude_dir / "sessions"
    paths.config_file = paths.claude_dir / "config.json"

    # Create directories
    for dir_path in [
        paths.claude_dir,
        paths.logs_dir,
        paths.data_dir,
        paths.sessions_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return paths


@pytest.fixture
def mock_session_logger(tmp_path: Path) -> Mock:
    """Create a mock SessionLogger for testing.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Mock SessionLogger with no-op logging methods.

    """
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.log_file = tmp_path / "test.log"
    return logger


@pytest.fixture
def mock_permissions_manager() -> Mock:
    """Create a mock SessionPermissionsManager for testing.

    Returns:
        Mock permissions manager with default allow/deny behavior.

    """
    manager = Mock()
    manager.is_trusted_operation = Mock(return_value=True)
    manager.add_trusted_operation = Mock()
    manager.remove_trusted_operation = Mock()
    manager.get_trusted_operations = Mock(return_value=[])
    return manager


@pytest.fixture
def mock_lifecycle_manager() -> Mock:
    """Create a mock SessionLifecycleManager for testing.

    Returns:
        Mock lifecycle manager with session state tracking.

    """
    manager = Mock()
    manager.session_active = False
    manager.session_id = None
    manager.start_time = None
    manager.checkpoint_count = 0

    async def mock_start(**kwargs: t.Any) -> dict[str, t.Any]:
        manager.session_active = True
        manager.session_id = "test-session-id"
        manager.checkpoint_count = 0
        return {
            "success": True,
            "session_id": manager.session_id,
            "message": "Session started",
        }

    async def mock_checkpoint(**kwargs: t.Any) -> dict[str, t.Any]:
        manager.checkpoint_count += 1
        return {
            "success": True,
            "checkpoint_number": manager.checkpoint_count,
            "message": f"Checkpoint {manager.checkpoint_count} created",
        }

    async def mock_end(**kwargs: t.Any) -> dict[str, t.Any]:
        manager.session_active = False
        return {
            "success": True,
            "message": "Session ended",
            "checkpoints": manager.checkpoint_count,
        }

    manager.start = AsyncMock(side_effect=mock_start)
    manager.checkpoint = AsyncMock(side_effect=mock_checkpoint)
    manager.end = AsyncMock(side_effect=mock_end)
    manager.get_status = AsyncMock(return_value={"active": manager.session_active})

    return manager


@pytest.fixture
async def mock_mcp_server_context(
    tmp_path: Path,
    mock_session_paths: Mock,
    mock_session_logger: Mock,
    mock_permissions_manager: Mock,
    mock_lifecycle_manager: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[dict[str, t.Any]]:
    """Create a complete mock MCP server context for integration testing.

    Args:
        tmp_path: pytest temporary directory fixture.
        mock_session_paths: Mock SessionPaths fixture.
        mock_session_logger: Mock SessionLogger fixture.
        mock_permissions_manager: Mock permissions manager fixture.
        mock_lifecycle_manager: Mock lifecycle manager fixture.
        monkeypatch: pytest monkeypatch fixture.

    Yields:
        Dictionary with all mock server components.

    """
    # Set up environment
    monkeypatch.setenv("PWD", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    # Create context with all mocks
    return {
        "paths": mock_session_paths,
        "logger": mock_session_logger,
        "permissions": mock_permissions_manager,
        "lifecycle": mock_lifecycle_manager,
        "tmp_path": tmp_path,
    }

    # Cleanup (if needed)


@pytest.fixture
def mock_quality_score_result() -> dict[str, t.Any]:
    """Create a mock quality score result for testing.

    Returns:
        Typical quality score dictionary with all expected fields.

    """
    return {
        "success": True,
        "quality_score": 75,
        "project_health": {
            "git_status": "clean",
            "has_tests": True,
            "has_ci": True,
            "documentation": "good",
        },
        "permissions_score": 80,
        "tools_available": 15,
        "recommendations": [
            "Consider adding more integration tests",
            "Update documentation for new features",
        ],
        "timestamp": "2025-10-29T12:00:00Z",
    }


@pytest.fixture
def mock_health_check_result() -> dict[str, t.Any]:
    """Create a mock health check result for testing.

    Returns:
        Typical health check dictionary with all expected fields.

    """
    return {
        "success": True,
        "status": "healthy",
        "checks": {
            "database": "ok",
            "filesystem": "ok",
            "permissions": "ok",
            "dependencies": "ok",
        },
        "warnings": [],
        "errors": [],
        "timestamp": "2025-10-29T12:00:00Z",
    }


@pytest.fixture
def mock_tool_result_factory() -> t.Callable[
    [bool, str, dict[str, t.Any]], dict[str, t.Any]
]:
    """Create a factory for generating mock tool results.

    Returns:
        Factory function that creates tool result dictionaries.

    Example:
        >>> factory = mock_tool_result_factory()
        >>> result = factory(success=True, message="Done", extra={"data": 42})
        >>> assert result["success"] is True
        >>> assert result["data"] == 42

    """

    def factory(
        success: bool = True,
        message: str = "Operation completed",
        extra: dict[str, t.Any] | None = None,
    ) -> dict[str, t.Any]:
        """Create a mock tool result with given parameters.

        Args:
            success: Whether operation succeeded.
            message: Result message.
            extra: Additional fields to include.

        Returns:
            Tool result dictionary.

        """
        result: dict[str, t.Any] = {
            "success": success,
            "message": message,
            "timestamp": "2025-10-29T12:00:00Z",
        }

        if extra:
            result.update(extra)

        return result

    return factory
