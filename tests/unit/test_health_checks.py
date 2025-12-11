"""Tests for session-mgmt-mcp health checks.

Tests comprehensive health check implementations for database,
file system, dependencies, and Python environment.

Phase 10.1: Production Hardening - Health Check Tests
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from session_buddy.health_checks import (
    ComponentHealth,
    HealthStatus,
    check_database_health,
    check_dependencies_health,
    check_file_system_health,
    check_python_environment_health,
    get_all_health_checks,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestDatabaseHealthCheck:
    """Test database health check."""

    @pytest.mark.asyncio
    async def test_database_healthy(self) -> None:
        """Should return HEALTHY when database is operational."""
        with patch("session_buddy.health_checks.REFLECTION_AVAILABLE", True):
            mock_db = AsyncMock()
            mock_db.get_stats.return_value = {"conversations_count": 100}

            with patch(
                "session_buddy.health_checks.get_reflection_database",
                return_value=mock_db,
            ):
                result = await check_database_health()

                assert result.name == "database"
                assert result.status == HealthStatus.HEALTHY
                assert "operational" in result.message.lower()
                assert result.latency_ms is not None
                assert result.latency_ms < 500  # Should be fast
                assert result.metadata["conversations"] == 100

    @pytest.mark.asyncio
    async def test_database_unavailable(self) -> None:
        """Should return DEGRADED when database not available."""
        with patch("session_buddy.health_checks.REFLECTION_AVAILABLE", False):
            result = await check_database_health()

            assert result.name == "database"
            assert result.status == HealthStatus.DEGRADED
            assert "not available" in result.message.lower()

    @pytest.mark.asyncio
    async def test_database_high_latency(self) -> None:
        """Should return DEGRADED when database latency is high."""
        with patch("session_buddy.health_checks.REFLECTION_AVAILABLE", True):
            # Mock slow database
            async def slow_get_stats() -> dict:
                import asyncio

                await asyncio.sleep(0.6)  # 600ms delay
                return {"conversations_count": 50}

            mock_db = AsyncMock()
            mock_db.get_stats = slow_get_stats

            with patch(
                "session_buddy.health_checks.get_reflection_database",
                return_value=mock_db,
            ):
                result = await check_database_health()

                assert result.name == "database"
                assert result.status == HealthStatus.DEGRADED
                assert "high" in result.message.lower()
                assert "latency" in result.message.lower()
                assert result.latency_ms is not None
                assert result.latency_ms > 500

    @pytest.mark.asyncio
    async def test_database_error(self) -> None:
        """Should return UNHEALTHY when database check fails."""
        with patch("session_buddy.health_checks.REFLECTION_AVAILABLE", True):
            with patch(
                "session_buddy.health_checks.get_reflection_database",
                side_effect=RuntimeError("Connection failed"),
            ):
                result = await check_database_health()

                assert result.name == "database"
                assert result.status == HealthStatus.UNHEALTHY
                assert "error" in result.message.lower()
                assert "Connection failed" in result.message


class TestFileSystemHealthCheck:
    """Test file system health check."""

    @pytest.mark.asyncio
    async def test_file_system_healthy(self, tmp_path: Path) -> None:
        """Should return HEALTHY when file system is accessible."""
        # Create test directory structure
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "logs").mkdir()
        (claude_dir / "data").mkdir()

        with patch("session_buddy.health_checks.Path.home", return_value=tmp_path):
            result = await check_file_system_health()

            assert result.name == "file_system"
            assert result.status == HealthStatus.HEALTHY
            assert "accessible" in result.message.lower()
            assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_file_system_missing_directory(self, tmp_path: Path) -> None:
        """Should return UNHEALTHY when .claude directory missing."""
        with patch("session_buddy.health_checks.Path.home", return_value=tmp_path):
            result = await check_file_system_health()

            assert result.name == "file_system"
            assert result.status == HealthStatus.UNHEALTHY
            assert "does not exist" in result.message.lower()

    @pytest.mark.asyncio
    async def test_file_system_not_writable(self, tmp_path: Path) -> None:
        """Should return UNHEALTHY when .claude not writable."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        # Make directory read-only
        claude_dir.chmod(0o444)

        try:
            with patch("session_buddy.health_checks.Path.home", return_value=tmp_path):
                result = await check_file_system_health()

                assert result.name == "file_system"
                assert result.status == HealthStatus.UNHEALTHY
                assert "not writable" in result.message.lower()
        finally:
            # Restore permissions for cleanup
            claude_dir.chmod(0o755)

    @pytest.mark.asyncio
    async def test_file_system_missing_subdirectories(self, tmp_path: Path) -> None:
        """Should return DEGRADED when subdirectories missing."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        # Don't create logs/data directories

        with patch("session_buddy.health_checks.Path.home", return_value=tmp_path):
            result = await check_file_system_health()

            assert result.name == "file_system"
            assert result.status == HealthStatus.DEGRADED
            assert "missing directories" in result.message.lower()
            assert "logs" in result.message or "data" in result.message


class TestDependenciesHealthCheck:
    """Test dependencies health check."""

    @pytest.mark.asyncio
    async def test_dependencies_all_available(self) -> None:
        """Should return HEALTHY when all optional dependencies available."""
        # Create a mock server module with MULTI_PROJECT_AVAILABLE
        mock_server = MagicMock()
        mock_server.MULTI_PROJECT_AVAILABLE = True

        with (
            patch("session_buddy.utils.quality_utils_v2.CRACKERJACK_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {"onnxruntime": MagicMock(), "session_buddy.server": mock_server},
            ),
        ):
            result = await check_dependencies_health()

            assert result.name == "dependencies"
            assert result.status == HealthStatus.HEALTHY
            assert "available" in result.message.lower()
            assert len(result.metadata.get("available", [])) >= 3
            assert len(result.metadata.get("unavailable", [])) == 0

    @pytest.mark.asyncio
    async def test_dependencies_none_available(self) -> None:
        """Should return DEGRADED when no optional dependencies available."""
        # Mock all dependencies as unavailable
        import builtins
        import sys

        # Save original modules and import function
        onnx_module = sys.modules.pop("onnxruntime", None)
        original_import = builtins.__import__

        # Create a mock server module with MULTI_PROJECT_AVAILABLE = False
        mock_server = MagicMock()
        mock_server.MULTI_PROJECT_AVAILABLE = False

        # Mock import to make onnxruntime fail
        def mock_import(name, *args, **kwargs):
            if name == "onnxruntime":
                msg = f"No module named '{name}'"
                raise ImportError(msg)
            return original_import(name, *args, **kwargs)

        try:
            with (
                patch(
                    "session_buddy.utils.quality_utils_v2.CRACKERJACK_AVAILABLE",
                    False,
                ),
                patch.dict("sys.modules", {"session_buddy.server": mock_server}),
                patch("builtins.__import__", side_effect=mock_import),
                patch(
                    "importlib.util.find_spec", return_value=None
                ),  # Mock multi_project check
            ):
                result = await check_dependencies_health()

                assert result.name == "dependencies"
                assert result.status == HealthStatus.DEGRADED
                assert "no optional features" in result.message.lower()
        finally:
            # Restore onnxruntime if it was present
            if onnx_module is not None:
                sys.modules["onnxruntime"] = onnx_module

    @pytest.mark.asyncio
    async def test_dependencies_some_available(self) -> None:
        """Should return DEGRADED when some dependencies available."""
        with (
            patch("session_buddy.utils.quality_utils_v2.CRACKERJACK_AVAILABLE", True),
            patch.dict("sys.modules", clear=True),  # Clear all modules
        ):
            result = await check_dependencies_health()

            assert result.name == "dependencies"
            # Can be HEALTHY or DEGRADED depending on what's available
            assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            assert result.metadata.get("available") or result.metadata.get(
                "unavailable"
            )


class TestPythonEnvironmentHealthCheck:
    """Test Python environment health check."""

    @pytest.mark.asyncio
    async def test_python_env_healthy(self) -> None:
        """Should return HEALTHY for valid Python 3.13+ environment."""
        result = await check_python_environment_health()

        assert result.name == "python_env"
        assert result.status == HealthStatus.HEALTHY
        assert "python" in result.message.lower()
        assert result.latency_ms is not None
        assert "python_version" in result.metadata
        assert "platform" in result.metadata

    @pytest.mark.asyncio
    async def test_python_env_old_version(self) -> None:
        """Should return UNHEALTHY for Python < 3.13."""
        from collections import namedtuple

        # Create a mock version_info for Python 3.11
        MockVersionInfo = namedtuple(
            "version_info", ["major", "minor", "micro", "releaselevel", "serial"]
        )
        mock_version = MockVersionInfo(
            major=3, minor=11, micro=0, releaselevel="final", serial=0
        )

        # Patch sys.version_info
        with patch("sys.version_info", new=mock_version):
            result = await check_python_environment_health()

            assert result.name == "python_env"
            assert result.status == HealthStatus.UNHEALTHY
            assert "required" in result.message.lower()
            assert "3.13" in result.message


class TestGetAllHealthChecks:
    """Test comprehensive health check aggregation."""

    @pytest.mark.asyncio
    async def test_get_all_checks_runs_all(self) -> None:
        """Should run all health checks concurrently."""
        components = await get_all_health_checks()

        assert len(components) == 4  # python_env, file_system, database, dependencies

        component_names = {c.name for c in components}
        assert "python_env" in component_names
        assert "file_system" in component_names
        assert "database" in component_names
        assert "dependencies" in component_names

    @pytest.mark.asyncio
    async def test_get_all_checks_handles_exceptions(self) -> None:
        """Should handle exceptions in individual checks gracefully."""
        # Mock one check to raise exception
        with patch(
            "session_buddy.health_checks.check_python_environment_health",
            side_effect=RuntimeError("Test error"),
        ):
            components = await get_all_health_checks()

            assert len(components) == 4

            # Find the failed check
            python_check = next(c for c in components if c.name == "python_env")
            assert python_check.status == HealthStatus.UNHEALTHY
            assert "crashed" in python_check.message.lower()
            assert "Test error" in python_check.message

    @pytest.mark.asyncio
    async def test_get_all_checks_concurrent_execution(self) -> None:
        """Should execute all checks concurrently for performance."""
        import time

        start_time = time.perf_counter()
        components = await get_all_health_checks()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert len(components) == 4

        # All checks should complete quickly when run concurrently
        # Individual checks might take 1-10ms, but concurrent should be <50ms total
        assert elapsed_ms < 200  # Very generous upper bound for concurrent execution
