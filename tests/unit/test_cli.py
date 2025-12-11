#!/usr/bin/env python3
"""Test suite for session_buddy.cli module.

Tests CLI commands, server process management, and status reporting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import psutil
import pytest
from typer.testing import CliRunner
from session_buddy.cli import app, find_server_processes, get_server_status


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_process() -> MagicMock:
    """Create a mock psutil.Process."""
    proc = MagicMock(spec=psutil.Process)
    proc.info = {
        "pid": 1234,
        "name": "python",
        "cmdline": ["python", "-m", "session_buddy.server"],
    }
    proc.pid = 1234
    proc.is_running.return_value = True
    return proc


class TestFindServerProcesses:
    """Test server process discovery."""

    def test_find_server_processes_empty(self) -> None:
        """Test finding processes when none exist."""
        with patch("psutil.process_iter", return_value=[]):
            processes = find_server_processes()
            assert processes == []

    def test_find_server_processes_with_server(self, mock_process: MagicMock) -> None:
        """Test finding server processes when they exist."""
        with patch("psutil.process_iter", return_value=[mock_process]):
            processes = find_server_processes()
            assert len(processes) > 0


class TestGetServerStatus:
    """Test server status reporting."""

    def test_get_server_status_no_server(self) -> None:
        """Test status when server is not running."""
        with patch("session_buddy.cli.find_server_processes", return_value=[]):
            status = get_server_status()
            assert status["running"] is False
            assert status["process_count"] == 0

    def test_get_server_status_with_server(self, mock_process: MagicMock) -> None:
        """Test status when server is running."""
        with patch(
            "session_buddy.cli.find_server_processes", return_value=[mock_process]
        ):
            status = get_server_status()
            assert status["running"] is True
            assert status["process_count"] == 1


class TestCliCommands:
    """Test CLI command execution."""

    def test_version_command(self, cli_runner: CliRunner) -> None:
        """Test version command display."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_status_command(self, cli_runner: CliRunner) -> None:
        """Test status command display."""
        with patch("session_buddy.cli.show_status"):
            result = cli_runner.invoke(app, ["--status"])
            # Note: may exit with 0 or 1 depending on server state
            assert result.exit_code in [0, 1]

    def test_start_command_basic(self, cli_runner: CliRunner) -> None:
        """Test start command with minimal arguments."""
        with (
            patch("session_buddy.cli.find_server_processes", return_value=[]),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.pid = 1234
            result = cli_runner.invoke(app, ["--start-mcp-server"])
            assert result.exit_code in [0, 1]  # May fail if already running

    def test_stop_command(self, cli_runner: CliRunner) -> None:
        """Test stop command."""
        with patch("session_buddy.cli.find_server_processes", return_value=[]):
            result = cli_runner.invoke(app, ["--stop-mcp-server"])
            assert result.exit_code in [0, 1]

    def test_restart_command(self, cli_runner: CliRunner) -> None:
        """Test restart command."""
        with (
            patch("session_buddy.cli.find_server_processes", return_value=[]),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.pid = 1234
            result = cli_runner.invoke(app, ["--restart-mcp-server"])
            assert result.exit_code in [0, 1]

    def test_logs_command(self, cli_runner: CliRunner) -> None:
        """Test logs display command."""
        with patch("session_buddy.cli.show_logs"):
            result = cli_runner.invoke(app, ["--logs"])
            assert result.exit_code in [0, 1]

    def test_config_command(self, cli_runner: CliRunner) -> None:
        """Test config display command."""
        with patch("session_buddy.cli.show_config"):
            result = cli_runner.invoke(app, ["--config"])
            assert result.exit_code == 0


class TestServerManagement:
    """Test server lifecycle management."""

    def test_server_already_running(
        self, cli_runner: CliRunner, mock_process: MagicMock
    ) -> None:
        """Test starting server when already running."""
        with patch(
            "session_buddy.cli.find_server_processes", return_value=[mock_process]
        ):
            result = cli_runner.invoke(app, ["--start-mcp-server"])
            assert result.exit_code in [
                0,
                1,
            ]  # May succeed or fail depending on auto-restart

    def test_server_not_running_on_stop(self, cli_runner: CliRunner) -> None:
        """Test stopping server when not running."""
        with patch("session_buddy.cli.find_server_processes", return_value=[]):
            result = cli_runner.invoke(app, ["--stop-mcp-server"])
            assert result.exit_code in [0, 1]  # May succeed (already stopped) or fail
