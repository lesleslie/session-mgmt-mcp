"""Tests for crackerjack_tools module.

Tests Crackerjack integration MCP tools for quality monitoring,
command execution, and metrics tracking.

Phase: Week 5 Day 1 - Crackerjack Tools Coverage
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestExecuteCrackerjackCommand:
    """Test execute_crackerjack_command validation and execution."""

    @pytest.mark.asyncio
    async def test_valid_command_executes(self) -> None:
        """Should execute valid command successfully."""
        from session_buddy.tools.crackerjack_tools import execute_crackerjack_command

        with patch(
            "session_buddy.tools.crackerjack_tools._execute_crackerjack_command_impl"
        ) as mock_impl:
            mock_impl.return_value = "Success"
            result = await execute_crackerjack_command(command="test")

            assert result == "Success"
            mock_impl.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_command_with_flags_returns_error(self) -> None:
        """Should reject commands starting with flags."""
        from session_buddy.tools.crackerjack_tools import execute_crackerjack_command

        result = await execute_crackerjack_command(command="--ai-fix -t")

        assert "❌" in result
        assert "Invalid Command" in result
        assert "semantic names" in result

    @pytest.mark.asyncio
    async def test_unknown_command_suggests_alternative(self) -> None:
        """Should suggest correct command for typos."""
        from session_buddy.tools.crackerjack_tools import execute_crackerjack_command

        result = await execute_crackerjack_command(
            command="tests"
        )  # typo: should be "test"

        assert "❌" in result
        assert "Unknown Command" in result
        assert "Did you mean" in result

    @pytest.mark.asyncio
    async def test_ai_fix_flag_in_args_returns_error(self) -> None:
        """Should reject --ai-fix in args parameter."""
        from session_buddy.tools.crackerjack_tools import execute_crackerjack_command

        result = await execute_crackerjack_command(
            command="test", args="--ai-fix --verbose"
        )

        assert "❌" in result
        assert "Invalid Args" in result
        assert "ai_agent_mode=True" in result

    @pytest.mark.asyncio
    async def test_valid_commands_accepted(self) -> None:
        """Should accept all valid command names."""
        from session_buddy.tools.crackerjack_tools import execute_crackerjack_command

        valid_commands = [
            "test",
            "lint",
            "check",
            "format",
            "security",
            "complexity",
            "all",
        ]

        with patch(
            "session_buddy.tools.crackerjack_tools._execute_crackerjack_command_impl"
        ) as mock_impl:
            mock_impl.return_value = "Success"

            for cmd in valid_commands:
                result = await execute_crackerjack_command(command=cmd)
                assert result == "Success"

    @pytest.mark.asyncio
    async def test_ai_agent_mode_parameter(self) -> None:
        """Should pass ai_agent_mode to implementation."""
        from session_buddy.tools.crackerjack_tools import execute_crackerjack_command

        with patch(
            "session_buddy.tools.crackerjack_tools._execute_crackerjack_command_impl"
        ) as mock_impl:
            mock_impl.return_value = "Success"

            await execute_crackerjack_command(command="test", ai_agent_mode=True)

            # Verify ai_agent_mode was passed
            call_args = mock_impl.call_args
            assert call_args[0][4] is True  # 5th positional arg is ai_agent_mode


class TestCrackerjackRun:
    """Test crackerjack_run wrapper function."""

    @pytest.mark.asyncio
    async def test_run_calls_implementation(self) -> None:
        """Should call implementation function."""
        from session_buddy.tools.crackerjack_tools import crackerjack_run

        with patch(
            "session_buddy.tools.crackerjack_tools._crackerjack_run_impl"
        ) as mock_impl:
            mock_impl.return_value = "Result"

            result = await crackerjack_run(command="lint", args="--verbose")

            assert result == "Result"
            mock_impl.assert_called_once()


class TestCrackerjackHistory:
    """Test crackerjack execution history tools."""

    @pytest.mark.asyncio
    async def test_history_calls_implementation(self) -> None:
        """Should call get_crackerjack_results_history."""
        from session_buddy.tools.crackerjack_tools import crackerjack_history

        # crackerjack_history just wraps get_crackerjack_results_history
        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await crackerjack_history(working_directory=".", days=7)

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_crackerjack_results_history_calls_implementation(self) -> None:
        """Should call reflection database for history."""
        from session_buddy.tools.crackerjack_tools import (
            get_crackerjack_results_history,
        )

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await get_crackerjack_results_history(
                working_directory=".", days=7
            )

            assert isinstance(result, str)


class TestCrackerjackMetrics:
    """Test quality metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_calls_implementation(self) -> None:
        """Should call get_crackerjack_quality_metrics."""
        from session_buddy.tools.crackerjack_tools import crackerjack_metrics

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await crackerjack_metrics(working_directory=".", days=30)

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_quality_metrics_implementation(self) -> None:
        """Should calculate quality metrics from history."""
        from session_buddy.tools.crackerjack_tools import (
            get_crackerjack_quality_metrics,
        )

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await get_crackerjack_quality_metrics(
                working_directory=".", days=30
            )

            assert isinstance(result, str)


class TestCrackerjackPatterns:
    """Test pattern analysis tools."""

    @pytest.mark.asyncio
    async def test_patterns_calls_implementation(self) -> None:
        """Should call analyze_crackerjack_test_patterns."""
        from session_buddy.tools.crackerjack_tools import crackerjack_patterns

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await crackerjack_patterns(days=7, working_directory=".")

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_analyze_test_patterns_implementation(self) -> None:
        """Should analyze test failure patterns."""
        from session_buddy.tools.crackerjack_tools import (
            analyze_crackerjack_test_patterns,
        )

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await analyze_crackerjack_test_patterns(
                days=7, working_directory="."
            )

            assert isinstance(result, str)


class TestCrackerjackHelp:
    """Test help and documentation tools."""

    @pytest.mark.asyncio
    async def test_help_returns_comprehensive_guide(self) -> None:
        """Should return help documentation."""
        from session_buddy.tools.crackerjack_tools import crackerjack_help

        result = await crackerjack_help()

        assert isinstance(result, str)
        assert len(result) > 100  # Should have substantial help text
        assert "crackerjack" in result.lower()


class TestQualityTrends:
    """Test quality trend analysis."""

    @pytest.mark.asyncio
    async def test_quality_trends_analyzes_history(self) -> None:
        """Should analyze quality trends over time."""
        from session_buddy.tools.crackerjack_tools import crackerjack_quality_trends

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.search_conversations = AsyncMock(return_value=[])
            mock_db.return_value = mock_db_instance

            result = await crackerjack_quality_trends(working_directory=".", days=30)

            assert isinstance(result, str)


class TestHealthCheck:
    """Test Crackerjack integration health monitoring."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self) -> None:
        """Should return health check status."""
        from session_buddy.tools.crackerjack_tools import crackerjack_health_check

        result = await crackerjack_health_check()

        assert isinstance(result, str)
        assert len(result) > 0


class TestQualityMonitor:
    """Test proactive quality monitoring."""

    @pytest.mark.asyncio
    async def test_quality_monitor_returns_insights(self) -> None:
        """Should return quality monitoring insights."""
        from session_buddy.tools.crackerjack_tools import quality_monitor

        with patch(
            "session_buddy.tools.crackerjack_tools._get_reflection_db"
        ) as mock_db:
            mock_db_instance = AsyncMock()
            mock_db.return_value = mock_db_instance

            result = await quality_monitor()

            assert isinstance(result, str)


class TestHelperFunctions:
    """Test utility and helper functions."""

    def test_suggest_command_finds_closest_match(self) -> None:
        """Should suggest closest matching command."""
        from session_buddy.tools.crackerjack_tools import _suggest_command

        valid_commands = {"test", "lint", "check", "format"}

        # Test close matches
        assert _suggest_command("tests", valid_commands) == "test"
        assert _suggest_command("linting", valid_commands) in valid_commands

    def test_build_execution_metadata_creates_dict(self) -> None:
        """Should build metadata dictionary."""
        from session_buddy.tools.crackerjack_tools import _build_execution_metadata

        # Create mock result and metrics objects
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.execution_time = 1.5

        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {"tests": 10}

        metadata = _build_execution_metadata(
            working_directory="/project", result=mock_result, metrics=mock_metrics
        )

        assert isinstance(metadata, dict)
        assert metadata["project"] == "project"  # Path().name
        assert metadata["exit_code"] == 0
        assert metadata["execution_time"] == 1.5
        assert metadata["metrics"] == {"tests": 10}


class TestFormatting:
    """Test output formatting functions."""

    def test_format_basic_result_formats_output(self) -> None:
        """Should format execution result with status."""
        from session_buddy.tools.crackerjack_tools import _format_basic_result

        # _format_basic_result expects a result object with .stdout attribute
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "All checks passed\n"
        mock_result.execution_time = 1.5

        formatted = _format_basic_result(mock_result, "test")

        assert isinstance(formatted, str)
        assert "Crackerjack test" in formatted
        assert "**Status**: Success" in formatted
        assert len(formatted) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_get_reflection_db_returns_none_when_unavailable(self) -> None:
        """Should return None when database is unavailable."""
        from session_buddy.tools.crackerjack_tools import _get_reflection_db

        with patch(
            "session_buddy.tools.crackerjack_tools.resolve_reflection_database"
        ) as mock_resolve:
            # Function returns None when resolve_reflection_database returns None
            mock_resolve.return_value = None

            result = await _get_reflection_db()

            # Should return None when database unavailable
            assert result is None
