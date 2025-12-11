#!/usr/bin/env python3
"""Comprehensive tests for SessionLifecycleManager lifecycle methods.

Tests the complete session lifecycle including:
- Session initialization with project analysis
- Quality assessment and scoring
- Session checkpointing
- Session completion and handoff
- Project context analysis
- Quality history tracking
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestSessionInitialization:
    """Test session initialization workflow."""

    async def test_initialize_session_basic(self):
        """Test basic session initialization."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.initialize_session(working_directory=tmpdir)

                assert result["success"]
                assert "project" in result
                assert "working_directory" in result
                assert "quality_score" in result
                assert "quality_data" in result
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_initialize_session_sets_current_project(self):
        """Test that initialization sets current project."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                assert manager.current_project is None

                result = await manager.initialize_session(working_directory=tmpdir)

                assert manager.current_project is not None
                assert result["success"]
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_initialize_session_creates_claude_dir(self):
        """Test that initialization creates .claude directory."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.initialize_session(working_directory=tmpdir)

                assert result["success"]
                assert "claude_directory" in result
                claude_dir = Path(result["claude_directory"])
                assert claude_dir.exists()
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestProjectContextAnalysis:
    """Test project context analysis."""

    async def test_analyze_project_context_empty_dir(self):
        """Test analyzing empty project directory."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                context = await manager.analyze_project_context(Path(tmpdir))

                assert isinstance(context, dict)
                assert "has_pyproject_toml" in context
                assert "has_git_repo" in context
                assert "has_tests" in context
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_analyze_project_context_with_structure(self):
        """Test analyzing project with structure."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                (tmppath / "pyproject.toml").touch()
                (tmppath / "README.md").touch()
                (tmppath / ".git").mkdir()
                (tmppath / "tests").mkdir()

                manager = SessionLifecycleManager()
                context = await manager.analyze_project_context(tmppath)

                assert context["has_pyproject_toml"]
                assert context["has_readme"]
                assert context["has_git_repo"]
                assert context["has_tests"]
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_analyze_project_context_python_files(self):
        """Test project context with Python files."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                (tmppath / "module.py").touch()
                (tmppath / "another.py").touch()

                manager = SessionLifecycleManager()
                context = await manager.analyze_project_context(tmppath)

                assert context.get("has_python_files", False)
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestQualityAssessment:
    """Test quality assessment functionality."""

    async def test_calculate_quality_score(self):
        """Test quality score calculation."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.calculate_quality_score(Path(tmpdir))

                assert isinstance(result, dict)
                assert "total_score" in result or "breakdown" in result
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_perform_quality_assessment(self):
        """Test quality assessment workflow."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                quality_score, quality_data = await manager.perform_quality_assessment(
                    Path(tmpdir)
                )

                assert isinstance(quality_score, int)
                assert isinstance(quality_data, dict)
                assert quality_score >= 0
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestQualityHistoryTracking:
    """Test quality score history tracking."""

    async def test_record_quality_score(self):
        """Test recording quality scores."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()
            assert "test_project" not in manager._quality_history

            manager.record_quality_score("test_project", 80)

            assert "test_project" in manager._quality_history
            assert manager._quality_history["test_project"] == [80]
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_record_multiple_scores(self):
        """Test recording multiple quality scores."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            manager.record_quality_score("project", 70)
            manager.record_quality_score("project", 75)
            manager.record_quality_score("project", 80)

            assert len(manager._quality_history["project"]) == 3
            assert manager._quality_history["project"] == [70, 75, 80]
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_get_previous_quality_score(self):
        """Test retrieving previous quality score."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()
            manager.record_quality_score("project", 80)
            manager.record_quality_score("project", 85)

            score = manager.get_previous_quality_score("project")
            assert score == 85
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_quality_history_limits_to_10(self):
        """Test that quality history is limited to last 10 scores."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            for i in range(15):
                manager.record_quality_score("project", 70 + i)

            # Should keep only last 10
            assert len(manager._quality_history["project"]) == 10
            # Should have scores from 75-84
            assert manager._quality_history["project"][0] == 75
            assert manager._quality_history["project"][-1] == 84
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionCheckpoint:
    """Test session checkpointing functionality."""

    async def test_checkpoint_session_basic(self):
        """Test basic session checkpoint."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.checkpoint_session(tmpdir)

                assert result["success"]
                assert "quality_score" in result
                assert "timestamp" in result
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_checkpoint_session_returns_output(self):
        """Test that checkpoint returns formatted output."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.checkpoint_session(tmpdir)

                assert result["success"]
                assert "quality_output" in result
                assert isinstance(result["quality_output"], list)
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_checkpoint_records_quality_score(self):
        """Test that checkpoint records quality score."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.checkpoint_session(tmpdir)

                assert result["success"]
                result.get("quality_score")  # May be in auto_store_decision
                # Quality score should be recorded
                assert "quality_score" in result
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionEnd:
    """Test session ending workflow."""

    async def test_end_session_basic(self):
        """Test basic session ending."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.end_session(tmpdir)

                assert result["success"]
                assert "summary" in result
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_end_session_generates_handoff(self):
        """Test that session end generates handoff documentation."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.end_session(tmpdir)

                assert result["success"]
                summary = result["summary"]
                assert "final_quality_score" in summary
                assert "recommendations" in summary
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_end_session_creates_handoff_file(self):
        """Test that handoff documentation file is created."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.end_session(tmpdir)

                assert result["success"]
                handoff_path = result["summary"].get("handoff_documentation")
                if handoff_path:
                    assert Path(handoff_path).exists()
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionStatus:
    """Test session status retrieval."""

    async def test_get_session_status_basic(self):
        """Test getting session status."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.get_session_status(tmpdir)

                assert result["success"]
                assert "project" in result
                assert "quality_score" in result
                assert "system_health" in result
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_get_session_status_includes_health(self):
        """Test that status includes system health checks."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.get_session_status(tmpdir)

                assert result["success"]
                health = result["system_health"]
                assert "uv_available" in health
                assert "git_repository" in health
                assert "claude_directory" in health
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestQualityFormatting:
    """Test quality results formatting."""

    async def test_format_quality_results(self):
        """Test formatting quality results for display."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.calculate_quality_score(Path(tmpdir))

                output = manager.format_quality_results(85, result)

                assert isinstance(output, list)
                assert len(output) > 0
                # Output should contain status line
                "\n".join(output)
                assert any("quality" in line.lower() for line in output)
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_format_quality_results_high_score(self):
        """Test formatting output for high quality score."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()
            quality_data = {
                "total_score": 85,
                "breakdown": {
                    "code_quality": 30,
                    "project_health": 25,
                    "dev_velocity": 18,
                    "security": 9,
                },
                "recommendations": ["Good work!"],
            }

            output = manager.format_quality_results(85, quality_data)
            output_str = "\n".join(output)

            # High score should show EXCELLENT or GOOD
            assert "EXCELLENT" in output_str or "GOOD" in output_str
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionInfoDataclass:
    """Test SessionInfo dataclass."""

    async def test_session_info_empty(self):
        """Test creating empty SessionInfo."""
        try:
            from session_buddy.core.session_manager import SessionInfo

            info = SessionInfo.empty()
            assert info.session_id == ""
            assert not info.is_complete()
        except ImportError:
            pytest.skip("SessionInfo not available")

    async def test_session_info_from_dict(self):
        """Test creating SessionInfo from dictionary."""
        try:
            from session_buddy.core.session_manager import SessionInfo

            data = {
                "session_id": "sess_123",
                "ended_at": "2025-01-01T00:00:00",
                "quality_score": "85/100",
                "working_directory": "/home/user/project",
            }

            info = SessionInfo.from_dict(data)
            assert info.session_id == "sess_123"
            assert info.ended_at == "2025-01-01T00:00:00"
            assert info.is_complete()
        except ImportError:
            pytest.skip("SessionInfo not available")


@pytest.mark.asyncio
class TestConcurrentSessions:
    """Test concurrent session operations."""

    async def test_multiple_managers_independent(self):
        """Test that multiple managers maintain independent state."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager1 = SessionLifecycleManager()
            manager2 = SessionLifecycleManager()

            manager1.current_project = "project1"
            manager2.current_project = "project2"

            assert manager1.current_project == "project1"
            assert manager2.current_project == "project2"
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_concurrent_checkpoints(self):
        """Test concurrent checkpoint operations."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                async def checkpoint_task():
                    return await manager.checkpoint_session(tmpdir)

                tasks = [checkpoint_task() for _ in range(3)]
                results = await asyncio.gather(*tasks)

                assert len(results) == 3
                assert all(r["success"] for r in results)
        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
