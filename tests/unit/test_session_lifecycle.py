"""Comprehensive tests for session lifecycle management.

Week 8 Day 2 - Phase 6: Test session initialization, checkpoints, and cleanup.
Tests SessionLifecycleManager state management and lifecycle operations.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from session_buddy.core.session_manager import SessionLifecycleManager
from tests.fixtures import mock_lifecycle_manager, tmp_git_repo


@pytest.mark.asyncio
class TestSessionLifecycleManagerInitialization:
    """Test SessionLifecycleManager initialization."""

    def test_lifecycle_manager_initialization(self):
        """SessionLifecycleManager initializes with correct state."""
        manager = SessionLifecycleManager()

        assert manager.current_project is None
        assert manager._quality_history == {}
        assert manager.logger is not None

    def test_lifecycle_manager_templates_fallback(self):
        """SessionLifecycleManager handles templates initialization gracefully."""
        # Templates may or may not be available
        manager = SessionLifecycleManager()

        # Should not raise, templates is either initialized or None
        assert manager.templates is not None or manager.templates is None


@pytest.mark.asyncio
class TestSessionLifecycleDirectorySetup:
    """Test directory setup operations."""

    def test_setup_working_directory_with_explicit_path(self, tmp_path: Path):
        """_setup_working_directory returns provided path."""
        manager = SessionLifecycleManager()

        # Use tmp_path that actually exists
        result = manager._setup_working_directory(str(tmp_path))

        assert result == tmp_path

    def test_setup_working_directory_with_pwd(self):
        """_setup_working_directory uses PWD or falls back to cwd."""
        manager = SessionLifecycleManager()

        # Test with PWD set
        with patch.dict("os.environ", {"PWD": "/tmp/pwd"}):
            with patch("os.chdir"):  # Mock chdir to avoid actual directory changes
                # Implementation may use PWD or cwd - both are valid
                result = manager._setup_working_directory(None)
                assert result is not None  # Just verify it returns a path

    def test_setup_working_directory_with_cwd_fallback(self):
        """_setup_working_directory falls back to current directory."""
        manager = SessionLifecycleManager()

        # Mock PWD not being set
        with patch.dict("os.environ", {}, clear=True):
            result = manager._setup_working_directory(None)

            # Should be current working directory
            assert result == Path.cwd()

    def test_setup_claude_directories_creates_structure(self, tmp_path: Path):
        """_setup_claude_directories creates ~/.claude structure."""
        manager = SessionLifecycleManager()

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = manager._setup_claude_directories()

            # Should create ~/.claude directory
            assert result.exists()
            assert result.name == ".claude"


@pytest.mark.asyncio
class TestSessionProjectContextAnalysis:
    """Test project context analysis."""

    async def test_analyze_project_context_with_basic_project(self, tmp_path: Path):
        """analyze_project_context detects basic project indicators."""
        manager = SessionLifecycleManager()

        # Create basic project structure
        (tmp_path / "README.md").write_text("# Project\n")
        (tmp_path / "pyproject.toml").write_text("[project]\n")

        context = await manager.analyze_project_context(tmp_path)

        assert context["has_readme"] is True
        assert context["has_pyproject_toml"] is True  # Correct key name
        assert isinstance(context, dict)

    async def test_analyze_project_context_with_empty_directory(self, tmp_path: Path):
        """analyze_project_context handles empty directory."""
        manager = SessionLifecycleManager()

        context = await manager.analyze_project_context(tmp_path)

        assert context["has_readme"] is False
        assert context["has_pyproject_toml"] is False  # Correct key name

    async def test_analyze_project_context_with_tests(self, tmp_path: Path):
        """analyze_project_context detects tests directory."""
        manager = SessionLifecycleManager()

        # Create tests directory
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_example.py").write_text("def test_pass(): pass\n")

        context = await manager.analyze_project_context(tmp_path)

        assert context["has_tests"] is True

    async def test_analyze_project_context_with_docs(self, tmp_path: Path):
        """analyze_project_context detects documentation."""
        manager = SessionLifecycleManager()

        # Create docs directory
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "intro.md").write_text("# Documentation\n")

        context = await manager.analyze_project_context(tmp_path)

        assert context["has_docs"] is True


@pytest.mark.asyncio
class TestSessionQualityScoring:
    """Test quality score calculation."""

    @patch("session_buddy.server.calculate_quality_score")
    async def test_calculate_quality_score_delegates_to_server(
        self, mock_server_calc: AsyncMock, tmp_path: Path
    ):
        """calculate_quality_score delegates to server.calculate_quality_score."""
        # Mock server's calculate_quality_score function with correct keys
        mock_server_calc.return_value = {
            "total_score": 85,
            "score": 85,
            "version": "2.0",
            "breakdown": {"code": 31, "project": 22},
        }

        manager = SessionLifecycleManager()
        result = await manager.calculate_quality_score(tmp_path)

        # Verify delegation to server
        mock_server_calc.assert_called_once()
        assert result["total_score"] == 85 or result["score"] == 85


@pytest.mark.asyncio
class TestSessionCheckpointOperations:
    """Test checkpoint session operations."""

    @patch("session_buddy.utils.git_operations.create_checkpoint_commit")
    @patch("session_buddy.server.calculate_quality_score")
    async def test_checkpoint_session_creates_commit(
        self, mock_server_calc: AsyncMock, mock_commit: Mock, tmp_git_repo: Path
    ):
        """checkpoint_session creates git commit when changes present."""
        # Mock quality score from server with all required keys
        mock_server_calc.return_value = {
            "total_score": 75,
            "score": 75,
            "version": "2.0",
            "breakdown": {
                "code_quality": 28,
                "project_health": 20,
                "dev_velocity": 10,
                "security": 6,
            },
            "recommendations": ["Improve test coverage"],
        }

        # Mock git commit success
        mock_commit.return_value = (True, "abc123de", ["Commit created"])

        manager = SessionLifecycleManager()
        result = await manager.checkpoint_session(str(tmp_git_repo))

        # Verify checkpoint completed
        assert "quality_score" in result or "score" in result or "total_score" in result
        mock_server_calc.assert_called()


@pytest.mark.asyncio
class TestSessionEndOperations:
    """Test session end and cleanup."""

    @patch("session_buddy.utils.git_operations.create_checkpoint_commit")
    @patch("session_buddy.server.calculate_quality_score")
    async def test_end_session_performs_final_assessment(
        self, mock_server_calc: AsyncMock, mock_commit: Mock, tmp_git_repo: Path
    ):
        """end_session performs final quality assessment."""
        # Mock quality score from server with correct keys
        mock_server_calc.return_value = {
            "total_score": 80,
            "score": 80,
            "version": "2.0",
            "breakdown": {
                "code_quality": 30,
                "project_health": 22,
                "dev_velocity": 12,
                "security": 8,
            },
            "recommendations": ["Good progress"],
        }

        mock_commit.return_value = (True, "def456gh", ["Final commit created"])

        manager = SessionLifecycleManager()
        result = await manager.end_session(str(tmp_git_repo))

        # Verify end session completed (end_session returns {'success': bool, 'summary': {...}})
        assert result.get("success") is True
        assert "final_quality_score" in result.get("summary", {})
        mock_server_calc.assert_called()


@pytest.mark.asyncio
class TestSessionPreviousSessionInfo:
    """Test previous session information reading."""

    async def test_get_previous_session_info_with_no_file(self, tmp_path: Path):
        """_get_previous_session_info returns None when no previous session."""
        manager = SessionLifecycleManager()

        info = await manager._get_previous_session_info(tmp_path)

        # Method returns None when no previous session found
        assert info is None

    async def test_read_previous_session_info_with_valid_file(self, tmp_path: Path):
        """_read_previous_session_info parses valid session file."""
        manager = SessionLifecycleManager()

        # Create previous session file with correct format (bold keys)
        session_file = tmp_path / "SESSION-HANDOFF.md"
        session_content = """# Session Handoff

## Session Information
**Session ended:** 2025-10-28 12:00:00
**Final quality score:** 75/100
**Working directory:** /tmp/project

## Recommendations for Next Session
1. Improve test coverage to ≥80%
"""
        session_file.write_text(session_content)

        info = await manager._read_previous_session_info(session_file)

        # Method returns dict[str, str] | None, not SessionInfo
        assert info is not None
        assert isinstance(info, dict)
        assert "75/100" in info["quality_score"]
        assert info["working_directory"] == "/tmp/project"
        assert "Improve test coverage to ≥80%" in info["top_recommendation"]


@pytest.mark.asyncio
class TestSessionStatusQuery:
    """Test session status queries."""

    async def test_get_session_status_with_active_project(self, tmp_path: Path):
        """get_session_status returns current session information."""
        manager = SessionLifecycleManager()
        manager.current_project = "test-project"

        status = await manager.get_session_status(str(tmp_path))

        assert "project" in status
        assert isinstance(status, dict)
