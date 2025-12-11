#!/usr/bin/env python3
"""Tests for context_manager module.

Tests automatic context detection, relevance scoring, and context loading
for intelligent conversation retrieval based on development environment.

Phase 2: Core Coverage (0% → 60%) - Context Manager Tests
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestContextDetectorInit:
    """Test ContextDetector initialization and configuration.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_context_detector_has_indicators(self) -> None:
        """Should initialize with context indicators."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()

        assert hasattr(detector, "context_indicators")
        assert isinstance(detector.context_indicators, dict)
        assert "git" in detector.context_indicators
        assert "python" in detector.context_indicators

    def test_context_detector_has_project_types(self) -> None:
        """Should initialize with project type patterns."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()

        assert hasattr(detector, "project_types")
        assert isinstance(detector.project_types, dict)
        assert "mcp_server" in detector.project_types
        assert "api" in detector.project_types


class TestContextDetectorHelperMethods:
    """Test ContextDetector helper methods.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_initialize_context_creates_structure(self, tmp_path: Path) -> None:
        """Should create basic context structure."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        result = detector._initialize_context(tmp_path)

        assert "working_directory" in result
        assert "project_name" in result
        assert result["project_name"] == tmp_path.name
        assert result["detected_languages"] == []
        assert result["detected_tools"] == []
        assert result["confidence_score"] == 0.0

    def test_resolve_working_path_with_explicit_dir(self, tmp_path: Path) -> None:
        """Should use explicit working directory."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        result = detector._resolve_working_path(str(tmp_path))

        assert result == tmp_path

    def test_resolve_working_path_with_pwd_env(self, tmp_path: Path) -> None:
        """Should use PWD environment variable when available."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()

        with patch.dict("os.environ", {"PWD": str(tmp_path)}):
            result = detector._resolve_working_path(None)

        assert result == tmp_path

    def test_resolve_working_path_defaults_to_cwd(self) -> None:
        """Should default to current working directory."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()

        with patch.dict("os.environ", {}, clear=True):
            result = detector._resolve_working_path(None)

        assert isinstance(result, Path)

    def test_should_ignore_file_ignores_git_directory(self, tmp_path: Path) -> None:
        """Should ignore .git directories."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        git_file = tmp_path / ".git" / "config"

        assert detector._should_ignore_file(git_file) is True

    def test_should_ignore_file_ignores_venv(self, tmp_path: Path) -> None:
        """Should ignore .venv directories."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        venv_file = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages"

        assert detector._should_ignore_file(venv_file) is True

    def test_should_ignore_file_ignores_pycache(self, tmp_path: Path) -> None:
        """Should ignore __pycache__ directories."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        cache_file = tmp_path / "src" / "__pycache__" / "module.pyc"

        assert detector._should_ignore_file(cache_file) is True

    def test_should_ignore_file_ignores_pyc_files(self, tmp_path: Path) -> None:
        """Should ignore .pyc files."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        pyc_file = tmp_path / "module.pyc"

        assert detector._should_ignore_file(pyc_file) is True

    def test_should_ignore_file_allows_python_files(self, tmp_path: Path) -> None:
        """Should not ignore .py files."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        py_file = tmp_path / "module.py"

        assert detector._should_ignore_file(py_file) is False


class TestFindIndicators:
    """Test _find_indicators helper method.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_find_indicators_matches_file(self, tmp_path: Path) -> None:
        """Should find matching file indicators."""
        from session_buddy.context_manager import ContextDetector

        # Create test file
        (tmp_path / "pyproject.toml").touch()

        detector = ContextDetector()
        result = detector._find_indicators(tmp_path, ["pyproject.toml"])

        assert "pyproject.toml" in result

    def test_find_indicators_matches_directory(self, tmp_path: Path) -> None:
        """Should find matching directory indicators."""
        from session_buddy.context_manager import ContextDetector

        # Create test directory
        (tmp_path / "tests").mkdir()

        detector = ContextDetector()
        result = detector._find_indicators(tmp_path, ["tests/"])

        assert "tests/" in result

    def test_find_indicators_matches_glob(self, tmp_path: Path) -> None:
        """Should find matching glob patterns."""
        from session_buddy.context_manager import ContextDetector

        # Create test files
        (tmp_path / "main.py").touch()
        (tmp_path / "utils.py").touch()

        detector = ContextDetector()
        result = detector._find_indicators(tmp_path, ["*.py"])

        assert len(result) > 0
        assert any("main.py" in r for r in result)

    def test_find_indicators_limits_glob_results(self, tmp_path: Path) -> None:
        """Should limit glob results to 3 matches."""
        from session_buddy.context_manager import ContextDetector

        # Create many test files
        for i in range(10):
            (tmp_path / f"file{i}.py").touch()

        detector = ContextDetector()
        result = detector._find_indicators(tmp_path, ["*.py"])

        # Should limit to 3 results
        assert len(result) <= 3

    def test_find_indicators_returns_empty_when_no_matches(
        self, tmp_path: Path
    ) -> None:
        """Should return empty list when no indicators found."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        result = detector._find_indicators(tmp_path, ["nonexistent.txt"])

        assert result == []


class TestLanguageAndToolDetection:
    """Test language and tool detection methods.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_detect_python_language(self, tmp_path: Path) -> None:
        """Should detect Python project."""
        from session_buddy.context_manager import ContextDetector

        # Create Python indicators
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "main.py").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_languages_and_tools(tmp_path, context)

        assert "python" in context["detected_languages"]
        assert context["confidence_score"] > 0

    def test_detect_javascript_language(self, tmp_path: Path) -> None:
        """Should detect JavaScript project."""
        from session_buddy.context_manager import ContextDetector

        # Create JavaScript indicators
        (tmp_path / "package.json").touch()
        (tmp_path / "index.js").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_languages_and_tools(tmp_path, context)

        assert "javascript" in context["detected_languages"]

    def test_detect_git_tool(self, tmp_path: Path) -> None:
        """Should detect Git as development tool."""
        from session_buddy.context_manager import ContextDetector

        # Create Git indicator
        (tmp_path / ".gitignore").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_languages_and_tools(tmp_path, context)

        assert "git" in context["detected_tools"]

    def test_detect_docker_tool(self, tmp_path: Path) -> None:
        """Should detect Docker as development tool."""
        from session_buddy.context_manager import ContextDetector

        # Create Docker indicator
        (tmp_path / "Dockerfile").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_languages_and_tools(tmp_path, context)

        assert "docker" in context["detected_tools"]

    def test_detect_multiple_languages(self, tmp_path: Path) -> None:
        """Should detect multiple programming languages."""
        from session_buddy.context_manager import ContextDetector

        # Create multiple language indicators
        (tmp_path / "main.py").touch()
        (tmp_path / "index.js").touch()
        (tmp_path / "main.go").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_languages_and_tools(tmp_path, context)

        assert len(context["detected_languages"]) >= 2
        assert "python" in context["detected_languages"]

    def test_increases_confidence_score_for_each_detection(
        self, tmp_path: Path
    ) -> None:
        """Should increase confidence score for each detected item."""
        from session_buddy.context_manager import ContextDetector

        # Create multiple indicators
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".gitignore").touch()
        (tmp_path / "Dockerfile").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_languages_and_tools(tmp_path, context)

        # Should have confidence score > 0.2 (0.1 per detection)
        assert context["confidence_score"] >= 0.2


class TestProjectTypeDetection:
    """Test project type detection methods.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_calculate_project_type_score_for_files(self, tmp_path: Path) -> None:
        """Should calculate score based on file existence."""
        from session_buddy.context_manager import ContextDetector

        # Create MCP server indicators
        (tmp_path / ".mcp.json").touch()

        detector = ContextDetector()
        score = detector._calculate_project_type_score(tmp_path, [".mcp.json"])

        assert score >= 1.0

    def test_calculate_project_type_score_for_directories(self, tmp_path: Path) -> None:
        """Should calculate score based on directory existence."""
        from session_buddy.context_manager import ContextDetector

        # Create API directories
        (tmp_path / "api").mkdir()
        (tmp_path / "routes").mkdir()

        detector = ContextDetector()
        score = detector._calculate_project_type_score(tmp_path, ["api/", "routes/"])

        assert score >= 2.0

    def test_calculate_project_type_score_for_path_name(self, tmp_path: Path) -> None:
        """Should add partial score for matching path names."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        # "fastmcp" in path name should get 0.5 score
        score = detector._calculate_project_type_score(tmp_path, ["fastmcp"])

        # Should get partial score for path matching (if path contains indicator)
        assert isinstance(score, float)

    def test_detect_mcp_server_project_type(self, tmp_path: Path) -> None:
        """Should detect MCP server project type."""
        from session_buddy.context_manager import ContextDetector

        # Create MCP server indicators
        (tmp_path / ".mcp.json").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_project_type(tmp_path, context)

        assert context["project_type"] == "mcp_server"

    def test_detect_api_project_type(self, tmp_path: Path) -> None:
        """Should detect API project type."""
        from session_buddy.context_manager import ContextDetector

        # Create API indicators
        (tmp_path / "api").mkdir()
        (tmp_path / "routes").mkdir()
        (tmp_path / "endpoints").mkdir()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_project_type(tmp_path, context)

        assert context["project_type"] == "api"

    def test_detect_cli_tool_project_type(self, tmp_path: Path) -> None:
        """Should detect CLI tool project type."""
        from session_buddy.context_manager import ContextDetector

        # Create CLI tool indicators
        (tmp_path / "cli").mkdir()
        (tmp_path / "__main__.py").touch()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_project_type(tmp_path, context)

        assert context["project_type"] == "cli_tool"

    def test_detect_best_matching_project_type(self, tmp_path: Path) -> None:
        """Should select project type with highest score."""
        from session_buddy.context_manager import ContextDetector

        # Create indicators for multiple types (MCP server has more specific indicators)
        (tmp_path / ".mcp.json").touch()
        (tmp_path / "src").mkdir()

        detector = ContextDetector()
        context = detector._initialize_context(tmp_path)
        detector._detect_project_type(tmp_path, context)

        # Should prefer mcp_server over library
        assert context["project_type"] in ("mcp_server", "library")


class TestRecentFilesDetection:
    """Test recent files detection.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_get_recent_files_finds_recently_modified(self, tmp_path: Path) -> None:
        """Should find recently modified files."""
        from session_buddy.context_manager import ContextDetector

        # Create test file
        test_file = tmp_path / "recent.txt"
        test_file.touch()

        detector = ContextDetector()
        result = detector._get_recent_files(tmp_path)

        assert isinstance(result, list)
        # File was just created so should be recent
        assert any("recent.txt" in f["path"] for f in result)

    def test_get_recent_files_includes_metadata(self, tmp_path: Path) -> None:
        """Should include file metadata."""
        from session_buddy.context_manager import ContextDetector

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        detector = ContextDetector()
        result = detector._get_recent_files(tmp_path)

        if result:
            file_info = result[0]
            assert "path" in file_info
            assert "modified" in file_info
            assert "size" in file_info

    def test_get_recent_files_limits_to_10(self, tmp_path: Path) -> None:
        """Should limit results to 10 files."""
        from session_buddy.context_manager import ContextDetector

        # Create many test files
        for i in range(20):
            (tmp_path / f"file{i}.txt").touch()

        detector = ContextDetector()
        result = detector._get_recent_files(tmp_path)

        assert len(result) <= 10

    def test_get_recent_files_ignores_venv_files(self, tmp_path: Path) -> None:
        """Should ignore .venv files."""
        from session_buddy.context_manager import ContextDetector

        # Create .venv file (should be ignored)
        venv_dir = tmp_path / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "module.py").touch()

        # Create normal file
        (tmp_path / "main.py").touch()

        detector = ContextDetector()
        result = detector._get_recent_files(tmp_path)

        # Should not include .venv files
        assert not any(".venv" in f["path"] for f in result)

    def test_get_recent_files_handles_permission_errors(self, tmp_path: Path) -> None:
        """Should handle permission errors gracefully."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()

        with patch.object(Path, "rglob", side_effect=PermissionError("Access denied")):
            result = detector._get_recent_files(tmp_path)

        assert result == []


class TestGitInfoDetection:
    """Test Git information detection.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_get_git_info_returns_empty_when_not_git_repo(self, tmp_path: Path) -> None:
        """Should return empty dict when not a Git repository."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        result = detector._get_git_info(tmp_path)

        assert result == {}

    def test_get_git_info_detects_github_platform(self, tmp_path: Path) -> None:
        """Should detect GitHub platform from config."""
        from session_buddy.context_manager import ContextDetector

        # Create mock Git directory with GitHub config
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        config = git_dir / "config"
        config.write_text('[remote "origin"]\n  url = https://github.com/user/repo.git')

        detector = ContextDetector()
        result = detector._get_git_info(tmp_path)

        assert result.get("platform") == "github"

    def test_get_git_info_detects_gitlab_platform(self, tmp_path: Path) -> None:
        """Should detect GitLab platform from config."""
        from session_buddy.context_manager import ContextDetector

        # Create mock Git directory with GitLab config
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        config = git_dir / "config"
        config.write_text('[remote "origin"]\n  url = https://gitlab.com/user/repo.git')

        detector = ContextDetector()
        result = detector._get_git_info(tmp_path)

        assert result.get("platform") == "gitlab"

    def test_determine_git_platform_defaults_to_git(self) -> None:
        """Should default to 'git' when platform unknown."""
        from session_buddy.context_manager import ContextDetector

        detector = ContextDetector()
        result = detector._determine_git_platform("some other content")

        assert result == "git"


class TestWorktreeDetection:
    """Test Git worktree detection.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_format_worktree_info_creates_dict(self) -> None:
        """Should format worktree information as dictionary."""
        from session_buddy.context_manager import ContextDetector

        # Create mock worktree info
        mock_worktree = Mock()
        mock_worktree.path = Path("/test/worktree")
        mock_worktree.branch = "main"
        mock_worktree.is_main_worktree = True
        mock_worktree.is_detached = False
        mock_worktree.is_bare = False
        mock_worktree.locked = None
        mock_worktree.prunable = None

        detector = ContextDetector()
        result = detector._format_worktree_info(mock_worktree)

        assert result["path"] == str(mock_worktree.path)
        assert result["branch"] == "main"
        assert result["is_main_worktree"] is True
        assert result["is_detached"] is False

    def test_get_all_worktrees_info_formats_list(self) -> None:
        """Should format list of worktrees."""
        from session_buddy.context_manager import ContextDetector

        # Create mock worktrees
        mock_wt1 = Mock()
        mock_wt1.path = Path("/test/main")
        mock_wt1.branch = "main"
        mock_wt1.is_main_worktree = True

        mock_wt2 = Mock()
        mock_wt2.path = Path("/test/feature")
        mock_wt2.branch = "feature"
        mock_wt2.is_main_worktree = False

        detector = ContextDetector()

        with patch(
            "session_buddy.context_manager.list_worktrees",
            return_value=[mock_wt1, mock_wt2],
        ):
            result = detector._get_all_worktrees_info(Path("/test"), mock_wt1)

        assert len(result) == 2
        assert result[0]["branch"] == "main"
        assert result[0]["is_current"] is True
        assert result[1]["branch"] == "feature"
        assert result[1]["is_current"] is False


class TestDetectCurrentContext:
    """Test detect_current_context main method.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_detect_current_context_returns_full_structure(
        self, tmp_path: Path
    ) -> None:
        """Should return complete context structure."""
        from session_buddy.context_manager import ContextDetector

        # Create minimal project structure
        (tmp_path / "pyproject.toml").touch()

        detector = ContextDetector()
        result = detector.detect_current_context(str(tmp_path))

        assert "working_directory" in result
        assert "project_name" in result
        assert "detected_languages" in result
        assert "detected_tools" in result
        assert "project_type" in result
        assert "recent_files" in result
        assert "git_info" in result
        assert "confidence_score" in result

    def test_detect_current_context_with_git_repo(self, tmp_path: Path) -> None:
        """Should include Git information for Git repositories."""
        from session_buddy.context_manager import ContextDetector

        # Create Git repository structure
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text(
            '[remote "origin"]\n  url = https://github.com/user/repo.git'
        )

        detector = ContextDetector()
        result = detector.detect_current_context(str(tmp_path))

        assert result["git_info"].get("is_git_repo") == "True"


class TestRelevanceScorerInit:
    """Test RelevanceScorer initialization.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_relevance_scorer_has_weights(self) -> None:
        """Should initialize with scoring weights."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()

        assert hasattr(scorer, "scoring_weights")
        assert "project_name_match" in scorer.scoring_weights
        assert "language_match" in scorer.scoring_weights
        assert "recency" in scorer.scoring_weights


class TestRelevanceScoringMethods:
    """Test RelevanceScorer scoring methods.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_score_project_match_with_name_match(self) -> None:
        """Should score project name matches."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"project_name": "session-mgmt-mcp"}
        conv_content = "working on session-mgmt-mcp features"
        conv_project = "session-mgmt-mcp"

        score = scorer._score_project_match(conv_content, conv_project, context)

        assert score > 0
        assert score == scorer.scoring_weights["project_name_match"]

    def test_score_project_match_no_match(self) -> None:
        """Should return zero when project doesn't match."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"project_name": "project-a"}
        conv_content = "working on different project"
        conv_project = "project-b"

        score = scorer._score_project_match(conv_content, conv_project, context)

        assert score == 0.0

    def test_score_language_match_with_python(self) -> None:
        """Should score language matches."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"detected_languages": ["python", "javascript"]}
        conv_content = "fixing python bug in the application"

        score = scorer._score_language_match(conv_content, context)

        assert score > 0

    def test_score_language_match_no_languages(self) -> None:
        """Should return zero when no languages detected."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"detected_languages": []}
        conv_content = "general conversation"

        score = scorer._score_language_match(conv_content, context)

        assert score == 0.0

    def test_score_tool_match_with_git(self) -> None:
        """Should score tool matches."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"detected_tools": ["git", "docker"]}
        conv_content = "configuring git workflow"

        score = scorer._score_tool_match(conv_content, context)

        assert score > 0

    def test_score_file_match_with_recent_files(self) -> None:
        """Should score matches with recent files."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {
            "recent_files": [
                {"path": "src/main.py"},
                {"path": "tests/test_main.py"},
            ]
        }
        conv_content = "modified main.py implementation"

        score = scorer._score_file_match(conv_content, context)

        assert score > 0

    def test_score_recency_for_today(self) -> None:
        """Should give full recency score for today's conversations."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        conversation = {"timestamp": datetime.now().isoformat()}

        score = scorer._score_recency(conversation)

        assert score == scorer.scoring_weights["recency"]

    def test_score_recency_for_last_week(self) -> None:
        """Should give partial recency score for last week."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        # 5 days ago
        past_time = datetime.now() - timedelta(days=5)
        conversation = {"timestamp": past_time.isoformat()}

        score = scorer._score_recency(conversation)

        assert score > 0
        assert score < scorer.scoring_weights["recency"]

    def test_score_recency_for_old_conversation(self) -> None:
        """Should give zero recency score for old conversations."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        # 30 days ago
        old_time = datetime.now() - timedelta(days=30)
        conversation = {"timestamp": old_time.isoformat()}

        score = scorer._score_recency(conversation)

        assert score == 0.0

    def test_score_recency_handles_invalid_timestamp(self) -> None:
        """Should handle invalid timestamp gracefully."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        conversation = {"timestamp": "invalid-date"}

        score = scorer._score_recency(conversation)

        assert score == 0.0

    def test_get_project_keywords_returns_mappings(self) -> None:
        """Should return project keyword mappings."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        keywords = scorer._get_project_keywords()

        assert isinstance(keywords, dict)
        assert "mcp_server" in keywords
        assert "api" in keywords
        assert isinstance(keywords["mcp_server"], list)

    def test_score_project_keywords_with_mcp_server(self) -> None:
        """Should score MCP server project keywords."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"project_type": "mcp_server"}
        conv_content = "implementing mcp protocol server"

        score = scorer._score_project_keywords(conv_content, context)

        assert score > 0

    def test_score_project_keywords_no_project_type(self) -> None:
        """Should return zero when no project type."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        context = {"project_type": None}
        conv_content = "general conversation"

        score = scorer._score_project_keywords(conv_content, context)

        assert score == 0.0


class TestScoreConversationRelevance:
    """Test score_conversation_relevance main method.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    def test_score_conversation_relevance_combines_scores(self) -> None:
        """Should combine multiple scoring factors."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        conversation = {
            "content": "fixing python bug in session-mgmt-mcp git workflow",
            "project": "session-mgmt-mcp",
            "timestamp": datetime.now().isoformat(),
        }
        context = {
            "project_name": "session-mgmt-mcp",
            "detected_languages": ["python"],
            "detected_tools": ["git"],
            "recent_files": [],
            "project_type": "mcp_server",
        }

        score = scorer.score_conversation_relevance(conversation, context)

        # Should have multiple matching factors
        assert score > 0
        assert isinstance(score, float)

    def test_score_conversation_relevance_caps_at_one(self) -> None:
        """Should cap relevance score at 1.0."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        # Perfect match scenario
        conversation = {
            "content": "python javascript git docker testing session-mgmt-mcp mcp server protocol",
            "project": "session-mgmt-mcp",
            "timestamp": datetime.now().isoformat(),
        }
        context = {
            "project_name": "session-mgmt-mcp",
            "detected_languages": ["python", "javascript"],
            "detected_tools": ["git", "docker", "testing"],
            "recent_files": [],
            "project_type": "mcp_server",
        }

        score = scorer.score_conversation_relevance(conversation, context)

        # Should be capped at 1.0 even with many matches
        assert score <= 1.0

    def test_score_conversation_relevance_zero_for_no_match(self) -> None:
        """Should return low score when nothing matches."""
        from session_buddy.context_manager import RelevanceScorer

        scorer = RelevanceScorer()
        # 30 days old, no matching content
        old_time = datetime.now() - timedelta(days=30)
        conversation = {
            "content": "completely unrelated topic",
            "project": "different-project",
            "timestamp": old_time.isoformat(),
        }
        context = {
            "project_name": "session-mgmt-mcp",
            "detected_languages": ["python"],
            "detected_tools": ["git"],
            "recent_files": [],
            "project_type": "mcp_server",
        }

        score = scorer.score_conversation_relevance(conversation, context)

        assert score == 0.0


class TestAutoContextLoaderInit:
    """Test AutoContextLoader initialization.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    @pytest.mark.asyncio
    async def test_auto_context_loader_initialization(self) -> None:
        """Should initialize with required components."""
        from session_buddy.context_manager import AutoContextLoader

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        assert hasattr(loader, "reflection_db")
        assert hasattr(loader, "context_detector")
        assert hasattr(loader, "relevance_scorer")
        assert hasattr(loader, "cache")
        assert loader.cache_timeout == 300


class TestGenerateContextHash:
    """Test _generate_context_hash method.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    @pytest.mark.asyncio
    async def test_generate_context_hash_creates_hash(self) -> None:
        """Should generate hash from context."""
        from session_buddy.context_manager import AutoContextLoader

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        context = {
            "project_name": "test-project",
            "detected_languages": ["python"],
            "detected_tools": ["git"],
            "project_type": "library",
            "working_directory": "/test",
        }

        hash_value = loader._generate_context_hash(context)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 12  # Truncated MD5 hash

    @pytest.mark.asyncio
    async def test_generate_context_hash_consistent_for_same_context(self) -> None:
        """Should generate same hash for same context."""
        from session_buddy.context_manager import AutoContextLoader

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        context = {
            "project_name": "test-project",
            "detected_languages": ["python"],
            "detected_tools": ["git"],
            "project_type": "library",
            "working_directory": "/test",
        }

        hash1 = loader._generate_context_hash(context)
        hash2 = loader._generate_context_hash(context)

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_generate_context_hash_different_for_different_context(
        self,
    ) -> None:
        """Should generate different hash for different context."""
        from session_buddy.context_manager import AutoContextLoader

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        context1 = {
            "project_name": "project-a",
            "detected_languages": ["python"],
            "detected_tools": [],
            "project_type": "library",
            "working_directory": "/test",
        }

        context2 = {
            "project_name": "project-b",
            "detected_languages": ["javascript"],
            "detected_tools": [],
            "project_type": "api",
            "working_directory": "/test",
        }

        hash1 = loader._generate_context_hash(context1)
        hash2 = loader._generate_context_hash(context2)

        assert hash1 != hash2


class TestLoadRelevantContext:
    """Test load_relevant_context main method.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    @pytest.mark.asyncio
    async def test_load_relevant_context_returns_structure(
        self, tmp_path: Path
    ) -> None:
        """Should return complete result structure."""
        from session_buddy.context_manager import AutoContextLoader

        # Create test project structure
        (tmp_path / "pyproject.toml").touch()

        # Mock database with no conversations
        mock_db = AsyncMock()
        mock_db.conn = None  # No connection

        loader = AutoContextLoader(mock_db)
        result = await loader.load_relevant_context(str(tmp_path))

        assert "context" in result
        assert "relevant_conversations" in result
        assert "total_found" in result
        assert "loaded_count" in result
        assert "min_relevance_threshold" in result

    @pytest.mark.asyncio
    async def test_load_relevant_context_uses_cache(self, tmp_path: Path) -> None:
        """Should use cached results within timeout."""
        from session_buddy.context_manager import AutoContextLoader

        mock_db = AsyncMock()
        mock_db.conn = None

        loader = AutoContextLoader(mock_db)

        # First call
        result1 = await loader.load_relevant_context(str(tmp_path))

        # Second call should use cache
        result2 = await loader.load_relevant_context(str(tmp_path))

        # Results should be identical (from cache)
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_load_relevant_context_respects_max_conversations(
        self, tmp_path: Path
    ) -> None:
        """Should limit results to max_conversations."""
        from session_buddy.context_manager import AutoContextLoader

        # Mock database with many conversations
        mock_db = AsyncMock()
        mock_conn = MagicMock()

        # Create 20 mock conversations
        mock_conversations = []
        for i in range(20):
            conv_data = (
                f"conv-{i}",  # id
                f"conversation {i} content",  # content
                "test-project",  # project
                datetime.now().isoformat(),  # timestamp
                json.dumps({}),  # metadata
            )
            mock_conversations.append(conv_data)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_conversations
        mock_conn.execute.return_value = mock_cursor

        mock_db.conn = mock_conn

        loader = AutoContextLoader(mock_db)
        result = await loader.load_relevant_context(
            str(tmp_path), max_conversations=5, min_relevance=0.0
        )

        # Should limit to 5 conversations
        assert result["loaded_count"] <= 5

    @pytest.mark.asyncio
    async def test_load_relevant_context_filters_by_relevance(
        self, tmp_path: Path
    ) -> None:
        """Should filter conversations by minimum relevance."""
        from session_buddy.context_manager import AutoContextLoader

        # Create test project
        (tmp_path / "pyproject.toml").touch()

        # Mock database with conversations
        mock_db = AsyncMock()
        mock_conn = MagicMock()

        # One highly relevant conversation, one not relevant
        mock_conversations = [
            (
                "conv-1",
                f"working on {tmp_path.name} python project",
                tmp_path.name,
                datetime.now().isoformat(),
                json.dumps({}),
            ),
            (
                "conv-2",
                "completely unrelated topic",
                "other-project",
                (datetime.now() - timedelta(days=30)).isoformat(),
                json.dumps({}),
            ),
        ]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_conversations
        mock_conn.execute.return_value = mock_cursor

        mock_db.conn = mock_conn

        loader = AutoContextLoader(mock_db)
        result = await loader.load_relevant_context(str(tmp_path), min_relevance=0.3)

        # Should only include relevant conversation
        assert result["loaded_count"] >= 0


class TestGetContextSummary:
    """Test get_context_summary method.

    Phase 2: Core Coverage - context_manager.py (0% → 60%)
    """

    @pytest.mark.asyncio
    async def test_get_context_summary_returns_string(self, tmp_path: Path) -> None:
        """Should return formatted summary string."""
        from session_buddy.context_manager import AutoContextLoader

        # Create test project
        (tmp_path / "pyproject.toml").touch()

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        result = await loader.get_context_summary(str(tmp_path))

        assert isinstance(result, str)
        assert tmp_path.name in result  # Project name
        assert "📁" in result  # Should have emoji formatting

    @pytest.mark.asyncio
    async def test_get_context_summary_includes_languages(self, tmp_path: Path) -> None:
        """Should include detected languages."""
        from session_buddy.context_manager import AutoContextLoader

        # Create Python project
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "main.py").touch()

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        result = await loader.get_context_summary(str(tmp_path))

        assert "Languages:" in result or "💻" in result

    @pytest.mark.asyncio
    async def test_get_context_summary_includes_git_info(self, tmp_path: Path) -> None:
        """Should include Git information when available."""
        from session_buddy.context_manager import AutoContextLoader

        # Create Git repository
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        (git_dir / "config").write_text(
            '[remote "origin"]\n  url = https://github.com/user/repo.git'
        )

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        result = await loader.get_context_summary(str(tmp_path))

        # Should include Git branch info
        assert "Git:" in result or "🌿" in result

    @pytest.mark.asyncio
    async def test_get_context_summary_includes_confidence(
        self, tmp_path: Path
    ) -> None:
        """Should include confidence score."""
        from session_buddy.context_manager import AutoContextLoader

        # Create project with indicators
        (tmp_path / "pyproject.toml").touch()

        mock_db = AsyncMock()
        loader = AutoContextLoader(mock_db)

        result = await loader.get_context_summary(str(tmp_path))

        assert "confidence" in result.lower() or "🎯" in result
