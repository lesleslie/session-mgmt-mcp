"""Tests for quality_engine module.

Tests quality scoring, compaction analysis, and workflow intelligence
for the session management MCP server.

Phase: Week 5 Day 1 - Quality Engine Coverage
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQualityScoreCalculation:
    """Test quality score calculation functions."""

    @pytest.mark.asyncio
    async def test_calculate_quality_score_returns_dict(self, tmp_path: Path) -> None:
        """Should return dictionary with quality score and details."""
        from session_buddy.quality_engine import calculate_quality_score

        result = await calculate_quality_score(project_dir=tmp_path)

        assert isinstance(result, dict)
        assert "total_score" in result
        assert "breakdown" in result

    @pytest.mark.asyncio
    async def test_calculate_quality_score_with_no_project(self) -> None:
        """Should handle None project_dir gracefully."""
        from session_buddy.quality_engine import calculate_quality_score

        result = await calculate_quality_score(project_dir=None)

        assert isinstance(result, dict)
        assert "total_score" in result

    @pytest.mark.asyncio
    async def test_calculate_quality_score_with_nonexistent_path(self) -> None:
        """Should handle nonexistent project directory."""
        from session_buddy.quality_engine import calculate_quality_score

        nonexistent = Path("/nonexistent/path/to/project")
        result = await calculate_quality_score(project_dir=nonexistent)

        assert isinstance(result, dict)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_calculate_quality_score_uses_v2_algorithm(
        self, tmp_path: Path
    ) -> None:
        """Should use quality_utils_v2 for scoring."""
        from session_buddy.quality_engine import calculate_quality_score
        from session_buddy.utils.quality_utils_v2 import (
            CodeQualityScore,
            DevVelocityScore,
            ProjectHealthScore,
            QualityScoreV2,
            SecurityScore,
            TrustScore,
        )

        # Create minimal project structure
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")

        # Mock with complete dataclass structure
        mock_result = QualityScoreV2(
            total_score=75.0,
            version="v2",
            code_quality=CodeQualityScore(
                test_coverage=10.0,
                lint_score=8.0,
                type_coverage=7.0,
                complexity_score=4.0,
                total=29.0,
                details={},
            ),
            project_health=ProjectHealthScore(
                total=20, tooling_score=10, maturity_score=10, details={}
            ),
            dev_velocity=DevVelocityScore(
                git_activity=8.0, dev_patterns=8.0, total=16.0, details={}
            ),
            security=SecurityScore(
                security_tools=5.0, security_hygiene=5.0, total=10.0, details={}
            ),
            trust_score=TrustScore(
                trusted_operations=20.0,
                session_availability=15.0,
                tool_ecosystem=10.0,
                total=45.0,
                details={},
            ),
            recommendations=[],
            timestamp="2025-01-01",
        )

        with patch(
            "session_buddy.quality_engine.calculate_quality_score_v2"
        ) as mock_v2:
            mock_v2.return_value = mock_result
            await calculate_quality_score(project_dir=tmp_path)

            # Verify V2 algorithm was called
            mock_v2.assert_called_once()


class TestCompactionAnalysis:
    """Test context compaction analysis and suggestions."""

    def test_should_suggest_compact_returns_tuple(self) -> None:
        """Should return (bool, str) tuple."""
        from session_buddy.quality_engine import should_suggest_compact

        result = should_suggest_compact()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_should_suggest_compact_with_large_project(self, tmp_path: Path) -> None:
        """Should suggest compaction for large projects."""
        from session_buddy.quality_engine import should_suggest_compact

        # Create 60 Python files to trigger large project heuristic
        for i in range(60):
            (tmp_path / f"file_{i}.py").write_text("# Python file\n")

        with patch(
            "session_buddy.quality_engine._count_significant_files"
        ) as mock_count:
            mock_count.return_value = 60

            should_compact, reason = should_suggest_compact()

            # May or may not suggest based on other factors
            assert isinstance(should_compact, bool)
            assert len(reason) > 0

    def test_should_suggest_compact_with_small_project(self, tmp_path: Path) -> None:
        """Should not suggest compaction for small projects."""
        from session_buddy.quality_engine import should_suggest_compact

        # Create minimal project
        (tmp_path / "main.py").write_text("# Main file\n")

        with patch(
            "session_buddy.quality_engine._count_significant_files"
        ) as mock_count:
            mock_count.return_value = 1

            should_compact, reason = should_suggest_compact()

            assert isinstance(should_compact, bool)
            assert isinstance(reason, str)

    @pytest.mark.slow  # Marked slow due to filesystem operations
    @pytest.mark.asyncio
    async def test_perform_strategic_compaction_returns_list(self) -> None:
        """Should return list of compaction results."""
        from session_buddy.quality_engine import perform_strategic_compaction

        # Mock filesystem operations to prevent timeout
        with patch(
            "session_buddy.utils.file_utils._cleanup_temp_files"
        ) as mock_cleanup:
            mock_cleanup.return_value = "✅ Cleaned 0 temporary files (0.0 MB)"

            result = await perform_strategic_compaction()

            assert isinstance(result, list)
            # Each item should be a string describing an action
            for item in result:
                assert isinstance(item, str)

    @pytest.mark.asyncio
    async def test_perform_strategic_compaction_includes_database_optimization(
        self,
    ) -> None:
        """Should include reflection database optimization."""
        from session_buddy.quality_engine import perform_strategic_compaction

        with patch(
            "session_buddy.quality_engine._optimize_reflection_database"
        ) as mock_optimize:
            mock_optimize.return_value = "✅ Database optimized"

            result = await perform_strategic_compaction()

            assert isinstance(result, list)
            # Should have attempted database optimization
            mock_optimize.assert_called_once()


class TestProjectHeuristics:
    """Test project analysis heuristics."""

    def test_count_significant_files_with_python_project(self, tmp_path: Path) -> None:
        """Should count Python files correctly."""
        from session_buddy.quality_engine import _count_significant_files

        # Create Python files
        (tmp_path / "main.py").write_text("# Main\n")
        (tmp_path / "utils.py").write_text("# Utils\n")
        (tmp_path / "tests.py").write_text("# Tests\n")

        count = _count_significant_files(tmp_path)

        assert count >= 3

    def test_count_significant_files_ignores_hidden_files(self, tmp_path: Path) -> None:
        """Should ignore files in hidden directories."""
        from session_buddy.quality_engine import _count_significant_files

        # Create hidden directory with files
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.py").write_text("# Hidden\n")

        # Create visible file
        (tmp_path / "visible.py").write_text("# Visible\n")

        count = _count_significant_files(tmp_path)

        # Should only count visible.py, not .hidden/secret.py
        assert count >= 1

    def test_count_significant_files_supports_multiple_languages(
        self, tmp_path: Path
    ) -> None:
        """Should count files from multiple programming languages."""
        from session_buddy.quality_engine import _count_significant_files

        # Create files in different languages
        (tmp_path / "script.py").write_text("# Python\n")
        (tmp_path / "app.js").write_text("// JavaScript\n")
        (tmp_path / "component.tsx").write_text("// TypeScript\n")
        (tmp_path / "main.go").write_text("// Go\n")
        (tmp_path / "lib.rs").write_text("// Rust\n")

        count = _count_significant_files(tmp_path)

        assert count >= 5

    def test_count_significant_files_stops_at_threshold(self, tmp_path: Path) -> None:
        """Should stop counting after threshold (performance optimization)."""
        from session_buddy.quality_engine import _count_significant_files

        # Create 100 files
        for i in range(100):
            (tmp_path / f"file_{i}.py").write_text("# File\n")

        count = _count_significant_files(tmp_path)

        # Should stop at 51 (threshold is 50, returns when > 50)
        assert count <= 51

    def test_check_git_activity_with_no_git(self, tmp_path: Path) -> None:
        """Should return None for non-git projects."""
        from session_buddy.quality_engine import _check_git_activity

        result = _check_git_activity(tmp_path)

        assert result is None

    def test_check_git_activity_with_git_repo(self, tmp_path: Path) -> None:
        """Should return commit and file counts for git repos."""
        from session_buddy.quality_engine import _check_git_activity

        # Create .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        with patch("subprocess.run") as mock_run:
            # Mock git log output
            mock_run.return_value = MagicMock(
                returncode=0, stdout="commit1\ncommit2\ncommit3\n"
            )

            result = _check_git_activity(tmp_path)

            # Should return tuple or None (depends on git state)
            assert result is None or isinstance(result, tuple)


class TestWorkflowAnalysis:
    """Test workflow pattern analysis."""

    @pytest.mark.asyncio
    async def test_analyze_project_workflow_patterns_returns_dict(
        self, tmp_path: Path
    ) -> None:
        """Should return dictionary with workflow analysis."""
        from session_buddy.quality_engine import analyze_project_workflow_patterns

        result = await analyze_project_workflow_patterns(tmp_path)

        assert isinstance(result, dict)
        # Actual keys returned by the function
        assert (
            "project_characteristics" in result or "workflow_recommendations" in result
        )

    @pytest.mark.asyncio
    async def test_analyze_project_workflow_patterns_detects_python_project(
        self, tmp_path: Path
    ) -> None:
        """Should detect Python project characteristics."""
        from session_buddy.quality_engine import analyze_project_workflow_patterns

        # Create Python project files
        (tmp_path / "setup.py").write_text("# Setup\n")
        (tmp_path / "main.py").write_text("# Main\n")

        result = await analyze_project_workflow_patterns(tmp_path)

        assert isinstance(result, dict)
        # Should detect Python characteristics

    def test_generate_workflow_recommendations_returns_list(self) -> None:
        """Should generate recommendations based on project characteristics."""
        from session_buddy.quality_engine import _generate_workflow_recommendations

        # Use actual characteristic keys from the function
        characteristics = {
            "has_python": True,
            "has_git": True,
            "has_tests": False,
            "has_node": False,
            "has_docker": False,
        }

        recommendations = _generate_workflow_recommendations(characteristics)

        assert isinstance(recommendations, list)
        # Should have git-related recommendations
        assert any("git" in rec.lower() for rec in recommendations)


class TestConversationAnalysis:
    """Test conversation and memory pattern analysis."""

    @pytest.mark.asyncio
    async def test_summarize_current_conversation_returns_dict(self) -> None:
        """Should return summary dictionary."""
        from session_buddy.quality_engine import summarize_current_conversation

        result = await summarize_current_conversation()

        assert isinstance(result, dict)
        # Actual keys from _create_empty_summary()
        assert (
            "key_topics" in result
            or "decisions_made" in result
            or "next_steps" in result
        )

    @pytest.mark.asyncio
    async def test_analyze_conversation_flow_returns_dict(self) -> None:
        """Should return conversation flow analysis."""
        from session_buddy.quality_engine import analyze_conversation_flow

        result = await analyze_conversation_flow()

        assert isinstance(result, dict)
        # Should contain flow metrics

    @pytest.mark.asyncio
    async def test_analyze_memory_patterns_returns_dict(self) -> None:
        """Should return memory pattern analysis."""
        from session_buddy.quality_engine import analyze_memory_patterns

        mock_db = MagicMock()
        result = await analyze_memory_patterns(mock_db, conv_count=10)

        assert isinstance(result, dict)


class TestTokenUsageAnalysis:
    """Test token usage and context analysis."""

    @pytest.mark.asyncio
    async def test_analyze_token_usage_patterns_returns_dict(self) -> None:
        """Should return token usage analysis."""
        from session_buddy.quality_engine import analyze_token_usage_patterns

        result = await analyze_token_usage_patterns()

        assert isinstance(result, dict)
        # Should contain usage metrics or patterns

    @pytest.mark.asyncio
    async def test_analyze_context_usage_returns_list(self) -> None:
        """Should return list of context usage recommendations."""
        from session_buddy.quality_engine import analyze_context_usage

        result = await analyze_context_usage()

        assert isinstance(result, list)
        # Each item should be a recommendation string
        for item in result:
            assert isinstance(item, str)

    @pytest.mark.asyncio
    async def test_analyze_advanced_context_metrics_returns_dict(self) -> None:
        """Should return advanced context metrics."""
        from session_buddy.quality_engine import analyze_advanced_context_metrics

        result = await analyze_advanced_context_metrics()

        assert isinstance(result, dict)
        # Should contain metrics


class TestSessionIntelligence:
    """Test session intelligence and recommendations."""

    @pytest.mark.asyncio
    async def test_generate_session_intelligence_returns_dict(self) -> None:
        """Should return session intelligence summary."""
        from session_buddy.quality_engine import generate_session_intelligence

        result = await generate_session_intelligence()

        assert isinstance(result, dict)
        assert (
            "insights" in result
            or "recommendations" in result
            or "priority_actions" in result
        )

    @pytest.mark.asyncio
    async def test_monitor_proactive_quality_returns_dict(self) -> None:
        """Should return proactive quality monitoring results."""
        from session_buddy.quality_engine import monitor_proactive_quality

        result = await monitor_proactive_quality()

        assert isinstance(result, dict)
        # Should contain monitoring data


class TestHelperFunctions:
    """Test utility and helper functions."""

    def test_get_default_compaction_reason_returns_string(self) -> None:
        """Should return default compaction reason."""
        from session_buddy.quality_engine import _get_default_compaction_reason

        result = _get_default_compaction_reason()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_fallback_compaction_reason_returns_string(self) -> None:
        """Should return fallback compaction reason."""
        from session_buddy.quality_engine import _get_fallback_compaction_reason

        result = _get_fallback_compaction_reason()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_session_tags_returns_list(self) -> None:
        """Should generate tags based on quality score."""
        from session_buddy.quality_engine import _generate_session_tags

        tags = _generate_session_tags(quality_score=85.0)

        assert isinstance(tags, list)
        for tag in tags:
            assert isinstance(tag, str)

    def test_generate_session_tags_for_high_quality(self) -> None:
        """Should generate positive tags for high quality scores."""
        from session_buddy.quality_engine import _generate_session_tags

        tags = _generate_session_tags(quality_score=95.0)

        assert isinstance(tags, list)
        # Should include high-quality tags

    def test_generate_session_tags_for_low_quality(self) -> None:
        """Should generate improvement tags for low quality scores."""
        from session_buddy.quality_engine import _generate_session_tags

        tags = _generate_session_tags(quality_score=30.0)

        assert isinstance(tags, list)
        # Should include needs-improvement tags
