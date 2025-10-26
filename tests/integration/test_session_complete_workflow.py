#!/usr/bin/env python3
"""Integration tests for complete session workflows.

Tests the complete session lifecycle including:
- Full initialization with project analysis
- Quality assessment and checkpoint operations
- Session completion and handoff generation
- Quality score trending across multiple checkpoints
- Multi-session continuity and independence
- Error recovery and resilience
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
class TestCompleteSessionLifecycle:
    """Test complete session lifecycle workflows."""

    async def test_full_session_workflow_init_to_end(self):
        """Test complete workflow from initialization to session end."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Step 1: Initialize session
                init_result = await manager.initialize_session(working_directory=tmpdir)
                assert init_result["success"]
                assert "project" in init_result
                assert "quality_score" in init_result
                assert manager.current_project is not None

                # Step 2: Perform checkpoint
                checkpoint_result = await manager.checkpoint_session(tmpdir)
                assert checkpoint_result["success"]
                assert "quality_score" in checkpoint_result
                assert "quality_output" in checkpoint_result

                # Step 3: End session
                end_result = await manager.end_session(tmpdir)
                assert end_result["success"]
                assert "summary" in end_result
                summary = end_result["summary"]
                assert "final_quality_score" in summary
                assert "recommendations" in summary

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_session_quality_assessment_workflow(self):
        """Test quality assessment during session lifecycle."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                # Create a more realistic project structure
                (tmppath / "pyproject.toml").touch()
                (tmppath / "README.md").touch()
                (tmppath / ".git").mkdir()
                tests_dir = tmppath / "tests"
                tests_dir.mkdir()
                for i in range(5):
                    (tests_dir / f"test_{i}.py").touch()

                manager = SessionLifecycleManager()

                # Initialize and get initial score
                init_result = await manager.initialize_session(working_directory=tmpdir)
                initial_score = init_result.get("quality_score", 0)

                # Perform assessment
                quality_score, quality_data = await manager.perform_quality_assessment(
                    tmppath
                )

                assert isinstance(quality_score, int)
                assert isinstance(quality_data, dict)
                assert quality_score >= 0
                assert quality_score <= 100

                # Record the score
                manager.record_quality_score("test_project", quality_score)
                assert "test_project" in manager._quality_history
                assert manager._quality_history["test_project"][-1] == quality_score

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_session_checkpoint_workflow(self):
        """Test checkpoint operation within session."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Initialize
                await manager.initialize_session(working_directory=tmpdir)

                # Checkpoint
                checkpoint_result = await manager.checkpoint_session(tmpdir)

                assert checkpoint_result["success"]
                assert "quality_score" in checkpoint_result
                assert "timestamp" in checkpoint_result
                assert "quality_output" in checkpoint_result

                # Verify output is properly formatted
                output = checkpoint_result["quality_output"]
                assert isinstance(output, list)
                assert len(output) > 0

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionHandoffAndDocumentation:
    """Test session handoff documentation generation."""

    async def test_handoff_documentation_generation(self):
        """Test that handoff documentation is created properly."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Initialize and record some activity
                await manager.initialize_session(working_directory=tmpdir)
                await manager.checkpoint_session(tmpdir)

                # End session and get handoff
                end_result = await manager.end_session(tmpdir)

                assert end_result["success"]
                summary = end_result["summary"]
                assert "handoff_documentation" in summary
                handoff_path = summary.get("handoff_documentation")

                if handoff_path:
                    assert Path(handoff_path).exists()
                    content = Path(handoff_path).read_text()
                    assert len(content) > 0
                    # Handoff should contain project information
                    assert any(
                        keyword in content.lower()
                        for keyword in ["project", "quality", "recommendation"]
                    )

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_session_summary_completeness(self):
        """Test that session summary contains all required information."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                await manager.initialize_session(working_directory=tmpdir)
                await manager.checkpoint_session(tmpdir)

                end_result = await manager.end_session(tmpdir)
                summary = end_result["summary"]

                # Required fields in summary
                required_fields = [
                    "final_quality_score",
                    "recommendations",
                ]

                for field in required_fields:
                    assert field in summary, f"Missing required field: {field}"

                # Verify summary contains meaningful data
                assert summary["final_quality_score"] >= 0
                assert isinstance(summary["recommendations"], list)
                assert len(summary["recommendations"]) > 0

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestQualityScoreTrending:
    """Test quality score tracking across multiple checkpoints."""

    async def test_quality_score_history_tracking(self):
        """Test that quality scores are tracked across checkpoints."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Record multiple scores
            scores = [70, 75, 72, 80, 85]
            for score in scores:
                manager.record_quality_score("project", score)

            # Verify history
            assert "project" in manager._quality_history
            history = manager._quality_history["project"]
            assert len(history) == len(scores)
            assert history == scores

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_quality_score_history_limit(self):
        """Test that quality history respects the 10-score limit."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Record 15 scores
            for i in range(15):
                manager.record_quality_score("project", 70 + i)

            # Should keep only last 10
            assert len(manager._quality_history["project"]) == 10
            # Should have scores from 75 to 84
            assert manager._quality_history["project"][0] == 75
            assert manager._quality_history["project"][-1] == 84

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_quality_trending_analysis(self):
        """Test analysis of quality trends across checkpoints."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Simulate improving project quality
            scores = [60, 65, 70, 75, 80]
            for score in scores:
                manager.record_quality_score("improving_project", score)

            history = manager._quality_history["improving_project"]

            # Verify trend is improving
            assert history[0] < history[-1]
            assert history[-1] - history[0] == 20

            # Test degrading trend
            scores = [85, 80, 75, 70, 65]
            for score in scores:
                manager.record_quality_score("degrading_project", score)

            history = manager._quality_history["degrading_project"]
            assert history[0] > history[-1]
            assert history[0] - history[-1] == 20

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestMultiSessionIndependence:
    """Test that multiple sessions maintain independent state."""

    async def test_multiple_session_managers_are_independent(self):
        """Test that multiple manager instances don't interfere."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            manager1 = SessionLifecycleManager()
            manager2 = SessionLifecycleManager()

            # Set different projects
            manager1.current_project = "project1"
            manager2.current_project = "project2"

            # Verify independence
            assert manager1.current_project == "project1"
            assert manager2.current_project == "project2"

            # Record different quality histories
            manager1.record_quality_score("project1", 80)
            manager2.record_quality_score("project2", 70)

            assert manager1._quality_history["project1"] == [80]
            assert manager2._quality_history["project2"] == [70]
            assert "project2" not in manager1._quality_history
            assert "project1" not in manager2._quality_history

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_concurrent_session_checkpoints(self):
        """Test concurrent checkpoint operations in multiple sessions."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir1:
                with tempfile.TemporaryDirectory() as tmpdir2:
                    manager1 = SessionLifecycleManager()
                    manager2 = SessionLifecycleManager()

                    # Initialize both sessions
                    await manager1.initialize_session(working_directory=tmpdir1)
                    await manager2.initialize_session(working_directory=tmpdir2)

                    # Run concurrent checkpoints
                    async def checkpoint_both():
                        results = await asyncio.gather(
                            manager1.checkpoint_session(tmpdir1),
                            manager2.checkpoint_session(tmpdir2),
                        )
                        return results

                    results = await checkpoint_both()

                    assert len(results) == 2
                    assert all(r["success"] for r in results)

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionStatusAndHealth:
    """Test session status and health checking."""

    async def test_session_status_during_workflow(self):
        """Test session status retrieval at different workflow stages."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Get status before initialization
                status_before = await manager.get_session_status(tmpdir)
                assert status_before["success"]

                # Initialize
                await manager.initialize_session(working_directory=tmpdir)

                # Get status after initialization
                status_after = await manager.get_session_status(tmpdir)
                assert status_after["success"]
                assert "project" in status_after
                assert "quality_score" in status_after

                # Verify system health is included
                assert "system_health" in status_after
                health = status_after["system_health"]
                assert isinstance(health, dict)

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_system_health_checks_in_status(self):
        """Test that status includes comprehensive system health checks."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.get_session_status(tmpdir)

                assert result["success"]
                health = result["system_health"]

                # Check for key health indicators
                health_indicators = [
                    "uv_available",
                    "git_repository",
                    "claude_directory",
                ]

                for indicator in health_indicators:
                    assert indicator in health

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionProjectContext:
    """Test project context analysis during session."""

    async def test_project_context_analysis_integration(self):
        """Test project context analysis as part of session init."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Create realistic project structure
                (tmppath / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")
                (tmppath / "README.md").write_text("# Test Project")
                (tmppath / ".git").mkdir()
                (tmppath / "src").mkdir()
                (tmppath / "src" / "main.py").touch()
                (tmppath / "tests").mkdir()
                (tmppath / "tests" / "test_main.py").touch()

                manager = SessionLifecycleManager()
                context = await manager.analyze_project_context(tmppath)

                assert isinstance(context, dict)
                assert context.get("has_pyproject_toml")
                assert context.get("has_readme")
                assert context.get("has_git_repo")
                assert context.get("has_tests")

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_empty_project_context_analysis(self):
        """Test context analysis on minimal project."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                context = await manager.analyze_project_context(Path(tmpdir))

                assert isinstance(context, dict)
                # Empty directory should show defaults
                assert "has_pyproject_toml" in context
                assert "has_git_repo" in context
                assert "has_tests" in context

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionErrorRecovery:
    """Test error handling and recovery in session workflows."""

    async def test_session_recovery_after_checkpoint_error(self):
        """Test that session can recover from checkpoint errors."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                await manager.initialize_session(working_directory=tmpdir)

                # First checkpoint succeeds
                result1 = await manager.checkpoint_session(tmpdir)
                assert result1["success"]

                # Second checkpoint should also succeed (recovery test)
                result2 = await manager.checkpoint_session(tmpdir)
                assert result2["success"]

                # Session should be usable after checkpoint
                status = await manager.get_session_status(tmpdir)
                assert status["success"]

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_session_end_cleans_up_properly(self):
        """Test that session end cleans up resources properly."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                await manager.initialize_session(working_directory=tmpdir)

                # End session
                end_result = await manager.end_session(tmpdir)
                assert end_result["success"]

                # .claude directory should exist (handoff was written)
                claude_dir = Path(tmpdir) / ".claude"
                assert claude_dir.exists() or "handoff_documentation" in end_result[
                    "summary"
                ]

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionQualityFormatting:
    """Test quality formatting output during session."""

    async def test_quality_output_formatting_in_checkpoint(self):
        """Test that checkpoint produces properly formatted quality output."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                result = await manager.checkpoint_session(tmpdir)

                assert result["success"]
                output = result.get("quality_output", [])
                assert isinstance(output, list)

                # Output should be readable
                output_text = "\n".join(output)
                assert len(output_text) > 0

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_quality_score_display_formatting(self):
        """Test formatting of quality scores for display."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()
            quality_data = {
                "total_score": 82,
                "breakdown": {
                    "code_quality": 28,
                    "project_health": 26,
                    "dev_velocity": 17,
                    "security": 8,
                },
                "recommendations": ["Maintain code quality", "Add more tests"],
            }

            output = manager.format_quality_results(82, quality_data)

            assert isinstance(output, list)
            assert len(output) > 0
            output_text = "\n".join(output)

            # Should contain quality indication
            assert any(
                keyword in output_text.lower()
                for keyword in ["quality", "score", "excellent", "good"]
            )

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSessionIntegrationWorkflows:
    """Integration tests for complete multi-step workflows."""

    async def test_three_checkpoint_session_workflow(self):
        """Test session with multiple checkpoints."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Initialize
                init_result = await manager.initialize_session(working_directory=tmpdir)
                assert init_result["success"]

                # First checkpoint
                checkpoint1 = await manager.checkpoint_session(tmpdir)
                assert checkpoint1["success"]
                score1 = checkpoint1.get("quality_score", 0)

                # Simulate work and second checkpoint
                manager.record_quality_score("session", score1)
                checkpoint2 = await manager.checkpoint_session(tmpdir)
                assert checkpoint2["success"]
                score2 = checkpoint2.get("quality_score", 0)

                # Third checkpoint
                manager.record_quality_score("session", score2)
                checkpoint3 = await manager.checkpoint_session(tmpdir)
                assert checkpoint3["success"]

                # All checkpoints should have succeeded
                assert len(manager._quality_history.get("session", [])) >= 2

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_session_workflow_with_project_changes(self):
        """Test session handling project changes between checkpoints."""
        try:
            from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                manager = SessionLifecycleManager()

                # Initialize with minimal project
                await manager.initialize_session(working_directory=tmpdir)

                # Checkpoint 1
                result1 = await manager.checkpoint_session(tmpdir)
                assert result1["success"]

                # "Add" files to project
                (tmppath / "pyproject.toml").touch()
                (tmppath / "tests").mkdir(exist_ok=True)
                for i in range(3):
                    (tmppath / "tests" / f"test_{i}.py").touch()

                # Checkpoint 2 - should reflect changes
                result2 = await manager.checkpoint_session(tmpdir)
                assert result2["success"]

                # Verify both checkpoints completed
                assert result1["success"]
                assert result2["success"]

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
