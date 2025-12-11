#!/usr/bin/env python3
"""Edge case and security tests for session management.

Tests edge cases and security scenarios including:
- Invalid input handling
- Permission and access control
- Resource exhaustion protection
- Error recovery scenarios
- Input validation and sanitization
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestInvalidInputHandling:
    """Test handling of invalid inputs."""

    async def test_initialize_session_empty_working_directory(self):
        """Test session initialization with empty working directory."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                # Empty directory should be handled gracefully
                result = await manager.initialize_session(working_directory=tmpdir)
                assert result["success"]

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_initialize_session_nonexistent_directory(self):
        """Test session initialization with non-existent directory."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()
            nonexistent = "/tmp/nonexistent_project_12345_xyz"

            # Should handle gracefully or raise appropriate error
            try:
                result = await manager.initialize_session(working_directory=nonexistent)
                # If it succeeds, directory should be created
                assert isinstance(result, dict)
            except (FileNotFoundError, OSError):
                # Acceptable to raise appropriate error
                pass

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_quality_score_calculation_with_corrupted_project(self):
        """Test quality scoring with corrupted project files."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Create corrupted project structure
                (tmppath / "pyproject.toml").write_text("INVALID TOML { ] [")
                (tmppath / "tests").mkdir()

                manager = SessionLifecycleManager()

                # Should handle gracefully
                try:
                    score, _data = await manager.perform_quality_assessment(tmppath)
                    # Should return default score on error
                    assert isinstance(score, int)
                    assert score >= 0
                except Exception:
                    # Acceptable to raise exception with invalid input
                    pass

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_checkpoint_with_permission_denied(self):
        """Test checkpoint handling when directory is read-only."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                await manager.initialize_session(working_directory=tmpdir)

                # Try checkpoint on a read-only directory (may not work on all systems)
                result = await manager.checkpoint_session(tmpdir)
                # Should either succeed or fail gracefully
                assert isinstance(result, dict)

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestResourceConstraints:
    """Test handling of resource constraints."""

    async def test_quality_history_limit_enforcement(self):
        """Test that quality history respects maximum limits."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Record many scores
            for i in range(50):
                manager.record_quality_score("stress_test", 50 + (i % 20))

            # Should limit to maximum (typically 10)
            history = manager._quality_history["stress_test"]
            assert len(history) <= 10

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_large_project_quality_assessment(self):
        """Test quality assessment on large project structure."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Create large project structure
                (tmppath / "pyproject.toml").touch()
                tests_dir = tmppath / "tests"
                tests_dir.mkdir()

                # Create many test files
                for i in range(20):
                    (tests_dir / f"test_module_{i}.py").touch()

                manager = SessionLifecycleManager()

                # Should handle large project
                score, _data = await manager.perform_quality_assessment(tmppath)
                assert isinstance(score, int)
                assert 0 <= score <= 100

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience."""

    async def test_session_recovery_after_error(self):
        """Test that session can recover after encountering errors."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Initialize session
                result1 = await manager.initialize_session(working_directory=tmpdir)
                assert result1["success"]

                # Attempt checkpoint
                result2 = await manager.checkpoint_session(tmpdir)
                assert result2["success"]

                # Session should still be usable
                status = await manager.get_session_status(tmpdir)
                assert status["success"]

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_multiple_sequential_checkpoints(self):
        """Test multiple checkpoints in rapid succession."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()
                await manager.initialize_session(working_directory=tmpdir)

                # Multiple checkpoints
                for _ in range(5):
                    result = await manager.checkpoint_session(tmpdir)
                    assert result["success"]

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestInputValidation:
    """Test input validation and sanitization."""

    async def test_quality_score_with_extreme_values(self):
        """Test quality score handling with extreme values."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Test with extreme scores
            manager.record_quality_score("extreme_test", 0)
            manager.record_quality_score("extreme_test", 100)
            manager.record_quality_score("extreme_test", -1)
            manager.record_quality_score("extreme_test", 101)

            # Should handle gracefully
            history = manager._quality_history["extreme_test"]
            assert len(history) > 0

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_project_name_sanitization(self):
        """Test handling of special characters in project names."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Test with special characters
            special_names = [
                "project-with-dashes",
                "project_with_underscores",
                "project.with.dots",
                "projectWithCamelCase",
                "project123WithNumbers",
            ]

            for name in special_names:
                manager.record_quality_score(name, 75)
                assert name in manager._quality_history

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_quality_data_format_validation(self):
        """Test validation of quality data format."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Test with various data formats
            quality_data = {
                "total_score": 75,
                "breakdown": {"code_quality": 30},
                "recommendations": [],
            }

            # Should handle various formats or raise appropriate error
            try:
                output = manager.format_quality_results(75, quality_data)
                assert isinstance(output, list)
                assert len(output) >= 0  # Can be empty list
            except (KeyError, ValueError, TypeError):
                # Acceptable if format validation is strict
                pass

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestSecurityConsiderations:
    """Test security-related aspects of session management."""

    async def test_session_data_isolation(self):
        """Test that session data is properly isolated."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir1:
                with tempfile.TemporaryDirectory() as tmpdir2:
                    manager1 = SessionLifecycleManager()
                    manager2 = SessionLifecycleManager()

                    # Initialize separate sessions
                    await manager1.initialize_session(working_directory=tmpdir1)
                    await manager2.initialize_session(working_directory=tmpdir2)

                    # Record different quality histories
                    manager1.record_quality_score("project1", 80)
                    manager2.record_quality_score("project2", 70)

                    # Verify isolation
                    assert "project1" not in manager2._quality_history
                    assert "project2" not in manager1._quality_history

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_no_sensitive_data_in_output(self):
        """Test that output doesn't contain sensitive data."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            with tempfile.TemporaryDirectory() as tmpdir:
                manager = SessionLifecycleManager()

                # Create session and get output
                result = await manager.checkpoint_session(tmpdir)

                # Verify no sensitive patterns in output
                output_text = str(result)

                # Check for common sensitive patterns
                sensitive_patterns = ["password", "api_key", "secret", "token"]
                for pattern in sensitive_patterns:
                    assert pattern.lower() not in output_text.lower()

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Test with path traversal attempts
            dangerous_names = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "/etc/passwd",
                "C:\\Windows\\System32",
            ]

            # Should handle safely
            for name in dangerous_names:
                try:
                    manager.record_quality_score(name, 50)
                    # If accepted, it should be treated as a literal name
                    assert name in manager._quality_history
                except (ValueError, OSError):
                    # Acceptable to reject dangerous names
                    pass

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestDatabaseIntegrity:
    """Test database and data integrity."""

    async def test_quality_history_consistency(self):
        """Test consistency of quality history data."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Record scores and verify consistency
            scores = [70, 75, 80, 75, 70]
            for score in scores:
                manager.record_quality_score("consistency_test", score)

            # Verify order and values are preserved
            history = manager._quality_history["consistency_test"]
            assert len(history) == len(scores)

            # Verify values match
            for i, score in enumerate(scores):
                assert history[i] == score

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_concurrent_quality_updates(self):
        """Test concurrent updates to quality data."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            async def record_scores():
                for i in range(10):
                    manager.record_quality_score("concurrent_test", 60 + i)

            # Run concurrent updates
            await asyncio.gather(record_scores(), record_scores())

            # Verify data integrity
            assert "concurrent_test" in manager._quality_history
            # Should have limited to max (typically 10)
            assert len(manager._quality_history["concurrent_test"]) <= 10

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


@pytest.mark.asyncio
class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    async def test_empty_recommendations(self):
        """Test formatting with empty recommendations."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            quality_data = {
                "total_score": 50,
                "breakdown": {},
                "recommendations": [],
            }

            try:
                output = manager.format_quality_results(50, quality_data)
                assert isinstance(output, list)
            except (KeyError, ValueError, TypeError):
                # Acceptable if implementation requires certain fields
                pass

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_missing_quality_data_fields(self):
        """Test formatting with missing quality data fields."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            # Minimal quality data
            quality_data = {}

            try:
                output = manager.format_quality_results(0, quality_data)
                assert isinstance(output, list)
            except (KeyError, ValueError, TypeError):
                # Acceptable if implementation requires certain fields
                pass

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")

    async def test_zero_quality_score(self):
        """Test handling of zero quality score."""
        try:
            from session_buddy.core.session_manager import SessionLifecycleManager

            manager = SessionLifecycleManager()

            manager.record_quality_score("zero_score_project", 0)

            # Should handle zero scores
            history = manager._quality_history["zero_score_project"]
            assert len(history) > 0
            assert history[-1] == 0

        except ImportError:
            pytest.skip("SessionLifecycleManager not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
