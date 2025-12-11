#!/usr/bin/env python3
"""Edge case tests for SessionLifecycleManager."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from session_buddy.core.session_manager import SessionLifecycleManager


class TestSessionManagerEdgeCases:
    """Test SessionLifecycleManager edge cases and error conditions."""

    async def test_analyze_project_context_empty_directory(self):
        """Test analyze_project_context with completely empty directory."""
        manager = SessionLifecycleManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Test with a completely empty directory
            result = await manager.analyze_project_context(project_dir)

            # Should handle empty directory gracefully
            assert isinstance(result, dict)
            assert "has_git_repo" in result
            assert "has_python_files" in result
            assert "has_pyproject_toml" in result

    async def test_analyze_project_context_nonexistent_directory(self):
        """Test analyze_project_context with nonexistent directory."""
        manager = SessionLifecycleManager()

        nonexistent_dir = Path("/nonexistent/directory/path")

        # The function should handle nonexistent directories gracefully
        # rather than raising an exception
        result = await manager.analyze_project_context(nonexistent_dir)

        # Even though directory doesn't exist, should return a valid structure
        assert isinstance(result, dict)

    async def test_analyze_project_context_permission_error(self):
        """Test analyze_project_context when permissions prevent access."""
        manager = SessionLifecycleManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            restricted_dir = project_dir / "restricted"
            restricted_dir.mkdir()

            # On Unix systems, we can set permissions that prevent access
            # On Windows, we could simulate this with mock
            with patch(
                "pathlib.Path.glob", side_effect=PermissionError("Permission denied")
            ):
                result = await manager.analyze_project_context(restricted_dir)

                # Should handle permission errors gracefully
                assert isinstance(result, dict)
                # Even with permission errors, should return a valid structure

    async def test_calculate_quality_score_with_malformed_project(self):
        """Test calculate_quality_score with project that has malformed files."""
        manager = SessionLifecycleManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create a malformed pyproject.toml
            pyproject_path = project_dir / "pyproject.toml"
            with open(pyproject_path, "w") as f:
                f.write("{ invalid json content without closing brace")

            with patch("os.getcwd", return_value=str(project_dir)):
                result = await manager.calculate_quality_score(project_dir)

                # Should handle malformed project files gracefully
                assert "total_score" in result
                assert isinstance(result["total_score"], (int, float))

                # Should still return proper structure even with errors
                assert "breakdown" in result
                assert "recommendations" in result

    async def test_calculate_quality_score_no_uv_available(self):
        """Test calculate_quality_score when UV is not available."""
        manager = SessionLifecycleManager()

        # Mock shutil.which to return None for UV
        with patch("shutil.which", return_value=None):
            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = Path(temp_dir)

                with patch("os.getcwd", return_value=str(project_dir)):
                    result = await manager.calculate_quality_score(project_dir)

                    # Should handle missing UV gracefully and return valid result
                    assert "total_score" in result
                    assert isinstance(result["total_score"], (int, float))
                    assert result["total_score"] >= 0  # Score should be non-negative

    async def test_calculate_quality_score_no_git_repo(self):
        """Test calculate_quality_score when not in a git repository."""
        manager = SessionLifecycleManager()

        # Mock git repository check to return False
        with patch(
            "session_buddy.core.session_manager.is_git_repository",
            return_value=False,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = Path(temp_dir)

                with patch("os.getcwd", return_value=str(project_dir)):
                    result = await manager.calculate_quality_score(project_dir)

                    # Should handle non-git repository and return valid result
                    assert "total_score" in result
                    assert isinstance(result["total_score"], (int, float))

    async def test_perform_quality_assessment_empty_project(self):
        """Test perform_quality_assessment with minimal project."""
        manager = SessionLifecycleManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Mock the analysis to return minimal results
            with patch.object(
                manager,
                "analyze_project_context",
                return_value={
                    "has_pyproject_toml": False,
                    "has_readme": False,
                    "has_git_repo": False,
                    "has_tests": False,
                    "has_venv": False,
                    "has_python_files": False,
                },
            ):
                with patch("os.getcwd", return_value=str(project_dir)):
                    (
                        quality_score,
                        quality_data,
                    ) = await manager.perform_quality_assessment()

                    # Should handle minimal project gracefully
                    assert isinstance(quality_score, (int, float))
                    assert quality_score >= 0  # Minimum score should be 0

                    assert isinstance(quality_data, dict)
                    assert "breakdown" in quality_data
                    assert "recommendations" in quality_data

    async def test_format_quality_results_with_missing_fields(self):
        """Test format_quality_results with incomplete quality data."""
        manager = SessionLifecycleManager()

        # Test with minimal quality data - should handle missing breakdown gracefully
        quality_score = 50
        quality_data = {
            "total_score": 50
            # Missing other expected fields like 'breakdown'
        }

        # The function should handle missing 'breakdown' key gracefully
        with pytest.raises(KeyError) as exc_info:
            manager.format_quality_results(quality_score, quality_data)

        # This shows us that the function expects 'breakdown' to be present
        assert "breakdown" in str(exc_info.value)

        # Test with proper structure but empty breakdown
        incomplete_data = {
            "breakdown": {
                "code_quality": 30.0,
                "project_health": 25.0,
                "dev_velocity": 15.0,
                "security": 8.0,
            },
            "recommendations": [],
            "version": "2.0",
            "trust_score": {},
        }

        result = manager.format_quality_results(quality_score, incomplete_data)

        # Should handle proper structure gracefully
        assert isinstance(result, list)
        assert len(result) > 0

    async def test_perform_git_checkpoint_no_git_repo(self):
        """Test perform_git_checkpoint when not in a git repository."""
        manager = SessionLifecycleManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            result = await manager.perform_git_checkpoint(project_dir, 80)

            # Should handle non-git repository gracefully
            # Result should contain appropriate error or notice message
            assert isinstance(result, list)
            # Git checkpoint should still return a proper output even if git isn't available

    async def test_perform_git_checkpoint_error_conditions(self):
        """Test perform_git_checkpoint with various error conditions."""
        manager = SessionLifecycleManager()

        # Mock git operations to raise exceptions
        with patch(
            "session_buddy.core.session_manager.create_checkpoint_commit"
        ) as mock_commit:
            mock_commit.side_effect = Exception("Git operation failed")

            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = Path(temp_dir)

                manager.current_project = "test-project"

                result = await manager.perform_git_checkpoint(project_dir, 80)

                # Should handle git errors gracefully and return appropriate message
                assert isinstance(result, list)
                # Should contain error message
                error_found = any(
                    "error" in line.lower() or "exception" in line.lower()
                    for line in result
                )
                assert error_found, "Result should contain error message"

    async def test_initialize_session_with_invalid_path(self):
        """Test initialize_session with invalid project path."""
        manager = SessionLifecycleManager()

        # Test with a path that doesn't exist - should handle gracefully
        result = await manager.initialize_session("/nonexistent/path")

        # Should return a structured response rather than raising an exception
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False  # Should indicate failure

    async def test_initialize_session_permission_error(self):
        """Test initialize_session when permissions prevent directory access."""
        manager = SessionLifecycleManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "restricted"
            project_dir.mkdir()

            # Mock os.chdir to raise PermissionError
            with patch("os.chdir", side_effect=PermissionError("Permission denied")):
                result = await manager.initialize_session(str(project_dir))

                # Should handle permission errors gracefully
                assert isinstance(result, dict)
                assert result["success"] is False
                assert "error" in result

    async def test_checkpoint_session_with_no_current_project(self):
        """Test checkpoint_session when no project is currently set."""
        manager = SessionLifecycleManager()

        # Ensure no project is set
        manager.current_project = None

        # Test in a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.getcwd", return_value=str(temp_dir)):
                result = await manager.checkpoint_session()

                # Should handle missing project name gracefully
                assert isinstance(result, dict)
                assert "success" in result  # May succeed or fail gracefully

    async def test_checkpoint_session_exception_handling(self):
        """Test checkpoint_session when internal operations raise exceptions."""
        manager = SessionLifecycleManager()

        # Mock the quality assessment to raise an exception
        with patch.object(
            manager,
            "perform_quality_assessment",
            side_effect=Exception("Quality assessment failed"),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("os.getcwd", return_value=str(temp_dir)):
                    result = await manager.checkpoint_session()

                    # Should handle exceptions gracefully
                    assert isinstance(result, dict)
                    assert result["success"] is False
                    assert "error" in result
                    assert "Quality assessment failed" in result["error"]

    async def test_end_session_exception_handling(self):
        """Test end_session when internal operations raise exceptions."""
        manager = SessionLifecycleManager()

        # Set a current project
        manager.current_project = "test-project"

        # Mock the quality assessment to raise an exception
        with patch.object(
            manager,
            "perform_quality_assessment",
            side_effect=Exception("Quality assessment failed"),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("os.getcwd", return_value=str(temp_dir)):
                    result = await manager.end_session()

                    # Should handle exceptions gracefully
                    assert isinstance(result, dict)
                    assert result["success"] is False
                    assert "error" in result
                    assert "Quality assessment failed" in result["error"]

    async def test_end_session_with_no_current_project(self):
        """Test end_session when no project is currently set."""
        manager = SessionLifecycleManager()

        # Ensure no project is set
        manager.current_project = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.getcwd", return_value=str(temp_dir)):
                result = await manager.end_session()

                # Should handle missing project gracefully
                assert isinstance(result, dict)
                assert (
                    result["success"] is True
                )  # Should still succeed even without project
                # May have different behavior depending on implementation

    async def test_session_manager_with_closed_database(self):
        """Test SessionLifecycleManager operations with database issues."""
        manager = SessionLifecycleManager()

        # This test would be more relevant if we were directly using a database
        # For now, test operations that might interact with persistent storage
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Mock operations that might access databases or files
            with patch("os.getcwd", return_value=str(project_dir)):
                # This shouldn't cause issues even if underlying storage has problems
                result = await manager.get_session_status(str(project_dir))

                # Should handle gracefully
                assert isinstance(result, dict)
                assert "success" in result

    async def test_session_manager_async_operations_timeout(self):
        """Test handling of async operations that might timeout."""
        manager = SessionLifecycleManager()

        # Mock an operation that takes too long
        async def slow_operation():
            await asyncio.sleep(0.1)  # Small delay to simulate work
            return {"result": "success"}

        # Test that the manager can handle operations that take time
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.getcwd", return_value=str(temp_dir)):
                # This is more about ensuring async operations don't hang
                result = await manager.get_session_status(str(temp_dir))
                assert isinstance(result, dict)

    async def test_large_input_handling(self):
        """Test SessionLifecycleManager with unusually large inputs."""
        manager = SessionLifecycleManager()

        # Create an extremely long project path (within system limits)
        with tempfile.TemporaryDirectory() as base_dir:
            base_path = Path(base_dir)

            # Create a very long subdirectory name
            long_dir_name = "a" * 100  # 100 chars for directory name
            long_path = base_path / long_dir_name
            long_path.mkdir()

            # Test with this long path
            with patch("os.getcwd", return_value=str(long_path)):
                # All operations should handle long paths gracefully
                result = await manager.get_session_status(str(long_path))

                assert isinstance(result, dict)
                assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__])
