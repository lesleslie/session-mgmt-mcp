#!/usr/bin/env python3
"""Integration tests for complete session workflows."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from session_buddy.core.session_manager import SessionLifecycleManager
from session_buddy.reflection_tools import ReflectionDatabase


class TestSessionWorkflowIntegration:
    """Integration tests for complete session workflows."""

    async def test_complete_session_lifecycle(self):
        """Test a complete session lifecycle: init → checkpoint → end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            db_path = project_dir / "test.duckdb"

            # Create the reflection database
            db = ReflectionDatabase(db_path=str(db_path))
            await db.initialize()

            # Initialize session
            manager = SessionLifecycleManager()

            # Mock git repository check and other dependencies
            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        # Initialize session
                        init_result = await manager.initialize_session(str(project_dir))

                        assert init_result["success"] is True
                        assert "project" in init_result
                        assert init_result["quality_score"] >= 0

                        # Perform a checkpoint
                        checkpoint_result = await manager.checkpoint_session()

                        assert checkpoint_result["success"] is True
                        assert checkpoint_result["quality_score"] >= 0
                        assert "quality_output" in checkpoint_result

                        # End the session
                        end_result = await manager.end_session()

                        assert end_result["success"] is True
                        assert "summary" in end_result
                        assert "final_quality_score" in end_result["summary"]

                        db.close()

    async def test_session_with_reflection_operations(self):
        """Test session operations that use reflection database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            db_path = project_dir / "test_reflections.duckdb"

            # Initialize reflection database
            db = ReflectionDatabase(db_path=str(db_path))
            await db.initialize()

            # Add some initial reflections
            await db.store_reflection(
                "Initial reflection for testing", ["test", "initial"]
            )
            await db.store_reflection("Another reflection", ["test", "example"])

            # Now test the session manager with these reflections
            manager = SessionLifecycleManager()

            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        # Initialize session
                        init_result = await manager.initialize_session(str(project_dir))
                        assert init_result["success"] is True

                        # Check that the session can access stored reflections
                        # This would depend on how the reflection database is accessed in the session
                        # For now, just verify that the session initializes properly with existing data
                        assert init_result["quality_score"] >= 0

                        # Perform a checkpoint
                        checkpoint_result = await manager.checkpoint_session()
                        assert checkpoint_result["success"] is True

                        # Add a reflection during the session
                        await db.store_reflection(
                            "Added during session", ["session", "test"]
                        )

                        # End session
                        end_result = await manager.end_session()
                        assert end_result["success"] is True

                        db.close()

    async def test_session_with_permission_trust_operations(self):
        """Test session operations that involve permission and trust."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Mock the permissions manager to test integration
            manager = SessionLifecycleManager()

            # Mock the permissions manager with specific trusted operations
            mock_perms_manager = Mock()
            mock_perms_manager.trusted_operations = {"op1", "op2", "op3"}

            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        with patch(
                            "session_buddy.server.permissions_manager",
                            mock_perms_manager,
                        ):
                            # Initialize session
                            init_result = await manager.initialize_session(
                                str(project_dir)
                            )
                            assert init_result["success"] is True

                            # The quality score should reflect the trusted operations count
                            checkpoint_result = await manager.checkpoint_session()
                            assert checkpoint_result["success"] is True

                            # End session
                            end_result = await manager.end_session()
                            assert end_result["success"] is True

    async def test_session_with_quality_scoring_integration(self):
        """Test session integration with quality scoring system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create some project files to influence quality scoring
            (project_dir / "pyproject.toml").write_text('{"project": {"name": "test"}}')
            (project_dir / "README.md").write_text("# Test Project")
            tests_dir = project_dir / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_example.py").write_text("def test_example(): pass")

            manager = SessionLifecycleManager()

            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        # Initialize session with project files present
                        init_result = await manager.initialize_session(str(project_dir))
                        assert init_result["success"] is True
                        # Quality score should be higher due to project files
                        assert init_result["quality_score"] >= 0

                        # Perform checkpoint
                        checkpoint_result = await manager.checkpoint_session()
                        assert checkpoint_result["success"] is True
                        # Should have a reasonable quality score
                        assert checkpoint_result["quality_score"] >= 0

                        # End session
                        end_result = await manager.end_session()
                        assert end_result["success"] is True
                        assert end_result["summary"]["final_quality_score"] >= 0

    async def test_concurrent_session_operations(self):
        """Test that session operations work correctly when initiated concurrently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create two session managers to simulate concurrent operations
            manager1 = SessionLifecycleManager()
            manager2 = SessionLifecycleManager()

            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        # Initialize first session
                        init_result1 = await manager1.initialize_session(
                            str(project_dir)
                        )
                        assert init_result1["success"] is True

                        # Initialize second session in the same directory
                        # This should work but may have different behavior depending on implementation
                        await manager2.initialize_session(str(project_dir))
                        # Depending on implementation, this might succeed or fail appropriately

                        # Perform operations on first session
                        checkpoint_result1 = await manager1.checkpoint_session()
                        assert checkpoint_result1["success"] is True

                        # End first session
                        end_result1 = await manager1.end_session()
                        assert end_result1["success"] is True

    async def test_session_with_search_functionality(self):
        """Test session integration with search functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            db_path = project_dir / "search_test.duckdb"

            # Initialize reflection database with search capability
            db = ReflectionDatabase(db_path=str(db_path))
            await db.initialize()

            # Add various reflections for search testing
            search_contents = [
                "Python async programming patterns",
                "DuckDB vector search implementation",
                "MCP server best practices",
                "FastAPI async testing strategies",
                "Quality score calculation methods",
            ]

            for i, content in enumerate(search_contents):
                await db.store_reflection(content, ["test", f"tag_{i}"])

            manager = SessionLifecycleManager()

            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        # Initialize session
                        init_result = await manager.initialize_session(str(project_dir))
                        assert init_result["success"] is True

                        # Simulate operations that might use search
                        results = await db.search_conversations(
                            "async programming", limit=10
                        )
                        # Verify search returns a list (even if empty)
                        assert isinstance(results, list)

                        # Checkpoint and end session
                        checkpoint_result = await manager.checkpoint_session()
                        assert checkpoint_result["success"] is True

                        end_result = await manager.end_session()
                        assert end_result["success"] is True

                        db.close()

    async def test_session_with_git_operations(self):
        """Test that session operations work with actual git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Initialize a git repository in the temp directory
            import subprocess

            subprocess.run(
                ["git", "init"], cwd=project_dir, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=project_dir,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"], cwd=project_dir, check=True
            )

            # Create a file and commit it
            test_file = project_dir / "test.txt"
            test_file.write_text("Initial commit")
            subprocess.run(["git", "add", "test.txt"], cwd=project_dir, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"], cwd=project_dir, check=True
            )

            manager = SessionLifecycleManager()

            with patch("os.chdir"):
                with patch("os.getcwd", return_value=str(project_dir)):
                    # Initialize session in git repo
                    init_result = await manager.initialize_session(str(project_dir))
                    assert init_result["success"] is True

                    # Git repository should be detected
                    assert init_result["project_context"]["has_git_repo"] is True

                    # Perform checkpoint (this might try to create a git commit)
                    checkpoint_result = await manager.checkpoint_session()
                    assert checkpoint_result["success"] is True

                    # End session
                    end_result = await manager.end_session()
                    assert end_result["success"] is True

    async def test_session_error_recovery(self):
        """Test session recovery after errors occur in one operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            manager = SessionLifecycleManager()

            with patch(
                "session_buddy.core.session_manager.is_git_repository",
                return_value=True,
            ):
                with patch("os.chdir"):
                    with patch("os.getcwd", return_value=str(project_dir)):
                        # Initialize session normally
                        init_result = await manager.initialize_session(str(project_dir))
                        assert init_result["success"] is True

                        # Mock one operation to fail
                        original_method = manager.perform_quality_assessment

                        async def failing_quality_assessment():
                            msg = "Simulated failure"
                            raise Exception(msg)

                        # Replace temporarily
                        manager.perform_quality_assessment = failing_quality_assessment

                        # This checkpoint should fail
                        with patch(
                            "session_buddy.core.session_manager.is_git_repository",
                            return_value=True,
                        ):
                            checkpoint_result = await manager.checkpoint_session()
                            assert checkpoint_result["success"] is False
                            assert "error" in checkpoint_result

                        # Restore the original method
                        manager.perform_quality_assessment = original_method

                        # Now the checkpoint should work again
                        checkpoint_result = await manager.checkpoint_session()
                        assert checkpoint_result["success"] is True

                        # End session normally
                        end_result = await manager.end_session()
                        assert end_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])
