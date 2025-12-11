#!/usr/bin/env python3
"""Functional tests for end-to-end session workflows."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from session_buddy.core.session_manager import SessionLifecycleManager


class TestSessionWorkflows:
    """Test complete session workflows."""

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_complete_session_workflow(self, mock_is_git_repo):
        """Test a complete session workflow: init -> checkpoint -> end."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create a basic project structure
            (project_dir / "pyproject.toml").touch()
            (project_dir / "README.md").touch()
            (project_dir / "tests").mkdir()

            # Mock the analyze_project_context to return consistent results
            mock_project_context = {
                "has_pyproject_toml": True,
                "has_readme": True,
                "has_git_repo": True,
                "has_tests": True,
                "has_venv": False,
                "has_src_structure": False,
                "has_docs": False,
                "has_ci_cd": False,
                "has_python_files": True,
            }

            with patch.object(
                manager, "analyze_project_context", return_value=mock_project_context
            ):
                # 1. Initialize session
                init_result = await manager.initialize_session(str(project_dir))

                assert init_result["success"] is True
                assert init_result["project"] == project_dir.name
                assert "quality_score" in init_result
                assert init_result["quality_score"] > 0

                # Verify .claude directory structure was created
                claude_dir = Path.home() / ".claude"
                assert (claude_dir / "data").exists()
                assert (claude_dir / "logs").exists()

                # 2. Create a checkpoint
                checkpoint_result = await manager.checkpoint_session()

                assert checkpoint_result["success"] is True
                assert "quality_score" in checkpoint_result
                assert checkpoint_result["quality_score"] > 0
                assert "quality_output" in checkpoint_result
                assert "git_output" in checkpoint_result

                # 3. End session
                end_result = await manager.end_session()

                assert end_result["success"] is True
                assert "summary" in end_result
                assert "final_quality_score" in end_result["summary"]
                assert end_result["summary"]["final_quality_score"] > 0
                assert "handoff_documentation" in end_result["summary"]

                # Verify handoff documentation was created
                handoff_path = end_result["summary"]["handoff_documentation"]
                if handoff_path:
                    handoff_file = Path(handoff_path)
                    assert handoff_file.exists()
                    assert handoff_file.suffix == ".md"

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_with_git_operations(self, mock_is_git_repo):
        """Test session workflow with git operations."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create a basic project structure
            (project_dir / "pyproject.toml").touch()
            (project_dir / "README.md").touch()

            # Create some files to simulate changes
            src_dir = project_dir / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text(
                "# Main application file\nprint('Hello World')\n"
            )

            mock_project_context = {
                "has_pyproject_toml": True,
                "has_readme": True,
                "has_git_repo": True,
                "has_tests": False,
                "has_venv": False,
                "has_src_structure": True,
                "has_docs": False,
                "has_ci_cd": False,
                "has_python_files": True,
            }

            with patch.object(
                manager, "analyze_project_context", return_value=mock_project_context
            ):
                # Initialize session
                init_result = await manager.initialize_session(str(project_dir))
                assert init_result["success"] is True

                # Create a checkpoint (should handle git status)
                with patch(
                    "session_buddy.core.session_manager.create_checkpoint_commit"
                ) as mock_checkpoint:
                    mock_checkpoint.return_value = (
                        True,
                        "abc123",
                        ["✅ Checkpoint created successfully"],
                    )

                    checkpoint_result = await manager.checkpoint_session()
                    assert checkpoint_result["success"] is True

                    # Verify git checkpoint was called
                    mock_checkpoint.assert_called_once()

                # End session
                end_result = await manager.end_session()
                assert end_result["success"] is True

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_status_check(self, mock_is_git_repo):
        """Test session status checking."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create a basic project structure
            (project_dir / "pyproject.toml").touch()
            (project_dir / "README.md").touch()
            (project_dir / "tests").mkdir()

            mock_project_context = {
                "has_pyproject_toml": True,
                "has_readme": True,
                "has_git_repo": True,
                "has_tests": True,
                "has_venv": False,
                "has_src_structure": False,
                "has_docs": False,
                "has_ci_cd": False,
                "has_python_files": True,
            }

            with patch.object(
                manager, "analyze_project_context", return_value=mock_project_context
            ):
                # Check session status
                status_result = await manager.get_session_status(str(project_dir))

                assert status_result["success"] is True
                assert "project" in status_result
                assert "quality_score" in status_result
                assert status_result["quality_score"] > 0
                assert "quality_breakdown" in status_result
                assert "recommendations" in status_result
                assert "project_context" in status_result
                assert "system_health" in status_result

    def _setup_test_project(self, temp_dir: Path, context: dict[str, bool]):
        """Create a test project directory with specified features."""
        if context.get("has_pyproject_toml"):
            (temp_dir / "pyproject.toml").write_text('[project]\nname = "test"\n')
        if context.get("has_readme"):
            (temp_dir / "README.md").write_text("# Test Project\n")
        if context.get("has_git_repo"):
            git_dir = temp_dir / ".git"
            git_dir.mkdir()
            # Create minimal git structure
            (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
            refs_dir = git_dir / "refs" / "heads"
            refs_dir.mkdir(parents=True)
        if context.get("has_tests"):
            tests_dir = temp_dir / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_example.py").write_text("def test_example(): pass\n")
        if context.get("has_venv"):
            venv_dir = temp_dir / ".venv"
            venv_dir.mkdir()
            (venv_dir / "pyvenv.cfg").write_text("version = 3.13\n")
        if context.get("has_src_structure"):
            src_dir = temp_dir / "src"
            src_dir.mkdir()
            (src_dir / "__init__.py").touch()
        if context.get("has_docs"):
            docs_dir = temp_dir / "docs"
            docs_dir.mkdir()
            (docs_dir / "index.md").write_text("# Documentation\n")
        if context.get("has_ci_cd"):
            gh_dir = temp_dir / ".github" / "workflows"
            gh_dir.mkdir(parents=True)
            (gh_dir / "test.yml").write_text("name: test\n")
        if context.get("has_python_files"):
            (temp_dir / "main.py").write_text("print('hello')\n")
        # Always create a lockfile if pyproject.toml exists (for dependency management scoring)
        if context.get("has_pyproject_toml"):
            (temp_dir / "uv.lock").write_text("# UV lockfile\n")

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_quality_scoring(self, mock_is_git_repo):
        """Test session quality scoring logic with V2 algorithm."""
        manager = SessionLifecycleManager(logger=Mock())

        test_cases = [
            # High quality project - V2 scores ~50 (project_health:20 + session:20 + code:13 + security:9)
            {
                "context": {
                    "has_pyproject_toml": True,
                    "has_readme": True,
                    "has_git_repo": True,
                    "has_tests": True,
                    "has_venv": True,
                    "has_src_structure": True,
                    "has_docs": True,
                    "has_ci_cd": True,
                    "has_python_files": True,
                },
                "expected_min_score": 45,  # V2 algorithm: baseline + high project_health
                "expected_max_score": 60,
                "uv_available": True,
            },
            # Medium quality project - V2 scores ~45 (project_health:15 + session:20 + code:13 + security:9)
            {
                "context": {
                    "has_pyproject_toml": True,
                    "has_readme": True,
                    "has_git_repo": True,
                    "has_tests": False,
                    "has_venv": False,
                    "has_src_structure": False,
                    "has_docs": False,
                    "has_ci_cd": False,
                    "has_python_files": True,
                },
                "expected_min_score": 40,
                "expected_max_score": 50,
                "uv_available": True,
            },
            # Low quality project - V2 scores ~22 (session:20 + baseline code/security)
            {
                "context": {
                    "has_pyproject_toml": False,
                    "has_readme": False,
                    "has_git_repo": False,
                    "has_tests": False,
                    "has_venv": False,
                    "has_src_structure": False,
                    "has_docs": False,
                    "has_ci_cd": False,
                    "has_python_files": False,
                },
                "expected_max_score": 30,  # Baseline session_management + minimal code/security
                "uv_available": False,
            },
        ]

        for i, test_case in enumerate(test_cases):
            await self._run_quality_scoring_test_case(
                manager, test_case, i, mock_is_git_repo
            )

    async def _run_quality_scoring_test_case(
        self, manager, test_case, i, mock_is_git_repo
    ):
        """Helper method to run a single quality scoring test case."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create project structure based on context
            self._setup_test_project(temp_path, test_case["context"])

            # Mock git repo check based on context
            mock_is_git_repo.return_value = test_case["context"].get(
                "has_git_repo", False
            )

            with patch(
                "shutil.which",
                return_value="/usr/local/bin/uv" if test_case["uv_available"] else None,
            ):
                with patch("session_buddy.server.permissions_manager") as mock_perms:
                    mock_perms.trusted_operations = {"op1", "op2"} if i < 2 else set()

                    # Pass temp_dir to quality scoring
                    quality_result = await manager.calculate_quality_score(
                        project_dir=temp_path
                    )

                    assert isinstance(quality_result["total_score"], int)
                    assert 0 <= quality_result["total_score"] <= 100

                    if "expected_min_score" in test_case:
                        assert (
                            quality_result["total_score"]
                            >= test_case["expected_min_score"]
                        ), (
                            f"Test case {i}: Expected min score {test_case['expected_min_score']}, got {quality_result['total_score']}"
                        )

                    if "expected_max_score" in test_case:
                        assert (
                            quality_result["total_score"]
                            <= test_case["expected_max_score"]
                        ), (
                            f"Test case {i}: Expected max score {test_case['expected_max_score']}, got {quality_result['total_score']}"
                        )

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_with_previous_handoff(self, mock_is_git_repo):
        """Test session initialization with previous handoff file."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create project structure
            (project_dir / "pyproject.toml").touch()

            # Create a previous handoff file
            handoff_dir = project_dir / ".crackerjack" / "session" / "handoff"
            handoff_dir.mkdir(parents=True)

            handoff_content = """# Session Handoff Report - previous-session

**Session ended:** 2024-01-01T10:00:00Z
**Final quality score:** 75/100
**Working directory:** /previous/path

## Recommendations for Next Session

1. Add test suite to improve code quality
2. Initialize git repository for version control

## Key Achievements

- Session successfully completed
"""

            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            handoff_file = handoff_dir / f"session_handoff_{timestamp}.md"
            with open(handoff_file, "w") as f:
                f.write(handoff_content)

            mock_project_context = {
                "has_pyproject_toml": True,
                "has_readme": False,
                "has_git_repo": True,
                "has_tests": False,
                "has_venv": False,
                "has_src_structure": False,
                "has_docs": False,
                "has_ci_cd": False,
                "has_python_files": True,
            }

            with patch.object(
                manager, "analyze_project_context", return_value=mock_project_context
            ):
                # Initialize session - should find previous handoff
                init_result = await manager.initialize_session(str(project_dir))

                assert init_result["success"] is True
                assert "previous_session" in init_result
                assert init_result["previous_session"] is not None
                assert "quality_score" in init_result["previous_session"]


class TestSessionErrorHandling:
    """Test session error handling scenarios."""

    async def test_session_with_invalid_directory(self):
        """Test session initialization with invalid directory."""
        manager = SessionLifecycleManager(logger=Mock())

        # Try to initialize with non-existent directory
        result = await manager.initialize_session("/non/existent/directory")

        assert result["success"] is False
        assert "error" in result

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_with_permissions_error(self, mock_is_git_repo):
        """Test session handling of permissions errors."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Mock analyze_project_context to raise permission error
            with patch.object(
                manager,
                "analyze_project_context",
                side_effect=PermissionError("Permission denied"),
            ):
                result = await manager.initialize_session(str(project_dir))

                # Should handle gracefully
                assert result["success"] is False
                assert "error" in result

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_with_handoff_write_error(self, mock_is_git_repo):
        """Test session handling of handoff file write errors."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            (project_dir / "pyproject.toml").touch()

            mock_project_context = {
                "has_pyproject_toml": True,
                "has_readme": False,
                "has_git_repo": True,
                "has_tests": False,
                "has_venv": False,
                "has_src_structure": False,
                "has_docs": False,
                "has_ci_cd": False,
                "has_python_files": True,
            }

            # Mock handoff documentation saving to fail
            with patch.object(
                manager, "analyze_project_context", return_value=mock_project_context
            ):
                with patch.object(
                    manager, "_save_handoff_documentation", return_value=None
                ):
                    result = await manager.end_session(
                        working_directory=str(project_dir)
                    )

                    # Should still succeed but without handoff path
                    assert result["success"] is True
                    assert "summary" in result
                    assert result["summary"]["handoff_documentation"] is None


class TestSessionCrossPlatform:
    """Test session functionality across different environments."""

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_in_different_environments(self, mock_is_git_repo):
        """Test session behavior with different environment configurations (V2 algorithm)."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        test_environments = [
            # Environment with UV available - V2 doesn't score UV separately anymore
            {
                "which_result": "/usr/local/bin/uv",
                "description": "with UV",
            },
            # Environment without UV
            {
                "which_result": None,
                "description": "without UV",
            },
        ]

        for env in test_environments:
            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = Path(temp_dir)
                (project_dir / "pyproject.toml").touch()

                mock_project_context = {
                    "has_pyproject_toml": True,
                    "has_readme": False,
                    "has_git_repo": True,
                    "has_tests": False,
                    "has_venv": False,
                    "has_src_structure": False,
                    "has_docs": False,
                    "has_ci_cd": False,
                    "has_python_files": True,
                }

                with patch.object(
                    manager,
                    "analyze_project_context",
                    return_value=mock_project_context,
                ):
                    with patch("shutil.which", return_value=env["which_result"]):
                        with patch(
                            "session_buddy.server.permissions_manager"
                        ) as mock_perms:
                            mock_perms.trusted_operations = set()

                            quality_result = await manager.calculate_quality_score(
                                project_dir=project_dir
                            )

                            # V2 algorithm: verify quality scoring works across environments
                            # tools score comes from trust_score.tool_ecosystem (MCP tool count), not UV
                            assert "total_score" in quality_result
                            assert "breakdown" in quality_result
                            assert "version" in quality_result
                            assert quality_result["version"] == "2.0"
                            assert 0 <= quality_result["total_score"] <= 100

    @patch("session_buddy.core.session_manager.is_git_repository")
    async def test_session_with_different_project_types(self, mock_is_git_repo):
        """Test session with different types of projects."""
        mock_is_git_repo.return_value = True

        manager = SessionLifecycleManager(logger=Mock())

        project_types = [
            # Python project with full structure
            {
                "files": [
                    "pyproject.toml",
                    "README.md",
                    "requirements.txt",
                    "setup.py",
                    "tests/test_main.py",
                ],
                "dirs": ["src", "docs", ".github"],
                "expected_indicators": 7,  # Number of True indicators we expect
            },
            # Minimal Python project
            {
                "files": ["main.py"],
                "dirs": [],
                "expected_indicators": 2,  # has_python_files, has_git_repo
            },
            # Non-Python project
            {
                "files": ["README.md", "package.json"],
                "dirs": [],
                "expected_indicators": 2,  # has_readme, has_git_repo
            },
        ]

        for project_type in project_types:
            await self._run_project_type_test_case(
                manager, project_type, mock_is_git_repo
            )

    async def _run_project_type_test_case(
        self, manager, project_type, mock_is_git_repo
    ):
        """Helper method to run a single project type test case."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            # Create files
            for filename in project_type["files"]:
                if "/" in filename:
                    # Handle nested files
                    subdir, fname = filename.rsplit("/", 1)
                    (project_dir / subdir).mkdir(parents=True, exist_ok=True)
                    (project_dir / subdir / fname).touch()
                else:
                    (project_dir / filename).touch()

            # Create directories
            for dirname in project_type["dirs"]:
                (project_dir / dirname).mkdir()

            # Create a basic Python file for detection
            if "main.py" not in project_type["files"]:
                (project_dir / "main.py").write_text("# Python file\n")

            # Test project context analysis
            context_result = await manager.analyze_project_context(str(project_dir))

            # Count True indicators
            sum(1 for detected in context_result.values() if detected)

            # For Python projects, we expect to detect Python files
            if (
                any("py" in f for f in project_type["files"])
                or "main.py" not in project_type["files"]
            ):
                assert context_result.get("has_python_files", False) is True


if __name__ == "__main__":
    pytest.main([__file__])
