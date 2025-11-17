"""Test fixtures for Git operations testing.

Week 8 Day 2 - Phase 2: Mock git repositories and operations.
Provides temporary git repositories with realistic commit history.
"""

from __future__ import annotations

import subprocess
import typing as t
from unittest.mock import Mock

import pytest

if t.TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary Git repository for testing.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to temporary git repository root.

    Example:
        >>> def test_git_status(tmp_git_repo):
        ...     # tmp_git_repo is already initialized with git
        ...     result = subprocess.run(
        ...         ["git", "status"], cwd=tmp_git_repo, capture_output=True, text=True
        ...     )
        ...     assert "On branch main" in result.stdout

    """
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    # Configure git user for commits
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    (tmp_path / "README.md").write_text(
        "# Test Project\n\nTest repository for fixtures.\n"
    )
    subprocess.run(
        ["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path


@pytest.fixture
def tmp_git_repo_with_commits(tmp_git_repo: Path) -> Path:
    """Create a temporary Git repository with multiple commits.

    Args:
        tmp_git_repo: Base git repository fixture.

    Returns:
        Path to git repository with commit history.

    """
    # Add additional commits
    for i in range(3):
        file_path = tmp_git_repo / f"file_{i}.txt"
        file_path.write_text(f"Content {i}\n")
        subprocess.run(
            ["git", "add", file_path.name],
            cwd=tmp_git_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add file_{i}.txt"],
            cwd=tmp_git_repo,
            check=True,
            capture_output=True,
        )

    return tmp_git_repo


@pytest.fixture
def tmp_git_repo_with_changes(tmp_git_repo: Path) -> Path:
    """Create a git repository with uncommitted changes.

    Args:
        tmp_git_repo: Base git repository fixture.

    Returns:
        Path to git repository with uncommitted changes.

    """
    # Create modified file (staged)
    modified_file = tmp_git_repo / "modified.txt"
    modified_file.write_text("Modified content\n")
    subprocess.run(
        ["git", "add", "modified.txt"],
        cwd=tmp_git_repo,
        check=True,
        capture_output=True,
    )

    # Create untracked file
    untracked_file = tmp_git_repo / "untracked.txt"
    untracked_file.write_text("Untracked content\n")

    return tmp_git_repo


@pytest.fixture
def mock_git_operations() -> Mock:
    """Create a mock for Git operations functions.

    Returns:
        Mock object with common git operation methods.

    Example:
        >>> def test_checkpoint_commit(mock_git_operations):
        ...     mock_git_operations.create_checkpoint_commit.return_value = "abc123"
        ...     sha = mock_git_operations.create_checkpoint_commit(
        ...         message="Checkpoint", metadata={"score": 75}
        ...     )
        ...     assert sha == "abc123"

    """
    mock = Mock()

    # Mock common git operations
    mock.create_checkpoint_commit = Mock(return_value="abc123def456")
    mock.get_git_status = Mock(
        return_value={
            "branch": "main",
            "status": "clean",
            "ahead": 0,
            "behind": 0,
            "staged": [],
            "unstaged": [],
            "untracked": [],
        }
    )
    mock.get_commit_history = Mock(
        return_value=[
            {
                "sha": "abc123",
                "message": "Initial commit",
                "author": "Test User",
                "date": "2025-10-29",
            }
        ]
    )
    mock.detect_branch = Mock(return_value="main")
    mock.is_git_repository = Mock(return_value=True)

    return mock


@pytest.fixture
def git_commit_data_factory() -> t.Callable[..., dict[str, t.Any]]:
    """Create a factory for generating git commit test data.

    Returns:
        Factory function that creates git commit data dictionaries.

    Example:
        >>> factory = git_commit_data_factory()
        >>> commit = factory(message="Test commit", quality_score=85)
        >>> assert commit["message"] == "Test commit"
        >>> assert commit["metadata"]["quality_score"] == 85

    """

    def factory(
        message: str = "Test commit",
        quality_score: int | None = None,
        checkpoint_number: int | None = None,
        metadata: dict[str, t.Any] | None = None,
    ) -> dict[str, t.Any]:
        """Create git commit data with given parameters.

        Args:
            message: Commit message.
            quality_score: Optional quality score for checkpoint.
            checkpoint_number: Optional checkpoint number.
            metadata: Additional metadata to include.

        Returns:
            Git commit data dictionary.

        """
        commit_data: dict[str, t.Any] = {
            "message": message,
            "author": "Test User <test@example.com>",
            "timestamp": "2025-10-29T12:00:00Z",
        }

        # Build metadata
        commit_metadata: dict[str, t.Any] = {}

        if quality_score is not None:
            commit_metadata["quality_score"] = quality_score

        if checkpoint_number is not None:
            commit_metadata["checkpoint_number"] = checkpoint_number

        if metadata:
            commit_metadata.update(metadata)

        if commit_metadata:
            commit_data["metadata"] = commit_metadata

        return commit_data

    return factory


@pytest.fixture
def mock_git_status_factory() -> t.Callable[..., dict[str, t.Any]]:
    """Create a factory for generating git status test data.

    Returns:
        Factory function that creates git status dictionaries.

    Example:
        >>> factory = mock_git_status_factory()
        >>> status = factory(branch="feature", staged=["file.py"], unstaged=["test.py"])
        >>> assert status["branch"] == "feature"
        >>> assert len(status["staged"]) == 1

    """

    def factory(
        branch: str = "main",
        status: str = "clean",
        ahead: int = 0,
        behind: int = 0,
        staged: list[str] | None = None,
        unstaged: list[str] | None = None,
        untracked: list[str] | None = None,
    ) -> dict[str, t.Any]:
        """Create git status data with given parameters.

        Args:
            branch: Current branch name.
            status: Repository status ("clean", "dirty", "ahead", "behind").
            ahead: Number of commits ahead of remote.
            behind: Number of commits behind remote.
            staged: List of staged files.
            unstaged: List of unstaged modified files.
            untracked: List of untracked files.

        Returns:
            Git status dictionary.

        """
        return {
            "branch": branch,
            "status": status,
            "ahead": ahead,
            "behind": behind,
            "staged": staged or [],
            "unstaged": unstaged or [],
            "untracked": untracked or [],
        }

    return factory


@pytest.fixture
def mock_checkpoint_metadata_factory() -> t.Callable[..., dict[str, t.Any]]:
    """Create a factory for generating checkpoint metadata.

    Returns:
        Factory function that creates checkpoint metadata dictionaries.

    Example:
        >>> factory = mock_checkpoint_metadata_factory()
        >>> metadata = factory(checkpoint_number=5, quality_score=85)
        >>> assert metadata["checkpoint_number"] == 5
        >>> assert metadata["quality"]["score"] == 85

    """

    def factory(
        checkpoint_number: int = 1,
        quality_score: int = 75,
        tests_passing: int | None = None,
        coverage: float | None = None,
        recommendations: list[str] | None = None,
    ) -> dict[str, t.Any]:
        """Create checkpoint metadata with given parameters.

        Args:
            checkpoint_number: Checkpoint number.
            quality_score: Quality score (0-100).
            tests_passing: Number of passing tests.
            coverage: Code coverage percentage.
            recommendations: Quality improvement recommendations.

        Returns:
            Checkpoint metadata dictionary.

        """
        metadata: dict[str, t.Any] = {
            "checkpoint_number": checkpoint_number,
            "timestamp": "2025-10-29T12:00:00Z",
            "quality": {
                "score": quality_score,
            },
        }

        if tests_passing is not None:
            metadata["quality"]["tests_passing"] = tests_passing

        if coverage is not None:
            metadata["quality"]["coverage"] = coverage

        if recommendations:
            metadata["recommendations"] = recommendations

        return metadata

    return factory
