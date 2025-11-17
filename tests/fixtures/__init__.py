"""Test fixtures and data factories for session-mgmt-mcp testing.

Week 8 Day 2 - Phase 2: Comprehensive fixture modules for server testing.

Available Fixtures:
- server_fixtures: MockMCP server, session paths, loggers, managers
- git_fixtures: Git repository fixtures, commit data, status factories
- crackerjack_fixtures: Mock crackerjack output, metrics, integration
"""

# Import commonly used fixtures for easy access
from tests.fixtures.crackerjack_fixtures import (
    crackerjack_metrics_factory,
    crackerjack_output_factory,
    mock_crackerjack_command_result,
    mock_crackerjack_integration,
    mock_crackerjack_metrics_failures,
    mock_crackerjack_metrics_success,
    mock_crackerjack_output_failures,
    mock_crackerjack_output_success,
)
from tests.fixtures.git_fixtures import (
    git_commit_data_factory,
    mock_checkpoint_metadata_factory,
    mock_git_operations,
    mock_git_status_factory,
    tmp_git_repo,
    tmp_git_repo_with_changes,
    tmp_git_repo_with_commits,
)
from tests.fixtures.server_fixtures import (
    mock_fastmcp_server,
    mock_health_check_result,
    mock_lifecycle_manager,
    mock_mcp_server_context,
    mock_permissions_manager,
    mock_quality_score_result,
    mock_session_logger,
    mock_session_paths,
    mock_tool_result_factory,
)

__all__ = [
    "crackerjack_metrics_factory",
    "crackerjack_output_factory",
    "git_commit_data_factory",
    "mock_checkpoint_metadata_factory",
    "mock_crackerjack_command_result",
    "mock_crackerjack_integration",
    "mock_crackerjack_metrics_failures",
    "mock_crackerjack_metrics_success",
    "mock_crackerjack_output_failures",
    # Crackerjack fixtures
    "mock_crackerjack_output_success",
    # Server fixtures
    "mock_fastmcp_server",
    "mock_git_operations",
    "mock_git_status_factory",
    "mock_health_check_result",
    "mock_lifecycle_manager",
    "mock_mcp_server_context",
    "mock_permissions_manager",
    "mock_quality_score_result",
    "mock_session_logger",
    "mock_session_paths",
    "mock_tool_result_factory",
    # Git fixtures
    "tmp_git_repo",
    "tmp_git_repo_with_changes",
    "tmp_git_repo_with_commits",
]
