"""Utility functions for session-mgmt-mcp."""

from .database_pool import DatabaseConnectionPool, get_database_pool
from .git_operations import (
    create_checkpoint_commit,
    create_commit,
    get_git_status,
    get_staged_files,
    is_git_repository,
    stage_files,
)
from .lazy_imports import (
    LazyImport,
    LazyLoader,
    get_dependency_status,
    lazy_loader,
    log_dependency_status,
    optional_dependency,
    require_dependency,
)
from .logging import SessionLogger, get_session_logger

__all__ = [
    "DatabaseConnectionPool",
    "LazyImport",
    "LazyLoader",
    "SessionLogger",
    "create_checkpoint_commit",
    "create_commit",
    "get_database_pool",
    "get_dependency_status",
    "get_git_status",
    "get_session_logger",
    "get_staged_files",
    "is_git_repository",
    "lazy_loader",
    "log_dependency_status",
    "optional_dependency",
    "require_dependency",
    "stage_files",
]
