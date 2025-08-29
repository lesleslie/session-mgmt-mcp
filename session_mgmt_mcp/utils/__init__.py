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
    "SessionLogger",
    "get_session_logger",
    "is_git_repository",
    "get_git_status",
    "stage_files",
    "get_staged_files",
    "create_commit",
    "create_checkpoint_commit",
    "DatabaseConnectionPool",
    "get_database_pool",
    "LazyImport",
    "LazyLoader",
    "lazy_loader",
    "require_dependency",
    "optional_dependency",
    "get_dependency_status",
    "log_dependency_status",
]
