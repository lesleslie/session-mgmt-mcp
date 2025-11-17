"""ACB adapter compatibility wrappers for session management.

This module provides compatibility wrappers that maintain the existing ReflectionDatabase
and KnowledgeGraphDatabase APIs while using ACB adapters under the hood.

Phase 1: Storage adapter integration for session state persistence
Phase 2.7 (Week 7 Day 1-5): Migration to ACB dependency injection
"""

from __future__ import annotations

from .session_storage_adapter import (
    DEFAULT_SESSION_BUCKET,
    SessionStorageAdapter,
    get_default_storage_adapter,
)
from .storage_registry import (
    SUPPORTED_BACKENDS,
    configure_storage_buckets,
    get_default_session_buckets,
    get_storage_adapter,
    register_storage_adapter,
)

__all__ = [
    "DEFAULT_SESSION_BUCKET",
    "SUPPORTED_BACKENDS",
    # Reflection adapter (Phase 2.7)
    "ReflectionDatabaseAdapter",
    # Storage adapters (Phase 1)
    "SessionStorageAdapter",
    "configure_storage_buckets",
    "get_default_session_buckets",
    "get_default_storage_adapter",
    "get_storage_adapter",
    # Storage registry
    "register_storage_adapter",
]
