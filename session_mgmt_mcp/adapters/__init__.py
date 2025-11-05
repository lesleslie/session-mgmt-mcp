"""ACB adapter compatibility wrappers for session management.

This module provides compatibility wrappers that maintain the existing ReflectionDatabase
and KnowledgeGraphDatabase APIs while using ACB adapters under the hood.

Phase 2.7 (Week 7 Day 1-5): Migration to ACB dependency injection
"""

from __future__ import annotations

__all__ = ["ReflectionDatabaseAdapter"]
