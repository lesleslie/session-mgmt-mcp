"""Caching layer for history analysis to improve performance."""

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Cache entry with expiration."""

    data: dict[str, Any]
    timestamp: float
    ttl: float = 300.0  # 5 minutes default TTL

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.timestamp) > self.ttl


class HistoryAnalysisCache:
    """In-memory cache for history analysis results.

    Caches expensive history analysis operations to avoid
    redundant database queries and pattern extraction.
    """

    def __init__(self, ttl: float = 300.0):
        """Initialize cache.

        Args:
            ttl: Time-to-live in seconds (default: 5 minutes)

        """
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl

    def _generate_key(self, project: str, days: int) -> str:
        """Generate cache key from parameters.

        Args:
            project: Project name
            days: Number of days analyzed

        Returns:
            Cache key string

        """
        # Create deterministic key from parameters
        params = f"{project}:{days}"
        return hashlib.md5(params.encode(), usedforsecurity=False).hexdigest()

    def get(self, project: str, days: int) -> dict[str, Any] | None:
        """Retrieve cached analysis result.

        Args:
            project: Project name
            days: Number of days analyzed

        Returns:
            Cached analysis dict or None if not found/expired

        """
        key = self._generate_key(project, days)
        entry = self._cache.get(key)

        if entry and not entry.is_expired():
            return entry.data

        # Remove expired entry
        if entry:
            del self._cache[key]

        return None

    def set(self, project: str, days: int, data: dict[str, Any]) -> None:
        """Store analysis result in cache.

        Args:
            project: Project name
            days: Number of days analyzed
            data: Analysis result dictionary

        """
        key = self._generate_key(project, days)
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=self._ttl,
        )

    def invalidate(self, project: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            project: Optional project name to invalidate specific project
                    If None, clears entire cache

        """
        if project is None:
            self._cache.clear()
        else:
            # Remove all entries for this project
            keys_to_remove = [
                key
                for key in self._cache  # FURB135: Value is unused
                if key.startswith(
                    hashlib.md5(project.encode(), usedforsecurity=False).hexdigest()[:8]
                )
            ]
            for key in keys_to_remove:
                del self._cache[key]

    def size(self) -> int:
        """Get number of cached entries.

        Returns:
            Number of cache entries

        """
        # Clean up expired entries first
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

        return len(self._cache)

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats

        """
        expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())

        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "active_entries": len(self._cache) - expired_count,
        }


# Global cache instance
_global_cache: HistoryAnalysisCache | None = None


def get_cache(ttl: float = 300.0) -> HistoryAnalysisCache:
    """Get or create global cache instance.

    Args:
        ttl: Time-to-live in seconds (default: 5 minutes)

    Returns:
        Global cache instance

    """
    global _global_cache
    if _global_cache is None:
        _global_cache = HistoryAnalysisCache(ttl=ttl)
    return _global_cache


def reset_cache() -> None:
    """Reset global cache instance.

    Useful for testing or clearing all cached data.
    """
    global _global_cache
    _global_cache = None
