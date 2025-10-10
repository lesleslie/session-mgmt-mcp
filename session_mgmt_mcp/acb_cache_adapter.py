"""ACB-backed cache adapters for session-mgmt-mcp.

This module provides cache adapters using aiocache (ACB's underlying cache),
maintaining backwards-compatible sync interfaces while leveraging ACB's
optimized serialization and lifecycle management.
"""

import asyncio
import hashlib
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta

from aiocache import SimpleMemoryCache
from aiocache.serializers import PickleSerializer


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def to_dict(self) -> dict[str, t.Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_entries": self.total_entries,
            "hit_rate_percent": round(self.hit_rate, 2),
        }


class ACBChunkCache:
    """ACB-backed chunk cache for token optimizer.

    Maintains backwards-compatible API with TokenOptimizer.chunk_cache
    while using ACB's aiocache for improved performance and lifecycle management.
    """

    def __init__(self, ttl: int = 3600) -> None:
        """Initialize chunk cache.

        Args:
            ttl: Default time-to-live in seconds (default: 1 hour)

        """
        self._cache = SimpleMemoryCache(
            serializer=PickleSerializer(),
            namespace="session_mgmt:chunks:",
        )
        self._cache.timeout = 0.0  # No operation timeout
        self._ttl = ttl

        # Stats tracking
        self.stats = CacheStats()

        # Event loop for sync API compatibility
        self._loop = asyncio.new_event_loop()

    def _run_async(self, coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
        """Execute async operation in sync context.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of the async operation

        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # We're in an async context, close coroutine and return None
            coro.close()
            return None
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            return self._loop.run_until_complete(coro)

    def set(self, key: str, value: t.Any, ttl: int | None = None) -> None:
        """Store chunk data in cache.

        Args:
            key: Cache key
            value: Value to cache (ChunkResult)
            ttl: Optional TTL override in seconds

        """
        effective_ttl = ttl or self._ttl
        self._run_async(self._cache.set(key, value, ttl=effective_ttl))
        self.stats.total_entries += 1

    def get(self, key: str) -> t.Any | None:
        """Retrieve chunk data from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired

        """
        result = self._run_async(self._cache.get(key))

        if result is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1

        return result

    def delete(self, key: str) -> None:
        """Delete chunk data from cache.

        Args:
            key: Cache key to delete

        """
        self._run_async(self._cache.delete(key))
        self.stats.evictions += 1

    def clear(self) -> None:
        """Clear all cached chunk data."""
        self._run_async(self._cache.clear())
        self.stats = CacheStats()  # Reset stats

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is not expired

        """
        return self._run_async(self._cache.exists(key))

    def __getitem__(self, key: str) -> t.Any:
        """Get item using dict syntax.

        Args:
            key: Cache key

        Returns:
            Cached value

        Raises:
            KeyError: If key not found in cache

        """
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: t.Any) -> None:
        """Set item using dict syntax.

        Args:
            key: Cache key
            value: Value to cache

        """
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item using dict syntax.

        Args:
            key: Cache key to delete

        """
        self.delete(key)

    def keys(self) -> list[str]:
        """Get all cache keys.

        Note: This is a best-effort implementation as aiocache
        doesn't provide efficient key listing.

        Returns:
            List of cache keys (may be incomplete)

        """
        # aiocache SimpleMemoryCache doesn't expose keys directly
        # This is a limitation we accept for ACB integration
        return []

    def get_stats(self) -> dict[str, t.Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics

        """
        return {"chunk_cache": self.stats.to_dict()}


class ACBHistoryCache:
    """ACB-backed history cache for analysis results.

    Maintains backwards-compatible API with HistoryAnalysisCache
    while using ACB's aiocache for improved performance.
    """

    def __init__(self, ttl: float = 300.0) -> None:
        """Initialize history cache.

        Args:
            ttl: Time-to-live in seconds (default: 5 minutes)

        """
        self._cache = SimpleMemoryCache(
            serializer=PickleSerializer(),
            namespace="session_mgmt:history:",
        )
        self._cache.timeout = 0.0  # No operation timeout
        self._ttl = int(ttl)  # Convert to int for aiocache

        # Stats tracking
        self.stats = CacheStats()

        # Event loop for sync API compatibility
        self._loop = asyncio.new_event_loop()

    def _run_async(self, coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
        """Execute async operation in sync context.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of the async operation

        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # We're in an async context, close coroutine and return None
            coro.close()
            return None
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            return self._loop.run_until_complete(coro)

    def _generate_key(self, project: str, days: int) -> str:
        """Generate cache key from parameters.

        Args:
            project: Project name
            days: Number of days analyzed

        Returns:
            Cache key string

        """
        params = f"{project}:{days}"
        return hashlib.md5(params.encode(), usedforsecurity=False).hexdigest()

    def get(self, project: str, days: int) -> dict[str, t.Any] | None:
        """Retrieve cached analysis result.

        Args:
            project: Project name
            days: Number of days analyzed

        Returns:
            Cached analysis dict or None if not found/expired

        """
        key = self._generate_key(project, days)
        result = self._run_async(self._cache.get(key))

        if result is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1

        return result

    def set(self, project: str, days: int, data: dict[str, t.Any]) -> None:
        """Store analysis result in cache.

        Args:
            project: Project name
            days: Number of days analyzed
            data: Analysis result dictionary

        """
        key = self._generate_key(project, days)
        self._run_async(self._cache.set(key, data, ttl=self._ttl))
        self.stats.total_entries += 1

    def invalidate(self, project: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            project: Optional project name (if None, clears entire cache)

        Note:
            ACB cache doesn't support efficient pattern-based deletion.
            If project is specified, this is a no-op with a warning.
            If project is None, clears all cache.

        """
        if project is None:
            self._run_async(self._cache.clear())
            self.stats = CacheStats()  # Reset stats
        else:
            import warnings

            warnings.warn(
                "ACB cache doesn't support selective invalidation by project. "
                "Use invalidate(None) to clear all cached data.",
                stacklevel=2,
            )

    def size(self) -> int:
        """Get number of cached entries.

        Returns:
            Number of cache entries (approximate)

        Note:
            ACB cache automatically handles expiration, so this returns
            the stats total_entries count which may include expired entries
            not yet cleaned up.

        """
        return self.stats.total_entries

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats

        """
        # Note: We can't accurately track expired entries with aiocache
        # as it handles expiration automatically via TTL
        return {
            "total_entries": self.stats.total_entries,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "expired_entries": 0,  # ACB handles this automatically
            "active_entries": self.stats.total_entries,  # Approximate
        }


# Global cache instances
_chunk_cache: ACBChunkCache | None = None
_history_cache: ACBHistoryCache | None = None


def get_chunk_cache(ttl: int = 3600) -> ACBChunkCache:
    """Get or create global chunk cache instance.

    Args:
        ttl: Time-to-live in seconds (default: 1 hour)

    Returns:
        Global chunk cache instance

    """
    global _chunk_cache
    if _chunk_cache is None:
        _chunk_cache = ACBChunkCache(ttl=ttl)
    return _chunk_cache


def get_history_cache(ttl: float = 300.0) -> ACBHistoryCache:
    """Get or create global history cache instance.

    Args:
        ttl: Time-to-live in seconds (default: 5 minutes)

    Returns:
        Global history cache instance

    """
    global _history_cache
    if _history_cache is None:
        _history_cache = ACBHistoryCache(ttl=ttl)
    return _history_cache


def reset_caches() -> None:
    """Reset global cache instances.

    Useful for testing or clearing all cached data.
    """
    global _chunk_cache, _history_cache
    if _chunk_cache:
        _chunk_cache.clear()
    if _history_cache:
        _history_cache.invalidate()
    _chunk_cache = None
    _history_cache = None
