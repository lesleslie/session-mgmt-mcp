"""Caching layer for history analysis to improve performance.

This module now uses ACB-backed cache for improved performance and
lifecycle management while maintaining backwards-compatible API.
"""

from session_buddy.acb_cache_adapter import ACBHistoryCache, get_history_cache

# Backwards-compatible alias
HistoryAnalysisCache = ACBHistoryCache


def get_cache(ttl: float = 300.0) -> ACBHistoryCache:
    """Get or create global cache instance.

    Args:
        ttl: Time-to-live in seconds (default: 5 minutes)

    Returns:
        Global cache instance using ACB-backed implementation

    """
    return get_history_cache(ttl=ttl)


async def reset_cache() -> None:
    """Reset global cache instance.

    Useful for testing or clearing all cached data.
    """
    from session_buddy.acb_cache_adapter import reset_caches

    await reset_caches()
