"""Storage backends for serverless session management.

This package provides various storage backend implementations for session state:
- RedisStorage: Redis-based storage with TTL support
- S3Storage: S3-compatible object storage
- LocalFileStorage: Local file system storage (development/testing)
- ACBCacheStorage: ACB cache-based storage

All backends implement the SessionStorage abstract interface.
"""

from session_buddy.backends.acb_cache_backend import ACBCacheStorage
from session_buddy.backends.base import SessionState, SessionStorage
from session_buddy.backends.local_backend import LocalFileStorage
from session_buddy.backends.redis_backend import RedisStorage
from session_buddy.backends.s3_backend import S3Storage

__all__ = [
    "ACBCacheStorage",
    "LocalFileStorage",
    "RedisStorage",
    "S3Storage",
    "SessionState",
    "SessionStorage",
]
