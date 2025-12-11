#!/usr/bin/env python3
"""Stateless/Serverless Mode for Session Management MCP Server.

Enables request-scoped sessions with external storage backends (Redis, S3, DynamoDB).
Allows the session management server to operate in cloud/serverless environments.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from session_buddy.adapters.serverless_storage_adapter import (
    ServerlessStorageAdapter,
)
from session_buddy.backends import (
    ACBCacheStorage,
    LocalFileStorage,
    RedisStorage,
    S3Storage,
    SessionState,
    SessionStorage,
)

CONFIG_LOGGER = logging.getLogger("serverless.config")


class ServerlessSessionManager:
    """Main session manager for serverless/stateless operation."""

    def __init__(self, storage_backend: SessionStorage) -> None:
        self.storage = storage_backend
        self.logger = logging.getLogger("serverless.session_manager")
        self.session_cache: dict[
            str,
            SessionState,
        ] = {}  # In-memory cache for current request

    async def create_session(
        self,
        user_id: str,
        project_id: str,
        session_data: dict[str, Any] | None = None,
        ttl_hours: int = 24,
    ) -> str:
        """Create new session."""
        session_id = self._generate_session_id(user_id, project_id)

        session_state = SessionState(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            created_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat(),
            permissions=[],
            conversation_history=[],
            reflection_data={},
            app_monitoring_state={},
            llm_provider_configs={},
            metadata=session_data or {},
        )

        # Store with TTL
        ttl_seconds = ttl_hours * 3600
        success = await self.storage.store_session(session_state, ttl_seconds)

        if success:
            self.session_cache[session_id] = session_state
            return session_id
        msg = "Failed to create session"
        raise RuntimeError(msg)

    async def get_session(self, session_id: str) -> SessionState | None:
        """Get session state."""
        # Check cache first
        if session_id in self.session_cache:
            return self.session_cache[session_id]

        # Load from storage
        session_state = await self.storage.retrieve_session(session_id)
        if session_state:
            self.session_cache[session_id] = session_state

        return session_state

    async def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
        ttl_hours: int | None = None,
    ) -> bool:
        """Update session state."""
        session_state = await self.get_session(session_id)
        if not session_state:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(session_state, key):
                setattr(session_state, key, value)

        # Update last activity
        session_state.last_activity = datetime.now().isoformat()

        # Store updated state
        ttl_seconds = ttl_hours * 3600 if ttl_hours else None
        success = await self.storage.store_session(session_state, ttl_seconds)

        if success:
            self.session_cache[session_id] = session_state

        return success

    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        # Remove from cache
        self.session_cache.pop(session_id, None)

        # Delete from storage
        return await self.storage.delete_session(session_id)

    async def list_user_sessions(self, user_id: str) -> list[str]:
        """List sessions for user."""
        return await self.storage.list_sessions(user_id=user_id)

    async def list_project_sessions(self, project_id: str) -> list[str]:
        """List sessions for project."""
        return await self.storage.list_sessions(project_id=project_id)

    async def cleanup_sessions(self) -> int:
        """Clean up expired sessions."""
        return await self.storage.cleanup_expired_sessions()

    def _generate_session_id(self, user_id: str, project_id: str) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        data = f"{user_id}:{project_id}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "cached_sessions": len(self.session_cache),
            "storage_backend": self.storage.__class__.__name__,
            "storage_config": {
                k: v for k, v in self.storage.config.items() if "key" not in k.lower()
            },
        }


class ServerlessConfigManager:
    """Manages configuration for serverless mode."""

    @staticmethod
    def load_config(config_path: str | None = None) -> dict[str, Any]:
        """Load serverless configuration."""
        default_config = {
            "storage_backend": "local",
            "session_ttl_hours": 24,
            "cleanup_interval_hours": 6,
            "backends": {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "key_prefix": "session_mgmt:",
                },
                "s3": {
                    "bucket_name": "session-buddy",
                    "region": "us-east-1",
                    "key_prefix": "sessions/",
                },
                "local": {
                    "storage_dir": str(Path.home() / ".claude" / "data" / "sessions"),
                },
            },
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except (OSError, json.JSONDecodeError):
                pass

        return default_config

    @staticmethod
    def create_storage_backend(config: dict[str, Any]) -> SessionStorage:
        """Create storage backend from config.

        Supports new ACB storage adapters (file, s3, azure, gcs, memory) and
        legacy backends (redis, local, acb cache) for backward compatibility.

        Recommended backends (ACB storage adapters):
        - "file": Local file storage (default, best for development)
        - "s3": AWS S3/MinIO (production cloud storage)
        - "azure": Azure Blob Storage (Azure deployments)
        - "gcs": Google Cloud Storage (GCP deployments)
        - "memory": In-memory storage (testing only)

        Legacy backends (deprecated, will be removed in v1.0):
        - "redis": Redis cache (use "acb" or "s3" instead)
        - "local": Old local file storage (use "file" instead)
        - "acb": Old ACB cache backend (use "file" or "s3" instead)
        """
        backend_type = config.get("storage_backend", "file")  # Default to file
        backend_config = config.get("backends", {}).get(backend_type, {})

        # New ACB storage adapters (recommended)
        if backend_type in ("file", "s3", "azure", "gcs", "memory"):
            CONFIG_LOGGER.info(
                f"Using ACB storage adapter: {backend_type} "
                "(recommended for production deployments)"
            )
            return ServerlessStorageAdapter(config=backend_config, backend=backend_type)

        # ACB-style cache backend (using aiocache directly)
        if backend_type == "acb":
            try:
                # Use aiocache directly (same as ACB uses internally)
                # This avoids ACB DI complexity while still getting benefits
                from aiocache import Cache as AIOCache
                from aiocache.serializers import PickleSerializer

                cache_type = backend_config.get("cache_type", "memory")

                # Configure aiocache backend
                if cache_type == "redis":
                    try:
                        from aiocache.backends.redis import RedisBackend

                        cache = AIOCache(
                            cache_class=RedisBackend,
                            serializer=PickleSerializer(),
                            endpoint=backend_config.get("host", "localhost"),
                            port=backend_config.get("port", 6379),
                            password=backend_config.get("password"),
                            db=backend_config.get("db", 0),
                        )
                    except ImportError:
                        CONFIG_LOGGER.warning("redis not available, using memory cache")
                        cache = AIOCache(serializer=PickleSerializer())
                else:
                    # Memory cache (default)
                    cache = AIOCache(serializer=PickleSerializer())

                namespace = backend_config.get("namespace", "session")
                return ACBCacheStorage(cache, namespace, backend_config)

            except ImportError as e:
                CONFIG_LOGGER.warning(
                    f"aiocache not available ({e}), falling back to local file storage. "
                    "Install with: pip install aiocache[redis]",
                )
                # Fallback to local storage if aiocache not available
                return LocalFileStorage(backend_config)

        # Legacy backends (deprecated)
        if backend_type == "redis":
            CONFIG_LOGGER.warning(
                "RedisStorage is deprecated. Use 'acb' backend for better "
                "connection pooling, SSL/TLS support, and automatic compression.",
            )
            return RedisStorage(backend_config)

        if backend_type == "s3":
            CONFIG_LOGGER.warning(
                "S3Storage is deprecated. Consider using ACB cache backends "
                "with Redis for production deployments.",
            )
            return S3Storage(backend_config)

        if backend_type == "local":
            return LocalFileStorage(backend_config)

        msg = f"Unsupported storage backend: {backend_type}"
        raise ValueError(msg)

    @staticmethod
    async def test_storage_backends(config: dict[str, Any]) -> dict[str, bool]:
        """Test all configured storage backends."""
        results: dict[str, bool] = {}

        for backend_name, backend_config in config.get("backends", {}).items():
            try:
                storage: SessionStorage
                match backend_name:
                    # New ACB storage adapters
                    case "file" | "s3" | "azure" | "gcs" | "memory":
                        storage = ServerlessStorageAdapter(
                            config=backend_config, backend=backend_name
                        )

                    case "acb":
                        # Test ACB-style cache backend
                        try:
                            from aiocache import Cache as AIOCache
                            from aiocache.serializers import PickleSerializer

                            backend_config.get("cache_type", "memory")
                            cache = AIOCache(serializer=PickleSerializer())
                            namespace = backend_config.get("namespace", "session")
                            storage = ACBCacheStorage(cache, namespace, backend_config)
                        except ImportError:
                            results[backend_name] = False
                            continue

                    case "redis":
                        storage = RedisStorage(backend_config)
                    case "s3":
                        storage = S3Storage(backend_config)
                    case "local":
                        storage = LocalFileStorage(backend_config)
                    case _:
                        results[backend_name] = False
                        continue

                results[backend_name] = await storage.is_available()

            except Exception:
                results[backend_name] = False

        return results
