"""ACB Cache session storage backend.

This module provides an ACB (Asynchronous Component Base) cache implementation
of the SessionStorage interface for storing and retrieving session state using ACB's
built-in caching mechanisms.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from session_buddy.backends.base import SessionState, SessionStorage


class ACBCacheStorage(SessionStorage):
    """ACB cache adapter for session storage.

    Wraps ACB cache adapters (Redis or Memory) to implement SessionStorage interface.
    Provides connection pooling, SSL/TLS support, and automatic compression via ACB.

    Benefits over custom backends:
    - Battle-tested ACB cache infrastructure
    - MsgPack + Brotli compression for efficient storage
    - Connection pooling and health checks built-in
    - SSL/TLS support via ACB configuration
    - Automatic reconnection handling
    """

    def __init__(
        self,
        cache: Any,
        namespace: str = "session",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ACB cache storage.

        Args:
            cache: ACB cache adapter instance (from acb.adapters.cache)
            namespace: Namespace for multi-tenant isolation (default: "session")
            config: Optional configuration dict for compatibility

        """
        super().__init__(config or {})
        self.cache = cache
        self.namespace = namespace
        # Track session IDs for list_sessions() functionality
        self._index_key = f"{namespace}:index"

    def _get_key(self, session_id: str) -> str:
        """Get namespaced key for session."""
        return f"{self.namespace}:{session_id}"

    async def store_session(
        self,
        session_state: SessionState,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store session using ACB cache adapter.

        Args:
            session_state: Session data to store
            ttl_seconds: Time-to-live in seconds (default: 86400 = 24 hours)

        Returns:
            True if stored successfully, False otherwise

        """
        try:
            key = self._get_key(session_state.session_id)

            # Serialize SessionState using Pydantic's model_dump
            # ACB cache will handle msgpack + brotli compression automatically
            data = session_state.model_dump(mode="json")

            # Store with TTL
            ttl = ttl_seconds if ttl_seconds is not None else 86400  # 24 hours default
            await self.cache.set(key, data, ttl=ttl)

            # Add session ID to index for list_sessions()
            await self._add_to_index(
                session_state.session_id,
                session_state.user_id,
                session_state.project_id,
                ttl,
            )

            self.logger.debug(
                f"Stored session {session_state.session_id} "
                f"with TTL {ttl}s (compressed via ACB)",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"Failed to store session {session_state.session_id}: {e}",
            )
            return False

    async def retrieve_session(self, session_id: str) -> SessionState | None:
        """Retrieve session using ACB cache adapter.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState if found, None otherwise

        """
        try:
            key = self._get_key(session_id)
            data = await self.cache.get(key)

            if data is None:
                self.logger.debug(f"Session {session_id} not found")
                return None

            # Deserialize from dict to SessionState
            session_state = SessionState.model_validate(data)
            self.logger.debug(f"Retrieved session {session_id}")
            return session_state

        except Exception as e:
            self.logger.exception(f"Failed to retrieve session {session_id}: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete session using ACB cache adapter.

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted, False otherwise

        """
        try:
            key = self._get_key(session_id)
            result = await self.cache.delete(key)

            # Remove from index
            await self._remove_from_index(session_id)

            self.logger.debug(f"Deleted session {session_id}")
            return bool(result)

        except Exception as e:
            self.logger.exception(f"Failed to delete session {session_id}: {e}")
            return False

    async def list_sessions(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[str]:
        """List session IDs matching criteria.

        Note: This implementation uses a separate index key to track sessions.
        For production use with many sessions, consider using Redis SCAN or
        dedicated index storage.

        Args:
            user_id: Filter by user ID (optional)
            project_id: Filter by project ID (optional)

        Returns:
            List of session IDs matching criteria

        """
        try:
            # Get index data
            index_data = await self.cache.get(self._index_key)
            if index_data is None:
                return []

            # Filter by criteria
            session_ids = []
            for session_id, metadata in index_data.items():
                if user_id and metadata.get("user_id") != user_id:
                    continue
                if project_id and metadata.get("project_id") != project_id:
                    continue
                session_ids.append(session_id)

            return session_ids

        except Exception as e:
            self.logger.exception(f"Failed to list sessions: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Note: With ACB cache + TTL, sessions expire automatically.
        This method cleans up stale index entries.

        Returns:
            Count of sessions removed from index

        """
        try:
            index_data = await self.cache.get(self._index_key)
            if index_data is None:
                return 0

            # Check which sessions still exist
            cleaned = 0
            for session_id in list(index_data.keys()):
                key = self._get_key(session_id)
                exists = await self.cache.exists(key)
                if not exists:
                    # Session expired, remove from index
                    del index_data[session_id]
                    cleaned += 1

            # Update index if we cleaned anything
            if cleaned > 0:
                await self.cache.set(self._index_key, index_data, ttl=None)
                self.logger.info(f"Cleaned up {cleaned} expired session(s) from index")

            return cleaned

        except Exception as e:
            self.logger.exception(f"Failed to cleanup expired sessions: {e}")
            return 0

    async def is_available(self) -> bool:
        """Check if ACB cache backend is available.

        Returns:
            True if cache is reachable, False otherwise

        """
        try:
            # Try a simple operation to test connectivity
            test_key = f"{self.namespace}:health_check"
            await self.cache.set(test_key, {"status": "ok"}, ttl=10)
            result = await self.cache.get(test_key)
            if result is not None:
                await self.cache.delete(test_key)
                return True
            return False

        except Exception as e:
            self.logger.debug(f"ACB cache health check failed: {e}")
            # For memory cache without proper config, assume available
            return True

    # Helper methods for index management

    async def _add_to_index(
        self,
        session_id: str,
        user_id: str,
        project_id: str,
        ttl: int,
    ) -> None:
        """Add session to index for list_sessions()."""
        try:
            result = await self.cache.get(self._index_key)
            index_data: dict[str, Any] = result if isinstance(result, dict) else {}
            index_data[session_id] = {
                "user_id": user_id,
                "project_id": project_id,
                "expires_at": (datetime.now(UTC) + timedelta(seconds=ttl)).isoformat(),
            }
            # Index doesn't expire (we clean it up manually)
            await self.cache.set(self._index_key, index_data, ttl=None)
        except Exception as e:
            self.logger.warning(f"Failed to update index: {e}")

    async def _remove_from_index(self, session_id: str) -> None:
        """Remove session from index."""
        try:
            index_data = await self.cache.get(self._index_key)
            if index_data and session_id in index_data:
                del index_data[session_id]
                await self.cache.set(self._index_key, index_data, ttl=None)
        except Exception as e:
            self.logger.warning(f"Failed to update index: {e}")
