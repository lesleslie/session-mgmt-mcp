"""Base classes for serverless session storage backends.

This module provides the abstract base class and data models for session storage
backends including Redis, S3, local file storage, and ACB cache.
"""

from __future__ import annotations

import gzip
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SessionState(BaseModel):
    """Represents complete session state for serialization."""

    session_id: str = Field(
        min_length=1,
        description="Unique identifier for the session",
    )
    user_id: str = Field(min_length=1, description="Identifier for the user")
    project_id: str = Field(min_length=1, description="Identifier for the project")
    created_at: str = Field(description="ISO timestamp when session was created")
    last_activity: str = Field(description="ISO timestamp of last activity")
    permissions: list[str] = Field(
        default_factory=list,
        description="List of permissions granted to the session",
    )
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of conversation entries",
    )
    reflection_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Stored reflection and memory data",
    )
    app_monitoring_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Application monitoring state",
    )
    llm_provider_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM provider configurations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata",
    )

    @field_validator("created_at", "last_activity")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        """Validate that timestamps are in ISO format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError as e:
            msg = f"Invalid ISO timestamp format: {v}"
            raise ValueError(msg) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Create from dictionary."""
        return cls.model_validate(data)

    def get_compressed_size(self) -> int:
        """Get compressed size of session state."""
        serialized = self.model_dump_json()
        compressed = gzip.compress(serialized.encode("utf-8"))
        return len(compressed)


class SessionStorage(ABC):
    """Abstract base class for session storage backends."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"serverless.{self.__class__.__name__.lower()}")

    @abstractmethod
    async def store_session(
        self,
        session_state: SessionState,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store session state with optional TTL."""

    @abstractmethod
    async def retrieve_session(self, session_id: str) -> SessionState | None:
        """Retrieve session state by ID."""

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session state."""

    @abstractmethod
    async def list_sessions(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[str]:
        """List session IDs matching criteria."""

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions, return count removed."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if storage backend is available."""
