"""Storage adapter registry for session state persistence.

This module provides registration and configuration for ACB storage adapters,
enabling multiple backend types (S3, Azure, GCS, File, Memory) for session
state persistence in serverless and distributed deployments.

The registry follows ACB's dependency injection pattern and provides:
- Type-safe adapter configuration via StorageSettings
- Runtime backend selection from config
- Automatic adapter initialization and cleanup
- Bucket management for organized storage

Example:
    >>> from session_mgmt_mcp.adapters.storage_registry import get_storage_adapter
    >>> storage = get_storage_adapter("file")  # or "s3", "azure", "gcs", "memory"
    >>> await storage.upload("sessions", "session_123/state.json", data)
    >>> state = await storage.download("sessions", "session_123/state.json")

"""

from __future__ import annotations

import typing as t
from contextlib import suppress
from pathlib import Path

from acb.adapters import import_adapter
from acb.config import Config
from acb.depends import depends

if t.TYPE_CHECKING:
    from acb.adapters.storage._base import StorageBase

# Supported storage backend types
SUPPORTED_BACKENDS = ("s3", "azure", "gcs", "file", "memory")


def register_storage_adapter(backend: str, config_overrides: dict[str, t.Any] | None = None, force: bool = False) -> StorageBase:
    """Register an ACB storage adapter with the given backend type.

    Args:
        backend: Storage backend type ("s3", "azure", "gcs", "file", "memory")
        config_overrides: Optional configuration overrides for the adapter
        force: If True, re-registers even if already registered

    Returns:
        Configured storage adapter instance

    Raises:
        ValueError: If backend type is not supported

    Example:
        >>> storage = register_storage_adapter("file", {"local_path": "/var/sessions"})
        >>> await storage.init()  # Initialize buckets

    """
    if backend not in SUPPORTED_BACKENDS:
        msg = f"Unsupported backend: {backend}. Must be one of {SUPPORTED_BACKENDS}"
        raise ValueError(msg)

    # Get Config singleton
    config = depends.get_sync(Config)
    config.ensure_initialized()

    # Import the appropriate storage adapter
    # ACB storage adapters auto-register as depends.set(Storage, "backend_name")
    storage_class = import_adapter("storage", backend)

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            existing = depends.get_sync(storage_class)
            if isinstance(existing, storage_class):
                return existing

    # Create adapter instance
    storage_adapter = storage_class()
    storage_adapter.config = config

    # Apply configuration overrides if provided
    if config_overrides:
        # Create or update storage settings
        if not hasattr(config, "storage"):
            from acb.adapters.storage._base import StorageBaseSettings

            config.storage = StorageBaseSettings()

        for key, value in config_overrides.items():
            setattr(config.storage, key, value)

    # Set logger from DI
    try:
        logger_class = import_adapter("logger")
        logger_instance = depends.get_sync(logger_class)
        storage_adapter.logger = logger_instance
    except Exception:
        import logging

        storage_adapter.logger = logging.getLogger(f"acb.storage.{backend}")

    # Register with DI container
    depends.set(storage_class, storage_adapter)

    return storage_adapter


def get_storage_adapter(backend: str | None = None) -> StorageBase:
    """Get a registered storage adapter by backend type.

    Args:
        backend: Storage backend type. If None, uses configured default.

    Returns:
        Storage adapter instance

    Raises:
        ValueError: If backend not registered or not supported
        KeyError: If adapter not found in DI container

    Example:
        >>> storage = get_storage_adapter("s3")
        >>> await storage.upload("sessions", "session_123/state.json", data)

    """
    # Get backend from config if not specified
    if backend is None:
        config = depends.get_sync(Config)
        backend = getattr(config.storage, "default_backend", "file")

    if backend not in SUPPORTED_BACKENDS:
        msg = f"Unsupported backend: {backend}. Must be one of {SUPPORTED_BACKENDS}"
        raise ValueError(msg)

    # Import and retrieve from DI
    storage_class = import_adapter("storage", backend)

    try:
        return depends.get_sync(storage_class)
    except KeyError as e:
        msg = f"Storage adapter '{backend}' not registered. Call register_storage_adapter() first."
        raise ValueError(msg) from e


def configure_storage_buckets(buckets: dict[str, str]) -> None:
    """Configure storage buckets for all registered adapters.

    Args:
        buckets: Mapping of bucket names to bucket identifiers
                 Example: {"sessions": "my-sessions-bucket", "test": "test-bucket"}

    Note:
        This should be called before initializing any storage adapters.
        Buckets are logical groupings used to organize stored files.

    Example:
        >>> configure_storage_buckets({
        ...     "sessions": "production-sessions",
        ...     "checkpoints": "session-checkpoints",
        ...     "test": "test-data"
        ... })

    """
    config = depends.get_sync(Config)
    config.ensure_initialized()

    if not hasattr(config, "storage"):
        from acb.adapters.storage._base import StorageBaseSettings

        config.storage = StorageBaseSettings()

    config.storage.buckets = buckets


def get_default_session_buckets(data_dir: Path) -> dict[str, str]:
    """Get default bucket configuration for session management.

    Args:
        data_dir: Base directory for session data storage

    Returns:
        Dictionary mapping bucket names to paths/identifiers

    Example:
        >>> buckets = get_default_session_buckets(Path.home() / ".claude" / "data")
        >>> configure_storage_buckets(buckets)

    """
    return {
        "sessions": str(data_dir / "sessions"),
        "checkpoints": str(data_dir / "checkpoints"),
        "handoffs": str(data_dir / "handoffs"),
        "test": str(data_dir / "test"),
    }


__all__ = [
    "SUPPORTED_BACKENDS",
    "register_storage_adapter",
    "get_storage_adapter",
    "configure_storage_buckets",
    "get_default_session_buckets",
]
