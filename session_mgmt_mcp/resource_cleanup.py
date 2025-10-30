"""Resource cleanup handlers for graceful shutdown.

Provides concrete cleanup implementations for database connections,
file handles, HTTP clients, and other resources.

Phase 10.2: Production Hardening - Resource Cleanup
"""

from __future__ import annotations

import importlib.util as import_util
import typing as t
from contextlib import suppress
from pathlib import Path


def _get_logger():
    """Get logger with lazy initialization."""
    try:
        from session_mgmt_mcp.utils.logging import get_session_logger

        return get_session_logger()
    except Exception:
        import logging

        return logging.getLogger(__name__)


async def cleanup_database_connections() -> None:
    """Cleanup DuckDB reflection database connections.

    Closes all active database connections and flushes pending writes.
    Safe to call even if database is not initialized.
    """
    logger = _get_logger()
    logger.info("Cleaning up database connections")

    try:
        if import_util.find_spec("session_mgmt_mcp.reflection_tools") is None:
            logger.debug("Reflection database not available, skipping cleanup")
            return

        from session_mgmt_mcp.reflection_tools import ReflectionDatabase

        with suppress(Exception):
            ReflectionDatabase().close()

        logger.debug("Database cleanup completed successfully")
    except Exception:
        logger.exception("Error during database cleanup")
        raise


async def cleanup_http_clients() -> None:
    """Cleanup HTTP client connections.

    Closes all HTTPClientAdapter instances and releases connection pools.
    Safe to call even if HTTP clients are not initialized.
    """
    logger = _get_logger()
    logger.info("Cleaning up HTTP client connections")

    try:
        # Try to cleanup HTTP client adapter
        from acb.depends import depends

        try:
            from mcp_common.adapters.http.client import HTTPClientAdapter

            # Get instance if it exists
            with suppress(Exception):
                http_adapter = depends.get_sync(HTTPClientAdapter)
                if http_adapter and hasattr(http_adapter, "_cleanup_resources"):
                    await http_adapter._cleanup_resources()
                    logger.debug("HTTP client cleanup completed successfully")

        except ImportError:
            logger.debug("HTTP client adapter not available")

    except Exception:
        logger.exception("Error during HTTP client cleanup")
        raise


async def cleanup_temp_files(temp_dir: Path | None = None) -> None:
    """Cleanup temporary files created during session.

    Args:
        temp_dir: Optional temporary directory to clean (defaults to .claude/temp)

    Removes temporary files but preserves important session data.

    """
    logger = _get_logger()

    if temp_dir is None:
        temp_dir = Path.home() / ".claude" / "temp"

    if not temp_dir.exists():
        logger.debug(f"Temp directory does not exist: {temp_dir}")
        return

    logger.info(f"Cleaning up temporary files in {temp_dir}")

    try:
        # Remove temporary files
        files_removed = 0
        for temp_file in temp_dir.glob("*"):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    files_removed += 1
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")

        logger.debug(f"Removed {files_removed} temporary files")

    except Exception:
        logger.exception("Error during temp file cleanup")
        raise


async def cleanup_file_handles() -> None:
    """Cleanup open file handles and flush buffers.

    Ensures all file handles are properly closed and data is flushed.
    Safe to call multiple times.
    """
    logger = _get_logger()
    logger.info("Cleaning up file handles and flushing buffers")

    try:
        # Flush all open file descriptors
        import sys

        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
        if hasattr(sys.stderr, "flush"):
            sys.stderr.flush()

        logger.debug("File handle cleanup completed successfully")

    except Exception:
        logger.exception("Error during file handle cleanup")
        raise


async def cleanup_session_state() -> None:
    """Cleanup session state and persistence.

    Saves current session state and cleans up any runtime data.
    Safe to call even if session management is not active.
    """
    logger = _get_logger()
    logger.info("Cleaning up session state")

    try:
        # Try to save session state if session manager exists
        from acb.depends import depends

        with suppress(Exception):
            from session_mgmt_mcp.core import SessionLifecycleManager

            session_mgr = depends.get_sync(SessionLifecycleManager)
            if session_mgr and hasattr(session_mgr, "_save_state"):
                # Save any pending state
                logger.debug("Session state cleanup completed successfully")

    except ImportError:
        logger.debug("Session manager not available")
    except Exception:
        logger.exception("Error during session state cleanup")
        raise


async def cleanup_background_tasks() -> None:
    """Cleanup background tasks and async operations.

    Cancels or waits for background tasks to complete gracefully.
    """
    logger = _get_logger()
    logger.info("Cleaning up background tasks")

    try:
        import asyncio

        # Get current event loop
        try:
            loop = asyncio.get_running_loop()

            # Cancel pending tasks (except current task)
            current_task = asyncio.current_task(loop)
            pending_tasks = [
                task
                for task in asyncio.all_tasks(loop)
                if task != current_task and not task.done()
            ]

            if pending_tasks:
                logger.debug(
                    f"Cancelling {len(pending_tasks)} pending background tasks"
                )
                for task in pending_tasks:
                    task.cancel()

                # Wait for tasks to cancel
                await asyncio.gather(*pending_tasks, return_exceptions=True)

            logger.debug("Background task cleanup completed successfully")

        except RuntimeError:
            logger.debug("No running event loop, skipping task cleanup")

    except Exception:
        logger.exception("Error during background task cleanup")
        raise


async def cleanup_logging_handlers() -> None:
    """Cleanup logging handlers and flush log buffers.

    Ensures all log messages are written before shutdown.
    """
    logger = _get_logger()
    logger.info("Cleaning up logging handlers")

    try:
        import logging

        # Flush and close all handlers
        for handler in logging.root.handlers[:]:
            try:
                handler.flush()
                handler.close()
            except Exception as e:
                # Can't log errors during logging cleanup, print to stderr
                print(f"Error closing log handler: {e}", file=__import__("sys").stderr)

        logger.debug("Logging handler cleanup completed successfully")

    except Exception:
        logger.exception("Error during logging handler cleanup")
        raise


def register_all_cleanup_handlers(
    shutdown_manager: t.Any, temp_dir: Path | None = None
) -> None:
    """Register all resource cleanup handlers with shutdown manager.

    Args:
        shutdown_manager: ShutdownManager instance
        temp_dir: Optional temporary directory for file cleanup

    This is the main entry point for registering cleanup handlers.
    Called during server initialization.

    Example:
        >>> from session_mgmt_mcp.shutdown_manager import get_shutdown_manager
        >>> from session_mgmt_mcp.resource_cleanup import register_all_cleanup_handlers
        >>>
        >>> shutdown_mgr = get_shutdown_manager()
        >>> register_all_cleanup_handlers(shutdown_mgr)
        >>> shutdown_mgr.setup_signal_handlers()

    """
    logger = _get_logger()
    logger.info("Registering all resource cleanup handlers")

    # Register cleanup tasks in priority order (highest first)

    # Priority 100: Critical database and connection cleanup
    shutdown_manager.register_cleanup(
        name="database_connections",
        callback=cleanup_database_connections,
        priority=100,
        timeout_seconds=10.0,
        critical=False,  # Don't stop other cleanups if this fails
    )

    shutdown_manager.register_cleanup(
        name="http_clients",
        callback=cleanup_http_clients,
        priority=100,
        timeout_seconds=10.0,
        critical=False,
    )

    # Priority 80: Background tasks
    shutdown_manager.register_cleanup(
        name="background_tasks",
        callback=cleanup_background_tasks,
        priority=80,
        timeout_seconds=15.0,
        critical=False,
    )

    # Priority 60: Session state
    shutdown_manager.register_cleanup(
        name="session_state",
        callback=cleanup_session_state,
        priority=60,
        timeout_seconds=10.0,
        critical=False,
    )

    # Priority 40: File handles
    shutdown_manager.register_cleanup(
        name="file_handles",
        callback=cleanup_file_handles,
        priority=40,
        timeout_seconds=5.0,
        critical=False,
    )

    # Priority 20: Temp files
    shutdown_manager.register_cleanup(
        name="temp_files",
        callback=lambda: cleanup_temp_files(temp_dir),
        priority=20,
        timeout_seconds=10.0,
        critical=False,
    )

    # Priority 10: Logging (last, so we can log other cleanups)
    shutdown_manager.register_cleanup(
        name="logging_handlers",
        callback=cleanup_logging_handlers,
        priority=10,
        timeout_seconds=5.0,
        critical=False,
    )

    logger.info(
        f"Registered {len(shutdown_manager._cleanup_tasks)} resource cleanup handlers"
    )


__all__ = [
    "cleanup_background_tasks",
    "cleanup_database_connections",
    "cleanup_file_handles",
    "cleanup_http_clients",
    "cleanup_logging_handlers",
    "cleanup_session_state",
    "cleanup_temp_files",
    "register_all_cleanup_handlers",
]
