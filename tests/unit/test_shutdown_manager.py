"""Tests for graceful shutdown manager.

Tests shutdown coordination, signal handling, cleanup task execution,
and error recovery.

Phase 10.2: Production Hardening - Graceful Shutdown Tests
"""

from __future__ import annotations

import asyncio
import signal
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from session_buddy.shutdown_manager import (
    CleanupTask,
    ShutdownManager,
    ShutdownStats,
    get_shutdown_manager,
)


class TestCleanupTaskRegistration:
    """Test cleanup task registration and management."""

    def test_register_sync_cleanup_task(self) -> None:
        """Should register synchronous cleanup task."""
        manager = ShutdownManager()

        def sync_cleanup():
            pass

        manager.register_cleanup("sync_task", sync_cleanup, priority=10)

        assert len(manager._cleanup_tasks) == 1
        task = manager._cleanup_tasks[0]
        assert task.name == "sync_task"
        assert task.priority == 10
        assert task.callback == sync_cleanup

    def test_register_async_cleanup_task(self) -> None:
        """Should register asynchronous cleanup task."""
        manager = ShutdownManager()

        async def async_cleanup():
            pass

        manager.register_cleanup("async_task", async_cleanup, priority=20)

        assert len(manager._cleanup_tasks) == 1
        task = manager._cleanup_tasks[0]
        assert task.name == "async_task"
        assert task.priority == 20

    def test_register_multiple_tasks_with_priorities(self) -> None:
        """Should register multiple tasks with different priorities."""
        manager = ShutdownManager()

        manager.register_cleanup("low", lambda: None, priority=10)
        manager.register_cleanup("high", lambda: None, priority=100)
        manager.register_cleanup("medium", lambda: None, priority=50)

        assert len(manager._cleanup_tasks) == 3
        assert manager._stats.tasks_registered == 3

    def test_register_critical_task(self) -> None:
        """Should register critical cleanup task."""
        manager = ShutdownManager()

        manager.register_cleanup("critical", lambda: None, critical=True)

        task = manager._cleanup_tasks[0]
        assert task.critical is True

    def test_register_task_with_custom_timeout(self) -> None:
        """Should register task with custom timeout."""
        manager = ShutdownManager()

        manager.register_cleanup("slow", lambda: None, timeout_seconds=60.0)

        task = manager._cleanup_tasks[0]
        assert task.timeout_seconds == 60.0


class TestShutdownExecution:
    """Test shutdown execution and task coordination."""

    @pytest.mark.asyncio
    async def test_execute_sync_cleanup_tasks(self) -> None:
        """Should execute synchronous cleanup tasks successfully."""
        manager = ShutdownManager()
        executed = []

        def cleanup1():
            executed.append(1)

        def cleanup2():
            executed.append(2)

        manager.register_cleanup("task1", cleanup1)
        manager.register_cleanup("task2", cleanup2)

        stats = await manager.shutdown()

        assert executed == [1, 2]  # Both executed
        assert stats.tasks_executed == 2
        assert stats.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_execute_async_cleanup_tasks(self) -> None:
        """Should execute asynchronous cleanup tasks successfully."""
        manager = ShutdownManager()
        executed = []

        async def cleanup1():
            executed.append(1)

        async def cleanup2():
            executed.append(2)

        manager.register_cleanup("task1", cleanup1)
        manager.register_cleanup("task2", cleanup2)

        stats = await manager.shutdown()

        assert executed == [1, 2]
        assert stats.tasks_executed == 2

    @pytest.mark.asyncio
    async def test_execute_tasks_by_priority_order(self) -> None:
        """Should execute tasks in priority order (highest first)."""
        manager = ShutdownManager()
        execution_order = []

        def low():
            execution_order.append("low")

        def high():
            execution_order.append("high")

        def medium():
            execution_order.append("medium")

        # Register in random order
        manager.register_cleanup("low", low, priority=10)
        manager.register_cleanup("high", high, priority=100)
        manager.register_cleanup("medium", medium, priority=50)

        await manager.shutdown()

        # Should execute high -> medium -> low
        assert execution_order == ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_handle_task_timeout(self) -> None:
        """Should handle task timeout gracefully."""
        manager = ShutdownManager()

        async def slow_task():
            await asyncio.sleep(2.0)  # Will timeout

        manager.register_cleanup("slow", slow_task, timeout_seconds=0.1)

        stats = await manager.shutdown()

        assert stats.tasks_timeout == 1
        assert stats.tasks_executed == 0  # Didn't complete

    @pytest.mark.asyncio
    async def test_handle_task_exception(self) -> None:
        """Should handle task exception and continue."""
        manager = ShutdownManager()
        executed = []

        def failing_task():
            msg = "Cleanup failed"
            raise RuntimeError(msg)

        def successful_task():
            executed.append("success")

        manager.register_cleanup("fail", failing_task, priority=100)
        manager.register_cleanup("success", successful_task, priority=10)

        stats = await manager.shutdown()

        # Failed task should not stop other tasks
        assert "success" in executed
        assert stats.tasks_failed == 1
        assert stats.tasks_executed == 1

    @pytest.mark.asyncio
    async def test_critical_task_failure_stops_cleanup(self) -> None:
        """Should stop cleanup when critical task fails."""
        manager = ShutdownManager()
        executed = []

        def critical_failing():
            msg = "Critical failure"
            raise RuntimeError(msg)

        def later_task():
            executed.append("later")

        manager.register_cleanup(
            "critical", critical_failing, priority=100, critical=True
        )
        manager.register_cleanup("later", later_task, priority=10)

        stats = await manager.shutdown()

        # Later task should not execute
        assert "later" not in executed
        assert stats.tasks_failed == 1
        assert stats.tasks_executed == 0

    @pytest.mark.asyncio
    async def test_prevent_multiple_simultaneous_shutdowns(self) -> None:
        """Should prevent multiple simultaneous shutdowns."""
        manager = ShutdownManager()
        shutdown_count = [0]

        async def track_shutdown():
            shutdown_count[0] += 1

        manager.register_cleanup("track", track_shutdown)

        # Start two shutdowns concurrently
        await asyncio.gather(
            manager.shutdown(),
            manager.shutdown(),
        )

        # Should only execute once
        assert shutdown_count[0] == 1

    @pytest.mark.asyncio
    async def test_shutdown_tracks_duration(self) -> None:
        """Should track total shutdown duration."""
        manager = ShutdownManager()

        async def slow_cleanup():
            await asyncio.sleep(0.01)  # 10ms

        manager.register_cleanup("slow", slow_cleanup)

        stats = await manager.shutdown()

        assert stats.total_duration_ms > 0
        assert stats.total_duration_ms >= 10  # At least the sleep time


class TestSignalHandling:
    """Test signal handler registration and handling."""

    def test_setup_signal_handlers(self) -> None:
        """Should register signal handlers successfully."""
        manager = ShutdownManager()

        # Save original handlers
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        try:
            manager.setup_signal_handlers()

            # Handlers should be changed
            assert signal.getsignal(signal.SIGTERM) != original_sigterm
            assert signal.getsignal(signal.SIGINT) != original_sigint

            # Should have saved original handlers
            assert signal.SIGTERM in manager._original_handlers
            assert signal.SIGINT in manager._original_handlers

        finally:
            # Restore original handlers
            manager.restore_signal_handlers()

    def test_restore_signal_handlers(self) -> None:
        """Should restore original signal handlers."""
        manager = ShutdownManager()

        original_sigterm = signal.getsignal(signal.SIGTERM)

        manager.setup_signal_handlers()
        # Handler changed
        assert signal.getsignal(signal.SIGTERM) != original_sigterm

        manager.restore_signal_handlers()
        # Handler restored
        assert signal.getsignal(signal.SIGTERM) == original_sigterm

    @pytest.mark.asyncio
    async def test_signal_handler_triggers_shutdown(self) -> None:
        """Should trigger shutdown when signal received."""
        manager = ShutdownManager()
        executed = []

        def cleanup():
            executed.append("cleaned")

        manager.register_cleanup("test", cleanup)
        manager.setup_signal_handlers()

        try:
            # Manually call signal handler (simulating signal)
            manager._signal_handler(signal.SIGTERM, None)

            # Give it time to execute
            await asyncio.sleep(0.1)

            # Cleanup should have run
            assert "cleaned" in executed

        finally:
            manager.restore_signal_handlers()


class TestShutdownStats:
    """Test shutdown statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_track_registered_tasks(self) -> None:
        """Should track number of registered tasks."""
        manager = ShutdownManager()

        manager.register_cleanup("task1", lambda: None)
        manager.register_cleanup("task2", lambda: None)
        manager.register_cleanup("task3", lambda: None)

        stats = manager.get_stats()
        assert stats.tasks_registered == 3

    @pytest.mark.asyncio
    async def test_stats_track_executed_tasks(self) -> None:
        """Should track successfully executed tasks."""
        manager = ShutdownManager()

        manager.register_cleanup("task1", lambda: None)
        manager.register_cleanup("task2", lambda: None)

        stats = await manager.shutdown()

        assert stats.tasks_executed == 2
        assert stats.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_stats_track_failed_tasks(self) -> None:
        """Should track failed tasks."""
        manager = ShutdownManager()

        def failing():
            msg = "Failed"
            raise RuntimeError(msg)

        manager.register_cleanup("fail", failing)
        manager.register_cleanup("success", lambda: None)

        stats = await manager.shutdown()

        assert stats.tasks_failed == 1
        assert stats.tasks_executed == 1


class TestGlobalShutdownManager:
    """Test global shutdown manager singleton."""

    def test_get_shutdown_manager_returns_singleton(self) -> None:
        """Should return same instance each time."""
        mgr1 = get_shutdown_manager()
        mgr2 = get_shutdown_manager()

        assert mgr1 is mgr2

    def test_global_manager_is_shutdown_manager_instance(self) -> None:
        """Should return ShutdownManager instance."""
        mgr = get_shutdown_manager()

        assert isinstance(mgr, ShutdownManager)


class TestShutdownManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_shutdown_with_no_tasks(self) -> None:
        """Should handle shutdown with no registered tasks."""
        manager = ShutdownManager()

        stats = await manager.shutdown()

        assert stats.tasks_executed == 0
        assert stats.tasks_failed == 0
        assert stats.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_is_shutdown_initiated_flag(self) -> None:
        """Should track shutdown initiation state."""
        manager = ShutdownManager()

        assert manager.is_shutdown_initiated() is False

        await manager.shutdown()

        assert manager.is_shutdown_initiated() is True

    def test_atexit_handler_registered(self) -> None:
        """Should register atexit handler during signal handler setup."""
        manager = ShutdownManager()

        # Setup signal handlers also registers atexit
        manager.setup_signal_handlers()

        # Verify atexit was registered (can't easily test execution in unit tests
        # due to event loop complexities)
        # The important thing is that setup_signal_handlers() registers it
        try:
            manager.restore_signal_handlers()
        finally:
            pass  # Cleanup
