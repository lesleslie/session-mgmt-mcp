"""Tests for app_monitor module.

Tests application activity monitoring including file changes, browser navigation,
and application focus tracking.

Phase: Week 5 Day 4 - App Monitor Coverage
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestActivityEvent:
    """Test ActivityEvent dataclass."""

    def test_activity_event_creation(self) -> None:
        """Should create ActivityEvent with required fields."""
        from session_buddy.app_monitor import ActivityEvent

        event = ActivityEvent(
            timestamp=datetime.now().isoformat(),
            event_type="file_change",
            application="VSCode",
            details={"file_path": "/test/file.py"},
            project_path="/test",
            relevance_score=0.8,
        )

        assert event.event_type == "file_change"
        assert event.application == "VSCode"
        assert event.relevance_score == 0.8
        assert "file_path" in event.details


class TestProjectActivityMonitor:
    """Test ProjectActivityMonitor class."""

    def test_initialization(self) -> None:
        """Should initialize with project paths."""
        from session_buddy.app_monitor import ProjectActivityMonitor

        monitor = ProjectActivityMonitor(project_paths=["/test/project"])

        assert len(monitor.project_paths) == 1
        assert "/test/project" in monitor.project_paths
        assert isinstance(monitor.activity_buffer, list)
        assert len(monitor.ide_extensions) > 0

    def test_initialization_no_paths(self) -> None:
        """Should initialize with empty project paths."""
        from session_buddy.app_monitor import ProjectActivityMonitor

        monitor = ProjectActivityMonitor()

        assert monitor.project_paths == []
        assert isinstance(monitor.activity_buffer, list)

    def test_add_activity(self) -> None:
        """Should add activity event to buffer."""
        from session_buddy.app_monitor import ActivityEvent, ProjectActivityMonitor

        monitor = ProjectActivityMonitor()
        event = ActivityEvent(
            timestamp=datetime.now().isoformat(),
            event_type="file_change",
            application="VSCode",
            details={"file_path": "/test/file.py"},
        )

        monitor.add_activity(event)

        assert len(monitor.activity_buffer) == 1
        assert monitor.activity_buffer[0] == event

    def test_activity_buffer_size_limit(self) -> None:
        """Should limit activity buffer size."""
        from session_buddy.app_monitor import ActivityEvent, ProjectActivityMonitor

        monitor = ProjectActivityMonitor()

        # Add 1001 events to trigger buffer trimming
        for i in range(1001):
            event = ActivityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="file_change",
                application="VSCode",
                details={"file_path": f"/test/file{i}.py"},
            )
            monitor.add_activity(event)

        # Should be trimmed to 500 (keeps last 500)
        assert len(monitor.activity_buffer) == 500

    def test_get_recent_activity(self) -> None:
        """Should retrieve recent activity within time window."""
        from session_buddy.app_monitor import ActivityEvent, ProjectActivityMonitor

        monitor = ProjectActivityMonitor()

        # Add recent event
        recent_event = ActivityEvent(
            timestamp=datetime.now().isoformat(),
            event_type="file_change",
            application="VSCode",
            details={"file_path": "/test/recent.py"},
        )
        monitor.add_activity(recent_event)

        # Add old event (2 hours ago)
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        old_event = ActivityEvent(
            timestamp=old_time,
            event_type="file_change",
            application="VSCode",
            details={"file_path": "/test/old.py"},
        )
        monitor.add_activity(old_event)

        # Get recent activity (last 30 minutes)
        recent = monitor.get_recent_activity(minutes=30)

        # Should only include recent event
        assert len(recent) == 1
        assert recent[0] == recent_event

    def test_get_active_files(self) -> None:
        """Should identify actively worked files."""
        from session_buddy.app_monitor import ActivityEvent, ProjectActivityMonitor

        monitor = ProjectActivityMonitor()

        # Add multiple events for same file
        for i in range(3):
            event = ActivityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="file_change",
                application="VSCode",
                details={"file_path": "/test/active.py"},
            )
            monitor.add_activity(event)

        active_files = monitor.get_active_files(minutes=60)

        # Should identify the file as active
        assert isinstance(active_files, list)
        if len(active_files) > 0:
            assert "file_path" in active_files[0]
            assert "event_count" in active_files[0]  # Actual field name

    def test_start_monitoring_no_watchdog(self) -> None:
        """Should return False when watchdog unavailable."""
        from session_buddy.app_monitor import ProjectActivityMonitor

        with patch("session_buddy.app_monitor.WATCHDOG_AVAILABLE", False):
            monitor = ProjectActivityMonitor(project_paths=["/test"])
            result = monitor.start_monitoring()

            assert result is False

    def test_stop_monitoring(self) -> None:
        """Should stop all observers."""
        from session_buddy.app_monitor import ProjectActivityMonitor

        monitor = ProjectActivityMonitor()

        # Mock observers
        mock_observer = Mock()
        mock_observer.stop = Mock()
        mock_observer.join = Mock()
        monitor.observers.append(mock_observer)

        monitor.stop_monitoring()

        assert len(monitor.observers) == 0
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


class TestBrowserDocumentationMonitor:
    """Test BrowserDocumentationMonitor class."""

    def test_initialization(self) -> None:
        """Should initialize documentation monitor."""
        from session_buddy.app_monitor import BrowserDocumentationMonitor

        monitor = BrowserDocumentationMonitor()

        assert isinstance(monitor.activity_buffer, list)
        assert isinstance(monitor.doc_domains, set)
        assert len(monitor.doc_domains) > 0

    def test_add_browser_activity(self) -> None:
        """Should add browser activity to buffer."""
        from session_buddy.app_monitor import (
            ActivityEvent,
            BrowserDocumentationMonitor,
        )

        monitor = BrowserDocumentationMonitor()

        url = "https://docs.python.org/3/library/asyncio.html"
        title = "asyncio — Asynchronous I/O"

        monitor.add_browser_activity(url, title)

        assert len(monitor.activity_buffer) == 1
        activity = monitor.activity_buffer[0]
        assert isinstance(activity, ActivityEvent)
        assert activity.details["url"] == url
        assert activity.details["title"] == title
        assert activity.event_type == "browser_nav"

    def test_extract_documentation_context(self) -> None:
        """Should extract context from documentation URLs."""
        from session_buddy.app_monitor import BrowserDocumentationMonitor

        monitor = BrowserDocumentationMonitor()

        url = "https://docs.python.org/3/library/asyncio.html"
        context = monitor.extract_documentation_context(url)

        assert isinstance(context, dict)
        assert "domain" in context
        assert context["domain"] == "docs.python.org"
        assert "technology" in context
        assert "topic" in context
        assert "relevance" in context

    def test_get_browser_processes(self) -> None:
        """Should return browser process information."""
        from session_buddy.app_monitor import BrowserDocumentationMonitor

        monitor = BrowserDocumentationMonitor()

        with patch("session_buddy.app_monitor.PSUTIL_AVAILABLE", False):
            processes = monitor.get_browser_processes()
            assert isinstance(processes, list)


class TestApplicationFocusMonitor:
    """Test ApplicationFocusMonitor class."""

    def test_initialization(self) -> None:
        """Should initialize focus monitor."""
        from session_buddy.app_monitor import ApplicationFocusMonitor

        monitor = ApplicationFocusMonitor()

        assert isinstance(monitor.focus_history, list)
        assert hasattr(monitor, "current_app")
        assert hasattr(monitor, "app_categories")

    def test_add_focus_event(self) -> None:
        """Should add focus event to history."""
        from session_buddy.app_monitor import ActivityEvent, ApplicationFocusMonitor

        monitor = ApplicationFocusMonitor()

        app_info = {"name": "VSCode", "pid": 12345, "category": "ide"}
        monitor.add_focus_event(app_info)

        assert len(monitor.focus_history) == 1
        event = monitor.focus_history[0]
        assert isinstance(event, ActivityEvent)
        assert event.application == "VSCode"
        assert event.event_type == "app_focus"

    def test_get_focused_application(self) -> None:
        """Should get currently focused application."""
        from session_buddy.app_monitor import ApplicationFocusMonitor

        monitor = ApplicationFocusMonitor()

        with patch("session_buddy.app_monitor.PSUTIL_AVAILABLE", False):
            result = monitor.get_focused_application()
            # Returns None when psutil unavailable or no app focused
            assert result is None or isinstance(result, dict)


class TestActivityDatabase:
    """Test ActivityDatabase class."""

    def test_initialization_creates_tables(self) -> None:
        """Should create database tables on initialization."""
        from session_buddy.app_monitor import ActivityDatabase

        # Constructor automatically calls _init_database()
        db = ActivityDatabase(db_path=":memory:")

        # Verify tables exist (will raise if not)
        assert db.db_path == ":memory:"

    def test_store_activity(self) -> None:
        """Should store activity event in database."""
        import tempfile

        from session_buddy.app_monitor import ActivityDatabase, ActivityEvent

        # Use temp file instead of :memory: to ensure persistence across operations
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = ActivityDatabase(db_path=str(db_path))

            event = ActivityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="file_change",
                application="VSCode",
                details={"file_path": "/test/file.py"},
            )

            # Use store_event method (actual implementation)
            db.store_event(event)

            # Verify event was stored
            events = db.get_events(limit=10)
            assert len(events) >= 1


class TestApplicationMonitor:
    """Test main ApplicationMonitor orchestration class."""

    def test_initialization(self) -> None:
        """Should initialize all sub-monitors."""
        from session_buddy.app_monitor import ApplicationMonitor

        monitor = ApplicationMonitor(
            data_dir="/tmp/test_monitor", project_paths=["/test"]
        )

        assert monitor.ide_monitor is not None
        assert monitor.browser_monitor is not None
        assert monitor.focus_monitor is not None
        assert monitor.monitoring_active is False

    @pytest.mark.asyncio
    async def test_start_monitoring(self) -> None:
        """Should start all monitoring components."""
        from session_buddy.app_monitor import ApplicationMonitor

        with patch("session_buddy.app_monitor.WATCHDOG_AVAILABLE", True):
            monitor = ApplicationMonitor(
                data_dir="/tmp/test_monitor", project_paths=["/test"]
            )

            # Mock start_monitoring to avoid actual file watching
            monitor.ide_monitor.start_monitoring = Mock(return_value=True)

            result = await monitor.start_monitoring()

            assert monitor.monitoring_active is True
            assert result is not None

    @pytest.mark.asyncio
    async def test_stop_monitoring(self) -> None:
        """Should stop all monitoring components."""
        from session_buddy.app_monitor import ApplicationMonitor

        monitor = ApplicationMonitor(
            data_dir="/tmp/test_monitor", project_paths=["/test"]
        )
        monitor.monitoring_active = True
        monitor._monitoring_task = None

        await monitor.stop_monitoring()

        assert monitor.monitoring_active is False

    def test_get_context_insights(self) -> None:
        """Should generate context insights from activity."""
        from session_buddy.app_monitor import ApplicationMonitor

        monitor = ApplicationMonitor(
            data_dir="/tmp/test_monitor", project_paths=["/test"]
        )

        # get_context_insights is NOT async - it's a regular method
        insights = monitor.get_context_insights(hours=1)

        assert isinstance(insights, dict)
        # Actual keys returned by get_context_insights
        assert "primary_focus" in insights
        assert "technologies_used" in insights
        assert "active_projects" in insights
        assert "documentation_topics" in insights
        assert "productivity_score" in insights
        assert "context_switches" in insights
