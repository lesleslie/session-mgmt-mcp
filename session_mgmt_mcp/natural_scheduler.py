"""Natural Language Scheduling module for time-based reminders and triggers.

This module provides intelligent scheduling capabilities including:
- Natural language time parsing ("in 30 minutes", "tomorrow at 9am")
- Recurring reminders and cron-like scheduling
- Context-aware reminder triggers
- Integration with session workflow
"""

import asyncio
import contextlib
import importlib.util
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from re import Match
from typing import Any

DATEUTIL_AVAILABLE = importlib.util.find_spec("dateutil") is not None
CRONTAB_AVAILABLE = importlib.util.find_spec("python_crontab") is not None
SCHEDULE_AVAILABLE = importlib.util.find_spec("schedule") is not None

if DATEUTIL_AVAILABLE:
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class ReminderType(Enum):
    """Types of reminders."""

    ONE_TIME = "one_time"
    RECURRING = "recurring"
    CONTEXT_TRIGGER = "context_trigger"
    SESSION_MILESTONE = "session_milestone"


class ReminderStatus(Enum):
    """Reminder execution status."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class NaturalReminder:
    """Natural language reminder with scheduling information."""

    id: str
    title: str
    description: str
    reminder_type: ReminderType
    status: ReminderStatus
    created_at: datetime
    scheduled_for: datetime
    executed_at: datetime | None
    user_id: str
    project_id: str | None
    context_triggers: list[str]
    recurrence_rule: str | None
    notification_method: str
    metadata: dict[str, Any]


@dataclass
class SchedulingContext:
    """Context information for scheduling decisions."""

    current_time: datetime
    timezone: str
    user_preferences: dict[str, Any]
    active_project: str | None
    session_duration: int
    recent_activity: list[dict[str, Any]]


class NaturalLanguageParser:
    """Parses natural language time expressions."""

    def __init__(self) -> None:
        """Initialize natural language parser."""
        self.time_patterns = self._create_time_patterns()
        self.recurrence_patterns = self._create_recurrence_patterns()

    def _create_time_patterns(self) -> dict[str, Any]:
        """Create time parsing patterns dictionary."""
        patterns = {}

        # Add relative time patterns
        patterns.update(self._get_relative_time_patterns())

        # Add specific time patterns
        patterns.update(self._get_specific_time_patterns())

        # Add session-relative patterns
        patterns.update(self._get_session_relative_patterns())

        return patterns

    def _get_relative_time_patterns(self) -> dict[str, Any]:
        """Get relative time patterns (in X minutes/hours/days)."""
        return {
            r"in (\d+) (minute|min|minutes|mins)": lambda m: timedelta(
                minutes=int(m.group(1))
            ),
            r"in (\d+) (hour|hours|hr|hrs)": lambda m: timedelta(hours=int(m.group(1))),
            r"in (\d+) (day|days)": lambda m: timedelta(days=int(m.group(1))),
            r"in (\d+) (week|weeks)": lambda m: timedelta(weeks=int(m.group(1))),
            r"in (\d+) (month|months)": self._create_month_handler(),
        }

    def _get_specific_time_patterns(self) -> dict[str, Any]:
        """Get specific time patterns (tomorrow, next monday, etc)."""
        return {
            r"tomorrow( at (\d{1,2}):?(\d{2})?)?(am|pm)?": self._parse_tomorrow,
            r"next (monday|tuesday|wednesday|thursday|friday|saturday|sunday)": self._parse_next_weekday,
            r"at (\d{1,2}):?(\d{2})?\s*(am|pm)?": self._parse_specific_time,
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday) at (\d{1,2}):?(\d{2})?\s*(am|pm)?": self._parse_weekday_time,
        }

    def _get_session_relative_patterns(self) -> dict[str, Any]:
        """Get session-relative time patterns."""
        return {
            r"end of (session|work)": lambda m: timedelta(hours=2),
            r"after (break|lunch)": lambda m: timedelta(hours=1),
            r"before (meeting|call)": lambda m: timedelta(minutes=15),
        }

    def _create_month_handler(self) -> Any:
        """Create month duration handler with dateutil fallback."""
        if DATEUTIL_AVAILABLE:
            return lambda m: relativedelta(months=int(m.group(1)))
        return lambda m: timedelta(days=int(m.group(1)) * 30)

    def _create_recurrence_patterns(self) -> dict[str, Any]:
        """Create recurrence parsing patterns dictionary."""
        return {
            r"every (day|daily)": "FREQ=DAILY",
            r"every (week|weekly)": "FREQ=WEEKLY",
            r"every (month|monthly)": "FREQ=MONTHLY",
            r"every (\d+) (minute|minutes)": lambda m: f"FREQ=MINUTELY;INTERVAL={m.group(1)}",
            r"every (\d+) (hour|hours)": lambda m: f"FREQ=HOURLY;INTERVAL={m.group(1)}",
            r"every (\d+) (day|days)": lambda m: f"FREQ=DAILY;INTERVAL={m.group(1)}",
        }

    def _try_parse_relative_pattern(
        self, expression: str, base_time: datetime, time_patterns: dict[str, Any]
    ) -> datetime | None:
        """Try to parse the expression using relative time patterns."""
        for pattern, handler in time_patterns.items():
            match = self._try_pattern_match(pattern, expression)
            if match:
                result = self._process_pattern_handler(handler, match)
                if result:
                    return self._convert_result_to_datetime(result, base_time)
        return None

    def _try_pattern_match(self, pattern: str, expression: str) -> Match[str] | None:
        """Try to match a pattern against the expression."""
        return re.search(pattern, expression, re.IGNORECASE)  # REGEX OK: Time parsing

    def _process_pattern_handler(self, handler: Any, match: Match[str]) -> Any:
        """Process a pattern handler with exception handling."""
        with contextlib.suppress(Exception):
            if callable(handler):
                return handler(match)  # type: ignore[no-untyped-call]
        return None

    def _convert_result_to_datetime(
        self, result: Any, base_time: datetime
    ) -> datetime | None:
        """Convert handler result to datetime with base time."""
        if isinstance(result, timedelta):
            return base_time + result
        if isinstance(result, datetime):
            return result
        if hasattr(result, "days") or hasattr(result, "months"):
            return base_time + result  # type: ignore[no-any-return]
        return None

    def _try_parse_absolute_date(
        self, expression: str, base_time: datetime
    ) -> datetime | None:
        """Try to parse the expression using absolute date parsing."""
        if DATEUTIL_AVAILABLE:
            try:
                parsed_date = date_parser.parse(expression, default=base_time)
                if parsed_date > base_time:  # Only future dates
                    return datetime(
                        parsed_date.year,
                        parsed_date.month,
                        parsed_date.day,
                        parsed_date.hour,
                        parsed_date.minute,
                        parsed_date.second,
                    )  # type: ignore[no-any-return]
            except (ValueError, TypeError):
                with contextlib.suppress(ValueError, TypeError):
                    pass
        return None

    def _validate_input(self, expression: str) -> str | None:
        """Validate and normalize input expression."""
        if not expression or not expression.strip():
            return None
        return expression.lower().strip()

    def _try_parsing_strategies(
        self, expression: str, base_time: datetime
    ) -> datetime | None:
        """Try multiple parsing strategies in order."""
        # Strategy 1: Relative patterns
        result = self._try_parse_relative_pattern(
            expression, base_time, self.time_patterns
        )
        if result:
            return result

        # Strategy 2: Absolute date parsing
        result = self._try_parse_absolute_date(expression, base_time)
        if result:
            return result

        return None

    def parse_time_expression(
        self,
        expression: str,
        base_time: datetime | None = None,
    ) -> datetime | None:
        """Parse natural language time expression."""
        normalized_expression = self._validate_input(expression)
        if not normalized_expression:
            return None

        base_time = base_time or datetime.now()
        return self._try_parsing_strategies(normalized_expression, base_time)

    def parse_recurrence(self, expression: str) -> str | None:
        """Parse recurrence pattern from natural language."""
        if not expression:
            return None

        expression = expression.lower().strip()

        for pattern, handler in self.recurrence_patterns.items():
            match = re.search(
                pattern, expression, re.IGNORECASE
            )  # REGEX OK: Recurrence parsing
            if match:
                if callable(handler):
                    result = handler(match)
                    if isinstance(result, str):
                        return result
                elif isinstance(handler, str):
                    return handler

        return None

    def _parse_tomorrow(self, match: Match[str]) -> datetime:
        """Parse 'tomorrow' with optional time."""
        tomorrow = datetime.now() + timedelta(days=1)

        if match.group(2) and match.group(3):  # Has time
            hour = int(match.group(2))
            minute = int(match.group(3))
            am_pm = match.group(4)

            if am_pm and am_pm.lower() == "pm" and hour != 12:
                hour += 12
            elif am_pm and am_pm.lower() == "am" and hour == 12:
                hour = 0

            return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # Default to 9 AM tomorrow
        return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)

    def _parse_next_weekday(self, match: Match[str]) -> datetime:
        """Parse 'next monday', etc."""
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        target_weekday = weekdays[match.group(1)]
        today = datetime.now()
        days_ahead = target_weekday - today.weekday()

        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7

        return today + timedelta(days=days_ahead)

    def _parse_specific_time(self, match: Match[str]) -> datetime:
        """Parse 'at 3:30pm' for today."""
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        am_pm = match.group(3)

        if am_pm and am_pm.lower() == "pm" and hour != 12:
            hour += 12
        elif am_pm and am_pm.lower() == "am" and hour == 12:
            hour = 0

        target_time = datetime.now().replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )

        # If time has passed today, schedule for tomorrow
        if target_time <= datetime.now():
            target_time += timedelta(days=1)

        return target_time

    def _parse_weekday_time(self, match: Match[str]) -> datetime:
        """Parse 'monday at 3pm'."""
        target_weekday = self._get_weekday_number(match.group(1))
        hour, minute = self._parse_hour_minute(
            match.group(2), match.group(3), match.group(4)
        )

        today = datetime.now()
        days_ahead = self._calculate_days_ahead(target_weekday, today, hour, minute)

        target_date = today + timedelta(days=days_ahead)
        return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _get_weekday_number(self, weekday_name: str) -> int:
        """Get weekday number from name."""
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        return weekdays[weekday_name]

    def _parse_hour_minute(
        self, hour_str: str, minute_str: str | None, am_pm: str | None
    ) -> tuple[int, int]:
        """Parse hour and minute from time components."""
        hour = int(hour_str)
        minute = int(minute_str) if minute_str else 0

        if am_pm and am_pm.lower() == "pm" and hour != 12:
            hour += 12
        elif am_pm and am_pm.lower() == "am" and hour == 12:
            hour = 0

        return hour, minute

    def _calculate_days_ahead(
        self, target_weekday: int, today: datetime, hour: int, minute: int
    ) -> int:
        """Calculate how many days ahead the target weekday is."""
        days_ahead = target_weekday - today.weekday()

        if days_ahead < 0:  # Target day already happened this week
            days_ahead += 7
        elif days_ahead == 0:  # Today - check if time has passed
            target_time = today.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if target_time <= today:
                days_ahead = 7

        return days_ahead


class ReminderScheduler:
    """Manages scheduling and execution of reminders."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize reminder scheduler."""
        self.db_path = db_path or str(
            Path.home() / ".claude" / "data" / "natural_scheduler.db",
        )
        self.parser = NaturalLanguageParser()
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: threading.Thread | None = None
        self._callbacks: dict[str, list[Callable[..., Any]]] = {}
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for reminders."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    reminder_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP,
                    scheduled_for TIMESTAMP,
                    executed_at TIMESTAMP,
                    user_id TEXT NOT NULL,
                    project_id TEXT,
                    context_triggers TEXT,  -- JSON array
                    recurrence_rule TEXT,
                    notification_method TEXT,
                    metadata TEXT  -- JSON object
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminder_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reminder_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP,
                    result TEXT,
                    details TEXT  -- JSON object
                )
            """)

            # Create indices
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_scheduled ON reminders(scheduled_for)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_id)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_project ON reminders(project_id)",
            )

    async def create_reminder(
        self,
        title: str,
        time_expression: str,
        description: str = "",
        user_id: str = "default",
        project_id: str | None = None,
        notification_method: str = "session",
        context_triggers: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Create a new reminder from natural language."""
        # Parse the time expression
        scheduled_time = self.parser.parse_time_expression(time_expression)
        if not scheduled_time:
            return None

        # Check for recurrence
        recurrence_rule = self.parser.parse_recurrence(time_expression)
        reminder_type = (
            ReminderType.RECURRING if recurrence_rule else ReminderType.ONE_TIME
        )

        # Generate reminder ID
        reminder_id = f"rem_{int(time.time() * 1000)}"

        reminder = NaturalReminder(
            id=reminder_id,
            title=title,
            description=description,
            reminder_type=reminder_type,
            status=ReminderStatus.PENDING,
            created_at=datetime.now(),
            scheduled_for=scheduled_time,
            executed_at=None,
            user_id=user_id,
            project_id=project_id,
            context_triggers=context_triggers or [],
            recurrence_rule=recurrence_rule,
            notification_method=notification_method,
            metadata=metadata or {},
        )

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO reminders (id, title, description, reminder_type, status, created_at,
                                     scheduled_for, executed_at, user_id, project_id, context_triggers,
                                     recurrence_rule, notification_method, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    reminder.id,
                    reminder.title,
                    reminder.description,
                    reminder.reminder_type.value,
                    reminder.status.value,
                    reminder.created_at,
                    reminder.scheduled_for,
                    reminder.executed_at,
                    reminder.user_id,
                    reminder.project_id,
                    json.dumps(reminder.context_triggers),
                    reminder.recurrence_rule,
                    reminder.notification_method,
                    json.dumps(reminder.metadata),
                ),
            )

        # Log creation
        await self._log_reminder_action(
            reminder_id,
            "created",
            "success",
            {
                "scheduled_for": scheduled_time.isoformat(),
                "time_expression": time_expression,
            },
        )

        return reminder_id

    async def get_pending_reminders(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get pending reminders."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            where_conditions = ["status IN ('pending', 'active')"]
            params = []

            if user_id:
                where_conditions.append("user_id = ?")
                params.append(user_id)

            if project_id:
                where_conditions.append("project_id = ?")
                params.append(project_id)

            query = f"SELECT * FROM reminders WHERE {' AND '.join(where_conditions)} ORDER BY scheduled_for"  # nosec B608 - where_conditions built from validated parameters with proper placeholders

            cursor = conn.execute(query, params)
            results = []

            for row in cursor.fetchall():
                result = dict(row)
                result["context_triggers"] = json.loads(
                    result["context_triggers"] or "[]",
                )
                result["metadata"] = json.loads(result["metadata"] or "{}")
                results.append(result)

            return results

    async def get_due_reminders(
        self,
        check_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get reminders that are due for execution."""
        check_time = check_time or datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT * FROM reminders
                WHERE status = 'pending' AND scheduled_for <= ?
                ORDER BY scheduled_for
            """,
                (check_time,),
            )

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result["context_triggers"] = json.loads(
                    result["context_triggers"] or "[]",
                )
                result["metadata"] = json.loads(result["metadata"] or "{}")
                results.append(result)

            return results

    async def execute_reminder(self, reminder_id: str) -> bool:
        """Execute a due reminder."""
        try:
            # Get reminder details
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM reminders WHERE id = ?",
                    (reminder_id,),
                ).fetchone()

                if not row:
                    return False

                reminder_data = dict(row)
                reminder_data["context_triggers"] = json.loads(
                    reminder_data["context_triggers"] or "[]",
                )
                reminder_data["metadata"] = json.loads(
                    reminder_data["metadata"] or "{}",
                )

            # Execute callbacks
            callbacks = self._callbacks.get(reminder_data["notification_method"], [])
            for callback in callbacks:
                try:
                    await callback(reminder_data)
                except Exception as e:
                    logger.exception(f"Callback error for reminder {reminder_id}: {e}")

            # Update status
            now = datetime.now()
            new_status = ReminderStatus.COMPLETED

            # Handle recurring reminders
            if reminder_data["recurrence_rule"]:
                # Schedule next occurrence
                next_time = self._calculate_next_occurrence(
                    reminder_data["scheduled_for"],
                    reminder_data["recurrence_rule"],
                )
                if next_time:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            """
                            UPDATE reminders
                            SET scheduled_for = ?, status = 'pending', executed_at = NULL
                            WHERE id = ?
                        """,
                            (next_time, reminder_id),
                        )

                    await self._log_reminder_action(
                        reminder_id,
                        "rescheduled",
                        "success",
                        {"next_occurrence": next_time.isoformat()},
                    )
                    return True

            # Mark as completed
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE reminders
                    SET status = ?, executed_at = ?
                    WHERE id = ?
                """,
                    (new_status.value, now, reminder_id),
                )

            await self._log_reminder_action(
                reminder_id,
                "executed",
                "success",
                {"executed_at": now.isoformat()},
            )

            return True

        except Exception as e:
            await self._log_reminder_action(
                reminder_id,
                "executed",
                "failed",
                {"error": str(e)},
            )
            return False

    async def cancel_reminder(self, reminder_id: str) -> bool:
        """Cancel a pending reminder."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    """
                    UPDATE reminders
                    SET status = ?
                    WHERE id = ? AND status IN ('pending', 'active')
                """,
                    (ReminderStatus.CANCELLED.value, reminder_id),
                )

                success = result.rowcount > 0

            if success:
                await self._log_reminder_action(reminder_id, "cancelled", "success", {})

            return success

        except Exception as e:
            await self._log_reminder_action(
                reminder_id,
                "cancelled",
                "failed",
                {"error": str(e)},
            )
            return False

    def register_notification_callback(
        self, method: str, callback: Callable[..., Any]
    ) -> None:
        """Register callback for notification method."""
        if method not in self._callbacks:
            self._callbacks[method] = []
        self._callbacks[method].append(callback)

    def start_scheduler(self) -> None:
        """Start the background scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
        )
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)

    def _scheduler_loop(self) -> None:
        """Background scheduler loop."""
        while self._running:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._check_and_execute_reminders())
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
            finally:
                if loop and not loop.is_closed():
                    loop.close()
                time.sleep(60)  # Check every minute

    async def _check_and_execute_reminders(self) -> None:
        """Check for due reminders and execute them."""
        due_reminders = await self.get_due_reminders()

        for reminder in due_reminders:
            await self.execute_reminder(reminder["id"])

    def _parse_recurrence_interval(
        self, recurrence_rule: str
    ) -> tuple[str | None, int]:
        """Parse frequency and interval from recurrence rule."""
        parts = recurrence_rule.split(";")
        interval = 1
        freq = None

        for part in parts:
            if part.startswith("FREQ="):
                freq = part.split("=")[1]
            elif part.startswith("INTERVAL="):
                interval = int(part.split("=")[1])

        return freq, interval

    def _calculate_simple_occurrence(
        self, last_time: datetime, recurrence_rule: str
    ) -> datetime | None:
        """Calculate simple recurrence occurrences (daily, weekly, monthly)."""
        if recurrence_rule.startswith("FREQ=DAILY"):
            return last_time + timedelta(days=1)  # type: ignore[no-any-return]
        if recurrence_rule.startswith("FREQ=WEEKLY"):
            return last_time + timedelta(weeks=1)  # type: ignore[no-any-return]
        if recurrence_rule.startswith("FREQ=MONTHLY"):
            return last_time + relativedelta(months=1)  # type: ignore[no-any-return]
        return None

    def _calculate_interval_occurrence(
        self, last_time: datetime, recurrence_rule: str
    ) -> datetime | None:
        """Calculate interval-based recurrence occurrences."""
        if "INTERVAL=" in recurrence_rule:
            freq, interval = self._parse_recurrence_interval(recurrence_rule)

            if freq == "HOURLY":
                return last_time + timedelta(hours=interval)  # type: ignore[no-any-return]
            if freq == "MINUTELY":
                return last_time + timedelta(minutes=interval)  # type: ignore[no-any-return]
            if freq == "DAILY":
                return last_time + timedelta(days=interval)  # type: ignore[no-any-return]
        return None

    def _check_dateutil_availability(self) -> bool:
        """Check if dateutil is available for processing."""
        return DATEUTIL_AVAILABLE

    def _attempt_simple_calculation(
        self, last_time: datetime, recurrence_rule: str
    ) -> datetime | None:
        """Attempt to calculate using simple occurrence rules."""
        try:
            return self._calculate_simple_occurrence(last_time, recurrence_rule)
        except Exception:
            return None

    def _attempt_interval_calculation(
        self, last_time: datetime, recurrence_rule: str
    ) -> datetime | None:
        """Attempt to calculate using interval occurrence rules."""
        try:
            return self._calculate_interval_occurrence(last_time, recurrence_rule)
        except Exception:
            return None

    def _calculate_next_occurrence(
        self,
        last_time: datetime,
        recurrence_rule: str,
    ) -> datetime | None:
        """Calculate next occurrence for recurring reminder."""
        if not DATEUTIL_AVAILABLE:
            return None

        try:
            # Try simple rule parsing first
            result = self._calculate_simple_occurrence(last_time, recurrence_rule)
            if result:
                return result

            # Try interval-based recurrence rules
            result = self._calculate_interval_occurrence(last_time, recurrence_rule)
            if result:
                return result

        except Exception as e:
            logger.exception(f"Error calculating next occurrence: {e}")

        return None

    async def _log_reminder_action(
        self,
        reminder_id: str,
        action: str,
        result: str,
        details: dict[str, Any],
    ) -> None:
        """Log reminder action for audit trail."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO reminder_history (reminder_id, action, timestamp, result, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (reminder_id, action, datetime.now(), result, json.dumps(details)),
            )


# Global scheduler instance
_reminder_scheduler = None


def get_reminder_scheduler() -> "ReminderScheduler":
    """Get global reminder scheduler instance."""
    global _reminder_scheduler
    if _reminder_scheduler is None:
        _reminder_scheduler = ReminderScheduler()
    return _reminder_scheduler


# Public API functions for MCP tools
async def create_natural_reminder(
    title: str,
    time_expression: str,
    description: str = "",
    user_id: str = "default",
    project_id: str | None = None,
    notification_method: str = "session",
) -> str | None:
    """Create reminder from natural language time expression."""
    scheduler = get_reminder_scheduler()
    return await scheduler.create_reminder(
        title,
        time_expression,
        description,
        user_id,
        project_id,
        notification_method,
    )


async def list_user_reminders(
    user_id: str = "default",
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """List pending reminders for user/project."""
    scheduler = get_reminder_scheduler()
    return await scheduler.get_pending_reminders(user_id, project_id)


async def cancel_user_reminder(reminder_id: str) -> bool:
    """Cancel a specific reminder."""
    scheduler = get_reminder_scheduler()
    return await scheduler.cancel_reminder(reminder_id)


async def check_due_reminders() -> list[dict[str, Any]]:
    """Check for reminders that are due now."""
    scheduler = get_reminder_scheduler()
    return await scheduler.get_due_reminders()


def start_reminder_service() -> None:
    """Start the background reminder service."""
    scheduler = get_reminder_scheduler()
    scheduler.start_scheduler()


def stop_reminder_service() -> None:
    """Stop the background reminder service."""
    scheduler = get_reminder_scheduler()
    scheduler.stop_scheduler()


def register_session_notifications() -> None:
    """Register session-based notification callbacks."""
    scheduler = get_reminder_scheduler()

    async def session_notification(reminder_data: dict[str, Any]) -> None:
        """Default session notification handler."""
        logger.info(
            f"Reminder: {reminder_data['title']} - {reminder_data['description']}",
        )

    scheduler.register_notification_callback("session", session_notification)
