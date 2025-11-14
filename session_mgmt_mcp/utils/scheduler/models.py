"""Data models for natural language scheduler.

This module provides data models for reminders, scheduling context,
and reminder types/statuses used throughout the scheduler system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReminderType(Enum):
    """Type of reminder."""

    TASK = "task"
    DEADLINE = "deadline"
    RECURRING = "recurring"
    SESSION_RELATIVE = "session_relative"


class ReminderStatus(Enum):
    """Status of a reminder."""

    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class NaturalReminder:
    """Represents a reminder created from natural language."""

    reminder_id: str
    reminder_type: ReminderType
    expression: str
    scheduled_time: datetime
    action: str
    status: ReminderStatus = ReminderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: datetime | None = None
    failure_reason: str | None = None
    recurrence_pattern: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingContext:
    """Context for scheduling operations."""

    session_start: datetime | None = None
    session_end: datetime | None = None
    current_task: str | None = None
    project: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
