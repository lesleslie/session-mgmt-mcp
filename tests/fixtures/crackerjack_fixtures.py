"""Test fixtures for Crackerjack integration testing.

Week 8 Day 2 - Phase 2: Mock crackerjack command output and quality metrics.
Provides realistic crackerjack output data for testing without execution.
"""

from __future__ import annotations

import typing as t
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_crackerjack_output_success() -> str:
    """Create mock successful crackerjack command output.

    Returns:
        Realistic crackerjack output string with quality metrics.

    """
    return """
╭─────────────────────────────────────────────────────────────╮
│                     Crackerjack Quality Report                │
╰─────────────────────────────────────────────────────────────╯

📊 Quality Score: 85/100

✅ Tests: 980 passed, 0 failed, 20 skipped
📈 Coverage: 14.4%
🔍 Linting: 0 errors, 5 warnings
🔒 Security: 0 vulnerabilities
⚡ Performance: No issues detected

Recommendations:
  • Increase test coverage to ≥80%
  • Fix remaining lint warnings
  • Consider adding integration tests

Duration: 42.3s
"""


@pytest.fixture
def mock_crackerjack_output_failures() -> str:
    """Create mock crackerjack output with failures.

    Returns:
        Realistic crackerjack output string showing test failures.

    """
    return """
╭─────────────────────────────────────────────────────────────╮
│                     Crackerjack Quality Report                │
╰─────────────────────────────────────────────────────────────╯

📊 Quality Score: 45/100

❌ Tests: 945 passed, 8 failed, 20 skipped
📈 Coverage: 14.4%
🔍 Linting: 12 errors, 25 warnings
🔒 Security: 2 vulnerabilities (low severity)
⚡ Performance: 3 slow tests detected

Failing Tests:
  • tests/unit/test_di_container.py::test_configure_registers_singletons
  • tests/unit/test_instance_managers.py::test_get_app_monitor_registers_singleton
  • tests/unit/test_server.py::TestServerQualityScoring::test_calculate_quality_score_with_no_args

Recommendations:
  • Fix failing tests immediately
  • Address security vulnerabilities
  • Refactor complex functions
  • Improve test coverage

Duration: 45.7s
"""


@pytest.fixture
def mock_crackerjack_metrics_success() -> dict[str, t.Any]:
    """Create mock successful crackerjack quality metrics.

    Returns:
        Structured quality metrics dictionary.

    """
    return {
        "quality_score": 85,
        "tests": {
            "total": 1000,
            "passed": 980,
            "failed": 0,
            "skipped": 20,
            "pass_rate": 0.98,
        },
        "coverage": {
            "percentage": 14.4,
            "statements": 13873,
            "missing": 11540,
            "covered": 2333,
        },
        "linting": {
            "errors": 0,
            "warnings": 5,
            "style_issues": 2,
        },
        "security": {
            "vulnerabilities": 0,
            "severity_distribution": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        },
        "performance": {
            "slow_tests": 0,
            "average_test_time": 0.05,
            "total_duration": 42.3,
        },
        "recommendations": [
            "Increase test coverage to ≥80%",
            "Fix remaining lint warnings",
            "Consider adding integration tests",
        ],
        "timestamp": "2025-10-29T12:00:00Z",
    }


@pytest.fixture
def mock_crackerjack_metrics_failures() -> dict[str, t.Any]:
    """Create mock crackerjack metrics with failures.

    Returns:
        Structured quality metrics dictionary showing failures.

    """
    return {
        "quality_score": 45,
        "tests": {
            "total": 973,
            "passed": 945,
            "failed": 8,
            "skipped": 20,
            "pass_rate": 0.97,
        },
        "coverage": {
            "percentage": 14.4,
            "statements": 13873,
            "missing": 11540,
            "covered": 2333,
        },
        "linting": {
            "errors": 12,
            "warnings": 25,
            "style_issues": 8,
        },
        "security": {
            "vulnerabilities": 2,
            "severity_distribution": {"critical": 0, "high": 0, "medium": 0, "low": 2},
        },
        "performance": {
            "slow_tests": 3,
            "average_test_time": 0.08,
            "total_duration": 45.7,
        },
        "failing_tests": [
            "tests/unit/test_di_container.py::test_configure_registers_singletons",
            "tests/unit/test_instance_managers.py::test_get_app_monitor_registers_singleton",
            "tests/unit/test_server.py::TestServerQualityScoring::test_calculate_quality_score_with_no_args",
        ],
        "recommendations": [
            "Fix failing tests immediately",
            "Address security vulnerabilities",
            "Refactor complex functions",
            "Improve test coverage",
        ],
        "timestamp": "2025-10-29T12:00:00Z",
    }


@pytest.fixture
def mock_crackerjack_integration() -> Mock:
    """Create a mock CrackerjackIntegration instance.

    Returns:
        Mock object with common crackerjack integration methods.

    Example:
        >>> def test_quality_parsing(mock_crackerjack_integration):
        ...     mock_crackerjack_integration.parse_output.return_value = {"score": 85}
        ...     result = mock_crackerjack_integration.parse_output("output")
        ...     assert result["score"] == 85

    """
    mock = Mock()

    # Mock common integration methods
    mock.parse_output = Mock(
        return_value={
            "quality_score": 85,
            "tests_passed": 980,
            "coverage": 14.4,
        }
    )

    mock.execute_command = Mock(
        return_value={
            "success": True,
            "output": "Quality Score: 85/100",
            "execution_time": 42.3,
            "command": "crackerjack",
        }
    )

    mock.get_quality_trends = Mock(
        return_value={
            "current_score": 85,
            "previous_score": 75,
            "trend": "improving",
            "delta": 10,
        }
    )

    mock.get_test_patterns = Mock(
        return_value={
            "common_failures": [],
            "flaky_tests": [],
            "slow_tests": [],
        }
    )

    return mock


@pytest.fixture
def crackerjack_output_factory() -> t.Callable[..., str]:
    """Create a factory for generating crackerjack output strings.

    Returns:
        Factory function that creates crackerjack output text.

    Example:
        >>> factory = crackerjack_output_factory()
        >>> output = factory(quality_score=90, tests_passed=1000, coverage=85.5)
        >>> assert "90/100" in output
        >>> assert "1000 passed" in output

    """

    def factory(
        quality_score: int = 75,
        tests_passed: int = 980,
        tests_failed: int = 0,
        tests_skipped: int = 20,
        coverage: float = 14.4,
        linting_errors: int = 0,
        linting_warnings: int = 5,
        security_vulns: int = 0,
        duration: float = 42.3,
    ) -> str:
        """Create crackerjack output with given parameters.

        Args:
            quality_score: Quality score (0-100).
            tests_passed: Number of passing tests.
            tests_failed: Number of failing tests.
            tests_skipped: Number of skipped tests.
            coverage: Code coverage percentage.
            linting_errors: Number of linting errors.
            linting_warnings: Number of linting warnings.
            security_vulns: Number of security vulnerabilities.
            duration: Execution duration in seconds.

        Returns:
            Formatted crackerjack output string.

        """
        status_icon = "✅" if tests_failed == 0 else "❌"
        security_icon = "🔒" if security_vulns == 0 else "⚠️"

        return f"""
╭─────────────────────────────────────────────────────────────╮
│                     Crackerjack Quality Report                │
╰─────────────────────────────────────────────────────────────╯

📊 Quality Score: {quality_score}/100

{status_icon} Tests: {tests_passed} passed, {tests_failed} failed, {tests_skipped} skipped
📈 Coverage: {coverage}%
🔍 Linting: {linting_errors} errors, {linting_warnings} warnings
{security_icon} Security: {security_vulns} vulnerabilities
⚡ Performance: No issues detected

Duration: {duration}s
"""

    return factory


@pytest.fixture
def crackerjack_metrics_factory() -> t.Callable[..., dict[str, t.Any]]:
    """Create a factory for generating crackerjack metrics dictionaries.

    Returns:
        Factory function that creates crackerjack metrics.

    Example:
        >>> factory = crackerjack_metrics_factory()
        >>> metrics = factory(quality_score=90, coverage=85.5)
        >>> assert metrics["quality_score"] == 90
        >>> assert metrics["coverage"]["percentage"] == 85.5

    """

    def factory(
        quality_score: int = 75,
        tests_total: int = 1000,
        tests_passed: int = 980,
        tests_failed: int = 0,
        coverage: float = 14.4,
        linting_errors: int = 0,
        security_vulns: int = 0,
    ) -> dict[str, t.Any]:
        """Create crackerjack metrics with given parameters.

        Args:
            quality_score: Quality score (0-100).
            tests_total: Total number of tests.
            tests_passed: Number of passing tests.
            tests_failed: Number of failing tests.
            coverage: Code coverage percentage.
            linting_errors: Number of linting errors.
            security_vulns: Number of security vulnerabilities.

        Returns:
            Structured metrics dictionary.

        """
        return {
            "quality_score": quality_score,
            "tests": {
                "total": tests_total,
                "passed": tests_passed,
                "failed": tests_failed,
                "skipped": tests_total - tests_passed - tests_failed,
                "pass_rate": tests_passed / tests_total if tests_total > 0 else 0.0,
            },
            "coverage": {
                "percentage": coverage,
                "statements": 13873,
                "missing": int(13873 * (1 - coverage / 100)),
                "covered": int(13873 * (coverage / 100)),
            },
            "linting": {
                "errors": linting_errors,
                "warnings": max(0, 10 - linting_errors),
                "style_issues": 2,
            },
            "security": {
                "vulnerabilities": security_vulns,
                "severity_distribution": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": security_vulns,
                },
            },
            "timestamp": "2025-10-29T12:00:00Z",
        }

    return factory


@pytest.fixture
def mock_crackerjack_command_result() -> dict[str, t.Any]:
    """Create a mock crackerjack command execution result.

    Returns:
        Command execution result dictionary.

    """
    return {
        "success": True,
        "command": "crackerjack",
        "args": ["-t"],
        "returncode": 0,
        "stdout": "Quality Score: 85/100\nTests: 980 passed\n",
        "stderr": "",
        "execution_time": 42.3,
        "timestamp": "2025-10-29T12:00:00Z",
    }
