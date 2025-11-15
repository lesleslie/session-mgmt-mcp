"""Quality metrics extraction from crackerjack output."""

import re
import typing as t
from dataclasses import dataclass
from typing import Any


@dataclass
class QualityMetrics:
    """Structured quality metrics from crackerjack execution."""

    coverage_percent: float | None = None
    max_complexity: int | None = None
    complexity_violations: int = 0
    security_issues: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    type_errors: int = 0
    formatting_issues: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None and zero values."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and (not isinstance(v, int) or v > 0)
        }

    def _format_coverage(self) -> str:
        """Format coverage metric."""
        if self.coverage_percent is None:
            return ""
        emoji = "✅" if self.coverage_percent >= 42 else "⚠️"
        result = f"- {emoji} Coverage: {self.coverage_percent:.1f}%"
        if self.coverage_percent < 42:
            result += " (below 42% baseline)"
        return result + "\n"

    def _format_complexity(self) -> str:
        """Format complexity metric."""
        if not self.max_complexity:
            return ""
        emoji = "✅" if self.max_complexity <= 15 else "❌"
        result = f"- {emoji} Max Complexity: {self.max_complexity}"
        if self.max_complexity > 15:
            result += " (exceeds limit of 15)"
        return result + "\n"

    def _format_violations(self) -> str:
        """Format violation metrics."""
        lines = []
        if self.complexity_violations:
            plural = "s" if self.complexity_violations != 1 else ""
            lines.append(
                f"- ⚠️ Complexity Violations: {self.complexity_violations} function{plural}",
            )
        if self.security_issues:
            plural = "s" if self.security_issues != 1 else ""
            lines.append(
                f"- 🔒 Security Issues: {self.security_issues} (Bandit finding{plural})",
            )
        return "\n".join(lines) + ("\n" if lines else "")

    def _format_tests(self) -> str:
        """Format test results."""
        if self.tests_failed:
            return f"- ❌ Tests Failed: {self.tests_failed}\n"
        if self.tests_passed:
            return f"- ✅ Tests Passed: {self.tests_passed}\n"
        return ""

    def _format_errors(self) -> str:
        """Format error metrics."""
        lines = []
        if self.type_errors:
            lines.append(f"- 📝 Type Errors: {self.type_errors}")
        if self.formatting_issues:
            lines.append(f"- ✨ Formatting Issues: {self.formatting_issues}")
        return "\n".join(lines) + ("\n" if lines else "")

    def format_for_display(self) -> str:
        """Format metrics for user-friendly display."""
        if not self.to_dict():
            return ""

        return (
            "\n📈 **Quality Metrics**:\n"
            + self._format_coverage()
            + self._format_complexity()
            + self._format_violations()
            + self._format_tests()
            + self._format_errors()
        )


class QualityMetricsExtractor:
    """Extract structured quality metrics from crackerjack output."""

    # Regex patterns for metric extraction
    PATTERNS: t.Final[dict[str, str]] = {
        "coverage": r"coverage:?\s*(\d+(?:\.\d+)?)%",
        "complexity": r"Complexity of (\d+) is too high",
        "security": r"B\d{3}:",  # Bandit security codes
        "tests": r"(\d+) passed(?:.*?(\d+) failed)?",
        "type_errors": r"error:|Found (\d+) error",
        "formatting": r"would reformat|line too long",
    }

    @classmethod
    def extract(cls, stdout: str, stderr: str) -> QualityMetrics:
        """Extract metrics from crackerjack output.

        Args:
            stdout: Standard output from crackerjack execution
            stderr: Standard error from crackerjack execution

        Returns:
            QualityMetrics object with extracted values

        """
        metrics = QualityMetrics()
        combined = stdout + stderr

        # Coverage
        if match := re.search(  # REGEX OK: coverage pattern from PATTERNS dict
            cls.PATTERNS["coverage"],
            combined,
        ):
            metrics.coverage_percent = float(match.group(1))

        # Complexity
        complexity_matches = (
            re.findall(  # REGEX OK: complexity pattern from PATTERNS dict
                cls.PATTERNS["complexity"],
                stderr,
            )
        )
        if complexity_matches:
            complexities = [int(c) for c in complexity_matches]
            metrics.max_complexity = max(complexities)
            metrics.complexity_violations = len(complexities)

        # Security (Bandit codes like B108, B603, etc.)
        metrics.security_issues = len(
            re.findall(  # REGEX OK: security pattern from PATTERNS dict
                cls.PATTERNS["security"],
                stderr,
            ),
        )

        # Tests
        if match := re.search(  # REGEX OK: test results pattern from PATTERNS dict
            cls.PATTERNS["tests"],
            stdout,
        ):
            metrics.tests_passed = int(match.group(1))
            if match.group(2):
                metrics.tests_failed = int(match.group(2))

        # Type errors
        type_error_match = re.search(  # REGEX OK: type error count extraction
            r"Found (\d+) error",
            stderr,
        )
        if type_error_match:
            metrics.type_errors = int(type_error_match.group(1))
        else:
            # Count error lines
            metrics.type_errors = len(
                [line for line in stderr.split("\n") if "error:" in line.lower()],
            )

        # Formatting
        metrics.formatting_issues = len(
            re.findall(  # REGEX OK: formatting pattern from PATTERNS dict
                cls.PATTERNS["formatting"],
                combined,
            ),
        )

        return metrics
