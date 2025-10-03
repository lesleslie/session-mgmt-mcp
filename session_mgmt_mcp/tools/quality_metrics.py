"""Quality metrics extraction from crackerjack output."""

import re
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

    def format_for_display(self) -> str:
        """Format metrics for user-friendly display."""
        if not self.to_dict():
            return ""

        output = "\n📈 **Quality Metrics**:\n"

        if self.coverage_percent is not None:
            # Coverage ratchet baseline is 42%
            emoji = "✅" if self.coverage_percent >= 42 else "⚠️"
            output += f"- {emoji} Coverage: {self.coverage_percent:.1f}%"
            if self.coverage_percent < 42:
                output += " (below 42% baseline)"
            output += "\n"

        if self.max_complexity:
            # Crackerjack enforces complexity ≤15
            emoji = "✅" if self.max_complexity <= 15 else "❌"
            output += f"- {emoji} Max Complexity: {self.max_complexity}"
            if self.max_complexity > 15:
                output += " (exceeds limit of 15)"
            output += "\n"

        if self.complexity_violations:
            output += (
                f"- ⚠️ Complexity Violations: {self.complexity_violations} "
                f"function{'s' if self.complexity_violations != 1 else ''}\n"
            )

        if self.security_issues:
            output += (
                f"- 🔒 Security Issues: {self.security_issues} "
                f"(Bandit finding{'s' if self.security_issues != 1 else ''})\n"
            )

        if self.tests_failed:
            output += f"- ❌ Tests Failed: {self.tests_failed}\n"
        elif self.tests_passed:
            output += f"- ✅ Tests Passed: {self.tests_passed}\n"

        if self.type_errors:
            output += f"- 📝 Type Errors: {self.type_errors}\n"

        if self.formatting_issues:
            output += f"- ✨ Formatting Issues: {self.formatting_issues}\n"

        return output


class QualityMetricsExtractor:
    """Extract structured quality metrics from crackerjack output."""

    # Regex patterns for metric extraction
    PATTERNS: t.Final[dict[str, str]] = {  # noqa: RUF012
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
            cls.PATTERNS["coverage"], combined
        ):
            metrics.coverage_percent = float(match.group(1))

        # Complexity
        complexity_matches = (
            re.findall(  # REGEX OK: complexity pattern from PATTERNS dict
                cls.PATTERNS["complexity"], stderr
            )
        )
        if complexity_matches:
            complexities = [int(c) for c in complexity_matches]
            metrics.max_complexity = max(complexities)
            metrics.complexity_violations = len(complexities)

        # Security (Bandit codes like B108, B603, etc.)
        metrics.security_issues = len(
            re.findall(  # REGEX OK: security pattern from PATTERNS dict
                cls.PATTERNS["security"], stderr
            )
        )

        # Tests
        if match := re.search(  # REGEX OK: test results pattern from PATTERNS dict
            cls.PATTERNS["tests"], stdout
        ):
            metrics.tests_passed = int(match.group(1))
            if match.group(2):
                metrics.tests_failed = int(match.group(2))

        # Type errors
        type_error_match = re.search(  # REGEX OK: type error count extraction
            r"Found (\d+) error", stderr
        )
        if type_error_match:
            metrics.type_errors = int(type_error_match.group(1))
        else:
            # Count error lines
            metrics.type_errors = len(
                [line for line in stderr.split("\n") if "error:" in line.lower()]
            )

        # Formatting
        metrics.formatting_issues = len(
            re.findall(  # REGEX OK: formatting pattern from PATTERNS dict
                cls.PATTERNS["formatting"], combined
            )
        )

        return metrics
