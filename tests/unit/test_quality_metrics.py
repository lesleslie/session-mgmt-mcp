"""Unit tests for QualityMetricsExtractor."""

import pytest
from session_buddy.tools.quality_metrics import (
    QualityMetrics,
    QualityMetricsExtractor,
)


class TestQualityMetrics:
    """Test suite for QualityMetrics dataclass."""

    def test_to_dict_excludes_none_and_zeros(self):
        """Test that to_dict excludes None and zero values."""
        metrics = QualityMetrics(
            coverage_percent=85.5,
            max_complexity=None,
            complexity_violations=0,
            security_issues=2,
            tests_passed=0,
            tests_failed=0,
            type_errors=5,
            formatting_issues=0,
        )

        result = metrics.to_dict()

        assert result == {
            "coverage_percent": 85.5,
            "security_issues": 2,
            "type_errors": 5,
        }

    def test_format_for_display_coverage_above_baseline(self):
        """Test display formatting for coverage above 42% baseline."""
        metrics = QualityMetrics(coverage_percent=85.5)
        output = metrics.format_for_display()

        assert "📈 **Quality Metrics**:" in output
        assert "✅ Coverage: 85.5%" in output
        assert "below 42% baseline" not in output

    def test_format_for_display_coverage_below_baseline(self):
        """Test display formatting for coverage below 42% baseline."""
        metrics = QualityMetrics(coverage_percent=35.0)
        output = metrics.format_for_display()

        assert "⚠️ Coverage: 35.0%" in output
        assert "(below 42% baseline)" in output

    def test_format_for_display_complexity_within_limit(self):
        """Test display formatting for complexity within 15 limit."""
        metrics = QualityMetrics(max_complexity=12)
        output = metrics.format_for_display()

        assert "✅ Max Complexity: 12" in output
        assert "exceeds limit" not in output

    def test_format_for_display_complexity_exceeds_limit(self):
        """Test display formatting for complexity exceeding 15 limit."""
        metrics = QualityMetrics(max_complexity=18, complexity_violations=1)
        output = metrics.format_for_display()

        assert "❌ Max Complexity: 18" in output
        assert "(exceeds limit of 15)" in output
        assert "⚠️ Complexity Violations: 1 function" in output

    def test_format_for_display_security_issues(self):
        """Test display formatting for security issues."""
        metrics = QualityMetrics(security_issues=3)
        output = metrics.format_for_display()

        assert "🔒 Security Issues: 3 (Bandit findings)" in output

    def test_format_for_display_tests(self):
        """Test display formatting for test results."""
        # Tests failed case
        metrics1 = QualityMetrics(tests_passed=10, tests_failed=2)
        output1 = metrics1.format_for_display()
        assert "❌ Tests Failed: 2" in output1

        # Tests passed case (no failures)
        metrics2 = QualityMetrics(tests_passed=10, tests_failed=0)
        output2 = metrics2.format_for_display()
        assert "✅ Tests Passed: 10" in output2

    def test_format_for_display_empty_metrics(self):
        """Test display formatting for empty metrics returns empty string."""
        metrics = QualityMetrics()
        output = metrics.format_for_display()
        assert output == ""


class TestQualityMetricsExtractor:
    """Test suite for QualityMetricsExtractor."""

    def test_extract_coverage(self):
        """Test coverage extraction from output."""
        stdout = "TOTAL coverage: 85.5%"
        stderr = ""

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.coverage_percent == 85.5

    def test_extract_complexity_violations(self):
        """Test complexity violation extraction."""
        stdout = ""
        stderr = """
        Complexity of 18 is too high (threshold 15)
        Complexity of 22 is too high (threshold 15)
        Complexity of 16 is too high (threshold 15)
        """

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.max_complexity == 22  # Maximum of [18, 22, 16]
        assert metrics.complexity_violations == 3

    def test_extract_security_issues(self):
        """Test security issue (Bandit code) extraction."""
        stdout = ""
        stderr = """
        test.py:10: B108: Probable insecure usage of temp file
        test.py:25: B603: subprocess call - check for execution
        test.py:40: B101: Assert statement detected
        """

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.security_issues == 3

    def test_extract_test_results_passed_only(self):
        """Test extraction when all tests pass."""
        stdout = "15 passed in 2.5s"
        stderr = ""

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.tests_passed == 15
        assert metrics.tests_failed == 0

    def test_extract_test_results_with_failures(self):
        """Test extraction when tests have failures."""
        stdout = "10 passed, 3 failed in 5.2s"
        stderr = ""

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.tests_passed == 10
        assert metrics.tests_failed == 3

    def test_extract_type_errors_counted_by_found(self):
        """Test type error extraction using 'Found N error' pattern."""
        stdout = ""
        stderr = "Found 5 errors in 3 files"

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.type_errors == 5

    def test_extract_type_errors_counted_by_lines(self):
        """Test type error extraction by counting error lines."""
        stdout = ""
        stderr = """
        file.py:10: error: Incompatible types
        file.py:20: error: Missing return statement
        file.py:30: error: Undefined name 'foo'
        """

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.type_errors == 3

    def test_extract_formatting_issues(self):
        """Test formatting issue extraction."""
        stdout = """
        would reformat file1.py
        would reformat file2.py
        line too long at file3.py:10
        """
        stderr = ""

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.formatting_issues == 3

    def test_extract_combined_metrics(self):
        """Test extraction of all metrics types together."""
        stdout = """
        coverage: 75.2%
        10 passed, 2 failed in 3.1s
        would reformat main.py
        """
        stderr = """
        Complexity of 20 is too high (threshold 15)
        test.py:15: B603: subprocess call - check for execution
        test.py:25: error: Incompatible types
        test.py:30: error: Missing return
        Found 2 errors in 1 file
        """

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.coverage_percent == 75.2
        assert metrics.max_complexity == 20
        assert metrics.complexity_violations == 1
        assert metrics.security_issues == 1
        assert metrics.tests_passed == 10
        assert metrics.tests_failed == 2
        assert metrics.type_errors == 2  # "Found 2 errors" takes precedence
        assert metrics.formatting_issues == 1

    def test_extract_no_metrics(self):
        """Test extraction when no metrics are present."""
        stdout = "Everything looks good!"
        stderr = ""

        metrics = QualityMetricsExtractor.extract(stdout, stderr)

        assert metrics.coverage_percent is None
        assert metrics.max_complexity is None
        assert metrics.complexity_violations == 0
        assert metrics.security_issues == 0
        assert metrics.tests_passed == 0
        assert metrics.tests_failed == 0
        assert metrics.type_errors == 0
        assert metrics.formatting_issues == 0
