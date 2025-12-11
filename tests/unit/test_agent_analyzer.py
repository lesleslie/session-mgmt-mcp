"""Unit tests for AgentAnalyzer."""

import pytest
from session_buddy.tools.agent_analyzer import (
    AgentAnalyzer,
    AgentRecommendation,
    AgentType,
)


class TestAgentAnalyzer:
    """Test suite for AgentAnalyzer pattern matching and recommendations."""

    def test_no_recommendations_on_success(self):
        """Test that no recommendations are given when exit code is 0."""
        stdout = "All checks passed!"
        stderr = ""
        exit_code = 0

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert not recommendations

    def test_complexity_violation_high_confidence(self):
        """Test RefactoringAgent recommendation for complexity violations."""
        stdout = ""
        stderr = "Complexity of 18 is too high (threshold: 15)"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.REFACTORING
        assert rec.confidence == 0.9
        assert "Complexity violation" in rec.reason
        assert "--ai-fix" in rec.quick_fix_command

    def test_complex_function_pattern(self):
        """Test RefactoringAgent recommendation for complex function pattern."""
        stdout = ""
        stderr = "Function process_data is too complex (22)"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.REFACTORING
        assert rec.confidence == 0.85
        assert "Complex function" in rec.reason

    def test_bandit_security_issue(self):
        """Test SecurityAgent recommendation for Bandit security codes."""
        stdout = ""
        stderr = """
        crackerjack/utils.py:45: B108: Probable insecure usage of temp file
        crackerjack/cli.py:120: B603: subprocess call - check for execution
        """
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.SECURITY
        assert rec.confidence == 0.8
        assert "Bandit security" in rec.reason

    def test_hardcoded_path_security_issue(self):
        """Test SecurityAgent recommendation for hardcoded paths."""
        stdout = ""
        stderr = "Warning: hardcoded path detected in config.py"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.SECURITY
        assert rec.confidence == 0.85
        assert "Security vulnerability" in rec.reason

    def test_test_failures_numeric(self):
        """Test TestCreationAgent recommendation for numeric test failures."""
        stdout = "5 passed, 3 failed in 2.5s"
        stderr = ""
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.TEST_CREATION
        assert rec.confidence == 0.8
        assert "Test failures" in rec.reason
        assert "--run-tests" in rec.quick_fix_command

    def test_specific_test_failure(self):
        """Test TestCreationAgent recommendation for specific test failures."""
        stdout = "FAILED tests/test_workflow.py::test_integration"
        stderr = ""
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.TEST_CREATION
        assert rec.confidence == 0.85
        assert "Specific test failure" in rec.reason

    def test_low_coverage_below_baseline(self):
        """Test TestSpecialistAgent recommendation for low coverage."""
        stdout = "coverage: 35.5%"
        stderr = ""
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.TEST_SPECIALIST
        assert rec.confidence == 0.7
        assert "Coverage below baseline" in rec.reason

    def test_high_coverage_no_recommendation(self):
        """Test that TestSpecialistAgent is NOT recommended for high coverage."""
        stdout = "coverage: 85.5%"
        stderr = ""
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        # Should not recommend TestSpecialistAgent for coverage above 42%
        agent_types = [rec.agent for rec in recommendations]
        assert AgentType.TEST_SPECIALIST not in agent_types

    def test_type_errors_found_pattern(self):
        """Test ImportOptimizationAgent recommendation for type errors."""
        stdout = ""
        stderr = "Found 5 errors in 3 files"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.IMPORT_OPTIMIZATION
        assert rec.confidence == 0.75
        assert "Type or import errors" in rec.reason

    def test_type_error_inline_pattern(self):
        """Test ImportOptimizationAgent recommendation for inline type errors."""
        stdout = ""
        stderr = "main.py:45: error: Incompatible types in assignment"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.IMPORT_OPTIMIZATION
        assert rec.confidence == 0.75

    def test_formatting_violations(self):
        """Test FormattingAgent recommendation for formatting issues."""
        stdout = "would reformat main.py\nwould reformat utils.py"
        stderr = ""
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.FORMATTING
        assert rec.confidence == 0.9
        assert "Code formatting" in rec.reason

    def test_code_duplication(self):
        """Test DRYAgent recommendation for code duplication."""
        stdout = ""
        stderr = "duplicate code detected in process_data functions"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.DRY
        assert rec.confidence == 0.8
        assert "Code duplication" in rec.reason

    def test_performance_issue(self):
        """Test PerformanceAgent recommendation for performance issues."""
        stdout = ""
        stderr = "Warning: O(n²) algorithm detected in loop"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.PERFORMANCE
        assert rec.confidence == 0.75
        assert "Performance issue" in rec.reason

    def test_documentation_missing(self):
        """Test DocumentationAgent recommendation for missing documentation."""
        stdout = ""
        stderr = "Warning: missing docstring for function process_data"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.agent == AgentType.DOCUMENTATION
        assert rec.confidence == 0.7
        assert "Documentation needs improvement" in rec.reason

    def test_multiple_issues_top_three(self):
        """Test that only top 3 recommendations are returned when multiple issues exist."""
        stdout = "would reformat main.py\n5 failed in 2.5s"
        stderr = """
        Complexity of 20 is too high
        B603: subprocess call detected
        Found 3 errors in 2 files
        duplicate code detected
        """
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        # Should return top 3 by confidence
        assert len(recommendations) == 3
        # Verify sorted by confidence (highest first)
        assert recommendations[0].confidence >= recommendations[1].confidence
        assert recommendations[1].confidence >= recommendations[2].confidence

    def test_duplicate_agent_keeps_highest_confidence(self):
        """Test that duplicate agent recommendations keep the highest confidence."""
        stdout = ""
        stderr = """
        Complexity of 18 is too high (threshold: 15)
        Function process_data is too complex (22)
        """
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        # Both patterns match RefactoringAgent, should keep highest confidence (0.9)
        assert len(recommendations) == 1
        assert recommendations[0].agent == AgentType.REFACTORING
        assert recommendations[0].confidence == 0.9

    def test_format_recommendations_empty(self):
        """Test formatting of empty recommendations list."""
        recommendations = []

        formatted = AgentAnalyzer.format_recommendations(recommendations)

        assert formatted == ""

    def test_format_recommendations_single(self):
        """Test formatting of single recommendation."""
        recommendations = [
            AgentRecommendation(
                agent=AgentType.REFACTORING,
                confidence=0.9,
                reason="Complexity violation detected",
                quick_fix_command="python -m crackerjack --ai-fix",
                pattern_matched=r"Complexity of (\d+) is too high",
            )
        ]

        formatted = AgentAnalyzer.format_recommendations(recommendations)

        assert "🤖 **AI Agent Recommendations**:" in formatted
        assert "🔥 **RefactoringAgent**" in formatted  # 🔥 for confidence >= 0.85
        assert "(confidence: 90%)" in formatted
        assert "Complexity violation detected" in formatted
        assert "python -m crackerjack --ai-fix" in formatted

    def test_format_recommendations_multiple_with_emoji_variation(self):
        """Test formatting of multiple recommendations with different confidence emojis."""
        recommendations = [
            AgentRecommendation(
                agent=AgentType.REFACTORING,
                confidence=0.9,
                reason="Complexity violation",
                quick_fix_command="python -m crackerjack --ai-fix",
                pattern_matched=r"Complexity of (\d+) is too high",
            ),
            AgentRecommendation(
                agent=AgentType.TEST_CREATION,
                confidence=0.8,
                reason="Test failures",
                quick_fix_command="python -m crackerjack --ai-fix --run-tests",
                pattern_matched=r"(\d+) failed",
            ),
            AgentRecommendation(
                agent=AgentType.DOCUMENTATION,
                confidence=0.7,
                reason="Documentation needs improvement",
                quick_fix_command="python -m crackerjack --ai-fix",
                pattern_matched=r"missing.*docstring",
            ),
        ]

        formatted = AgentAnalyzer.format_recommendations(recommendations)

        # High confidence (≥0.85) gets 🔥
        assert "🔥 **RefactoringAgent**" in formatted
        # Lower confidence gets ✨
        assert "✨ **TestCreationAgent**" in formatted
        assert "✨ **DocumentationAgent**" in formatted
        # Verify all three are numbered
        assert "1. " in formatted
        assert "2. " in formatted
        assert "3. " in formatted

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case-insensitive."""
        stdout = ""
        stderr = "COMPLEXITY OF 20 IS TOO HIGH"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        # Should match despite uppercase
        assert len(recommendations) == 1
        assert recommendations[0].agent == AgentType.REFACTORING

    def test_combined_stdout_stderr_analysis(self):
        """Test that both stdout and stderr are analyzed together."""
        stdout = "would reformat main.py"
        stderr = "Complexity of 18 is too high"
        exit_code = 1

        recommendations = AgentAnalyzer.analyze(stdout, stderr, exit_code)

        # Should find patterns in both stdout and stderr
        agent_types = [rec.agent for rec in recommendations]
        assert AgentType.FORMATTING in agent_types
        assert AgentType.REFACTORING in agent_types
