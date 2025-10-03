"""Recommendation engine for learning from crackerjack execution history."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .agent_analyzer import AgentRecommendation, AgentType


@dataclass
class FailurePattern:
    """Detected failure pattern from historical executions."""

    pattern_signature: str  # Unique identifier for the pattern
    occurrences: int
    last_seen: datetime
    successful_fixes: list[AgentType]  # Agents that successfully fixed this pattern
    failed_fixes: list[AgentType]  # Agents that failed to fix this pattern
    avg_fix_time: float  # Average time to fix in seconds


@dataclass
class AgentEffectiveness:
    """Track effectiveness of an agent over time."""

    agent: AgentType
    total_recommendations: int
    successful_fixes: int
    failed_fixes: int
    avg_confidence: float
    success_rate: float  # 0.0-1.0


class RecommendationEngine:
    """Learn from execution history to improve recommendations."""

    @classmethod
    async def analyze_history(
        cls,
        db: Any,  # ReflectionDatabase
        project: str,
        days: int = 30,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Analyze execution history for patterns and effectiveness.

        Args:
            db: ReflectionDatabase instance
            project: Project name
            days: Number of days to analyze
            use_cache: Whether to use cached results (default: True)

        Returns:
            Dictionary with patterns, agent effectiveness, and insights

        """
        # Check cache first
        if use_cache:
            from .history_cache import get_cache

            cache = get_cache()
            cached_result = cache.get(project, days)
            if cached_result:
                return cached_result

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Search for crackerjack executions with agent recommendations
        results = await db.search_conversations(
            query="crackerjack agent_recommendations",
            project=project,
            limit=100,
        )

        # Filter by date
        filtered_results = []
        for result in results:
            timestamp_str = result.get("timestamp")
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        result_date = datetime.fromisoformat(timestamp_str)
                    else:
                        result_date = timestamp_str
                    if result_date >= start_date:
                        filtered_results.append(result)
                except (ValueError, AttributeError):
                    filtered_results.append(result)

        # Extract patterns and effectiveness
        patterns = cls._extract_patterns(filtered_results)
        effectiveness = cls._calculate_agent_effectiveness(filtered_results)
        insights = cls._generate_insights(patterns, effectiveness)

        result = {
            "patterns": patterns,
            "agent_effectiveness": effectiveness,
            "insights": insights,
            "total_executions": len(filtered_results),
            "date_range": {"start": start_date, "end": end_date},
        }

        # Cache the result
        if use_cache:
            from .history_cache import get_cache

            cache = get_cache()
            cache.set(project, days, result)

        return result

    @classmethod
    def _extract_patterns(cls, results: list[dict[str, Any]]) -> list[FailurePattern]:
        """Extract failure patterns from execution history."""
        pattern_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "occurrences": 0,
                "last_seen": None,
                "successful_fixes": [],
                "failed_fixes": [],
                "fix_times": [],
            }
        )

        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            content = result.get("content", "")

            # Extract error signature
            signature = cls._generate_signature(content, metadata)
            if not signature:
                continue

            pattern_data[signature]["occurrences"] += 1

            # Update last seen timestamp
            timestamp_str = result.get("timestamp")
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str)
                    else:
                        timestamp = timestamp_str
                    if (
                        not pattern_data[signature]["last_seen"]
                        or timestamp > pattern_data[signature]["last_seen"]
                    ):
                        pattern_data[signature]["last_seen"] = timestamp
                except (ValueError, AttributeError):
                    pass

            # Track which agents were recommended
            recommendations = metadata.get("agent_recommendations", [])
            if recommendations:
                # Check next execution to see if it was fixed
                if i + 1 < len(results):
                    next_result = results[i + 1]
                    next_metadata = next_result.get("metadata", {})
                    next_exit_code = next_metadata.get("exit_code", 1)

                    for rec in recommendations:
                        agent_name = rec.get("agent")
                        if agent_name:
                            try:
                                agent = AgentType(agent_name)
                                if next_exit_code == 0:
                                    # Success - agent recommendation worked
                                    pattern_data[signature]["successful_fixes"].append(
                                        agent
                                    )
                                    # Track fix time if available
                                    exec_time = next_metadata.get("execution_time")
                                    if exec_time:
                                        pattern_data[signature]["fix_times"].append(
                                            exec_time
                                        )
                                else:
                                    # Failed - agent recommendation didn't work
                                    pattern_data[signature]["failed_fixes"].append(
                                        agent
                                    )
                            except ValueError:
                                pass

        # Convert to FailurePattern objects
        patterns = []
        for signature, data in pattern_data.items():
            avg_fix_time = (
                sum(data["fix_times"]) / len(data["fix_times"])
                if data["fix_times"]
                else 0.0
            )

            pattern = FailurePattern(
                pattern_signature=signature,
                occurrences=data["occurrences"],
                last_seen=data["last_seen"] or datetime.now(),
                successful_fixes=data["successful_fixes"],
                failed_fixes=data["failed_fixes"],
                avg_fix_time=avg_fix_time,
            )
            patterns.append(pattern)

        # Sort by occurrences (most common first)
        return sorted(patterns, key=lambda p: p.occurrences, reverse=True)

    @classmethod
    def _calculate_agent_effectiveness(
        cls, results: list[dict[str, Any]]
    ) -> list[AgentEffectiveness]:
        """Calculate effectiveness metrics for each agent."""
        agent_stats: dict[AgentType, dict[str, Any]] = defaultdict(
            lambda: {
                "total_recommendations": 0,
                "successful_fixes": 0,
                "failed_fixes": 0,
                "confidences": [],
            }
        )

        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            recommendations = metadata.get("agent_recommendations", [])

            if recommendations and i + 1 < len(results):
                next_result = results[i + 1]
                next_metadata = next_result.get("metadata", {})
                next_exit_code = next_metadata.get("exit_code", 1)

                for rec in recommendations:
                    agent_name = rec.get("agent")
                    if agent_name:
                        try:
                            agent = AgentType(agent_name)
                            agent_stats[agent]["total_recommendations"] += 1
                            agent_stats[agent]["confidences"].append(
                                rec.get("confidence", 0.0)
                            )

                            if next_exit_code == 0:
                                agent_stats[agent]["successful_fixes"] += 1
                            else:
                                agent_stats[agent]["failed_fixes"] += 1
                        except ValueError:
                            pass

        # Convert to AgentEffectiveness objects
        effectiveness = []
        for agent, stats in agent_stats.items():
            total = stats["total_recommendations"]
            if total == 0:
                continue

            successful = stats["successful_fixes"]
            success_rate = successful / total if total > 0 else 0.0

            avg_confidence = (
                sum(stats["confidences"]) / len(stats["confidences"])
                if stats["confidences"]
                else 0.0
            )

            effectiveness.append(
                AgentEffectiveness(
                    agent=agent,
                    total_recommendations=total,
                    successful_fixes=successful,
                    failed_fixes=stats["failed_fixes"],
                    avg_confidence=avg_confidence,
                    success_rate=success_rate,
                )
            )

        # Sort by success rate (highest first)
        return sorted(effectiveness, key=lambda e: e.success_rate, reverse=True)

    @classmethod
    def _generate_signature(cls, content: str, metadata: dict[str, Any]) -> str:
        """Generate unique signature for a failure pattern."""
        # Extract key error indicators
        exit_code = metadata.get("exit_code", 0)
        if exit_code == 0:
            return ""  # Not a failure

        metrics = metadata.get("metrics", {})

        # Build signature from error characteristics
        signature_parts = []

        # Complexity violations
        if metrics.get("complexity_violations", 0) > 0:
            signature_parts.append(f"complexity:{metrics['max_complexity']}")

        # Security issues
        if metrics.get("security_issues", 0) > 0:
            signature_parts.append(f"security:{metrics['security_issues']}")

        # Test failures
        if metrics.get("tests_failed", 0) > 0:
            signature_parts.append(f"test_failures:{metrics['tests_failed']}")

        # Type errors
        if metrics.get("type_errors", 0) > 0:
            signature_parts.append(f"type_errors:{metrics['type_errors']}")

        # Formatting issues
        if metrics.get("formatting_issues", 0) > 0:
            signature_parts.append("formatting")

        # Extract specific error patterns from content
        error_patterns = [
            r"B\d{3}",  # Bandit codes
            r"E\d{3}",  # Ruff codes
            r"F\d{3}",  # Pyflakes codes
        ]

        for pattern in error_patterns:
            matches = re.findall(  # REGEX OK: error code extraction from patterns
                pattern, content
            )
            if matches:
                signature_parts.extend(sorted(set(matches))[:3])  # Top 3 unique codes

        return "|".join(signature_parts) if signature_parts else "unknown_failure"

    @classmethod
    def _generate_insights(
        cls,
        patterns: list[FailurePattern],
        effectiveness: list[AgentEffectiveness],
    ) -> list[str]:
        """Generate actionable insights from patterns and effectiveness data."""
        insights = []

        # Pattern insights
        if patterns:
            most_common = patterns[0]
            insights.append(
                f"🔄 Most common failure: '{most_common.pattern_signature}' "
                f"({most_common.occurrences} occurrences)"
            )

            # Check for recent patterns
            recent_patterns = [
                p for p in patterns if (datetime.now() - p.last_seen).days <= 7
            ]
            if len(recent_patterns) > 3:
                insights.append(
                    f"⚠️ {len(recent_patterns)} different failure patterns in last 7 days - "
                    f"consider addressing root causes"
                )

        # Agent effectiveness insights
        if effectiveness:
            top_agent = effectiveness[0]
            if top_agent.success_rate >= 0.8:
                insights.append(
                    f"⭐ {top_agent.agent.value} has {top_agent.success_rate:.0%} success rate - "
                    f"highly effective!"
                )

            low_performers = [e for e in effectiveness if e.success_rate < 0.3]
            if low_performers:
                agents = ", ".join(e.agent.value for e in low_performers[:2])
                insights.append(
                    f"📉 Low success rate for: {agents} - "
                    f"review recommendations or patterns"
                )

        # Cross-pattern insights
        if patterns and effectiveness:
            # Find patterns with consistent successful fixes
            reliable_fixes = [
                p
                for p in patterns
                if len(p.successful_fixes) > 0
                and len(p.failed_fixes) == 0
                and p.occurrences >= 2
            ]
            if reliable_fixes:
                insights.append(
                    f"✅ {len(reliable_fixes)} patterns have consistent successful fixes - "
                    f"good agent-pattern matching"
                )

        if not insights:
            insights.append(
                "📊 Insufficient data - continue using AI mode to build history"
            )

        return insights

    @classmethod
    def adjust_confidence(
        cls,
        recommendations: list[AgentRecommendation],
        effectiveness: list[AgentEffectiveness],
    ) -> list[AgentRecommendation]:
        """Adjust recommendation confidence scores based on historical effectiveness.

        Args:
            recommendations: Original recommendations from AgentAnalyzer
            effectiveness: Historical effectiveness data

        Returns:
            Recommendations with adjusted confidence scores

        """
        # Create effectiveness lookup
        effectiveness_map = {e.agent: e for e in effectiveness}

        adjusted = []
        for rec in recommendations:
            agent_eff = effectiveness_map.get(rec.agent)

            if agent_eff and agent_eff.total_recommendations >= 5:
                # Enough data to adjust - blend original and learned confidence
                learned_confidence = agent_eff.success_rate

                # Weighted average: 60% learned, 40% original
                adjusted_confidence = (0.6 * learned_confidence) + (
                    0.4 * rec.confidence
                )

                # Create adjusted recommendation
                adjusted_rec = AgentRecommendation(
                    agent=rec.agent,
                    confidence=min(adjusted_confidence, 1.0),  # Cap at 1.0
                    reason=f"{rec.reason} (adjusted based on {agent_eff.success_rate:.0%} historical success)",
                    quick_fix_command=rec.quick_fix_command,
                    pattern_matched=rec.pattern_matched,
                )
                adjusted.append(adjusted_rec)
            else:
                # Not enough data - keep original
                adjusted.append(rec)

        # Re-sort by adjusted confidence
        return sorted(adjusted, key=lambda r: r.confidence, reverse=True)[:3]
