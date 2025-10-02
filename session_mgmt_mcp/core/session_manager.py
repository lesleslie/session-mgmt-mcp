#!/usr/bin/env python3
"""Session lifecycle management for session-mgmt-mcp.

This module handles session initialization, quality assessment, checkpoints,
and cleanup operations.
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    """Immutable session information."""

    session_id: str = field(default="")
    ended_at: str = field(default="")
    quality_score: str = field(default="")
    working_directory: str = field(default="")
    top_recommendation: str = field(default="")

    def is_complete(self) -> bool:
        """Check if session info has required fields."""
        return bool(self.ended_at and self.quality_score and self.working_directory)

    @classmethod
    def empty(cls) -> "SessionInfo":
        """Create empty session info."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "SessionInfo":
        """Create from dictionary with validation."""
        return cls(  # type: ignore[call-arg]
            session_id=data.get("session_id", ""),
            ended_at=data.get("ended_at", ""),
            quality_score=data.get("quality_score", ""),
            working_directory=data.get("working_directory", ""),
            top_recommendation=data.get("top_recommendation", ""),
        )


from session_mgmt_mcp.utils.git_operations import (
    create_checkpoint_commit,
    is_git_repository,
)
from session_mgmt_mcp.utils.logging import get_session_logger


class SessionLifecycleManager:
    """Manages session lifecycle operations."""

    def __init__(self) -> None:
        self.logger = get_session_logger()
        self.current_project: str | None = None
        self._quality_history: dict[str, list[int]] = {}  # project -> [scores]

    async def calculate_quality_score(self) -> dict[str, Any]:
        """Calculate session quality score using V2 algorithm.

        Delegates to the centralized quality scoring in server.py to avoid
        code duplication and ensure consistent scoring across the system.
        """
        # Import to avoid circular dependencies
        from session_mgmt_mcp import server

        return await server.calculate_quality_score()

    def _calculate_project_score(self, project_context: dict[str, bool]) -> float:
        """Calculate project health score (40% of total)."""
        return (
            sum(1 for detected in project_context.values() if detected)
            / len(project_context)
        ) * 40

    def _calculate_permissions_score(self) -> int:
        """Calculate permissions health score (20% of total)."""
        try:
            from session_mgmt_mcp.server import permissions_manager

            if hasattr(permissions_manager, "trusted_operations"):
                trusted_count = len(permissions_manager.trusted_operations)
                return min(
                    trusted_count * 4, 20
                )  # 4 points per trusted operation, max 20
            return 10  # Basic score if we can't access trusted operations
        except (ImportError, AttributeError):
            return 10  # Fallback score

    def _calculate_session_score(self) -> int:
        """Calculate session management score (20% of total)."""
        return 20  # Always available in this refactored version

    def _calculate_tool_score(self) -> int:
        """Calculate tool availability score (20% of total)."""
        uv_available = shutil.which("uv") is not None
        return 20 if uv_available else 10

    def _format_quality_score_result(
        self,
        total_score: int,
        project_score: float,
        permissions_score: int,
        session_score: int,
        tool_score: int,
        project_context: dict[str, bool],
        uv_available: bool,
    ) -> dict[str, Any]:
        """Format the quality score calculation result."""
        return {
            "total_score": total_score,
            "breakdown": {
                "project_health": project_score,
                "permissions": permissions_score,
                "session_management": session_score,
                "tools": tool_score,
            },
            "recommendations": self._generate_quality_recommendations(
                total_score, project_context, uv_available
            ),
        }

    def _generate_quality_recommendations(
        self,
        score: int,
        project_context: dict[str, Any],
        uv_available: bool,
    ) -> list[str]:
        """Generate quality improvement recommendations based on score factors."""
        recommendations = []

        if score < 50:
            recommendations.append(
                "Session needs attention - multiple areas for improvement",
            )

        if not project_context.get("has_pyproject_toml", False):
            recommendations.append(
                "Consider adding pyproject.toml for modern Python project structure",
            )

        if not project_context.get("has_git_repo", False):
            recommendations.append("Initialize git repository for version control")

        if not uv_available:
            recommendations.append(
                "Install UV package manager for improved dependency management",
            )

        if not project_context.get("has_tests", False):
            recommendations.append("Add test suite to improve code quality")

        if score >= 80:
            recommendations.append("Excellent session setup! Keep up the good work.")
        elif score >= 60:
            recommendations.append("Good session quality with room for optimization.")

        return recommendations[:5]  # Limit to top 5 recommendations

    async def analyze_project_context(self, project_dir: Path) -> dict[str, bool]:
        """Analyze project directory for common indicators and patterns."""
        indicators = self._get_basic_project_indicators(project_dir)
        self._add_python_context_indicators(project_dir, indicators)
        return indicators

    def _get_basic_project_indicators(self, project_dir: Path) -> dict[str, bool]:
        """Get basic project structure indicators."""
        return {
            "has_pyproject_toml": (project_dir / "pyproject.toml").exists(),
            "has_setup_py": (project_dir / "setup.py").exists(),
            "has_requirements_txt": (project_dir / "requirements.txt").exists(),
            "has_readme": self._check_readme_exists(project_dir),
            "has_git_repo": is_git_repository(project_dir),
            "has_venv": self._check_venv_exists(project_dir),
            "has_tests": self._check_tests_exist(project_dir),
            "has_src_structure": (project_dir / "src").exists(),
            "has_docs": self._check_docs_exist(project_dir),
            "has_ci_cd": self._check_ci_cd_exists(project_dir),
        }

    def _check_readme_exists(self, project_dir: Path) -> bool:
        """Check if README file exists."""
        return any(
            (project_dir / name).exists()
            for name in ("README.md", "README.rst", "README.txt", "readme.md")
        )

    def _check_venv_exists(self, project_dir: Path) -> bool:
        """Check if virtual environment exists."""
        return any(
            (project_dir / name).exists() for name in (".venv", "venv", ".env", "env")
        )

    def _check_tests_exist(self, project_dir: Path) -> bool:
        """Check if test directories exist."""
        return any(
            (project_dir / name).exists() for name in ("tests", "test", "testing")
        )

    def _check_docs_exist(self, project_dir: Path) -> bool:
        """Check if documentation directories exist."""
        return any((project_dir / name).exists() for name in ("docs", "documentation"))

    def _check_ci_cd_exists(self, project_dir: Path) -> bool:
        """Check if CI/CD configuration exists."""
        return any(
            (project_dir / name).exists()
            for name in (".github", ".gitlab-ci.yml", ".travis.yml", "Jenkinsfile")
        )

    def _add_python_context_indicators(
        self, project_dir: Path, indicators: dict[str, bool]
    ) -> None:
        """Add Python-specific context indicators."""
        try:
            python_files = list(project_dir.glob("**/*.py"))
            indicators["has_python_files"] = len(python_files) > 0
            self._detect_python_frameworks(python_files, indicators)
        except Exception as e:
            self.logger.warning(f"Error analyzing Python files: {e}")

    def _detect_python_frameworks(
        self, python_files: list[Path], indicators: dict[str, bool]
    ) -> None:
        """Detect Python frameworks from file content."""
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                with py_file.open("r", encoding="utf-8") as f:
                    content = f.read(1000)  # Read first 1000 chars
                    self._check_framework_imports(content, indicators)
            except (UnicodeDecodeError, PermissionError):
                continue

    def _check_framework_imports(
        self, content: str, indicators: dict[str, bool]
    ) -> None:
        """Check for framework imports in file content."""
        if "import fastapi" in content or "from fastapi" in content:
            indicators["uses_fastapi"] = True
        if "import django" in content or "from django" in content:
            indicators["uses_django"] = True
        if "import flask" in content or "from flask" in content:
            indicators["uses_flask"] = True

    async def perform_quality_assessment(self) -> tuple[int, dict[str, Any]]:
        """Perform quality assessment and return score and data."""
        quality_data = await self.calculate_quality_score()
        quality_score = quality_data["total_score"]
        return quality_score, quality_data

    def format_quality_results(
        self,
        quality_score: int,
        quality_data: dict[str, Any],
        checkpoint_result: dict[str, Any] | None = None,
    ) -> list[str]:
        """Format quality assessment results for display."""
        output = []

        # Quality status
        if quality_score >= 80:
            output.append(f"✅ Session quality: EXCELLENT (Score: {quality_score}/100)")
        elif quality_score >= 60:
            output.append(f"✅ Session quality: GOOD (Score: {quality_score}/100)")
        else:
            output.append(
                f"⚠️ Session quality: NEEDS ATTENTION (Score: {quality_score}/100)",
            )

        # Quality breakdown - V2 format (actual code quality metrics)
        output.append("\n📈 Quality breakdown (code health metrics):")
        breakdown = quality_data["breakdown"]
        output.append(f"   • Code quality: {breakdown['code_quality']:.1f}/40")
        output.append(f"   • Project health: {breakdown['project_health']:.1f}/30")
        output.append(f"   • Dev velocity: {breakdown['dev_velocity']:.1f}/20")
        output.append(f"   • Security: {breakdown['security']:.1f}/10")

        # Trust score (separate from quality)
        if "trust_score" in quality_data:
            trust = quality_data["trust_score"]
            output.append(
                f"\n🔐 Trust score: {trust['total']:.0f}/100 (separate metric)"
            )
            output.append(
                f"   • Trusted operations: {trust['breakdown']['trusted_operations']:.0f}/40"
            )
            output.append(
                f"   • Session features: {trust['breakdown']['session_availability']:.0f}/30"
            )
            output.append(
                f"   • Tool ecosystem: {trust['breakdown']['tool_ecosystem']:.0f}/30"
            )

        # Recommendations
        recommendations = quality_data["recommendations"]
        if recommendations:
            output.append("\n💡 Recommendations:")
            for rec in recommendations[:3]:
                output.append(f"   • {rec}")

        # Session management specific results
        if checkpoint_result:
            strengths = checkpoint_result.get("strengths", [])
            if strengths:
                output.append("\n🌟 Session strengths:")
                for strength in strengths[:3]:
                    output.append(f"   • {strength}")

            session_stats = checkpoint_result.get("session_stats", {})
            if session_stats:
                output.append("\n⏱️ Session progress:")
                output.append(
                    f"   • Duration: {session_stats.get('duration_minutes', 0)} minutes",
                )
                output.append(
                    f"   • Checkpoints: {session_stats.get('total_checkpoints', 0)}",
                )
                output.append(
                    f"   • Success rate: {session_stats.get('success_rate', 0):.1f}%",
                )

        return output

    async def perform_git_checkpoint(
        self,
        current_dir: Path,
        quality_score: int,
    ) -> list[str]:
        """Handle git operations for checkpoint commit using the new git utilities."""
        output = []
        output.append("\n" + "=" * 50)
        output.append("📦 Git Checkpoint Commit")
        output.append("=" * 50)

        try:
            # Use the new git utilities
            success, result, git_output = create_checkpoint_commit(
                current_dir,
                self.current_project or "Unknown",
                quality_score,
            )

            output.extend(git_output)

            if success and result != "clean":
                self.logger.info(
                    "Checkpoint commit created",
                    project=self.current_project,
                    commit_hash=result,
                    quality_score=quality_score,
                )

        except Exception as e:
            output.append(f"\n⚠️ Git operations error: {e}")
            self.logger.exception(
                "Git checkpoint error occurred",
                error=str(e),
                project=self.current_project,
            )

        return output

    def _setup_working_directory(self, working_directory: str | None) -> Path:
        """Set up working directory and project name."""
        if working_directory:
            os.chdir(working_directory)

        current_dir = Path.cwd()
        self.current_project = current_dir.name
        return current_dir

    def _setup_claude_directories(self) -> Path:
        """Create .claude directory structure."""
        claude_dir = Path.home() / ".claude"
        claude_dir.mkdir(exist_ok=True)
        (claude_dir / "data").mkdir(exist_ok=True)
        (claude_dir / "logs").mkdir(exist_ok=True)
        return claude_dir

    def _get_previous_session_info(self, current_dir: Path) -> dict[str, Any] | None:
        """Get previous session information if available. Target complexity: ≤5."""
        session_files = self._discover_session_files(current_dir)

        for file_path in session_files:
            session_info = self._read_previous_session_info(file_path)
            if session_info:
                return session_info

        # Fallback to old method
        latest_handoff = self._find_latest_handoff_file(current_dir)
        if latest_handoff:
            return self._read_previous_session_info(latest_handoff)

        return None

    async def initialize_session(
        self,
        working_directory: str | None = None,
    ) -> dict[str, Any]:
        """Initialize a new session with comprehensive setup."""
        try:
            # Setup directories and project
            current_dir = self._setup_working_directory(working_directory)
            claude_dir = self._setup_claude_directories()

            # Analyze project and assess quality
            project_context = await self.analyze_project_context(current_dir)
            quality_score, quality_data = await self.perform_quality_assessment()

            # Get previous session info
            previous_session_info = self._get_previous_session_info(current_dir)

            self.logger.info(
                "Session initialized",
                project=self.current_project,
                quality_score=quality_score,
                working_directory=str(current_dir),
                has_previous_session=previous_session_info is not None,
            )

            return {
                "success": True,
                "project": self.current_project,
                "working_directory": str(current_dir),
                "quality_score": quality_score,
                "quality_data": quality_data,
                "project_context": project_context,
                "claude_directory": str(claude_dir),
                "previous_session": previous_session_info,
            }

        except Exception as e:
            self.logger.exception("Session initialization failed", error=str(e))
            return {"success": False, "error": str(e)}

    def get_previous_quality_score(self, project: str) -> int | None:
        """Get the most recent quality score for a project."""
        scores = self._quality_history.get(project, [])
        return scores[-1] if scores else None

    def record_quality_score(self, project: str, score: int) -> None:
        """Record a quality score for quality trend tracking."""
        if project not in self._quality_history:
            self._quality_history[project] = []
        self._quality_history[project].append(score)
        # Keep only last 10 scores to prevent unbounded growth
        if len(self._quality_history[project]) > 10:
            self._quality_history[project] = self._quality_history[project][-10:]

    async def checkpoint_session(
        self,
        working_directory: str | None = None,
        is_manual: bool = False,
    ) -> dict[str, Any]:
        """Perform a comprehensive session checkpoint.

        Args:
            working_directory: Optional working directory override
            is_manual: Whether this is a manually-triggered checkpoint

        Returns:
            Dictionary containing checkpoint results and auto-store decision

        """
        try:
            current_dir = Path(working_directory) if working_directory else Path.cwd()
            self.current_project = current_dir.name

            # Quality assessment
            quality_score, quality_data = await self.perform_quality_assessment()

            # Get previous score for trend analysis
            previous_score = self.get_previous_quality_score(self.current_project)

            # Record this score for future comparisons
            self.record_quality_score(self.current_project, quality_score)

            # Determine if reflection should be auto-stored
            from session_mgmt_mcp.utils.reflection_utils import (
                format_auto_store_summary,
                should_auto_store_checkpoint,
            )

            auto_store_decision = should_auto_store_checkpoint(
                quality_score=quality_score,
                previous_score=previous_score,
                is_manual=is_manual,
                session_phase="checkpoint",
            )

            # Git checkpoint
            git_output = await self.perform_git_checkpoint(current_dir, quality_score)

            # Format results
            quality_output = self.format_quality_results(quality_score, quality_data)

            self.logger.info(
                "Session checkpoint completed",
                project=self.current_project,
                quality_score=quality_score,
                auto_store_decision=auto_store_decision.should_store,
                auto_store_reason=auto_store_decision.reason.value,
            )

            return {
                "success": True,
                "quality_score": quality_score,
                "quality_output": quality_output,
                "git_output": git_output,
                "timestamp": datetime.now().isoformat(),
                "auto_store_decision": auto_store_decision,
                "auto_store_summary": format_auto_store_summary(auto_store_decision),
            }

        except Exception as e:
            self.logger.exception("Session checkpoint failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def end_session(self, working_directory: str | None = None) -> dict[str, Any]:
        """End the current session with cleanup and summary."""
        try:
            current_dir = Path(working_directory) if working_directory else Path.cwd()
            self.current_project = current_dir.name

            # Final quality assessment
            quality_score, quality_data = await self.perform_quality_assessment()

            # Create session summary
            summary = {
                "project": self.current_project,
                "final_quality_score": quality_score,
                "session_end_time": datetime.now().isoformat(),
                "working_directory": str(current_dir),
                "recommendations": quality_data.get("recommendations", []),
            }

            # Generate handoff documentation
            handoff_content = self._generate_handoff_documentation(
                summary, quality_data
            )

            # Save handoff documentation
            handoff_path = self._save_handoff_documentation(
                handoff_content, current_dir
            )

            self.logger.info(
                "Session ended",
                project=self.current_project,
                final_quality_score=quality_score,
            )

            summary["handoff_documentation"] = (
                str(handoff_path) if handoff_path else None
            )

            return {"success": True, "summary": summary}

        except Exception as e:
            self.logger.exception("Session end failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _build_handoff_header(self, summary: dict[str, Any]) -> list[str]:
        """Build handoff documentation header section."""
        return [
            f"# Session Handoff Report - {summary['project']}",
            "",
            f"**Session ended:** {summary['session_end_time']}",
            f"**Final quality score:** {summary['final_quality_score']}/100",
            f"**Working directory:** {summary['working_directory']}",
            "",
        ]

    def _build_quality_section(self, quality_data: dict[str, Any]) -> list[str]:
        """Build quality assessment section of handoff documentation."""
        lines = ["## Quality Assessment", ""]
        breakdown = quality_data.get("breakdown", {})
        lines.extend(
            [
                f"- **Project health:** {breakdown.get('project_health', 0):.1f}/40",
                f"- **Permissions:** {breakdown.get('permissions', 0):.1f}/20",
                f"- **Session tools:** {breakdown.get('session_management', 0):.1f}/20",
                f"- **Tool availability:** {breakdown.get('tools', 0):.1f}/20",
                "",
            ]
        )
        return lines

    def _build_recommendations_section(self, recommendations: list[str]) -> list[str]:
        """Build recommendations section of handoff documentation."""
        if not recommendations:
            return []

        lines = ["## Recommendations for Next Session", ""]
        lines.extend([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])
        lines.append("")
        return lines

    def _build_static_sections(self) -> list[str]:
        """Build static sections of handoff documentation."""
        return [
            "## Key Achievements",
            "",
            "- Session successfully completed",
            "- Quality metrics captured",
            "- Temporary files cleaned up",
            "",
            "## Next Steps",
            "",
            "1. Review the recommendations above",
            "2. Check the working directory for any uncommitted changes",
            "3. Ensure all necessary files are committed to version control",
            "4. Address any outstanding issues before starting next session",
            "",
        ]

    def _generate_handoff_documentation(
        self, summary: dict[str, Any], quality_data: dict[str, Any]
    ) -> str:
        """Generate comprehensive handoff documentation in markdown format."""
        lines = []

        # Build each section
        lines.extend(self._build_handoff_header(summary))
        lines.extend(self._build_quality_section(quality_data))
        lines.extend(
            self._build_recommendations_section(summary.get("recommendations", []))
        )
        lines.extend(self._build_static_sections())

        return "\n".join(lines)

    def _save_handoff_documentation(
        self, content: str, working_dir: Path
    ) -> Path | None:
        """Save handoff documentation to file."""
        try:
            # Create organized directory structure
            handoff_dir = working_dir / ".crackerjack" / "session" / "handoff"
            handoff_dir.mkdir(parents=True, exist_ok=True)

            # Create handoff filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_handoff_{timestamp}.md"
            handoff_path = handoff_dir / filename

            # Write content to file
            with handoff_path.open("w", encoding="utf-8") as f:
                f.write(content)

            return handoff_path
        except Exception as e:
            self.logger.exception("Failed to save handoff documentation", error=str(e))
            return None

    def _find_latest_handoff_file(self, working_dir: Path) -> Path | None:
        """Find the most recent session handoff file."""
        try:
            handoff_dir = working_dir / ".crackerjack" / "session" / "handoff"

            if not handoff_dir.exists():
                # Check for legacy handoff files in project root
                legacy_files = list(working_dir.glob("session_handoff_*.md"))
                if legacy_files:
                    # Return the most recent legacy file
                    return max(legacy_files, key=lambda f: f.stat().st_mtime)
                return None

            # Find all handoff files
            handoff_files = list(handoff_dir.glob("session_handoff_*.md"))

            if not handoff_files:
                return None

            # Return the most recent file based on timestamp in filename
            return max(handoff_files, key=lambda f: f.name)

        except Exception as e:
            self.logger.debug(f"Error finding handoff files: {e}")
            return None

    def _discover_session_files(self, working_dir: Path) -> list[Path]:
        """Find potential session files in priority order. Target complexity: ≤3."""
        candidates = [
            working_dir / "session_handoff.md",
            working_dir / ".claude" / "session_handoff.md",
            working_dir / "session_summary.md",
        ]
        return [path for path in candidates if path.exists()]

    async def _read_file_safely(self, file_path: Path) -> str:
        """Read file content safely. Target complexity: ≤3."""
        try:
            with file_path.open(encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.debug(f"Failed to read file {file_path}: {e}")
            return ""

    async def _parse_session_file(self, file_path: Path) -> SessionInfo:
        """Parse single session file with error handling. Target complexity: ≤8."""
        try:
            content = await self._read_file_safely(file_path)
            if not content:
                return SessionInfo.empty()

            lines = content.split("\n")
            info_dict = self._extract_session_metadata(lines)
            self._extract_session_recommendations(lines, info_dict)

            return SessionInfo.from_dict(info_dict)

        except Exception as e:
            self.logger.debug(f"Failed to parse session file {file_path}: {e}")
            return SessionInfo.empty()

    def _read_previous_session_info(self, handoff_file: Path) -> dict[str, str] | None:
        """Read previous session information. Target complexity: ≤8."""
        import asyncio

        try:
            # Use the new async parsing method
            session_info = asyncio.run(self._parse_session_file(handoff_file))

            if session_info.is_complete():
                return {
                    "ended_at": session_info.ended_at,
                    "quality_score": session_info.quality_score,
                    "working_directory": session_info.working_directory,
                    "top_recommendation": session_info.top_recommendation,
                }

            return None

        except Exception as e:
            self.logger.debug(f"Error reading previous session info: {e}")
            return None

    def _extract_session_metadata(self, lines: list[str]) -> dict[str, str]:
        """Extract session metadata from handoff file lines."""
        info = {}
        for line in lines:
            if line.startswith("**Session ended:**"):
                info["ended_at"] = line.split("**Session ended:**")[1].strip()
            elif line.startswith("**Final quality score:**"):
                info["quality_score"] = line.split("**Final quality score:**")[
                    1
                ].strip()
            elif line.startswith("**Working directory:**"):
                info["working_directory"] = line.split("**Working directory:**")[
                    1
                ].strip()
        return info

    def _extract_session_recommendations(
        self, lines: list[str], info: dict[str, str]
    ) -> None:
        """Extract first recommendation from recommendations section."""
        in_recommendations = False
        for line in lines:
            if "## Recommendations for Next Session" in line:
                in_recommendations = True
                continue
            if in_recommendations and line.strip().startswith("1."):
                info["top_recommendation"] = line.strip()[3:].strip()  # Remove "1. "
                break
            if in_recommendations and line.startswith("##"):
                break  # End of recommendations section

    async def get_session_status(
        self,
        working_directory: str | None = None,
    ) -> dict[str, Any]:
        """Get current session status and health information."""
        try:
            current_dir = Path(working_directory) if working_directory else Path.cwd()

            self.current_project = current_dir.name

            # Get comprehensive status
            project_context = await self.analyze_project_context(current_dir)
            quality_score, quality_data = await self.perform_quality_assessment()

            # Check system health
            uv_available = shutil.which("uv") is not None
            git_available = is_git_repository(current_dir)
            claude_dir = Path.home() / ".claude"
            claude_dir_exists = claude_dir.exists()

            return {
                "success": True,
                "project": self.current_project,
                "working_directory": str(current_dir),
                "quality_score": quality_score,
                "quality_breakdown": quality_data["breakdown"],
                "recommendations": quality_data["recommendations"],
                "project_context": project_context,
                "system_health": {
                    "uv_available": uv_available,
                    "git_repository": git_available,
                    "claude_directory": claude_dir_exists,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.exception("Failed to get session status", error=str(e))
            return {"success": False, "error": str(e)}
