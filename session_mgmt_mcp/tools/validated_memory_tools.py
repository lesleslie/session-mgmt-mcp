#!/usr/bin/env python3
"""Example integration of parameter validation models with MCP tools.

This module demonstrates how to integrate Pydantic parameter validation
models with existing MCP tools for improved type safety and error handling.

Following crackerjack patterns:
- EVERY LINE IS A LIABILITY: Clean, focused tool implementations
- DRY: Reusable validation across all tools
- KISS: Simple integration without over-engineering

Refactored to use utility modules for reduced code duplication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from session_mgmt_mcp.parameter_models import (
    ConceptSearchParams,
    FileSearchParams,
    ReflectionStoreParams,
    SearchQueryParams,
    validate_mcp_params,
)
from session_mgmt_mcp.utils.error_handlers import ValidationError, _get_logger
from session_mgmt_mcp.utils.tool_wrapper import execute_database_tool

# ============================================================================
# Validated Tool Implementations
# ============================================================================


async def _store_reflection_validated_impl(**params: Any) -> str:
    """Implementation for store_reflection tool with parameter validation."""
    # Validate parameters using Pydantic model
    validated = validate_mcp_params(ReflectionStoreParams, params)
    if not validated.is_valid:
        msg = f"Parameter validation failed: {validated.errors}"
        raise ValidationError(msg)

    params_obj = validated.params

    async def operation(db: Any) -> dict[str, Any]:
        """Store reflection operation."""
        reflection_id = await db.store_reflection(
            params_obj.content,
            tags=params_obj.tags,
        )
        return {
            "success": True,
            "id": reflection_id,
            "content": params_obj.content,
            "tags": params_obj.tags,
            "timestamp": datetime.now().isoformat(),
        }

    def formatter(result: dict[str, Any]) -> str:
        """Format reflection storage result."""
        lines = [
            "💾 Reflection stored successfully!",
            f"🆔 ID: {result['id']}",
            f"📝 Content: {result['content'][:100]}...",
        ]
        if result["tags"]:
            lines.append(f"🏷️  Tags: {', '.join(result['tags'])}")
        lines.append(f"📅 Stored: {result['timestamp']}")

        _get_logger().info(
            "Validated reflection stored",
            reflection_id=result["id"],
            content_length=len(result["content"]),
            tags_count=len(result["tags"]),
        )
        return "\n".join(lines)

    return await execute_database_tool(
        operation,
        formatter,
        "Store validated reflection",
    )


async def _quick_search_validated_impl(**params: Any) -> str:
    """Implementation for quick_search tool with parameter validation."""
    # Validate parameters
    validated = validate_mcp_params(SearchQueryParams, params)
    if not validated.is_valid:
        msg = f"Parameter validation failed: {validated.errors}"
        raise ValidationError(msg)

    params_obj = validated.params

    async def operation(db: Any) -> dict[str, Any]:
        """Quick search operation."""
        results = await db.search_conversations(
            query=params_obj.query,
            project=params_obj.project,
            min_score=params_obj.min_score,
            limit=1,
        )

        return {
            "query": params_obj.query,
            "results": results,
            "total_count": len(results),
        }

    def formatter(result: dict[str, Any]) -> str:
        """Format quick search results."""
        lines = [f"🔍 Quick search for: '{result['query']}'"]

        if not result["results"]:
            lines.extend(
                [
                    "🔍 No results found",
                    "💡 Try adjusting your search terms or lowering min_score",
                ]
            )
        else:
            top_result = result["results"][0]
            lines.extend(
                [
                    "📊 Found results (showing top 1)",
                    f"📝 {top_result['content'][:150]}...",
                ]
            )
            if top_result.get("project"):
                lines.append(f"📁 Project: {top_result['project']}")
            if top_result.get("score") is not None:
                lines.append(f"⭐ Relevance: {top_result['score']:.2f}")
            if top_result.get("timestamp"):
                lines.append(f"📅 Date: {top_result['timestamp']}")

        _get_logger().info(
            "Validated quick search executed",
            query=result["query"],
            results_count=result["total_count"],
        )
        return "\n".join(lines)

    return await execute_database_tool(
        operation,
        formatter,
        "Validated quick search",
    )


async def _search_by_file_validated_impl(**params: Any) -> str:
    """Implementation for search_by_file tool with parameter validation."""
    # Validate parameters
    validated = validate_mcp_params(FileSearchParams, params)
    if not validated.is_valid:
        msg = f"Parameter validation failed: {validated.errors}"
        raise ValidationError(msg)

    params_obj = validated.params

    async def operation(db: Any) -> dict[str, Any]:
        """File search operation."""
        results = await db.search_conversations(
            query=params_obj.file_path,
            project=params_obj.project,
            limit=params_obj.limit,
        )

        return {
            "file_path": params_obj.file_path,
            "results": results,
        }

    def formatter(result: dict[str, Any]) -> str:
        """Format file search results."""
        file_path = result["file_path"]
        results = result["results"]

        lines = [
            f"📁 Searching conversations about: {file_path}",
            "=" * 50,
        ]

        if not results:
            lines.extend(
                [
                    "🔍 No conversations found about this file",
                    "💡 The file might not have been discussed in previous sessions",
                ]
            )
        else:
            lines.append(f"📈 Found {len(results)} relevant conversations:")

            for i, res in enumerate(results, 1):
                lines.append(f"\n{i}. 📝 {res['content'][:200]}...")
                if res.get("project"):
                    lines.append(f"   📁 Project: {res['project']}")
                if res.get("score") is not None:
                    lines.append(f"   ⭐ Relevance: {res['score']:.2f}")
                if res.get("timestamp"):
                    lines.append(f"   📅 Date: {res['timestamp']}")

        _get_logger().info(
            "Validated file search executed",
            file_path=file_path,
            results_count=len(results),
        )
        return "\n".join(lines)

    return await execute_database_tool(
        operation,
        formatter,
        "Validated file search",
    )


async def _search_by_concept_validated_impl(**params: Any) -> str:
    """Implementation for search_by_concept tool with parameter validation."""
    # Validate parameters
    validated = validate_mcp_params(ConceptSearchParams, params)
    if not validated.is_valid:
        msg = f"Parameter validation failed: {validated.errors}"
        raise ValidationError(msg)

    params_obj = validated.params

    async def operation(db: Any) -> dict[str, Any]:
        """Concept search operation."""
        results = await db.search_conversations(
            query=params_obj.concept,
            project=params_obj.project,
            limit=params_obj.limit,
        )

        return {
            "concept": params_obj.concept,
            "include_files": params_obj.include_files,
            "results": results,
        }

    def formatter(result: dict[str, Any]) -> str:
        """Format concept search results."""
        concept = result["concept"]
        results = result["results"]

        lines = [
            f"🧠 Searching for concept: '{concept}'",
            "=" * 50,
        ]

        if not results:
            lines.extend(
                [
                    "🔍 No conversations found about this concept",
                    "💡 Try related terms or broader concepts",
                ]
            )
        else:
            lines.append(f"📈 Found {len(results)} related conversations:")

            for i, res in enumerate(results, 1):
                lines.append(f"\n{i}. 📝 {res['content'][:250]}...")
                if res.get("project"):
                    lines.append(f"   📁 Project: {res['project']}")
                if res.get("score") is not None:
                    lines.append(f"   ⭐ Relevance: {res['score']:.2f}")
                if res.get("timestamp"):
                    lines.append(f"   📅 Date: {res['timestamp']}")

                if result["include_files"] and res.get("files"):
                    files = res["files"][:3]
                    if files:
                        lines.append(f"   📄 Files: {', '.join(files)}")

        _get_logger().info(
            "Validated concept search executed",
            concept=concept,
            results_count=len(results),
        )
        return "\n".join(lines)

    return await execute_database_tool(
        operation,
        formatter,
        "Validated concept search",
    )


# ============================================================================
# MCP Tool Registration
# ============================================================================


def register_validated_memory_tools(mcp_server: Any) -> None:
    """Register all validated memory tools with the MCP server.

    These tools demonstrate parameter validation using Pydantic models
    while using the same utility-based refactoring patterns as other tools.
    """

    @mcp_server.tool()  # type: ignore[misc]
    async def store_reflection_validated(**params: Any) -> str:
        """Store a reflection with validated parameters.

        This demonstrates how to integrate Pydantic parameter validation
        with MCP tools for improved type safety.
        """
        return await _store_reflection_validated_impl(**params)

    @mcp_server.tool()  # type: ignore[misc]
    async def quick_search_validated(**params: Any) -> str:
        """Quick search with validated parameters."""
        return await _quick_search_validated_impl(**params)

    @mcp_server.tool()  # type: ignore[misc]
    async def search_by_file_validated(**params: Any) -> str:
        """Search by file with validated parameters."""
        return await _search_by_file_validated_impl(**params)

    @mcp_server.tool()  # type: ignore[misc]
    async def search_by_concept_validated(**params: Any) -> str:
        """Search by concept with validated parameters."""
        return await _search_by_concept_validated_impl(**params)
