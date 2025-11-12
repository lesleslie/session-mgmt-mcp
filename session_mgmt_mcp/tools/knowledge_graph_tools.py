#!/usr/bin/env python3
"""Knowledge Graph MCP tools for semantic memory management.

This module provides MCP tools for interacting with the DuckPGQ-based knowledge graph,
enabling semantic memory through entity-relationship modeling.
"""

from __future__ import annotations

import importlib.util
import re
import typing as t
from typing import TYPE_CHECKING, Any

from acb.adapters import import_adapter
from acb.depends import depends


def _get_logger() -> t.Any:
    """Lazy logger resolution using ACB's logger adapter from DI container."""
    logger_class = import_adapter("logger")
    return depends.get_sync(logger_class)


# Lazy detection flag
_knowledge_graph_available: bool | None = None

if TYPE_CHECKING:
    from session_mgmt_mcp.adapters.knowledge_graph_adapter import (
        KnowledgeGraphDatabaseAdapter as KnowledgeGraphDatabase,
    )


async def _get_knowledge_graph() -> KnowledgeGraphDatabase:
    """Get knowledge graph database instance.

    Note:
        Migration Phase 2.7: Now returns KnowledgeGraphDatabaseAdapter which provides
        the same API as KnowledgeGraphDatabase but uses ACB graph adapter.

    """
    global _knowledge_graph_available

    if _knowledge_graph_available is False:
        msg = "Knowledge graph not available"
        raise ImportError(msg)

    try:
        from session_mgmt_mcp.adapters.knowledge_graph_adapter import (
            KnowledgeGraphDatabaseAdapter,
        )
        from session_mgmt_mcp.di import configure

        # Ensure DI is configured before creating adapter
        configure()

        kg = KnowledgeGraphDatabaseAdapter()
        await kg.initialize()
        _knowledge_graph_available = True
        return kg
    except Exception as e:
        _knowledge_graph_available = False
        msg = f"Knowledge graph not available: {e}"
        raise ImportError(msg) from e


def _check_knowledge_graph_available() -> bool:
    """Check if knowledge graph is available.

    Note:
        Migration Phase 2.7: Checks for KnowledgeGraphDatabaseAdapter instead of old class.

    """
    global _knowledge_graph_available

    if _knowledge_graph_available is None:
        try:
            spec = importlib.util.find_spec(
                "session_mgmt_mcp.adapters.knowledge_graph_adapter"
            )
            _knowledge_graph_available = spec is not None
        except ImportError:
            _knowledge_graph_available = False

    return bool(_knowledge_graph_available)


# Entity extraction patterns for auto-extraction
ENTITY_PATTERNS = {
    "project": r"\b([A-Z][a-z]+-[a-z]+(?:-[a-z]+)*)\b",  # kebab-case projects
    "library": r"\b(ACB|FastMCP|DuckDB|pytest|pydantic|uvicorn)\b",  # Common libs
    "technology": r"\b(Python|JavaScript|TypeScript|Docker|Kubernetes)\b",
    "concept": r"\b(dependency injection|semantic memory|property graph|vector search)\b",
}


# Tool implementations
async def _create_entity_impl(
    name: str,
    entity_type: str,
    observations: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create an entity in the knowledge graph."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            entity = await kg.create_entity(
                name=name,
                entity_type=entity_type,
                observations=observations or [],
                properties=properties or {},
            )

            output = []
            output.append(f"✅ Entity '{name}' created successfully!")
            output.append(f"📊 Type: {entity_type}")
            output.append(f"🆔 ID: {entity['id']}")
            if observations:
                output.append(f"📝 Observations: {len(observations)}")
            if properties:
                output.append(f"⚙️ Properties: {', '.join(properties.keys())}")

            _get_logger().info(
                "Entity created",
                entity_name=name,
                entity_type=entity_type,
                observations_count=len(observations or []),
            )
            return "\n".join(output)

    except Exception as e:
        _get_logger().exception(f"Error creating entity: {e}")
        return f"❌ Error creating entity: {e}"


async def _add_observation_impl(entity_name: str, observation: str) -> str:
    """Add an observation (fact) to an existing entity."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            success = await kg.add_observation(entity_name, observation)

            if success:
                output = []
                output.append(f"✅ Observation added to '{entity_name}'")
                output.append(f"📝 Observation: {observation}")

                _get_logger().info(
                    "Observation added",
                    entity_name=entity_name,
                    observation=observation[:100],
                )
                return "\n".join(output)
            return f"❌ Entity '{entity_name}' not found"

    except Exception as e:
        _get_logger().exception(f"Error adding observation: {e}")
        return f"❌ Error adding observation: {e}"


async def _create_relation_impl(
    from_entity: str,
    to_entity: str,
    relation_type: str,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create a relationship between two entities."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            relation = await kg.create_relation(
                from_entity=from_entity,
                to_entity=to_entity,
                relation_type=relation_type,
                properties=properties or {},
            )

            if relation:
                output = []
                output.append(
                    f"✅ Relationship created: {from_entity} --[{relation_type}]--> {to_entity}"
                )
                output.append(f"🆔 Relation ID: {relation['id']}")
                if properties:
                    output.append(f"⚙️ Properties: {', '.join(properties.keys())}")

                _get_logger().info(
                    "Relation created",
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relation_type=relation_type,
                )
                return "\n".join(output)
            return f"❌ One or both entities not found: {from_entity}, {to_entity}"

    except Exception as e:
        _get_logger().exception(f"Error creating relation: {e}")
        return f"❌ Error creating relation: {e}"


def _format_entity_result(entity: dict[str, t.Any]) -> list[str]:
    """Format a single entity search result."""
    lines = [f"📌 {entity['name']} ({entity['entity_type']})"]

    observations = entity.get("observations")
    if observations:
        lines.append(f"   📝 Observations: {len(observations)}")
        # Show first observation as preview
        if observations:
            preview = observations[0]
            lines.append(f"   └─ {preview[:80]}{'...' if len(preview) > 80 else ''}")

    lines.append("")
    return lines


async def _search_entities_impl(
    query: str,
    entity_type: str | None = None,
    limit: int = 10,
) -> str:
    """Search for entities by name or observations."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            results = await kg.search_entities(
                query=query,
                entity_type=entity_type,
                limit=limit,
            )

            if not results:
                return f"🔍 No entities found matching '{query}'"

            output = [
                f"🔍 Found {len(results)} entities matching '{query}':",
                "",
            ]

            for entity in results:
                output.extend(_format_entity_result(entity))

            _get_logger().info(
                "Entities searched",
                query=query,
                entity_type=entity_type,
                results_count=len(results),
            )
            return "\n".join(output)

    except Exception as e:
        _get_logger().exception(f"Error searching entities: {e}")
        return f"❌ Error searching entities: {e}"


async def _get_entity_relationships_impl(
    entity_name: str,
    relation_type: str | None = None,
    direction: str = "both",
) -> str:
    """Get all relationships for an entity."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            relationships = await kg.get_relationships(
                entity_name=entity_name,
                relation_type=relation_type,
                direction=direction,
            )

            if not relationships:
                return f"🔍 No relationships found for '{entity_name}'"

            output_lines = _build_relationship_output_lines(
                relationships, direction, entity_name
            )

            _get_logger().info(
                "Relationships retrieved",
                entity_name=entity_name,
                relation_type=relation_type,
                direction=direction,
                count=len(relationships),
            )
            return "\n".join(output_lines)

    except Exception as e:
        _get_logger().exception(f"Error getting relationships: {e}")
        return f"❌ Error getting relationships: {e}"


def _build_relationship_output_lines(
    relationships: list[dict[str, Any]], direction: str, entity_name: str
) -> list[str]:
    """Build output lines for relationships based on direction."""
    output = []
    output.append(f"🔗 Found {len(relationships)} relationships for '{entity_name}':")
    output.append("")

    for rel in relationships:
        formatted_rel = _format_relationship(rel, direction, entity_name)
        output.append(formatted_rel)

    return output


def _format_relationship(rel: dict[str, Any], direction: str, entity_name: str) -> str:
    """Format a single relationship based on direction."""
    if direction == "outgoing" or (
        direction == "both" and rel["from_entity"] == entity_name
    ):
        return (
            f"  {rel['from_entity']} --[{rel['relation_type']}]--> {rel['to_entity']}"
        )
    return f"  {rel['from_entity']} <--[{rel['relation_type']}]-- {rel['to_entity']}"


async def _find_path_impl(
    from_entity: str,
    to_entity: str,
    max_depth: int = 5,
) -> str:
    """Find paths between two entities using SQL/PGQ."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            paths = await kg.find_path(
                from_entity=from_entity,
                to_entity=to_entity,
                max_depth=max_depth,
            )

            if not paths:
                return f"🔍 No path found between '{from_entity}' and '{to_entity}'"

            output = []
            output.append(
                f"🛤️ Found {len(paths)} path(s) from '{from_entity}' to '{to_entity}':"
            )
            output.append("")

            for i, path in enumerate(paths, 1):
                output.append(f"{i}. Path length: {path['path_length']} hop(s)")
                output.append(f"   {path['from_entity']} ➜ ... ➜ {path['to_entity']}")
                output.append("")

            _get_logger().info(
                "Paths found",
                from_entity=from_entity,
                to_entity=to_entity,
                paths_count=len(paths),
            )
            return "\n".join(output)

    except Exception as e:
        _get_logger().exception(f"Error finding path: {e}")
        return f"❌ Error finding path: {e}"


def _format_entity_types(entity_types: dict[str, int]) -> list[str]:
    """Format entity types section."""
    if not entity_types:
        return []

    lines = ["📊 Entity Types:"]
    lines.extend(f"   • {etype}: {count}" for etype, count in entity_types.items())
    lines.append("")
    return lines


def _format_relationship_types(relationship_types: dict[str, int]) -> list[str]:
    """Format relationship types section."""
    if not relationship_types:
        return []

    lines = ["🔗 Relationship Types:"]
    lines.extend(
        f"   • {rtype}: {count}" for rtype, count in relationship_types.items()
    )
    lines.append("")
    return lines


async def _get_knowledge_graph_stats_impl() -> str:
    """Get knowledge graph statistics."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            stats = await kg.get_stats()

            output = [
                "📊 Knowledge Graph Statistics",
                "",
                f"📌 Total Entities: {stats['total_entities']}",
                f"🔗 Total Relationships: {stats['total_relationships']}",
                "",
            ]

            output.extend(_format_entity_types(stats.get("entity_types", {})))
            output.extend(
                _format_relationship_types(stats.get("relationship_types", {}))
            )

            output.extend(
                [
                    f"💾 Database: {stats['database_path']}",
                    f"🔧 DuckPGQ: {'✅ Installed' if stats['duckpgq_installed'] else '❌ Not installed'}",
                ]
            )

            _get_logger().info("Knowledge graph stats retrieved", **stats)
            return "\n".join(output)

    except Exception as e:
        _get_logger().exception(f"Error getting stats: {e}")
        return f"❌ Error getting stats: {e}"


def _extract_patterns_from_context(context: str) -> dict[str, set[str]]:
    """Extract entity patterns from context text."""
    extracted: dict[str, set[str]] = {}
    for entity_type, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(
            pattern, context, re.IGNORECASE
        )  # REGEX OK: Entity extraction from user context with predefined safe patterns
        if matches:
            extracted[entity_type] = set(matches)
    return extracted


async def _auto_create_entity_if_new(
    kg: t.Any,
    entity_name: str,
    entity_type: str,
) -> bool:
    """Create entity if it doesn't exist. Returns True if created."""
    existing = await kg.find_entity_by_name(entity_name)
    if not existing:
        await kg.create_entity(
            name=entity_name,
            entity_type=entity_type,
            observations=["Extracted from conversation context"],
        )
        return True
    return False


async def _format_extraction_output(
    extracted: dict[str, set[str]],
    auto_create: bool,
) -> tuple[list[str], int, int]:
    """Format extraction output and optionally auto-create entities."""
    output = ["🔍 Extracted Entities from Context:", ""]
    total_extracted = 0
    created_count = 0

    for entity_type, entities in extracted.items():
        output.append(f"📊 {entity_type.capitalize()}:")

        for entity_name in sorted(entities):
            output.append(f"   • {entity_name}")
            total_extracted += 1

            if auto_create:
                async with await _get_knowledge_graph() as kg:
                    if await _auto_create_entity_if_new(kg, entity_name, entity_type):
                        created_count += 1

        output.append("")

    return output, total_extracted, created_count


async def _extract_entities_from_context_impl(
    context: str,
    auto_create: bool = False,
) -> str:
    """Extract entities from conversation context using pattern matching."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        extracted = _extract_patterns_from_context(context)

        if not extracted:
            return "🔍 No entities detected in context"

        output, total_extracted, created_count = await _format_extraction_output(
            extracted, auto_create
        )

        output.append(f"📊 Total Extracted: {total_extracted}")
        if auto_create:
            output.append(f"✅ Auto-created: {created_count} new entities")

        _get_logger().info(
            "Entities extracted from context",
            total_extracted=total_extracted,
            auto_created=created_count if auto_create else 0,
        )
        return "\n".join(output)

    except Exception as e:
        _get_logger().exception(f"Error extracting entities: {e}")
        return f"❌ Error extracting entities: {e}"


async def _create_single_entity(
    kg: t.Any,
    entity_data: dict[str, Any],
) -> tuple[str | None, tuple[str, str] | None]:
    """Create a single entity. Returns (created_name, None) or (None, (name, error))."""
    try:
        entity = await kg.create_entity(
            name=entity_data["name"],
            entity_type=entity_data["entity_type"],
            observations=entity_data.get("observations", []),
            properties=entity_data.get("properties", {}),
        )
        return entity["name"], None
    except Exception as e:
        return None, (entity_data["name"], str(e))


def _format_batch_results(
    created: list[str],
    failed: list[tuple[str, str]],
) -> list[str]:
    """Format batch creation results."""
    output = [
        "📦 Batch Entity Creation Results:",
        "",
        f"✅ Successfully Created: {len(created)}",
    ]

    if created:
        for name in created[:10]:  # Show first 10
            output.append(f"   • {name}")
        if len(created) > 10:
            output.append(f"   ... and {len(created) - 10} more")
    output.append("")

    if failed:
        output.append(f"❌ Failed: {len(failed)}")
        for name, error in failed[:5]:  # Show first 5 failures
            output.append(f"   • {name}: {error}")
        if len(failed) > 5:
            output.append(f"   ... and {len(failed) - 5} more")

    return output


async def _batch_create_entities_impl(
    entities: list[dict[str, Any]],
) -> str:
    """Bulk create multiple entities."""
    if not _check_knowledge_graph_available():
        return "❌ Knowledge graph not available. Install dependencies: uv sync"

    try:
        async with await _get_knowledge_graph() as kg:
            created = []
            failed = []

            for entity_data in entities:
                created_name, failure = await _create_single_entity(kg, entity_data)
                if created_name:
                    created.append(created_name)
                elif failure:
                    failed.append(failure)

            output = _format_batch_results(created, failed)

            _get_logger().info(
                "Batch entities created",
                total=len(entities),
                created=len(created),
                failed=len(failed),
            )
            return "\n".join(output)

    except Exception as e:
        _get_logger().exception(f"Error in batch create: {e}")
        return f"❌ Error in batch create: {e}"


# Registration function for FastMCP
def register_knowledge_graph_tools(mcp_server: Any) -> None:
    """Register all knowledge graph MCP tools with the server.

    Args:
        mcp_server: FastMCP server instance

    """

    @mcp_server.tool()  # type: ignore[misc]
    async def create_entity(
        name: str,
        entity_type: str,
        observations: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create an entity (node) in the knowledge graph.

        Args:
            name: Entity name (e.g., "session-mgmt-mcp", "Python 3.13")
            entity_type: Type of entity (e.g., "project", "language", "library", "concept")
            observations: List of facts/observations about this entity
            properties: Additional structured properties as key-value pairs

        Returns:
            Success message with entity details

        """
        return await _create_entity_impl(name, entity_type, observations, properties)

    @mcp_server.tool()  # type: ignore[misc]
    async def add_observation(entity_name: str, observation: str) -> str:
        """Add an observation (fact) to an existing entity.

        Args:
            entity_name: Name of the entity to add observation to
            observation: Fact or note to add

        Returns:
            Success message or error if entity not found

        """
        return await _add_observation_impl(entity_name, observation)

    @mcp_server.tool()  # type: ignore[misc]
    async def create_relation(
        from_entity: str,
        to_entity: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a relationship between two entities in the knowledge graph.

        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relation_type: Type of relationship (e.g., "uses", "depends_on", "developed_by")
            properties: Additional properties for the relationship

        Returns:
            Success message with relationship details

        """
        return await _create_relation_impl(
            from_entity, to_entity, relation_type, properties
        )

    @mcp_server.tool()  # type: ignore[misc]
    async def search_entities(
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> str:
        """Search for entities by name or observations.

        Args:
            query: Search query (matches name and observations)
            entity_type: Optional filter by entity type
            limit: Maximum number of results (default: 10)

        Returns:
            List of matching entities with their observations

        """
        return await _search_entities_impl(query, entity_type, limit)

    @mcp_server.tool()  # type: ignore[misc]
    async def get_entity_relationships(
        entity_name: str,
        relation_type: str | None = None,
        direction: str = "both",
    ) -> str:
        """Get all relationships for a specific entity.

        Args:
            entity_name: Name of entity to get relationships for
            relation_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both" (default: "both")

        Returns:
            List of relationships involving this entity

        """
        return await _get_entity_relationships_impl(
            entity_name, relation_type, direction
        )

    @mcp_server.tool()  # type: ignore[misc]
    async def find_path(
        from_entity: str,
        to_entity: str,
        max_depth: int = 5,
    ) -> str:
        """Find paths between two entities using DuckPGQ's SQL/PGQ graph queries.

        This uses SQL:2023 standard graph pattern matching to find connections.

        Args:
            from_entity: Starting entity name
            to_entity: Target entity name
            max_depth: Maximum path length to search (default: 5)

        Returns:
            Paths found between entities with hop counts

        """
        return await _find_path_impl(from_entity, to_entity, max_depth)

    @mcp_server.tool()  # type: ignore[misc]
    async def get_knowledge_graph_stats() -> str:
        """Get statistics about the knowledge graph.

        Returns:
            Summary with entity count, relationship count, type distributions, and database info

        """
        return await _get_knowledge_graph_stats_impl()

    @mcp_server.tool()  # type: ignore[misc]
    async def extract_entities_from_context(
        context: str,
        auto_create: bool = False,
    ) -> str:
        """Extract entities from conversation context using pattern matching.

        Automatically detects:
        - Projects (kebab-case names)
        - Libraries (ACB, FastMCP, DuckDB, pytest, etc.)
        - Technologies (Python, JavaScript, Docker, etc.)
        - Concepts (dependency injection, semantic memory, etc.)

        Args:
            context: Text to extract entities from
            auto_create: If True, automatically create detected entities in the graph

        Returns:
            List of extracted entities by type, with creation status if auto_create enabled

        """
        return await _extract_entities_from_context_impl(context, auto_create)

    @mcp_server.tool()  # type: ignore[misc]
    async def batch_create_entities(
        entities: list[dict[str, Any]],
    ) -> str:
        """Bulk create multiple entities in one operation.

        Args:
            entities: List of entity dictionaries with keys: name, entity_type, observations (optional), properties (optional)

        Returns:
            Summary of successful and failed entity creations

        Example:
            entities = [
                {"name": "FastMCP", "entity_type": "library", "observations": ["MCP server framework"]},
                {"name": "session-mgmt-mcp", "entity_type": "project"}
            ]

        """
        return await _batch_create_entities_impl(entities)
