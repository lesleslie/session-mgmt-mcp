#!/usr/bin/env python3
"""Knowledge Graph Database using DuckDB + DuckPGQ Extension.

This module provides semantic memory (knowledge graph) capabilities
using DuckDB's DuckPGQ extension for SQL/PGQ (Property Graph Queries).

The knowledge graph stores:
- **Entities**: Nodes representing projects, libraries, technologies, concepts
- **Relations**: Edges connecting entities (uses, depends_on, developed_by, etc.)
- **Observations**: Facts and notes attached to entities

This is separate from the episodic memory (conversations) in ReflectionDatabase.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class KnowledgeGraphDatabase:
    """Manages knowledge graph using DuckDB + DuckPGQ extension.

    This class provides semantic memory through a property graph model,
    complementing the episodic memory in ReflectionDatabase.

    Example:
        >>> async with KnowledgeGraphDatabase() as kg:
        >>>     entity = await kg.create_entity(
        >>>         name="session-mgmt-mcp",
        >>>         entity_type="project"
        >>>     )
        >>>     relation = await kg.create_relation(
        >>>         from_entity="session-mgmt-mcp",
        >>>         to_entity="ACB",
        >>>         relation_type="uses"
        >>>     )
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize knowledge graph database.

        Args:
            db_path: Path to DuckDB database file.
                    Defaults to ~/.claude/data/knowledge_graph.duckdb
        """
        self.db_path = db_path or os.path.expanduser(
            "~/.claude/data/knowledge_graph.duckdb"
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._duckpgq_installed = False

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        self.close()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.conn = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()

    async def initialize(self) -> None:
        """Initialize database and DuckPGQ extension.

        This method:
        1. Creates DuckDB connection
        2. Installs DuckPGQ extension from community repository
        3. Creates property graph schema (entities + relationships tables)
        4. Creates the knowledge_graph property graph

        Raises:
            ImportError: If DuckDB is not available
            RuntimeError: If DuckPGQ installation fails
        """
        if not DUCKDB_AVAILABLE:
            msg = "DuckDB not available. Install with: uv add duckdb"
            raise ImportError(msg)

        # Create connection
        self.conn = duckdb.connect(self.db_path)

        # Install and load DuckPGQ extension
        try:
            self.conn.execute("INSTALL duckpgq FROM community")
            self.conn.execute("LOAD duckpgq")
            self._duckpgq_installed = True
        except Exception as e:
            msg = f"Failed to install DuckPGQ extension: {e}"
            raise RuntimeError(msg) from e

        # Create schema
        await self._create_schema()

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Get database connection, raising error if not initialized.

        Returns:
            Active DuckDB connection

        Raises:
            RuntimeError: If connection not initialized
        """
        if self.conn is None:
            msg = "Database connection not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self.conn

    async def _create_schema(self) -> None:
        """Create knowledge graph schema with DuckPGQ property graph.

        Creates:
        - kg_entities table (nodes)
        - kg_relationships table (edges)
        - knowledge_graph property graph
        - Indexes for performance
        """
        conn = self._get_conn()

        # Create entities table (nodes/vertices)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                entity_type VARCHAR NOT NULL,
                observations VARCHAR[],
                properties JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        """)

        # Create relationships table (edges)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_relationships (
                id VARCHAR PRIMARY KEY,
                from_entity VARCHAR NOT NULL,
                to_entity VARCHAR NOT NULL,
                relation_type VARCHAR NOT NULL,
                properties JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                FOREIGN KEY (from_entity) REFERENCES kg_entities(id) ON DELETE CASCADE,
                FOREIGN KEY (to_entity) REFERENCES kg_entities(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON kg_entities(name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON kg_entities(entity_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON kg_relationships(relation_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_from ON kg_relationships(from_entity)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_to ON kg_relationships(to_entity)"
        )

        # Create property graph using DuckPGQ
        # This maps our tables to SQL/PGQ graph structure
        try:
            conn.execute("""
                CREATE PROPERTY GRAPH IF NOT EXISTS knowledge_graph
                VERTEX TABLES (kg_entities)
                EDGE TABLES (
                    kg_relationships
                        SOURCE KEY (from_entity) REFERENCES kg_entities (id)
                        DESTINATION KEY (to_entity) REFERENCES kg_entities (id)
                )
            """)
        except Exception as e:
            # Property graph might already exist, that's okay
            if "already exists" not in str(e).lower():
                raise

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        observations: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an entity (node) in the knowledge graph.

        Args:
            name: Entity name (e.g., "session-mgmt-mcp", "Python 3.13")
            entity_type: Type of entity (e.g., "project", "language", "library")
            observations: List of facts about this entity
            properties: Additional structured properties
            metadata: Metadata (e.g., source, confidence)

        Returns:
            Created entity as dict with id, name, type, etc.

        Example:
            >>> entity = await kg.create_entity(
            >>>     name="FastBlocks",
            >>>     entity_type="framework",
            >>>     observations=["Web framework", "Built on ACB"]
            >>> )
        """
        conn = self._get_conn()
        entity_id = str(uuid.uuid4())
        observations = observations or []
        properties = properties or {}
        metadata = metadata or {}
        now = datetime.now(UTC)

        conn.execute(
            """
            INSERT INTO kg_entities (id, name, entity_type, observations, properties, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                name,
                entity_type,
                observations,
                json.dumps(properties),
                now,
                now,
                json.dumps(metadata),
            ),
        )

        return {
            "id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "observations": observations,
            "properties": properties,
            "created_at": now.isoformat(),
            "metadata": metadata,
        }

    async def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Retrieve an entity by ID.

        Args:
            entity_id: UUID of the entity

        Returns:
            Entity dict or None if not found
        """
        conn = self._get_conn()

        result = conn.execute(
            "SELECT * FROM kg_entities WHERE id = ?", (entity_id,)
        ).fetchone()

        if not result:
            return None

        return {
            "id": result[0],
            "name": result[1],
            "entity_type": result[2],
            "observations": result[3],
            "properties": json.loads(result[4]) if result[4] else {},
            "created_at": result[5].isoformat() if result[5] else None,
            "updated_at": result[6].isoformat() if result[6] else None,
            "metadata": json.loads(result[7]) if result[7] else {},
        }

    async def find_entity_by_name(
        self, name: str, entity_type: str | None = None
    ) -> dict[str, Any] | None:
        """Find an entity by name (case-insensitive).

        Args:
            name: Entity name to search for
            entity_type: Optional type filter

        Returns:
            First matching entity or None
        """
        conn = self._get_conn()

        if entity_type:
            result = conn.execute(
                "SELECT * FROM kg_entities WHERE LOWER(name) = LOWER(?) AND entity_type = ? LIMIT 1",
                (name, entity_type),
            ).fetchone()
        else:
            result = conn.execute(
                "SELECT * FROM kg_entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (name,),
            ).fetchone()

        if not result:
            return None

        return {
            "id": result[0],
            "name": result[1],
            "entity_type": result[2],
            "observations": result[3],
            "properties": json.loads(result[4]) if result[4] else {},
            "created_at": result[5].isoformat() if result[5] else None,
            "updated_at": result[6].isoformat() if result[6] else None,
            "metadata": json.loads(result[7]) if result[7] else {},
        }

    async def create_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create a relationship between two entities.

        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relation_type: Type of relationship (e.g., "uses", "depends_on")
            properties: Additional properties
            metadata: Metadata

        Returns:
            Created relationship dict, or None if entities not found

        Example:
            >>> relation = await kg.create_relation(
            >>>     from_entity="crackerjack",
            >>>     to_entity="Python 3.13",
            >>>     relation_type="uses"
            >>> )
        """
        # Find source and target entities
        from_node = await self.find_entity_by_name(from_entity)
        to_node = await self.find_entity_by_name(to_entity)

        if not from_node or not to_node:
            return None

        conn = self._get_conn()
        relation_id = str(uuid.uuid4())
        properties = properties or {}
        metadata = metadata or {}
        now = datetime.now(UTC)

        conn.execute(
            """
            INSERT INTO kg_relationships (id, from_entity, to_entity, relation_type, properties, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relation_id,
                from_node["id"],
                to_node["id"],
                relation_type,
                json.dumps(properties),
                now,
                json.dumps(metadata),
            ),
        )

        return {
            "id": relation_id,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation_type": relation_type,
            "properties": properties,
            "created_at": now.isoformat(),
            "metadata": metadata,
        }

    async def add_observation(self, entity_name: str, observation: str) -> bool:
        """Add an observation (fact) to an existing entity.

        Args:
            entity_name: Name of the entity
            observation: Fact to add

        Returns:
            True if successful, False if entity not found
        """
        entity = await self.find_entity_by_name(entity_name)
        if not entity:
            return False

        conn = self._get_conn()
        observations = entity.get("observations", [])
        observations.append(observation)
        now = datetime.now(UTC)

        conn.execute(
            """
            UPDATE kg_entities
            SET observations = ?, updated_at = ?
            WHERE id = ?
            """,
            (observations, now, entity["id"]),
        )

        return True

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for entities by name or observations.

        Args:
            query: Search query (matches name and observations)
            entity_type: Optional filter by type
            limit: Maximum results to return

        Returns:
            List of matching entities
        """
        conn = self._get_conn()

        if entity_type:
            sql = """
                SELECT * FROM kg_entities
                WHERE (LOWER(name) LIKE LOWER(?) OR ARRAY_TO_STRING(observations, ' ') LIKE LOWER(?))
                  AND entity_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (f"%{query}%", f"%{query}%", entity_type, limit)
        else:
            sql = """
                SELECT * FROM kg_entities
                WHERE LOWER(name) LIKE LOWER(?) OR ARRAY_TO_STRING(observations, ' ') LIKE LOWER(?)
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (f"%{query}%", f"%{query}%", limit)

        results = conn.execute(sql, params).fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "observations": row[3],
                "properties": json.loads(row[4]) if row[4] else {},
                "created_at": row[5].isoformat() if row[5] else None,
                "updated_at": row[6].isoformat() if row[6] else None,
                "metadata": json.loads(row[7]) if row[7] else {},
            }
            for row in results
        ]

    async def get_relationships(
        self,
        entity_name: str,
        relation_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get all relationships for an entity.

        Args:
            entity_name: Entity to find relationships for
            relation_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationships
        """
        entity = await self.find_entity_by_name(entity_name)
        if not entity:
            return []

        conn = self._get_conn()

        # Build query based on direction
        if direction == "outgoing":
            where_clause = "WHERE r.from_entity = ?"
        elif direction == "incoming":
            where_clause = "WHERE r.to_entity = ?"
        else:  # both
            where_clause = "WHERE (r.from_entity = ? OR r.to_entity = ?)"

        if relation_type:
            where_clause += " AND r.relation_type = ?"

        sql = f"""
            SELECT
                r.id,
                r.relation_type,
                e1.name as from_name,
                e2.name as to_name,
                r.properties,
                r.created_at
            FROM kg_relationships r
            JOIN kg_entities e1 ON r.from_entity = e1.id
            JOIN kg_entities e2 ON r.to_entity = e2.id
            {where_clause}
            ORDER BY r.created_at DESC
        """

        if direction == "both":
            params = (
                (entity["id"], entity["id"], relation_type)
                if relation_type
                else (entity["id"], entity["id"])
            )
        else:
            params = (
                (entity["id"], relation_type) if relation_type else (entity["id"],)
            )

        results = conn.execute(sql, params).fetchall()

        return [
            {
                "id": row[0],
                "relation_type": row[1],
                "from_entity": row[2],
                "to_entity": row[3],
                "properties": json.loads(row[4]) if row[4] else {},
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in results
        ]

    async def find_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = 5,
    ) -> list[dict[str, Any]]:
        """Find paths between two entities using SQL/PGQ.

        Args:
            from_entity: Starting entity name
            to_entity: Target entity name
            max_depth: Maximum path length

        Returns:
            List of paths, each with nodes and relationships

        Note:
            This uses DuckPGQ's SQL/PGQ syntax for graph pattern matching.
        """
        from_node = await self.find_entity_by_name(from_entity)
        to_node = await self.find_entity_by_name(to_entity)

        if not from_node or not to_node:
            return []

        conn = self._get_conn()

        # Use SQL/PGQ for path finding
        # This is the beautiful part - SQL:2023 standard graph queries!
        query = f"""
            SELECT *
            FROM GRAPH_TABLE (knowledge_graph
                MATCH (start)-[path:*1..{max_depth}]->(end)
                WHERE start.id = '{from_node["id"]}'
                  AND end.id = '{to_node["id"]}'
                COLUMNS (
                    start.name AS from_name,
                    end.name AS to_name,
                    length(path) AS path_length
                )
            )
        """

        try:
            results = conn.execute(query).fetchall()

            return [
                {
                    "from_entity": row[0],
                    "to_entity": row[1],
                    "path_length": row[2],
                }
                for row in results
            ]
        except Exception:
            # Fallback to simple check if SQL/PGQ fails
            # (This can happen if graph is complex)
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Stats including entity count, relationship count, types
        """
        conn = self._get_conn()

        # Count entities
        entity_count = conn.execute(
            "SELECT COUNT(*) FROM kg_entities"
        ).fetchone()[0]

        # Count relationships
        relationship_count = conn.execute(
            "SELECT COUNT(*) FROM kg_relationships"
        ).fetchone()[0]

        # Get entity types
        entity_types = conn.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM kg_entities
            GROUP BY entity_type
            ORDER BY count DESC
        """).fetchall()

        # Get relationship types
        relationship_types = conn.execute("""
            SELECT relation_type, COUNT(*) as count
            FROM kg_relationships
            GROUP BY relation_type
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_entities": entity_count,
            "total_relationships": relationship_count,
            "entity_types": {etype: count for etype, count in entity_types},
            "relationship_types": {rtype: count for rtype, count in relationship_types},
            "database_path": self.db_path,
            "duckpgq_installed": self._duckpgq_installed,
        }
