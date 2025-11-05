"""Migration script for knowledge graph database to ACB adapter.

This script migrates data from the original KnowledgeGraphDatabase to the new
KnowledgeGraphDatabaseAdapter that uses ACB configuration.

Usage:
    # Dry run (preview changes without modifying data)
    python scripts/migrate_graph_database.py --dry-run

    # Create backup before migration
    python scripts/migrate_graph_database.py --backup

    # Verbose output
    python scripts/migrate_graph_database.py --verbose

    # Combination
    python scripts/migrate_graph_database.py --backup --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path.home() / ".claude" / "data"


def get_old_db_path() -> Path:
    """Get path to old knowledge graph database."""
    return get_data_dir() / "knowledge_graph.duckdb"


def get_new_db_path() -> Path:
    """Get path for new ACB-managed database."""
    return get_data_dir() / "knowledge_graph_acb.duckdb"


def get_backup_path() -> Path:
    """Get timestamped backup path."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return get_data_dir() / f"knowledge_graph_backup_{timestamp}.duckdb"


async def migrate_graph_database(
    *,
    dry_run: bool = False,
    backup: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """Migrate knowledge graph data from old schema to new ACB adapter.

    Args:
        dry_run: If True, preview changes without modifying data
        backup: If True, create backup before migration
        verbose: If True, print detailed progress information

    Returns:
        Dictionary with migration statistics

    Raises:
        FileNotFoundError: If old database doesn't exist
        RuntimeError: If migration fails
    """
    import duckdb

    from session_mgmt_mcp.adapters.knowledge_graph_adapter import (
        KnowledgeGraphDatabaseAdapter,
    )

    old_db_path = get_old_db_path()
    new_db_path = get_new_db_path()

    # Check if old database exists
    if not old_db_path.exists():
        msg = f"Old database not found at {old_db_path}"
        raise FileNotFoundError(msg)

    if verbose:
        print(f"📊 Migration Configuration:")
        print(f"  Old DB: {old_db_path}")
        print(f"  New DB: {new_db_path}")
        print(f"  Dry Run: {dry_run}")
        print(f"  Backup: {backup}")
        print()

    # Create backup if requested
    if backup and not dry_run:
        backup_path = get_backup_path()
        if verbose:
            print(f"💾 Creating backup at {backup_path}...")
        shutil.copy2(old_db_path, backup_path)
        if verbose:
            print(f"✅ Backup created successfully")
            print()

    # Connect to old database (read-only)
    if verbose:
        print("📖 Reading from old database...")

    old_conn = duckdb.connect(str(old_db_path), read_only=True)

    # Get table list
    tables_result = old_conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN ('kg_entities', 'kg_relationships')
    """
    ).fetchall()
    tables = [row[0] for row in tables_result]

    if verbose:
        print(f"  Found tables: {', '.join(tables)}")

    # Read entities
    entities = []
    if "kg_entities" in tables:
        entity_result = old_conn.execute("SELECT * FROM kg_entities").fetchall()
        for row in entity_result:
            entities.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "entity_type": row[2],
                    "observations": list(row[3]) if row[3] else [],
                    "properties": row[4] if row[4] else {},
                    "created_at": row[5],
                    "updated_at": row[6],
                    "metadata": row[7] if row[7] else {},
                }
            )

    if verbose:
        print(f"  Read {len(entities)} entities")

    # Read relationships
    relationships = []
    if "kg_relationships" in tables:
        rel_result = old_conn.execute("SELECT * FROM kg_relationships").fetchall()
        for row in rel_result:
            relationships.append(
                {
                    "id": row[0],
                    "from_entity": row[1],
                    "to_entity": row[2],
                    "relation_type": row[3],
                    "properties": row[4] if row[4] else {},
                    "created_at": row[5],
                    "updated_at": row[6],
                    "metadata": row[7] if row[7] else {},
                }
            )

    if verbose:
        print(f"  Read {len(relationships)} relationships")
        print()

    old_conn.close()

    # If dry run, just report what would be migrated
    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
        print()
        print("Would migrate:")
        print(f"  📦 {len(entities)} entities")
        print(f"  🔗 {len(relationships)} relationships")
        print()
        return {"entities": len(entities), "relationships": len(relationships)}

    # Write to new database using adapter
    if verbose:
        print("✍️  Writing to new ACB-managed database...")

    async with KnowledgeGraphDatabaseAdapter(db_path=new_db_path) as new_db:
        # Migrate entities (preserve IDs and timestamps)
        if verbose:
            print(f"  Migrating {len(entities)} entities...")

        for entity in entities:
            # Direct insert to preserve IDs and timestamps
            conn = new_db._get_conn()
            conn.execute(
                """
                INSERT INTO kg_entities
                (id, name, entity_type, observations, properties, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity["id"],
                    entity["name"],
                    entity["entity_type"],
                    entity["observations"],
                    entity["properties"],
                    entity["created_at"],
                    entity["updated_at"],
                    entity["metadata"],
                ),
            )

        if verbose:
            print(f"  ✅ Migrated {len(entities)} entities")

        # Migrate relationships (preserve IDs and timestamps)
        if verbose:
            print(f"  Migrating {len(relationships)} relationships...")

        for rel in relationships:
            conn = new_db._get_conn()
            conn.execute(
                """
                INSERT INTO kg_relationships
                (id, from_entity, to_entity, relation_type, properties, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel["id"],
                    rel["from_entity"],
                    rel["to_entity"],
                    rel["relation_type"],
                    rel["properties"],
                    rel["created_at"],
                    rel["updated_at"],
                    rel["metadata"],
                ),
            )

        if verbose:
            print(f"  ✅ Migrated {len(relationships)} relationships")

    # Validate migration
    if verbose:
        print()
        print("🔍 Validating migration...")

    async with KnowledgeGraphDatabaseAdapter(db_path=new_db_path) as new_db:
        stats = await new_db.get_stats()

        # Compare counts
        entities_match = stats["total_entities"] == len(entities)
        relationships_match = stats["total_relationships"] == len(relationships)

        if verbose:
            print(f"  Entities: {stats['total_entities']} (expected {len(entities)}) "
                  f"{'✅' if entities_match else '❌'}")
            print(
                f"  Relationships: {stats['total_relationships']} (expected {len(relationships)}) "
                f"{'✅' if relationships_match else '❌'}"
            )

        if not (entities_match and relationships_match):
            msg = "Migration validation failed - record counts don't match"
            raise RuntimeError(msg)

    if verbose:
        print()
        print("✅ Migration completed successfully!")
        print()

    return {
        "entities_migrated": len(entities),
        "relationships_migrated": len(relationships),
        "total_records": len(entities) + len(relationships),
    }


def main() -> None:
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate knowledge graph database to ACB adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration without making changes
  python scripts/migrate_graph_database.py --dry-run

  # Migrate with backup
  python scripts/migrate_graph_database.py --backup --verbose

  # Just migrate (no backup)
  python scripts/migrate_graph_database.py
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying data",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before migration"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed progress information"
    )

    args = parser.parse_args()

    try:
        result = asyncio.run(
            migrate_graph_database(
                dry_run=args.dry_run,
                backup=args.backup,
                verbose=args.verbose,
            )
        )

        if not args.dry_run:
            print()
            print("Migration Summary:")
            print(f"  Entities: {result['entities_migrated']} migrated")
            print(f"  Relationships: {result['relationships_migrated']} migrated")
            print(f"  Total: {result['total_records']} records")
            print()
            print(f"✅ Migration complete!")
            print()
            print(f"Old database: {get_old_db_path()}")
            print(f"New database: {get_new_db_path()}")
            if args.backup:
                print(f"Backup: {get_backup_path()}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print()
        print("No migration needed - old database doesn't exist.")
        sys.exit(0)  # Not an error, just nothing to migrate

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
