#!/usr/bin/env python3
"""Data migration script for vector database (Phase 2.7).

Migrates conversation and reflection data from the old ReflectionDatabase schema
to the new ACB vector adapter schema.

Usage:
    python scripts/migrate_vector_database.py [--dry-run] [--backup]

Options:
    --dry-run    Show what would be migrated without making changes
    --backup     Create backup of old database before migration
    --verbose    Show detailed migration progress

Migration Process:
    1. Connects to old reflection.duckdb database
    2. Reads all conversations and reflections with embeddings
    3. Stores data in new ACB vector adapter schema (vectors.default table)
    4. Validates migration by comparing counts
    5. Optionally creates backup of old database

Safety:
    - Non-destructive: Old database remains intact
    - Validation: Compares counts before/after
    - Backup: Optional backup creation
    - Dry-run: Preview changes without applying them
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from datetime import UTC, datetime

import duckdb


async def migrate_vector_database(
    *,
    dry_run: bool = False,
    backup: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """Migrate data from old schema to new ACB vector adapter schema.

    Args:
        dry_run: If True, show what would be migrated without making changes
        backup: If True, create backup of old database before migration
        verbose: If True, show detailed progress

    Returns:
        Dictionary with migration statistics:
        - conversations_migrated: Number of conversations migrated
        - reflections_migrated: Number of reflections migrated
        - total_migrated: Total items migrated
        - errors: Number of errors encountered

    """
    from session_mgmt_mcp.di import SessionPaths, configure

    configure()  # Configure DI to get paths
    from acb.depends import depends

    paths = depends.get_sync(SessionPaths)
    old_db_path = paths.data_dir / "reflection.duckdb"

    print("=" * 70)
    print("VECTOR DATABASE MIGRATION (Phase 2.7)")
    print("=" * 70)
    print(f"\n📂 Old database: {old_db_path}")

    if not old_db_path.exists():
        print("\n⚠️  Old database not found - nothing to migrate")
        return _create_empty_migration_result()

    # Create backup if requested
    if backup and not dry_run:
        await _create_backup(old_db_path)

    # Read data from old database
    conversations, reflections = await _read_old_data(old_db_path, verbose)

    total_items = len(conversations) + len(reflections)
    if total_items == 0:
        print("\n⚠️  No data found to migrate")
        return _create_empty_migration_result()

    # Handle dry run
    if dry_run:
        return await _handle_dry_run(conversations, reflections)

    # Perform migration
    return await _perform_migration(conversations, reflections, verbose)


async def _create_backup(old_db_path):
    """Create backup of old database if requested."""
    backup_path = old_db_path.with_suffix(".duckdb.backup")
    print(f"\n💾 Creating backup: {backup_path}")
    shutil.copy2(old_db_path, backup_path)
    print("✅ Backup created")


async def _read_old_data(old_db_path, verbose: bool):
    """Read conversations and reflections from old database."""
    print("\n🔍 Reading data from old schema...")
    old_conn = duckdb.connect(str(old_db_path), read_only=True)

    try:
        # Check if old tables exist
        tables_result = old_conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        tables = {row[0] for row in tables_result}

        if verbose:
            print(f"   Found tables: {tables}")

        conversations = []
        reflections = []

        # Read conversations if table exists
        if "conversations" in tables:
            conversations = await _read_conversations(old_conn, verbose)

        # Read reflections if table exists
        if "reflections" in tables:
            reflections = await _read_reflections(old_conn, verbose)

        return conversations, reflections

    finally:
        old_conn.close()


async def _read_conversations(old_conn, verbose: bool):
    """Read conversations from old database."""
    conv_result = old_conn.execute(
        """
        SELECT id, content, embedding, project, timestamp, metadata
        FROM conversations
        ORDER BY timestamp DESC
        """
    ).fetchall()

    conversations = [
        {
            "id": row[0],
            "content": row[1],
            "embedding": row[2],
            "project": row[3],
            "timestamp": row[4],
            "metadata": row[5] or {},
        }
        for row in conv_result
    ]

    print(f"   ✅ Found {len(conversations)} conversations")
    return conversations


async def _read_reflections(old_conn, verbose: bool):
    """Read reflections from old database."""
    refl_result = old_conn.execute(
        """
        SELECT id, content, embedding, tags, timestamp
        FROM reflections
        ORDER BY timestamp DESC
        """
    ).fetchall()

    reflections = [
        {
            "id": row[0],
            "content": row[1],
            "embedding": row[2],
            "tags": row[3] or [],
            "timestamp": row[4],
        }
        for row in refl_result
    ]

    print(f"   ✅ Found {len(reflections)} reflections")
    return reflections


async def _handle_dry_run(conversations, reflections):
    """Handle dry run scenario."""
    total_items = len(conversations) + len(reflections)
    print("\n🔍 DRY RUN - No changes will be made")
    print("\nWould migrate:")
    print(f"   - {len(conversations)} conversations")
    print(f"   - {len(reflections)} reflections")
    print(f"   - {total_items} total items")
    return {
        "conversations_migrated": 0,
        "reflections_migrated": 0,
        "total_migrated": 0,
        "errors": 0,
    }


def _create_empty_migration_result():
    """Create a result dictionary for empty migration."""
    return {
        "conversations_migrated": 0,
        "reflections_migrated": 0,
        "total_migrated": 0,
        "errors": 0,
    }


async def _perform_migration(conversations, reflections, verbose: bool):
    """Perform the actual migration of data."""
    from session_mgmt_mcp.adapters.reflection_adapter import ReflectionDatabaseAdapter

    total_items = len(conversations) + len(reflections)
    print(f"\n📥 Migrating {total_items} items to new ACB schema...")

    errors = 0
    conv_migrated = 0
    refl_migrated = 0

    async with ReflectionDatabaseAdapter() as db:
        # Get direct access to ACB adapter for bulk operations
        adapter = db._get_adapter()
        await adapter.get_client()

        # Migrate conversations
        if conversations:
            conv_migrated, conv_errors = await _migrate_conversations(
                conversations, db, adapter, verbose
            )
            errors += conv_errors

        # Migrate reflections
        if reflections:
            refl_migrated, refl_errors = await _migrate_reflections(
                reflections, db, adapter, verbose
            )
            errors += refl_errors

    # Validation
    await _validate_migration()

    total_migrated = conv_migrated + refl_migrated
    await _print_migration_summary(
        conv_migrated, refl_migrated, total_migrated, total_items, errors
    )

    return {
        "conversations_migrated": conv_migrated,
        "reflections_migrated": refl_migrated,
        "total_migrated": total_migrated,
        "errors": errors,
    }


async def _migrate_conversations(conversations, db, adapter, verbose: bool):
    """Migrate conversations to the new schema."""
    print(f"\n   Migrating {len(conversations)} conversations...")
    errors = 0
    migrated = 0

    for i, conv in enumerate(conversations, 1):
        try:
            if verbose and i % 10 == 0:
                print(f"      Progress: {i}/{len(conversations)}")

            # Prepare metadata
            metadata = {
                "id": conv["id"],
                "content": conv["content"],
                "project": conv.get("project"),
                "timestamp": conv.get("timestamp", datetime.now(UTC).isoformat()),
                "type": "conversation",
                **conv.get("metadata", {}),
            }

            # Convert embedding to list if needed
            embedding = conv.get("embedding")
            if embedding is None:
                # Generate embedding if missing
                embedding = await db.get_embedding(conv["content"])

            # Store via adapter
            from acb.adapters.vector._base import VectorDocument

            doc = VectorDocument(id=conv["id"], vector=embedding, metadata=metadata)

            await adapter.insert(collection=db.collection_name, documents=[doc])
            migrated += 1

        except Exception as e:
            errors += 1
            print(f"      ❌ Error migrating conversation {conv['id'][:8]}: {e}")

    print(f"   ✅ Migrated {migrated}/{len(conversations)} conversations")
    return migrated, errors


async def _migrate_reflections(reflections, db, adapter, verbose: bool):
    """Migrate reflections to the new schema."""
    print(f"\n   Migrating {len(reflections)} reflections...")
    errors = 0
    migrated = 0

    for i, refl in enumerate(reflections, 1):
        try:
            if verbose and i % 10 == 0:
                print(f"      Progress: {i}/{len(reflections)}")

            # Prepare metadata
            metadata = {
                "id": refl["id"],
                "content": refl["content"],
                "tags": refl.get("tags", []),
                "timestamp": refl.get("timestamp", datetime.now(UTC).isoformat()),
                "type": "reflection",
            }

            # Convert embedding to list if needed
            embedding = refl.get("embedding")
            if embedding is None:
                # Generate embedding if missing
                embedding = await db.get_embedding(refl["content"])

            # Store via adapter
            from acb.adapters.vector._base import VectorDocument

            doc = VectorDocument(id=refl["id"], vector=embedding, metadata=metadata)

            await adapter.insert(collection=db.collection_name, documents=[doc])
            migrated += 1

        except Exception as e:
            errors += 1
            print(f"      ❌ Error migrating reflection {refl['id'][:8]}: {e}")

    print(f"   ✅ Migrated {migrated}/{len(reflections)} reflections")
    return migrated, errors


async def _validate_migration():
    """Validate the migration by checking database stats."""
    from session_mgmt_mcp.adapters.reflection_adapter import ReflectionDatabaseAdapter

    print("\n🔍 Validating migration...")
    async with ReflectionDatabaseAdapter() as db:
        stats = await db.get_stats()
        print("   New database stats:")
        print(f"      - Total vectors: {stats.get('total_vectors')}")
        print(f"      - Conversations: {stats.get('conversations')}")
        print(f"      - Reflections: {stats.get('reflections')}")


async def _print_migration_summary(
    conv_migrated, refl_migrated, total_migrated, total_items, errors
):
    """Print migration summary."""
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"   - Conversations migrated: {conv_migrated}/{total_migrated}")
    print(f"   - Reflections migrated: {refl_migrated}/{total_migrated}")
    print(f"   - Total migrated: {total_migrated}/{total_items}")
    if errors > 0:
        print(f"   - Errors: {errors}")

    if errors == 0 and total_migrated == total_items:
        print("\n✅ Migration successful!")
    elif errors > 0:
        print(f"\n⚠️  Migration completed with {errors} errors")
    else:
        print("\n⚠️  Migration incomplete - some items were not migrated")


def main() -> int:
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate vector database to ACB adapter schema (Phase 2.7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of old database before migration",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed migration progress"
    )

    args = parser.parse_args()

    try:
        result = asyncio.run(
            migrate_vector_database(
                dry_run=args.dry_run,
                backup=args.backup,
                verbose=args.verbose,
            )
        )

        # Return 0 if successful, 1 if errors
        return 0 if result["errors"] == 0 else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Migration cancelled by user")
        return 130
    except Exception as e:
        print(f"\n\n❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
