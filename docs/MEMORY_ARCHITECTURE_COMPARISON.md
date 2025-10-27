# Memory Architecture Comparison

**Date**: 2025-10-26

---

## Executive Summary

Session-mgmt-mcp now implements **dual memory** system:

| System | Type | Database | Purpose |
|--------|------|----------|---------|
| **ReflectionDatabase** | Episodic | reflection.duckdb | Full conversation history with vector search |
| **KnowledgeGraphDatabase** | Semantic | knowledge_graph.duckdb | Structured entities and relationships |

---

## Memory Systems Explained

### Episodic Memory (Existing)

**What it stores**: Full conversations with context

**How it works**:
- Stores complete conversation text
- Generates 384-dim embeddings (ONNX all-MiniLM-L6-v2)
- Vector similarity search
- Time-decay prioritization

**Example query**:
> "How did I fix the authentication bug?"

Returns: Full conversation with implementation details

### Semantic Memory (New - Knowledge Graph)

**What it stores**: Structured facts and relationships

**How it works**:
- Entities (nodes): Projects, libraries, technologies
- Relations (edges): "uses", "depends_on", "developed_by"
- Observations: Facts attached to entities
- Graph queries via SQL/PGQ

**Example query**:
> "Which projects use Python 3.13?"

Returns: List of project entities with "uses" relationship to Python

---

## Why Both?

Think of it like your brain:

1. **Episodic Memory** = "I remember when..."
   - Full context of experiences
   - How things happened
   - Problem-solving processes

2. **Semantic Memory** = "I know that..."
   - Factual knowledge
   - Relationships between concepts
   - Instant recall of facts

**Combined example**:

**Episodic**: "How did I integrate ACB?"
→ Returns: Full conversation with step-by-step implementation

**Semantic**: "What does session-mgmt-mcp use?"
→ Returns: Graph showing dependencies (ACB, DuckDB, FastMCP, etc.)

---

## Comparison with External Servers

### Option 1: External `@modelcontextprotocol/server-memory`

**Security**: 🔴 45/100 (command injection risk)
**Installation**: External Node.js process
**Integration**: Loose (separate MCP server)
**Verdict**: ❌ **Security concerns**

### Option 2: External `mcp-knowledge-graph`

**Security**: ✅ 72/100 (safety markers)
**Installation**: External Node.js process
**Integration**: Loose (separate MCP server)
**Verdict**: ⚠️ **Better, but still external**

### Option 3: DuckPGQ (Chosen Solution)

**Security**: ✅ 100/100 (local, no external deps)
**Installation**: ❌ **None!** (DuckDB already installed)
**Integration**: ✅ **Tight** (same codebase, DI)
**Verdict**: ✅ **BEST** - Embedded, secure, SQL standard

---

## Architecture Benefits

### Unified Codebase
- Everything in Python
- Single dependency injection system
- Consistent testing patterns
- Same configuration system (ACB Settings)

### No External Processes
- No Node.js runtime needed
- No separate MCP servers
- No network communication overhead
- Simpler deployment

### SQL Standard
- SQL/PGQ is official SQL:2023
- Future-proof
- Familiar syntax for database users
- DuckDB performance optimizations

### ACB Integration Ready
- Can use ACB graph adapters later (Neo4j, ArangoDB)
- Just change configuration
- Same code, different backend
- Production scaling path available

---

## Data Flow

### Storing a Conversation with Auto-Extraction

```
1. User stores conversation via MCP tool
   ↓
2. ReflectionDatabase stores full text + embedding
   ↓
3. EntityExtractor analyzes text
   ↓
4. Extracts: ["session-mgmt-mcp", "DuckPGQ", "ACB"]
   ↓
5. KnowledgeGraphDatabase creates entities
   ↓
6. Creates relations: session-mgmt-mcp --uses--> DuckPGQ
```

### Searching Across Both Systems

```
1. User searches: "How do I use DuckPGQ?"
   ↓
2. ReflectionDatabase: Vector search for conversations
   → Returns: Full implementation discussion
   ↓
3. KnowledgeGraphDatabase: Find entity "DuckPGQ"
   → Returns: Projects using it, relationships
   ↓
4. Combined result: Conversations + structured knowledge
```

---

## Migration Path

### Current State (Before)

```
External Memory Server (risky)
  ├── memory.jsonl files
  └── External Node.js process
```

### New State (After Migration)

```
Session-Mgmt-MCP (unified)
  ├── reflection.duckdb (conversations)
  └── knowledge_graph.duckdb (entities from migrated data)
```

---

## Future Enhancements

### Potential ACB Integration

Once the DuckPGQ implementation is stable:

```yaml
# Could later switch to production graph DB via ACB
# settings/adapters.yaml
graph: neo4j  # or arangodb

# settings/session-mgmt.yaml
graph:
  host: "bolt://localhost"
  port: 7687
```

**Code changes needed**: ❌ **ZERO!**
- KnowledgeGraphDatabase becomes ACB facade
- Same MCP tools work
- Just configuration change

---

## Summary

**Chosen Architecture**: DuckPGQ embedded graph database

**Advantages**:
1. ✅ Zero installation (DuckDB already present)
2. ✅ No external processes
3. ✅ 100% security (local, no network)
4. ✅ SQL/PGQ standard (SQL:2023)
5. ✅ ACB integration path available
6. ✅ Unified Python codebase

**Result**: Best of both worlds - simple embedded solution now, production scaling path later.
