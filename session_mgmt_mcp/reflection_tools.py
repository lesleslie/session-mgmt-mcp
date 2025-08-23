#!/usr/bin/env python3
"""
Reflection Tools for Claude Session Management

Provides memory and conversation search capabilities using DuckDB and local embeddings.
"""

import asyncio
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import hashlib

# Database and embedding imports
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import numpy as np

class ReflectionDatabase:
    """Manages DuckDB database for conversation memory and reflection"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.expanduser("~/.claude/data/reflection.duckdb")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
    async def initialize(self):
        """Initialize database and embedding models"""
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not available. Install with: pip install duckdb")
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect(self.db_path)
        
        # Initialize ONNX embedding model
        if ONNX_AVAILABLE:
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                
                # Try to load ONNX model
                model_path = os.path.expanduser("~/.claude/all-MiniLM-L6-v2/onnx/model.onnx")
                if not os.path.exists(model_path):
                    print("ONNX model not found, will use text search fallback")
                    self.onnx_session = None
                else:
                    self.onnx_session = ort.InferenceSession(model_path)
                    self.embedding_dim = 384
            except Exception as e:
                print(f"ONNX model loading failed, using text search: {e}")
                self.onnx_session = None
        else:
            print("ONNX not available, using text search fallback")
        
        # Create tables if they don't exist
        await self._ensure_tables()
    
    async def _ensure_tables(self):
        """Ensure required tables exist"""
        # Create conversations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                embedding FLOAT[384],
                project VARCHAR,
                timestamp TIMESTAMP,
                metadata JSON
            )
        """)
        
        # Create reflections table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                embedding FLOAT[384],
                tags VARCHAR[],
                timestamp TIMESTAMP,
                metadata JSON
            )
        """)
        
        self.conn.commit()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using ONNX model"""
        if self.onnx_session and self.tokenizer:
            def _get_embedding():
                # Tokenize text
                encoded = self.tokenizer(text, truncation=True, padding=True, return_tensors='np')
                
                # Run inference
                outputs = self.onnx_session.run(None, {
                    'input_ids': encoded['input_ids'],
                    'attention_mask': encoded['attention_mask'],
                    'token_type_ids': encoded.get('token_type_ids', np.zeros_like(encoded['input_ids']))
                })
                
                # Mean pooling
                embeddings = outputs[0]
                attention_mask = encoded['attention_mask']
                masked_embeddings = embeddings * np.expand_dims(attention_mask, axis=-1)
                summed = np.sum(masked_embeddings, axis=1)
                counts = np.sum(attention_mask, axis=1, keepdims=True)
                mean_pooled = summed / counts
                
                # Normalize
                norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                normalized = mean_pooled / norms
                
                # Convert to float32 to match DuckDB FLOAT type
                return normalized[0].astype(np.float32).tolist()
            
            return await asyncio.get_event_loop().run_in_executor(None, _get_embedding)
        
        raise RuntimeError("No embedding model available")
    
    async def store_conversation(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store conversation with optional embedding"""
        conversation_id = hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()
        
        if ONNX_AVAILABLE and self.onnx_session:
            try:
                embedding = await self.get_embedding(content)
            except Exception:
                embedding = None  # Fallback to no embedding
        else:
            embedding = None  # Store without embedding
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.conn.execute(
                """
                INSERT INTO conversations (id, content, embedding, project, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    conversation_id,
                    content,
                    embedding,
                    metadata.get("project"),
                    datetime.now(timezone.utc),
                    json.dumps(metadata)
                ]
            )
        )
        
        self.conn.commit()
        return conversation_id
    
    async def store_reflection(self, content: str, tags: Optional[List[str]] = None) -> str:
        """Store reflection/insight with optional embedding"""
        reflection_id = hashlib.md5(f"reflection_{content}_{time.time()}".encode()).hexdigest()
        
        if ONNX_AVAILABLE and self.onnx_session:
            try:
                embedding = await self.get_embedding(content)
            except Exception:
                embedding = None  # Fallback to no embedding
        else:
            embedding = None  # Store without embedding
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.conn.execute(
                """
                INSERT INTO reflections (id, content, embedding, tags, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    reflection_id,
                    content,
                    embedding,
                    tags or [],
                    datetime.now(timezone.utc),
                    json.dumps({"type": "reflection"})
                ]
            )
        )
        
        self.conn.commit()
        return reflection_id
    
    async def search_conversations(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.7,
        project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search conversations by text similarity (fallback to text search if no embeddings)"""
        if ONNX_AVAILABLE and self.onnx_session:
            # Use semantic search with embeddings
            try:
                query_embedding = await self.get_embedding(query)
                
                sql = """
                    SELECT 
                        id, content, embedding, project, timestamp, metadata,
                        array_cosine_similarity(embedding, CAST(? AS FLOAT[384])) as score
                    FROM conversations
                    WHERE embedding IS NOT NULL
                """
                params = [query_embedding]
                
                if project:
                    sql += " AND project = ?"
                    params.append(project)
            
                sql += """
                    ORDER BY score DESC
                    LIMIT ?
                """
                params.append(limit)
                
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.conn.execute(sql, params).fetchall()
                )
                
                return [
                    {
                        "content": row[1],
                        "score": float(row[6]),
                        "timestamp": row[4],
                        "project": row[3],
                        "metadata": json.loads(row[5]) if row[5] else {}
                    }
                    for row in results
                    if float(row[6]) >= min_score
                ]
            except Exception as e:
                print(f"Semantic search failed, falling back to text search: {e}")
                # Fall through to text search
        
        # Fallback to text search (if ONNX failed or not available)
        search_terms = query.lower().split()
        sql = "SELECT id, content, project, timestamp, metadata FROM conversations"
        params = []
        
        if project:
            sql += " WHERE project = ?"
            params.append(project)
        
        sql += " ORDER BY timestamp DESC"
        
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.conn.execute(sql, params).fetchall()
        )
        
        # Simple text matching score
        matches = []
        for row in results:
            content_lower = row[1].lower()
            score = sum(1 for term in search_terms if term in content_lower) / len(search_terms)
            
            if score > 0:  # At least one term matches
                matches.append({
                    "content": row[1],
                    "score": score,
                    "timestamp": row[3],
                    "project": row[2],
                    "metadata": json.loads(row[4]) if row[4] else {}
                })
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:limit]
    
    async def search_reflections(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search stored reflections by semantic similarity"""
        if not (ONNX_AVAILABLE and self.onnx_session):
            return []  # No semantic search available
            
        query_embedding = await self.get_embedding(query)
        
        sql = """
            SELECT 
                id, content, embedding, tags, timestamp, metadata,
                array_cosine_similarity(embedding, CAST(? AS FLOAT[384])) as score
            FROM reflections
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.conn.execute(sql, [query_embedding, limit]).fetchall()
        )
        
        return [
            {
                "content": row[1],
                "score": float(row[6]),
                "tags": row[3] if row[3] else [],
                "timestamp": row[4],
                "metadata": json.loads(row[5]) if row[5] else {}
            }
            for row in results
            if float(row[6]) >= min_score
        ]
    
    async def search_by_file(
        self,
        file_path: str,
        limit: int = 10,
        project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search conversations that mention a specific file"""
        sql = """
            SELECT id, content, project, timestamp, metadata
            FROM conversations
            WHERE content LIKE ?
        """
        params = [f"%{file_path}%"]
        
        if project:
            sql += " AND project = ?"
            params.append(project)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.conn.execute(sql, params).fetchall()
        )
        
        return [
            {
                "content": row[1],
                "project": row[2],
                "timestamp": row[3],
                "metadata": json.loads(row[4]) if row[4] else {}
            }
            for row in results
        ]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conv_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            )
            
            refl_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
            )
            
            provider = "onnx-runtime" if (self.onnx_session and ONNX_AVAILABLE) else "text-search-only"
            return {
                "conversations_count": conv_count,
                "reflections_count": refl_count,
                "embedding_provider": provider,
                "embedding_dimension": self.embedding_dim,
                "database_path": str(self.db_path)
            }
        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}

# Global database instance
_reflection_db: Optional[ReflectionDatabase] = None

async def get_reflection_database() -> ReflectionDatabase:
    """Get or create reflection database instance"""
    global _reflection_db
    if _reflection_db is None:
        _reflection_db = ReflectionDatabase()
        await _reflection_db.initialize()
    return _reflection_db

def get_current_project() -> Optional[str]:
    """Get current project name from working directory"""
    try:
        cwd = Path.cwd()
        # Try to detect project from common indicators
        if (cwd / "pyproject.toml").exists() or (cwd / "package.json").exists():
            return cwd.name
        # Fallback to directory name
        return cwd.name if cwd.name != "." else None
    except Exception:
        return None