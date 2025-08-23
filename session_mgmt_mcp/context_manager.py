#!/usr/bin/env python3
"""
Auto-Context Loading for Session Management MCP Server

Automatically detects current development context and loads relevant conversations.
"""

import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import json
import hashlib

from .reflection_tools import ReflectionDatabase

class ContextDetector:
    """Detects current development context from environment and files"""
    
    def __init__(self):
        self.context_indicators = {
            'git': ['.git', '.gitignore', '.github'],
            'python': ['pyproject.toml', 'setup.py', 'requirements.txt', '*.py'],
            'javascript': ['package.json', 'node_modules', '*.js', '*.ts'],
            'rust': ['Cargo.toml', 'Cargo.lock', '*.rs'],
            'go': ['go.mod', 'go.sum', '*.go'],
            'java': ['pom.xml', 'build.gradle', '*.java'],
            'docker': ['Dockerfile', 'docker-compose.yml', '.dockerignore'],
            'web': ['index.html', '*.css', '*.scss'],
            'testing': ['tests/', 'test/', '*test*', 'pytest.ini'],
            'documentation': ['README.md', 'docs/', '*.md'],
            'config': ['.env', '.envrc', 'config/', '*.ini', '*.yaml', '*.yml']
        }
        
        self.project_types = {
            'mcp_server': ['mcp.json', '.mcp.json', 'fastmcp'],
            'api': ['api/', 'routes/', 'endpoints/'],
            'web_app': ['templates/', 'static/', 'public/'],
            'cli_tool': ['cli/', 'commands/', '__main__.py'],
            'library': ['src/', 'lib/', '__init__.py'],
            'data_science': ['*.ipynb', 'data/', 'notebooks/'],
            'ml_project': ['model/', 'models/', 'training/', '*.pkl'],
            'devops': ['terraform/', 'ansible/', 'k8s/', 'kubernetes/']
        }
    
    def detect_current_context(self, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Detect current development context"""
        if not working_dir:
            working_dir = os.environ.get('PWD', os.getcwd())
        
        working_path = Path(working_dir)
        context = {
            'working_directory': str(working_path),
            'project_name': working_path.name,
            'detected_languages': [],
            'detected_tools': [],
            'project_type': None,
            'current_files': [],
            'recent_files': [],
            'git_info': {},
            'confidence_score': 0.0
        }
        
        # Detect languages and tools
        for category, indicators in self.context_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator.startswith('*'):
                    # Glob pattern
                    pattern = indicator
                    matches = list(working_path.glob(pattern))
                    if matches:
                        found_indicators.extend([m.name for m in matches[:3]])  # Limit to 3
                elif indicator.endswith('/'):
                    # Directory
                    if (working_path / indicator.rstrip('/')).exists():
                        found_indicators.append(indicator)
                else:
                    # File
                    if (working_path / indicator).exists():
                        found_indicators.append(indicator)
            
            if found_indicators:
                if category in ['python', 'javascript', 'rust', 'go', 'java']:
                    context['detected_languages'].append(category)
                else:
                    context['detected_tools'].append(category)
                context['confidence_score'] += 0.1
        
        # Detect project type
        for proj_type, indicators in self.project_types.items():
            type_score = 0
            for indicator in indicators:
                if indicator.startswith('*'):
                    if list(working_path.glob(indicator)):
                        type_score += 1
                elif indicator.endswith('/'):
                    if (working_path / indicator.rstrip('/')).exists():
                        type_score += 1
                else:
                    if (working_path / indicator).exists():
                        type_score += 1
                    elif indicator in str(working_path):  # Check if it's in path name
                        type_score += 0.5
            
            if type_score > 0:
                if not context['project_type'] or type_score > context.get('_type_score', 0):
                    context['project_type'] = proj_type
                    context['_type_score'] = type_score
        
        # Get current files (recently modified)
        try:
            recent_threshold = datetime.now() - timedelta(hours=2)
            for file_path in working_path.rglob('*'):
                if file_path.is_file() and not self._should_ignore_file(file_path):
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time > recent_threshold:
                        context['recent_files'].append({
                            'path': str(file_path.relative_to(working_path)),
                            'modified': mod_time.isoformat(),
                            'size': file_path.stat().st_size
                        })
            
            # Sort by modification time
            context['recent_files'].sort(key=lambda x: x['modified'], reverse=True)
            context['recent_files'] = context['recent_files'][:10]  # Top 10
            
        except (OSError, PermissionError):
            pass
        
        # Get git information
        context['git_info'] = self._get_git_info(working_path)
        
        # Clean up temporary fields
        context.pop('_type_score', None)
        
        return context
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = {
            '.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache',
            '.mypy_cache', '.ruff_cache', 'dist', 'build', '.DS_Store'
        }
        
        # Check if any part of the path matches ignore patterns
        for part in file_path.parts:
            if part in ignore_patterns or part.startswith('.') and len(part) > 4:
                return True
        
        # Check file extensions to ignore
        ignore_extensions = {'.pyc', '.pyo', '.log', '.tmp', '.cache'}
        if file_path.suffix in ignore_extensions:
            return True
        
        return False
    
    def _get_git_info(self, working_path: Path) -> Dict[str, Any]:
        """Get git repository information"""
        git_info = {}
        
        git_dir = working_path / '.git'
        if git_dir.exists():
            try:
                # Get current branch
                head_file = git_dir / 'HEAD'
                if head_file.exists():
                    head_content = head_file.read_text().strip()
                    if head_content.startswith('ref: refs/heads/'):
                        git_info['current_branch'] = head_content.split('/')[-1]
                
                # Get remote info (simplified)
                config_file = git_dir / 'config'
                if config_file.exists():
                    config_content = config_file.read_text()
                    if 'github.com' in config_content:
                        git_info['platform'] = 'github'
                    elif 'gitlab.com' in config_content:
                        git_info['platform'] = 'gitlab'
                    else:
                        git_info['platform'] = 'git'
                
                git_info['is_git_repo'] = True
                
            except (OSError, PermissionError):
                pass
        
        return git_info

class RelevanceScorer:
    """Scores conversation relevance based on context"""
    
    def __init__(self):
        self.scoring_weights = {
            'project_name_match': 0.3,
            'language_match': 0.2,
            'tool_match': 0.15,
            'file_match': 0.15,
            'recency': 0.1,
            'keyword_match': 0.1
        }
    
    def score_conversation_relevance(self, conversation: Dict[str, Any], 
                                   context: Dict[str, Any]) -> float:
        """Score how relevant a conversation is to current context"""
        score = 0.0
        
        conv_content = conversation.get('content', '').lower()
        conv_project = conversation.get('project', '').lower()
        conv_metadata = conversation.get('metadata', {})
        
        # Project name matching
        current_project = context['project_name'].lower()
        if current_project in conv_project or current_project in conv_content:
            score += self.scoring_weights['project_name_match']
        
        # Language matching
        for lang in context['detected_languages']:
            if lang in conv_content:
                score += self.scoring_weights['language_match'] / len(context['detected_languages'])
        
        # Tool matching
        for tool in context['detected_tools']:
            if tool in conv_content:
                score += self.scoring_weights['tool_match'] / len(context['detected_tools'])
        
        # File matching
        for file_info in context['recent_files']:
            file_name = Path(file_info['path']).name.lower()
            if file_name in conv_content:
                score += self.scoring_weights['file_match'] / len(context['recent_files'])
        
        # Recency (conversations from same day get boost)
        try:
            conv_time = datetime.fromisoformat(conversation.get('timestamp', ''))
            time_diff = datetime.now() - conv_time
            if time_diff.days == 0:
                score += self.scoring_weights['recency']
            elif time_diff.days <= 7:
                score += self.scoring_weights['recency'] * 0.5
        except (ValueError, TypeError):
            pass
        
        # Project type keywords
        if context.get('project_type'):
            project_keywords = {
                'mcp_server': ['mcp', 'server', 'fastmcp', 'protocol'],
                'api': ['api', 'endpoint', 'route', 'request', 'response'],
                'web_app': ['web', 'app', 'frontend', 'backend', 'html', 'css'],
                'cli_tool': ['cli', 'command', 'argument', 'terminal'],
                'library': ['library', 'package', 'module', 'import'],
                'data_science': ['data', 'analysis', 'pandas', 'numpy', 'jupyter'],
                'ml_project': ['machine learning', 'model', 'training', 'neural'],
                'devops': ['deploy', 'infrastructure', 'docker', 'kubernetes']
            }
            
            keywords = project_keywords.get(context['project_type'], [])
            for keyword in keywords:
                if keyword in conv_content:
                    score += self.scoring_weights['keyword_match'] / len(keywords)
        
        return min(score, 1.0)  # Cap at 1.0

class AutoContextLoader:
    """Main class for automatic context loading"""
    
    def __init__(self, reflection_db: ReflectionDatabase):
        self.reflection_db = reflection_db
        self.context_detector = ContextDetector()
        self.relevance_scorer = RelevanceScorer()
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def load_relevant_context(self, working_dir: Optional[str] = None,
                                  max_conversations: int = 10,
                                  min_relevance: float = 0.3) -> Dict[str, Any]:
        """Load relevant conversations based on current context"""
        
        # Detect current context
        current_context = self.context_detector.detect_current_context(working_dir)
        
        # Generate cache key based on context
        context_hash = self._generate_context_hash(current_context)
        
        # Check cache
        if context_hash in self.cache:
            cached_time, cached_result = self.cache[context_hash]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                return cached_result
        
        # Get all conversations from database
        relevant_conversations = []
        
        if hasattr(self.reflection_db, 'conn') and self.reflection_db.conn:
            cursor = self.reflection_db.conn.execute(
                "SELECT id, content, project, timestamp, metadata FROM conversations"
            )
            conversations = cursor.fetchall()
            
            for conv in conversations:
                conv_id, content, project, timestamp, metadata = conv
                
                conversation_data = {
                    'id': conv_id,
                    'content': content,
                    'project': project,
                    'timestamp': timestamp,
                    'metadata': json.loads(metadata) if metadata else {}
                }
                
                # Score relevance
                relevance = self.relevance_scorer.score_conversation_relevance(
                    conversation_data, current_context
                )
                
                if relevance >= min_relevance:
                    conversation_data['relevance_score'] = relevance
                    relevant_conversations.append(conversation_data)
        
        # Sort by relevance and limit results
        relevant_conversations.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_conversations = relevant_conversations[:max_conversations]
        
        result = {
            'context': current_context,
            'relevant_conversations': top_conversations,
            'total_found': len(relevant_conversations),
            'loaded_count': len(top_conversations),
            'min_relevance_threshold': min_relevance
        }
        
        # Cache result
        self.cache[context_hash] = (datetime.now(), result)
        
        return result
    
    def _generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for context caching"""
        # Use key context elements for hashing
        hash_data = {
            'project_name': context['project_name'],
            'detected_languages': sorted(context['detected_languages']),
            'detected_tools': sorted(context['detected_tools']),
            'project_type': context.get('project_type'),
            'working_directory': context['working_directory']
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:12]
    
    async def get_context_summary(self, working_dir: Optional[str] = None) -> str:
        """Get a human-readable summary of current context"""
        context = self.context_detector.detect_current_context(working_dir)
        
        summary_parts = []
        summary_parts.append(f"📁 Project: {context['project_name']}")
        summary_parts.append(f"📂 Directory: {context['working_directory']}")
        
        if context['detected_languages']:
            langs = ', '.join(context['detected_languages'])
            summary_parts.append(f"💻 Languages: {langs}")
        
        if context['detected_tools']:
            tools = ', '.join(context['detected_tools'])
            summary_parts.append(f"🔧 Tools: {tools}")
        
        if context['project_type']:
            summary_parts.append(f"📋 Type: {context['project_type'].replace('_', ' ').title()}")
        
        if context['git_info'].get('is_git_repo'):
            git_info = context['git_info']
            branch = git_info.get('current_branch', 'unknown')
            platform = git_info.get('platform', 'git')
            summary_parts.append(f"🌿 Git: {branch} branch on {platform}")
        
        if context['recent_files']:
            count = len(context['recent_files'])
            summary_parts.append(f"📄 Recent files: {count} modified in last 2 hours")
        
        confidence = context['confidence_score'] * 100
        summary_parts.append(f"🎯 Detection confidence: {confidence:.0f}%")
        
        return '\n'.join(summary_parts)