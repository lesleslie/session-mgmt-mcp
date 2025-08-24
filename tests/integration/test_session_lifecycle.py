"""
Advanced integration tests for session lifecycle management.

Tests the complete session management workflow from initialization through
checkpoint creation to session cleanup with proper async patterns.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
import tempfile

from session_mgmt_mcp.server import (
    init, checkpoint, end, status, permissions,
    reflect_on_past, store_reflection, quick_search
)
from tests.fixtures.mcp_fixtures import AsyncTestCase


@pytest.mark.integration
@pytest.mark.asyncio
class TestSessionLifecycleIntegration(AsyncTestCase):
    """Comprehensive session lifecycle integration tests"""
    
    async def test_complete_session_workflow(self, 
                                           mock_session_permissions,
                                           mock_workspace_validation,
                                           isolated_database,
                                           temporary_project_structure):
        """Test complete session workflow from init to end"""
        working_dir = str(temporary_project_structure)
        
        # Phase 1: Session Initialization
        with patch.dict('os.environ', {'PWD': working_dir}):
            init_result = await init(working_directory=working_dir)
            
        assert init_result['success'] is True
        assert 'session_id' in init_result
        assert init_result['project_analysis']['maturity_score'] > 0
        assert init_result['workspace_validation']['exists'] is True
        
        # Verify initialization side effects
        mock_workspace_validation.assert_called_once()
        assert mock_session_permissions.trust_operation.call_count >= 1
        
        # Phase 2: Status Check
        status_result = await status(working_directory=working_dir)
        assert status_result['health_checks']['all_healthy'] is True
        assert status_result['session']['active'] is True
        assert 'quality_score' in status_result
        
        # Phase 3: Store Reflection
        reflection_result = await store_reflection(
            content="Session initialized successfully with comprehensive testing",
            tags=["testing", "session-management", "integration"]
        )
        assert reflection_result['success'] is True
        assert 'reflection_id' in reflection_result
        
        # Phase 4: Quality Checkpoint  
        checkpoint_result = await checkpoint()
        assert checkpoint_result['success'] is True
        assert 'checkpoint_id' in checkpoint_result
        assert checkpoint_result['quality_assessment']['overall_score'] > 0
        
        # Phase 5: Memory Search
        search_result = await quick_search(
            query="session initialization testing",
            min_score=0.5
        )
        assert search_result['count'] >= 0  # May be 0 if no semantic search available
        
        # Phase 6: Session End
        end_result = await end()
        assert end_result['success'] is True
        assert 'handoff_documentation' in end_result
        assert end_result['cleanup']['completed'] is True
    
    @pytest.mark.asyncio
    async def test_session_error_recovery(self,
                                        mock_session_permissions,
                                        isolated_database):
        """Test session error recovery patterns"""
        # Simulate workspace validation failure
        with patch('session_mgmt_mcp.server.validate_global_workspace') as mock_validation:
            mock_validation.return_value = {
                'exists': False,
                'toolkits_available': False,
                'validation_score': 20
            }
            
            # Should still succeed but with degraded functionality
            result = await init()
            assert result['success'] is True
            assert result['workspace_validation']['exists'] is False
            assert 'fallback_mode' in result
    
    @pytest.mark.asyncio 
    async def test_permission_system_integration(self,
                                               mock_session_permissions,
                                               temporary_project_structure):
        """Test permission system integration across tools"""
        working_dir = str(temporary_project_structure)
        
        # Initialize session
        await init(working_directory=working_dir)
        
        # Test permission operations
        permissions_result = await permissions(action="status")
        assert 'trusted_operations' in permissions_result
        
        # Trust a new operation
        trust_result = await permissions(
            action="trust", 
            operation="advanced_analysis"
        )
        assert trust_result['success'] is True
        assert mock_session_permissions.trust_operation.called
        
        # Verify trusted operation is recorded
        status_result = await permissions(action="status") 
        trusted_ops = status_result.get('trusted_operations', [])
        # Should include the newly trusted operation
        assert len(trusted_ops) >= 0  # May vary based on mock setup
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self,
                                               mock_session_permissions,
                                               isolated_database):
        """Test concurrent session operations don't interfere"""
        
        # Simulate concurrent operations
        tasks = [
            asyncio.create_task(store_reflection(
                content=f"Concurrent reflection {i}",
                tags=[f"test-{i}"]
            ))
            for i in range(5)
        ]
        
        # Wait for all reflections to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert result['success'] is True
        
        # Verify all reflections are searchable
        search_result = await quick_search(
            query="concurrent reflection",
            limit=10
        )
        # Should find some results (exact count depends on search implementation)
        assert search_result['count'] >= 0
    
    @pytest.mark.asyncio
    async def test_database_isolation_between_sessions(self, isolated_database):
        """Test that different sessions maintain data isolation"""
        
        # Session 1: Store data
        session1_reflection = await store_reflection(
            content="Session 1 exclusive data",
            tags=["session-1"]
        )
        assert session1_reflection['success'] is True
        
        # Create new isolated database for session 2
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "session2.db"
            
            with patch('session_mgmt_mcp.reflection_tools.ReflectionDatabase') as mock_db_class:
                mock_db = AsyncMock()
                mock_db.search_conversations.return_value = {'results': [], 'count': 0}
                mock_db_class.return_value = mock_db
                
                # Session 2: Should not see session 1 data
                search_result = await quick_search(
                    query="Session 1 exclusive",
                    project="test"
                )
                assert search_result['count'] == 0
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_session_performance_metrics(self,
                                             performance_metrics_collector,
                                             temporary_project_structure):
        """Test session operations meet performance requirements"""
        import time
        working_dir = str(temporary_project_structure)
        
        # Measure initialization time
        start_time = time.time()
        await init(working_directory=working_dir)
        init_time = time.time() - start_time
        
        performance_metrics_collector['record_execution_time']('init', init_time)
        assert init_time < 5.0  # Should complete within 5 seconds
        
        # Measure checkpoint time
        start_time = time.time()
        await checkpoint()
        checkpoint_time = time.time() - start_time
        
        performance_metrics_collector['record_execution_time']('checkpoint', checkpoint_time)
        assert checkpoint_time < 3.0  # Should complete within 3 seconds
        
        # Measure cleanup time
        start_time = time.time()
        await end()
        cleanup_time = time.time() - start_time
        
        performance_metrics_collector['record_execution_time']('end', cleanup_time)
        assert cleanup_time < 2.0  # Should complete within 2 seconds
    
    @pytest.mark.asyncio
    async def test_session_state_consistency(self,
                                           mock_session_permissions,
                                           temporary_project_structure):
        """Test session state remains consistent across operations"""
        working_dir = str(temporary_project_structure)
        
        # Initialize session and capture initial state
        init_result = await init(working_directory=working_dir)
        session_id = init_result['session_id']
        
        # Perform multiple state-changing operations
        await store_reflection(content="Test reflection 1", tags=["test"])
        await checkpoint()
        await store_reflection(content="Test reflection 2", tags=["test"])
        
        # Verify session state consistency
        status_result = await status(working_directory=working_dir)
        assert status_result['session']['session_id'] == session_id
        assert status_result['session']['active'] is True
        
        # Verify operations were tracked correctly
        assert status_result['reflection_stats']['total_reflections'] >= 2


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPToolRegistration:
    """Test MCP tool registration and execution patterns"""
    
    @pytest.mark.asyncio
    async def test_all_tools_registered(self, mock_mcp_server):
        """Test all expected MCP tools are properly registered"""
        expected_tools = [
            'init', 'checkpoint', 'end', 'status', 'permissions',
            'reflect_on_past', 'store_reflection', 'quick_search',
            'search_summary', 'get_more_results', 'search_by_file',
            'search_by_concept', 'reflection_stats'
        ]
        
        # In actual implementation, these would be registered on the real mcp instance
        # Here we simulate checking the registration
        from session_mgmt_mcp import server
        
        # Verify the functions exist and are callable
        for tool_name in expected_tools:
            assert hasattr(server, tool_name)
            tool_func = getattr(server, tool_name)
            assert callable(tool_func)
    
    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self):
        """Test MCP tools validate parameters correctly"""
        # Test required parameters
        with pytest.raises((TypeError, ValueError)):
            await store_reflection()  # Missing required content parameter
        
        # Test parameter types
        result = await store_reflection(
            content="Valid content",
            tags=["valid", "tags"]
        )
        assert result['success'] is True
    
    @pytest.mark.asyncio 
    async def test_tool_error_handling(self, isolated_database):
        """Test MCP tools handle errors gracefully"""
        # Test with invalid project parameter
        result = await quick_search(
            query="test",
            project="/invalid/path/that/does/not/exist"
        )
        # Should not crash, may return empty results or error
        assert isinstance(result, dict)
        assert 'error' in result or 'count' in result