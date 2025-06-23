"""
Unit tests for core HTTP handlers and utilities.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from pydantic import BaseModel

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.handlers import (
    AsyncEndpointHandler, endpoint_handler, BatchEndpointHandler,
    handle_github_auth, handle_qdrant_errors
)
from core.utils import (
    OperationLogger, MemoryManagementMixin, ModelLoadingStrategy,
    CollectionNameGenerator, get_collection_name
)


class TestRequest(BaseModel):
    """Test request model."""
    name: str
    value: int = 42


class TestAsyncEndpointHandler:
    """Test AsyncEndpointHandler class."""
    
    @pytest.mark.asyncio
    async def test_successful_sync_handler(self):
        """Test handling of successful sync function."""
        handler = AsyncEndpointHandler("test_operation")
        
        def sync_func(name: str, value: int):
            return {"result": f"{name}: {value}"}
        
        result = await handler.handle(sync_func, name="test", value=123)
        assert result == {"result": "test: 123"}
    
    @pytest.mark.asyncio
    async def test_successful_async_handler(self):
        """Test handling of successful async function."""
        handler = AsyncEndpointHandler("test_operation")
        
        async def async_func(name: str, value: int):
            await asyncio.sleep(0.01)  # Simulate async work
            return {"result": f"{name}: {value}"}
        
        result = await handler.handle(async_func, name="test", value=123)
        assert result == {"result": "test: 123"}
    
    @pytest.mark.asyncio
    async def test_error_in_result(self):
        """Test handling of error in result dict."""
        handler = AsyncEndpointHandler("test_operation")
        
        def error_func():
            return {"error": "Something went wrong"}
        
        with pytest.raises(HTTPException) as exc_info:
            await handler.handle(error_func)
        
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Something went wrong"
    
    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test handling of exceptions."""
        handler = AsyncEndpointHandler("test_operation")
        
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(HTTPException) as exc_info:
            await handler.handle(failing_func)
        
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Test error"
    
    @pytest.mark.asyncio
    async def test_with_pydantic_request(self):
        """Test handling with Pydantic request model."""
        handler = AsyncEndpointHandler("test_operation")
        
        def func_with_request(name: str, value: int):
            return {"name": name, "value": value}
        
        request = TestRequest(name="test", value=99)
        result = await handler.handle(func_with_request, request=request)
        
        assert result == {"name": "test", "value": 99}
    
    def test_sanitize_params(self):
        """Test parameter sanitization."""
        handler = AsyncEndpointHandler("test")
        
        params = {
            "name": "test",
            "password": "secret123",
            "api_key": "key123",
            "data": {"token": "token123", "value": 42},
            "long_list": list(range(20)),
            "long_string": "x" * 300
        }
        
        sanitized = handler._sanitize_params(params)
        
        assert sanitized["name"] == "test"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["data"]["token"] == "***REDACTED***"
        assert sanitized["data"]["value"] == 42
        assert sanitized["long_list"] == "[20 items]"
        assert len(sanitized["long_string"]) == 203  # 200 + "..."


class TestEndpointDecorator:
    """Test endpoint_handler decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_with_request_model(self):
        """Test decorator with Pydantic request."""
        
        @endpoint_handler("test_endpoint")
        async def test_endpoint(request: TestRequest):
            return {"echoed": request.name}
        
        request = TestRequest(name="hello")
        result = await test_endpoint(request)
        assert result == {"echoed": "hello"}
    
    @pytest.mark.asyncio
    async def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        
        @endpoint_handler("test_endpoint")
        async def test_endpoint(name: str, value: int = 10):
            return {"name": name, "value": value}
        
        result = await test_endpoint(name="test", value=20)
        assert result == {"name": "test", "value": 20}


class TestBatchEndpointHandler:
    """Test BatchEndpointHandler class."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of items."""
        handler = BatchEndpointHandler("test_batch")
        
        # Track progress
        progress_updates = []
        handler.set_progress_callback(
            lambda current, total: progress_updates.append((current, total))
        )
        
        async def process_item(item: int):
            await asyncio.sleep(0.01)
            if item == 3:
                raise ValueError("Item 3 fails")
            return {"processed": item * 2}
        
        items = list(range(5))
        result = await handler.handle_batch(
            items, process_item, batch_size=2
        )
        
        assert result["total"] == 5
        assert result["processed"] == 4  # One failed
        assert len(result["errors"]) == 1
        assert result["errors"][0]["item"] == 3
        assert len(progress_updates) > 0


class TestOperationLogger:
    """Test OperationLogger class."""
    
    def test_successful_operation(self):
        """Test logging of successful operation."""
        logger = OperationLogger("test")
        
        with logger.log_operation("sample", query="test") as result:
            result["summary"] = {"items": 10}
        
        assert result["status"] == "success"
    
    def test_failed_operation(self):
        """Test logging of failed operation."""
        logger = OperationLogger("test")
        
        with pytest.raises(ValueError):
            with logger.log_operation("sample", query="test") as result:
                raise ValueError("Test error")


class TestMemoryManagementMixin:
    """Test MemoryManagementMixin class."""
    
    def test_memory_manager_property(self):
        """Test memory manager property."""
        
        class TestClass(MemoryManagementMixin):
            pass
        
        obj = TestClass()
        # This will create the memory manager
        manager = obj.memory_manager
        assert manager is not None
        
        # Should return same instance
        assert obj.memory_manager is manager
    
    @patch('core.utils.get_memory_manager')
    def test_check_memory_pressure(self, mock_get_manager):
        """Test memory pressure checking."""
        mock_manager = Mock()
        mock_manager.get_memory_usage.return_value = {"percentage": 85}
        mock_get_manager.return_value = mock_manager
        
        class TestClass(MemoryManagementMixin):
            pass
        
        obj = TestClass()
        
        # Should trigger at 80% threshold
        assert obj.check_memory_pressure(threshold=0.8) is True
        assert obj.check_memory_pressure(threshold=0.9) is False


class TestCollectionNameGenerator:
    """Test CollectionNameGenerator class."""
    
    def test_force_global(self):
        """Test force_global flag."""
        generator = CollectionNameGenerator()
        name = generator.get_collection_name(
            file_path="/some/path", 
            file_type="code",
            force_global=True
        )
        assert name == "global_code"
    
    def test_sanitize_project_name(self):
        """Test project name sanitization."""
        generator = CollectionNameGenerator()
        
        # Test various cases
        assert generator._sanitize_project_name("my-project") == "my_project"
        assert generator._sanitize_project_name("my project") == "my_project"
        assert generator._sanitize_project_name("123project") == "project_123project"
        assert generator._sanitize_project_name("project@#$%") == "project"
        assert generator._sanitize_project_name("") == "unnamed"
        
        # Test long name
        long_name = "a" * 60
        sanitized = generator._sanitize_project_name(long_name)
        assert len(sanitized) <= 50
        assert sanitized.startswith("a" * 40)
    
    @patch('core.utils.CollectionNameGenerator.current_project', 
           new_callable=lambda: property(lambda self: {
               "name": "test-project",
               "root": "/projects/test",
               "collection_prefix": "project_test_project"
           }))
    def test_with_current_project(self, mock_project):
        """Test with current project context."""
        generator = CollectionNameGenerator()
        
        # File in current project
        name = generator.get_collection_name(
            file_path="/projects/test/src/main.py",
            file_type="code"
        )
        assert name == "project_test_project_code"
    
    def test_get_collection_name_helper(self):
        """Test get_collection_name helper function."""
        name = get_collection_name(file_type="config", force_global=True)
        assert name == "global_config"


class TestContextManagers:
    """Test utility context managers."""
    
    @pytest.mark.asyncio
    async def test_handle_github_auth_401(self):
        """Test GitHub auth error handling for 401."""
        async with pytest.raises(HTTPException) as exc_info:
            async with handle_github_auth():
                raise Exception("401 Unauthorized")
        
        assert exc_info.value.status_code == 401
        assert "authentication" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_handle_github_auth_403(self):
        """Test GitHub auth error handling for 403."""
        async with pytest.raises(HTTPException) as exc_info:
            async with handle_github_auth():
                raise Exception("403 Forbidden")
        
        assert exc_info.value.status_code == 403
        assert "rate limit" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_handle_qdrant_errors_connection(self):
        """Test Qdrant connection error handling."""
        async with pytest.raises(HTTPException) as exc_info:
            async with handle_qdrant_errors():
                raise Exception("Connection refused")
        
        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_handle_qdrant_errors_not_found(self):
        """Test Qdrant not found error handling."""
        async with pytest.raises(HTTPException) as exc_info:
            async with handle_qdrant_errors():
                raise Exception("Collection does not exist")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])