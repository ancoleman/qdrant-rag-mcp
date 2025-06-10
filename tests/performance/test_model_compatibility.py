#!/usr/bin/env python3
"""Unit tests for model compatibility checking"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# We need to mock these imports since they depend on external services
with patch('qdrant_client.QdrantClient'):
    with patch('utils.logging.get_project_logger', return_value=Mock()):
        from qdrant_mcp_context_aware import (
            check_model_compatibility,
            get_collection_metadata,
            get_query_embedding_for_collection
        )


class TestModelCompatibility(unittest.TestCase):
    """Test cases for model compatibility checking"""
    
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    @patch('qdrant_mcp_context_aware.get_logger')
    def test_no_metadata_assumes_compatible(self, mock_logger, mock_get_metadata):
        """Test that collections without metadata are assumed compatible"""
        mock_get_metadata.return_value = None
        
        is_compatible, recommended_model, metadata = check_model_compatibility(
            "test_collection", "test-model"
        )
        
        self.assertTrue(is_compatible)
        self.assertEqual(recommended_model, "test-model")
        self.assertIsNone(metadata)
        
        # Check warning was logged
        mock_logger.return_value.warning.assert_called_once()
    
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    def test_no_query_model_uses_collection_model(self, mock_get_metadata):
        """Test that when no query model specified, collection model is used"""
        mock_metadata = {
            "embedding_model": "collection-model",
            "embedding_dimension": 384
        }
        mock_get_metadata.return_value = mock_metadata
        
        is_compatible, recommended_model, metadata = check_model_compatibility(
            "test_collection", None
        )
        
        self.assertTrue(is_compatible)
        self.assertEqual(recommended_model, "collection-model")
        self.assertEqual(metadata, mock_metadata)
    
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    def test_exact_model_match(self, mock_get_metadata):
        """Test that exact model matches are compatible"""
        mock_metadata = {
            "embedding_model": "same-model",
            "embedding_dimension": 384
        }
        mock_get_metadata.return_value = mock_metadata
        
        is_compatible, recommended_model, metadata = check_model_compatibility(
            "test_collection", "same-model"
        )
        
        self.assertTrue(is_compatible)
        self.assertEqual(recommended_model, "same-model")
        self.assertEqual(metadata, mock_metadata)
    
    @patch('qdrant_mcp_context_aware.get_model_registry')
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    @patch('qdrant_mcp_context_aware.get_logger')
    def test_dimension_mismatch_incompatible(self, mock_logger, mock_get_metadata, mock_get_registry):
        """Test that different dimensions are incompatible"""
        mock_metadata = {
            "embedding_model": "model-384",
            "embedding_dimension": 384
        }
        mock_get_metadata.return_value = mock_metadata
        
        # Mock registry to return different dimension
        mock_registry = Mock()
        mock_registry.get_model_dimension.return_value = 768
        mock_get_registry.return_value = mock_registry
        
        is_compatible, recommended_model, metadata = check_model_compatibility(
            "test_collection", "model-768"
        )
        
        self.assertFalse(is_compatible)
        self.assertEqual(recommended_model, "model-384")  # Recommends collection model
        self.assertEqual(metadata, mock_metadata)
        
        # Check warning was logged
        mock_logger.return_value.warning.assert_called_once()
    
    @patch('qdrant_mcp_context_aware.get_model_registry')
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    @patch('qdrant_mcp_context_aware.get_logger')
    def test_same_dimension_potentially_compatible(self, mock_logger, mock_get_metadata, mock_get_registry):
        """Test that same dimensions are potentially compatible with warning"""
        mock_metadata = {
            "embedding_model": "model-a",
            "embedding_dimension": 384
        }
        mock_get_metadata.return_value = mock_metadata
        
        # Mock registry to return same dimension
        mock_registry = Mock()
        mock_registry.get_model_dimension.return_value = 384
        mock_get_registry.return_value = mock_registry
        
        is_compatible, recommended_model, metadata = check_model_compatibility(
            "test_collection", "model-b"
        )
        
        self.assertTrue(is_compatible)
        self.assertEqual(recommended_model, "model-b")
        self.assertEqual(metadata, mock_metadata)
        
        # Check info log about potential compatibility
        mock_logger.return_value.info.assert_called_once()


class TestGetQueryEmbeddingForCollection(unittest.TestCase):
    """Test cases for get_query_embedding_for_collection"""
    
    @patch('qdrant_mcp_context_aware.get_embeddings_manager_instance')
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    def test_specialized_collection_uses_content_type(self, mock_get_metadata, mock_get_manager):
        """Test that specialized collections use the correct content type"""
        mock_metadata = {
            "specialized_embeddings": True,
            "content_type": "code",
            "embedding_model": "code-model"
        }
        mock_get_metadata.return_value = mock_metadata
        
        mock_manager = Mock()
        mock_manager.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_manager.get_model_name.return_value = "code-model"
        mock_get_manager.return_value = mock_manager
        
        result = get_query_embedding_for_collection("test query", "code_collection")
        
        # Check that encode was called with correct content type
        mock_manager.encode.assert_called_once_with("test query", content_type="code")
        self.assertEqual(result, [0.1, 0.2, 0.3])
    
    @patch('qdrant_mcp_context_aware.get_embeddings_manager_instance')
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    def test_legacy_collection_uses_general(self, mock_get_metadata, mock_get_manager):
        """Test that legacy collections use general content type"""
        mock_metadata = {
            "specialized_embeddings": False,
            "embedding_model": "old-model"
        }
        mock_get_metadata.return_value = mock_metadata
        
        mock_manager = Mock()
        mock_manager.encode.return_value.tolist.return_value = [0.4, 0.5, 0.6]
        mock_get_manager.return_value = mock_manager
        
        result = get_query_embedding_for_collection("test query", "legacy_collection")
        
        # Check that encode was called with general content type
        mock_manager.encode.assert_called_once_with("test query", content_type="general")
        self.assertEqual(result, [0.4, 0.5, 0.6])
    
    @patch('qdrant_mcp_context_aware.get_embeddings_manager_instance')
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    def test_no_metadata_uses_general(self, mock_get_metadata, mock_get_manager):
        """Test that collections without metadata use general content type"""
        mock_get_metadata.return_value = None
        
        mock_manager = Mock()
        mock_manager.encode.return_value.tolist.return_value = [0.7, 0.8, 0.9]
        mock_get_manager.return_value = mock_manager
        
        result = get_query_embedding_for_collection("test query", "unknown_collection")
        
        # Check that encode was called with general content type
        mock_manager.encode.assert_called_once_with("test query", content_type="general")
        self.assertEqual(result, [0.7, 0.8, 0.9])
    
    @patch('qdrant_mcp_context_aware.get_logger')
    @patch('qdrant_mcp_context_aware.get_embeddings_manager_instance')
    @patch('qdrant_mcp_context_aware.get_collection_metadata')
    def test_model_mismatch_warning(self, mock_get_metadata, mock_get_manager, mock_logger):
        """Test that model mismatch triggers warning"""
        mock_metadata = {
            "specialized_embeddings": True,
            "content_type": "code",
            "embedding_model": "expected-model"
        }
        mock_get_metadata.return_value = mock_metadata
        
        mock_manager = Mock()
        mock_manager.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_manager.get_model_name.return_value = "different-model"
        mock_get_manager.return_value = mock_manager
        
        result = get_query_embedding_for_collection("test query", "code_collection")
        
        # Check that warning was logged
        mock_logger.return_value.warning.assert_called_once()
        warning_message = mock_logger.return_value.warning.call_args[0][0]
        self.assertIn("Model mismatch", warning_message)
        self.assertIn("expected-model", warning_message)
        self.assertIn("different-model", warning_message)
    
    def test_uses_provided_embeddings_manager(self):
        """Test that provided embeddings manager is used"""
        mock_manager = Mock()
        mock_manager.encode.return_value.tolist.return_value = [1.0, 2.0, 3.0]
        
        with patch('qdrant_mcp_context_aware.get_collection_metadata', return_value=None):
            result = get_query_embedding_for_collection(
                "test query", 
                "collection", 
                embeddings_manager=mock_manager
            )
        
        # Should use provided manager, not create new one
        mock_manager.encode.assert_called_once_with("test query", content_type="general")
        self.assertEqual(result, [1.0, 2.0, 3.0])


if __name__ == '__main__':
    unittest.main()