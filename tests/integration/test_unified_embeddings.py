#!/usr/bin/env python3
"""Unit tests for UnifiedEmbeddingsManager"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.embeddings import UnifiedEmbeddingsManager, should_use_specialized_embeddings


class TestUnifiedEmbeddingsManager(unittest.TestCase):
    """Test cases for UnifiedEmbeddingsManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'specialized_embeddings': {
                'enabled': True,
                'models': {
                    'code': {'primary': 'test-code-model'},
                    'documentation': {'primary': 'test-doc-model'}
                }
            },
            'embeddings': {
                'model': 'test-single-model',
                'dimension': 384
            }
        }
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch('utils.embeddings.get_specialized_embedding_manager')
    def test_specialized_mode_initialization(self, mock_get_specialized):
        """Test initialization in specialized embeddings mode"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_specialized_manager = Mock()
        mock_get_specialized.return_value = mock_specialized_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        
        # Check that specialized mode is active
        self.assertTrue(manager.use_specialized)
        self.assertEqual(manager.manager, mock_specialized_manager)
        mock_get_specialized.assert_called_once_with(self.config['specialized_embeddings'])
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', False)
    @patch('utils.embeddings.EmbeddingsManager')
    def test_single_mode_initialization(self, mock_embeddings_manager):
        """Test initialization in single model mode when specialized not available"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_single_manager = Mock()
        mock_embeddings_manager.return_value = mock_single_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        
        # Check that single mode is active
        self.assertFalse(manager.use_specialized)
        self.assertEqual(manager.manager, mock_single_manager)
        mock_embeddings_manager.assert_called_once()
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch.dict(os.environ, {'QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED': 'false'})
    @patch('utils.embeddings.EmbeddingsManager')
    def test_env_var_disables_specialized(self, mock_embeddings_manager):
        """Test that environment variable can disable specialized embeddings"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_single_manager = Mock()
        mock_embeddings_manager.return_value = mock_single_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        
        # Should use single mode despite config enabling specialized
        self.assertFalse(manager.use_specialized)
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch('utils.embeddings.get_specialized_embedding_manager')
    def test_encode_with_content_type(self, mock_get_specialized):
        """Test encoding with content type in specialized mode"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_specialized_manager = Mock()
        mock_specialized_manager.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_get_specialized.return_value = mock_specialized_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        result = manager.encode("test text", content_type="code")
        
        # Check that specialized manager was used with content type
        mock_specialized_manager.encode.assert_called_once_with(
            "test text", content_type="code"
        )
        self.assertEqual(result.shape, (1, 3))
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch('utils.embeddings.get_specialized_embedding_manager')
    def test_encode_without_content_type_defaults_to_general(self, mock_get_specialized):
        """Test that encoding without content type defaults to general"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_specialized_manager = Mock()
        mock_specialized_manager.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_get_specialized.return_value = mock_specialized_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        result = manager.encode("test text")
        
        # Should default to general content type
        mock_specialized_manager.encode.assert_called_once_with(
            "test text", content_type="general"
        )
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', False)
    @patch('utils.embeddings.EmbeddingsManager')
    def test_encode_in_single_mode(self, mock_embeddings_manager):
        """Test encoding in single model mode"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_single_manager = Mock()
        # The UnifiedEmbeddingsManager checks for encode_batch first
        mock_single_manager.encode_batch = Mock(return_value=[[0.4, 0.5, 0.6]])
        mock_embeddings_manager.return_value = mock_single_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        result = manager.encode("test text", content_type="code")  # content_type ignored
        
        # Check that single manager's encode_batch was used
        mock_single_manager.encode_batch.assert_called_once_with(["test text"])
        self.assertEqual(result.shape, (1, 3))
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch('utils.embeddings.get_specialized_embedding_manager')
    def test_get_dimension_specialized(self, mock_get_specialized):
        """Test getting dimension in specialized mode"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_specialized_manager = Mock()
        mock_specialized_manager.get_dimension = Mock(return_value=768)
        mock_get_specialized.return_value = mock_specialized_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        
        # Test with content type
        dim = manager.get_dimension("code")
        mock_specialized_manager.get_dimension.assert_called_with("code")
        self.assertEqual(dim, 768)
        
        # Test without content type - should default to general
        dim = manager.get_dimension()
        mock_specialized_manager.get_dimension.assert_called_with("general")
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', False)
    @patch('utils.embeddings.EmbeddingsManager')
    def test_get_dimension_single(self, mock_embeddings_manager):
        """Test getting dimension in single mode"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_single_manager = Mock()
        mock_single_manager.dimension = 384
        mock_embeddings_manager.return_value = mock_single_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        dim = manager.get_dimension("code")  # content_type ignored
        
        self.assertEqual(dim, 384)
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch('utils.embeddings.get_specialized_embedding_manager')
    def test_backward_compatibility_methods(self, mock_get_specialized):
        """Test backward compatibility methods"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_specialized_manager = Mock()
        mock_specialized_manager.get_dimension = Mock(return_value=512)
        mock_specialized_manager.get_model_name = Mock(return_value="test-model")
        mock_get_specialized.return_value = mock_specialized_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        
        # Test get_sentence_embedding_dimension
        dim = manager.get_sentence_embedding_dimension()
        self.assertEqual(dim, 512)
        
        # Test dimension property
        dim = manager.dimension
        self.assertEqual(dim, 512)
        
        # Test model_name property
        name = manager.model_name
        self.assertEqual(name, "test-model")
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch('utils.embeddings.get_specialized_embedding_manager')
    def test_get_model_info(self, mock_get_specialized):
        """Test getting model info"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        mock_specialized_manager = Mock()
        mock_specialized_manager.get_model_info = Mock(return_value={"info": "test"})
        mock_specialized_manager.get_all_models_info = Mock(return_value={"all": "info"})
        mock_get_specialized.return_value = mock_specialized_manager
        
        manager = UnifiedEmbeddingsManager(self.config)
        
        # Test with content type
        info = manager.get_model_info("code")
        mock_specialized_manager.get_model_info.assert_called_with("code")
        self.assertEqual(info, {"info": "test"})
        
        # Test without content type
        info = manager.get_model_info()
        mock_specialized_manager.get_all_models_info.assert_called_once()
        self.assertEqual(info, {"all": "info"})


class TestShouldUseSpecializedEmbeddings(unittest.TestCase):
    """Test cases for should_use_specialized_embeddings function"""
    
    def setUp(self):
        """Reset global state before each test"""
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
    def tearDown(self):
        """Clean up after each test"""
        import utils.embeddings
        utils.embeddings._use_specialized = None
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    def test_default_enabled_when_available(self):
        """Test that specialized embeddings are enabled by default when available"""
        self.assertTrue(should_use_specialized_embeddings())
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', False)
    def test_disabled_when_not_available(self):
        """Test that specialized embeddings are disabled when not available"""
        self.assertFalse(should_use_specialized_embeddings())
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    @patch.dict(os.environ, {'QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED': 'false'})
    def test_env_var_override(self):
        """Test that environment variable can override default"""
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        self.assertFalse(should_use_specialized_embeddings())
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    def test_config_override(self):
        """Test that config can override default"""
        config = {'specialized_embeddings': {'enabled': False}}
        
        # Clear cached value
        import utils.embeddings
        utils.embeddings._use_specialized = None
        
        self.assertFalse(should_use_specialized_embeddings(config))
    
    @patch('utils.embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE', True)
    def test_caching(self):
        """Test that the decision is cached"""
        # First call
        result1 = should_use_specialized_embeddings()
        
        # Change environment (shouldn't affect cached result)
        with patch.dict(os.environ, {'QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED': 'false'}):
            result2 = should_use_specialized_embeddings()
        
        # Both should be the same due to caching
        self.assertEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()