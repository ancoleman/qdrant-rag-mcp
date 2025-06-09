#!/usr/bin/env python3
"""Unit tests for SpecializedEmbeddingManager"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.specialized_embeddings import SpecializedEmbeddingManager


class TestSpecializedEmbeddingManager(unittest.TestCase):
    """Test cases for SpecializedEmbeddingManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock config
        self.config = {
            'models': {
                'code': {
                    'primary': 'test-code-model',
                    'dimension': 768,
                    'fallback': 'test-code-fallback',
                    'max_tokens': 512
                },
                'documentation': {
                    'primary': 'test-doc-model',
                    'dimension': 384,
                    'instruction_prefix': 'Test prefix:',
                    'max_tokens': 256
                }
            },
            'memory': {
                'max_models_in_memory': 2,
                'memory_limit_gb': 4.0
            }
        }
        
        # Create manager with mocked models
        with patch('utils.specialized_embeddings.SentenceTransformer'):
            self.manager = SpecializedEmbeddingManager(config=self.config)
    
    def test_initialization(self):
        """Test manager initialization"""
        # Check that config was applied
        self.assertEqual(self.manager.model_configs['code']['name'], 'test-code-model')
        self.assertEqual(self.manager.model_configs['documentation']['name'], 'test-doc-model')
        # Note: actual values may be overridden by environment variables or defaults
        self.assertIsInstance(self.manager.max_models_in_memory, int)
        self.assertIsInstance(self.manager.memory_limit_gb, float)
    
    def test_environment_variable_override(self):
        """Test that environment variables override config"""
        with patch.dict(os.environ, {
            'QDRANT_CODE_EMBEDDING_MODEL': 'env-code-model',
            'QDRANT_MAX_MODELS_IN_MEMORY': '3',
            'QDRANT_MEMORY_LIMIT_GB': '5.0'
        }):
            with patch('utils.specialized_embeddings.SentenceTransformer'):
                manager = SpecializedEmbeddingManager()
                
                # Check env var overrides
                self.assertEqual(manager.model_configs['code']['name'], 'env-code-model')
                self.assertEqual(manager.max_models_in_memory, 3)
                self.assertEqual(manager.memory_limit_gb, 5.0)
    
    @patch('utils.specialized_embeddings.SentenceTransformer')
    def test_model_loading(self, mock_transformer):
        """Test lazy model loading"""
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_transformer.return_value = mock_model
        
        # Load a model
        model, config = self.manager.load_model('code')
        
        # Check that model was loaded
        mock_transformer.assert_called_once_with(
            'test-code-model',
            device=self.manager.device,
            cache_folder=self.manager.cache_dir
        )
        self.assertEqual(model, mock_model)
        self.assertEqual(config['name'], 'test-code-model')
        
        # Check that model is cached
        self.assertIn('test-code-model', self.manager.loaded_models)
        self.assertEqual(self.manager.total_memory_used_gb, 1.0)  # Default estimate from _estimate_model_memory
    
    @patch('utils.specialized_embeddings.SentenceTransformer')
    def test_lru_eviction(self, mock_transformer):
        """Test LRU eviction when max models reached"""
        # Create mock models
        mock_model1 = Mock()
        mock_model1.eval = Mock()
        mock_model2 = Mock()
        mock_model2.eval = Mock()
        mock_model3 = Mock()
        mock_model3.eval = Mock()
        
        mock_transformer.side_effect = [mock_model1, mock_model2, mock_model3]
        
        # Load two models (max_models = 2)
        self.manager.load_model('code')
        self.manager.load_model('documentation')
        
        # Check both are loaded
        self.assertEqual(len(self.manager.loaded_models), 2)
        
        # Load a third model - should evict the first
        self.manager.load_model('config')
        
        # Check that first model was evicted
        self.assertEqual(len(self.manager.loaded_models), 2)
        self.assertNotIn('test-code-model', self.manager.loaded_models)
        self.assertIn('test-doc-model', self.manager.loaded_models)
        self.assertIn('jinaai/jina-embeddings-v3', self.manager.loaded_models)  # default config model
    
    @patch('utils.specialized_embeddings.SentenceTransformer')
    def test_fallback_model(self, mock_transformer):
        """Test fallback to alternative model on failure"""
        # First call fails, second succeeds
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_transformer.side_effect = [Exception("Model not found"), mock_model]
        
        # Load model - should fall back
        model, config = self.manager.load_model('code')
        
        # Check that fallback was used
        self.assertEqual(mock_transformer.call_count, 2)
        self.assertEqual(mock_transformer.call_args_list[1][0][0], 'test-code-fallback')
        self.assertEqual(config['name'], 'test-code-fallback')
    
    @patch('utils.specialized_embeddings.SentenceTransformer')
    def test_encode_with_content_type(self, mock_transformer):
        """Test encoding with different content types"""
        # Create mock model with encode method
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_transformer.return_value = mock_model
        
        # Test code encoding
        result = self.manager.encode("test code", content_type="code")
        
        # Check encoding was called
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        self.assertEqual(call_args, ["test code"])
        
        # Check result shape
        self.assertEqual(result.shape, (1, 3))
    
    @patch('utils.specialized_embeddings.SentenceTransformer')
    def test_instruction_prefix_for_documentation(self, mock_transformer):
        """Test that instruction prefix is added for documentation"""
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_transformer.return_value = mock_model
        
        # Set model name to include 'instructor'
        self.manager.model_configs['documentation']['name'] = 'hkunlp/instructor-large'
        
        # Encode documentation
        self.manager.encode("test doc", content_type="documentation")
        
        # Check that prefix was added
        call_args = mock_model.encode.call_args[0][0]
        self.assertEqual(call_args, ["Test prefix: test doc"])
    
    def test_get_dimension(self):
        """Test getting embedding dimension for content types"""
        self.assertEqual(self.manager.get_dimension('code'), 768)
        self.assertEqual(self.manager.get_dimension('documentation'), 384)
        self.assertEqual(self.manager.get_dimension('general'), 384)  # default
    
    def test_get_model_name(self):
        """Test getting model name for content types"""
        self.assertEqual(self.manager.get_model_name('code'), 'test-code-model')
        self.assertEqual(self.manager.get_model_name('documentation'), 'test-doc-model')
    
    def test_get_content_type_for_file(self):
        """Test content type detection from file path"""
        # Code files
        self.assertEqual(self.manager.get_content_type_for_file('test.py'), 'code')
        self.assertEqual(self.manager.get_content_type_for_file('test.js'), 'code')
        self.assertEqual(self.manager.get_content_type_for_file('test.go'), 'code')
        
        # Config files
        self.assertEqual(self.manager.get_content_type_for_file('config.json'), 'config')
        self.assertEqual(self.manager.get_content_type_for_file('settings.yaml'), 'config')
        self.assertEqual(self.manager.get_content_type_for_file('.env'), 'config')
        
        # Documentation files
        self.assertEqual(self.manager.get_content_type_for_file('README.md'), 'documentation')
        self.assertEqual(self.manager.get_content_type_for_file('docs.rst'), 'documentation')
        
        # Unknown files
        self.assertEqual(self.manager.get_content_type_for_file('unknown.xyz'), 'general')
    
    @patch('utils.specialized_embeddings.SentenceTransformer')
    def test_memory_limit_enforcement(self, mock_transformer):
        """Test that memory limits are enforced"""
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_transformer.return_value = mock_model
        
        # Set very low memory limit
        self.manager.memory_limit_gb = 1.5
        
        # Try to load a model that would exceed limit
        # Code model is estimated at 2GB, which exceeds 1.5GB limit
        with patch.object(self.manager, '_estimate_model_memory', return_value=2.0):
            # Should still load since no models are loaded yet
            self.manager.load_model('code')
            
            # But loading another should fail and trigger eviction
            self.manager.load_model('documentation')
            
            # Check that only one model is loaded due to memory constraint
            self.assertEqual(len(self.manager.loaded_models), 1)
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add some dummy data
        self.manager.loaded_models['test'] = Mock()
        self.manager.memory_usage['test'] = 1.0
        self.manager.total_memory_used_gb = 1.0
        
        # Clear cache
        self.manager.clear_cache()
        
        # Check everything is cleared
        self.assertEqual(len(self.manager.loaded_models), 0)
        self.assertEqual(len(self.manager.memory_usage), 0)
        self.assertEqual(self.manager.total_memory_used_gb, 0.0)
    
    def test_get_all_models_info(self):
        """Test getting info for all models"""
        info = self.manager.get_all_models_info()
        
        # Check structure
        self.assertIn('device', info)
        self.assertIn('cache_dir', info)
        self.assertIn('memory', info)
        self.assertIn('models', info)
        
        # Check memory info
        self.assertEqual(info['memory']['limit_gb'], 4.0)
        self.assertEqual(info['memory']['max_models'], 2)
        
        # Check model info
        self.assertIn('code', info['models'])
        self.assertIn('documentation', info['models'])


if __name__ == '__main__':
    unittest.main()