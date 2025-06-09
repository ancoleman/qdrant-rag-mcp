"""
Model Registry for managing embedding models and their metadata

This module provides a central registry for managing embedding models,
their configurations, and the mapping between content types and models.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from .logging import get_project_logger

logger = get_project_logger()


class ModelRegistry:
    """Central registry for managing embedding models and their configurations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model registry
        
        Args:
            config: Optional configuration dictionary
        """
        # Model configurations by content type
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        
        # Collection to model mapping
        self.collection_model_mapping: Dict[str, str] = {}
        
        # Model metadata (dimensions, capabilities, etc.)
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Registry file for persistence
        self.registry_file = os.path.expanduser("~/.mcp-servers/qdrant-rag/model_registry.json")
        
        # Initialize with default models
        self._initialize_defaults()
        
        # Load persisted registry if exists
        self._load_registry()
        
        # Apply custom configuration
        if config:
            self._apply_config(config)
        
        logger.info(f"Initialized ModelRegistry with {len(self.registered_models)} content types")
    
    def _initialize_defaults(self):
        """Initialize default model configurations"""
        # Default models from environment or hardcoded fallbacks
        self.registered_models = {
            'code': {
                'primary': os.getenv('QDRANT_CODE_EMBEDDING_MODEL', 'nomic-ai/CodeRankEmbed'),
                'fallback': os.getenv('QDRANT_CODE_EMBEDDING_FALLBACK', 'microsoft/codebert-base'),
                'dimension': 768,
                'max_tokens': 8192
            },
            'config': {
                'primary': os.getenv('QDRANT_CONFIG_EMBEDDING_MODEL', 'jinaai/jina-embeddings-v3'),
                'fallback': os.getenv('QDRANT_CONFIG_EMBEDDING_FALLBACK', 'jinaai/jina-embeddings-v2-base-en'),
                'dimension': 1024,
                'max_tokens': 8192
            },
            'documentation': {
                'primary': os.getenv('QDRANT_DOC_EMBEDDING_MODEL', 'hkunlp/instructor-large'),
                'fallback': os.getenv('QDRANT_DOC_EMBEDDING_FALLBACK', 'sentence-transformers/all-mpnet-base-v2'),
                'dimension': 768,
                'max_tokens': 512,
                'instruction_prefix': os.getenv('QDRANT_DOC_INSTRUCTION_PREFIX', 
                                               'Represent the technical documentation for retrieval:')
            },
            'general': {
                'primary': os.getenv('QDRANT_GENERAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                'fallback': None,
                'dimension': 384,
                'max_tokens': 256
            }
        }
        
        # Initialize model metadata
        self._update_model_metadata()
    
    def _update_model_metadata(self):
        """Update model metadata from registered models"""
        for content_type, config in self.registered_models.items():
            primary_model = config['primary']
            if primary_model not in self.model_metadata:
                self.model_metadata[primary_model] = {
                    'dimension': config['dimension'],
                    'max_tokens': config.get('max_tokens', 512),
                    'content_types': [content_type],
                    'is_primary': True
                }
            else:
                self.model_metadata[primary_model]['content_types'].append(content_type)
            
            # Add fallback model metadata
            fallback_model = config.get('fallback')
            if fallback_model and fallback_model not in self.model_metadata:
                # Estimate fallback dimensions based on model name
                fallback_dim = self._estimate_dimension(fallback_model)
                self.model_metadata[fallback_model] = {
                    'dimension': fallback_dim,
                    'max_tokens': config.get('max_tokens', 512) // 2,  # Assume half for fallback
                    'content_types': [content_type],
                    'is_primary': False
                }
    
    def _estimate_dimension(self, model_name: str) -> int:
        """Estimate embedding dimension based on model name"""
        # Common model dimension patterns
        if 'large' in model_name.lower():
            return 768
        elif 'base' in model_name.lower():
            return 768
        elif 'small' in model_name.lower() or 'mini' in model_name.lower():
            return 384
        elif 'jina' in model_name.lower() and 'v3' in model_name:
            return 1024
        elif 'codebert' in model_name.lower():
            return 768
        elif 'instructor' in model_name.lower():
            return 768
        else:
            return 384  # Default
    
    def _load_registry(self):
        """Load persisted registry from file"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    
                # Load collection mappings
                if 'collection_model_mapping' in data:
                    self.collection_model_mapping = data['collection_model_mapping']
                
                # Load any custom model registrations
                if 'custom_models' in data:
                    for content_type, config in data['custom_models'].items():
                        if content_type not in self.registered_models:
                            self.registered_models[content_type] = config
                
                logger.info(f"Loaded model registry from {self.registry_file}")
                
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
    
    def _save_registry(self):
        """Save registry to file"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            
            data = {
                'collection_model_mapping': self.collection_model_mapping,
                'custom_models': {},
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            # Save only custom (non-default) models
            default_types = {'code', 'config', 'documentation', 'general'}
            for content_type, config in self.registered_models.items():
                if content_type not in default_types:
                    data['custom_models'][content_type] = config
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved model registry to {self.registry_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration to the registry"""
        if 'models' in config:
            for content_type, model_config in config['models'].items():
                self.register_model(content_type, model_config)
    
    def register_model(self, content_type: str, model_config: Dict[str, Any]):
        """
        Register a model configuration for a content type
        
        Args:
            content_type: Type of content (e.g., 'code', 'config', 'documentation')
            model_config: Model configuration dictionary
        """
        # Validate required fields
        if 'primary' not in model_config and 'name' not in model_config:
            raise ValueError(f"Model config for {content_type} must have 'primary' or 'name' field")
        
        # Normalize field names
        if 'primary' in model_config and 'name' not in model_config:
            model_config['name'] = model_config['primary']
        elif 'name' in model_config and 'primary' not in model_config:
            model_config['primary'] = model_config['name']
        
        # Ensure dimension is set
        if 'dimension' not in model_config:
            model_config['dimension'] = self._estimate_dimension(model_config['primary'])
        
        # Register the model
        self.registered_models[content_type] = model_config
        
        # Update metadata
        self._update_model_metadata()
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model for content type '{content_type}': {model_config['primary']}")
    
    def get_model_for_content_type(self, content_type: str) -> str:
        """
        Get the appropriate model name for a content type
        
        Args:
            content_type: Type of content
            
        Returns:
            Model name
        """
        if content_type in self.registered_models:
            return self.registered_models[content_type]['primary']
        else:
            # Default to general model
            return self.registered_models['general']['primary']
    
    def get_model_config(self, content_type: str) -> Dict[str, Any]:
        """
        Get the full model configuration for a content type
        
        Args:
            content_type: Type of content
            
        Returns:
            Model configuration dictionary
        """
        if content_type in self.registered_models:
            return self.registered_models[content_type].copy()
        else:
            return self.registered_models['general'].copy()
    
    def get_model_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Embedding dimension
        """
        if model_name in self.model_metadata:
            return self.model_metadata[model_name]['dimension']
        else:
            # Try to find in registered models
            for config in self.registered_models.values():
                if config['primary'] == model_name:
                    return config['dimension']
                if config.get('fallback') == model_name:
                    # Estimate fallback dimension
                    return self._estimate_dimension(model_name)
            
            # Default estimate
            return self._estimate_dimension(model_name)
    
    def register_collection(self, collection_name: str, model_name: str, content_type: str):
        """
        Register a collection with its associated model
        
        Args:
            collection_name: Name of the Qdrant collection
            model_name: Name of the embedding model used
            content_type: Type of content in the collection
        """
        self.collection_model_mapping[collection_name] = {
            'model': model_name,
            'content_type': content_type,
            'registered_at': datetime.now().isoformat()
        }
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered collection '{collection_name}' with model '{model_name}' for {content_type}")
    
    def get_collection_model(self, collection_name: str) -> Optional[str]:
        """
        Get the model associated with a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Model name or None if not registered
        """
        if collection_name in self.collection_model_mapping:
            return self.collection_model_mapping[collection_name]['model']
        return None
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full information about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection info or None if not registered
        """
        if collection_name in self.collection_model_mapping:
            return self.collection_model_mapping[collection_name].copy()
        return None
    
    def list_registered_models(self) -> List[Tuple[str, str]]:
        """
        List all registered content types and their models
        
        Returns:
            List of (content_type, model_name) tuples
        """
        return [(ct, config['primary']) for ct, config in self.registered_models.items()]
    
    def list_collections(self) -> List[Tuple[str, str, str]]:
        """
        List all registered collections
        
        Returns:
            List of (collection_name, model_name, content_type) tuples
        """
        result = []
        for coll_name, info in self.collection_model_mapping.items():
            result.append((coll_name, info['model'], info.get('content_type', 'unknown')))
        return result
    
    def validate_model_compatibility(self, collection_name: str, model_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a model is compatible with a collection
        
        Args:
            collection_name: Name of the collection
            model_name: Name of the model to check
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        # Get collection's registered model
        collection_model = self.get_collection_model(collection_name)
        
        if not collection_model:
            # Collection not registered, can't validate
            return True, None
        
        if collection_model == model_name:
            return True, None
        
        # Check dimensions
        collection_dim = self.get_model_dimension(collection_model)
        model_dim = self.get_model_dimension(model_name)
        
        if collection_dim != model_dim:
            return False, (f"Dimension mismatch: collection uses {collection_model} "
                         f"({collection_dim}D) but trying to use {model_name} ({model_dim}D)")
        
        # Models have same dimensions, might be compatible
        return True, None
    
    def get_fallback_model(self, content_type: str) -> Optional[str]:
        """
        Get the fallback model for a content type
        
        Args:
            content_type: Type of content
            
        Returns:
            Fallback model name or None
        """
        if content_type in self.registered_models:
            return self.registered_models[content_type].get('fallback')
        return None
    
    def get_content_type_for_file(self, file_path: str) -> str:
        """
        Determine the appropriate content type for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content type
        """
        file_path = file_path.lower()
        ext = Path(file_path).suffix
        
        # Code files
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', 
                          '.cpp', '.c', '.h', '.php', '.rb', '.swift', '.kt', '.scala',
                          '.sh', '.bash', '.zsh', '.fish'}
        
        # Config files
        config_extensions = {'.json', '.yaml', '.yml', '.xml', '.toml', '.ini', 
                           '.cfg', '.conf', '.env', '.properties'}
        
        # Documentation files
        doc_extensions = {'.md', '.markdown', '.rst', '.txt', '.mdx'}
        
        if ext in code_extensions:
            return 'code'
        elif ext in config_extensions:
            return 'config'
        elif ext in doc_extensions:
            return 'documentation'
        else:
            return 'general'
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry"""
        return {
            'content_types': len(self.registered_models),
            'unique_models': len(self.model_metadata),
            'registered_collections': len(self.collection_model_mapping),
            'models': {
                ct: config['primary'] for ct, config in self.registered_models.items()
            },
            'registry_file': self.registry_file
        }


# Global instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry(config: Optional[Dict[str, Any]] = None) -> ModelRegistry:
    """Get or create the global model registry instance"""
    global _model_registry
    
    if _model_registry is None:
        _model_registry = ModelRegistry(config=config)
    
    return _model_registry