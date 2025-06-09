"""
Specialized Embeddings Manager for content-type specific models

This module provides advanced embedding management with support for multiple
specialized models optimized for different content types (code, config, docs).
"""

import os
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from .logging import get_project_logger
from .memory_manager import MemoryComponent, get_memory_manager

logger = get_project_logger()


class SpecializedEmbeddingManager(MemoryComponent):
    """Manages multiple specialized embedding models with lazy loading and memory optimization"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 max_models_in_memory: int = None,
                 memory_limit_gb: float = None):
        """
        Initialize the specialized embedding manager
        
        Args:
            config: Optional configuration dictionary
            cache_dir: Directory to cache downloaded models
            device: Device to use (cpu, cuda, mps, or auto)
            max_models_in_memory: Maximum number of models to keep loaded
            memory_limit_gb: Total memory limit for all models in GB
        """
        # Get full config from server config if not provided
        if config is None:
            from config import get_config
            server_config = get_config()
            config = server_config.get('specialized_embeddings', {})
        
        # Get memory limits from memory management config
        memory_manager = get_memory_manager()
        memory_config = memory_manager.config
        component_limits = memory_config.get('component_limits', {}).get('specialized_embeddings', {})
        
        # Use memory manager limits if not explicitly provided
        if memory_limit_gb is None:
            memory_limit_gb = float(component_limits.get('max_memory_mb', 4000)) / 1024
        if max_models_in_memory is None:
            max_models_in_memory = int(component_limits.get('max_items', 3))
        
        # Initialize parent MemoryComponent
        super().__init__(
            name="specialized_embeddings",
            max_memory_mb=memory_limit_gb * 1024
        )
        
        # Register with memory manager
        memory_manager.register_component("specialized_embeddings", self)
        
        # Initialize default model configurations
        self.model_configs = self._build_default_configs()
        
        # Apply custom configuration if provided
        if config:
            if 'models' in config:
                self._merge_config(config['models'])
        
        # Initialize properties (order matters - set defaults first, then let config override)
        self.device = device or self._auto_detect_device()
        self.max_models_in_memory = max_models_in_memory
        self.memory_limit_gb = memory_limit_gb
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/qdrant-mcp/models")
        
        # Apply Apple Silicon optimizations if detected
        if memory_manager.is_apple_silicon and self.device == "mps":
            # Conservative limits for unified memory architecture
            if self.memory_limit_gb > 3.0:
                logger.info(f"Reducing memory limit from {self.memory_limit_gb}GB to 3.0GB for Apple Silicon")
                self.memory_limit_gb = 3.0
            if self.max_models_in_memory > 2:
                logger.info(f"Reducing max models from {self.max_models_in_memory} to 2 for Apple Silicon")
                self.max_models_in_memory = 2
        
        # Apply memory settings from config or env vars
        if config and 'memory' in config:
            memory_config = config['memory']
            if isinstance(memory_config.get('max_models_in_memory'), (int, str)):
                try:
                    self.max_models_in_memory = int(memory_config['max_models_in_memory'])
                except ValueError:
                    pass
            if isinstance(memory_config.get('memory_limit_gb'), (float, str)):
                try:
                    self.memory_limit_gb = float(memory_config['memory_limit_gb'])
                except ValueError:
                    pass
            if 'cache_dir' in memory_config:
                self.cache_dir = os.path.expanduser(memory_config['cache_dir'])
        
        # Override with environment variables if set
        if os.getenv('QDRANT_MAX_MODELS_IN_MEMORY'):
            try:
                self.max_models_in_memory = int(os.getenv('QDRANT_MAX_MODELS_IN_MEMORY'))
            except ValueError:
                pass
        if os.getenv('QDRANT_MEMORY_LIMIT_GB'):
            try:
                self.memory_limit_gb = float(os.getenv('QDRANT_MEMORY_LIMIT_GB'))
            except ValueError:
                pass
        if os.getenv('QDRANT_MODEL_CACHE_DIR'):
            self.cache_dir = os.path.expanduser(os.getenv('QDRANT_MODEL_CACHE_DIR'))
        
        # LRU cache for loaded models
        self.loaded_models: OrderedDict[str, SentenceTransformer] = OrderedDict()
        
        # Memory tracking
        self.memory_usage: Dict[str, float] = {}
        self.total_memory_used_gb = 0.0
        
        # Model usage statistics
        self.usage_stats: Dict[str, Dict[str, int]] = {
            model_type: {'loads': 0, 'encodings': 0, 'errors': 0}
            for model_type in self.model_configs
        }
        
        logger.info(f"Initialized SpecializedEmbeddingManager with device: {self.device}, "
                   f"max_models: {max_models_in_memory}, memory_limit: {memory_limit_gb}GB")
    
    def _build_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build default model configurations from environment variables"""
        return {
            'code': {
                'name': os.getenv('QDRANT_CODE_EMBEDDING_MODEL', 'nomic-ai/CodeRankEmbed'),
                'dimension': 768,
                'fallback': os.getenv('QDRANT_CODE_EMBEDDING_FALLBACK', 'microsoft/codebert-base'),
                'max_tokens': 2048,  # Reduced from 8192 to prevent memory issues
                'description': 'Optimized for code understanding across multiple languages',
                'query_prefix': os.getenv('QDRANT_CODE_QUERY_PREFIX', None),  # Allow override via env
                'requires_query_prefix': True  # Flag to indicate if model needs special query handling
            },
            'config': {
                'name': os.getenv('QDRANT_CONFIG_EMBEDDING_MODEL', 'jinaai/jina-embeddings-v3'),
                'dimension': 1024,
                'fallback': os.getenv('QDRANT_CONFIG_EMBEDDING_FALLBACK', 'jinaai/jina-embeddings-v2-base-en'),
                'max_tokens': 8192,
                'description': 'Specialized for configuration files (JSON, YAML, etc.)'
            },
            'documentation': {
                'name': os.getenv('QDRANT_DOC_EMBEDDING_MODEL', 'hkunlp/instructor-large'),
                'dimension': 768,
                'fallback': os.getenv('QDRANT_DOC_EMBEDDING_FALLBACK', 'sentence-transformers/all-mpnet-base-v2'),
                'instruction_prefix': os.getenv('QDRANT_DOC_INSTRUCTION_PREFIX', 
                                               'Represent the technical documentation for retrieval:'),
                'max_tokens': 512,
                'description': 'Optimized for technical documentation with instruction support'
            },
            'general': {
                'name': os.getenv('QDRANT_GENERAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                'dimension': 384,
                'fallback': None,
                'max_tokens': 256,
                'description': 'General purpose embeddings and backward compatibility'
            }
        }
    
    def _merge_config(self, custom_models: Dict[str, Any]):
        """Merge custom model configuration with defaults"""
        for content_type, model_config in custom_models.items():
            if content_type in self.model_configs:
                # For model configs, handle 'primary' vs 'name' field
                if 'primary' in model_config:
                    model_config['name'] = model_config.pop('primary')
                self.model_configs[content_type].update(model_config)
            else:
                self.model_configs[content_type] = model_config
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA GPU detected for embeddings")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Metal Performance Shaders detected")
        else:
            device = "cpu"
            logger.info("Using CPU for embeddings")
        
        return device
    
    def _estimate_model_memory(self, model_name: str) -> float:
        """Estimate memory usage for a model in GB"""
        # More accurate estimates based on actual measured usage on MPS
        # These include model weights + PyTorch overhead + device buffers
        memory_estimates = {
            'nomic-ai/CodeRankEmbed': 1.0,  # Measured: ~0.91GB on MPS
            'jinaai/jina-embeddings-v3': 1.5,  # Measured: ~1.47GB on MPS
            'hkunlp/instructor-large': 1.2,  # 335M params
            'sentence-transformers/all-MiniLM-L6-v2': 0.2,  # 22M params
            'microsoft/codebert-base': 0.7,  # 125M params
            'jinaai/jina-embeddings-v2-base-en': 0.8,
            'sentence-transformers/all-mpnet-base-v2': 0.6  # 110M params
        }
        
        # Return estimate or default based on model size keywords
        if model_name in memory_estimates:
            return memory_estimates[model_name]
        elif 'large' in model_name.lower():
            return 1.5
        elif 'base' in model_name.lower():
            return 0.5
        elif 'small' in model_name.lower() or 'mini' in model_name.lower():
            return 0.2
        else:
            return 1.0  # Default estimate
    
    def _check_memory_before_loading(self, model_name: str) -> bool:
        """Check if we have enough memory to load a model"""
        estimated_memory = self._estimate_model_memory(model_name)
        
        # Get memory manager for Apple Silicon checks
        memory_manager = get_memory_manager()
        
        # Check actual system memory if possible
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # More conservative safety margin for Apple Silicon
            if memory_manager.is_apple_silicon and self.device == "mps":
                # For CodeRankEmbed on Apple Silicon, use 2.5x safety margin
                safety_margin = 2.5 if 'CodeRankEmbed' in model_name else 2.0
                
                # Also check if we should trigger cleanup first
                if available_memory_gb < 3.0:  # Less than 3GB available
                    logger.info("Low memory on Apple Silicon, triggering cleanup before model load")
                    memory_manager.perform_apple_silicon_cleanup("aggressive")
                    # Re-check after cleanup
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
            else:
                safety_margin = 1.5  # Standard safety margin
            
            if available_memory_gb < estimated_memory * safety_margin:
                logger.warning(f"Insufficient system memory for {model_name}: "
                              f"{available_memory_gb:.1f}GB available, need {estimated_memory * safety_margin:.1f}GB")
                return False
        except ImportError:
            pass  # psutil not available, fall back to tracking
        
        if self.total_memory_used_gb + estimated_memory > self.memory_limit_gb:
            logger.warning(f"Loading {model_name} would exceed memory limit "
                          f"({self.total_memory_used_gb + estimated_memory:.1f}GB > {self.memory_limit_gb}GB)")
            return False
        return True
    
    def _evict_lru_model(self):
        """Evict the least recently used model"""
        if not self.loaded_models:
            return
        
        # Get the least recently used model (first in OrderedDict)
        lru_model_name, lru_model = self.loaded_models.popitem(last=False)
        
        # Update memory tracking
        if lru_model_name in self.memory_usage:
            self.total_memory_used_gb -= self.memory_usage[lru_model_name]
            del self.memory_usage[lru_model_name]
        
        # Clean up
        del lru_model
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache if applicable
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS doesn't have explicit cache clearing, but we can try to free memory
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        logger.info(f"Evicted model {lru_model_name} from memory. "
                   f"Current memory usage: {self.total_memory_used_gb:.1f}GB")
    
    def load_model(self, content_type: str) -> Tuple[SentenceTransformer, Dict[str, Any]]:
        """
        Load a model for the specified content type with LRU eviction
        
        Args:
            content_type: Type of content (code, config, documentation, general)
            
        Returns:
            Tuple of (model, model_config)
        """
        # Get model configuration
        model_config = self.model_configs.get(content_type, self.model_configs['general'])
        model_name = model_config['name']
        
        # Update usage statistics
        self.usage_stats[content_type]['loads'] += 1
        
        # Check if already loaded
        if model_name in self.loaded_models:
            # Move to end (most recently used)
            self.loaded_models.move_to_end(model_name)
            logger.debug(f"Using cached model {model_name}")
            return self.loaded_models[model_name], model_config
        
        # Check memory constraints
        if len(self.loaded_models) >= self.max_models_in_memory:
            self._evict_lru_model()
        
        if not self._check_memory_before_loading(model_name):
            # Try to free up memory by evicting models
            while self.loaded_models and not self._check_memory_before_loading(model_name):
                self._evict_lru_model()
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load the model
        try:
            logger.info(f"Loading embedding model: {model_name} for {content_type}")
            
            # Ensure HuggingFace uses our cache directory for modules
            # This is critical for models like jina-embeddings-v3 that have custom code
            if not os.environ.get('HF_HOME'):
                os.environ['HF_HOME'] = str(Path(self.cache_dir).resolve())
            if not os.environ.get('HF_HUB_CACHE'):
                os.environ['HF_HUB_CACHE'] = str(Path(self.cache_dir).resolve())
            
            # Load model with MPS optimization if applicable
            model = self._load_model_with_mps_optimization(model_name, content_type)
            
            # Track memory usage
            estimated_memory = self._estimate_model_memory(model_name)
            self.memory_usage[model_name] = estimated_memory
            self.total_memory_used_gb += estimated_memory
            
            # Cache the model
            self.loaded_models[model_name] = model
            
            logger.info(f"Successfully loaded {model_name}. "
                       f"Memory usage: {self.total_memory_used_gb:.1f}/{self.memory_limit_gb}GB")
            
            return model, model_config
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            self.usage_stats[content_type]['errors'] += 1
            
            # Try fallback model
            fallback_name = model_config.get('fallback')
            if fallback_name and fallback_name != model_name:
                logger.info(f"Attempting to load fallback model: {fallback_name}")
                model_config = model_config.copy()
                model_config['name'] = fallback_name
                
                try:
                    # Load fallback model with MPS optimization
                    model = self._load_model_with_mps_optimization(fallback_name, content_type)
                    
                    # Track memory usage
                    estimated_memory = self._estimate_model_memory(fallback_name)
                    self.memory_usage[fallback_name] = estimated_memory
                    self.total_memory_used_gb += estimated_memory
                    
                    # Cache the model
                    self.loaded_models[fallback_name] = model
                    
                    logger.info(f"Successfully loaded fallback {fallback_name}")
                    return model, model_config
                    
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback {fallback_name}: {fallback_e}")
            
            # Last resort: fall back to general model
            if content_type != 'general':
                logger.warning(f"Falling back to general model for {content_type}")
                return self.load_model('general')
            
            raise RuntimeError(f"Unable to load any embedding model for {content_type}")
    
    def _load_model_with_mps_optimization(self, model_name: str, content_type: str) -> SentenceTransformer:
        """Load model with Apple Silicon MPS optimizations"""
        memory_manager = get_memory_manager()
        
        # Pre-loading memory check for Apple Silicon
        if self.device == "mps" and memory_manager.is_apple_silicon:
            import psutil
            import os
            
            # Check available memory before loading
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            estimated_memory = self._estimate_model_memory(model_name)
            
            # For CodeRankEmbed on Apple Silicon, be extra conservative
            if 'CodeRankEmbed' in model_name:
                safety_multiplier = 2.5  # More conservative for Apple Silicon
                if available_memory_gb < estimated_memory * safety_multiplier:
                    # Aggressive cleanup before loading CodeRankEmbed
                    logger.info(f"Low memory for {model_name} on Apple Silicon, performing cleanup")
                    memory_manager.perform_apple_silicon_cleanup("aggressive")
                    
                    # Re-check after cleanup
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
                    if available_memory_gb < estimated_memory * 2.0:
                        logger.warning(f"Insufficient memory for {model_name} on MPS, attempting CPU fallback")
                        return self._load_model_cpu_fallback(model_name, content_type)
            
            # Set MPS-specific environment variables for CodeRankEmbed
            if 'CodeRankEmbed' in model_name:
                original_env = {}
                mps_env_vars = {
                    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                    'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                    'MPS_CAPTURE_STDERR': '1'
                }
                
                # Set environment variables temporarily
                for key, value in mps_env_vars.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                try:
                    model = self._load_model_standard(model_name)
                    
                    # Post-loading MPS optimization for CodeRankEmbed
                    import torch
                    if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                        torch.mps.set_per_process_memory_fraction(0.7)  # Limit MPS usage to 70%
                    
                    return model
                    
                finally:
                    # Restore original environment
                    for key, original_value in original_env.items():
                        if original_value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = original_value
        
        # Standard loading for non-Apple Silicon or non-MPS devices
        return self._load_model_standard(model_name)
    
    def _load_model_standard(self, model_name: str) -> SentenceTransformer:
        """Standard model loading without special optimizations"""
        # Some models require trust_remote_code
        trust_remote_code = model_name in [
            'nomic-ai/CodeRankEmbed',
            'jinaai/jina-embeddings-v3'
        ]
        
        # Set environment variables for model caching
        import os
        if not os.environ.get('SENTENCE_TRANSFORMERS_HOME'):
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(Path(self.cache_dir).resolve())
        if not os.environ.get('HF_HOME'):
            os.environ['HF_HOME'] = str(Path(self.cache_dir).resolve())
        if not os.environ.get('HF_HUB_CACHE'):
            os.environ['HF_HUB_CACHE'] = str(Path(self.cache_dir).resolve())
        
        if trust_remote_code:
            model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=True
            )
        else:
            model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
        
        model.eval()
        return model
    
    def _load_model_cpu_fallback(self, model_name: str, content_type: str) -> SentenceTransformer:
        """Fallback to CPU loading when MPS memory is insufficient"""
        logger.warning(f"Loading {model_name} on CPU due to memory constraints")
        
        original_device = self.device
        try:
            # Temporarily switch to CPU
            self.device = "cpu"
            
            # Some models require trust_remote_code
            trust_remote_code = model_name in [
                'nomic-ai/CodeRankEmbed',
                'jinaai/jina-embeddings-v3'
            ]
            
            if trust_remote_code:
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    cache_folder=self.cache_dir,
                    trust_remote_code=True
                )
            else:
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    cache_folder=self.cache_dir
                )
            
            model.eval()
            
            logger.info(f"Successfully loaded {model_name} on CPU as fallback")
            return model
            
        finally:
            # Don't restore device - keep it as CPU for this model
            # The device will be set correctly for future models
            pass
    
    def encode(self,
               texts: Union[str, List[str]],
               content_type: str = 'general',
               batch_size: int = 32,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode texts using the appropriate model for the content type
        
        Args:
            texts: Single text or list of texts to encode
            content_type: Type of content being encoded
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Embeddings as numpy array
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Load appropriate model
        model, model_config = self.load_model(content_type)
        
        # Update usage statistics
        self.usage_stats[content_type]['encodings'] += len(texts)
        
        # Apply instruction prefix for documentation if using instructor model
        if content_type == 'documentation' and 'instructor' in model_config['name']:
            instruction = model_config.get('instruction_prefix', '')
            if instruction:
                texts = [f"{instruction} {text}" for text in texts]
        
        # Apply query prefix for models that require it
        # This provides flexibility while maintaining backward compatibility:
        # 1. Check if model requires a query prefix (via config flag)
        # 2. Use custom prefix if configured (via env var or config)
        # 3. Fall back to known model-specific prefixes
        # 4. This ensures CodeRankEmbed continues to work correctly with its required prefix
        if len(texts) == 1 and model_config.get('requires_query_prefix', False):
            # This is likely a query, not a document
            # First check if a custom prefix is configured
            if model_config.get('query_prefix'):
                texts = [f"{model_config['query_prefix']} {texts[0]}"]
            # Fallback to known model-specific prefixes
            elif 'CodeRankEmbed' in model_config['name']:
                # CodeRankEmbed requires this specific prefix as per their documentation
                texts = [f"Represent this query for searching relevant code: {texts[0]}"]
            # Add more model-specific prefixes here as needed
            # elif 'SomeOtherModel' in model_config['name']:
            #     texts = [f"Their specific prefix: {texts[0]}"]
        
        # Truncate texts if they exceed max tokens
        max_tokens = model_config.get('max_tokens', 512)
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        truncated_texts = []
        for text in texts:
            if len(text) > max_chars:
                truncated_texts.append(text[:max_chars] + "...")
                logger.debug(f"Truncated text from {len(text)} to {max_chars} chars")
            else:
                truncated_texts.append(text)
        
        try:
            # Special handling for CodeRankEmbed - it's memory intensive
            if 'CodeRankEmbed' in model_config['name']:
                # Use smaller batch size for CodeRankEmbed
                actual_batch_size = min(batch_size, 4)
                logger.debug(f"Using reduced batch size {actual_batch_size} for CodeRankEmbed (requested: {batch_size})")
                
                # If we have many texts, process in smaller chunks to avoid memory issues
                if len(truncated_texts) > actual_batch_size:
                    all_embeddings = []
                    for i in range(0, len(truncated_texts), actual_batch_size):
                        batch_texts = truncated_texts[i:i+actual_batch_size]
                        batch_embeddings = model.encode(
                            batch_texts,
                            batch_size=actual_batch_size,
                            show_progress_bar=False,  # No progress for sub-batches
                            normalize_embeddings=normalize_embeddings,
                            convert_to_numpy=True,
                            convert_to_tensor=False
                        )
                        all_embeddings.append(batch_embeddings)
                        
                        # Force cleanup after each sub-batch for CodeRankEmbed
                        if self.device == "mps":
                            memory_manager = get_memory_manager()
                            if memory_manager.is_apple_silicon:
                                # Check memory pressure after each batch
                                import psutil
                                available_gb = psutil.virtual_memory().available / (1024**3)
                                if available_gb < 2.0:  # Critical threshold
                                    logger.debug(f"Low memory during encoding: {available_gb:.1f}GB, triggering cleanup")
                                    memory_manager.perform_apple_silicon_cleanup("aggressive")
                                else:
                                    # Standard MPS cleanup
                                    import torch
                                    if hasattr(torch.mps, 'empty_cache'):
                                        torch.mps.empty_cache()
                    
                    embeddings = np.vstack(all_embeddings)
                else:
                    embeddings = model.encode(
                        truncated_texts,
                        batch_size=actual_batch_size,
                        show_progress_bar=show_progress_bar,
                        normalize_embeddings=normalize_embeddings,
                        convert_to_numpy=True,
                        convert_to_tensor=False
                    )
            else:
                # Regular encoding for other models
                embeddings = model.encode(
                    truncated_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                    convert_to_tensor=False  # Ensure numpy output
                )
            
            # Ensure correct data type (Qdrant expects float32)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Ensure 2D shape even for single input
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            logger.debug(f"Encoded {len(texts)} texts using {model_config['name']} "
                        f"into {embeddings.shape} embeddings with dtype {embeddings.dtype}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding with {model_config['name']}: {e}")
            self.usage_stats[content_type]['errors'] += 1
            
            # If not already using general model, fall back to it
            if content_type != 'general':
                logger.warning(f"Falling back to general model for encoding")
                return self.encode(texts, 'general', batch_size, show_progress_bar, normalize_embeddings)
            
            raise
    
    def get_dimension(self, content_type: str = 'general') -> int:
        """Get the embedding dimension for a content type"""
        model_config = self.model_configs.get(content_type, self.model_configs['general'])
        return model_config['dimension']
    
    def get_model_name(self, content_type: str = 'general') -> str:
        """Get the model name for a content type"""
        model_config = self.model_configs.get(content_type, self.model_configs['general'])
        return model_config['name']
    
    def get_model_info(self, content_type: str = 'general') -> Dict[str, Any]:
        """Get detailed information about a model"""
        model_config = self.model_configs.get(content_type, self.model_configs['general'])
        
        # Check if model is loaded
        model_name = model_config['name']
        is_loaded = model_name in self.loaded_models
        
        info = {
            'content_type': content_type,
            'model_name': model_name,
            'dimension': model_config['dimension'],
            'max_tokens': model_config.get('max_tokens', 512),
            'description': model_config.get('description', ''),
            'fallback': model_config.get('fallback'),
            'is_loaded': is_loaded,
            'memory_usage_gb': self.memory_usage.get(model_name, 0),
            'usage_stats': self.usage_stats.get(content_type, {})
        }
        
        # Add instruction prefix for documentation models
        if content_type == 'documentation':
            info['instruction_prefix'] = model_config.get('instruction_prefix', '')
        
        return info
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Get information about all models and system status"""
        return {
            'device': self.device,
            'cache_dir': self.cache_dir,
            'memory': {
                'used_gb': self.total_memory_used_gb,
                'limit_gb': self.memory_limit_gb,
                'models_loaded': len(self.loaded_models),
                'max_models': self.max_models_in_memory
            },
            'models': {
                content_type: self.get_model_info(content_type)
                for content_type in self.model_configs
            },
            'loaded_models': list(self.loaded_models.keys()),
            'available_devices': {
                'cpu': True,
                'cuda': torch.cuda.is_available(),
                'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
        }
    
    def warmup(self, content_types: Optional[List[str]] = None):
        """Warmup models by loading them and doing a test encoding"""
        if content_types is None:
            content_types = ['general']  # Only warmup general by default
        
        logger.info(f"Warming up models for content types: {content_types}")
        
        for content_type in content_types:
            try:
                # Load model
                _, config = self.load_model(content_type)
                
                # Do a test encoding
                test_text = f"This is a warmup sentence for {content_type} embeddings."
                _ = self.encode(test_text, content_type, show_progress_bar=False)
                
                logger.info(f"Warmed up {config['name']} for {content_type}")
                
            except Exception as e:
                logger.error(f"Failed to warmup {content_type}: {e}")
    
    def clear_cache(self):
        """Clear all loaded models from memory"""
        logger.info(f"Clearing {len(self.loaded_models)} models from cache")
        
        self.loaded_models.clear()
        self.memory_usage.clear()
        self.total_memory_used_gb = 0.0
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if self.device == "cuda" else None
        
        logger.info("Model cache cleared")
    
    def set_query_prefix(self, content_type: str, prefix: str, requires_prefix: bool = True):
        """Set a custom query prefix for a content type
        
        Args:
            content_type: The content type to configure
            prefix: The prefix to use for queries (set to None to disable)
            requires_prefix: Whether the model requires a query prefix
        """
        if content_type in self.model_configs:
            self.model_configs[content_type]['query_prefix'] = prefix
            self.model_configs[content_type]['requires_query_prefix'] = requires_prefix
            logger.info(f"Updated query prefix for {content_type}: {prefix}")
    
    def get_content_type_for_file(self, file_path: str) -> str:
        """Determine the appropriate content type for a file"""
        file_path = file_path.lower()
        filename = Path(file_path).name
        
        # Special config files (dot files and specific names)
        special_config_files = {'.env', '.gitignore', '.dockerignore', '.prettierrc', 
                               '.eslintrc', '.editorconfig', '.npmrc', '.yarnrc', '.ragignore',
                               'dockerfile', 'makefile', 'requirements.txt', 'package.json',
                               'composer.json', 'cargo.toml', 'pyproject.toml'}
        
        # Check for special config files first
        if filename in special_config_files:
            return 'config'
        
        # Code files
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', 
                          '.cpp', '.c', '.h', '.php', '.rb', '.swift', '.kt', '.scala',
                          '.sh', '.bash', '.zsh', '.fish'}
        
        # Config files
        config_extensions = {'.json', '.yaml', '.yml', '.xml', '.toml', '.ini', 
                           '.cfg', '.conf', '.env', '.properties'}
        
        # Documentation files
        doc_extensions = {'.md', '.markdown', '.rst', '.txt', '.mdx'}
        
        # Get file extension
        ext = Path(file_path).suffix
        
        if ext in code_extensions:
            return 'code'
        elif ext in config_extensions:
            return 'config'
        elif ext in doc_extensions:
            return 'documentation'
        else:
            return 'general'
    
    def get_item_count(self) -> int:
        """Get number of models currently loaded"""
        return len(self.loaded_models)
    
    def cleanup(self, aggressive: bool = False) -> int:
        """Perform cleanup and return number of models removed"""
        if aggressive:
            # Remove all but one model
            target_keep = 1
        else:
            # Remove half of the models
            target_keep = max(1, len(self.loaded_models) // 2)
        
        removed = 0
        while len(self.loaded_models) > target_keep:
            self._evict_lru_model()
            removed += 1
        
        self.mark_cleanup()
        return removed


# Global instance
_specialized_manager: Optional[SpecializedEmbeddingManager] = None


def get_specialized_embedding_manager(config: Optional[Dict[str, Any]] = None) -> SpecializedEmbeddingManager:
    """Get or create the global specialized embedding manager instance"""
    global _specialized_manager
    
    if _specialized_manager is None:
        _specialized_manager = SpecializedEmbeddingManager(config=config)
    
    return _specialized_manager