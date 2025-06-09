# src/utils/embeddings.py
"""
Embeddings utility module for Qdrant MCP RAG Server

Handles embedding model management, caching, and optimization
for different platforms (especially macOS with MPS).

This module now supports both single-model mode (legacy) and
specialized embeddings mode for content-type specific models.
"""

import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

# Import specialized embeddings if available
try:
    from .specialized_embeddings import SpecializedEmbeddingManager, get_specialized_embedding_manager
    SPECIALIZED_EMBEDDINGS_AVAILABLE = True
except ImportError:
    SPECIALIZED_EMBEDDINGS_AVAILABLE = False
    logger.debug("Specialized embeddings not available, using single model mode")


class EmbeddingsManager:
    """Manages embedding models and generation"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 normalize_embeddings: bool = True,
                 show_progress_bar: bool = True):
        """
        Initialize the embeddings manager
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache downloaded models
            device: Device to use (cpu, cuda, mps, or auto)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings
            show_progress_bar: Whether to show progress bar during encoding
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/qdrant-mcp/models")
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        
        # Auto-detect device if not specified
        self.device = device or self._auto_detect_device()
        
        # Initialize model
        self._model = None
        self._model_dimension = None
        
        logger.info(f"Initialized EmbeddingsManager with model: {model_name}, device: {self.device}")
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA GPU detected")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Metal Performance Shaders detected")
        else:
            device = "cpu"
            logger.info("Using CPU for embeddings")
        
        return device
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Create cache directory if it doesn't exist
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_dir
                )
                
                # Set model to eval mode
                self._model.eval()
                
                # Get embedding dimension
                self._model_dimension = self._model.get_sentence_embedding_dimension()
                
                logger.info(f"Model loaded successfully. Dimension: {self._model_dimension}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                # Fallback to a simpler model
                logger.info("Falling back to all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"
                self._model = SentenceTransformer(
                    self.model_name,
                    device="cpu",  # Use CPU for fallback
                    cache_folder=self.cache_dir
                )
                self._model_dimension = self._model.get_sentence_embedding_dimension()
        
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self._model_dimension is None:
            _ = self.model  # Force model loading
        return self._model_dimension
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               convert_to_tensor: bool = False,
               show_progress_bar: Optional[bool] = None) -> np.ndarray:
        """
        Encode text(s) into embeddings
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Override default batch size
            convert_to_tensor: Return PyTorch tensor instead of numpy array
            show_progress_bar: Override default progress bar setting
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Use default values if not specified
        batch_size = batch_size or self.batch_size
        show_progress_bar = show_progress_bar if show_progress_bar is not None else self.show_progress_bar
        
        try:
            # Encode texts
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=self.normalize_embeddings
            )
            
            logger.debug(f"Encoded {len(texts)} texts into embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Fallback: encode one by one
            logger.info("Attempting fallback encoding...")
            embeddings = []
            for text in texts:
                try:
                    emb = self.model.encode(
                        [text],
                        show_progress_bar=False,
                        normalize_embeddings=self.normalize_embeddings
                    )
                    embeddings.append(emb[0])
                except Exception as inner_e:
                    logger.error(f"Failed to encode text: {inner_e}")
                    # Return zero vector for failed encoding
                    embeddings.append(np.zeros(self.dimension))
            
            embeddings = np.array(embeddings)
            
            if convert_to_tensor:
                embeddings = torch.tensor(embeddings)
            
            return embeddings
    
    def encode_batch(self, texts: List[str], max_length: int = 512) -> List[List[float]]:
        """
        Encode a batch of texts with length limiting
        
        Args:
            texts: List of texts to encode
            max_length: Maximum length of text to consider
            
        Returns:
            List of embedding vectors
        """
        # Truncate long texts
        truncated_texts = []
        for text in texts:
            if len(text) > max_length:
                truncated_texts.append(text[:max_length] + "...")
            else:
                truncated_texts.append(text)
        
        # Encode
        embeddings = self.encode(truncated_texts)
        
        # Convert to list of lists for JSON serialization
        return embeddings.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "available_devices": {
                "cpu": True,
                "cuda": torch.cuda.is_available(),
                "mps": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
        }
    
    def switch_model(self, model_name: str):
        """Switch to a different embedding model"""
        logger.info(f"Switching from {self.model_name} to {model_name}")
        
        # Clear current model
        self._model = None
        self._model_dimension = None
        
        # Update model name
        self.model_name = model_name
        
        # Force load new model
        _ = self.model
        
        logger.info(f"Successfully switched to {model_name}")
    
    def switch_device(self, device: str):
        """Switch to a different device"""
        if device not in ["cpu", "cuda", "mps", "auto"]:
            raise ValueError(f"Invalid device: {device}")
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        
        if device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"
        
        if device == "auto":
            device = self._auto_detect_device()
        
        logger.info(f"Switching from {self.device} to {device}")
        
        # Update device
        self.device = device
        
        # Reload model on new device
        self._model = None
        _ = self.model
    
    @lru_cache(maxsize=1000)
    def encode_cached(self, text: str) -> List[float]:
        """
        Encode text with caching (useful for repeated queries)
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as list
        """
        embedding = self.encode([text])[0]
        return embedding.tolist()
    
    def clear_cache(self):
        """Clear the encoding cache"""
        self.encode_cached.cache_clear()
        logger.info("Cleared embedding cache")
    
    def warmup(self):
        """Warmup the model with a dummy encoding"""
        logger.info("Warming up embedding model...")
        _ = self.encode(["This is a warmup sentence."], show_progress_bar=False)
        logger.info("Model warmed up")
    
    def benchmark(self, num_texts: int = 100, text_length: int = 100) -> Dict[str, float]:
        """
        Benchmark the embedding model
        
        Args:
            num_texts: Number of texts to encode
            text_length: Length of each text
            
        Returns:
            Benchmark results
        """
        import time
        import string
        import random
        
        # Generate random texts
        texts = []
        for _ in range(num_texts):
            text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=text_length))
            texts.append(text)
        
        # Benchmark encoding
        start_time = time.time()
        embeddings = self.encode(texts, show_progress_bar=False)
        end_time = time.time()
        
        total_time = end_time - start_time
        texts_per_second = num_texts / total_time
        
        return {
            "num_texts": num_texts,
            "text_length": text_length,
            "total_time": total_time,
            "texts_per_second": texts_per_second,
            "batch_size": self.batch_size,
            "device": self.device,
            "model": self.model_name
        }


# Global instances
_embeddings_manager = None
_use_specialized = None


def should_use_specialized_embeddings(config: Optional[Dict[str, Any]] = None) -> bool:
    """Check if specialized embeddings should be used"""
    global _use_specialized
    
    if _use_specialized is None:
        # Check environment variable first
        if os.getenv('QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED', '').lower() == 'false':
            _use_specialized = False
        # Check config
        elif config and 'specialized_embeddings' in config:
            _use_specialized = config['specialized_embeddings'].get('enabled', True)
        else:
            # Default to True if available
            _use_specialized = SPECIALIZED_EMBEDDINGS_AVAILABLE
    
    return _use_specialized and SPECIALIZED_EMBEDDINGS_AVAILABLE


class UnifiedEmbeddingsManager:
    """Unified manager that wraps both single and specialized embedding modes"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.use_specialized = should_use_specialized_embeddings(config)
        
        if self.use_specialized:
            # Use specialized embeddings
            specialized_config = self.config.get('specialized_embeddings', {})
            self.manager = get_specialized_embedding_manager(specialized_config)
            logger.info("Using specialized embeddings mode")
        else:
            # Use single model mode
            embeddings_config = self.config.get('embeddings', {})
            self.manager = EmbeddingsManager(
                model_name=embeddings_config.get("model", "all-MiniLM-L6-v2"),
                cache_dir=embeddings_config.get("cache_dir"),
                device=embeddings_config.get("device"),
                batch_size=embeddings_config.get("batch_size", 32),
                normalize_embeddings=embeddings_config.get("normalize_embeddings", True),
                show_progress_bar=embeddings_config.get("show_progress_bar", True)
            )
            logger.info(f"Using single model mode: {self.manager.model_name}")
    
    def encode(self, texts: Union[str, List[str]], 
               content_type: Optional[str] = None,
               **kwargs) -> np.ndarray:
        """
        Encode texts with appropriate model
        
        Args:
            texts: Texts to encode
            content_type: Content type for specialized embeddings
            **kwargs: Additional arguments for encoding
        """
        if self.use_specialized:
            # Use specialized manager
            # Default to "general" content type if not specified for backward compatibility
            if content_type is None:
                content_type = "general"
            return self.manager.encode(texts, content_type=content_type, **kwargs)
        else:
            # Use single model manager
            if hasattr(self.manager, 'encode_batch'):
                # Handle batch encoding for legacy manager
                if isinstance(texts, str):
                    texts = [texts]
                return np.array(self.manager.encode_batch(texts))
            else:
                return self.manager.encode(texts, **kwargs)
    
    def get_dimension(self, content_type: Optional[str] = None) -> int:
        """Get embedding dimension"""
        if self.use_specialized:
            # Default to "general" content type if not specified
            if content_type is None:
                content_type = "general"
            return self.manager.get_dimension(content_type)
        else:
            return self.manager.dimension
    
    def get_model_name(self, content_type: Optional[str] = None) -> str:
        """Get model name"""
        if self.use_specialized:
            # Default to "general" content type if not specified
            if content_type is None:
                content_type = "general"
            return self.manager.get_model_name(content_type)
        else:
            return self.manager.model_name
    
    def get_model_info(self, content_type: Optional[str] = None) -> Dict[str, Any]:
        """Get model information"""
        if self.use_specialized:
            if content_type:
                return self.manager.get_model_info(content_type)
            else:
                return self.manager.get_all_models_info()
        else:
            return self.manager.get_model_info()
    
    # Backward compatibility methods
    def get_sentence_embedding_dimension(self) -> int:
        """Backward compatibility method for getting embedding dimension"""
        return self.get_dimension()
    
    @property
    def dimension(self) -> int:
        """Property for backward compatibility"""
        return self.get_dimension()
    
    @property
    def model_name(self) -> str:
        """Property for backward compatibility"""
        return self.get_model_name()


# Global unified manager
_unified_manager = None


def get_embeddings_manager(config: Optional[Dict[str, Any]] = None) -> Union[EmbeddingsManager, UnifiedEmbeddingsManager]:
    """
    Get global embeddings manager instance
    
    Returns either UnifiedEmbeddingsManager (if specialized embeddings enabled)
    or legacy EmbeddingsManager for backward compatibility
    """
    global _embeddings_manager, _unified_manager
    
    # Check if we should use specialized embeddings
    if should_use_specialized_embeddings(config):
        if _unified_manager is None:
            _unified_manager = UnifiedEmbeddingsManager(config)
        return _unified_manager
    else:
        # Legacy single model mode
        if _embeddings_manager is None:
            if config is None:
                config = {}
            
            embeddings_config = config.get('embeddings', config)
            _embeddings_manager = EmbeddingsManager(
                model_name=embeddings_config.get("model", "all-MiniLM-L6-v2"),
                cache_dir=embeddings_config.get("cache_dir"),
                device=embeddings_config.get("device"),
                batch_size=embeddings_config.get("batch_size", 32),
                normalize_embeddings=embeddings_config.get("normalize_embeddings", True),
                show_progress_bar=embeddings_config.get("show_progress_bar", True)
            )
        
        return _embeddings_manager
