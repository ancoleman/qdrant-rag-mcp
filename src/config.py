# src/config.py
"""
Configuration handler for Qdrant MCP RAG Server

Manages configuration loading from JSON files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for the MCP server"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config/server_config.json")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables"""
        # Default configuration
        default_config = {
            "server": {
                "name": "qdrant-rag-server",
                "host": "0.0.0.0",
                "port": 8080,
                "log_level": "INFO"
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "api_key": None,
                "grpc_port": 6334,
                "https": False
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "cache_dir": "./data/models",
                "device": "auto",
                "batch_size": 32,
                "normalize_embeddings": True,
                "show_progress_bar": True
            },
            "indexing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "code_chunk_size": 1500,
                "code_chunk_overlap": 300,
                "batch_size": 100
            },
            "search": {
                "max_results": 5,
                "score_threshold": 0.7,
                "rerank": True
            },
            "collections": {
                "code": "code_collection",
                "config": "config_collection",
                "docs": "docs_collection"
            }
        }
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                default_config = self._deep_merge(default_config, file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        # Override with environment variables
        config = self._apply_env_vars(default_config)
        
        return config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variables to configuration"""
        # Map of config paths to environment variables
        env_mappings = {
            "server.port": "SERVER_PORT",
            "server.log_level": "LOG_LEVEL",
            "qdrant.host": "QDRANT_HOST",
            "qdrant.port": "QDRANT_PORT",
            "qdrant.api_key": "QDRANT_API_KEY",
            "embeddings.model": "EMBEDDING_MODEL",
            "embeddings.cache_dir": "SENTENCE_TRANSFORMERS_HOME",
            "embeddings.batch_size": "EMBEDDING_BATCH_SIZE"
        }
        
        for config_path, env_var in env_mappings.items():
            if env_value := os.getenv(env_var):
                # Navigate to the config location
                keys = config_path.split('.')
                current = config
                
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value
                last_key = keys[-1]
                
                # Type conversion
                if isinstance(current.get(last_key), int):
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        logger.warning(f"Failed to convert {env_var}={env_value} to int")
                        continue
                elif isinstance(current.get(last_key), bool):
                    env_value = env_value.lower() in ('true', '1', 'yes')
                
                current[last_key] = env_value
                logger.info(f"Applied environment variable {env_var}")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def reload(self):
        """Reload configuration from file and environment"""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config({self.config_path})"


# Global config instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _config
    
    if _config is None:
        _config = Config(config_path)
    
    return _config
