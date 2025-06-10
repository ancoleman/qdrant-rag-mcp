#!/usr/bin/env python3
"""
Verify specialized embeddings configuration without modifying any data
This script only reads configuration and checks model availability
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_env_config():
    """Check environment variables for specialized embeddings"""
    print("=" * 70)
    print("Specialized Embeddings Configuration Check")
    print("=" * 70)
    
    print("\n1. Environment Variables:")
    
    env_vars = {
        "QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED": "Enable specialized embeddings",
        "QDRANT_CODE_EMBEDDING_MODEL": "Code embedding model",
        "QDRANT_CONFIG_EMBEDDING_MODEL": "Config embedding model", 
        "QDRANT_DOC_EMBEDDING_MODEL": "Documentation embedding model",
        "QDRANT_GENERAL_EMBEDDING_MODEL": "General/fallback embedding model",
        "QDRANT_MAX_MODELS_IN_MEMORY": "Max models in memory",
        "QDRANT_MEMORY_LIMIT_GB": "Memory limit (GB)",
        "SENTENCE_TRANSFORMERS_HOME": "Model cache directory"
    }
    
    for var, desc in env_vars.items():
        value = os.getenv(var, "NOT SET")
        # Clean up potential comments
        if value != "NOT SET" and "#" in value:
            value = value.split('#')[0].strip()
        print(f"  {var}: {value}")
        print(f"    ({desc})")
    
    enabled = os.getenv("QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED", "false").lower() == "true"
    return enabled

def check_server_config():
    """Check server_config.json for specialized embeddings config"""
    print("\n2. Server Configuration (config/server_config.json):")
    
    config_path = Path("config/server_config.json")
    if not config_path.exists():
        print("  ✗ server_config.json not found")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        if "specialized_embeddings" in config:
            spec_config = config["specialized_embeddings"]
            print("  ✓ Specialized embeddings configuration found")
            print(f"    Enabled: {spec_config.get('enabled', False)}")
            
            if "models" in spec_config:
                print("    Models:")
                for content_type, model_config in spec_config["models"].items():
                    print(f"      {content_type}: {model_config.get('name', 'NOT SET')}")
                    print(f"        Dimension: {model_config.get('dimension', 'NOT SET')}")
                    if 'fallback' in model_config:
                        print(f"        Fallback: {model_config['fallback']}")
            
            return spec_config.get('enabled', False)
        else:
            print("  ✗ No specialized embeddings configuration found")
            return False
            
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False

def check_models_downloaded():
    """Check which models are actually downloaded"""
    print("\n3. Downloaded Models:")
    
    # Always use local data/models directory
    cache_dir = "data/models"
    print(f"  Using local cache: {cache_dir}")
    
    if not Path(cache_dir).exists():
        print("  ✗ Cache directory does not exist")
        return {}
    
    # Expected models
    expected_models = {
        "nomic-ai/CodeRankEmbed": "Code embeddings (768D, ~2GB)",
        "jinaai/jina-embeddings-v3": "Config embeddings (1024D, ~2GB)",
        "hkunlp/instructor-large": "Documentation embeddings (768D, ~1.5GB)",
        "sentence-transformers/all-MiniLM-L6-v2": "General/fallback (384D, ~90MB)",
        "microsoft/codebert-base": "Code fallback (768D, ~440MB)",
        "jinaai/jina-embeddings-v2-base-en": "Config fallback (768D, ~1GB)",
        "sentence-transformers/all-mpnet-base-v2": "Docs fallback (768D, ~420MB)"
    }
    
    downloaded = {}
    for model_name, description in expected_models.items():
        model_dir_name = f"models--{model_name.replace('/', '--')}"
        model_path = Path(cache_dir) / model_dir_name
        
        if model_path.exists():
            # Get size
            size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
            if size_mb > 1024:
                size_str = f"{size_mb/1024:.1f}GB"
            else:
                size_str = f"{size_mb:.0f}MB"
            
            print(f"  ✓ {model_name}")
            print(f"    {description} - Downloaded: {size_str}")
            downloaded[model_name] = True
        else:
            print(f"  ✗ {model_name}")
            print(f"    {description} - NOT DOWNLOADED")
            downloaded[model_name] = False
    
    return downloaded

def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("\n4. Python Dependencies:")
    
    dependencies = {
        "einops": "Required for CodeRankEmbed and jina-embeddings-v3",
        "InstructorEmbedding": "Required for instructor-large",
        "sentence_transformers": "Base requirement for all models"
    }
    
    all_installed = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"  ✓ {package}: Installed")
        except ImportError:
            print(f"  ✗ {package}: NOT INSTALLED - {description}")
            all_installed = False
    
    return all_installed

def verify_embeddings_import():
    """Try to import and check the embeddings modules"""
    print("\n5. Module Import Test:")
    
    try:
        from src.utils.embeddings import UnifiedEmbeddingsManager
        print("  ✓ UnifiedEmbeddingsManager imported successfully")
        
        # Check if specialized embeddings are available
        from src.utils import embeddings
        if hasattr(embeddings, 'SPECIALIZED_EMBEDDINGS_AVAILABLE'):
            print(f"  ✓ Specialized embeddings available: {embeddings.SPECIALIZED_EMBEDDINGS_AVAILABLE}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to import embeddings: {e}")
        return False

def main():
    """Run all configuration checks"""
    
    # Check environment
    env_enabled = check_env_config()
    
    # Check server config
    config_enabled = check_server_config()
    
    # Check downloaded models
    downloaded_models = check_models_downloaded()
    
    # Check Python dependencies
    deps_ok = check_python_dependencies()
    
    # Try imports
    imports_ok = verify_embeddings_import()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    all_ready = True
    
    if env_enabled:
        print("✓ Specialized embeddings ENABLED in environment")
    else:
        print("✗ Specialized embeddings DISABLED in environment")
        all_ready = False
    
    if config_enabled:
        print("✓ Specialized embeddings ENABLED in server config")
    else:
        print("⚠️  Specialized embeddings not configured in server_config.json")
    
    # Check critical models
    critical_models = [
        "nomic-ai/CodeRankEmbed",
        "jinaai/jina-embeddings-v3", 
        "hkunlp/instructor-large",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    missing_models = [m for m in critical_models if not downloaded_models.get(m, False)]
    if missing_models:
        print(f"✗ Missing {len(missing_models)} critical models:")
        for model in missing_models:
            print(f"  - {model}")
        all_ready = False
    else:
        print("✓ All critical models downloaded")
    
    if not deps_ok:
        print("✗ Missing Python dependencies")
        print("  Run: uv pip install -e '.[models]'")
        all_ready = False
    else:
        print("✓ All Python dependencies installed")
    
    if not imports_ok:
        print("✗ Module import failed")
        all_ready = False
    else:
        print("✓ Modules import successfully")
    
    print("\n" + "=" * 70)
    if all_ready and env_enabled:
        print("✅ Specialized embeddings are READY TO USE!")
        print("\nWhen you restart Claude Code with MCP, it will use:")
        print("  - CodeRankEmbed for code files")
        print("  - jina-embeddings-v3 for config files")
        print("  - instructor-large for documentation")
        print("  - all-MiniLM-L6-v2 for general/unknown files")
    else:
        print("⚠️  Specialized embeddings are NOT fully configured")
        print("\nTo enable:")
        print("  1. Set QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=true in .env")
        print("  2. Download missing models: ./scripts/download_models.sh")
        print("  3. Install dependencies: uv pip install -e '.[models]'")
    print("=" * 70)
    
    return 0 if all_ready else 1

if __name__ == "__main__":
    sys.exit(main())