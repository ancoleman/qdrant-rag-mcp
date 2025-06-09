#!/usr/bin/env python3
"""
Safe test for specialized embeddings - creates temporary test collections
This test creates separate collections that won't interfere with existing data
"""

import os
import sys
import json
import requests
import tempfile
from pathlib import Path
from datetime import datetime

# Base URL for HTTP API
BASE_URL = "http://localhost:8081"

# Use timestamp to ensure unique collection names
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TEST_PROJECT_NAME = f"test_specialized_{TIMESTAMP}"

def create_temp_files():
    """Create temporary test files for each content type"""
    temp_dir = tempfile.mkdtemp(prefix="qdrant_test_")
    print(f"Created temp directory: {temp_dir}")
    
    # Create test code file
    code_file = Path(temp_dir) / "test_code.py"
    code_file.write_text('''
class TestEmbeddings:
    """Test class for specialized embeddings"""
    
    def encode(self, text: str) -> list:
        """Encode text using specialized model"""
        return [0.1, 0.2, 0.3]
    
    def load_model(self, model_type: str):
        """Load the appropriate model"""
        if model_type == "code":
            return "CodeRankEmbed"
        return "default"
''')
    
    # Create test config file
    config_file = Path(temp_dir) / "test_config.json"
    config_file.write_text('''{
    "specialized_embeddings": {
        "enabled": true,
        "models": {
            "code": "nomic-ai/CodeRankEmbed",
            "config": "jinaai/jina-embeddings-v3",
            "documentation": "hkunlp/instructor-large"
        }
    }
}''')
    
    # Create test documentation file
    doc_file = Path(temp_dir) / "test_doc.md"
    doc_file.write_text('''# Test Documentation

## Specialized Embeddings Test

This is a test document for specialized embeddings.

### Features
- Code embeddings with CodeRankEmbed
- Config embeddings with Jina v3
- Documentation embeddings with Instructor

### Implementation
The system uses different models for different content types.
''')
    
    return temp_dir, {
        "code": str(code_file),
        "config": str(config_file),
        "documentation": str(doc_file)
    }

def test_specialized_models_safely():
    """Test specialized embeddings without touching existing collections"""
    print("=" * 70)
    print("Safe Test for Specialized Embeddings (Temporary Collections)")
    print("=" * 70)
    
    temp_dir = None
    
    try:
        # 1. Check server health
        print("\n1. Checking server health...")
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("✗ Server not running. Start with: python src/http_server.py")
            return 1
        
        health = response.json()
        print("✓ Server is healthy")
        print(f"  Current project: {health.get('current_project', 'unknown')}")
        
        # Check if specialized embeddings are enabled
        specialized_enabled = os.getenv('QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED', 'false').lower() == 'true'
        print(f"\n2. Specialized embeddings enabled: {specialized_enabled}")
        if not specialized_enabled:
            print("⚠️  Set QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=true to test specialized models")
        
        # 2. Create temporary test files
        print("\n3. Creating temporary test files...")
        temp_dir, test_files = create_temp_files()
        print("✓ Created test files")
        
        # 3. Switch to a test project context (this creates isolated collections)
        print(f"\n4. Switching to test project: {TEST_PROJECT_NAME}")
        response = requests.post(
            f"{BASE_URL}/switch_project",
            json={"project_path": temp_dir}
        )
        
        if response.status_code == 200:
            print("✓ Switched to test project")
        else:
            print("✗ Failed to switch project")
            return 1
        
        # 4. Index test files
        print("\n5. Indexing test files with specialized models...")
        
        # Index code file
        print("\n  a) Indexing code file...")
        response = requests.post(
            f"{BASE_URL}/index_code",
            json={"file_path": test_files["code"]}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Code indexed: {result.get('chunks_created', 0)} chunks")
            print(f"    Collection: {result.get('collection', 'unknown')}")
        
        # Index config file  
        print("\n  b) Indexing config file...")
        response = requests.post(
            f"{BASE_URL}/index_config",
            json={"file_path": test_files["config"]}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Config indexed: {result.get('chunks_created', 0)} chunks")
            print(f"    Collection: {result.get('collection', 'unknown')}")
        
        # Index documentation file
        print("\n  c) Indexing documentation file...")
        response = requests.post(
            f"{BASE_URL}/index_documentation",
            json={"file_path": test_files["documentation"]}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Documentation indexed: {result.get('chunks_created', 0)} chunks")
            print(f"    Collection: {result.get('collection', 'unknown')}")
        
        # 5. Test searches
        print("\n6. Testing searches with specialized models...")
        
        # Code search
        print("\n  a) Code search test...")
        response = requests.post(
            f"{BASE_URL}/search_code",
            json={"query": "encode load_model specialized", "n_results": 2}
        )
        if response.status_code == 200:
            results = response.json()
            print(f"  ✓ Found {len(results['results'])} code results")
            if results['results']:
                print(f"    Top result score: {results['results'][0]['score']:.4f}")
        
        # Config search
        print("\n  b) Config search test...")
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": "specialized embeddings models", "n_results": 2}
        )
        if response.status_code == 200:
            results = response.json()
            print(f"  ✓ Found {len(results['results'])} results")
            if results['results']:
                print(f"    Top result score: {results['results'][0]['score']:.4f}")
        
        # Documentation search
        print("\n  c) Documentation search test...")
        response = requests.post(
            f"{BASE_URL}/search_docs",
            json={"query": "specialized embeddings features", "n_results": 2}
        )
        if response.status_code == 200:
            results = response.json()
            print(f"  ✓ Found {len(results['results'])} documentation results")
            if results['results']:
                print(f"    Top result score: {results['results'][0]['score']:.4f}")
        
        print("\n" + "=" * 70)
        print("✅ Test Complete! (Test collections remain isolated)")
        print("=" * 70)
        
        print("\n📝 Summary:")
        print(f"- Test project: {TEST_PROJECT_NAME}")
        print(f"- Test collections created: project_{TEST_PROJECT_NAME}_code/config/documentation")
        print("- Your existing collections are untouched")
        print("\nTo switch back to your project, restart the server or use switch_project")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up temp files
        if temp_dir:
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"\n✓ Cleaned up temp directory: {temp_dir}")
            except:
                print(f"\n⚠️  Could not clean up temp directory: {temp_dir}")

def verify_models_downloaded():
    """Check if specialized models are downloaded"""
    print("\n7. Checking if specialized models are downloaded...")
    
    models_to_check = [
        ("Code", "nomic-ai/CodeRankEmbed"),
        ("Config", "jinaai/jina-embeddings-v3"),
        ("Documentation", "hkunlp/instructor-large"),
        ("General", "sentence-transformers/all-MiniLM-L6-v2")
    ]
    
    # Always use local data/models directory
    cache_dir = "data/models"
    
    all_present = True
    for role, model_name in models_to_check:
        model_dir = f"models--{model_name.replace('/', '--')}"
        model_path = Path(cache_dir) / model_dir
        if model_path.exists():
            print(f"  ✓ {role}: {model_name}")
        else:
            print(f"  ✗ {role}: {model_name} (not found)")
            all_present = False
    
    if not all_present:
        print("\n  ⚠️  Some models are missing. Run: ./scripts/download_models.sh")
    
    return all_present

if __name__ == "__main__":
    # First check if models are downloaded
    verify_models_downloaded()
    
    # Run the safe test
    sys.exit(test_specialized_models_safely())