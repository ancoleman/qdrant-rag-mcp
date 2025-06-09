#!/usr/bin/env python3
"""Test Apple Silicon optimizations via HTTP API"""

import os
import sys
import time
import subprocess
import requests
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def start_http_server():
    """Start the HTTP server in the background"""
    print("Starting HTTP server...")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent / "src")
    
    # Start the server
    process = subprocess.Popen(
        [sys.executable, "-m", "http_server"],
        cwd=str(Path(__file__).parent.parent / "src"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8081/health")
            if response.status_code == 200:
                print("✅ HTTP server started successfully")
                return process
        except:
            time.sleep(0.5)
    
    # If we get here, server failed to start
    process.terminate()
    stdout, stderr = process.communicate()
    print(f"Server stdout: {stdout.decode()}")
    print(f"Server stderr: {stderr.decode()}")
    raise RuntimeError("Failed to start HTTP server")


def test_apple_silicon_status(base_url="http://localhost:8081"):
    """Test Apple Silicon status endpoint"""
    print("\n=== Testing Apple Silicon Status ===")
    
    try:
        response = requests.get(f"{base_url}/apple_silicon_status")
        assert response.status_code == 200, f"Status code: {response.status_code}"
        
        data = response.json()
        print(f"Apple Silicon detected: {data.get('is_apple_silicon', False)}")
        
        if data.get('is_apple_silicon'):
            print("\nApple Silicon Configuration:")
            
            # Memory limits
            if 'memory_limits' in data:
                limits = data['memory_limits']
                print(f"  Total limit: {limits.get('total_mb', 'N/A')}MB")
                print(f"  Cleanup threshold: {limits.get('cleanup_threshold_mb', 'N/A')}MB")
                print(f"  Aggressive threshold: {limits.get('aggressive_threshold_mb', 'N/A')}MB")
            
            # MPS status
            print(f"\n  MPS available: {data.get('mps_available', False)}")
            print(f"  Unified memory: {data.get('unified_memory', False)}")
            print(f"  Memory pressure: {data.get('memory_pressure', 'Unknown')}")
            
            # Embeddings info
            if 'embeddings' in data:
                emb = data['embeddings']
                print(f"\nEmbeddings Manager:")
                print(f"  Device: {emb.get('device', 'Unknown')}")
                print(f"  Max models: {emb.get('max_models_in_memory', 'N/A')}")
                print(f"  Memory limit: {emb.get('memory_limit_gb', 'N/A')}GB")
                print(f"  Models loaded: {emb.get('models_loaded', 0)}")
                print(f"  Memory used: {emb.get('total_memory_used_gb', 0):.2f}GB")
            
            # System memory
            if 'system_memory' in data:
                mem = data['system_memory']
                print(f"\nSystem Memory:")
                print(f"  Total: {mem.get('total_gb', 0):.1f}GB")
                print(f"  Available: {mem.get('available_gb', 0):.1f}GB")
                print(f"  Used: {mem.get('percent_used', 0):.1f}%")
        else:
            print("Not running on Apple Silicon")
        
        return data
        
    except Exception as e:
        print(f"❌ Error testing Apple Silicon status: {e}")
        return None


def test_apple_silicon_cleanup(base_url="http://localhost:8081", level="standard"):
    """Test Apple Silicon cleanup endpoint"""
    print(f"\n=== Testing Apple Silicon Cleanup ({level}) ===")
    
    try:
        response = requests.post(
            f"{base_url}/apple_silicon_cleanup",
            json={"level": level}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "error" in data:
                print(f"⚠️  {data['error']}")
                return data
            
            print(f"Cleanup level: {data.get('cleanup_level', 'Unknown')}")
            
            # Before stats
            if 'before' in data:
                before = data['before']
                print(f"\nBefore cleanup:")
                print(f"  Available: {before.get('available_gb', 0):.2f}GB")
                print(f"  Process: {before.get('process_gb', 0):.2f}GB")
                print(f"  System: {before.get('system_percent', 0):.1f}%")
            
            # After stats
            if 'after' in data:
                after = data['after']
                print(f"\nAfter cleanup:")
                print(f"  Available: {after.get('available_gb', 0):.2f}GB")
                print(f"  Process: {after.get('process_gb', 0):.2f}GB")
                print(f"  System: {after.get('system_percent', 0):.1f}%")
            
            # Memory freed
            if 'memory_freed' in data:
                freed = data['memory_freed']
                print(f"\nMemory freed:")
                print(f"  System: {freed.get('system_gb', 0):.3f}GB")
                print(f"  Process: {freed.get('process_gb', 0):.3f}GB")
            
            return data
        else:
            print(f"❌ Cleanup failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing cleanup: {e}")
        return None


def test_indexing_with_monitoring(base_url="http://localhost:8081"):
    """Test indexing while monitoring memory"""
    print("\n=== Testing Indexing with Memory Monitoring ===")
    
    # Get initial status
    initial_status = test_apple_silicon_status(base_url)
    if not initial_status:
        return
    
    initial_available = initial_status.get('system_memory', {}).get('available_gb', 0)
    
    # Index a test file
    test_file = Path(__file__)
    print(f"\nIndexing test file: {test_file}")
    
    try:
        response = requests.post(
            f"{base_url}/index_code",
            json={"file_path": str(test_file)}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Indexed {result.get('indexed', 0)} chunks")
        else:
            print(f"❌ Indexing failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
    
    # Check memory after indexing
    time.sleep(1)  # Let things settle
    after_status = test_apple_silicon_status(base_url)
    if after_status:
        after_available = after_status.get('system_memory', {}).get('available_gb', 0)
        memory_used = initial_available - after_available
        print(f"\nMemory used during indexing: {memory_used:.3f}GB")


def main():
    """Run all Apple Silicon tests"""
    print("🍎 Apple Silicon Optimization Tests via HTTP API")
    print("=" * 50)
    
    # Start HTTP server
    server_process = None
    try:
        server_process = start_http_server()
        
        # Run tests
        base_url = "http://localhost:8081"
        
        # Test 1: Get status
        status = test_apple_silicon_status(base_url)
        
        # Test 2: Standard cleanup (only if on Apple Silicon)
        if status and status.get('is_apple_silicon'):
            test_apple_silicon_cleanup(base_url, "standard")
            
            # Test 3: Indexing with monitoring
            test_indexing_with_monitoring(base_url)
            
            # Test 4: Aggressive cleanup
            test_apple_silicon_cleanup(base_url, "aggressive")
        else:
            print("\n⚠️  Skipping cleanup tests - not on Apple Silicon")
        
        print("\n✅ All tests completed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    finally:
        # Clean up server
        if server_process:
            print("\nShutting down HTTP server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("Server stopped")


if __name__ == "__main__":
    main()