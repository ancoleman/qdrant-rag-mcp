#!/usr/bin/env python3
"""
Simple script to check memory usage of the Qdrant RAG MCP server
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_manager import get_memory_manager
import json

def main():
    print("Qdrant RAG Memory Report")
    print("=" * 50)
    
    # Get memory manager
    manager = get_memory_manager()
    
    # Get memory report
    report = manager.get_memory_report()
    
    # Print system memory
    system = report['system']
    print(f"\nSystem Memory:")
    print(f"  Process RSS: {system.get('process_rss_mb', 0):.1f} MB")
    print(f"  Process VMS: {system.get('process_vms_mb', 0):.1f} MB")
    print(f"  System Available: {system.get('system_available_mb', 0):.1f} MB")
    print(f"  System Usage: {system.get('system_percent', 0):.1f}%")
    
    # Print component memory
    print(f"\nComponent Memory Usage:")
    components = report['components']
    total_component_mb = report['component_total_mb']
    
    for name, stats in components.items():
        if 'error' not in stats:
            print(f"  {name}:")
            print(f"    Memory: {stats['memory_mb']:.1f} MB")
            print(f"    Items: {stats['items_count']}")
            if stats['last_cleanup']:
                print(f"    Last Cleanup: {stats['last_cleanup']}")
    
    print(f"\n  Total Component Memory: {total_component_mb:.1f} MB")
    
    # Print limits
    limits = report['limits']
    print(f"\nMemory Limits:")
    print(f"  Total Limit: {limits['total_mb']} MB")
    print(f"  Cleanup Threshold: {limits['cleanup_threshold_mb']} MB")
    print(f"  Aggressive Threshold: {limits['aggressive_threshold_mb']} MB")
    
    # Check status
    process_mb = system.get('process_rss_mb', 0)
    if process_mb > limits['aggressive_threshold_mb']:
        print(f"\n⚠️  WARNING: Memory usage is CRITICAL!")
    elif process_mb > limits['cleanup_threshold_mb']:
        print(f"\n⚠️  WARNING: Memory usage is HIGH!")
    else:
        print(f"\n✅ Memory usage is normal")
    
    # Save report
    report_path = os.path.expanduser("~/.mcp-servers/qdrant-rag/memory_reports/latest.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    main()