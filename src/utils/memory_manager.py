"""
Unified Memory Management System for Qdrant RAG

This module provides centralized memory management for all components including:
- Model loading and caching (specialized embeddings)
- Semantic result caching (progressive context)
- Context tracking and session data
- Periodic garbage collection
- Memory pressure handling
"""

import gc
import threading
import time
import psutil
import subprocess
from typing import Dict, Any, Optional, Callable, List
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import json

from .logging import get_project_logger

logger = get_project_logger()


class MemoryRegistry:
    """Registry for all memory-consuming components"""
    
    def __init__(self):
        self.components: Dict[str, 'MemoryComponent'] = {}
        self.lock = threading.RLock()
    
    def register(self, name: str, component: 'MemoryComponent'):
        """Register a memory-consuming component"""
        with self.lock:
            self.components[name] = component
            logger.info(f"Registered memory component: {name}")
    
    def unregister(self, name: str):
        """Unregister a component"""
        with self.lock:
            if name in self.components:
                del self.components[name]
                logger.info(f"Unregistered memory component: {name}")
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all components in MB"""
        with self.lock:
            total = 0.0
            for name, component in self.components.items():
                try:
                    usage = component.get_memory_usage()
                    total += usage
                except Exception as e:
                    logger.warning(f"Failed to get memory usage for {name}: {e}")
            return total
    
    def get_component_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed stats for all components"""
        with self.lock:
            stats = {}
            for name, component in self.components.items():
                try:
                    stats[name] = {
                        'memory_mb': component.get_memory_usage(),
                        'items_count': component.get_item_count(),
                        'last_cleanup': component.last_cleanup,
                        'cleanup_count': component.cleanup_count
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats for {name}: {e}")
                    stats[name] = {'error': str(e)}
            return stats


class MemoryComponent:
    """Base class for memory-managed components"""
    
    def __init__(self, name: str, max_memory_mb: float = 500.0):
        self.name = name
        self.max_memory_mb = max_memory_mb
        self.last_cleanup = None
        self.cleanup_count = 0
        self.lock = threading.RLock()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB - must be implemented by subclasses"""
        raise NotImplementedError
    
    def get_item_count(self) -> int:
        """Get number of items in cache/storage - must be implemented by subclasses"""
        raise NotImplementedError
    
    def cleanup(self, aggressive: bool = False) -> int:
        """Perform cleanup and return number of items removed"""
        raise NotImplementedError
    
    def mark_cleanup(self):
        """Mark that cleanup was performed"""
        self.last_cleanup = datetime.now()
        self.cleanup_count += 1


class LRUMemoryCache(MemoryComponent):
    """Generic LRU cache with memory management"""
    
    def __init__(self, name: str, max_memory_mb: float = 200.0, max_items: int = 1000):
        super().__init__(name, max_memory_mb)
        self.max_items = max_items
        self.cache: OrderedDict = OrderedDict()
        self.memory_estimates: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any, memory_estimate_mb: float = 0.001):
        """Put item in cache with memory estimate"""
        with self.lock:
            # Remove if already exists to update position
            if key in self.cache:
                del self.cache[key]
                if key in self.memory_estimates:
                    del self.memory_estimates[key]
            
            # Add to cache
            self.cache[key] = value
            self.memory_estimates[key] = memory_estimate_mb
            
            # Enforce limits
            self._enforce_limits()
    
    def _enforce_limits(self):
        """Enforce item count and memory limits"""
        # Check item count
        while len(self.cache) > self.max_items:
            self._evict_lru()
        
        # Check memory usage
        while self.get_memory_usage() > self.max_memory_mb:
            if not self.cache:
                break
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            if key in self.memory_estimates:
                del self.memory_estimates[key]
    
    def get_memory_usage(self) -> float:
        """Get estimated memory usage in MB"""
        with self.lock:
            return sum(self.memory_estimates.values())
    
    def get_item_count(self) -> int:
        """Get number of items in cache"""
        with self.lock:
            return len(self.cache)
    
    def cleanup(self, aggressive: bool = False) -> int:
        """Perform cleanup"""
        with self.lock:
            if aggressive:
                # Remove 50% of items
                target_remove = len(self.cache) // 2
            else:
                # Remove 20% of items
                target_remove = len(self.cache) // 5
            
            removed = 0
            for _ in range(target_remove):
                if not self.cache:
                    break
                self._evict_lru()
                removed += 1
            
            self.mark_cleanup()
            return removed
    
    def clear(self):
        """Clear all items"""
        with self.lock:
            self.cache.clear()
            self.memory_estimates.clear()


class UnifiedMemoryManager:
    """Unified memory manager for the entire system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Get memory management config from server config
        if config is None:
            from config import get_config
            server_config = get_config()
            config = server_config.get('memory_management', {})
        
        self.config = config
        self.enabled = self._parse_bool(config.get('enabled', 'true'))
        
        # Memory limits - parse from config with type conversion
        self.total_memory_limit_mb = float(config.get('total_memory_limit_mb', 8000))
        self.cleanup_threshold_mb = float(config.get('cleanup_threshold_mb', 6000))
        self.aggressive_threshold_mb = float(config.get('aggressive_threshold_mb', 7000))
        
        # Cleanup intervals
        self.cleanup_interval_seconds = int(config.get('cleanup_interval_seconds', 180))
        self.gc_interval_seconds = int(config.get('gc_interval_seconds', 300))
        
        # Component limits
        self.component_limits = config.get('component_limits', {})
        
        # Component registry
        self.registry = MemoryRegistry()
        
        # Monitoring
        self.cleanup_thread = None
        self.gc_thread = None
        self.running = False
        self.stats_history: List[Dict[str, Any]] = []
        self.max_stats_history = 100
        
        # Callbacks for memory pressure
        self.memory_pressure_callbacks: List[Callable[[float], None]] = []
        
        # Apple Silicon optimization
        self.is_apple_silicon = self._detect_apple_silicon()
        if self.is_apple_silicon:
            self._apply_apple_silicon_optimizations()
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean from string or bool"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def start(self):
        """Start memory management threads"""
        if not self.enabled:
            logger.info("Memory management is disabled")
            return
            
        if self.running:
            return
        
        self.running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="MemoryCleanup"
        )
        self.cleanup_thread.start()
        
        # Start GC thread
        self.gc_thread = threading.Thread(
            target=self._gc_loop,
            daemon=True,
            name="GarbageCollection"
        )
        self.gc_thread.start()
        
        logger.info(f"Started unified memory manager with limits: "
                   f"total={self.total_memory_limit_mb}MB, "
                   f"cleanup={self.cleanup_threshold_mb}MB, "
                   f"aggressive={self.aggressive_threshold_mb}MB")
    
    def stop(self):
        """Stop memory management threads"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        if self.gc_thread:
            self.gc_thread.join(timeout=5)
        logger.info("Stopped unified memory manager")
    
    def register_component(self, name: str, component: MemoryComponent):
        """Register a memory-consuming component"""
        self.registry.register(name, component)
    
    def unregister_component(self, name: str):
        """Unregister a component"""
        self.registry.unregister(name)
    
    def add_memory_pressure_callback(self, callback: Callable[[float], None]):
        """Add callback for memory pressure events"""
        self.memory_pressure_callbacks.append(callback)
    
    def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while self.running:
            time.sleep(self.cleanup_interval_seconds)
            if self.running:
                self._perform_cleanup()
    
    def _gc_loop(self):
        """Periodic garbage collection loop"""
        while self.running:
            time.sleep(self.gc_interval_seconds)
            if self.running:
                self._perform_gc()
    
    def _get_system_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory
            virtual_mem = psutil.virtual_memory()
            
            return {
                'process_rss_mb': memory_info.rss / (1024**2),
                'process_vms_mb': memory_info.vms / (1024**2),
                'system_total_mb': virtual_mem.total / (1024**2),
                'system_available_mb': virtual_mem.available / (1024**2),
                'system_percent': virtual_mem.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get system memory stats: {e}")
            return {}
    
    def _perform_cleanup(self):
        """Perform memory cleanup based on usage"""
        try:
            # Get current memory usage
            system_stats = self._get_system_memory_usage()
            component_total = self.registry.get_total_memory_usage()
            process_memory = system_stats.get('process_rss_mb', 0)
            
            # Record stats
            stats = {
                'timestamp': datetime.now().isoformat(),
                'system': system_stats,
                'components': self.registry.get_component_stats(),
                'component_total_mb': component_total,
                'process_total_mb': process_memory
            }
            
            self.stats_history.append(stats)
            if len(self.stats_history) > self.max_stats_history:
                self.stats_history.pop(0)
            
            # Determine cleanup level
            cleanup_needed = False
            aggressive = False
            
            if process_memory > self.aggressive_threshold_mb:
                cleanup_needed = True
                aggressive = True
                logger.warning(f"Memory usage critical: {process_memory:.1f}MB > {self.aggressive_threshold_mb}MB")
            elif process_memory > self.cleanup_threshold_mb:
                cleanup_needed = True
                logger.info(f"Memory usage high: {process_memory:.1f}MB > {self.cleanup_threshold_mb}MB")
            
            # Notify callbacks about memory pressure
            if cleanup_needed:
                memory_pressure = process_memory / self.total_memory_limit_mb
                for callback in self.memory_pressure_callbacks:
                    try:
                        callback(memory_pressure)
                    except Exception as e:
                        logger.error(f"Memory pressure callback failed: {e}")
            
            # Perform cleanup if needed
            if cleanup_needed:
                total_removed = 0
                for name, component in self.registry.components.items():
                    try:
                        removed = component.cleanup(aggressive=aggressive)
                        total_removed += removed
                        logger.info(f"Cleaned up {removed} items from {name}")
                    except Exception as e:
                        logger.error(f"Cleanup failed for {name}: {e}")
                
                # Force garbage collection after cleanup
                gc.collect()
                
                # Log results
                new_process_memory = self._get_system_memory_usage().get('process_rss_mb', 0)
                logger.info(f"Cleanup complete: removed {total_removed} items, "
                           f"memory {process_memory:.1f}MB -> {new_process_memory:.1f}MB")
        
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def _perform_gc(self):
        """Perform garbage collection"""
        try:
            before_memory = self._get_system_memory_usage().get('process_rss_mb', 0)
            
            # Run garbage collection
            collected = gc.collect()
            gc.collect()  # Run twice for thorough cleanup
            
            after_memory = self._get_system_memory_usage().get('process_rss_mb', 0)
            
            if collected > 0:
                logger.debug(f"GC collected {collected} objects, "
                            f"memory {before_memory:.1f}MB -> {after_memory:.1f}MB")
        
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon (M1/M2/M3/M4)"""
        try:
            import platform
            import subprocess
            
            if platform.system() != "Darwin":
                return False
            
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"], 
                capture_output=True, text=True, timeout=5
            )
            
            # If the command returns "1", it's Apple Silicon
            if result.returncode == 0 and result.stdout.strip() == "1":
                logger.info("Apple Silicon detected - enabling MPS optimizations")
                return True
                
            return False
            
        except Exception as e:
            logger.debug(f"Could not detect Apple Silicon: {e}")
            return False
    
    def _apply_apple_silicon_optimizations(self):
        """Apply memory optimizations for Apple Silicon with unified memory"""
        logger.info("Applying Apple Silicon memory optimizations")
        
        # Get total system memory
        try:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Conservative limits for unified memory architecture
            # Leave at least 2-3GB for system and GPU operations
            if total_memory_gb <= 8:  # M1/M2 base models
                self.total_memory_limit_mb = 5000
                self.cleanup_threshold_mb = 3500
                self.aggressive_threshold_mb = 4000
            elif total_memory_gb <= 16:  # M1/M2 Pro base models
                self.total_memory_limit_mb = 10000
                self.cleanup_threshold_mb = 7000
                self.aggressive_threshold_mb = 8500
            elif total_memory_gb <= 24:  # M1/M2/M3 Pro with 18GB
                self.total_memory_limit_mb = 14000
                self.cleanup_threshold_mb = 10000
                self.aggressive_threshold_mb = 12000
            else:  # M1/M2 Max, Ultra models
                self.total_memory_limit_mb = 20000
                self.cleanup_threshold_mb = 15000
                self.aggressive_threshold_mb = 18000
            
            # Update component limits for Apple Silicon
            self.component_limits['embeddings'] = self.total_memory_limit_mb * 0.3  # 30% for models
            self.component_limits['progressive_context'] = self.total_memory_limit_mb * 0.2  # 20% for cache
            self.component_limits['context_tracking'] = self.total_memory_limit_mb * 0.1  # 10% for tracking
            
            logger.info(f"Apple Silicon limits set for {total_memory_gb:.1f}GB system: "
                       f"total={self.total_memory_limit_mb}MB, "
                       f"cleanup={self.cleanup_threshold_mb}MB, "
                       f"aggressive={self.aggressive_threshold_mb}MB")
            
        except Exception as e:
            logger.warning(f"Could not optimize for Apple Silicon: {e}")
    
    def get_apple_silicon_memory_status(self) -> Dict[str, Any]:
        """Get Apple Silicon specific memory status including MPS usage"""
        status = {
            "is_apple_silicon": self.is_apple_silicon,
            "unified_memory": False,
            "mps_available": False,
            "memory_pressure": None
        }
        
        if not self.is_apple_silicon:
            return status
        
        try:
            import torch
            
            # Check MPS availability
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                status["mps_available"] = True
                status["unified_memory"] = True
                
                # Try to get memory pressure (macOS specific)
                try:
                    result = subprocess.run(
                        ["memory_pressure"], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        output = result.stdout.lower()
                        if "system-wide memory pressure" in output:
                            if "critical" in output:
                                status["memory_pressure"] = "critical"
                            elif "warning" in output:
                                status["memory_pressure"] = "warning"
                            else:
                                status["memory_pressure"] = "normal"
                except:
                    pass
                    
        except ImportError:
            pass
        
        return status
    
    def perform_apple_silicon_cleanup(self, level: str = "standard"):
        """Perform Apple Silicon specific cleanup including MPS cache"""
        if not self.is_apple_silicon:
            return
        
        try:
            import torch
            
            # Standard cleanup
            self._perform_cleanup()
            
            # MPS specific cleanup
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS cache")
                
                if level == "aggressive" and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                    logger.debug("Synchronized MPS operations")
                    
        except Exception as e:
            logger.warning(f"Apple Silicon cleanup failed: {e}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        system_stats = self._get_system_memory_usage()
        component_stats = self.registry.get_component_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': system_stats,
            'components': component_stats,
            'component_total_mb': self.registry.get_total_memory_usage(),
            'limits': {
                'total_mb': self.total_memory_limit_mb,
                'cleanup_threshold_mb': self.cleanup_threshold_mb,
                'aggressive_threshold_mb': self.aggressive_threshold_mb
            },
            'history': self.stats_history[-10:] if self.stats_history else []
        }
        
        # Add Apple Silicon status if applicable
        if self.is_apple_silicon:
            report['apple_silicon'] = self.get_apple_silicon_memory_status()
        
        return report
    
    def save_memory_report(self, file_path: Optional[Path] = None):
        """Save memory report to file"""
        if file_path is None:
            file_path = Path.home() / '.mcp-servers' / 'qdrant-rag' / 'memory_reports' / f'memory_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.get_memory_report(), f, indent=2)
        
        logger.info(f"Saved memory report to {file_path}")


# Global instance
_memory_manager: Optional[UnifiedMemoryManager] = None


def get_memory_manager(config: Optional[Dict[str, Any]] = None) -> UnifiedMemoryManager:
    """Get or create the global memory manager instance"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = UnifiedMemoryManager(config)
        _memory_manager.start()
        
        # Register cleanup on exit
        import atexit
        atexit.register(_memory_manager.stop)
    
    return _memory_manager