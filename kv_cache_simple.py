"""
Simple KV Cache implementation from scratch.
Clean and minimal implementation focusing on core functionality.
"""

import time
from typing import Any, Optional
from collections import OrderedDict


class SimpleKVCache:
    """
    Simple Key-Value Cache with LRU eviction and basic TTL support.
    """
    
    def __init__(self, max_size: int = 100, default_ttl: Optional[float] = None):
        """
        Initialize simple KV Cache
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds (None for no expiry)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.expiry_times = {}
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Check if expired
        if self._is_expired(key):
            self._remove(key)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        # Handle TTL
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        
        # If key exists, update it
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                self._remove(oldest)
            
            self.cache[key] = value
        
        # Set expiry
        if ttl_to_use is not None:
            self.expiry_times[key] = time.time() + ttl_to_use
        elif key in self.expiry_times:
            del self.expiry_times[key]
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            self._remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items"""
        self.cache.clear()
        self.expiry_times.clear()
    
    def size(self) -> int:
        """Get current size"""
        return len(self.cache)
    
    def keys(self):
        """Get all keys"""
        return list(self.cache.keys())
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired"""
        if key not in self.expiry_times:
            return False
        return time.time() > self.expiry_times[key]
    
    def _remove(self, key: str) -> None:
        """Remove key from cache and expiry tracking"""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    # Magic methods for convenience
    def __len__(self):
        return self.size()
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key, value):
        self.set(key, value)


if __name__ == "__main__":
    print("Simple KV Cache Test")
    print("=" * 30)
    
    # Create cache
    cache = SimpleKVCache(max_size=3, default_ttl=2.0)
    
    # Basic operations
    cache.set("user1", {"name": "Alice", "age": 30})
    cache.set("user2", {"name": "Bob", "age": 25})
    cache.set("user3", {"name": "Charlie", "age": 35})
    
    print(f"Cache size: {cache.size()}")
    print(f"Keys: {cache.keys()}")
    
    # Test LRU eviction
    cache.set("user4", {"name": "David", "age": 40})
    print(f"After adding user4: {cache.keys()}")
    
    # Test get operations
    print(f"Get user2: {cache.get('user2')}")
    print(f"Get nonexistent: {cache.get('nonexistent')}")
    
    # Test TTL
    cache.set("temp", "temporary", ttl=1.0)
    print(f"Temp data immediately: {cache.get('temp')}")
    
    time.sleep(1.1)
    print(f"Temp data after 1.1s: {cache.get('temp')}")
    
    # Test dictionary-like access
    cache["api_key"] = "secret123"
    print(f"API key: {cache['api_key']}")
    print(f"'api_key' in cache: {'api_key' in cache}")
    
    # Test delete
    cache.delete("api_key")
    print(f"After deletion: {'api_key' in cache}")
    
    print(f"Final cache size: {len(cache)}")
