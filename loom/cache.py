"""
Fast LRU (Least Recently Used) Cache implementation.

Uses lru-dict (C implementation) for maximum performance.
"""

from lru import LRU
from typing import Any

_SENTINEL = object()


class LRUCache:
    """Fast LRU cache wrapping lru-dict C implementation.

    Uses lru-dict for O(1) get/set/delete with C-level performance.
    When capacity is reached, least recently used items are evicted.

    Usage:
        cache = LRUCache(capacity=1000)

        # Set items
        cache['key'] = 'value'

        # Get items
        value = cache['key']
        value = cache.get('key', default=None)

        # Check membership
        if 'key' in cache:
            ...

        # Delete items
        del cache['key']

    Performance:
        - Get: O(1) - C implementation
        - Set: O(1) - C implementation
        - Delete: O(1) - C implementation
        - Memory: O(capacity)
    """

    def __init__(self, capacity: int):
        """Initialize LRU cache.

        Args:
            capacity: Maximum number of items to store (must be > 0)
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self._cache = LRU(capacity)
        self.hits = 0
        self.misses = 0

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value for key.

        Args:
            key: Key to look up
            default: Value to return if key not found

        Returns:
            Value if found, else default
        """
        result = self._cache.get(key, _SENTINEL)
        if result is _SENTINEL:
            self.misses += 1
            return default
        self.hits += 1
        return result

    def set(self, key: Any, value: Any) -> None:
        """Set key to value.

        Args:
            key: Key to set
            value: Value to store
        """
        self._cache[key] = value

    def invalidate(self, key: Any) -> None:
        """Invalidate (remove) a key from cache.

        Args:
            key: Key to invalidate
        """
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __getitem__(self, key: Any) -> Any:
        """Get item using cache[key] syntax."""
        result = self._cache.get(key, _SENTINEL)
        if result is _SENTINEL:
            self.misses += 1
            raise KeyError(key)
        self.hits += 1
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item using cache[key] = value syntax."""
        self._cache[key] = value

    def __delitem__(self, key: Any) -> None:
        """Delete item using del cache[key] syntax."""
        del self._cache[key]

    def __contains__(self, key: Any) -> bool:
        """Check if key in cache using 'key in cache' syntax."""
        return key in self._cache

    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def __bool__(self) -> bool:
        """Treat an LRUCache instance as truthy even when empty.

        Many call sites use `if cache:` to mean "is caching enabled?".
        Since LRUCache implements __len__, an empty cache would otherwise be
        falsy and caching would never warm up.
        """
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"LRUCache(capacity={self.capacity}, size={len(self)})"


class NamespacedCache:
    """Prefixed view into a shared LRUCache.

    Multiple data structures share one LRU cache. Each structure gets
    its own namespace so keys can't collide. A generation counter lets
    clear() logically invalidate all entries without scanning the cache
    (old entries are simply never matched again and get evicted by LRU).

    Key format: (namespace, generation, item_key)

    Usage:
        shared = LRUCache(capacity=50_000)
        cache_a = NamespacedCache(shared, "dict_users")
        cache_b = NamespacedCache(shared, "dict_posts")

        cache_a["alice"] = 42
        cache_b["hello"] = 99
        cache_a.clear()          # only invalidates cache_a entries
    """

    def __init__(self, shared: LRUCache, namespace: str):
        self._shared = shared
        self._ns = namespace
        self._gen = 0

    def _k(self, key: Any):
        return (self._ns, self._gen, key)

    def get(self, key: Any, default: Any = None) -> Any:
        return self._shared.get(self._k(key), default)

    def set(self, key: Any, value: Any) -> None:
        self._shared[self._k(key)] = value

    def invalidate(self, key: Any) -> None:
        self._shared.invalidate(self._k(key))

    def clear(self) -> None:
        """Logically invalidate all entries by bumping the generation.

        Old entries remain in the shared cache until evicted by LRU
        pressure — no O(n) scan needed.
        """
        self._gen += 1

    @property
    def hit_rate(self) -> float:
        return self._shared.hit_rate

    def __setitem__(self, key: Any, value: Any) -> None:
        self._shared[self._k(key)] = value

    def __getitem__(self, key: Any) -> Any:
        return self._shared[self._k(key)]

    def __delitem__(self, key: Any) -> None:
        del self._shared[self._k(key)]

    def __contains__(self, key: Any) -> bool:
        return self._k(key) in self._shared

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return 0  # accurate count per-namespace would require a scan

    def __repr__(self) -> str:
        return f"NamespacedCache(ns={self._ns!r}, gen={self._gen}, shared={self._shared!r})"


class NullCache:
    """Null object pattern - cache that does nothing.

    Used when caching is disabled (cache_size=0).
    Provides same interface as LRUCache but stores nothing.
    """

    def __init__(self):
        self.hits = 0
        self.misses = 0

    def get(self, key: Any, default: Any = None) -> Any:
        self.misses += 1
        return default

    def set(self, key: Any, value: Any) -> None:
        pass

    def delete(self, key: Any) -> None:
        pass

    def invalidate(self, key: Any) -> None:
        pass

    def clear(self) -> None:
        pass

    def __getitem__(self, key: Any) -> Any:
        self.misses += 1
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def __delitem__(self, key: Any) -> None:
        pass

    def __contains__(self, key: Any) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False

    @property
    def hit_rate(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "NullCache(disabled)"
