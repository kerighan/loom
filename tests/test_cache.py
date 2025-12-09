"""
Tests for LRU Cache implementation.

Tests:
- Basic operations (get, set, delete)
- LRU eviction
- Statistics tracking
- Edge cases
- Performance
"""

import pytest
from loom.cache import LRUCache, NullCache


class TestLRUCacheBasics:
    """Test basic cache operations."""

    def test_create_cache(self):
        """Test creating a cache."""
        cache = LRUCache(capacity=10)
        assert len(cache) == 0
        assert cache.capacity == 10

    def test_set_and_get(self):
        """Test setting and getting items."""
        cache = LRUCache(capacity=10)

        cache["key1"] = "value1"
        cache.set("key2", "value2")

        assert cache["key1"] == "value1"
        assert cache.get("key2") == "value2"
        assert len(cache) == 2

    def test_get_default(self):
        """Test get with default value."""
        cache = LRUCache(capacity=10)

        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_contains(self):
        """Test membership checking."""
        cache = LRUCache(capacity=10)

        cache["key"] = "value"

        assert "key" in cache
        assert "missing" not in cache

    def test_delete(self):
        """Test deleting items."""
        cache = LRUCache(capacity=10)

        cache["key"] = "value"
        assert "key" in cache

        del cache["key"]
        assert "key" not in cache
        assert len(cache) == 0

    def test_update_existing(self):
        """Test updating existing key."""
        cache = LRUCache(capacity=10)

        cache["key"] = "value1"
        cache["key"] = "value2"

        assert cache["key"] == "value2"
        assert len(cache) == 1  # Still only one item


class TestLRUEviction:
    """Test LRU eviction behavior."""

    def test_evict_when_full(self):
        """Test that least recently used item is evicted."""
        cache = LRUCache(capacity=3)

        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Cache is full, adding 'd' should evict 'a' (least recently used)
        cache["d"] = 4

        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache
        assert "d" in cache
        assert len(cache) == 3

    def test_access_updates_lru(self):
        """Test that accessing an item makes it most recently used."""
        cache = LRUCache(capacity=3)

        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Access 'a' to make it recently used
        _ = cache["a"]

        # Add 'd' - should evict 'b' (now least recently used)
        cache["d"] = 4

        assert "a" in cache  # Still there (was accessed)
        assert "b" not in cache  # Evicted
        assert "c" in cache
        assert "d" in cache

    def test_update_makes_recent(self):
        """Test that updating a key makes it most recently used."""
        cache = LRUCache(capacity=3)

        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Update 'a'
        cache["a"] = 10

        # Add 'd' - should evict 'b'
        cache["d"] = 4

        assert "a" in cache  # Still there (was updated)
        assert cache["a"] == 10
        assert "b" not in cache  # Evicted


class TestStatistics:
    """Test cache statistics."""

    def test_hit_tracking(self):
        """Test hit counting."""
        cache = LRUCache(capacity=10)

        cache["key"] = "value"

        _ = cache["key"]  # Hit
        _ = cache.get("key")  # Hit

        assert cache.hits == 2
        assert cache.misses == 0

    def test_miss_tracking(self):
        """Test miss counting."""
        cache = LRUCache(capacity=10)

        _ = cache.get("missing1")  # Miss
        _ = cache.get("missing2")  # Miss

        try:
            _ = cache["missing3"]  # Miss (raises KeyError)
        except KeyError:
            pass

        assert cache.hits == 0
        assert cache.misses == 3

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = LRUCache(capacity=10)

        cache["key"] = "value"

        _ = cache["key"]  # Hit
        _ = cache.get("missing")  # Miss
        _ = cache["key"]  # Hit

        assert cache.hit_rate == 2 / 3  # 2 hits, 1 miss

    def test_clear_resets_stats(self):
        """Test that clear resets statistics."""
        cache = LRUCache(capacity=10)

        cache["key"] = "value"
        _ = cache["key"]
        _ = cache.get("missing")

        cache.clear()

        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache) == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_capacity_one(self):
        """Test cache with capacity of 1."""
        cache = LRUCache(capacity=1)

        cache["a"] = 1
        assert "a" in cache

        cache["b"] = 2
        assert "a" not in cache
        assert "b" in cache

    def test_invalid_capacity(self):
        """Test that invalid capacity raises error."""
        with pytest.raises(ValueError):
            LRUCache(capacity=0)

        with pytest.raises(ValueError):
            LRUCache(capacity=-1)

    def test_delete_missing_key(self):
        """Test deleting non-existent key raises error."""
        cache = LRUCache(capacity=10)

        with pytest.raises(KeyError):
            del cache["missing"]

    def test_contains_does_not_update_lru(self):
        """Test that 'in' check doesn't affect LRU order."""
        cache = LRUCache(capacity=2)

        cache["a"] = 1
        cache["b"] = 2

        # Check 'a' existence (should NOT make it recent)
        _ = "a" in cache

        # Add 'c' - should still evict 'a'
        cache["c"] = 3

        assert "a" not in cache


class TestNullCache:
    """Test NullCache (disabled cache)."""

    def test_null_cache_stores_nothing(self):
        """Test that NullCache doesn't store anything."""
        cache = NullCache()

        cache["key"] = "value"
        cache.set("key2", "value2")

        assert len(cache) == 0
        assert "key" not in cache
        assert cache.get("key") is None

    def test_null_cache_tracks_misses(self):
        """Test that NullCache tracks misses."""
        cache = NullCache()

        _ = cache.get("key1")
        _ = cache.get("key2")

        assert cache.misses == 2
        assert cache.hits == 0
        assert cache.hit_rate == 0.0


class TestPerformance:
    """Test cache performance."""

    def test_large_cache(self):
        """Test cache with many items."""
        cache = LRUCache(capacity=10000)

        # Add 10K items
        for i in range(10000):
            cache[i] = i * 2

        assert len(cache) == 10000

        # Access all items
        for i in range(10000):
            assert cache[i] == i * 2

        # Add more - should evict oldest
        cache[10000] = 20000
        assert 0 not in cache
        assert 10000 in cache

    def test_repr(self):
        """Test string representation."""
        cache = LRUCache(capacity=100)

        cache["key"] = "value"
        _ = cache["key"]
        _ = cache.get("missing")

        repr_str = repr(cache)
        assert "LRUCache" in repr_str
        assert "capacity=100" in repr_str
        assert "size=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
