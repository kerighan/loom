"""
Tests for BloomFilter data structure.

Tests:
- Basic add and membership testing
- False positive rate
- Persistence across sessions
- Different data types
- Edge cases
"""

import os
import tempfile
import pytest
from loom.database import DB
from loom.datastructures import BloomFilter


class TestBloomFilterBasics:
    """Test basic Bloom filter operations."""

    def test_create_bloomfilter(self):
        """Test creating a Bloom filter."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter(
                    "test", db, expected_items=1000, false_positive_rate=0.01
                )

                assert bf.name == "test"
                assert bf.num_items == 0
                assert bf.num_bits > 0
                assert bf.num_hashes > 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_add_and_check(self):
        """Test adding items and checking membership."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)

                # Add items
                bf.add(1)
                bf.add(2)
                bf.add(3)

                # Check membership
                assert 1 in bf
                assert 2 in bf
                assert 3 in bf

                # Items not added should (probably) not be in filter
                # Note: Small chance of false positive
                assert 999 not in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_string_items(self):
        """Test with string items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)

                bf.add("alice")
                bf.add("bob")
                bf.add("charlie")

                assert "alice" in bf
                assert "bob" in bf
                assert "charlie" in bf
                assert "diana" not in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_mixed_types(self):
        """Test with different data types."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)

                bf.add(123)
                bf.add("test")
                bf.add(45.67)

                assert 123 in bf
                assert "test" in bf
                assert 45.67 in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestBloomFilterPersistence:
    """Test persistence across sessions."""

    def test_persistence(self):
        """Test that Bloom filter persists across sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Create and add items
            with DB(filename) as db:
                bf = BloomFilter("users", db, expected_items=1000)
                bf.add("alice")
                bf.add("bob")
                bf.add("charlie")

                assert len(bf) == 3

            # Session 2: Load and check
            with DB(filename) as db:
                bf = BloomFilter("users", db)

                assert len(bf) == 3
                assert "alice" in bf
                assert "bob" in bf
                assert "charlie" in bf
                assert "diana" not in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_add_after_reload(self):
        """Test adding items after reloading."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)
                bf.add(1)
                bf.add(2)

            # Session 2
            with DB(filename) as db:
                bf = BloomFilter("test", db)
                bf.add(3)
                bf.add(4)

                assert 1 in bf
                assert 2 in bf
                assert 3 in bf
                assert 4 in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestBloomFilterAccuracy:
    """Test false positive rate."""

    def test_no_false_negatives(self):
        """Test that there are never false negatives."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=1000)

                # Add many items
                items = list(range(500))
                for item in items:
                    bf.add(item)

                # All added items must be found
                for item in items:
                    assert item in bf, f"False negative for {item}"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_false_positive_rate(self):
        """Test that false positive rate is reasonable."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                target_fp_rate = 0.01
                bf = BloomFilter(
                    "test", db, expected_items=1000, false_positive_rate=target_fp_rate
                )

                # Add items
                added = set(range(1000))
                for item in added:
                    bf.add(item)

                # Test items not added
                false_positives = 0
                test_items = 1000
                for item in range(10000, 10000 + test_items):
                    if item in bf:
                        false_positives += 1

                actual_fp_rate = false_positives / test_items

                # Should be close to target (within 2x)
                assert (
                    actual_fp_rate < target_fp_rate * 2
                ), f"FP rate {actual_fp_rate} exceeds {target_fp_rate * 2}"
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestBloomFilterOperations:
    """Test various operations."""

    def test_len(self):
        """Test length tracking."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)

                assert len(bf) == 0

                bf.add(1)
                assert len(bf) == 1

                bf.add(2)
                assert len(bf) == 2

                # Adding same item increases count (not a set)
                bf.add(1)
                assert len(bf) == 3
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_clear(self):
        """Test clearing the filter."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)

                bf.add(1)
                bf.add(2)
                bf.add(3)

                assert len(bf) == 3
                assert 1 in bf

                bf.clear()

                assert len(bf) == 0
                assert 1 not in bf
                assert 2 not in bf
                assert 3 not in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_repr(self):
        """Test string representation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=1000)

                repr_str = repr(bf)
                assert "test" in repr_str
                assert "BloomFilter" in repr_str
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestBloomFilterEdgeCases:
    """Test edge cases."""

    def test_empty_filter(self):
        """Test operations on empty filter."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100)

                assert len(bf) == 0
                assert 1 not in bf
                assert "test" not in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_small_filter(self):
        """Test with very small expected items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=10)

                for i in range(5):
                    bf.add(i)

                for i in range(5):
                    assert i in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_large_filter(self):
        """Test with large expected items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                bf = BloomFilter("test", db, expected_items=100000)

                # Add some items
                for i in range(100):
                    bf.add(i)

                # Check they're all there
                for i in range(100):
                    assert i in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
