"""
Tests for CountingBloomFilter data structure.

Tests:
- Basic add/remove/check operations
- Removal functionality (key feature!)
- Persistence across sessions
- Counter overflow handling
- Edge cases
"""

import os
import tempfile
import pytest
from loom.database import DB
from loom.datastructures import CountingBloomFilter, CounterOverflowError


class TestCountingBloomFilterBasics:
    """Test basic operations."""

    def test_create(self):
        """Test creating a Counting Bloom filter."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=1000)

                assert cbf.name == "test"
                assert cbf.num_items == 0
                assert cbf.max_count == 255
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_add_and_check(self):
        """Test adding and checking items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                cbf.add(1)
                cbf.add(2)
                cbf.add(3)

                assert 1 in cbf
                assert 2 in cbf
                assert 3 in cbf
                assert 999 not in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestRemoval:
    """Test removal functionality - the key feature!"""

    def test_remove_item(self):
        """Test removing an item."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                # Add items
                cbf.add("alice")
                cbf.add("bob")
                cbf.add("charlie")

                assert "alice" in cbf
                assert "bob" in cbf
                assert "charlie" in cbf

                # Remove one
                cbf.remove("bob")

                # Bob should be gone
                assert "bob" not in cbf

                # Others still there
                assert "alice" in cbf
                assert "charlie" in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_add_remove_add(self):
        """Test adding, removing, then adding again."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                # Add
                cbf.add("test")
                assert "test" in cbf

                # Remove
                cbf.remove("test")
                assert "test" not in cbf

                # Add again
                cbf.add("test")
                assert "test" in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_multiple_same_item(self):
        """Test adding same item multiple times."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                # Add same item 3 times
                cbf.add("test")
                cbf.add("test")
                cbf.add("test")

                assert "test" in cbf

                # Remove once - still there
                cbf.remove("test")
                assert "test" in cbf

                # Remove again - still there
                cbf.remove("test")
                assert "test" in cbf

                # Remove third time - now gone
                cbf.remove("test")
                assert "test" not in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_len_with_removal(self):
        """Test length tracking with add/remove."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                assert len(cbf) == 0

                cbf.add(1)
                assert len(cbf) == 1

                cbf.add(2)
                assert len(cbf) == 2

                cbf.remove(1)
                assert len(cbf) == 1

                cbf.remove(2)
                assert len(cbf) == 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestPersistence:
    """Test persistence across sessions."""

    def test_persistence(self):
        """Test that filter persists across sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Add items
            with DB(filename) as db:
                cbf = CountingBloomFilter("cache", db, expected_items=1000)
                cbf.add("alice")
                cbf.add("bob")
                cbf.add("charlie")

            # Session 2: Check and remove
            with DB(filename) as db:
                cbf = CountingBloomFilter("cache", db)

                assert "alice" in cbf
                assert "bob" in cbf
                assert "charlie" in cbf

                cbf.remove("bob")

            # Session 3: Verify removal persisted
            with DB(filename) as db:
                cbf = CountingBloomFilter("cache", db)

                assert "alice" in cbf
                assert "bob" not in cbf
                assert "charlie" in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestOperations:
    """Test various operations."""

    def test_del_operator(self):
        """Test del operator for removal."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                # Add items
                cbf.add("alice")
                cbf.add("bob")

                assert "alice" in cbf
                assert "bob" in cbf

                # Remove using del operator
                del cbf["alice"]

                # Alice should be gone
                assert "alice" not in cbf
                # Bob still there
                assert "bob" in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_clear(self):
        """Test clearing the filter."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                cbf.add(1)
                cbf.add(2)
                cbf.add(3)

                assert len(cbf) == 3
                assert 1 in cbf

                cbf.clear()

                assert len(cbf) == 0
                assert 1 not in cbf
                assert 2 not in cbf
                assert 3 not in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_repr(self):
        """Test string representation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=1000)

                repr_str = repr(cbf)
                assert "test" in repr_str
                assert "CountingBloomFilter" in repr_str
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestEdgeCases:
    """Test edge cases."""

    def test_remove_nonexistent(self):
        """Test removing item that was never added."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100)

                cbf.add("alice")

                # Remove item that wasn't added
                # This decrements counters incorrectly but shouldn't crash
                cbf.remove("bob")

                # Alice should still be there
                assert "alice" in cbf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_custom_max_count(self):
        """Test with custom max_count."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                cbf = CountingBloomFilter("test", db, expected_items=100, max_count=10)

                assert cbf.max_count == 10
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_counter_overflow(self):
        """Test that counter overflow raises error."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                # Very small filter to force collisions
                cbf = CountingBloomFilter("test", db, expected_items=10, max_count=5)

                # Add same item multiple times up to max
                for i in range(5):
                    cbf.add("test")

                # Next add should raise overflow error
                with pytest.raises(CounterOverflowError, match="counter at maximum"):
                    cbf.add("test")
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestComparison:
    """Compare with regular Bloom filter behavior."""

    def test_same_false_positive_rate(self):
        """Test that FP rate is similar to regular Bloom filter."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                target_fp_rate = 0.01
                cbf = CountingBloomFilter(
                    "test", db, expected_items=1000, false_positive_rate=target_fp_rate
                )

                # Add items
                added = set(range(1000))
                for item in added:
                    cbf.add(item)

                # Test items not added
                false_positives = 0
                test_items = 1000
                for item in range(10000, 10000 + test_items):
                    if item in cbf:
                        false_positives += 1

                actual_fp_rate = false_positives / test_items

                # Should be close to target (within 2x)
                assert actual_fp_rate < target_fp_rate * 2
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
