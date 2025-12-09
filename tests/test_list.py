"""
Comprehensive tests for List data structure.

Tests:
- Basic operations (append, get, set, len)
- Slicing
- Iteration
- Persistence
- Edge cases
- Multiple data structures on same file
"""

import os
import tempfile
import pytest
from loom.database import DB
from loom.datastructures import List


class TestListBasics:
    """Test basic list operations."""

    def test_create_list(self):
        """Test creating a list."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})
                assert len(lst) == 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_append(self):
        """Test appending items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                lst.append({"id": 1})
                lst.append({"id": 2})
                lst.append({"id": 3})

                assert len(lst) == 3
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_getitem(self):
        """Test getting items by index."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "name": "U50"})

                lst.append({"id": 1, "name": "Alice"})
                lst.append({"id": 2, "name": "Bob"})
                lst.append({"id": 3, "name": "Charlie"})

                assert lst[0] == {"id": 1, "name": "Alice"}
                assert lst[1] == {"id": 2, "name": "Bob"}
                assert lst[2] == {"id": 3, "name": "Charlie"}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_negative_indexing(self):
        """Test negative indices."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(5):
                    lst.append({"id": i})

                assert lst[-1] == {"id": 4}
                assert lst[-2] == {"id": 3}
                assert lst[-5] == {"id": 0}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_setitem(self):
        """Test setting items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                lst.append({"id": 1, "value": "original"})
                lst[0] = {"id": 1, "value": "modified"}

                assert lst[0] == {"id": 1, "value": "modified"}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_len(self):
        """Test length."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                assert len(lst) == 0

                for i in range(10):
                    lst.append({"id": i})
                    assert len(lst) == i + 1
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestListSlicing:
    """Test list slicing."""

    def test_simple_slice(self):
        """Test simple slicing."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Simple slice
                items = lst[2:5]
                assert len(items) == 3
                assert items[0] == {"id": 2}
                assert items[1] == {"id": 3}
                assert items[2] == {"id": 4}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_slice_array_basic(self):
        """Test slice_array returns correct NumPy array."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "float64"})

                for i in range(100):
                    lst.append({"id": i, "value": float(i * 2)})

                # Get slice as array
                arr = lst.slice_array(10, 20)

                # Verify shape and values
                assert len(arr) == 10
                for i, rec in enumerate(arr):
                    assert rec["id"] == 10 + i
                    assert rec["value"] == float((10 + i) * 2)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_slice_array_matches_dict_slice(self):
        """Test that slice_array returns same data as dict slicing."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "name": "U50"})

                for i in range(50):
                    lst.append({"id": i, "name": f"item_{i}"})

                # Get both slice types
                dict_slice = lst[5:15]
                arr_slice = lst.slice_array(5, 15)

                # Verify they match
                assert len(dict_slice) == len(arr_slice)
                for i, (d, rec) in enumerate(zip(dict_slice, arr_slice)):
                    assert d["id"] == rec["id"]
                    # NumPy strings may have trailing spaces, strip them
                    assert d["name"] == str(rec["name"]).strip()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_slice_array_across_blocks(self):
        """Test slice_array works correctly across block boundaries."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                # Add enough items to span multiple blocks (first block ~57 items)
                for i in range(200):
                    lst.append({"id": i})

                # Slice across block boundary
                arr = lst.slice_array(50, 70)

                assert len(arr) == 20
                for i, rec in enumerate(arr):
                    assert rec["id"] == 50 + i
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_slice_array_full_list(self):
        """Test slice_array on entire list."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                n = 100
                for i in range(n):
                    lst.append({"id": i})

                arr = lst.slice_array(0, n)

                assert len(arr) == n
                for i, rec in enumerate(arr):
                    assert rec["id"] == i
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_slice_with_step(self):
        """Test slicing with step."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Every other item
                items = lst[::2]
                assert len(items) == 5
                assert [item["id"] for item in items] == [0, 2, 4, 6, 8]

                # Reverse step
                items = lst[::-1]
                assert len(items) == 10
                assert items[0] == {"id": 9}
                assert items[-1] == {"id": 0}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_slice_negative_indices(self):
        """Test slicing with negative indices."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Last 3 items
                items = lst[-3:]
                assert len(items) == 3
                assert [item["id"] for item in items] == [7, 8, 9]

                # All but last 2
                items = lst[:-2]
                assert len(items) == 8
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestListIteration:
    """Test list iteration."""

    def test_iterate(self):
        """Test iterating over list."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Iterate and collect
                items = list(lst)
                assert len(items) == 10
                assert [item["id"] for item in items] == list(range(10))
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_enumerate(self):
        """Test enumerate on list."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                lst.append({"id": 1, "value": "a"})
                lst.append({"id": 2, "value": "b"})
                lst.append({"id": 3, "value": "c"})

                for i, item in enumerate(lst):
                    assert item["id"] == i + 1
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestListPersistence:
    """Test list persistence."""

    def test_persist_and_reload(self):
        """Test that list persists across sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Create and populate
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "name": "U50"})

                for i in range(100):
                    lst.append({"id": i, "name": f"item_{i}"})

                assert len(lst) == 100

            # Session 2: Reload and verify
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "name": "U50"})

                assert len(lst) == 100
                assert lst[0] == {"id": 0, "name": "item_0"}
                assert lst[50] == {"id": 50, "name": "item_50"}
                assert lst[-1] == {"id": 99, "name": "item_99"}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_append_after_reload(self):
        """Test appending after reload."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})
                for i in range(50):
                    lst.append({"id": i})

            # Session 2: Append more
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})
                assert len(lst) == 50

                for i in range(50, 100):
                    lst.append({"id": i})

                assert len(lst) == 100

            # Session 3: Verify
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})
                assert len(lst) == 100
                assert lst[0] == {"id": 0}
                assert lst[99] == {"id": 99}
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestListEdgeCases:
    """Test edge cases."""

    def test_empty_list(self):
        """Test operations on empty list."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                assert len(lst) == 0
                assert list(lst) == []

                with pytest.raises(IndexError):
                    _ = lst[0]
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_single_item(self):
        """Test list with single item."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                lst.append({"id": 42})

                assert len(lst) == 1
                assert lst[0] == {"id": 42}
                assert lst[-1] == {"id": 42}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_large_list(self):
        """Test list with many items (multiple blocks)."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                # Add enough items to span multiple blocks
                # First block: ~57 items, second: ~86 items
                n = 200
                for i in range(n):
                    lst.append({"id": i})

                assert len(lst) == n
                assert lst[0] == {"id": 0}
                assert lst[56] == {"id": 56}  # End of first block
                assert lst[57] == {"id": 57}  # Start of second block
                assert lst[-1] == {"id": n - 1}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_index_out_of_range(self):
        """Test index out of range errors."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(5):
                    lst.append({"id": i})

                with pytest.raises(IndexError):
                    _ = lst[5]

                with pytest.raises(IndexError):
                    _ = lst[-6]
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestMultipleDataStructures:
    """Test multiple data structures on same file."""

    def test_two_lists(self):
        """Test two lists on same file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                list1 = db.create_list("list1", {"id": "uint64", "value": "U50"})
                list2 = db.create_list("list2", {"x": "float64", "y": "float64"})

                # Add to list1
                for i in range(10):
                    list1.append({"id": i, "value": f"item_{i}"})

                # Add to list2
                for i in range(5):
                    list2.append({"x": float(i), "y": float(i * 2)})

                assert len(list1) == 10
                assert len(list2) == 5

                # Verify data
                assert list1[0] == {"id": 0, "value": "item_0"}
                assert list2[0] == {"x": 0.0, "y": 0.0}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_alternating_inserts(self):
        """Test alternating inserts between structures."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                list1 = db.create_list("list1", {"id": "uint64"})
                list2 = db.create_list("list2", {"id": "uint64"})
                bf = db.create_bloomfilter("bloom", expected_items=100)

                # Alternate between structures
                for i in range(50):
                    list1.append({"id": i})
                    list2.append({"id": i * 2})
                    bf.add(f"item_{i}")

                # Verify all structures
                assert len(list1) == 50
                assert len(list2) == 50
                assert len(bf) == 50

                assert list1[0] == {"id": 0}
                assert list2[0] == {"id": 0}
                assert "item_0" in bf

                assert list1[49] == {"id": 49}
                assert list2[49] == {"id": 98}
                assert "item_49" in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_persistence_multiple_structures(self):
        """Test persistence with multiple structures."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Create and populate
            with DB(filename) as db:
                list1 = db.create_list("tasks", {"id": "uint64", "task": "U200"})
                list2 = db.create_list("users", {"id": "uint64", "name": "U50"})
                bf = db.create_bloomfilter("seen", expected_items=1000)

                for i in range(100):
                    list1.append({"id": i, "task": f"task_{i}"})
                    list2.append({"id": i, "name": f"user_{i}"})
                    bf.add(f"item_{i}")

            # Session 2: Reload and verify
            with DB(filename) as db:
                list1 = db.create_list("tasks", {"id": "uint64", "task": "U200"})
                list2 = db.create_list("users", {"id": "uint64", "name": "U50"})
                bf = db.create_bloomfilter("seen", expected_items=1000)

                assert len(list1) == 100
                assert len(list2) == 100
                assert len(bf) == 100

                assert list1[50] == {"id": 50, "task": "task_50"}
                assert list2[50] == {"id": 50, "name": "user_50"}
                assert "item_50" in bf
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
