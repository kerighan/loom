"""
Comprehensive tests for List deletion.

Tests:
- Basic deletion
- Multiple deletions
- Deletion with existing deletions
- Set after deletion
- Auto-compaction
- Persistence after deletion
"""

import os
import tempfile
import pytest
from loom.database import DB


class TestListDeletion:
    """Test list deletion operations."""

    def test_delete_single_item(self):
        """Test deleting a single item."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                # Add items
                for i in range(5):
                    lst.append({"id": i, "value": f"item_{i}"})

                assert len(lst) == 5

                # Delete middle item
                del lst[2]

                assert len(lst) == 4
                assert lst[0] == {"id": 0, "value": "item_0"}
                assert lst[1] == {"id": 1, "value": "item_1"}
                assert lst[2] == {"id": 3, "value": "item_3"}  # Shifted!
                assert lst[3] == {"id": 4, "value": "item_4"}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delete_first_item(self):
        """Test deleting first item."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(5):
                    lst.append({"id": i})

                del lst[0]

                assert len(lst) == 4
                assert lst[0] == {"id": 1}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delete_last_item(self):
        """Test deleting last item."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(5):
                    lst.append({"id": i})

                del lst[-1]

                assert len(lst) == 4
                assert lst[-1] == {"id": 3}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_multiple_deletions(self):
        """Test multiple deletions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Delete multiple items
                # After each deletion, indices shift!
                del lst[5]  # Delete item with id=5, list now: [0,1,2,3,4,6,7,8,9]
                assert len(lst) == 9

                del lst[3]  # Delete item at index 3 (id=3), list now: [0,1,2,4,6,7,8,9]
                assert len(lst) == 8

                del lst[0]  # Delete item at index 0 (id=0), list now: [1,2,4,6,7,8,9]
                assert len(lst) == 7

                # Verify remaining items
                expected = [1, 2, 4, 6, 7, 8, 9]
                for i, item in enumerate(lst):
                    assert item["id"] == expected[i]
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delete_with_negative_index(self):
        """Test deletion with negative indices."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Delete using negative indices
                del lst[-1]  # Delete 9
                del lst[-2]  # Delete 7 (was at -2)

                assert len(lst) == 8
                assert lst[-1] == {"id": 8}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_setitem_after_deletion(self):
        """Test setting items after deletion."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                for i in range(10):
                    lst.append({"id": i, "value": f"item_{i}"})

                # Delete some items
                del lst[2]
                del lst[4]  # Was at index 5

                assert len(lst) == 8

                # Set item after deletion
                lst[2] = {"id": 99, "value": "modified"}

                assert lst[2] == {"id": 99, "value": "modified"}
                assert len(lst) == 8
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_iteration_after_deletion(self):
        """Test iteration skips deleted items."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(10):
                    lst.append({"id": i})

                # Delete some items
                del lst[2]
                del lst[4]  # Was at 5
                del lst[6]  # Was at 8

                # Iterate and collect
                items = list(lst)
                assert len(items) == 7

                expected = [0, 1, 3, 4, 6, 7, 9]
                for i, item in enumerate(items):
                    assert item["id"] == expected[i]
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delete_out_of_range(self):
        """Test deleting out of range raises error."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(5):
                    lst.append({"id": i})

                with pytest.raises(IndexError):
                    del lst[5]

                with pytest.raises(IndexError):
                    del lst[-6]
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestAutoCompaction:
    """Test automatic compaction."""

    def test_auto_compact_at_threshold(self):
        """Test auto-compaction at 30% waste."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                # Add 100 items
                for i in range(100):
                    lst.append({"id": i})

                # Delete 30 items (30% waste)
                for i in range(30):
                    del lst[0]  # Always delete first

                # Should have auto-compacted
                assert len(lst) == 70

                # Verify data integrity after compaction
                for i, item in enumerate(lst):
                    assert item["id"] == i + 30
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_manual_compact(self):
        """Test manual compaction."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                # Add items
                for i in range(100):
                    lst.append({"id": i, "value": f"item_{i}"})

                # Delete some items (not enough to trigger auto-compact)
                for i in range(20):
                    del lst[0]

                assert len(lst) == 80

                # Manual compact
                lst.compact()

                # Verify all data preserved
                assert len(lst) == 80
                for i, item in enumerate(lst):
                    assert item["id"] == i + 20
                    assert item["value"] == f"item_{i + 20}"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_compact_resets_performance(self):
        """Test that compaction resets to O(1) performance."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                # Add items
                for i in range(100):
                    lst.append({"id": i})

                # Delete many items
                for i in range(50):
                    del lst[0]

                # Compact
                lst.compact()

                # After compaction, length should equal valid_count
                assert lst.length == lst.valid_count == 50

                # Verify sequential access works
                for i in range(50):
                    assert lst[i]["id"] == i + 50
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDeletionPersistence:
    """Test deletion persistence."""

    def test_deletions_persist(self):
        """Test that deletions persist across sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Create and delete
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(20):
                    lst.append({"id": i})

                # Delete some items
                del lst[5]
                del lst[10]  # Was at 11
                del lst[15]  # Was at 17

                assert len(lst) == 17

            # Session 2: Verify deletions persisted
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                assert len(lst) == 17

                # Verify correct items remain
                items = [item["id"] for item in lst]
                assert 5 not in items
                assert 11 not in items
                assert 17 not in items
                assert 0 in items
                assert 10 in items
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_compact_persists(self):
        """Test that compaction persists."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Delete and compact
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                for i in range(100):
                    lst.append({"id": i})

                # Delete many items
                for i in range(40):
                    del lst[0]

                # Manual compact
                lst.compact()

                assert len(lst) == 60

            # Session 2: Verify compaction persisted
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                assert len(lst) == 60
                assert lst.length == lst.valid_count  # No waste

                # Verify data
                for i, item in enumerate(lst):
                    assert item["id"] == i + 40
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
