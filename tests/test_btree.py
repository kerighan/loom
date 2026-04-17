"""
Tests for BTree data structure.

BTree provides ordered key-value storage with O(log n) operations
and efficient range queries.
"""

import os
import tempfile
import pytest
from loom.database import DB
from loom.datastructures import BTree


class TestBTreeBasics:
    """Test basic BTree operations."""

    def test_create_btree(self):
        """Test creating a BTree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)
                assert len(btree) == 0

    def test_insert_and_get(self):
        """Test inserting and retrieving items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}
                btree["charlie"] = {"id": 3, "name": "Charlie"}

                assert len(btree) == 3
                assert btree["alice"]["name"] == "Alice"
                assert btree["bob"]["id"] == 2
                assert btree["charlie"]["name"] == "Charlie"

    def test_update_existing(self):
        """Test updating an existing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["alice"] = {"id": 1, "name": "Alice Updated"}

                assert len(btree) == 1
                assert btree["alice"]["name"] == "Alice Updated"

    def test_contains(self):
        """Test membership checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}

                assert "alice" in btree
                assert "bob" not in btree

    def test_delete(self):
        """Test deleting items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}
                assert len(btree) == 2

                del btree["alice"]
                assert len(btree) == 1
                assert "alice" not in btree
                assert "bob" in btree

    def test_delete_missing_raises(self):
        """Test that deleting missing key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                with pytest.raises(KeyError):
                    del btree["missing"]

    def test_get_missing_raises(self):
        """Test that getting missing key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                with pytest.raises(KeyError):
                    _ = btree["missing"]

    def test_get_with_default(self):
        """Test get() with default value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}

                assert btree.get("alice")["name"] == "Alice"
                assert btree.get("missing") is None
                assert btree.get("missing", "default") == "default"


class TestBTreeOrdering:
    """Test ordered operations."""

    def test_keys_sorted(self):
        """Test that keys() returns keys in sorted order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                # Insert in random order
                btree["charlie"] = {"id": 3, "name": "Charlie"}
                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}
                btree["diana"] = {"id": 4, "name": "Diana"}

                keys = list(btree.keys())
                assert keys == ["alice", "bob", "charlie", "diana"]

    def test_items_sorted(self):
        """Test that items() returns items in sorted order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["charlie"] = {"id": 3, "name": "Charlie"}
                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}

                items = list(btree.items())
                assert items[0][0] == "alice"
                assert items[1][0] == "bob"
                assert items[2][0] == "charlie"

    def test_min_max(self):
        """Test min() and max() operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["charlie"] = {"id": 3, "name": "Charlie"}
                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}

                assert btree.min() == "alice"
                assert btree.max() == "charlie"

    def test_min_max_empty(self):
        """Test min/max on empty tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                assert btree.min() is None
                assert btree.max() is None


class TestBTreeRangeQueries:
    """Test range query operations."""

    def test_range_inclusive(self):
        """Test range query with inclusive bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                for c in "abcdefghij":
                    btree[c] = {"id": ord(c), "name": c.upper()}

                # Range from 'c' to 'g' inclusive
                result = list(btree.range("c", "g"))
                keys = [k for k, v in result]
                assert keys == ["c", "d", "e", "f", "g"]

    def test_range_exclusive(self):
        """Test range query with exclusive bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                for c in "abcdefghij":
                    btree[c] = {"id": ord(c), "name": c.upper()}

                # Range from 'c' to 'g' exclusive
                result = list(btree.range("c", "g", inclusive=(False, False)))
                keys = [k for k, v in result]
                assert keys == ["d", "e", "f"]

    def test_range_open_start(self):
        """Test range with no lower bound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                for c in "abcdefghij":
                    btree[c] = {"id": ord(c), "name": c.upper()}

                # All keys up to 'c'
                result = list(btree.range(None, "c"))
                keys = [k for k, v in result]
                assert keys == ["a", "b", "c"]

    def test_range_open_end(self):
        """Test range with no upper bound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                for c in "abcdefghij":
                    btree[c] = {"id": ord(c), "name": c.upper()}

                # All keys from 'h' onwards
                result = list(btree.range("h", None))
                keys = [k for k, v in result]
                assert keys == ["h", "i", "j"]

    def test_prefix_search(self):
        """Test prefix-based search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["user_alice"] = {"id": 1, "name": "Alice"}
                btree["user_bob"] = {"id": 2, "name": "Bob"}
                btree["admin_charlie"] = {"id": 3, "name": "Charlie"}
                btree["user_diana"] = {"id": 4, "name": "Diana"}

                result = list(btree.prefix("user_"))
                keys = [k for k, v in result]
                assert keys == ["user_alice", "user_bob", "user_diana"]


class TestBTreePersistence:
    """Test persistence across sessions."""

    def test_persistence(self):
        """Test that BTree persists across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Session 1: Create and populate
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}
                btree["charlie"] = {"id": 3, "name": "Charlie"}

            # Session 2: Verify data persists
            with DB(db_path) as db:
                user_ds = db.get_dataset("users")
                btree = db.create_btree("users_btree", user_ds)

                assert len(btree) == 3
                assert btree["alice"]["name"] == "Alice"
                assert btree["bob"]["id"] == 2

                # Verify ordering persists
                keys = list(btree.keys())
                assert keys == ["alice", "bob", "charlie"]

    def test_persistence_after_delete(self):
        """Test that deletions persist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Session 1: Create, populate, delete
            with DB(db_path) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                btree["alice"] = {"id": 1, "name": "Alice"}
                btree["bob"] = {"id": 2, "name": "Bob"}
                del btree["alice"]

            # Session 2: Verify deletion persists
            with DB(db_path) as db:
                user_ds = db.get_dataset("users")
                btree = db.create_btree("users_btree", user_ds)

                assert len(btree) == 1
                assert "alice" not in btree
                assert "bob" in btree


class TestBTreeScaling:
    """Test BTree with larger datasets."""

    def test_many_inserts(self):
        """Test inserting many items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, header_size=1024 * 1024) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                # Insert 1000 items
                for i in range(1000):
                    key = f"user_{i:04d}"
                    btree[key] = {"id": i, "name": f"User {i}"}

                assert len(btree) == 1000

                # Verify ordering
                keys = list(btree.keys())
                assert keys == sorted(keys)

                # Verify random access
                assert btree["user_0500"]["id"] == 500

    def test_many_deletes(self):
        """Test deleting many items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, header_size=1024 * 1024) as db:
                user_ds = db.create_dataset("users", id="uint32", name="U50")
                btree = db.create_btree("users_btree", user_ds)

                # Insert 500 items
                for i in range(500):
                    key = f"user_{i:04d}"
                    btree[key] = {"id": i, "name": f"User {i}"}

                # Delete half
                for i in range(0, 500, 2):
                    key = f"user_{i:04d}"
                    del btree[key]

                assert len(btree) == 250

                # Verify remaining items
                for i in range(1, 500, 2):
                    key = f"user_{i:04d}"
                    assert key in btree


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
