"""
Tests for Set data structure.
"""

import os
import tempfile
import pytest
from loom.database import DB
from loom.datastructures import Set


class TestSetBasics:
    """Test basic set operations."""

    def test_create_set(self):
        """Test creating a set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")
                assert len(s) == 0
                assert repr(s) == "Set(name='users', size=0)"

    def test_add_and_contains(self):
        """Test adding items and checking membership."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.add("bob")
                s.add("charlie")

                assert "alice" in s
                assert "bob" in s
                assert "charlie" in s
                assert "diana" not in s
                assert len(s) == 3

    def test_add_duplicate(self):
        """Test that adding duplicate has no effect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.add("alice")
                s.add("alice")

                assert len(s) == 1
                assert "alice" in s

    def test_remove(self):
        """Test removing items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.add("bob")
                assert len(s) == 2

                s.remove("alice")
                assert "alice" not in s
                assert "bob" in s
                assert len(s) == 1

    def test_remove_missing_raises(self):
        """Test that removing missing item raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                with pytest.raises(KeyError):
                    s.remove("missing")

    def test_discard(self):
        """Test discard (no error if missing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.discard("alice")
                assert "alice" not in s

                # Should not raise
                s.discard("missing")

    def test_pop(self):
        """Test pop removes and returns an item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.add("bob")

                item = s.pop()
                assert item in ("alice", "bob")
                assert len(s) == 1

    def test_pop_empty_raises(self):
        """Test pop on empty set raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                with pytest.raises(KeyError):
                    s.pop()

    def test_clear(self):
        """Test clearing the set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.add("bob")
                s.add("charlie")
                assert len(s) == 3

                s.clear()
                assert len(s) == 0
                assert "alice" not in s


class TestSetIteration:
    """Test set iteration."""

    def test_iterate(self):
        """Test iterating over set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                items = {"alice", "bob", "charlie"}
                for item in items:
                    s.add(item)

                result = set(s)
                assert result == items

    def test_keys(self):
        """Test keys() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.add("alice")
                s.add("bob")

                keys = list(s.keys())
                assert set(keys) == {"alice", "bob"}


class TestSetBulkOperations:
    """Test bulk operations."""

    def test_update(self):
        """Test adding multiple items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")

                s.update(["alice", "bob", "charlie"])

                assert len(s) == 3
                assert "alice" in s
                assert "bob" in s
                assert "charlie" in s


class TestSetOperations:
    """Test set comparison operations."""

    def test_issubset(self):
        """Test subset checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s1 = db.create_set("s1")
                s2 = db.create_set("s2")

                s1.update(["a", "b"])
                s2.update(["a", "b", "c", "d"])

                assert s1.issubset(s2)
                assert not s2.issubset(s1)

    def test_issuperset(self):
        """Test superset checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s1 = db.create_set("s1")
                s2 = db.create_set("s2")

                s1.update(["a", "b", "c", "d"])
                s2.update(["a", "b"])

                assert s1.issuperset(s2)
                assert not s2.issuperset(s1)

    def test_isdisjoint(self):
        """Test disjoint checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s1 = db.create_set("s1")
                s2 = db.create_set("s2")
                s3 = db.create_set("s3")

                s1.update(["a", "b"])
                s2.update(["c", "d"])
                s3.update(["b", "c"])

                assert s1.isdisjoint(s2)
                assert not s1.isdisjoint(s3)


class TestSetPersistence:
    """Test set persistence."""

    def test_persistence(self):
        """Test that set persists across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Session 1: Create and populate
            with DB(db_path) as db:
                s = db.create_set("users")
                s.add("alice")
                s.add("bob")
                s.add("charlie")

            # Session 2: Verify data persists
            with DB(db_path) as db:
                s = db.create_set("users")
                assert len(s) == 3
                assert "alice" in s
                assert "bob" in s
                assert "charlie" in s

    def test_persistence_after_remove(self):
        """Test that removals persist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Session 1: Create, populate, remove
            with DB(db_path) as db:
                s = db.create_set("users")
                s.add("alice")
                s.add("bob")
                s.remove("alice")

            # Session 2: Verify removal persists
            with DB(db_path) as db:
                s = db.create_set("users")
                assert len(s) == 1
                assert "alice" not in s
                assert "bob" in s


class TestSetBool:
    """Test boolean behavior."""

    def test_bool_empty(self):
        """Test empty set is falsy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")
                assert not s
                assert bool(s) is False

    def test_bool_nonempty(self):
        """Test non-empty set is truthy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")
                s.add("alice")
                assert s
                assert bool(s) is True


class TestSetAtomic:
    """Test atomic operations."""

    def test_atomic_add(self):
        """Test atomic add."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")
                s.add("alice", atomic=True)
                assert "alice" in s

    def test_atomic_update(self):
        """Test atomic bulk update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                s = db.create_set("users")
                s.update(["alice", "bob", "charlie"], atomic=True)
                assert len(s) == 3
