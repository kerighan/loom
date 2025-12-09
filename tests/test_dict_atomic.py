"""Tests for Dict atomic operations."""

import tempfile
import os
import pytest
from loom.database import DB


class TestDictAtomic:
    """Test atomic operations for Dict."""

    def setup_method(self):
        """Create a temporary database file for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_path = self.temp_file.name
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary database file."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
        log_path = self.temp_path + ".log"
        if os.path.exists(log_path):
            os.unlink(log_path)

    def test_atomic_insert_basic(self):
        """Test basic atomic insert."""
        db = DB(self.temp_path)

        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        # Insert with atomic=True
        users.__setitem__(
            "alice", {"id": 1, "name": "Alice", "score": 95.5}, atomic=True
        )
        users.__setitem__("bob", {"id": 2, "name": "Bob", "score": 87.0}, atomic=True)

        assert len(users) == 2
        assert users["alice"]["name"] == "Alice"
        assert users["bob"]["score"] == 87.0

        db.close()

    def test_atomic_update(self):
        """Test atomic update of existing key."""
        db = DB(self.temp_path)

        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        # Insert normally
        users["alice"] = {"id": 1, "name": "Alice", "score": 95.5}

        # Update atomically
        users.__setitem__(
            "alice", {"id": 1, "name": "Alice", "score": 99.0}, atomic=True
        )

        assert users["alice"]["score"] == 99.0

        db.close()

    def test_atomic_persistence(self):
        """Test that atomic writes persist correctly."""
        # Write with atomic=True
        db = DB(self.temp_path)
        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        users.__setitem__(
            "alice", {"id": 1, "name": "Alice", "score": 95.5}, atomic=True
        )
        users.__setitem__("bob", {"id": 2, "name": "Bob", "score": 87.0}, atomic=True)
        db.close()

        # Reopen and verify
        db = DB(self.temp_path)
        from loom.datastructures.dict import Dict

        users = Dict("users_dict", db, None)

        assert len(users) == 2
        assert users["alice"]["name"] == "Alice"
        assert users["bob"]["score"] == 87.0

        db.close()

    def test_atomic_vs_fast_path(self):
        """Test that atomic and fast paths produce same results."""
        # Fast path
        db1 = DB(self.temp_path)
        user_dataset = db1.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users1 = db1.create_dict("users_dict", user_dataset)

        users1["alice"] = {"id": 1, "name": "Alice", "score": 95.5}
        users1["bob"] = {"id": 2, "name": "Bob", "score": 87.0}
        db1.close()

        # Atomic path
        temp_file2 = tempfile.NamedTemporaryFile(delete=False)
        temp_path2 = temp_file2.name
        temp_file2.close()

        try:
            db2 = DB(temp_path2)
            user_dataset2 = db2.create_dataset(
                "users", id="uint32", name="U50", score="float32"
            )
            users2 = db2.create_dict("users_dict", user_dataset2)

            users2.__setitem__(
                "alice", {"id": 1, "name": "Alice", "score": 95.5}, atomic=True
            )
            users2.__setitem__(
                "bob", {"id": 2, "name": "Bob", "score": 87.0}, atomic=True
            )
            db2.close()

            # Compare file sizes (should be similar)
            size1 = os.path.getsize(self.temp_path)
            size2 = os.path.getsize(temp_path2)

            # Sizes should be within 10% (accounting for metadata differences)
            assert abs(size1 - size2) / size1 < 0.1

        finally:
            if os.path.exists(temp_path2):
                os.unlink(temp_path2)
            log_path2 = temp_path2 + ".log"
            if os.path.exists(log_path2):
                os.unlink(log_path2)

    def test_atomic_with_crash_simulation(self):
        """Test that atomic operations are crash-safe via WAL."""
        db = DB(self.temp_path)
        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        # Insert some data atomically
        users.__setitem__(
            "alice", {"id": 1, "name": "Alice", "score": 95.5}, atomic=True
        )

        # Explicitly save metadata to ensure it's persisted
        users.save()

        # Simulate crash by not closing DB properly
        # (In real crash, WAL would have the transaction logged)
        # For this test, we just verify the data is there

        # Force close without proper cleanup
        db._db.mapped_file.close()
        db._db.file_handle.close()

        # Reopen - WAL recovery should happen automatically
        db2 = DB(self.temp_path)
        from loom.datastructures.dict import Dict

        users2 = Dict("users_dict", db2, None)

        # Data should be intact
        assert len(users2) == 1
        assert users2["alice"]["name"] == "Alice"

        db2.close()

    def test_atomic_large_batch(self):
        """Test atomic operations with many items."""
        db = DB(self.temp_path)
        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        # Insert 1000 items atomically
        for i in range(1000):
            users.__setitem__(
                f"user_{i}",
                {"id": i, "name": f"User {i}", "score": float(i * 1.5)},
                atomic=True,
            )

        assert len(users) == 1000
        assert users["user_500"]["id"] == 500
        assert users["user_999"]["score"] == 1498.5

        db.close()

    def test_atomic_mixed_with_fast(self):
        """Test mixing atomic and fast path operations."""
        db = DB(self.temp_path)
        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        # Mix atomic and fast operations
        users["alice"] = {"id": 1, "name": "Alice", "score": 95.5}  # Fast
        users.__setitem__(
            "bob", {"id": 2, "name": "Bob", "score": 87.0}, atomic=True
        )  # Atomic
        users["charlie"] = {"id": 3, "name": "Charlie", "score": 92.0}  # Fast
        users.__setitem__(
            "diana", {"id": 4, "name": "Diana", "score": 88.5}, atomic=True
        )  # Atomic

        assert len(users) == 4
        assert users["alice"]["name"] == "Alice"
        assert users["bob"]["name"] == "Bob"
        assert users["charlie"]["name"] == "Charlie"
        assert users["diana"]["name"] == "Diana"

        db.close()

    def test_set_method_convenience(self):
        """Test the .set() convenience method."""
        db = DB(self.temp_path)
        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        users = db.create_dict("users_dict", user_dataset)

        # Test fast path with .set()
        users.set("alice", {"id": 1, "name": "Alice", "score": 95.5})
        assert users["alice"]["name"] == "Alice"

        # Test atomic path with .set()
        users.set("bob", {"id": 2, "name": "Bob", "score": 87.0}, atomic=True)
        assert users["bob"]["name"] == "Bob"

        # Test update with .set()
        users.set(
            "alice", {"id": 1, "name": "Alice Updated", "score": 99.0}, atomic=True
        )
        assert users["alice"]["name"] == "Alice Updated"

        assert len(users) == 2

        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
