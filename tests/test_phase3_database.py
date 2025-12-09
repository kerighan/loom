"""
Phase 3 Tests: Database Orchestrator

Tests for:
- Dataset creation and retrieval
- Schema persistence
- Registry management
- Context manager
- Dict-like access
"""

import os
import tempfile
import pytest
from loom.database import DB


class TestDatabaseBasics:
    """Test basic database operations."""

    def test_create_and_open(self):
        """Test creating and opening a database."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = DB(filename)
            db.open()

            assert db._is_open
            assert len(db.list_datasets()) == 0

            db.close()
            assert not db._is_open
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_context_manager(self):
        """Test context manager support."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                assert db._is_open
                users = db.create_dataset("users", id="uint64", name="U50")
                assert "users" in db

            # Should be closed after context
            assert not db._is_open
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDatasetCreation:
    """Test dataset creation and management."""

    def test_create_dataset(self):
        """Test creating a dataset."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                users = db.create_dataset("users", id="uint64", name="U50", age="int32")

                assert users.name == "users"
                assert users.identifier == 1  # First dataset gets ID 1
                assert "id" in users.user_schema.names
                assert "name" in users.user_schema.names
                assert "age" in users.user_schema.names
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_create_multiple_datasets(self):
        """Test creating multiple datasets."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                users = db.create_dataset("users", id="uint64", name="U50")
                posts = db.create_dataset("posts", id="uint64", title="U100")
                comments = db.create_dataset("comments", id="uint64", text="U200")

                assert users.identifier == 1
                assert posts.identifier == 2
                assert comments.identifier == 3

                assert len(db.list_datasets()) == 3
                assert "users" in db.list_datasets()
                assert "posts" in db.list_datasets()
                assert "comments" in db.list_datasets()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_duplicate_dataset_name(self):
        """Test that duplicate names are rejected."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")

                with pytest.raises(ValueError, match="already exists"):
                    db.create_dataset("users", id="uint64", email="U100")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_auto_increment_identifiers(self):
        """Test that identifiers auto-increment."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                ds1 = db.create_dataset("ds1", value="int32")
                ds2 = db.create_dataset("ds2", value="int32")
                ds3 = db.create_dataset("ds3", value="int32")

                assert ds1.identifier == 1
                assert ds2.identifier == 2
                assert ds3.identifier == 3
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestSchemaPersistence:
    """Test schema persistence across sessions."""

    def test_schema_persists(self):
        """Test that schema is saved and loaded."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create dataset
            with DB(filename) as db:
                users = db.create_dataset("users", id="uint64", name="U50", age="int32")
                assert users.identifier == 1

            # Reopen and verify
            with DB(filename) as db:
                assert "users" in db
                users = db["users"]

                assert users.name == "users"
                assert users.identifier == 1
                assert "id" in users.user_schema.names
                assert "name" in users.user_schema.names
                assert "age" in users.user_schema.names
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_multiple_datasets_persist(self):
        """Test that multiple datasets persist."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create datasets
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")
                db.create_dataset("posts", id="uint64", title="U100")
                db.create_dataset("comments", id="uint64", text="U200")

            # Reopen and verify
            with DB(filename) as db:
                assert len(db.list_datasets()) == 3
                assert "users" in db
                assert "posts" in db
                assert "comments" in db

                users = db["users"]
                posts = db["posts"]
                comments = db["comments"]

                assert users.identifier == 1
                assert posts.identifier == 2
                assert comments.identifier == 3
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_data_persists_with_schema(self):
        """Test that data and schema both persist."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create and write data
            with DB(filename) as db:
                users = db.create_dataset("users", id="uint64", name="U50", age="int32")
                addr = users.allocate_block(1)
                users[addr] = {"id": 1, "name": "Alice", "age": 30}

            # Reopen and read data
            with DB(filename) as db:
                users = db["users"]
                record = users[addr]

                assert record["id"] == 1
                assert record["name"] == "Alice"
                assert record["age"] == 30
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDatasetRetrieval:
    """Test dataset retrieval methods."""

    def test_get_dataset(self):
        """Test get_dataset method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")

                users = db.get_dataset("users")
                assert users.name == "users"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_get_nonexistent_dataset(self):
        """Test getting a dataset that doesn't exist."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                with pytest.raises(KeyError, match="not found"):
                    db.get_dataset("nonexistent")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_has_dataset(self):
        """Test has_dataset method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                assert not db.has_dataset("users")

                db.create_dataset("users", id="uint64", name="U50")

                assert db.has_dataset("users")
                assert not db.has_dataset("posts")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_list_datasets(self):
        """Test list_datasets method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                assert db.list_datasets() == []

                db.create_dataset("users", id="uint64", name="U50")
                db.create_dataset("posts", id="uint64", title="U100")

                datasets = db.list_datasets()
                assert len(datasets) == 2
                assert "users" in datasets
                assert "posts" in datasets
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDictLikeAccess:
    """Test dict-like access to datasets."""

    def test_getitem(self):
        """Test accessing dataset with []."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")

                users = db["users"]
                assert users.name == "users"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_contains(self):
        """Test 'in' operator."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                assert "users" not in db

                db.create_dataset("users", id="uint64", name="U50")

                assert "users" in db
                assert "posts" not in db
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_datasets_property(self):
        """Test datasets property."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")
                db.create_dataset("posts", id="uint64", title="U100")

                datasets = db.datasets
                assert len(datasets) == 2
                assert "users" in datasets
                assert "posts" in datasets
                assert datasets["users"].name == "users"
                assert datasets["posts"].name == "posts"
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDatasetDeletion:
    """Test dataset deletion."""

    def test_delete_dataset(self):
        """Test deleting a dataset."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")
                db.create_dataset("posts", id="uint64", title="U100")

                assert "users" in db
                assert "posts" in db

                db.delete_dataset("users")

                assert "users" not in db
                assert "posts" in db
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delete_nonexistent_dataset(self):
        """Test deleting a dataset that doesn't exist."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                with pytest.raises(KeyError, match="not found"):
                    db.delete_dataset("nonexistent")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_deletion_persists(self):
        """Test that deletion persists across sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create and delete
            with DB(filename) as db:
                db.create_dataset("users", id="uint64", name="U50")
                db.create_dataset("posts", id="uint64", title="U100")
                db.delete_dataset("users")

            # Reopen and verify
            with DB(filename) as db:
                assert "users" not in db
                assert "posts" in db
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delete_dataset_with_data(self):
        """Test deleting a dataset that contains data."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create dataset and add data
            with DB(filename) as db:
                users = db.create_dataset("users", id="uint64", name="U50")
                addr = users.allocate_block(1)
                users[addr] = {"id": 1, "name": "Alice"}

                # Verify data exists
                assert users[addr]["name"] == "Alice"

                # Delete dataset
                db.delete_dataset("users")

                # Dataset should be gone from registry
                assert "users" not in db

            # Reopen - dataset should still be gone
            with DB(filename) as db:
                assert "users" not in db
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_recreate_deleted_dataset(self):
        """Test creating a new dataset with the same name after deletion."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                # First dataset
                users1 = db.create_dataset("users", id="uint64", name="U50")
                assert users1.identifier == 1

                # Delete it
                db.delete_dataset("users")

                # Create new dataset with same name
                users2 = db.create_dataset("users", id="uint64", email="U100")
                # Should get a NEW identifier
                assert users2.identifier == 2
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_identifier_not_reused(self):
        """Test that identifiers are not reused after deletion."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            with DB(filename) as db:
                ds1 = db.create_dataset("ds1", value="int32")
                ds2 = db.create_dataset("ds2", value="int32")
                ds3 = db.create_dataset("ds3", value="int32")

                assert ds1.identifier == 1
                assert ds2.identifier == 2
                assert ds3.identifier == 3

                # Delete middle dataset
                db.delete_dataset("ds2")

                # Create new dataset - should get ID 4, not reuse 2
                ds4 = db.create_dataset("ds4", value="int32")
                assert ds4.identifier == 4
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestIntegration:
    """Test complete workflows."""

    def test_complete_workflow(self):
        """Test a complete realistic workflow."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Day 1: Create database and add data
            with DB(filename) as db:
                users = db.create_dataset("users", id="uint64", name="U50", age="int32")
                posts = db.create_dataset(
                    "posts", id="uint64", title="U100", author_id="uint64"
                )

                # Add users
                user_addr = users.allocate_block(2)
                users[user_addr] = {"id": 1, "name": "Alice", "age": 30}
                users[user_addr + users.record_size] = {
                    "id": 2,
                    "name": "Bob",
                    "age": 25,
                }

                # Add posts
                post_addr = posts.allocate_block(2)
                posts[post_addr] = {"id": 100, "title": "Hello World", "author_id": 1}
                posts[post_addr + posts.record_size] = {
                    "id": 101,
                    "title": "Python Tips",
                    "author_id": 2,
                }

            # Day 2: Different file, load and use
            with DB(filename) as db:
                # List what's available
                assert len(db.list_datasets()) == 2

                # Get datasets (schema loaded automatically!)
                users = db["users"]
                posts = db["posts"]

                # Read data
                alice = users[user_addr]
                assert alice["name"] == "Alice"
                assert alice["age"] == 30

                bob = users[user_addr + users.record_size]
                assert bob["name"] == "Bob"

                post1 = posts[post_addr]
                assert post1["title"] == "Hello World"
                assert post1["author_id"] == 1

                # Add more data
                user_addr2 = users.allocate_block(1)
                users[user_addr2] = {"id": 3, "name": "Charlie", "age": 35}

            # Day 3: Verify everything persisted
            with DB(filename) as db:
                users = db["users"]
                charlie = users[user_addr2]
                assert charlie["name"] == "Charlie"
                assert charlie["age"] == 35
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = DB(filename, auto_open=False)
            repr_str = repr(db)
            assert filename in repr_str
            assert "closed" in repr_str

            db.open()
            db.create_dataset("users", id="uint64", name="U50")
            repr_str = repr(db)
            assert "open" in repr_str
            assert "1 datasets" in repr_str

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
