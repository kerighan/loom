"""Test nested Set data structures: List[Set] and Dict[Set].

These tests verify that we can create and use:
- List[Set] - A list where each element is a Set
- Dict[Set] - A dict where each value is a Set
"""

import os
import tempfile
import pytest
from loom import DB
from loom.datastructures import List, Set
from loom.datastructures.dict import Dict


class TestListOfSets:
    """Test List of Sets (List[Set])."""

    def setup_method(self):
        """Create temp file for each test."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix=".loom")
        os.close(self.temp_fd)

    def teardown_method(self):
        """Clean up temp file."""
        try:
            os.unlink(self.temp_path)
        except (PermissionError, FileNotFoundError):
            pass

    def test_list_of_sets_creation(self):
        """Test creating a List of Sets."""
        db = DB(self.temp_path)

        # Create template for nested sets
        UserSet = Set.template(key_size=50)

        # Create list of sets
        groups = db.create_list("groups", UserSet)

        assert groups._is_nested == True
        assert len(groups) == 0

        db.close()

    def test_list_of_sets_append_and_access(self):
        """Test appending and accessing sets in a list."""
        db = DB(self.temp_path)

        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        # Append creates a new set
        admins = groups.append()
        assert isinstance(admins, Set)

        # Add items to the set
        admins.add("alice")
        admins.add("bob")

        assert len(admins) == 2
        assert "alice" in admins
        assert "bob" in admins

        # Update the stored reference
        groups.update_nested_ref(0, admins)

        # Access via index
        retrieved = groups[0]
        assert len(retrieved) == 2
        assert "alice" in retrieved

        db.close()

    def test_list_of_sets_multiple_items(self):
        """Test list with multiple sets."""
        db = DB(self.temp_path)

        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        # Create multiple groups
        group_names = ["admins", "editors", "viewers"]
        for i, group_name in enumerate(group_names):
            group = groups.append()
            group.add(f"user_{i}_1")
            group.add(f"user_{i}_2")
            groups.update_nested_ref(i, group)

        assert len(groups) == 3

        # Verify each group
        for i in range(3):
            group = groups[i]
            assert len(group) == 2

        db.close()

    def test_list_of_sets_iteration(self):
        """Test iterating over list of sets."""
        db = DB(self.temp_path)

        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        # Create groups
        for i in range(5):
            group = groups.append()
            group.add(f"user_{i}")
            groups.update_nested_ref(i, group)

        # Iterate
        count = 0
        for group in groups:
            assert isinstance(group, Set)
            count += 1

        assert count == 5

        db.close()

    def test_list_of_sets_persistence(self):
        """Test that list of sets persists across reopen."""
        db = DB(self.temp_path)

        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        # Create and populate
        admins = groups.append()
        admins.add("alice")
        admins.add("bob")
        groups.update_nested_ref(0, admins)

        editors = groups.append()
        editors.add("charlie")
        groups.update_nested_ref(1, editors)

        groups.save(force=True)
        db.close()

        # Reopen and verify
        db2 = DB(self.temp_path)
        groups2 = db2._datastructures.get("groups")
        if groups2 is None:
            UserSet2 = Set.template(key_size=50)
            groups2 = db2.create_list("groups", UserSet2)

        assert len(groups2) == 2

        admins2 = groups2[0]
        assert len(admins2) == 2
        assert "alice" in admins2

        editors2 = groups2[1]
        assert len(editors2) == 1
        assert "charlie" in editors2

        db2.close()

    def test_list_of_sets_many_items(self):
        """Test creating many sets in a list."""
        db = DB(self.temp_path, header_size=1024 * 1024)

        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        # Create 50 sets
        for i in range(50):
            group = groups.append()
            group.add(f"user_{i}")
            groups.update_nested_ref(i, group)

        groups.save(force=True)

        assert len(groups) == 50

        # Verify random access
        group_25 = groups[25]
        assert f"user_25" in group_25

        db.close()


class TestDictOfSets:
    """Test Dict of Sets (Dict[Set])."""

    def setup_method(self):
        """Create temp file for each test."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix=".loom")
        os.close(self.temp_fd)

    def teardown_method(self):
        """Clean up temp file."""
        try:
            os.unlink(self.temp_path)
        except (PermissionError, FileNotFoundError):
            pass

    def test_dict_of_sets_creation(self):
        """Test creating a Dict of Sets."""
        db = DB(self.temp_path)

        # Create template for nested sets
        TagSet = Set.template(key_size=50)

        # Create dict of sets
        user_tags = db.create_dict("user_tags", TagSet)

        assert user_tags._is_nested == True
        assert len(user_tags) == 0

        db.close()

    def test_dict_of_sets_set_and_get(self):
        """Test setting and getting sets in a dict."""
        db = DB(self.temp_path)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Access key auto-creates a set
        alice_tags = user_tags["alice"]
        assert isinstance(alice_tags, Set)

        # Add items to the set
        alice_tags.add("python")
        alice_tags.add("rust")

        assert len(alice_tags) == 2
        assert "python" in alice_tags
        assert "rust" in alice_tags

        db.close()

    def test_dict_of_sets_multiple_keys(self):
        """Test dict with multiple sets."""
        db = DB(self.temp_path)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Create sets for multiple users
        users = ["alice", "bob", "charlie"]
        for i, user in enumerate(users):
            tags = user_tags[user]
            for j in range(3):
                tags.add(f"tag_{i}_{j}")

        assert len(user_tags) == 3

        # Verify each user's tags
        for i, user in enumerate(users):
            tags = user_tags[user]
            assert len(tags) == 3
            assert f"tag_{i}_0" in tags

        db.close()

    def test_dict_of_sets_iteration(self):
        """Test iterating over dict of sets."""
        db = DB(self.temp_path)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Create sets
        for i in range(5):
            user = f"user_{i}"
            tags = user_tags[user]
            tags.add(f"tag_for_{user}")

        # Iterate over keys
        keys = list(user_tags.keys())
        assert len(keys) == 5
        assert set(keys) == {f"user_{i}" for i in range(5)}

        # Iterate over values
        for tags in user_tags.values():
            assert isinstance(tags, Set)
            assert len(tags) == 1

        db.close()

    def test_dict_of_sets_persistence(self):
        """Test that dict of sets persists across reopen."""
        db = DB(self.temp_path)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Create and populate
        alice_tags = user_tags["alice"]
        alice_tags.add("python")
        alice_tags.add("rust")

        bob_tags = user_tags["bob"]
        bob_tags.add("javascript")

        user_tags.save(force=True)
        db.close()

        # Reopen and verify
        db2 = DB(self.temp_path)
        user_tags2 = db2._datastructures.get("user_tags")
        if user_tags2 is None:
            TagSet2 = Set.template(key_size=50)
            user_tags2 = db2.create_dict("user_tags", TagSet2)

        assert len(user_tags2) == 2

        alice_tags2 = user_tags2["alice"]
        assert len(alice_tags2) == 2
        assert "python" in alice_tags2

        bob_tags2 = user_tags2["bob"]
        assert len(bob_tags2) == 1
        assert "javascript" in bob_tags2

        db2.close()

    def test_dict_of_sets_many_keys(self):
        """Test creating many sets in a dict."""
        db = DB(self.temp_path, header_size=1024 * 1024)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Create 50 sets
        for i in range(50):
            user = f"user_{i}"
            tags = user_tags[user]
            tags.add(f"tag_for_{user}")

        user_tags.save(force=True)

        assert len(user_tags) == 50

        # Verify random access
        tags_25 = user_tags["user_25"]
        assert len(tags_25) == 1
        assert "tag_for_user_25" in tags_25

        db.close()


class TestNestedSetCombined:
    """Test combined scenarios with both List[Set] and Dict[Set]."""

    def setup_method(self):
        """Create temp file for each test."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix=".loom")
        os.close(self.temp_fd)

    def teardown_method(self):
        """Clean up temp file."""
        try:
            os.unlink(self.temp_path)
        except (PermissionError, FileNotFoundError):
            pass

    def test_both_structures_same_db(self):
        """Test having both List[Set] and Dict[Set] in same database."""
        db = DB(self.temp_path)

        # Create List of Sets
        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        # Create Dict of Sets
        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Populate List of Sets
        admins = groups.append()
        admins.add("alice")
        groups.update_nested_ref(0, admins)

        # Populate Dict of Sets
        alice_tags = user_tags["alice"]
        alice_tags.add("python")

        # Verify both work
        assert len(groups) == 1
        assert len(user_tags) == 1
        assert "alice" in groups[0]
        assert "python" in user_tags["alice"]

        db.close()

    def test_persistence_both_structures(self):
        """Test persistence with both structures."""
        db = DB(self.temp_path)

        # Create structures
        UserSet = Set.template(key_size=50)
        groups = db.create_list("groups", UserSet)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Populate
        admins = groups.append()
        admins.add("alice")
        groups.update_nested_ref(0, admins)

        alice_tags = user_tags["alice"]
        alice_tags.add("python")

        groups.save(force=True)
        user_tags.save(force=True)
        db.close()

        # Reopen and verify
        db2 = DB(self.temp_path)

        # Reload groups
        groups2 = db2._datastructures.get("groups")
        if groups2 is None:
            UserSet2 = Set.template(key_size=50)
            groups2 = db2.create_list("groups", UserSet2)

        # Reload user_tags
        user_tags2 = db2._datastructures.get("user_tags")
        if user_tags2 is None:
            TagSet2 = Set.template(key_size=50)
            user_tags2 = db2.create_dict("user_tags", TagSet2)

        assert len(groups2) == 1
        assert len(user_tags2) == 1
        assert "alice" in groups2[0]
        assert "python" in user_tags2["alice"]

        db2.close()

    def test_set_operations_in_nested(self):
        """Test that set operations work correctly in nested context."""
        db = DB(self.temp_path)

        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict("user_tags", TagSet)

        # Test add/remove/discard
        alice_tags = user_tags["alice"]
        alice_tags.add("python")
        alice_tags.add("rust")
        alice_tags.add("go")
        assert len(alice_tags) == 3

        alice_tags.remove("go")
        assert len(alice_tags) == 2
        assert "go" not in alice_tags

        alice_tags.discard("nonexistent")  # Should not raise
        assert len(alice_tags) == 2

        # Test update
        alice_tags.update(["javascript", "typescript"])
        assert len(alice_tags) == 4

        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
