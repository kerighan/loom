"""Test cross-nested data structures: List of Dicts and Dict of Lists.

These tests verify that we can create and use:
- List[Dict] - A list where each element is a Dict
- Dict[List] - A dict where each value is a List
"""

import os
import tempfile
import pytest
from loom import DB
from loom.datastructures import List
from loom.datastructures.dict import Dict


class TestListOfDicts:
    """Test List of Dicts (List[Dict])."""

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

    def test_list_of_dicts_creation(self):
        """Test creating a List of Dicts."""
        db = DB(self.temp_path)

        # Create dataset for dict items
        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )

        # Create template for nested dicts
        UserDict = Dict.template(user_dataset, cache_size=10)

        # Create list of dicts
        teams = db.create_list("teams", UserDict)

        assert teams._is_nested == True
        assert len(teams) == 0

        db.close()

    def test_list_of_dicts_append_and_access(self):
        """Test appending and accessing dicts in a list."""
        db = DB(self.temp_path)

        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_list("teams", UserDict)

        # Append creates a new dict
        eng_team = teams.append()
        assert isinstance(eng_team, Dict)

        # Add items to the dict
        eng_team["alice"] = {"id": 1, "name": "Alice", "score": 95.5}
        eng_team["bob"] = {"id": 2, "name": "Bob", "score": 87.0}

        assert len(eng_team) == 2
        assert eng_team["alice"]["name"] == "Alice"
        assert eng_team["bob"]["score"] == 87.0

        # Update the stored reference
        teams.update_nested_ref(0, eng_team)

        # Access via index
        retrieved = teams[0]
        assert len(retrieved) == 2
        assert retrieved["alice"]["id"] == 1

        db.close()

    def test_list_of_dicts_multiple_items(self):
        """Test list with multiple dicts."""
        db = DB(self.temp_path)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_list("teams", UserDict)

        # Create multiple teams
        team_names = ["engineering", "sales", "marketing"]
        for i, team_name in enumerate(team_names):
            team = teams.append()
            team[f"member_{i}_1"] = {"id": i * 10 + 1, "name": f"{team_name}_lead"}
            team[f"member_{i}_2"] = {"id": i * 10 + 2, "name": f"{team_name}_member"}
            teams.update_nested_ref(i, team)

        assert len(teams) == 3

        # Verify each team
        for i in range(3):
            team = teams[i]
            assert len(team) == 2

        db.close()

    def test_list_of_dicts_iteration(self):
        """Test iterating over list of dicts."""
        db = DB(self.temp_path)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_list("teams", UserDict)

        # Create teams
        for i in range(5):
            team = teams.append()
            team[f"user_{i}"] = {"id": i, "name": f"User {i}"}
            teams.update_nested_ref(i, team)

        # Iterate
        count = 0
        for team in teams:
            assert isinstance(team, Dict)
            count += 1

        assert count == 5

        db.close()

    def test_list_of_dicts_persistence(self):
        """Test that list of dicts persists across reopen."""
        db = DB(self.temp_path)

        user_dataset = db.create_dataset(
            "users", id="uint32", name="U50", score="float32"
        )
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_list("teams", UserDict)

        # Create and populate
        eng = teams.append()
        eng["alice"] = {"id": 1, "name": "Alice", "score": 95.5}
        eng["bob"] = {"id": 2, "name": "Bob", "score": 87.0}
        teams.update_nested_ref(0, eng)

        sales = teams.append()
        sales["charlie"] = {"id": 3, "name": "Charlie", "score": 92.0}
        teams.update_nested_ref(1, sales)

        teams.save(force=True)
        db.close()

        # Reopen and verify
        db2 = DB(self.temp_path)
        teams2 = db2._datastructures.get("teams")
        if teams2 is None:
            user_dataset2 = db2.get_dataset("users")
            UserDict2 = Dict.template(user_dataset2, cache_size=10)
            teams2 = db2.create_list("teams", UserDict2)

        assert len(teams2) == 2

        eng2 = teams2[0]
        assert len(eng2) == 2
        assert eng2["alice"]["name"] == "Alice"

        sales2 = teams2[1]
        assert len(sales2) == 1
        assert sales2["charlie"]["score"] == 92.0

        db2.close()

    def test_list_of_dicts_many_items(self):
        """Test creating many dicts in a list."""
        db = DB(self.temp_path, header_size=1024 * 1024)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=0)
        departments = db.create_list("departments", UserDict)

        # Create 50 dicts
        for i in range(50):
            dept = departments.append()
            dept[f"user_{i}"] = {"id": i, "name": f"User {i}"}
            departments.update_nested_ref(i, dept)

        departments.save(force=True)

        assert len(departments) == 50

        # Verify random access
        dept_25 = departments[25]
        assert dept_25[f"user_25"]["id"] == 25

        db.close()


class TestDictOfLists:
    """Test Dict of Lists (Dict[List])."""

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

    def test_dict_of_lists_creation(self):
        """Test creating a Dict of Lists."""
        db = DB(self.temp_path)

        # Create dataset for list items
        task_dataset = db.create_dataset(
            "tasks", id="uint32", title="U100", priority="uint8"
        )

        # Create template for nested lists
        TaskList = List.template(task_dataset, cache_size=10)

        # Create dict of lists
        user_tasks = db.create_dict("user_tasks", TaskList)

        assert user_tasks._is_nested == True
        assert len(user_tasks) == 0

        db.close()

    def test_dict_of_lists_set_and_get(self):
        """Test setting and getting lists in a dict."""
        db = DB(self.temp_path)

        task_dataset = db.create_dataset(
            "tasks", id="uint32", title="U100", priority="uint8"
        )
        TaskList = List.template(task_dataset, cache_size=10)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Access key auto-creates a list
        alice_tasks = user_tasks["alice"]
        assert isinstance(alice_tasks, List)

        # Add items to the list
        alice_tasks.append({"id": 1, "title": "Review PR", "priority": 1})
        alice_tasks.append({"id": 2, "title": "Write tests", "priority": 2})

        assert len(alice_tasks) == 2
        assert alice_tasks[0]["title"] == "Review PR"
        assert alice_tasks[1]["priority"] == 2

        db.close()

    def test_dict_of_lists_multiple_keys(self):
        """Test dict with multiple lists."""
        db = DB(self.temp_path)

        task_dataset = db.create_dataset("tasks", id="uint32", title="U100")
        TaskList = List.template(task_dataset, cache_size=10)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Create lists for multiple users
        users = ["alice", "bob", "charlie"]
        for i, user in enumerate(users):
            tasks = user_tasks[user]
            for j in range(3):
                tasks.append({"id": i * 10 + j, "title": f"Task {j} for {user}"})

        assert len(user_tasks) == 3

        # Verify each user's tasks
        for i, user in enumerate(users):
            tasks = user_tasks[user]
            assert len(tasks) == 3
            assert tasks[0]["title"] == f"Task 0 for {user}"

        db.close()

    def test_dict_of_lists_iteration(self):
        """Test iterating over dict of lists."""
        db = DB(self.temp_path)

        task_dataset = db.create_dataset("tasks", id="uint32", title="U100")
        TaskList = List.template(task_dataset, cache_size=10)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Create lists
        for i in range(5):
            user = f"user_{i}"
            tasks = user_tasks[user]
            tasks.append({"id": i, "title": f"Task for {user}"})

        # Iterate over keys
        keys = list(user_tasks.keys())
        assert len(keys) == 5
        assert set(keys) == {f"user_{i}" for i in range(5)}

        # Iterate over values
        for tasks in user_tasks.values():
            assert isinstance(tasks, List)
            assert len(tasks) == 1

        db.close()

    def test_dict_of_lists_persistence(self):
        """Test that dict of lists persists across reopen."""
        db = DB(self.temp_path)

        task_dataset = db.create_dataset(
            "tasks", id="uint32", title="U100", priority="uint8"
        )
        TaskList = List.template(task_dataset, cache_size=10)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Create and populate
        alice_tasks = user_tasks["alice"]
        alice_tasks.append({"id": 1, "title": "Review PR", "priority": 1})
        alice_tasks.append({"id": 2, "title": "Write tests", "priority": 2})

        bob_tasks = user_tasks["bob"]
        bob_tasks.append({"id": 3, "title": "Deploy", "priority": 1})

        user_tasks.save(force=True)
        db.close()

        # Reopen and verify
        db2 = DB(self.temp_path)
        user_tasks2 = db2._datastructures.get("user_tasks")
        if user_tasks2 is None:
            task_dataset2 = db2.get_dataset("tasks")
            TaskList2 = List.template(task_dataset2, cache_size=10)
            user_tasks2 = db2.create_dict("user_tasks", TaskList2)

        assert len(user_tasks2) == 2

        alice_tasks2 = user_tasks2["alice"]
        assert len(alice_tasks2) == 2
        assert alice_tasks2[0]["title"] == "Review PR"

        bob_tasks2 = user_tasks2["bob"]
        assert len(bob_tasks2) == 1
        assert bob_tasks2[0]["priority"] == 1

        db2.close()

    def test_dict_of_lists_many_keys(self):
        """Test creating many lists in a dict."""
        db = DB(self.temp_path, header_size=1024 * 1024)

        task_dataset = db.create_dataset("tasks", id="uint32", title="U50")
        TaskList = List.template(task_dataset, cache_size=0)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Create 50 lists
        for i in range(50):
            user = f"user_{i}"
            tasks = user_tasks[user]
            tasks.append({"id": i, "title": f"Task for {user}"})

        user_tasks.save(force=True)

        assert len(user_tasks) == 50

        # Verify random access
        tasks_25 = user_tasks["user_25"]
        assert len(tasks_25) == 1
        assert tasks_25[0]["id"] == 25

        db.close()


class TestCrossNestedCombined:
    """Test combined scenarios with both List[Dict] and Dict[List]."""

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
        """Test having both List[Dict] and Dict[List] in same database."""
        db = DB(self.temp_path)

        # Create List of Dicts
        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_list("teams", UserDict)

        # Create Dict of Lists
        task_dataset = db.create_dataset("tasks", id="uint32", title="U100")
        TaskList = List.template(task_dataset, cache_size=10)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Populate List of Dicts
        eng = teams.append()
        eng["alice"] = {"id": 1, "name": "Alice"}
        teams.update_nested_ref(0, eng)

        # Populate Dict of Lists
        alice_tasks = user_tasks["alice"]
        alice_tasks.append({"id": 1, "title": "Review PR"})

        # Verify both work
        assert len(teams) == 1
        assert len(user_tasks) == 1
        assert teams[0]["alice"]["name"] == "Alice"
        assert user_tasks["alice"][0]["title"] == "Review PR"

        db.close()

    def test_persistence_both_structures(self):
        """Test persistence with both structures."""
        db = DB(self.temp_path)

        # Create structures
        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_list("teams", UserDict)

        task_dataset = db.create_dataset("tasks", id="uint32", title="U100")
        TaskList = List.template(task_dataset, cache_size=10)
        user_tasks = db.create_dict("user_tasks", TaskList)

        # Populate
        eng = teams.append()
        eng["alice"] = {"id": 1, "name": "Alice"}
        teams.update_nested_ref(0, eng)

        alice_tasks = user_tasks["alice"]
        alice_tasks.append({"id": 1, "title": "Review PR"})

        teams.save(force=True)
        user_tasks.save(force=True)
        db.close()

        # Reopen and verify
        db2 = DB(self.temp_path)

        # Reload teams
        teams2 = db2._datastructures.get("teams")
        if teams2 is None:
            user_dataset2 = db2.get_dataset("users")
            UserDict2 = Dict.template(user_dataset2, cache_size=10)
            teams2 = db2.create_list("teams", UserDict2)

        # Reload user_tasks
        user_tasks2 = db2._datastructures.get("user_tasks")
        if user_tasks2 is None:
            task_dataset2 = db2.get_dataset("tasks")
            TaskList2 = List.template(task_dataset2, cache_size=10)
            user_tasks2 = db2.create_dict("user_tasks", TaskList2)

        assert len(teams2) == 1
        assert len(user_tasks2) == 1
        assert teams2[0]["alice"]["name"] == "Alice"
        assert user_tasks2["alice"][0]["title"] == "Review PR"

        db2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
