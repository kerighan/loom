"""Test nested Dict functionality."""

import os
import tempfile
import pytest
from loom import DB
from loom.datastructures.dict import Dict


class TestNestedDict:
    """Test nested Dict (Dict of Dicts)."""

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

    def test_nested_dict_creation(self):
        """Test creating a nested Dict."""
        db = DB(self.temp_path)
        
        # Create inner dict schema
        user_dataset = db.create_dataset("users", id="uint32", name="U50", score="float32")
        
        # Create template for nested dicts
        UserDict = Dict.template(user_dataset, cache_size=10)
        
        # Create outer dict (dict of dicts)
        teams = db.create_dict("teams", UserDict)
        
        assert teams._is_nested == True
        assert len(teams) == 0
        
        db.close()

    def test_nested_dict_set_and_get(self):
        """Test setting and getting nested dicts."""
        db = DB(self.temp_path)
        
        user_dataset = db.create_dataset("users", id="uint32", name="U50", score="float32")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_dict("teams", UserDict)
        
        # Create nested dict by accessing key
        eng_team = teams["engineering"]
        
        # Add items to nested dict
        eng_team["alice"] = {"id": 1, "name": "Alice", "score": 95.5}
        eng_team["bob"] = {"id": 2, "name": "Bob", "score": 87.0}
        
        assert len(eng_team) == 2
        assert eng_team["alice"]["name"] == "Alice"
        assert eng_team["bob"]["score"] == 87.0
        
        db.close()

    def test_nested_dict_persistence(self):
        """Test that nested dicts persist across reopen."""
        db = DB(self.temp_path)
        
        user_dataset = db.create_dataset("users", id="uint32", name="U50", score="float32")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_dict("teams", UserDict)
        
        # Create and populate nested dict
        eng = teams["engineering"]
        eng["alice"] = {"id": 1, "name": "Alice", "score": 95.5}
        eng["bob"] = {"id": 2, "name": "Bob", "score": 87.0}
        
        sales = teams["sales"]
        sales["charlie"] = {"id": 3, "name": "Charlie", "score": 92.0}
        
        teams.save(force=True)
        db.close()
        
        # Reopen and verify
        db2 = DB(self.temp_path)
        # Use _datastructures to get existing dict (loaded from registry)
        teams2 = db2._datastructures.get("teams")
        if teams2 is None:
            # Fallback: recreate with same params (will load from metadata)
            user_dataset2 = db2.get_dataset("users")
            UserDict2 = Dict.template(user_dataset2, cache_size=10)
            teams2 = db2.create_dict("teams", UserDict2)
        
        assert len(teams2) == 2
        
        eng2 = teams2["engineering"]
        assert len(eng2) == 2
        assert eng2["alice"]["name"] == "Alice"
        
        sales2 = teams2["sales"]
        assert len(sales2) == 1
        assert sales2["charlie"]["score"] == 92.0
        
        db2.close()

    def test_many_nested_dicts(self):
        """Test creating many nested dicts (stress test for header pollution)."""
        db = DB(self.temp_path, header_size=1024*1024)  # 1MB header
        
        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=0)
        departments = db.create_dict("departments", UserDict)
        
        # Create 100 nested dicts
        for i in range(100):
            dept = departments[f"dept_{i}"]
            dept[f"user_{i}"] = {"id": i, "name": f"User {i}"}
        
        departments.save(force=True)
        
        assert len(departments) == 100
        
        # Verify random access
        dept_50 = departments["dept_50"]
        assert dept_50["user_50"]["id"] == 50
        
        db.close()

    def test_nested_dict_iteration(self):
        """Test iterating over nested dicts."""
        db = DB(self.temp_path)
        
        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_dict("teams", UserDict)
        
        # Create nested dicts
        for team_name in ["alpha", "beta", "gamma"]:
            team = teams[team_name]
            team["member1"] = {"id": 1, "name": f"{team_name}_1"}
            team["member2"] = {"id": 2, "name": f"{team_name}_2"}
        
        # Iterate over teams
        team_names = list(teams.keys())
        assert len(team_names) == 3
        assert set(team_names) == {"alpha", "beta", "gamma"}
        
        # Iterate over values (nested dicts)
        for team in teams.values():
            assert len(team) == 2
        
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
