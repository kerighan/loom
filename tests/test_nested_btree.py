"""Test cross-nested structures involving BTree.

These tests verify that we can create and use:
- Dict[BTree] - a dict where each value is a BTree
- BTree[Dict] - a btree where each value is a Dict
"""

import os
import tempfile

import pytest

from loom import DB
from loom.datastructures.btree import BTree
from loom.datastructures.dict import Dict


class TestDictOfBtrees:
    def setup_method(self):
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix=".loom")
        os.close(self.temp_fd)

    def teardown_method(self):
        try:
            os.unlink(self.temp_path)
        except (PermissionError, FileNotFoundError):
            pass

    def test_creation_and_basic_ops(self):
        db = DB(self.temp_path)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserBTree = BTree.template(user_dataset, cache_size=10)

        teams = db.create_dict("teams", UserBTree)

        eng = teams["engineering"]
        eng["alice"] = {"id": 1, "name": "Alice"}
        eng["bob"] = {"id": 2, "name": "Bob"}

        assert eng["alice"]["name"] == "Alice"
        assert list(eng.keys()) == ["alice", "bob"]

        db.close()

    def test_persistence(self):
        db = DB(self.temp_path)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserBTree = BTree.template(user_dataset, cache_size=10)
        teams = db.create_dict("teams", UserBTree)

        eng = teams["engineering"]
        eng["alice"] = {"id": 1, "name": "Alice"}
        db.close()

        db = DB(self.temp_path)
        teams = db.create_dict("teams", None)

        eng = teams["engineering"]
        assert eng["alice"]["name"] == "Alice"
        db.close()


class TestBtreeOfDicts:
    def setup_method(self):
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix=".loom")
        os.close(self.temp_fd)

    def teardown_method(self):
        try:
            os.unlink(self.temp_path)
        except (PermissionError, FileNotFoundError):
            pass

    def test_creation_and_nested_mutation_updates_parent(self):
        db = DB(self.temp_path, header_size=1024 * 1024)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)

        teams = db.create_btree("teams", UserDict)

        eng = teams["engineering"]
        eng["alice"] = {"id": 1, "name": "Alice"}

        # Re-fetch (forces rebuild from stored ref)
        eng2 = teams["engineering"]
        assert eng2["alice"]["name"] == "Alice"

        # Mutate nested and ensure ref update persists
        eng2["alice"] = {"id": 1, "name": "Alice Smith"}
        eng3 = teams["engineering"]
        assert eng3["alice"]["name"] == "Alice Smith"

        db.close()

    def test_persistence(self):
        db = DB(self.temp_path, header_size=1024 * 1024)

        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDict = Dict.template(user_dataset, cache_size=10)
        teams = db.create_btree("teams", UserDict)

        eng = teams["engineering"]
        eng["alice"] = {"id": 1, "name": "Alice"}

        db.close()

        db = DB(self.temp_path, header_size=1024 * 1024)
        teams = db.create_btree("teams", None)

        eng = teams["engineering"]
        assert eng["alice"]["name"] == "Alice"
        db.close()
