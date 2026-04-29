import os
import tempfile
import pytest

from loom import DB


class TestRefBasics:
    def test_dataset_insert_returns_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                users = db.create_dataset("users", id="uint32", name="U50")
                ref = users.insert({"id": 1, "name": "Alice"})

                rec = users[ref.addr]
                assert rec["id"] == 1
                assert rec["name"] == "Alice"

                assert ref.get()["name"] == "Alice"

    def test_ref_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                users = db.create_dataset("users", id="uint32", name="U50")
                ref = users.insert({"id": 1, "name": "Alice"})

                ref.update(name="Bob")
                assert users[ref.addr]["name"] == "Bob"


class TestRefAcrossStructures:
    def test_ref_can_be_shared_between_dict_and_btree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, header_size=1024 * 1024) as db:
                users = db.create_dataset("users", id="uint32", name="U50")
                # store_key=False keeps the Dict's values dataset
                # pointing at the user's dataset (no internal wrapper),
                # which is required for cross-structure Ref sharing.
                d = db.create_dict("d", users, cache_size=100, store_key=False)
                b = db.create_btree("b", users)

                ref = users.insert({"id": 1, "name": "Alice"})

                d["truc"] = ref
                b["autre_truc"] = ref

                assert d["truc"]["name"] == "Alice"
                assert b["autre_truc"]["name"] == "Alice"

                ref.update(name="Bob")

                assert d["truc"]["name"] == "Bob"
                assert b["autre_truc"]["name"] == "Bob"

    def test_ref_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                users1 = db.create_dataset("users1", id="uint32", name="U50")
                users2 = db.create_dataset("users2", id="uint32", name="U50")

                d = db.create_dict("d", users1)
                ref = users2.insert({"id": 1, "name": "Alice"})

                with pytest.raises(TypeError):
                    d["truc"] = ref

    def test_get_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                users = db.create_dataset("users", id="uint32", name="U50")
                d = db.create_dict("d", users, cache_size=100)

                d["truc"] = {"id": 1, "name": "Alice"}
                ref = d.get_ref("truc")

                assert ref.get()["name"] == "Alice"

                ref.update(name="Bob")
                assert d["truc"]["name"] == "Bob"
