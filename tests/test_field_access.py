"""Single-field access on Dict: d[key, field] get / d[key, field] = v set.

Reads/writes one field of a record in place (no full deserialization), via the
dataset's read_field / write_field.  The record's address is unchanged on a
field write, so cached addresses and the other fields stay valid — ideal for
counters (likes / views) and partial updates.
"""

import os
import tempfile

import pytest

from loom.database import DB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


def _dict(db):
    ds = db.create_dataset("u", id="uint32", name="U20", likes="int64",
                           bio="text", tag="utf8[16]")
    d = db.create_dict("d", ds)
    d["alice"] = {"id": 1, "name": "Alice", "likes": 5, "bio": "hello world", "tag": "vip"}
    return d


class TestFieldGet:
    def test_get_each_field_type(self, db):
        d = _dict(db)
        assert int(d["alice", "likes"]) == 5      # int
        assert str(d["alice", "name"]) == "Alice"  # U
        assert d["alice", "bio"] == "hello world"  # text (resolved)
        assert d["alice", "tag"] == "vip"          # utf8 (resolved)

    def test_get_missing_key_raises(self, db):
        d = _dict(db)
        with pytest.raises(KeyError):
            d["bob", "likes"]

    def test_get_unknown_field_raises(self, db):
        d = _dict(db)
        with pytest.raises(ValueError):
            d["alice", "nope"]


class TestFieldSet:
    def test_set_leaves_other_fields_intact(self, db):
        d = _dict(db)
        d["alice", "likes"] = 42
        assert int(d["alice", "likes"]) == 42
        assert d["alice"] == {"id": 1, "name": "Alice", "likes": 42,
                              "bio": "hello world", "tag": "vip"}

    def test_increment_pattern(self, db):
        d = _dict(db)
        for _ in range(10):
            d["alice", "likes"] = d["alice", "likes"] + 1
        assert int(d["alice", "likes"]) == 15

    def test_set_text_field(self, db):
        d = _dict(db)
        d["alice", "bio"] = "updated café 日本"
        assert d["alice", "bio"] == "updated café 日本"
        assert d["alice"]["bio"] == "updated café 日本"

    def test_set_missing_key_raises(self, db):
        d = _dict(db)
        with pytest.raises(KeyError):
            d["bob", "likes"] = 1

    def test_survives_reopen(self, db):
        d = _dict(db)
        d["alice", "likes"] = 99
        d["alice", "bio"] = "persisted"
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            d2 = db2._datastructures["d"]
            assert int(d2["alice", "likes"]) == 99
            assert d2["alice", "bio"] == "persisted"
            assert d2["alice"]["name"] == "Alice"
        finally:
            db2.close()

    def test_keys_intact_after_field_set(self, db):
        d = _dict(db)
        d["bob"] = {"id": 2, "name": "Bob", "likes": 0, "bio": "", "tag": ""}
        d["alice", "likes"] = 7
        assert sorted(d.keys()) == ["alice", "bob"]   # _key untouched by field write


class TestGetRef:
    """Ref handles from Dict/List/BTree: cached-address reads/writes, with the
    internal _key (Dict/BTree) and valid (List) fields hidden on get() and
    preserved on a full set()."""

    def test_dict_ref_get_hides_key_and_set_preserves_it(self, db):
        d = _dict(db)
        d["bob"] = {"id": 2, "name": "Bob", "likes": 0, "bio": "", "tag": ""}
        r = d.get_ref("alice")
        assert "_key" not in r.get()
        assert r.get()["name"] == "Alice"
        r["likes"] = r["likes"] + 5                # single field
        r.update(name="Alicia")                    # partial
        assert d["alice"]["likes"] == 10 and d["alice"]["name"] == "Alicia"
        r.set({"id": 1, "name": "A2", "likes": 1, "bio": "x", "tag": "y"})  # full
        assert d["alice"]["name"] == "A2"
        assert sorted(d.keys()) == ["alice", "bob"]   # _key preserved through set()

    def test_dict_ref_missing_raises(self, db):
        d = _dict(db)
        with pytest.raises(KeyError):
            d.get_ref("nobody")

    def test_list_ref_hides_valid(self, db):
        ds = db.create_dataset("li", v="int64", tag="U10")
        lst = db.create_list("l", ds)
        lst.append({"v": 5, "tag": "a"})
        r = lst.get_ref(0)
        assert r.get() == {"v": 5, "tag": "a"}     # no 'valid'
        r["v"] = r["v"] + 10
        assert lst[0] == {"v": 15, "tag": "a"}

    def test_btree_ref(self, db):
        ds = db.create_dataset("bd", v="int64", name="U10")
        bt = db.create_btree("bt", ds)
        bt["k"] = {"v": 1, "name": "x"}
        r = bt.get_ref("k")
        assert "_key" not in r.get()
        r["v"] = 42
        assert bt["k"]["v"] == 42
        assert list(bt.keys()) == ["k"]            # _key intact

    def test_ref_survives_reopen(self, db):
        d = _dict(db)
        d.get_ref("alice").update(likes=77)
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            assert db2._datastructures["d"]["alice"]["likes"] == 77
        finally:
            db2.close()
