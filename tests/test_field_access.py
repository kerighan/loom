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
