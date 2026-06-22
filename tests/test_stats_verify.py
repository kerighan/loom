"""db.stats(), db.verify(), Dict.get_many()."""

import os
import tempfile

import pytest

from loom import DB, Many


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = DB(os.path.join(d, "s.db"))
    database.open()
    yield database
    database.close()


def test_stats(db):
    ds = db.create_dataset("users", id="int64", name="utf8[16]")
    d = db.create_dict("by_name", ds)
    for i in range(10):
        d[f"u{i}"] = {"id": i, "name": f"n{i}"}
    col = db.collection("posts", {"pid": "utf8[8]", "cat": "utf8[8]"},
                        indexes={"pid": "primary", "cat": Many(sort="pid")})
    col.insert_many([{"pid": f"p{i}", "cat": "a"} for i in range(5)])

    s = db.stats()
    assert s["file_size"] > 0 and s["allocated_bytes"] > 0
    assert 0.0 <= s["fragmentation"] <= 1.0
    assert s["collections"]["posts"]["records"] == 5
    assert s["collections"]["posts"]["indexes"] == ["cat"]
    # top-level structures only (collection internals excluded)
    assert s["structures"]["by_name"] == {"type": "Dict", "length": 10}
    assert "posts__primary" not in s["structures"]


def test_verify_clean(db):
    col = db.collection("posts", {"pid": "utf8[8]", "cat": "utf8[8]"},
                        indexes={"pid": "primary", "cat": Many(sort="pid")})
    col.insert_many([{"pid": f"p{i}", "cat": "a"} for i in range(5)])
    rep = db.verify()
    assert rep["ok"] is True and rep["issues"] == []


def test_verify_detects_dangling_index(db):
    col = db.collection("posts", {"pid": "utf8[8]", "cat": "utf8[8]"},
                        indexes={"pid": "primary", "cat": Many(sort="pid")})
    col.insert_many([{"pid": f"p{i}", "cat": "a"} for i in range(5)])
    del col._primary["p2"]                       # desync: index still references p2
    rep = db.verify()
    assert rep["ok"] is False
    assert any("p2" in issue for issue in rep["issues"])


def test_get_many(db):
    ds = db.create_dataset("users", id="int64", name="utf8[16]")
    d = db.create_dict("by_name", ds)
    for i in range(20):
        d[f"u{i}"] = {"id": i, "name": f"n{i}"}
    got = d.get_many(["u1", "u5", "u999", "u10"])
    assert set(got) == {"u1", "u5", "u10"}        # missing key dropped
    assert got["u5"] == d["u5"]                    # consistent with []
    assert "_key" not in got["u1"]                 # internal field hidden
