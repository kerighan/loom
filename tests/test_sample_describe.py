"""sample() and describe() across List, Dict, BTree and Collection — the
data-inspection helpers an agent/LLM uses to grasp a structure's contents."""

import os
import tempfile

import pytest

from loom import DB, Many


@pytest.fixture
def db():
    path = os.path.join(tempfile.mkdtemp(), "sd.db")
    database = DB(path, header_size=131072)
    yield database
    database.close()


# ── List ──────────────────────────────────────────────────────────────────────

def test_list_sample(db):
    lst = db.create_list("events", {"id": "uint64", "kind": "utf8[12]"})
    lst.append_many([{"id": i, "kind": "view"} for i in range(20)])
    s = lst.sample(5, seed=1)
    assert len(s) == 5
    assert all(isinstance(r, dict) and "id" in r for r in s)
    # reproducible
    assert s == lst.sample(5, seed=1)
    # head, no scan
    head = lst.sample(3, random=False)
    assert len(head) == 3 and head == lst.sample(3, random=False)


def test_list_describe(db):
    lst = db.create_list("events", {"id": "uint64", "kind": "utf8[12]"})
    lst.append_many([{"id": i, "kind": "view"} for i in range(5)])
    text = lst.describe(n=2, seed=1)
    assert "List 'events'" in text
    assert "5 element(s)" in text
    assert "id: uint64" in text          # friendly dtype name, not '<u8'
    assert "kind: utf8[12]" in text
    assert "sample (2 of 5):" in text


# ── Dict (natural element = (key, value)) ──────────────────────────────────────

def test_dict_sample_returns_pairs(db):
    d = db.create_dict("meta", {"version": "int64", "note": "text"})
    for i in range(15):
        d[f"v{i}"] = {"version": i, "note": f"n{i}"}
    s = d.sample(4, seed=2)
    assert len(s) == 4
    for key, value in s:                  # pairs
        assert isinstance(key, str)
        assert value["version"] == int(key[1:])


def test_dict_describe(db):
    d = db.create_dict("meta", {"version": "int64", "note": "text"})
    d["v1"] = {"version": 1, "note": "hi"}
    text = d.describe(seed=1)
    assert "Dict 'meta'" in text
    assert "version: int64" in text and "note: text" in text


# ── BTree ───────────────────────────────────────────────────────────────────

def test_btree_sample_and_describe(db):
    bt = db.create_btree("idx", {"score": "float64"}, key_size=8)
    for i in range(15):
        bt[f"k{i:02d}"] = {"score": i / 2}
    s = bt.sample(3, seed=1)
    assert len(s) == 3
    assert all(isinstance(k, str) for k, _ in s)
    text = bt.describe(n=2, seed=1)
    assert "BTree 'idx'" in text and "score: float64" in text


# ── Collection ────────────────────────────────────────────────────────────────

def _posts(db):
    col = db.collection(
        "posts",
        {"id": "utf8[8]", "user": "utf8[16]", "eng": "int64", "body": "text"},
        indexes={"id": "primary", "user": Many(sort="eng", desc=True), "eng": "range"},
    )
    col.insert_many(
        [{"id": f"p{i}", "user": "alice" if i % 2 else "bob", "eng": i * 10,
          "body": f"post {i}"} for i in range(12)]
    )
    return col


def test_collection_describe(db):
    text = _posts(db).describe(n=2, seed=3)
    assert "Collection 'posts'" in text
    assert "12 record(s)" in text
    assert "key='id'" in text
    assert "eng: int64" in text and "body: text" in text
    # indexes section
    assert "user: many (sort=eng, desc)" in text
    assert "eng: range" in text
    assert "sample (2 of 12):" in text


def test_set_sample_and_describe(db):
    # A Set has no item_schema — describe must still work (no schema section).
    s = db.create_set("active", key_size=12)
    for u in ["alice", "bob", "carol", "dave", "erin"]:
        s.add(u)
    sample = s.sample(3, seed=1)
    assert len(sample) == 3 and all(isinstance(x, str) for x in sample)
    text = s.describe(n=2, seed=1)
    assert "Set 'active'" in text and "5 element(s)" in text
    assert "schema:" not in text


def test_describe_empty_structure(db):
    d = db.create_dict("empty", {"x": "int64"})
    text = d.describe()
    assert "Dict 'empty'" in text
    assert "0 element(s)" in text
    # no sample section when empty
    assert "sample" not in text
