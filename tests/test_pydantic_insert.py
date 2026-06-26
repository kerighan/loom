"""Inserting Pydantic model instances directly (no manual .model_dump())."""

import os
import tempfile
from datetime import datetime

import pytest
from pydantic import BaseModel

from loom import DB


class Post(BaseModel):
    id: str
    score: float
    created_at: datetime
    text: str


def _post(i, score=1.0):
    return Post(id=f"p{i}", score=score, created_at=datetime(2026, 1, 1 + i), text=f"t{i}")


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    database = DB(os.path.join(tmp, "pyd.db"))
    database.open()
    yield database
    database.close()


def test_dataset_accepts_model(db):
    ds = db.create_dataset("posts", Post)
    ref = ds.insert(_post(1, 9.5))
    assert ds[ref.addr]["id"] == "p1"
    assert ds[ref.addr]["score"] == 9.5
    assert isinstance(ds[ref.addr]["created_at"], datetime)
    # __setitem__ path too
    ds[ref.addr] = _post(2, 3.0)
    assert ds[ref.addr]["id"] == "p2"


def test_list_accepts_model(db):
    ds = db.create_dataset("posts", Post)
    lst = db.create_list("feed", ds)
    lst.append(_post(1))
    lst.append_many([_post(2), _post(3)])
    lst[0] = _post(9)
    assert [r["id"] for r in lst] == ["p9", "p2", "p3"]


def test_dict_accepts_model(db):
    ds = db.create_dataset("posts", Post)
    d = db.create_dict("by_id", ds)
    d["p1"] = _post(1, 7.0)
    assert d["p1"]["score"] == 7.0
    d.set_batch([("p2", _post(2)), ("p3", _post(3))])
    assert set(d.keys()) == {"p1", "p2", "p3"}


def test_btree_accepts_model(db):
    ds = db.create_dataset("posts", Post)
    bt = db.create_btree("bt", ds, key_size=16)
    bt["p1"] = _post(1, 5.0)
    assert bt["p1"]["score"] == 5.0
    # bulk_load on an empty tree
    bt2 = db.create_btree("bt2", ds, key_size=16)
    bt2.bulk_load([("p2", _post(2)), ("p3", _post(3))])
    assert [k for k, _ in bt2.items()] == ["p2", "p3"]


def test_queue_accepts_model(db):
    q = db.create_queue("q", Post)
    q.push(_post(1))
    q.push_many([_post(2), _post(3)])
    assert [q.pop()["id"] for _ in range(3)] == ["p1", "p2", "p3"]


def test_priority_queue_accepts_model(db):
    pq = db.create_priority_queue("pq", Post)
    pq.push(_post(1), 5.0)
    pq.push_many([(_post(2), 9.0), (_post(3), 1.0)])
    assert [pq.pop()["id"] for _ in range(3)] == ["p2", "p1", "p3"]


def test_collection_accepts_model(db):
    col = db.collection("coll", Post,
                        indexes={"id": "primary", "score": "range"})
    col.insert(_post(1, 10.0))
    col.insert_many([_post(2, 50.0), _post(3, 99.0)])
    assert col["p1"]["id"] == "p1"
    assert {r["id"] for r in col.range("score", 40, None)} == {"p2", "p3"}


# ── auto-backing-dataset shorthand: pass a model / schema directly ────────────

def test_dict_auto_dataset_from_model(db):
    # No explicit create_dataset — backing dataset is created automatically.
    d = db.create_dict("by_id", Post)
    d["p1"] = _post(1, 7.0)
    assert d["p1"]["score"] == 7.0
    assert "_by_id_ds" in db._datasets  # hidden backing dataset


def test_dict_auto_dataset_from_schema(db):
    d = db.create_dict("cfg", {"k": "utf8[20]", "n": "int64"})
    d["a"] = {"k": "a", "n": 42}
    assert d["a"] == {"k": "a", "n": 42}


def test_list_auto_dataset_from_model(db):
    lst = db.create_list("feed", Post)
    lst.append(_post(1))
    assert lst[0]["id"] == "p1"


def test_btree_auto_dataset_from_model(db):
    bt = db.create_btree("bt", Post, key_size=16)
    bt["p1"] = _post(1, 5.0)
    assert bt["p1"]["score"] == 5.0


def test_auto_dataset_survives_reopen(tmp_path):
    path = str(tmp_path / "reopen.db")
    db = DB(path)
    db.open()
    db.create_dict("meta", Post)["p1"] = _post(1, 3.0)
    db.create_list("feed", Post).append(_post(2))
    db.create_btree("bt", Post, key_size=16)["p3"] = _post(3)
    db.close()

    db2 = DB(path)
    db2.open()
    assert db2["meta"]["p1"]["id"] == "p1"
    assert db2["feed"][0]["id"] == "p2"
    assert db2["bt"]["p3"]["id"] == "p3"
    db2.close()
