"""BTree with integer keys (int_keys=True): numeric ordering, not lexicographic."""

import os
import tempfile

import pytest

from loom import DB


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = DB(os.path.join(d, "i.db"))
    database.open()
    yield database
    database.close()


def _bt(db):
    ds = db.create_dataset("v", n="int64")
    return db.create_btree("bt", ds, int_keys=True)


def test_numeric_ordering(db):
    bt = _bt(db)
    for k in [5, 10, 2, 100, -3, 1, -100, 50]:
        bt[k] = {"n": k}
    assert list(bt.keys()) == [-100, -3, 1, 2, 5, 10, 50, 100]   # numeric, not "10"<"5"
    assert bt.min() == -100 and bt.max() == 100
    assert bt[10] == {"n": 10} and 10 in bt and 999 not in bt


def test_range_and_reverse(db):
    bt = _bt(db)
    for k in range(0, 100, 10):
        bt[k] = {"n": k}
    assert [k for k, _ in bt.range(20, 50)] == [20, 30, 40, 50]
    assert [k for k, _ in bt.range(70, None, reverse=True)] == [90, 80, 70]
    assert [k for k, _ in bt.range(None, 20)] == [0, 10, 20]


def test_delete_and_items(db):
    bt = _bt(db)
    for k in [3, 1, 2]:
        bt[k] = {"n": k * 10}
    del bt[2]
    assert [(k, v["n"]) for k, v in bt.items()] == [(1, 10), (3, 30)]


def test_bulk_load_int_keys(db):
    ds = db.create_dataset("v", n="int64")
    bt = db.create_btree("bt", ds, int_keys=True)
    bt.bulk_load([(k, {"n": k}) for k in [3, 1, 2, 20, 10]])
    assert list(bt.keys()) == [1, 2, 3, 10, 20]


def test_persists():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "i.db")
    with DB(path) as db:
        ds = db.create_dataset("v", n="int64")
        bt = db.create_btree("bt", ds, int_keys=True)
        for k in [10, 2, 30, -5]:
            bt[k] = {"n": k}
    with DB(path) as db:
        bt = db["bt"]
        assert bt._int_keys is True
        assert list(bt.keys()) == [-5, 2, 10, 30]
        assert bt[-5] == {"n": -5}
