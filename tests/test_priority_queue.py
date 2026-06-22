"""Tests for PriorityQueue and the order-preserving encode_value extensions."""

import os
import tempfile
from datetime import date, datetime

import pytest

from loom import DB, PriorityQueue, PriorityQueueEmpty
from loom.collection import encode_value


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "pq.db")
    database = DB(path)
    database.open()
    yield database
    database.close()


# ── encode_value ordering ─────────────────────────────────────────────────────

def test_encode_value_int_order_preserving():
    vals = [-100, -1, 0, 1, 9, 12, 100, 2**40]
    assert sorted(vals, key=encode_value) == sorted(vals)
    assert sorted(vals, key=lambda v: encode_value(v, desc=True)) == sorted(vals, reverse=True)


def test_encode_value_float_order_preserving():
    vals = [-10.5, -2.0, 0.0, 0.5, 3.5, 9.25, 12.0, 100.0]
    assert sorted(vals, key=encode_value) == sorted(vals)
    assert sorted(vals, key=lambda v: encode_value(v, desc=True)) == sorted(vals, reverse=True)


def test_encode_value_numpy_float_matches_python_float():
    np = pytest.importorskip("numpy")
    assert encode_value(np.float64(9.25)) == encode_value(9.25)
    assert encode_value(np.int64(42)) == encode_value(42)


def test_encode_value_datetime_chronological():
    vals = [datetime(2024, 1, 1), datetime(2026, 6, 1, 12, 0, 0),
            datetime(2025, 3, 1), date(2025, 12, 31)]
    keys = [encode_value(v) for v in vals]
    # keys sort in the same order as the chronological values
    order = sorted(range(len(vals)), key=lambda i: keys[i])
    expected = sorted(range(len(vals)),
                      key=lambda i: vals[i] if isinstance(vals[i], datetime)
                      else datetime(vals[i].year, vals[i].month, vals[i].day))
    assert order == expected


# ── PriorityQueue ─────────────────────────────────────────────────────────────

def test_priority_queue_max_first_and_fifo_ties(db):
    pq = db.create_priority_queue("jobs", {"task": "utf8[40]"})
    pq.push({"task": "low"}, 1)
    pq.push({"task": "high"}, 9)
    pq.push({"task": "mid"}, 5)
    pq.push({"task": "high2"}, 9)   # tie with 'high' → FIFO after it
    assert len(pq) == 4
    assert pq.peek()["task"] == "high"
    assert [pq.pop()["task"] for _ in range(4)] == ["high", "high2", "mid", "low"]


def test_priority_queue_empty_behaviour(db):
    pq = db.create_priority_queue("jobs", {"task": "utf8[40]"})
    assert pq.pop(default=None) is None
    assert pq.peek(default="none") == "none"
    with pytest.raises(PriorityQueueEmpty):
        pq.pop()


def test_priority_queue_min_first(db):
    pq = db.create_priority_queue("jobs", {"x": "int64"}, max_first=False)
    for x, p in [(0, 9), (1, 1), (2, 5)]:
        pq.push({"x": x}, p)
    assert [pq.pop()["x"] for _ in range(3)] == [1, 2, 0]


def test_priority_queue_float_priority(db):
    pq = db.create_priority_queue("jobs", {"task": "utf8[20]"})
    pq.push({"task": "a"}, 0.5)
    pq.push({"task": "b"}, 9.75)
    pq.push({"task": "c"}, 3.2)
    assert [pq.pop()["task"] for _ in range(3)] == ["b", "c", "a"]


def test_priority_queue_bulk_push(db):
    pq = db.create_priority_queue("jobs", {"x": "int64"})
    pq.push_many([({"x": i}, i) for i in range(50)])
    assert len(pq) == 50
    assert pq.pop()["x"] == 49        # highest priority first
    pq.push_many([({"x": 100}, 100)])  # non-empty path
    assert pq.pop()["x"] == 100


def test_priority_queue_reopen():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "pq.db")
    with DB(path) as db:
        pq = db.create_priority_queue("jobs", {"task": "utf8[40]"})
        pq.push({"task": "a"}, 1)
        pq.push({"task": "b"}, 5)

    with DB(path) as db:
        pq = db.create_priority_queue("jobs")  # reopen, no schema
        assert len(pq) == 2
        assert pq.peek()["task"] == "b"
        # a fresh push must not collide with persisted sequence numbers
        pq.push({"task": "c"}, 5)
        assert [pq.pop()["task"] for _ in range(3)] == ["b", "c", "a"]


def test_btree_reverse_range(db):
    ds = db.create_dataset("vals", v="int64")
    bt = db.create_btree("bt", ds, key_size=24)
    for i in range(100):
        bt[f"{i:03d}"] = {"v": i}
    # full reverse == reversed ascending
    asc = [k for k, _ in bt.range()]
    desc = [k for k, _ in bt.range(reverse=True)]
    assert desc == list(reversed(asc))
    # top-N from the high end, seeked
    top3 = [k for k, _ in bt.range(reverse=True)][:3]
    assert top3 == ["099", "098", "097"]
    # bounded reverse: keys <= "050", highest first
    bounded = [k for k, _ in bt.range(None, "050", reverse=True)][:3]
    assert bounded == ["050", "049", "048"]
    # lower bound stops the descent
    low_bounded = [k for k, _ in bt.range("095", None, reverse=True)]
    assert low_bounded == ["099", "098", "097", "096", "095"]


def test_collection_range_desc_inbox(db):
    inbox = db.collection("inbox", {"id": "utf8[16]", "created_at": "int64"},
                          indexes={"id": "primary", "created_at": "range"})
    for i in range(200):
        inbox.insert({"id": f"m{i:03d}", "created_at": i})
    # the 5 most recent — no grouping field needed
    latest = inbox.range("created_at", limit=5, desc=True)
    assert [r["created_at"] for r in latest] == [199, 198, 197, 196, 195]
    # oldest still works ascending
    assert [r["created_at"] for r in inbox.range("created_at", limit=3)] == [0, 1, 2]
    # bounded desc
    assert [r["created_at"] for r in
            inbox.range("created_at", None, 100, limit=3, desc=True)] == [100, 99, 98]


def test_priority_queue_datetime_priority(db):
    pq = db.create_priority_queue("events", {"x": "int64"})
    pq.push({"x": 1}, datetime(2024, 1, 1))
    pq.push({"x": 2}, datetime(2026, 6, 1))
    pq.push({"x": 3}, datetime(2025, 3, 1))
    assert [pq.pop()["x"] for _ in range(3)] == [2, 3, 1]   # most recent first
