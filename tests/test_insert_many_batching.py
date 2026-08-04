"""insert_many into a non-empty collection must be correct regardless of how
the rows are chunked, and (perf) must not pay the old ~2x per-row penalty.

The write path for a non-empty BTree index defers node writes (a leaf touched K
times is serialised once, not K times) — so the *results* must be byte-for-byte
independent of the batch split, across every index kind.
"""

import os
import tempfile
from datetime import datetime, timedelta

from pydantic import BaseModel

from loom import DB, Many
from loom.schema import Utf8


class Ref(BaseModel):
    id: Utf8(32)
    run_id: Utf8(24)
    parent_id: Utf8(24)
    group: Utf8(32)
    collection: Utf8(32)
    position: int
    timestamp: datetime


INDEXES = {
    "id": "primary",
    "run_id": Many(sort="position"),
    "parent_id": Many(sort="timestamp", desc=True),
    "group": Many(sort="timestamp", desc=True, counted=True),
    "timestamp": "range",
}

T0 = datetime(2026, 1, 1)


def _rows(n):
    return [{
        "id": f"run{i // 14:06d}:{i % 14:03d}",
        "run_id": f"run{i // 14:06d}",
        "parent_id": f"prompt{i // 70:05d}",
        "group": f"g{i % 3}",
        "collection": f"theme{i % 7}",
        "position": i % 14,
        "timestamp": T0 + timedelta(seconds=i),
    } for i in range(n)]


def _snapshot(col):
    """A batching-invariant fingerprint of every index."""
    groups = col.groups("group")
    finds = {g: [r["id"] for r in col.find("group", g)] for g, _ in groups}
    runs = {rid: [r["id"] for r in col.find("run_id", rid)]
            for rid in {r["run_id"] for r in col.values()}}
    rng = [r["id"] for r in col.range("timestamp", T0, T0 + timedelta(seconds=500))]
    counts = {g: col.count("group", g) for g, _ in groups}
    return {
        "len": len(col),
        "groups": groups,
        "finds": finds,
        "runs": runs,
        "range": rng,
        "counts": counts,
    }


def _build(path, rows, n_batches):
    if os.path.exists(path):
        os.remove(path)
    size = (len(rows) + n_batches - 1) // n_batches
    with DB(path) as db:
        col = db.collection("refs", Ref, indexes=INDEXES)
        for i in range(0, len(rows), size):
            col.insert_many(rows[i:i + size])
        return _snapshot(col)


class TestBatchingInvariance:
    def test_same_result_1_vs_10_vs_200_batches(self):
        rows = _rows(4000)   # >> ORDER, so every index splits many times
        with tempfile.TemporaryDirectory() as d:
            base = _build(os.path.join(d, "b1.loom"), rows, 1)
            for nb in (2, 10, 200):
                got = _build(os.path.join(d, f"b{nb}.loom"), rows, nb)
                assert got == base, f"{nb} batches diverged from 1"

    def test_survives_reopen(self):
        rows = _rows(2000)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.loom")
            live = _build(path, rows, 20)
            with DB(path) as db:
                assert _snapshot(db.collection("refs")) == live

    def test_upsert_across_batches(self):
        rows = _rows(500)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "u.loom")
            with DB(path) as db:
                col = db.collection("refs", Ref, indexes=INDEXES)
                col.insert_many(rows)                      # first pass
                # re-insert an overlapping slice with a changed indexed field
                changed = [{**r, "group": "gX"} for r in rows[:100]]
                col.insert_many(changed)
                assert len(col) == 500                     # upsert, no dup rows
                assert col.count("group", "gX") == 100
                # the 100 moved rows left their old groups
                assert sum(c for _, c in col.groups("group")) == 500
                assert col["run000000:000"]["group"] == "gX"


class TestDeferredNodeWrites:
    """The BTree deferred-write buffer itself: correct under heavy splitting."""

    def test_bulk_insert_into_nonempty_tree(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "t.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(100):
                    bt[f"{i:04d}"] = {"v": i}          # seed (non-empty)
                extra = [(f"{i:04d}", {"v": i}) for i in range(100, 1000)]
                extra_shuffled = extra[300:] + extra[:300]  # not pre-sorted
                with bt.deferred_node_writes():
                    for k, v in extra_shuffled:
                        bt[k] = v
                assert len(bt) == 1000
                assert [k for k in bt.keys()] == [f"{i:04d}" for i in range(1000)]
                assert bt["0777"] == {"v": 777}
                # range still correct after the buffered splits
                assert [v["v"] for _, v in bt.range("0010", "0015")] == [10, 11, 12, 13, 14, 15]

    def test_deferred_updates_existing_keys(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "t.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(200):
                    bt[f"{i:04d}"] = {"v": i}
                with bt.deferred_node_writes():
                    for i in range(0, 200, 2):
                        bt[f"{i:04d}"] = {"v": i * 1000}   # update, no size change
                assert len(bt) == 200
                assert bt["0100"] == {"v": 100000}
                assert bt["0101"] == {"v": 101}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
