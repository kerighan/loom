"""Data-integrity guards: updating or re-inserting a key must never silently
drop data that iteration / traversal depends on.

Regression for a real bug: a non-nested Dict with store_key=True (the default)
re-wrote the value record on update WITHOUT the stored `_key`, zeroing it — so
keys()/items() returned an empty string for any key that had been updated.
That surfaced as silent edge loss in Graph.add_edges whenever a (src, dst) pair
appeared more than once (very common in real knowledge graphs), because the
duplicate became an in-place update.

These tests verify the full key set survives updates, duplicate edges, and a
close/reopen — across Dict and Graph.
"""

import os
import random
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


# ── Dict: key recovery survives updates ─────────────────────────────────────


class TestDictUpdateKeyRecovery:
    def test_keys_intact_after_update(self, db):
        ds = db.create_dataset("u", id="uint32", name="U30")
        d = db.create_dict("d", ds)
        d["alice"] = {"id": 1, "name": "A"}
        d["bob"] = {"id": 2, "name": "B"}
        d["alice"] = {"id": 9, "name": "AA"}            # update
        assert sorted(d.keys()) == ["alice", "bob"]      # not ['', 'bob']
        assert "" not in set(d.keys())
        assert d["alice"] == {"id": 9, "name": "AA"}

    def test_items_intact_after_update(self, db):
        ds = db.create_dataset("u", id="uint32")
        d = db.create_dict("d", ds)
        for i in range(20):
            d[f"k{i}"] = {"id": i}
        for i in range(0, 20, 3):                        # update every 3rd
            d[f"k{i}"] = {"id": i + 1000}
        items = dict(d.items())
        assert "" not in items
        assert set(items.keys()) == {f"k{i}" for i in range(20)}
        assert items["k0"]["id"] == 1000

    def test_repeated_updates_no_key_loss(self, db):
        ds = db.create_dataset("u", val="int64")
        d = db.create_dict("d", ds)
        keys = [f"key_{i}" for i in range(200)]
        for k in keys:
            d[k] = {"val": 0}
        rng = random.Random(0)
        for _ in range(2000):                            # hammer with updates
            k = rng.choice(keys)
            d[k] = {"val": rng.randint(1, 1_000_000)}
        recovered = set(d.keys())
        assert "" not in recovered
        assert recovered == set(keys)
        assert len(list(d.keys())) == len(keys)

    def test_update_survives_reopen(self, db):
        ds = db.create_dataset("u", val="int64")
        d = db.create_dict("d", ds)
        d["x"] = {"val": 1}
        d["x"] = {"val": 2}                              # update
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            d2 = db2._datastructures["d"]
            assert list(d2.keys()) == ["x"]
            assert d2["x"] == {"val": 2}
        finally:
            db2.close()


# ── Graph: no edge loss through bulk insert / duplicates ────────────────────


class TestGraphNoEdgeLoss:
    def test_duplicate_edges_keep_neighbor(self, db):
        g = db.create_graph("g", {"n": "U10"}, {"w": "float32"}, directed=True)
        g.add_edges([
            ("a", "b", {"w": 1.0}),
            ("a", "b", {"w": 2.0}),     # duplicate (src, dst) → update
            ("a", "c", {"w": 3.0}),
        ])
        assert sorted(g.neighbors("a")) == ["b", "c"]    # not ['', 'b', 'c']
        assert "" not in set(g.neighbors("a"))
        assert g.get_edge("a", "b")["w"] == pytest.approx(2.0)

    def test_bulk_add_edges_full_roundtrip(self, db):
        """Every inserted edge must be recoverable via neighbors() — no key
        is dropped, even with many duplicates and hub growth."""
        g = db.create_graph("g", {"n": "U10"}, {"w": "float32"}, directed=True)
        rng = random.Random(1)
        expected = {}
        edges = []
        for _ in range(20000):
            s = f"n{rng.randint(0, 800)}"
            d = f"n{rng.randint(0, 800)}"
            edges.append((s, d, {"w": 1.0}))
            expected.setdefault(s, set()).add(d)
        g.add_edges(edges)

        for s, dsts in expected.items():
            got = set(g.neighbors(s))
            assert "" not in got, f"empty neighbor under {s}"
            assert got == dsts, f"neighbors mismatch for {s}"

    def test_no_empty_neighbor_after_reopen(self, db):
        g = db.create_graph("g", {"n": "U10"}, {"w": "float32"}, directed=True)
        rng = random.Random(2)
        expected = {}
        edges = []
        for _ in range(10000):
            s = f"n{rng.randint(0, 300)}"
            d = f"n{rng.randint(0, 300)}"
            edges.append((s, d, {"w": 1.0}))
            expected.setdefault(s, set()).add(d)
        g.add_edges(edges)
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            g2 = db2._datastructures["g"]
            for s, dsts in expected.items():
                got = set(g2.neighbors(s))
                assert "" not in got
                assert got == dsts
        finally:
            db2.close()
