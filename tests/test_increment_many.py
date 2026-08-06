"""Collection.increment_many — batched counter bumps.

Same result as calling increment() per key, but the index-entry moves for a
sorted (range/Many) counter are applied in one deferred-write block per index
(~2x cheaper once non-empty) and only the counter field is rewritten in place.
"""

import os
import random
import tempfile

from pydantic import BaseModel

from loom import DB, Many
from loom.schema import Utf8


class Agg(BaseModel):
    id: Utf8(64)
    grp: Utf8(16)
    n: int


def _make(db):
    return db.collection("agg", Agg, indexes={
        "id": "primary",
        "grp": Many(sort="n", desc=True),
    })


def _seed(col, n=4000):
    col.insert_many([{"id": f"k{i:05d}", "grp": f"g{i % 5}", "n": 1}
                     for i in range(n)])


class TestIncrementMany:
    def test_matches_per_key_increment(self):
        bumps = [f"k{i:05d}" for i in range(2000)]
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                a = _make(db)
                _seed(a)
                res = a.increment_many([(k, 1) for k in bumps], "n")
                top_a = [(r["id"], r["n"]) for r in a.find("grp", "g2", limit=20)]
                len_a = len(a)
            with DB(os.path.join(d, "b.loom")) as db:
                b = db.collection("agg", Agg, indexes={
                    "id": "primary", "grp": Many(sort="n", desc=True)})
                _seed(b)
                for k in bumps:
                    b.increment(k, "n", 1)
                top_b = [(r["id"], r["n"]) for r in b.find("grp", "g2", limit=20)]
            assert top_a == top_b
            assert len_a == len(b)
            assert all(res[k] == 2 for k in bumps)

    def test_input_forms_and_coalescing(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = _make(db)
                _seed(col, 100)
                assert col.increment_many({}, "n") == {}
                assert col.increment_many({"k00000": 5}, "n")["k00000"] == 6
                # duplicate pk in an iterable → summed
                r = col.increment_many([("k00001", 3), ("k00001", 2)], "n")
                assert r["k00001"] == 1 + 3 + 2
                # bare pks use the default amount
                r = col.increment_many(["k00002", "k00003"], "n", amount=4)
                assert r == {"k00002": 5, "k00003": 5}
                assert col["k00002"]["n"] == 5

    def test_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = _make(db)
                _seed(col, 50)
                try:
                    col.increment_many([("nope", 1)], "n")
                    assert False, "expected KeyError"
                except KeyError:
                    pass

    def test_non_indexed_field(self):
        class Doc(BaseModel):
            id: Utf8(16)
            views: int
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = db.collection("d", Doc, indexes={"id": "primary"})
                col.insert_many([{"id": f"k{i}", "views": 0} for i in range(100)])
                col.increment_many({f"k{i}": i for i in range(100)}, "views")
                assert col["k42"]["views"] == 42

    def test_counted_index_group_bump(self):
        # Incrementing the GROUP field itself must move the maintained counter.
        class Rec(BaseModel):
            id: Utf8(16)
            g: int
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = db.collection("r", Rec, indexes={
                    "id": "primary", "g": Many(sort="id", counted=True)})
                col.insert_many([{"id": f"k{i:03d}", "g": 0} for i in range(300)])
                assert col.count("g", 0) == 300
                col.increment_many([f"k{i:03d}" for i in range(100)], "g", amount=1)
                assert col.count("g", 0) == 200
                assert col.count("g", 1) == 100

    def test_batched_moves_bounded_growth(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = _make(db)
                _seed(col, 4000)
                db.flush()
                base = db._db.get_allocation_index()
                rng = random.Random(0)
                for _ in range(20):                       # 20 flush-sized batches
                    keys = [f"k{rng.randint(0, 4000 - 1):05d}" for _ in range(500)]
                    col.increment_many([(k, 1) for k in keys], "n")
                db.flush()
                grew = db._db.get_allocation_index() - base
                # 10k bumps: must not scale like a per-op leak (was ~390 B/op)
                assert grew < 1_000_000, grew


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
