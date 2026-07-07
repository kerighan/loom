"""Vector index on a Collection — pre-filtered exact (flat) similarity.

Not ANN: the spec is pure config (the vectors live inline in the records,
nothing is maintained on write) and nearest() narrows candidates through the
regular indexes first, then scores the survivors' vectors exactly.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from loom import DB, Many, Vector, Vec, Utf8
from pydantic import BaseModel

DIM = 16


class Doc(BaseModel):
    id:         Utf8(16)
    topic:      Utf8(16)
    created_at: datetime
    emb:        Vec(DIM)


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


def make_docs(db, n=300):
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(n, DIM)).astype(np.float32)
    col = db.collection("docs", Doc, indexes={
        "id": "primary",
        "topic": Many(sort="created_at", desc=True),
        "created_at": "range",
        "emb": Vector(metric="cosine"),
    })
    dates = [datetime(2026, 1 + i % 6, 1 + i % 27) for i in range(n)]
    col.insert_many([Doc(id=f"d{i}", topic=("pol" if i % 3 == 0 else "eco"),
                         created_at=dates[i], emb=vecs[i]) for i in range(n)])
    return col, vecs, dates


def cosine(vecs, q):
    return (vecs @ q) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q))


class TestNearest:
    def test_flat_matches_brute_force(self, db):
        col, vecs, _ = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        hits = col.nearest("emb", q, k=5, with_scores=True)
        sims = cosine(vecs, q)
        assert [h["id"] for h, _ in hits] == [f"d{i}" for i in np.argsort(-sims)[:5]]
        assert hits[0][1] == pytest.approx(float(sims.max()), abs=1e-5)

    def test_where_group_prefilters(self, db):
        col, vecs, _ = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        hits = col.nearest("emb", q, k=4, where={"topic": "pol"})
        assert all(h["topic"] == "pol" for h in hits)
        sims = cosine(vecs, q)
        expected = [f"d{i}" for i in np.argsort(-sims) if i % 3 == 0][:4]
        assert [h["id"] for h in hits] == expected

    def test_where_group_and_time_window(self, db):
        col, vecs, dates = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        lo, hi = datetime(2026, 3, 1), datetime(2026, 4, 30)
        hits = col.nearest("emb", q, k=4,
                           where={"topic": "pol", "created_at": (lo, hi)})
        assert hits and all(h["topic"] == "pol" and lo <= h["created_at"] <= hi
                            for h in hits)
        sims = cosine(vecs, q)
        expected = [f"d{i}" for i in np.argsort(-sims)
                    if i % 3 == 0 and lo <= dates[i] <= hi][:4]
        assert [h["id"] for h in hits] == expected

    def test_where_range_index(self, db):
        col, _, _ = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        hits = col.nearest("emb", q, k=4,
                           where={"created_at": (datetime(2026, 6, 1), None)})
        assert hits and all(h["created_at"] >= datetime(2026, 6, 1) for h in hits)

    def test_where_callable_and_fields(self, db):
        col, _, _ = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        hits = col.nearest("emb", q, k=3, where=lambda r: r["topic"] == "eco",
                           fields=["id", "topic"])
        assert hits and all(set(h) == {"id", "topic"} and h["topic"] == "eco"
                            for h in hits)

    def test_l2_ranks_ascending(self, db):
        col = db.collection("l2docs", Doc, indexes={
            "id": "primary", "emb": Vector(metric="l2")})
        rng = np.random.default_rng(3)
        vecs = rng.normal(size=(50, DIM)).astype(np.float32)
        col.insert_many([Doc(id=f"d{i}", topic="x",
                             created_at=datetime(2026, 1, 1), emb=vecs[i])
                         for i in range(50)])
        q = np.zeros(DIM, dtype=np.float32)
        hits = col.nearest("emb", q, k=3, with_scores=True)
        dists = np.linalg.norm(vecs - q, axis=1)
        assert [h["id"] for h, _ in hits] == [f"d{i}" for i in np.argsort(dists)[:3]]
        assert [s for _, s in hits] == sorted(s for _, s in hits)

    def test_empty_candidates(self, db):
        col, _, _ = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        assert col.nearest("emb", q, k=5, where={"topic": "nope"}) == []

    def test_k_larger_than_candidates(self, db):
        col, _, _ = make_docs(db, n=3)
        q = np.ones(DIM, dtype=np.float32)
        assert len(col.nearest("emb", q, k=50)) == 3


class TestVectorSpec:
    def test_survives_reopen_and_vacuum(self, db):
        col, vecs, _ = make_docs(db)
        q = np.ones(DIM, dtype=np.float32)
        expected = [h["id"] for h in col.nearest("emb", q, k=3)]
        path = db.filename
        db.close()
        db2 = DB(path)
        db2.open()
        col = db2.collection("docs")
        assert [h["id"] for h in col.nearest("emb", q, k=3)] == expected
        db2.vacuum()
        col = db2.collection("docs")
        assert [h["id"] for h in col.nearest("emb", q, k=3)] == expected
        db2.close()

    def test_non_vector_field_rejected(self, db):
        with pytest.raises(ValueError, match="1-D vector field"):
            db.collection("bad", {"id": "utf8[8]", "x": "int64"},
                          indexes={"id": "primary", "x": "vector"})

    def test_unknown_metric_rejected(self, db):
        with pytest.raises(ValueError, match="metric"):
            db.collection("bad2", Doc, indexes={
                "id": "primary", "emb": Vector(metric="manhattan")})

    def test_writes_dont_touch_vector_index(self, db):
        """The spec is pure config: update/delete work with no vector-side
        bookkeeping, and nearest() sees the mutations immediately."""
        col, vecs, _ = make_docs(db, n=20)
        q = np.ones(DIM, dtype=np.float32)
        top = col.nearest("emb", q, k=1)[0]["id"]
        col.delete(top)
        assert col.nearest("emb", q, k=1)[0]["id"] != top
        col.update("d5", emb=np.full(DIM, 100.0, dtype=np.float32))
        assert col.nearest("emb", q, k=1)[0]["id"] == "d5"
