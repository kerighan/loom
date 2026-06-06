"""Tests for the vector similarity indexes (FlatIndex / IVFIndex).

Covers exact-search correctness across all three metrics, top-k ordering,
update/remove semantics, persistence across reopen, IVF training guards,
approximate recall (nprobe=K → exact), Product Quantization, and two
regressions found while writing these tests:

  * ``_topk`` raised ``argpartition kth out of bounds`` for the
    lower-is-better branch when ``k == len(scores)`` (any l2 search where a
    probed cell — or the whole index — held ``<= k`` vectors).
  * l2 search returned the *farthest* vectors instead of the nearest,
    because the ``higher_is_better`` flag was set to ``False`` for l2 even
    though ``_scores`` already negates the distance (so higher is always
    better).
"""

import os
import tempfile

import numpy as np
import pytest

from loom.database import DB
from loom.datastructures.vector_index import FlatIndex, IVFIndex


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


def _orthobasis(dim):
    """Identity rows — mutually orthogonal unit vectors, unambiguous NN."""
    return np.eye(dim, dtype=np.float32)


def _brute_force_top1(data, query, metric):
    """Reference nearest-neighbour index for a single query."""
    if metric == "cosine":
        d = data / np.linalg.norm(data, axis=1, keepdims=True)
        q = query / np.linalg.norm(query)
        return int(np.argmax(d @ q))
    if metric == "dot":
        return int(np.argmax(data @ query))
    # l2
    return int(np.argmin(np.sum((data - query) ** 2, axis=1)))


# ── FlatIndex ───────────────────────────────────────────────────────────────


class TestFlatIndexBasic:
    def test_add_and_len(self, db):
        idx = db.create_flat_index("f", dim=4)
        assert len(idx) == 0
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
        assert len(idx) == 2

    def test_search_returns_nearest(self, db):
        idx = db.create_flat_index("f", dim=4, metric="cosine")
        basis = _orthobasis(4)
        for i in range(4):
            idx.add(f"v{i}", basis[i])
        res = idx.search(basis[2], k=1)
        assert len(res) == 1
        assert res[0][0] == "v2"
        assert res[0][1] == pytest.approx(1.0)

    def test_search_topk_order_descending(self, db):
        idx = db.create_flat_index("f", dim=3, metric="cosine")
        idx.add("near", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        idx.add("mid", np.array([1.0, 1.0, 0.0], dtype=np.float32))
        idx.add("far", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        res = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3)
        ids = [r[0] for r in res]
        scores = [r[1] for r in res]
        assert ids == ["near", "mid", "far"]
        assert scores == sorted(scores, reverse=True)

    def test_empty_search_returns_empty(self, db):
        idx = db.create_flat_index("f", dim=4)
        assert idx.search(np.ones(4, dtype=np.float32), k=5) == []

    def test_k_larger_than_n(self, db):
        idx = db.create_flat_index("f", dim=4)
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
        res = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=10)
        assert len(res) == 2

    # Only cosine/l2 guarantee the identical vector ranks first; for the raw
    # dot product a higher-magnitude aligned vector can outscore it.
    @pytest.mark.parametrize("metric", ["cosine", "l2"])
    def test_metric_finds_identical_vector(self, db, metric):
        idx = db.create_flat_index("f", dim=8, metric=metric)
        rng = np.random.RandomState(1)
        data = rng.randn(50, 8).astype(np.float32)
        idx.add_batch([(f"v{i}", data[i]) for i in range(50)])
        for probe in (0, 17, 49):
            res = idx.search(data[probe], k=1)
            assert res[0][0] == f"v{probe}", metric

    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
    def test_matches_brute_force_top1(self, db, metric):
        idx = db.create_flat_index("f", dim=8, metric=metric)
        rng = np.random.RandomState(2)
        data = rng.randn(80, 8).astype(np.float32)
        idx.add_batch([(f"v{i}", data[i]) for i in range(80)])
        qrng = np.random.RandomState(99)
        for _ in range(10):
            q = qrng.randn(8).astype(np.float32)
            expected = f"v{_brute_force_top1(data, q, metric)}"
            assert idx.search(q, k=1)[0][0] == expected, metric

    def test_update_existing_id_keeps_len(self, db):
        idx = db.create_flat_index("f", dim=4, metric="cosine")
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        assert len(idx) == 1
        idx.add("a", np.array([0, 1, 0, 0], dtype=np.float32))  # overwrite
        assert len(idx) == 1
        res = idx.search(np.array([0, 1, 0, 0], dtype=np.float32), k=1)
        assert res[0][0] == "a"
        assert res[0][1] == pytest.approx(1.0)

    def test_remove(self, db):
        idx = db.create_flat_index("f", dim=4, metric="cosine")
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
        idx.remove("a")
        assert len(idx) == 1
        ids = [r[0] for r in idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=5)]
        assert "a" not in ids
        assert "b" in ids

    def test_remove_missing_raises(self, db):
        idx = db.create_flat_index("f", dim=4)
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        with pytest.raises(KeyError):
            idx.remove("ghost")

    def test_persistence_reopen(self, db):
        idx = db.create_flat_index("f", dim=8, metric="cosine")
        rng = np.random.RandomState(7)
        data = rng.randn(40, 8).astype(np.float32)
        idx.add_batch([(f"v{i}", data[i]) for i in range(40)])
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            idx2 = db2._datastructures["f"]
            assert isinstance(idx2, FlatIndex)
            assert len(idx2) == 40
            res = idx2.search(data[10], k=1)
            assert res[0][0] == "v10"
        finally:
            db2.close()


# ── IVFIndex — training guards ────────────────────────────────────────────


class TestIVFTrainingGuards:
    def test_add_before_train_raises(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=4)
        with pytest.raises(RuntimeError):
            ivf.add("x", np.ones(8, dtype=np.float32))

    def test_search_before_train_raises(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=4)
        with pytest.raises(RuntimeError):
            ivf.search(np.ones(8, dtype=np.float32), k=5)

    def test_is_trained_flag(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=4)
        assert ivf.is_trained is False
        rng = np.random.RandomState(0)
        ivf.train(rng.randn(60, 8).astype(np.float32))
        assert ivf.is_trained is True


# ── IVFIndex — search correctness ──────────────────────────────────────────


class TestIVFSearch:
    @pytest.mark.parametrize("metric", ["cosine", "l2"])
    def test_full_nprobe_finds_identical(self, db, metric):
        """nprobe == n_clusters probes every cell → exact for the query itself.

        dot is excluded: the identical vector is not guaranteed to maximise
        the raw inner product (a longer aligned vector can score higher).
        """
        ivf = db.create_ivf_index("v", dim=8, n_clusters=8, metric=metric)
        rng = np.random.RandomState(3)
        data = rng.randn(200, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(200)])
        for probe in (0, 100, 199):
            res = ivf.search(data[probe], k=1, nprobe=8)
            assert res[0][0] == f"v{probe}", metric

    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
    def test_full_nprobe_matches_brute_force(self, db, metric):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=8, metric=metric)
        rng = np.random.RandomState(4)
        data = rng.randn(200, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(200)])
        qrng = np.random.RandomState(123)
        for _ in range(8):
            q = qrng.randn(8).astype(np.float32)
            expected = f"v{_brute_force_top1(data, q, metric)}"
            assert ivf.search(q, k=1, nprobe=8)[0][0] == expected, metric

    def test_topk_order_descending(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=8, metric="cosine")
        rng = np.random.RandomState(5)
        data = rng.randn(200, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(200)])
        res = ivf.search(data[0], k=10, nprobe=8)
        scores = [s for _, s in res]
        assert scores == sorted(scores, reverse=True)

    def test_empty_search(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=4)
        ivf.train(np.random.RandomState(0).randn(50, 8).astype(np.float32))
        assert ivf.search(np.ones(8, dtype=np.float32), k=5) == []

    def test_k_larger_than_n(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=8, metric="cosine")
        rng = np.random.RandomState(6)
        data = rng.randn(50, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(50)])
        res = ivf.search(data[0], k=100, nprobe=8)
        assert len(res) <= 50
        assert res[0][0] == "v0"

    def test_default_nprobe_runs(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=16, metric="cosine")
        rng = np.random.RandomState(8)
        data = rng.randn(300, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(300)])
        res = ivf.search(data[0], k=5)  # nprobe defaults to ~sqrt(K)
        assert 1 <= len(res) <= 5

    def test_remove(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=8, metric="cosine")
        rng = np.random.RandomState(9)
        data = rng.randn(100, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(100)])
        assert len(ivf) == 100
        ivf.remove("v0")
        assert len(ivf) == 99
        ids = [r[0] for r in ivf.search(data[0], k=10, nprobe=8)]
        assert "v0" not in ids

    def test_persistence_reopen_no_retrain(self, db):
        ivf = db.create_ivf_index("v", dim=8, n_clusters=8, metric="cosine")
        rng = np.random.RandomState(10)
        data = rng.randn(200, 8).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(200)])
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            ivf2 = db2._datastructures["v"]
            assert isinstance(ivf2, IVFIndex)
            assert ivf2.is_trained is True
            assert len(ivf2) == 200
            res = ivf2.search(data[42], k=1, nprobe=8)  # no retrain
            assert res[0][0] == "v42"
        finally:
            db2.close()


# ── IVFIndex — Product Quantization ──────────────────────────────────────


class TestIVFProductQuantization:
    def test_pq_search_returns_valid_ids(self, db):
        ivf = db.create_ivf_index(
            "v", dim=16, n_clusters=8, metric="l2", pq=True, n_sub=4
        )
        rng = np.random.RandomState(11)
        data = rng.randn(300, 16).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(300)])
        res = ivf.search(data[0], k=10, nprobe=8)
        assert len(res) == 10
        valid = {f"v{i}" for i in range(300)}
        assert all(vid in valid for vid, _ in res)

    def test_pq_persistence_reopen(self, db):
        ivf = db.create_ivf_index(
            "v", dim=16, n_clusters=8, metric="l2", pq=True, n_sub=4
        )
        rng = np.random.RandomState(12)
        data = rng.randn(300, 16).astype(np.float32)
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(300)])
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            ivf2 = db2._datastructures["v"]
            assert ivf2.pq is True
            assert ivf2.is_trained is True
            assert len(ivf2) == 300
            res = ivf2.search(data[0], k=5, nprobe=8)
            assert len(res) == 5
        finally:
            db2.close()


# ── Regressions ──────────────────────────────────────────────────────────


class TestVectorRegressions:
    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
    def test_search_returns_nearest_not_farthest(self, db, metric):
        """l2 used to invert the ranking and return the farthest vectors."""
        idx = db.create_flat_index("f", dim=4, metric=metric)
        idx.add("identical", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        idx.add("opposite", np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        idx.add("orthogonal", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        res = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=3)
        assert res[0][0] == "identical", metric

    def test_flat_l2_k_greater_than_n_no_crash(self, db):
        """Regression: _topk(argpartition kth=len) out of bounds for l2."""
        idx = db.create_flat_index("f", dim=4, metric="l2")
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
        idx.add("c", np.array([0, 0, 1, 0], dtype=np.float32))
        res = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=5)  # k > n
        assert res[0][0] == "a"
        assert len(res) == 3

    def test_ivf_l2_small_cells_no_crash(self, db):
        """Regression: per-cell l2 _topk crashed when a cell held <= k vectors."""
        ivf = db.create_ivf_index("v", dim=4, n_clusters=8, metric="l2")
        rng = np.random.RandomState(13)
        data = rng.randn(20, 4).astype(np.float32)  # ~2-3 vectors per cell
        ivf.train(data)
        ivf.add_batch([(f"v{i}", data[i]) for i in range(20)])
        res = ivf.search(data[0], k=5, nprobe=8)
        assert res[0][0] == "v0"
