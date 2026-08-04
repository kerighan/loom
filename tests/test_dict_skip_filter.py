"""In-RAM per-table skip filter for Dict (loom/datastructures/dict.py).

It speeds the multi-table existence check by skipping tables that can't hold a
key. Correctness rule: it may only produce **false positives** (an occasional
un-skipped table), never a false negative — otherwise a present key would be
missed (duplicate insert / lost update). These tests pin that guarantee across
inserts, upserts, deletes, reopen (lazy rebuild) and the on-disk format.
"""

import os
import tempfile

from pydantic import BaseModel

from loom import DB
from loom.schema import Utf8


class Rec(BaseModel):
    id: Utf8(24)
    v: int


def _fresh(path):
    if os.path.exists(path):
        os.remove(path)


class TestSkipFilterCorrectness:
    def test_no_false_negatives_across_many_tables(self):
        # Enough rows to force several exponential tables (P_INIT=10 → 1024,
        # 2048, …) so the multi-table existence check — and the filter skip —
        # actually kicks in.
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = db.collection("c", Rec, indexes={"id": "primary"})
                n = 50_000
                col.insert_many([{"id": f"k{i:08d}", "v": i} for i in range(n)])
                dk = col._primary
                assert dk.use_bloom and len(dk._blooms) >= 4  # several tables
                assert len(col) == n
                # every inserted key must be found (no false negative)
                for i in range(0, n, 91):
                    assert col[f"k{i:08d}"]["v"] == i
                # keys never inserted must not be found
                for i in range(n, n + 500):
                    assert f"k{i:08d}" not in col

    def test_upsert_and_delete_stay_correct(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = db.collection("c", Rec, indexes={"id": "primary"})
                col.insert_many([{"id": f"k{i:05d}", "v": i} for i in range(5000)])
                col.insert_many([{"id": f"k{i:05d}", "v": i * 10} for i in range(2000)])
                assert len(col) == 5000                      # upsert, not dup
                assert col["k01000"]["v"] == 10000
                col.delete("k01000")
                assert "k01000" not in col and len(col) == 4999
                # a fresh insert of the just-deleted key works (filter is a
                # false-positive-only structure: the stale bit never blocks it)
                col.insert({"id": "k01000", "v": 7})
                assert col["k01000"]["v"] == 7 and len(col) == 5000

    def test_reopen_lazy_rebuild_is_correct(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            with DB(path) as db:
                col = db.collection("c", Rec, indexes={"id": "primary"})
                col.insert_many([{"id": f"k{i:06d}", "v": i} for i in range(20000)])
            # reopen: filters are not persisted → rebuilt on first write
            with DB(path) as db:
                col = db.collection("c")
                dk = col._primary
                assert dk._blooms == []                      # not built until a write
                # a new key whose absence must be seen correctly after rebuild
                col.insert({"id": "k999999", "v": -1})
                assert dk._blooms and dk.use_bloom           # rebuilt on that write
                assert len(col) == 20001
                assert col["k010000"]["v"] == 10000          # old key still found
                assert col["k999999"]["v"] == -1
                # re-inserting an existing key updates (no dup) — proves the
                # rebuilt filter didn't hide the existing key
                col.insert({"id": "k010000", "v": 42})
                assert len(col) == 20001 and col["k010000"]["v"] == 42

    def test_on_disk_format_unchanged(self):
        # The filter is in-RAM: a Dict written by this code must carry no bloom
        # structures / bloom metadata on disk (full back-compat).
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            with DB(path) as db:
                col = db.collection("c", Rec, indexes={"id": "primary"})
                col.insert_many([{"id": f"k{i:05d}", "v": i} for i in range(3000)])
                meta = db._db.get_header_field("_ds_c__primary_metadata")
                assert "bloom_names" not in meta and "bloom_name" not in meta
                # no bloom datasets were created either
                assert not any("bloom" in n for n in db._datasets)


class TestDeletionSafety:
    def test_delete_reinsert_churn_never_false_negative(self):
        # A non-counting filter with a no-op remove can only accumulate false
        # POSITIVES (extra scans), never a false negative. This churn would
        # fail loudly if a delete ever cleared a bit a live key needs.
        import random
        rng = random.Random(0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            with DB(path) as db:
                col = db.collection("c", Rec, indexes={"id": "primary"})
                col.insert_many([{"id": f"k{i:06d}", "v": i} for i in range(8000)])
                live = set(range(8000))
                for _ in range(8000):
                    i = rng.randint(0, 12000)
                    k = f"k{i:06d}"
                    if i in live and rng.random() < 0.5:
                        col.delete(k); live.discard(i)
                    else:
                        col.insert({"id": k, "v": i}); live.add(i)
                assert len(col) == len(live)
                assert all(f"k{i:06d}" in col for i in live)          # no false neg
                gone = set(range(12001)) - live
                assert not any(f"k{i:06d}" in col for i in list(gone)[:2000])
            # reopen → rebuild from valid slots only: deleted keys stay gone
            with DB(path) as db:
                col = db.collection("c")
                assert len(col) == len(live)
                col.insert({"id": "k000000", "v": 1})   # force lazy rebuild
                assert all(f"k{i:06d}" in col for i in list(live)[:2000])


class TestVectorisedRebuildEquivalence:
    def test_add_many_sets_identical_bits_to_per_key_add(self):
        # The numpy bulk path used by the rebuild must set exactly the bits the
        # scalar add() sets — otherwise a reopened Dict's filter would diverge.
        import numpy as np
        from loom.datastructures.dict import _HashSkipFilter

        rng = np.random.default_rng(0)
        his = rng.integers(0, 2**64, size=20000, dtype=np.uint64)
        los = rng.integers(0, 2**64, size=20000, dtype=np.uint64)

        scalar = _HashSkipFilter(20000)
        for hi, lo in zip(his.tolist(), los.tolist()):
            scalar.add((hi, lo))

        bulk = _HashSkipFilter(20000)
        bulk.add_many(his, los)

        assert scalar._bits == bulk._bits
        # and every key is a member of the bulk-built filter (no false negative)
        assert all((int(hi), int(lo)) in bulk for hi, lo in zip(his[:1000], los[:1000]))


class TestReadOnlyOpenSkipsRebuild:
    def test_read_only_open_does_not_build_filters(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            with DB(path) as db:
                col = db.collection("c", Rec, indexes={"id": "primary"})
                col.insert_many([{"id": f"k{i:05d}", "v": i} for i in range(4000)])
            with DB(path) as db:
                col = db.collection("c")
                # pure reads: filters must stay unbuilt (no O(n) rebuild scan)
                assert col["k00042"]["v"] == 42
                assert "k99999" not in col
                assert col._primary._blooms == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
