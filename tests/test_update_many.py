"""Collection.update_many — batched, index-amortised, no-op-preserving updates.

Same result as calling update(pk, **changes) on each row, but the ordered
indexes do all their entry moves in one deferred-write block. These tests pin
functional parity against one-by-one update() and check the no-op short-circuit,
both call forms, and the error/edge cases.
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from loom import DB, Many, Unique, Search


def _build(path):
    db = DB(path)
    col = db.collection(
        "v",
        {"id": "utf8[16]", "status": "utf8[12]", "grp": "utf8[8]",
         "ts": "datetime", "cost": "float64", "email": "utf8[24]",
         "body": "text"},
        indexes={"id": "primary",
                 "status": Many(sort="ts", counted=True),
                 "grp": Many(counted=True),
                 "email": Unique(),
                 "ft": Search(fields=["body"])},
    )
    base = datetime(2026, 1, 1)
    col.insert_many([
        {"id": f"r{i}", "status": "processing", "grp": "A" if i % 2 else "B",
         "ts": base + timedelta(seconds=i), "cost": 1.0 * i,
         "email": f"e{i}@x.com", "body": f"doc number {i}"}
        for i in range(60)
    ])
    db.flush()
    return db, col


@pytest.fixture
def two_cols():
    d = tempfile.TemporaryDirectory()
    dbA, colA = _build(os.path.join(d.name, "a.loom"))   # reference: update()
    dbB, colB = _build(os.path.join(d.name, "b.loom"))   # test: update_many()
    yield colA, colB
    dbA.close()
    dbB.close()
    d.cleanup()


def _assert_same_state(colA, colB, ids):
    for pk in ids:
        assert dict(colA[pk]) == dict(colB[pk]), pk
    assert ([r["id"] for r in colA.find("status", "completed")]
            == [r["id"] for r in colB.find("status", "completed")])
    assert colA.count("status", "completed") == colB.count("status", "completed")
    assert dict(colA.groups("grp")) == dict(colB.groups("grp"))


class TestParity:
    def test_form1_same_change(self, two_cols):
        colA, colB = two_cols
        ids = [f"r{i}" for i in range(60)]
        for pk in ids:
            colA.update(pk, status="completed")
        n = colB.update_many(ids, status="completed")
        assert n == 60
        _assert_same_state(colA, colB, ids)

    def test_form2_dict_per_row(self, two_cols):
        colA, colB = two_cols
        per = {f"r{i}": {"cost": 100.0 + i, "grp": "C"} for i in range(0, 60, 2)}
        for pk, c in per.items():
            colA.update(pk, **c)
        n = colB.update_many(per)
        assert n == 30
        _assert_same_state(colA, colB, [f"r{i}" for i in range(60)])

    def test_form2_pairs(self, two_cols):
        colA, colB = two_cols
        pairs = [(f"r{i}", {"status": "completed"}) for i in range(10)]
        for pk, c in pairs:
            colA.update(pk, **c)
        assert colB.update_many(pairs) == 10
        _assert_same_state(colA, colB, [f"r{i}" for i in range(60)])

    def test_sort_only_change_moves_within_group(self, two_cols):
        colA, colB = two_cols
        base = datetime(2026, 1, 1)
        per = {f"r{i}": {"ts": base + timedelta(days=i)} for i in range(0, 60, 2)}
        for pk, c in per.items():
            colA.update(pk, **c)
        colB.update_many(per)
        # group counts unchanged (sort-only move), order matches
        assert colA.count("status", "processing") == colB.count("status", "processing")
        assert ([r["id"] for r in colA.find("status", "processing")]
                == [r["id"] for r in colB.find("status", "processing")])

    def test_search_reindex(self, two_cols):
        colA, colB = two_cols
        per = {f"r{i}": {"body": f"rewritten alpha {i}"} for i in range(5)}
        for pk, c in per.items():
            colA.update(pk, **c)
        colB.update_many(per)
        a = sorted(r["id"] for r in colA.search("ft", "alpha"))
        b = sorted(r["id"] for r in colB.search("ft", "alpha"))
        assert a == b == [f"r{i}" for i in range(5)]


class TestNoOp:
    def test_reapplying_current_value_changes_nothing(self, two_cols):
        _, colB = two_cols
        ids = [f"r{i}" for i in range(60)]
        colB.update_many(ids, status="completed")
        assert colB.update_many(ids, status="completed") == 0   # all no-ops

    def test_partial_noop_counts_only_real_changes(self, two_cols):
        _, colB = two_cols
        # r0..r9 already "processing"; set half to processing (no-op), half done
        per = {}
        for i in range(10):
            per[f"r{i}"] = {"status": "processing" if i < 5 else "done"}
        assert colB.update_many(per) == 5

    def test_unchanged_blob_not_rewritten(self, two_cols):
        _, colB = two_cols
        # changing only status must not touch the body blob reference
        before = colB._primary.get_fields("r0", ["body"])["body"]
        colB.update_many(["r0"], status="done")
        after = colB._primary.get_fields("r0", ["body"])["body"]
        assert before == after


class TestErrorsAndForms:
    def test_missing_pk_raises_before_writes(self, two_cols):
        _, colB = two_cols
        with pytest.raises(KeyError):
            colB.update_many(["r0", "nope"], status="done")
        # r0 must be untouched (validation happens before any write)
        assert colB["r0"]["status"] == "processing"

    def test_cannot_change_primary_key(self, two_cols):
        _, colB = two_cols
        with pytest.raises(ValueError):
            colB.update_many({"r0": {"id": "zzz"}})

    def test_empty_inputs(self, two_cols):
        _, colB = two_cols
        assert colB.update_many([], status="x") == 0
        assert colB.update_many({}) == 0
        assert colB.update_many(None) == 0

    def test_unique_conflict_raises(self, two_cols):
        _, colB = two_cols
        with pytest.raises(ValueError):
            colB.update_many({"r0": {"email": "e1@x.com"}})   # already r1's


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
