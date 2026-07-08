"""Property-based fuzz harness for Collections.

Thousands of random operations (insert/upsert/insert_many/update/increment/
field-write/delete, interleaved with vacuum, migrate, reopen) run against a
Collection carrying every index kind — primary, unique, many, range, search,
vector — while a plain in-memory model (dict pk → record) tracks the expected
state. After every mutation the query surface is cross-checked against the
model: point reads, find (order included), count, range, unique get,
full-text hit sets, nearest top-k vs numpy brute force, projections.

This is the pre-prod insurance for the "works until you read it back" bug
class (str→'<U0', unknown-field defaults, stale post-vacuum handles): those
were all found by downstream apps, and all three would be caught here.

LOOM_FUZZ_OPS=20000 python -m pytest tests/test_fuzz_collection.py  # soak run
"""

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from loom import DB, Many, Search, Vector

DIM = 8
VOCAB = ("alpha bravo charlie delta echo foxtrot golf hotel india juliett "
         "kilo lima mike november oscar papa quebec romeo sierra tango").split()

SCHEMA = {
    "id":    "utf8[16]",
    "grp":   "utf8[8]",
    "score": "float64",
    "n":     "int64",
    "email": "utf8[32]",
    "ts":    "datetime",
    "body":  "text",
    "meta":  "json",
    "emb":   f"float32[{DIM}]",
}

INDEXES = {
    "id":    "primary",
    "grp":   Many(sort="score", desc=True),
    "score": "range",
    "email": "unique",
    "body":  Search(fields=["body"], scoring="bm25"),
    "emb":   Vector(metric="cosine"),
}

N_OPS = int(os.environ.get("LOOM_FUZZ_OPS", "1000"))
FULL_CHECK_EVERY = 250


def _make_record(rng, pk):
    return {
        "id": pk,
        "grp": f"g{rng.integers(0, 8)}",
        "score": float(rng.random()),
        "n": int(rng.integers(0, 100)),
        "email": f"{pk}@x",
        "ts": datetime(2026, 1, 1) + timedelta(seconds=int(rng.integers(0, 10**7))),
        "body": " ".join(rng.choice(VOCAB, size=6)),
        "meta": {"k": int(rng.integers(0, 5)), "tag": str(rng.choice(VOCAB))},
        "emb": rng.normal(size=DIM).astype(np.float32),
    }


def _assert_record(got, want, pk):
    assert got is not None, f"{pk} missing from collection"
    for f, w in want.items():
        g = got[f]
        if f == "emb":
            assert np.allclose(g, w, atol=0), f"{pk}.{f}: {g} != {w}"
        else:
            assert g == w, f"{pk}.{f}: {g!r} != {w!r}"


def _check_point_reads(col, model, rng, k=5):
    if not model:
        assert len(col) == 0
        return
    pks = list(model)
    for pk in rng.choice(pks, size=min(k, len(pks)), replace=False):
        _assert_record(col[pk], model[pk], pk)
    absent = "zz_nope"
    assert absent not in col and col.get_primary(absent) is None


def _check_find_count(col, model, rng):
    grp = f"g{rng.integers(0, 8)}"
    in_grp = [(pk, r) for pk, r in model.items() if r["grp"] == grp]
    expected = [pk for pk, r in sorted(in_grp, key=lambda x: (-x[1]["score"], x[0]))]
    got = [r["id"] for r in col.find("grp", grp)]
    assert got == expected, f"find({grp}): {got} != {expected}"
    assert col.count("grp", grp) == len(expected)
    if in_grp:
        lo = float(rng.random())
        want = sum(1 for _, r in in_grp if r["score"] >= lo)
        assert col.count("grp", grp, start=lo) == want
    # projection consistency
    for rec in col.find("grp", grp, fields=["id", "n"], limit=5):
        assert set(rec) == {"id", "n"} and rec["n"] == model[rec["id"]]["n"]


def _check_range(col, model, rng):
    lo, hi = sorted((float(rng.random()), float(rng.random())))
    expected = [pk for pk, r in sorted(model.items(), key=lambda x: (x[1]["score"], x[0]))
                if lo <= r["score"] <= hi]
    got = [r["id"] for r in col.range("score", lo, hi)]
    assert got == expected, f"range({lo:.3f},{hi:.3f}): {got} != {expected}"


def _check_unique(col, model, rng):
    if not model:
        return
    pk = str(rng.choice(list(model)))
    hit = col.get("email", f"{pk}@x")
    assert hit is not None and hit["id"] == pk
    assert col.get("email", "ghost@x") is None


def _check_search(col, model, rng):
    word = str(rng.choice(VOCAB))
    expected = {pk for pk, r in model.items() if word in r["body"].split()}
    got = {r["id"] for r in col.search("body", word)}
    assert got == expected, f"search({word}): {got ^ expected} differ"


def _check_nearest(col, model, rng, where_grp=None):
    if not model:
        return
    q = rng.normal(size=DIM).astype(np.float32)
    pool = {pk: r for pk, r in model.items()
            if where_grp is None or r["grp"] == where_grp}
    if not pool:
        assert col.nearest("emb", q, k=3, where={"grp": where_grp}) == []
        return
    pks = list(pool)
    M = np.stack([pool[pk]["emb"] for pk in pks])
    sims = (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))
    k = min(5, len(pks))
    where = None if where_grp is None else {"grp": where_grp}
    hits = col.nearest("emb", q, k=k, where=where, with_scores=True)
    assert len(hits) == k
    brute = sorted(sims, reverse=True)[:k]
    by_pk = dict(zip(pks, sims))
    for (rec, score), bscore in zip(hits, brute):
        assert score == pytest.approx(by_pk[rec["id"]], abs=1e-5)   # true score
        assert score == pytest.approx(bscore, abs=1e-4)             # is k-best


def _full_check(db, col, model, rng):
    assert len(col) == len(model)
    for pk, want in model.items():
        _assert_record(col.get_primary(pk), want, pk)
    for g in range(8):
        _check_find_count(col, model, np.random.default_rng(g))
    _check_range(col, model, rng)
    _check_search(col, model, rng)
    _check_nearest(col, model, rng)
    _check_nearest(col, model, rng, where_grp=f"g{rng.integers(0, 8)}")
    report = db.verify()
    problems = [r for r in (report or []) if r] if isinstance(report, list) else []
    assert not problems, f"db.verify(): {problems}"


@pytest.mark.parametrize("seed", [1, 2])
def test_fuzz_collection(seed):
    rng = np.random.default_rng(seed)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "fuzz.db")
    db = DB(path, header_size=131072)
    db.open()
    col = db.collection("posts", SCHEMA, indexes=INDEXES)
    model = {}
    next_id = 0

    def new_pk():
        nonlocal next_id
        next_id += 1
        return f"p{next_id}"

    def existing_pk():
        return str(rng.choice(list(model)))

    for step in range(N_OPS):
        op = rng.random()

        if op < 0.20 or not model:                       # insert new
            pk = new_pk()
            rec = _make_record(rng, pk)
            col.insert(rec)
            model[pk] = rec

        elif op < 0.28:                                  # upsert existing pk
            pk = existing_pk()
            rec = _make_record(rng, pk)
            col.insert(rec)
            model[pk] = rec

        elif op < 0.34:                                  # insert_many w/ dups
            batch = []
            for _ in range(int(rng.integers(2, 8))):
                pk = new_pk() if rng.random() < 0.7 else existing_pk()
                batch.append(_make_record(rng, pk))
            if rng.random() < 0.3 and batch:             # duplicate inside batch
                dup = dict(batch[0]); dup["score"] = float(rng.random())
                batch.append(dup)
            col.insert_many(batch)
            for rec in batch:                            # last occurrence wins
                model[rec["id"]] = rec

        elif op < 0.44:                                  # update random fields
            pk = existing_pk()
            changes = {}
            for f in rng.choice(["grp", "score", "body", "meta", "ts", "emb"],
                                size=int(rng.integers(1, 4)), replace=False):
                changes[f] = _make_record(rng, pk)[f]
            col.update(pk, **changes)
            model[pk] = {**model[pk], **changes}

        elif op < 0.50:                                  # increment
            pk = existing_pk()
            delta = int(rng.integers(-5, 20))
            col.increment(pk, "n", delta)
            model[pk] = {**model[pk], "n": model[pk]["n"] + delta}

        elif op < 0.56:                                  # col[pk, field] = v
            pk = existing_pk()
            f = str(rng.choice(["score", "n", "body"]))
            v = _make_record(rng, pk)[f]
            col[pk, f] = v
            model[pk] = {**model[pk], f: v}

        elif op < 0.64:                                  # delete
            pk = existing_pk()
            col.delete(pk)
            del model[pk]

        elif op < 0.90:                                  # query cross-checks
            _check_point_reads(col, model, rng)
            which = rng.random()
            if which < 0.3:
                _check_find_count(col, model, rng)
            elif which < 0.5:
                _check_range(col, model, rng)
            elif which < 0.65:
                _check_unique(col, model, rng)
            elif which < 0.8:
                _check_search(col, model, rng)
            else:
                _check_nearest(col, model, rng,
                               where_grp=(None if rng.random() < 0.5
                                          else f"g{rng.integers(0, 8)}"))

        elif op < 0.955:                                 # vacuum — keep handle
            db.vacuum()

        elif op < 0.97:                                  # migrate (rebuild)
            db.migrate_collection("posts", SCHEMA)

        else:                                            # close + reopen
            db.close()
            db = DB(path)
            db.open()
            col = db.collection("posts")

        if step % FULL_CHECK_EVERY == FULL_CHECK_EVERY - 1:
            _full_check(db, col, model, rng)

    _full_check(db, col, model, rng)
    db.close()
