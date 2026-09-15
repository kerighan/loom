"""Benchmark loom-accel accelerated paths vs pure-Python fallbacks.

Measures the three accelerated paths on a Collection with many/range indexes:
  1. BTree range_keys (key-only walk)
  2. Dict get_fields_many (batch hash resolve + gather)
  3. Collection find(fields=...) end-to-end
  4. Collection find(as_columns=True) end-to-end

Each test runs twice: once with accel enabled, once with accel monkey-patched
out, so the table shows the exact before/after on the same data.

Run:  PYTHONPATH=. python benchmarks/benchmark_accel.py
"""
from __future__ import annotations

import gc
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB
from loom import Many

# ── config ──────────────────────────────────────────────────────────────────

N = 50_000          # records
N_USERS = 1_000     # distinct group values
FIND_LIMIT = None   # None = full group scan
FIELDS = ["id", "username", "created_at", "engagement", "likes"]
REPS = 20           # repetitions per measurement


# ── helpers ─────────────────────────────────────────────────────────────────

def timed(fn, reps=REPS):
    """Best of `reps` runs, return (result, seconds)."""
    best = float("inf")
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def disable_accel():
    """Monkey-patch loom modules to hide loom_accel, return a restore fn."""
    import loom.fileio as fio
    import loom.datastructures.dict as dmod
    import loom.datastructures.btree as bmod
    import loom.collection as cmod

    saved = (fio._accel_gather, dmod._accel_resolve,
             bmod._accel_range_keys, cmod._accel_fused)
    fio._accel_gather = None
    dmod._accel_resolve = None
    bmod._accel_range_keys = None
    cmod._accel_fused = None

    def restore():
        fio._accel_gather = saved[0]
        dmod._accel_resolve = saved[1]
        bmod._accel_range_keys = saved[2]
        cmod._accel_fused = saved[3]
    return restore


def has_accel():
    try:
        import loom_accel  # noqa: F401
        return True
    except ImportError:
        return False


# ── build the dataset ───────────────────────────────────────────────────────

def build():
    rng = random.Random(42)
    users = [f"user{u:04d}" for u in range(N_USERS)]
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    db = DB(path)
    col = db.collection("posts", {
        "id": "utf8[16]", "username": "utf8[24]", "created_at": "int64",
        "engagement": "int64", "likes": "int64", "text": "text",
    }, indexes={
        "id": "primary",
        "username": Many(sort="created_at", desc=True, counted=True),
        "engagement": "range",
    })
    records = [{
        "id": f"p{i:012d}",
        "username": rng.choice(users),
        "created_at": i,
        "engagement": rng.randint(0, 10_000),
        "likes": 0,
        "text": f"post {i} about topic {rng.randint(0, 99)}",
    } for i in range(N)]
    col.insert_many(records)
    db.flush()
    return db, col, users, path


# ── benchmarks ──────────────────────────────────────────────────────────────

def bench_find(col, users, label):
    """find(username, fields=FIELDS) — the projected group scan."""
    rng = random.Random(7)
    targets = [rng.choice(users) for _ in range(40)]

    def fn():
        total = 0
        for u in targets:
            total += len(col.find("username", u, fields=FIELDS, limit=FIND_LIMIT))
        return total

    n, best = timed(fn)
    ops = n / best
    us = best / n * 1e6
    print(f"  {label:<42} {ops:>12,.0f} rows/s  ({us:5.2f} µs/row)  [{n} rows]")
    return ops


def bench_find_columns(col, users, label):
    """find(as_columns=True) — columnar return."""
    rng = random.Random(7)
    targets = [rng.choice(users) for _ in range(40)]

    def fn():
        total = 0
        for u in targets:
            c = col.find("username", u, fields=FIELDS, limit=FIND_LIMIT,
                         as_columns=True)
            total += len(c["id"])
        return total

    n, best = timed(fn)
    ops = n / best
    us = best / n * 1e6
    print(f"  {label:<42} {ops:>12,.0f} rows/s  ({us:5.2f} µs/row)  [{n} rows]")
    return ops


def bench_range(col, label):
    """range(engagement, 8000, 10000, fields=FIELDS)."""
    def fn():
        return len(col.range("engagement", 8000, 10000, fields=FIELDS))

    n, best = timed(fn)
    ops = n / best
    us = best / n * 1e6
    print(f"  {label:<42} {ops:>12,.0f} rows/s  ({us:5.2f} µs/row)  [{n} rows]")
    return ops


def bench_btree_walk(col, users, label):
    """BTree.range_keys — the key-only index walk."""
    rng = random.Random(7)
    targets = [rng.choice(users) for _ in range(40)]

    def fn():
        total = 0
        for u in targets:
            ix, lo, hi = col._many_bounds("username", "find", u, None, None)
            total += sum(1 for _ in ix["struct"].range_keys(lo, hi,
                                                            inclusive=(True, False)))
        return total

    n, best = timed(fn)
    ops = n / best
    us = best / n * 1e6
    print(f"  {label:<42} {ops:>12,.0f} keys/s  ({us:5.2f} µs/key)  [{n} keys]")
    return ops


def bench_resolve_batch(col, users, label):
    """Dict.get_fields_many — batch hash resolve + gather."""
    rng = random.Random(7)
    targets = [rng.choice(users) for _ in range(40)]
    # collect pks to resolve
    all_pks = []
    for u in targets:
        ix, lo, hi = col._many_bounds("username", "find", u, None, None)
        for k, entry in ix["struct"].range(lo, hi, inclusive=(True, False)):
            all_pks.append(str(entry["pk"]))

    def fn():
        return len(col._primary.get_fields_many(all_pks, FIELDS))

    n, best = timed(fn)
    ops = n / best
    us = best / n * 1e6
    print(f"  {label:<42} {ops:>12,.0f} rows/s  ({us:5.2f} µs/row)  [{n} rows]")
    return ops


# ── main ────────────────────────────────────────────────────────────────────

def main():
    accel = has_accel()
    print(f"loom-accel: {'installed' if accel else 'NOT installed'}")
    print(f"Building dataset: {N:,} posts, {N_USERS:,} users ...", flush=True)

    db, col, users, path = build()
    # warm caches
    col.find("username", users[0], fields=FIELDS)

    results = {}

    if accel:
        # ── with accel ──────────────────────────────────────────────────
        print("\n── with loom-accel ──")
        # clear pk_in_key cache so it recomputes
        for ix in col._indexes.values():
            ix.pop("pk_in_key", None)
        results["find_accel"] = bench_find(col, users, "find(fields=) [fused]")
        results["find_col_accel"] = bench_find_columns(col, users,
                                                       "find(as_columns=True) [fused]")
        results["range_accel"] = bench_range(col, "range(fields=) [accel]")
        results["btree_accel"] = bench_btree_walk(col, users,
                                                  "BTree.range_keys [accel]")
        results["resolve_accel"] = bench_resolve_batch(col, users,
                                                       "Dict.get_fields_many [accel]")

        # ── without accel ───────────────────────────────────────────────
        print("\n── without loom-accel (pure Python) ──")
        restore = disable_accel()
        for ix in col._indexes.values():
            ix.pop("pk_in_key", None)
        results["find_pure"] = bench_find(col, users, "find(fields=) [pure Python]")
        results["range_pure"] = bench_range(col, "range(fields=) [pure Python]")
        results["btree_pure"] = bench_btree_walk(col, users,
                                                 "BTree.range_keys [pure Python]")
        results["resolve_pure"] = bench_resolve_batch(col, users,
                                                      "Dict.get_fields_many [pure Python]")
        restore()
    else:
        print("\n── pure Python (no accel) ──")
        results["find_pure"] = bench_find(col, users, "find(fields=) [pure Python]")
        results["range_pure"] = bench_range(col, "range(fields=) [pure Python]")
        results["btree_pure"] = bench_btree_walk(col, users,
                                                 "BTree.range_keys [pure Python]")
        results["resolve_pure"] = bench_resolve_batch(col, users,
                                                      "Dict.get_fields_many [pure Python]")

    db.close()
    os.unlink(path)

    # ── summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MARKDOWN SUMMARY")
    print("=" * 70)

    if accel:
        print("\n### loom-accel — projected scan (find / range with fields=)\n")
        print("| Operation | Pure Python | + loom-accel | Speedup |")
        print("|---|---:|---:|---:|")
        rows = [
            ("Collection find(fields=)",
             results["find_pure"], results["find_accel"]),
            ("Collection find(as_columns=True)",
             results["find_pure"], results["find_col_accel"]),
            ("Collection range(fields=)",
             results["range_pure"], results["range_accel"]),
            ("BTree range_keys (key-only walk)",
             results["btree_pure"], results["btree_accel"]),
            ("Dict get_fields_many (batch resolve)",
             results["resolve_pure"], results["resolve_accel"]),
        ]
        for name, pure, fast in rows:
            ratio = fast / pure if pure > 0 else 0
            print(f"| {name} | {pure:,.0f} rows/s | "
                  f"**{fast:,.0f} rows/s** | **{ratio:.1f}×** |")
    else:
        print("\n(loom-accel not installed — only pure-Python numbers shown)")


if __name__ == "__main__":
    main()
