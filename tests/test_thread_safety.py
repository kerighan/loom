"""
Thread-safety stress tests for loom DataStructures.

Runs N_THREADS threads concurrently doing interleaved reads and writes
on Dict, List, and Queue to surface race conditions (torn writes,
corrupted hash slots, wrong counts, etc.).
"""

import random
import tempfile
import threading
import os

import pytest

from loom import DB


N_THREADS  = 8
OPS_THREAD = 300


def _run_stress(db):
    """Core stress worker: hammer Dict + List + Queue from multiple threads."""
    ds  = db.create_dataset("kv", key="U30", val="int64")
    dct = db.create_dict("store", ds)
    lst = db.create_list("log", db.create_dataset("log_ds", msg="U30"))
    q   = db.create_queue("q", db.create_dataset("q_ds", n="int64"), block_size=32)

    errors = []
    barrier = threading.Barrier(N_THREADS)

    def worker(tid):
        try:
            barrier.wait()
            rng = random.Random(tid)
            for i in range(OPS_THREAD):
                key = f"k{rng.randint(0, 30)}"   # high collision → many overwrites
                val = tid * 10_000 + i
                dct[key] = {"key": key, "val": val}
                lst.append({"msg": f"t{tid}_{i}"})
                q.push({"n": val})
                if i % 5 == 0:
                    _ = dct.get(key)
                    _ = len(lst)
        except Exception as e:
            errors.append((tid, str(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return errors, dct, lst, q


def test_multithread_no_errors():
    """No exceptions under concurrent reads + writes."""
    with tempfile.TemporaryDirectory() as tmp:
        with DB(os.path.join(tmp, "stress.db")) as db:
            errors, _, _, _ = _run_stress(db)
    assert errors == [], f"Thread errors: {errors}"


def test_multithread_list_count():
    """List length equals total appends (N_THREADS × OPS_THREAD)."""
    with tempfile.TemporaryDirectory() as tmp:
        with DB(os.path.join(tmp, "stress2.db")) as db:
            errors, _, lst, _ = _run_stress(db)
            assert errors == []
            assert len(lst) == N_THREADS * OPS_THREAD


def test_multithread_dict_keys_bounded():
    """Dict never holds more keys than the key space (0..30)."""
    with tempfile.TemporaryDirectory() as tmp:
        with DB(os.path.join(tmp, "stress3.db")) as db:
            errors, dct, _, _ = _run_stress(db)
            assert errors == []
            assert len(list(dct.keys())) <= 31


def test_write_lock_explicit():
    """db.write_lock() context manager serialises compound multi-step operations."""
    with tempfile.TemporaryDirectory() as tmp:
        with DB(os.path.join(tmp, "wl.db")) as db:
            ds  = db.create_dataset("ds", val="int64")
            dct = db.create_dict("d", ds)
            errors = []
            barrier = threading.Barrier(4)

            def writer(tid):
                try:
                    barrier.wait()
                    for i in range(200):
                        with db.write_lock():
                            dct[f"k{i}"] = {"val": tid * 1000 + i}
                            _ = dct.get(f"k{i}")  # read within lock
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == []
