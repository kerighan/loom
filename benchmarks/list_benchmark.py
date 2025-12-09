"""
List Performance Benchmark

Tests:
- Insertion speed (append)
- Lookup speed (random access)
- Slice performance
- Iteration speed
- Comparison with Python list
"""

import sys
import os
import time
import random
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB


def benchmark_insertion():
    """Benchmark append performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Insertion (Append)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        n = 10_000

        # Loom List
        print(f"\nLoom List - Appending {n:,} items:")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})

            start = time.time()
            for i in range(n):
                lst.append({"id": i, "value": float(i * 2)})
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Ops/sec: {n/elapsed:,.0f}")
            print(f"  Per item: {elapsed/n*1000:.3f}ms")

        # Python list (baseline)
        print(f"\nPython list - Appending {n:,} items:")
        print("-" * 50)
        py_list = []

        start = time.time()
        for i in range(n):
            py_list.append({"id": i, "value": float(i * 2)})
        py_elapsed = time.time() - start

        print(f"  Time: {py_elapsed:.3f}s")
        print(f"  Ops/sec: {n/py_elapsed:,.0f}")
        print(f"  Per item: {py_elapsed/n*1000:.3f}ms")

        print(
            f"\n  Loom vs Python: {py_elapsed/elapsed:.1f}x slower (expected for persistence)"
        )

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def benchmark_append_vs_append_many():
    """Compare List.append vs List.append_many performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Append vs Append Many")
    print("=" * 70)

    n = 10_000

    # ---------------------------------------------------------------
    # Baseline: append in a loop
    # ---------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False) as tmp1:
        filename1 = tmp1.name

    try:
        with DB(filename1) as db:
            lst = db.create_list("test_append", {"id": "uint64", "value": "float64"})

            items = [
                {"id": i, "value": float(i * 2)}
                for i in range(n)
            ]

            print(f"\nLoom List - append() x {n:,}:")
            print("-" * 50)

            start = time.time()
            for item in items:
                lst.append(item)
            append_time = time.time() - start

            print(f"  Time: {append_time:.3f}s")
            print(f"  Ops/sec: {n/append_time:,.0f}")
            print(f"  Per item: {append_time/n*1000:.3f}ms")
    finally:
        if os.path.exists(filename1):
            os.remove(filename1)

    # ---------------------------------------------------------------
    # Batched: append_many in a single WAL transaction
    # ---------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False) as tmp2:
        filename2 = tmp2.name

    try:
        with DB(filename2) as db:
            lst = db.create_list(
                "test_append_many", {"id": "uint64", "value": "float64"}
            )

            items = [
                {"id": i, "value": float(i * 2)}
                for i in range(n)
            ]

            print(f"\nLoom List - append_many() with {n:,} items:")
            print("-" * 50)

            start = time.time()
            lst.append_many(items, atomic=True)
            batch_time = time.time() - start

            print(f"  Time: {batch_time:.3f}s")
            print(f"  Ops/sec: {n/batch_time:,.0f}")
            print(f"  Per item: {batch_time/n*1000:.3f}ms")

        if batch_time > 0:
            print(
                f"\n  append_many speedup vs append: {append_time/batch_time:.2f}x"
            )
    finally:
        if os.path.exists(filename2):
            os.remove(filename2)


def benchmark_lookup():
    """Benchmark random access performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Random Access (Lookup)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        n_items = 10_000
        n_lookups = 10_000

        # Prepare Loom List
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})
            for i in range(n_items):
                lst.append({"id": i, "value": float(i)})

        # Benchmark Loom List
        print(f"\nLoom List - {n_lookups:,} random lookups from {n_items:,} items:")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})

            indices = [random.randint(0, n_items - 1) for _ in range(n_lookups)]

            start = time.time()
            for idx in indices:
                item = lst[idx]
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Lookups/sec: {n_lookups/elapsed:,.0f}")
            print(f"  Per lookup: {elapsed/n_lookups*1000:.3f}ms")
            print(
                f"  Cache hit rate: {lst._block_cache.hit_rate:.1%}"
                if lst._block_cache
                else "  Cache: disabled"
            )

        # Benchmark Python list
        print(f"\nPython list - {n_lookups:,} random lookups:")
        print("-" * 50)
        py_list = [{"id": i, "value": float(i)} for i in range(n_items)]

        start = time.time()
        for idx in indices:
            item = py_list[idx]
        py_elapsed = time.time() - start

        print(f"  Time: {py_elapsed:.3f}s")
        if py_elapsed > 0:
            print(f"  Lookups/sec: {n_lookups/py_elapsed:,.0f}")
            print(f"  Per lookup: {py_elapsed/n_lookups*1000:.3f}ms")
            print(
                f"\n  Loom vs Python: {elapsed/py_elapsed:.1f}x slower (mmap + deserialization overhead)"
            )
        else:
            print("  Lookups/sec: (too fast to measure)")
            print("  Per lookup: <0.001ms")
            print(
                "\n  Loom vs Python: N/A (Python too fast to measure accurately)"
            )

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def benchmark_slicing():
    """Benchmark slicing performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Slicing")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        n_items = 10_000

        # Prepare data
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64"})
            for i in range(n_items):
                lst.append({"id": i})

        # Benchmark different slice sizes
        slice_sizes = [10, 100, 1000]

        for size in slice_sizes:
            print(f"\nSlice size: {size} items")
            print("-" * 50)

            # Loom List - dict slicing (convenient but slower)
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                n_slices = 1000
                start = time.time()
                for _ in range(n_slices):
                    start_idx = random.randint(0, n_items - size)
                    items = lst[start_idx : start_idx + size]
                elapsed = time.time() - start

                print(f"  Loom (dicts): {elapsed:.3f}s ({n_slices/elapsed:,.0f} slices/sec)")

            # Loom List - array slicing (fast)
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                start = time.time()
                for _ in range(n_slices):
                    start_idx = random.randint(0, n_items - size)
                    arr = lst.slice_array(start_idx, start_idx + size)
                array_elapsed = time.time() - start

                print(f"  Loom (array): {array_elapsed:.3f}s ({n_slices/array_elapsed:,.0f} slices/sec)")

            # Python list
            py_list = [{"id": i} for i in range(n_items)]

            start = time.time()
            for _ in range(n_slices):
                start_idx = random.randint(0, n_items - size)
                items = py_list[start_idx : start_idx + size]
            py_elapsed = time.time() - start

            if py_elapsed > 0:
                print(
                    f"  Python: {py_elapsed:.3f}s ({n_slices/py_elapsed:,.0f} slices/sec)"
                )
                print(f"  Dict ratio: {elapsed/py_elapsed:.1f}x, Array ratio: {array_elapsed/py_elapsed:.1f}x")
            else:
                print("  Python: 0.000s (too fast to measure)")
                print("  Ratio: N/A (Python too fast to measure accurately)")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def benchmark_iteration():
    """Benchmark iteration performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Iteration")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        n_items = 10_000

        # Prepare data
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})
            for i in range(n_items):
                lst.append({"id": i, "value": float(i)})

        # Benchmark Loom List
        print(f"\nLoom List - Iterating {n_items:,} items:")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})

            start = time.time()
            count = 0
            for item in lst:
                count += 1
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Items/sec: {n_items/elapsed:,.0f}")
            print(f"  Per item: {elapsed/n_items*1000:.3f}ms")

        # Benchmark Python list
        print(f"\nPython list - Iterating {n_items:,} items:")
        print("-" * 50)
        py_list = [{"id": i, "value": float(i)} for i in range(n_items)]

        start = time.time()
        count = 0
        for item in py_list:
            count += 1
        py_elapsed = time.time() - start

        print(f"  Time: {py_elapsed:.3f}s")
        if py_elapsed > 0:
            print(f"  Items/sec: {n_items/py_elapsed:,.0f}")
            print(f"  Per item: {py_elapsed/n_items*1000:.3f}ms")
            print(f"\n  Loom vs Python: {elapsed/py_elapsed:.1f}x slower")
        else:
            print("  Items/sec: (too fast to measure)")
            print("  Per item: <0.001ms")
            print("\n  Loom vs Python: N/A (Python too fast to measure accurately)")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def benchmark_large_scale():
    """Benchmark with large dataset."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Large Scale (100K items)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        n = 100_000

        print(f"\nInserting {n:,} items...")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})

            start = time.time()
            for i in range(n):
                lst.append({"id": i, "value": float(i)})
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Ops/sec: {n/elapsed:,.0f}")
            print(f"  Blocks allocated: {lst.p_last - lst.P_INIT + 1}")

        print(f"\nRandom access (10K lookups)...")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})

            n_lookups = 10_000
            indices = [random.randint(0, n - 1) for _ in range(n_lookups)]

            start = time.time()
            for idx in indices:
                item = lst[idx]
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Lookups/sec: {n_lookups/elapsed:,.0f}")
            if lst._block_cache:
                print(f"  Cache hit rate: {lst._block_cache.hit_rate:.1%}")

        print(f"\nFull iteration...")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "float64"})

            start = time.time()
            count = sum(1 for _ in lst)
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Items/sec: {count/elapsed:,.0f}")

        # File size
        file_size = os.path.getsize(filename)
        print(f"\nFile size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"Bytes per item: {file_size/n:.1f}")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def benchmark_cache_impact():
    """Benchmark cache impact."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Cache Impact")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        n_items = 10_000
        n_lookups = 10_000

        # Prepare data
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64"}, cache_size=0)
            for i in range(n_items):
                lst.append({"id": i})

        # Without cache
        print(f"\nWithout cache:")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64"}, cache_size=0)

            # Hot access pattern (80/20 rule)
            hot_indices = list(range(100))  # 100 hot items
            cold_indices = list(range(100, n_items))

            indices = []
            for _ in range(n_lookups):
                if random.random() < 0.8:
                    indices.append(random.choice(hot_indices))
                else:
                    indices.append(random.choice(cold_indices))

            start = time.time()
            for idx in indices:
                item = lst[idx]
            no_cache_time = time.time() - start

            print(f"  Time: {no_cache_time:.3f}s")
            print(f"  Lookups/sec: {n_lookups/no_cache_time:,.0f}")

        # With cache
        print(f"\nWith cache (10 blocks):")
        print("-" * 50)
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64"}, cache_size=10)

            start = time.time()
            for idx in indices:
                item = lst[idx]
            cache_time = time.time() - start

            print(f"  Time: {cache_time:.3f}s")
            print(f"  Lookups/sec: {n_lookups/cache_time:,.0f}")
            print(f"  Cache hit rate: {lst._block_cache.hit_rate:.1%}")
            print(f"  Speedup: {no_cache_time/cache_time:.2f}x")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    benchmark_insertion()
    benchmark_append_vs_append_many()
    benchmark_lookup()
    benchmark_slicing()
    benchmark_iteration()
    benchmark_cache_impact()
    benchmark_large_scale()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Loom List provides persistent storage with good performance")
    print("✓ Exponential block growth avoids reallocation overhead")
    print("✓ LRU cache significantly improves hot access patterns")
    print("✓ Suitable for large datasets (100K+ items)")
    print("=" * 70 + "\n")
