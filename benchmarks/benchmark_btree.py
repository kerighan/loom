"""
Benchmark BTree vs Dict performance.

Compares:
- Insert performance
- Lookup performance
- Iteration performance
- Range query performance (BTree only)
- Delete performance
"""

import os
import sys
import tempfile
import time
import random
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB
from loom.datastructures import BTree, Dict


def generate_keys(n, key_length=20):
    """Generate n random string keys."""
    keys = []
    for _ in range(n):
        key = "".join(random.choices(string.ascii_lowercase, k=key_length))
        keys.append(key)
    return keys


def generate_sorted_keys(n, prefix="key_"):
    """Generate n sorted keys."""
    return [f"{prefix}{i:08d}" for i in range(n)]


def benchmark_insert(structure, keys, values):
    """Benchmark insert performance."""
    start = time.perf_counter()
    for key, value in zip(keys, values):
        structure[key] = value
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_lookup(structure, keys):
    """Benchmark lookup performance."""
    start = time.perf_counter()
    for key in keys:
        _ = structure[key]
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_contains(structure, keys):
    """Benchmark contains check performance."""
    start = time.perf_counter()
    for key in keys:
        _ = key in structure
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_iteration(structure):
    """Benchmark iteration performance."""
    start = time.perf_counter()
    count = 0
    for key in structure.keys():
        count += 1
    elapsed = time.perf_counter() - start
    return elapsed, count


def benchmark_range_query(btree, start_key, end_key):
    """Benchmark range query (BTree only)."""
    start = time.perf_counter()
    count = 0
    for key, value in btree.range(start_key, end_key):
        count += 1
    elapsed = time.perf_counter() - start
    return elapsed, count


def benchmark_delete(structure, keys):
    """Benchmark delete performance."""
    start = time.perf_counter()
    for key in keys:
        del structure[key]
    elapsed = time.perf_counter() - start
    return elapsed


def run_benchmarks(n_items=10000):
    """Run all benchmarks."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: BTree vs Dict ({n_items:,} items)")
    print(f"{'='*60}\n")

    # Generate test data
    keys = generate_sorted_keys(n_items)
    random.shuffle(keys)  # Randomize insertion order
    values = [{"id": i, "name": f"User {i}"} for i in range(n_items)]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "benchmark.db")

        with DB(db_path, header_size=10 * 1024 * 1024) as db:
            # Create datasets
            btree_ds = db.create_dataset("btree_users", id="uint32", name="U50")
            dict_ds = db.create_dataset("dict_users", id="uint32", name="U50")

            # Create structures
            btree = db.create_btree("users_btree", btree_ds, cache_size=1000)
            dct = db.create_dict("users_dict", dict_ds, cache_size=1000)

            # ========== INSERT BENCHMARK ==========
            print("INSERT PERFORMANCE")
            print("-" * 40)

            btree_insert = benchmark_insert(btree, keys, values)
            print(f"  BTree: {btree_insert:.3f}s ({n_items/btree_insert:,.0f} ops/sec)")

            dct_insert = benchmark_insert(dct, keys, values)
            print(f"  Dict:  {dct_insert:.3f}s ({n_items/dct_insert:,.0f} ops/sec)")

            ratio = btree_insert / dct_insert
            print(
                f"  Ratio: BTree is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}"
            )
            print()

            # ========== LOOKUP BENCHMARK ==========
            print("LOOKUP PERFORMANCE (random access)")
            print("-" * 40)

            # Shuffle keys for random access pattern
            lookup_keys = keys.copy()
            random.shuffle(lookup_keys)

            btree_lookup = benchmark_lookup(btree, lookup_keys)
            print(f"  BTree: {btree_lookup:.3f}s ({n_items/btree_lookup:,.0f} ops/sec)")

            dict_lookup = benchmark_lookup(dct, lookup_keys)
            print(f"  Dict:  {dict_lookup:.3f}s ({n_items/dict_lookup:,.0f} ops/sec)")

            ratio = btree_lookup / dict_lookup
            print(
                f"  Ratio: BTree is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}"
            )
            print()

            # ========== CONTAINS BENCHMARK ==========
            print("CONTAINS CHECK PERFORMANCE")
            print("-" * 40)

            btree_contains = benchmark_contains(btree, lookup_keys)
            print(
                f"  BTree: {btree_contains:.3f}s ({n_items/btree_contains:,.0f} ops/sec)"
            )

            dict_contains = benchmark_contains(dct, lookup_keys)
            print(
                f"  Dict:  {dict_contains:.3f}s ({n_items/dict_contains:,.0f} ops/sec)"
            )

            ratio = btree_contains / dict_contains
            print(
                f"  Ratio: BTree is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}"
            )
            print()

            # ========== ITERATION BENCHMARK ==========
            print("ITERATION PERFORMANCE")
            print("-" * 40)

            btree_iter, btree_count = benchmark_iteration(btree)
            print(f"  BTree: {btree_iter:.3f}s ({btree_count:,} items, SORTED)")

            dict_iter, dict_count = benchmark_iteration(dct)
            print(f"  Dict:  {dict_iter:.3f}s ({dict_count:,} items, unsorted)")

            ratio = btree_iter / dict_iter
            print(
                f"  Ratio: BTree is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}"
            )
            print()

            # ========== RANGE QUERY BENCHMARK (BTree only) ==========
            print("RANGE QUERY PERFORMANCE (BTree only)")
            print("-" * 40)

            # Query for ~10% of keys
            sorted_keys = sorted(keys)
            start_idx = len(sorted_keys) // 4
            end_idx = start_idx + len(sorted_keys) // 10
            start_key = sorted_keys[start_idx]
            end_key = sorted_keys[end_idx]

            range_time, range_count = benchmark_range_query(btree, start_key, end_key)
            print(f"  BTree range({start_key[:15]}..., {end_key[:15]}...):")
            print(f"    {range_time:.4f}s ({range_count:,} items)")
            print(f"    {range_count/range_time:,.0f} items/sec")
            print()

            # ========== MIN/MAX BENCHMARK (BTree only) ==========
            print("MIN/MAX PERFORMANCE (BTree only)")
            print("-" * 40)

            start = time.perf_counter()
            for _ in range(1000):
                _ = btree.min()
                _ = btree.max()
            minmax_time = time.perf_counter() - start
            print(f"  1000 min()+max() calls: {minmax_time:.4f}s")
            print(f"  {2000/minmax_time:,.0f} ops/sec")
            print()

            # ========== DELETE BENCHMARK ==========
            print("DELETE PERFORMANCE")
            print("-" * 40)

            # Delete half the items
            delete_keys = keys[: n_items // 2]
            random.shuffle(delete_keys)

            btree_delete = benchmark_delete(btree, delete_keys)
            print(
                f"  BTree: {btree_delete:.3f}s ({len(delete_keys)/btree_delete:,.0f} ops/sec)"
            )

            dict_delete = benchmark_delete(dct, delete_keys)
            print(
                f"  Dict:  {dict_delete:.3f}s ({len(delete_keys)/dict_delete:,.0f} ops/sec)"
            )

            ratio = btree_delete / dict_delete
            print(
                f"  Ratio: BTree is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}"
            )
            print()

    # ========== SUMMARY ==========
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        """
BTree advantages:
  ✓ Ordered iteration (keys always sorted)
  ✓ Range queries (get all keys between A and B)
  ✓ Prefix searches (get all keys starting with X)
  ✓ Min/Max in O(log n)

Dict advantages:
  ✓ Faster insert/lookup/delete (O(1) vs O(log n))
  ✓ Bloom filter acceleration for negative lookups
  
Use BTree when you need ordering or range queries.
Use Dict when you need maximum speed for point lookups.
"""
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark BTree vs Dict")
    parser.add_argument(
        "-n",
        "--items",
        type=int,
        default=10000,
        help="Number of items to benchmark (default: 10000)",
    )
    args = parser.parse_args()

    run_benchmarks(args.items)
