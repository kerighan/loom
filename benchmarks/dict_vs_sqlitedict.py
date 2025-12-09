"""Benchmark comparison: Loom Dict vs SqliteDict.

Compares performance of:
- Loom Dict (our implementation)
- SqliteDict (popular persistent dict library)

Tests:
- Sequential inserts
- Random reads
- Mixed read/write workload
- Iteration
- Storage size
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import time
import random
import string
from loom.database import DB
from loom.datastructures.dict import Dict

# Try to import sqlitedict
try:
    from sqlitedict import SqliteDict

    HAS_SQLITEDICT = True
except ImportError:
    HAS_SQLITEDICT = False
    print("WARNING: sqlitedict not installed. Run: pip install sqlitedict")
    print("Only Loom Dict benchmarks will run.\n")


def generate_random_string(length=20):
    """Generate a random string of fixed length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def benchmark_loom_dict(num_items, value_size=50):
    """Benchmark Loom Dict operations."""
    results = {}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".loom") as tmp:
        filename = tmp.name

    try:
        # Setup
        db = DB(filename)
        dataset = db.create_dataset("items", name=f"U{value_size}", value="uint64")
        # Disable bloom filter for large dicts (>50K) due to known scaling issue
        use_bloom = num_items <= 50000
        d = db.create_dict(
            "benchmark_dict", dataset, cache_size=10000, use_bloom=use_bloom
        )

        # Generate test data
        keys = [f"key_{i:08d}" for i in range(num_items)]
        values = [
            {"name": generate_random_string(value_size - 10), "value": i}
            for i in range(num_items)
        ]

        # 1. Sequential inserts
        start = time.time()
        for key, value in zip(keys, values):
            d[key] = value
        insert_time = time.time() - start
        results["insert_time"] = insert_time
        results["insert_rate"] = num_items / insert_time

        # 2. Sequential reads
        start = time.time()
        for key in keys:
            _ = d[key]
        seq_read_time = time.time() - start
        results["seq_read_time"] = seq_read_time
        results["seq_read_rate"] = num_items / seq_read_time

        # 3. Random reads
        random_keys = random.sample(keys, min(10000, num_items))
        start = time.time()
        for key in random_keys:
            _ = d[key]
        random_read_time = time.time() - start
        results["random_read_time"] = random_read_time
        results["random_read_rate"] = len(random_keys) / random_read_time

        # 4. Mixed workload (80% read, 20% write)
        num_ops = min(10000, num_items)
        start = time.time()
        for i in range(num_ops):
            key = random.choice(keys)
            if random.random() < 0.8:
                _ = d[key]
            else:
                d[key] = {"name": generate_random_string(value_size - 10), "value": i}
        mixed_time = time.time() - start
        results["mixed_time"] = mixed_time
        results["mixed_rate"] = num_ops / mixed_time

        # 5. Iteration
        start = time.time()
        count = 0
        for key in d.keys():
            count += 1
        iter_time = time.time() - start
        results["iter_time"] = iter_time
        results["iter_rate"] = count / iter_time if iter_time > 0 else float("inf")

        # 6. Contains check (membership)
        test_keys = random.sample(keys, min(5000, num_items))
        missing_keys = [f"missing_{i}" for i in range(min(5000, num_items))]
        all_test_keys = test_keys + missing_keys
        random.shuffle(all_test_keys)

        start = time.time()
        for key in all_test_keys:
            _ = key in d
        contains_time = time.time() - start
        results["contains_time"] = contains_time
        results["contains_rate"] = len(all_test_keys) / contains_time

        # Save and get file size
        d.save()
        db.close()

        results["file_size"] = os.path.getsize(filename)
        results["bytes_per_item"] = results["file_size"] / num_items

    finally:
        os.unlink(filename)

    return results


def benchmark_sqlitedict(num_items, value_size=50):
    """Benchmark SqliteDict operations."""
    if not HAS_SQLITEDICT:
        return None

    results = {}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tmp:
        filename = tmp.name

    try:
        # Generate test data
        keys = [f"key_{i:08d}" for i in range(num_items)]
        values = [
            {"name": generate_random_string(value_size - 10), "value": i}
            for i in range(num_items)
        ]

        # 1. Sequential inserts
        d = SqliteDict(filename, autocommit=True)
        start = time.time()
        for key, value in zip(keys, values):
            d[key] = value
        insert_time = time.time() - start
        results["insert_time"] = insert_time
        results["insert_rate"] = num_items / insert_time
        d.close()

        # 2. Sequential reads
        d = SqliteDict(filename)
        start = time.time()
        for key in keys:
            _ = d[key]
        seq_read_time = time.time() - start
        results["seq_read_time"] = seq_read_time
        results["seq_read_rate"] = num_items / seq_read_time

        # 3. Random reads
        random_keys = random.sample(keys, min(10000, num_items))
        start = time.time()
        for key in random_keys:
            _ = d[key]
        random_read_time = time.time() - start
        results["random_read_time"] = random_read_time
        results["random_read_rate"] = len(random_keys) / random_read_time

        # 4. Mixed workload (80% read, 20% write)
        d.close()
        d = SqliteDict(filename, autocommit=True)
        num_ops = min(10000, num_items)
        start = time.time()
        for i in range(num_ops):
            key = random.choice(keys)
            if random.random() < 0.8:
                _ = d[key]
            else:
                d[key] = {"name": generate_random_string(value_size - 10), "value": i}
        mixed_time = time.time() - start
        results["mixed_time"] = mixed_time
        results["mixed_rate"] = num_ops / mixed_time

        # 5. Iteration
        start = time.time()
        count = 0
        for key in d.keys():
            count += 1
        iter_time = time.time() - start
        results["iter_time"] = iter_time
        results["iter_rate"] = count / iter_time if iter_time > 0 else float("inf")

        # 6. Contains check (membership)
        test_keys = random.sample(keys, min(5000, num_items))
        missing_keys = [f"missing_{i}" for i in range(min(5000, num_items))]
        all_test_keys = test_keys + missing_keys
        random.shuffle(all_test_keys)

        start = time.time()
        for key in all_test_keys:
            _ = key in d
        contains_time = time.time() - start
        results["contains_time"] = contains_time
        results["contains_rate"] = len(all_test_keys) / contains_time

        d.close()

        results["file_size"] = os.path.getsize(filename)
        results["bytes_per_item"] = results["file_size"] / num_items

    finally:
        os.unlink(filename)

    return results


def print_comparison(num_items, loom_results, sqlite_results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS: {num_items:,} items")
    print(f"{'='*70}")

    def format_rate(rate):
        if rate >= 1000000:
            return f"{rate/1000000:.2f}M"
        elif rate >= 1000:
            return f"{rate/1000:.1f}K"
        else:
            return f"{rate:.0f}"

    def format_time(t):
        if t < 0.001:
            return f"{t*1000000:.1f}µs"
        elif t < 1:
            return f"{t*1000:.1f}ms"
        else:
            return f"{t:.2f}s"

    def format_size(size):
        if size >= 1024 * 1024:
            return f"{size/1024/1024:.1f}MB"
        elif size >= 1024:
            return f"{size/1024:.1f}KB"
        else:
            return f"{size}B"

    headers = ["Operation", "Loom Dict", "SqliteDict", "Winner", "Speedup"]

    comparisons = [
        ("Insert", "insert_rate", "ops/sec", True),
        ("Sequential Read", "seq_read_rate", "ops/sec", True),
        ("Random Read", "random_read_rate", "ops/sec", True),
        ("Mixed (80r/20w)", "mixed_rate", "ops/sec", True),
        ("Iteration", "iter_rate", "keys/sec", True),
        ("Contains Check", "contains_rate", "ops/sec", True),
    ]

    print(
        f"\n{'Operation':<20} {'Loom Dict':>15} {'SqliteDict':>15} {'Winner':>12} {'Speedup':>10}"
    )
    print("-" * 75)

    for name, key, unit, higher_better in comparisons:
        loom_val = loom_results[key]

        if sqlite_results:
            sqlite_val = sqlite_results[key]

            if higher_better:
                if loom_val > sqlite_val:
                    winner = "Loom"
                    speedup = loom_val / sqlite_val
                else:
                    winner = "SqliteDict"
                    speedup = sqlite_val / loom_val
            else:
                if loom_val < sqlite_val:
                    winner = "Loom"
                    speedup = sqlite_val / loom_val
                else:
                    winner = "SqliteDict"
                    speedup = loom_val / sqlite_val

            print(
                f"{name:<20} {format_rate(loom_val):>12}/s {format_rate(sqlite_val):>12}/s {winner:>12} {speedup:>9.1f}x"
            )
        else:
            print(
                f"{name:<20} {format_rate(loom_val):>12}/s {'N/A':>15} {'N/A':>12} {'N/A':>10}"
            )

    # Storage comparison
    print("-" * 75)
    loom_size = loom_results["file_size"]
    loom_per_item = loom_results["bytes_per_item"]

    if sqlite_results:
        sqlite_size = sqlite_results["file_size"]
        sqlite_per_item = sqlite_results["bytes_per_item"]

        if loom_size < sqlite_size:
            winner = "Loom"
            ratio = sqlite_size / loom_size
        else:
            winner = "SqliteDict"
            ratio = loom_size / sqlite_size

        print(
            f"{'Storage Size':<20} {format_size(loom_size):>15} {format_size(sqlite_size):>15} {winner:>12} {ratio:>9.1f}x"
        )
        print(f"{'Bytes/Item':<20} {loom_per_item:>13.0f}B {sqlite_per_item:>13.0f}B")
    else:
        print(f"{'Storage Size':<20} {format_size(loom_size):>15} {'N/A':>15}")
        print(f"{'Bytes/Item':<20} {loom_per_item:>13.0f}B {'N/A':>15}")


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("LOOM DICT vs SQLITEDICT BENCHMARK")
    print("=" * 70)

    if HAS_SQLITEDICT:
        print("SqliteDict: AVAILABLE")
    else:
        print("SqliteDict: NOT INSTALLED (pip install sqlitedict)")

    # Test sizes
    sizes = [1000, 10000, 100000]

    all_results = []

    for num_items in sizes:
        print(f"\n{'#'*70}")
        print(f"# Testing with {num_items:,} items")
        print(f"{'#'*70}")

        print("\nRunning Loom Dict benchmark...")
        loom_results = benchmark_loom_dict(num_items)

        sqlite_results = None
        if HAS_SQLITEDICT:
            print("Running SqliteDict benchmark...")
            sqlite_results = benchmark_sqlitedict(num_items)

        print_comparison(num_items, loom_results, sqlite_results)
        all_results.append((num_items, loom_results, sqlite_results))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if HAS_SQLITEDICT:
        print("\nAverage speedups (Loom vs SqliteDict):")

        metrics = [
            "insert_rate",
            "seq_read_rate",
            "random_read_rate",
            "mixed_rate",
            "iter_rate",
            "contains_rate",
        ]
        metric_names = [
            "Insert",
            "Seq Read",
            "Random Read",
            "Mixed",
            "Iteration",
            "Contains",
        ]

        for metric, name in zip(metrics, metric_names):
            speedups = []
            for num_items, loom, sqlite in all_results:
                if sqlite:
                    speedup = loom[metric] / sqlite[metric]
                    speedups.append(speedup)

            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"  {name:<15}: {avg_speedup:.1f}x faster")
    else:
        print("\nInstall sqlitedict for comparison: pip install sqlitedict")

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_benchmarks()
