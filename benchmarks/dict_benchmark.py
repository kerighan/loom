"""Benchmark standalone Dict performance."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import time
import random
from loom.database import DB


def benchmark_dict(num_items=10000, key_length=20):
    """Benchmark Dict operations."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        print("=" * 70)
        print(f"BENCHMARK: Dict - {num_items:,} items")
        print("=" * 70)

        # ================================================================
        # CREATE & INSERT
        # ================================================================
        start = time.time()
        with DB(filename) as db:
            # Create dataset first, then dict
            user_dataset = db.create_dataset(
                "users",
                id="uint64",
                name="U50",
                email="U100",
                score="float64",
            )
            users = db.create_dict("users_dict", user_dataset)

            # Insert items
            for i in range(num_items):
                key = f"user_{i:0{key_length}d}"
                users[key] = {
                    "id": i,
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "score": i * 1.5,
                }

        insert_time = time.time() - start

        print(f"\n✓ INSERT:")
        print(f"  Items: {num_items:,}")
        print(f"  Time: {insert_time:.3f}s")
        print(f"  Rate: {num_items/insert_time:,.0f} inserts/sec")
        print(f"  Avg latency: {insert_time/num_items*1000:.3f}ms")

        # ================================================================
        # SEQUENTIAL READ
        # ================================================================
        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            read_count = 0
            for key in users.keys():
                user = users[key]
                read_count += 1

        seq_read_time = time.time() - start

        print(f"\n✓ SEQUENTIAL READ:")
        print(f"  Items read: {read_count:,}")
        print(f"  Time: {seq_read_time:.3f}s")
        print(f"  Rate: {read_count/seq_read_time:,.0f} reads/sec")

        # ================================================================
        # RANDOM READ (with cache warmup)
        # ================================================================
        random.seed(42)
        sample_size = min(10000, num_items)
        random_keys = [
            f"user_{random.randint(0, num_items-1):0{key_length}d}"
            for _ in range(sample_size)
        ]

        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            for key in random_keys:
                user = users[key]

        random_time = time.time() - start

        print(f"\n✓ RANDOM READ:")
        print(f"  Lookups: {sample_size:,}")
        print(f"  Time: {random_time:.3f}s")
        print(f"  Rate: {sample_size/random_time:,.0f} lookups/sec")
        print(f"  Avg latency: {random_time/sample_size*1000:.3f}ms")

        # ================================================================
        # RANDOM READ (cache cold - new session)
        # ================================================================
        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            for key in random_keys:
                user = users[key]

        cold_time = time.time() - start

        print(f"\n✓ RANDOM READ (Cold Cache):")
        print(f"  Lookups: {sample_size:,}")
        print(f"  Time: {cold_time:.3f}s")
        print(f"  Rate: {sample_size/cold_time:,.0f} lookups/sec")
        print(f"  Avg latency: {cold_time/sample_size*1000:.3f}ms")
        print(f"  Cache speedup: {cold_time/random_time:.2f}x")

        # ================================================================
        # UPDATE
        # ================================================================
        update_keys = random_keys[: min(1000, len(random_keys))]

        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            for key in update_keys:
                user = users[key]
                user["score"] = user["score"] * 2
                users[key] = user

        update_time = time.time() - start

        print(f"\n✓ UPDATE:")
        print(f"  Updates: {len(update_keys):,}")
        print(f"  Time: {update_time:.3f}s")
        print(f"  Rate: {len(update_keys)/update_time:,.0f} updates/sec")

        # ================================================================
        # DELETE
        # ================================================================
        # Use actual keys from the dict (not random_keys which may not exist)
        # Delete keys from the second half to avoid overlap with update_keys
        delete_count = min(1000, num_items // 2)
        delete_keys = [
            f"user_{i:0{key_length}d}"
            for i in range(num_items // 2, num_items // 2 + delete_count)
        ]

        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            for key in delete_keys:
                del users[key]

        delete_time = time.time() - start

        print(f"\n✓ DELETE:")
        print(f"  Deletes: {len(delete_keys):,}")
        print(f"  Time: {delete_time:.3f}s")
        if len(delete_keys) > 0:
            print(f"  Rate: {len(delete_keys)/delete_time:,.0f} deletes/sec")
        else:
            print(f"  Rate: N/A")

        # ================================================================
        # CONTAINS
        # ================================================================
        # Test with actual keys that exist (excluding deleted ones)
        contains_count = min(1000, num_items)
        contains_keys = [
            f"user_{i:0{key_length}d}"
            for i in range(contains_count)
            if i < num_items // 2 or i >= num_items // 2 + delete_count
        ]

        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            hit_count = 0
            for key in contains_keys:
                if key in users:
                    hit_count += 1

        contains_time = time.time() - start

        print(f"\n✓ CONTAINS:")
        print(f"  Checks: {len(contains_keys):,}")
        print(f"  Hits: {hit_count:,}")
        print(f"  Time: {contains_time:.3f}s")
        print(f"  Rate: {len(contains_keys)/contains_time:,.0f} checks/sec")

        # ================================================================
        # FILE SIZE
        # ================================================================
        file_size = os.path.getsize(filename)
        bytes_per_item = file_size / num_items

        print(f"\n✓ STORAGE:")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"  Per item: {bytes_per_item:.0f} bytes")
        print(f"  Overhead: {(bytes_per_item / 200) * 100:.1f}% (schema ~200 bytes)")

        # ================================================================
        # ITERATION
        # ================================================================
        start = time.time()
        with DB(filename) as db:
            from loom.datastructures.dict import Dict

            users = Dict("users_dict", db, None)

            count = 0
            for key, value in users.items():
                count += 1

        iter_time = time.time() - start

        print(f"\n✓ ITERATION:")
        print(f"  Items: {count:,}")
        print(f"  Time: {iter_time:.3f}s")
        print(f"  Rate: {count/iter_time:,.0f} items/sec")

        return {
            "num_items": num_items,
            "insert_time": insert_time,
            "seq_read_time": seq_read_time,
            "random_time": random_time,
            "cold_time": cold_time,
            "update_time": update_time,
            "delete_time": delete_time,
            "contains_time": contains_time,
            "iter_time": iter_time,
            "file_size": file_size,
        }

    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    print("\n" + "🚀 " * 35)
    print("DICT PERFORMANCE BENCHMARK")
    print("🚀 " * 35 + "\n")

    # Small scale
    print("\n" + "─" * 70)
    print("SMALL SCALE")
    print("─" * 70)
    small = benchmark_dict(num_items=1000, key_length=10)

    # Medium scale
    print("\n" + "─" * 70)
    print("MEDIUM SCALE")
    print("─" * 70)
    medium = benchmark_dict(num_items=10000, key_length=15)

    # Large scale
    print("\n" + "─" * 70)
    print("LARGE SCALE")
    print("─" * 70)
    large = benchmark_dict(num_items=100000, key_length=20)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nInsert Performance:")
    print(f"  1K items:   {small['num_items']/small['insert_time']:,.0f} inserts/sec")
    print(f"  10K items:  {medium['num_items']/medium['insert_time']:,.0f} inserts/sec")
    print(f"  100K items: {large['num_items']/large['insert_time']:,.0f} inserts/sec")

    print("\nRandom Read Performance (warm cache):")
    print(
        f"  1K items:   {10000/small['random_time']:,.0f} reads/sec ({small['random_time']/10000*1000:.3f}ms avg)"
    )
    print(
        f"  10K items:  {10000/medium['random_time']:,.0f} reads/sec ({medium['random_time']/10000*1000:.3f}ms avg)"
    )
    print(
        f"  100K items: {10000/large['random_time']:,.0f} reads/sec ({large['random_time']/10000*1000:.3f}ms avg)"
    )

    print("\nRandom Read Performance (cold cache):")
    print(
        f"  1K items:   {10000/small['cold_time']:,.0f} reads/sec ({small['cold_time']/10000*1000:.3f}ms avg)"
    )
    print(
        f"  10K items:  {10000/medium['cold_time']:,.0f} reads/sec ({medium['cold_time']/10000*1000:.3f}ms avg)"
    )
    print(
        f"  100K items: {10000/large['cold_time']:,.0f} reads/sec ({large['cold_time']/10000*1000:.3f}ms avg)"
    )

    print("\nStorage Efficiency:")
    print(
        f"  1K items:   {small['file_size']/1024:.0f}KB ({small['file_size']/small['num_items']:.0f} bytes/item)"
    )
    print(
        f"  10K items:  {medium['file_size']/1024:.0f}KB ({medium['file_size']/medium['num_items']:.0f} bytes/item)"
    )
    print(
        f"  100K items: {large['file_size']/1024/1024:.1f}MB ({large['file_size']/large['num_items']:.0f} bytes/item)"
    )

    print("\n" + "✅ " * 35)
    print("BENCHMARK COMPLETE!")
    print("✅ " * 35 + "\n")
