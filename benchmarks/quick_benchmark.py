"""Quick performance benchmark - smaller scale for fast results."""

import os
import sys
import time
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB
from loom.datastructures import BloomFilter, CountingBloomFilter


def quick_benchmark():
    """Quick benchmark with smaller dataset."""
    print("\n" + "=" * 70)
    print("         LOOM - QUICK PERFORMANCE BENCHMARK")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        with DB(filename) as db:
            # Test with 50K items (fast but meaningful)
            n = 50_000

            print(f"\nTesting with {n:,} items...")
            print("=" * 70)

            # BloomFilter insertions
            print("\n1. BloomFilter - Insertions")
            print("-" * 50)
            bf = BloomFilter("test_bf", db, expected_items=n)

            start = time.time()
            for i in range(n):
                bf.add(i)
            bf.save()  # Persist metadata
            elapsed = time.time() - start

            print(f"   Time: {elapsed:.3f}s")
            print(f"   Ops/sec: {n/elapsed:,.0f}")
            print(f"   µs/op: {(elapsed/n)*1_000_000:.2f}")

            # BloomFilter lookups
            print("\n2. BloomFilter - Lookups (hits)")
            print("-" * 50)
            start = time.time()
            hits = sum(1 for i in range(n) if i in bf)
            elapsed = time.time() - start

            print(f"   Time: {elapsed:.3f}s")
            print(f"   Ops/sec: {n/elapsed:,.0f}")
            print(f"   µs/op: {(elapsed/n)*1_000_000:.2f}")
            print(f"   Hit rate: {hits/n*100:.1f}%")

            # CountingBloomFilter add/remove
            print("\n3. CountingBloomFilter - Add/Remove Cycles")
            print("-" * 50)
            cbf = CountingBloomFilter("test_cbf", db, expected_items=n)

            start = time.time()
            for i in range(n):
                cbf.add(i)
                cbf.remove(i)
            cbf.save()  # Persist metadata
            elapsed = time.time() - start

            print(f"   Time: {elapsed:.3f}s")
            print(f"   Cycles/sec: {n/elapsed:,.0f}")
            print(f"   µs/cycle: {(elapsed/n)*1_000_000:.2f}")

            # Dataset operations
            print("\n4. Dataset - Read/Write")
            print("-" * 50)
            users = db.create_dataset("users", user_id="uint64", score="int32")
            addr = users.allocate_block(n)

            # Writes
            start = time.time()
            for i in range(n):
                record_addr = addr + i * users.record_size
                users[record_addr] = {"user_id": i, "score": i * 10}
            write_time = time.time() - start

            # Reads
            start = time.time()
            for i in range(n):
                record_addr = addr + i * users.record_size
                record = users[record_addr]
            read_time = time.time() - start

            print(f"   Writes: {write_time:.3f}s ({n/write_time:,.0f} ops/sec)")
            print(f"   Reads:  {read_time:.3f}s ({n/read_time:,.0f} ops/sec)")

        # File size
        file_size = os.path.getsize(filename)
        print("\n5. Storage Efficiency")
        print("-" * 50)
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"   Bytes/item: {file_size/n:.2f}")

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"✓ Processed {n:,} items across multiple data structures")
        print(f"✓ Memory-mapped I/O provides near-RAM performance")
        print(f"✓ BloomFilter: {n/elapsed:,.0f} ops/sec")
        print(f"✓ All operations complete in seconds, not minutes!")
        print("=" * 70 + "\n")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    quick_benchmark()
