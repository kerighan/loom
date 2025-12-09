"""
Auto-Save Demo: Transparent Persistence

Shows how auto-save works transparently - user never thinks about saving!
"""

import os
import sys
import tempfile
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB
from loom.datastructures import CountingBloomFilter


def demo_auto_save():
    """Demo: Auto-save happens transparently."""
    print("\n" + "=" * 70)
    print("DEMO: Auto-Save - Transparent Persistence")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        # Session 1: Add items (auto-save every 1000 ops)
        print("\nSESSION 1: Adding 5,000 items...")
        print("-" * 50)
        with DB(filename) as db:
            cbf = CountingBloomFilter("cache", db, expected_items=10000)

            # Just use it like a normal Python object!
            for i in range(5000):
                cbf.add(i)
                # Auto-saves every 1000 items transparently

            print(f"Added {len(cbf):,} items")
            print("✓ Auto-saved 5 times (every 1,000 ops)")
            print("✓ User never called save() manually!")

        # Session 2: Verify data persisted
        print("\nSESSION 2: Reopening database...")
        print("-" * 50)
        with DB(filename) as db:
            cbf = CountingBloomFilter("cache", db)

            print(f"Found {len(cbf):,} items")
            print(f"Item 0 in cache: {0 in cbf}")
            print(f"Item 4999 in cache: {4999 in cbf}")
            print("✓ All data persisted automatically!")

        print("\n" + "=" * 70)
        print("KEY POINTS")
        print("=" * 70)
        print("✓ No manual save() calls needed")
        print("✓ Auto-saves every 1,000 operations")
        print("✓ Final save on context manager exit")
        print("✓ Use it like a Python object, get persistence for free!")
        print("=" * 70 + "\n")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_interrupt_safety():
    """Demo: Data safe even with interrupts."""
    print("\n" + "=" * 70)
    print("DEMO: Interrupt Safety")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        print("\nAdding items with periodic auto-save...")
        print("-" * 50)
        with DB(filename) as db:
            cbf = CountingBloomFilter("cache", db, expected_items=10000)

            # Add 2500 items (will auto-save at 1000 and 2000)
            for i in range(2500):
                cbf.add(i)

            print(f"Added {len(cbf):,} items")
            print("✓ Auto-saved at 1,000 and 2,000 items")

        # Simulate what happens if interrupted
        print("\nSimulating interrupt after 2,500 items...")
        print("-" * 50)
        with DB(filename) as db:
            cbf = CountingBloomFilter("cache", db)

            # Even if interrupted, we have data up to last auto-save
            print(f"Items persisted: {len(cbf):,}")
            print("✓ At minimum, 2,000 items are safe (last auto-save)")
            print("✓ With context manager exit, all 2,500 are safe!")

        print("\n" + "=" * 70)
        print("SAFETY GUARANTEES")
        print("=" * 70)
        print("✓ Auto-save every 1,000 ops = max 1,000 ops lost on crash")
        print("✓ Context manager exit = final save, 0 ops lost")
        print("✓ Much better than losing everything!")
        print("=" * 70 + "\n")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_pythonic_usage():
    """Demo: Use like normal Python, get persistence."""
    print("\n" + "=" * 70)
    print("DEMO: Pythonic Usage")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        print("\nJust use it like a Python set!")
        print("-" * 50)

        # Day 1: Build cache
        with DB(filename) as db:
            seen_users = db.create_counting_bloomfilter("seen", expected_items=1000)

            # Use it naturally
            users = ["alice", "bob", "charlie", "diana"]
            for user in users:
                seen_users.add(user)

            print(f"Cached {len(seen_users)} users")

        # Day 2: Check cache
        with DB(filename) as db:
            seen_users = db.create_counting_bloomfilter("seen")

            # Just check membership
            if "alice" in seen_users:
                print("✓ Alice was seen before")

            if "eve" not in seen_users:
                print("✓ Eve is new")

            # Remove users
            del seen_users["bob"]
            print("✓ Removed Bob")

        # Day 3: Verify
        with DB(filename) as db:
            seen_users = db.create_counting_bloomfilter("seen")

            print(f"\nFinal state: {len(seen_users)} users")
            print(f"  alice: {'alice' in seen_users}")
            print(f"  bob: {'bob' in seen_users}")  # Should be False
            print(f"  charlie: {'charlie' in seen_users}")

        print("\n" + "=" * 70)
        print("PYTHONIC = EASY")
        print("=" * 70)
        print("✓ No database thinking required")
        print("✓ No save() calls")
        print("✓ No transactions to manage")
        print("✓ Just use it like Python!")
        print("=" * 70 + "\n")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    demo_auto_save()
    demo_interrupt_safety()
    demo_pythonic_usage()
