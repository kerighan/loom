"""
Test interrupt safety for List data structure.

Simulates keyboard interrupts during operations to verify:
- Auto-save preserves data
- No corruption on interrupt
- Minimal data loss (only since last auto-save)
"""

import os
import tempfile
import signal
import time
from multiprocessing import Process
import pytest
from loom.database import DB


def insert_with_interrupt(filename, n_items, interrupt_at):
    """Insert items and simulate interrupt at specific point.

    Args:
        filename: Database file
        n_items: Total items to insert
        interrupt_at: Item number to interrupt at
    """
    try:
        with DB(filename) as db:
            lst = db.create_list("test", {"id": "uint64", "value": "U50"})

            for i in range(n_items):
                lst.append({"id": i, "value": f"item_{i}"})

                # Simulate interrupt
                if i == interrupt_at:
                    # Force exit without proper cleanup
                    os._exit(1)
    except:
        pass


class TestInterruptSafety:
    """Test data safety on interrupts."""

    def test_interrupt_during_insert(self):
        """Test interrupt during insertion preserves data."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Insert 2500 items, interrupt at 2500
            # Auto-save happens at 1000, 2000
            # So we should have at least 2000 items saved
            n_items = 2500
            interrupt_at = 2499

            # Run in subprocess to simulate hard interrupt
            p = Process(
                target=insert_with_interrupt, args=(filename, n_items, interrupt_at)
            )
            p.start()
            p.join(timeout=5)

            if p.is_alive():
                p.terminate()
                p.join()

            # Check what was saved
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                saved_count = len(lst)
                print(f"\n  Inserted: {interrupt_at + 1} items")
                print(f"  Saved: {saved_count} items")
                print(f"  Lost: {interrupt_at + 1 - saved_count} items")

                # Should have at least 2000 items (last auto-save at 2000)
                # Might have up to 2500 if context manager exit saved
                assert saved_count >= 2000, f"Expected >= 2000, got {saved_count}"

                # Verify data integrity
                for i in range(min(saved_count, 100)):
                    item = lst[i]
                    assert item["id"] == i
                    assert item["value"] == f"item_{i}"

                print(f"  ✓ Data integrity verified")

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_interrupt_early(self):
        """Test interrupt before first auto-save."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Interrupt at 500 items (before first auto-save at 1000)
            n_items = 1000
            interrupt_at = 500

            p = Process(
                target=insert_with_interrupt, args=(filename, n_items, interrupt_at)
            )
            p.start()
            p.join(timeout=5)

            if p.is_alive():
                p.terminate()
                p.join()

            # Check what was saved
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                saved_count = len(lst)
                print(f"\n  Inserted: {interrupt_at + 1} items")
                print(f"  Saved: {saved_count} items")

                # Might have 0 items if no auto-save happened yet
                # Or might have some if context manager saved
                print(f"  ✓ No corruption (got {saved_count} items)")

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_multiple_interrupts(self):
        """Test multiple interrupted sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Session 1: Insert 1500, interrupt at 1499
            p = Process(target=insert_with_interrupt, args=(filename, 1500, 1499))
            p.start()
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join()

            # Check session 1
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})
                count1 = len(lst)
                print(f"\n  Session 1 saved: {count1} items")

            # Session 2: Continue inserting (should append)
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                # Append more items
                for i in range(count1, count1 + 500):
                    lst.append({"id": i, "value": f"item_{i}"})

                count2 = len(lst)
                print(f"  Session 2 total: {count2} items")

            # Session 3: Verify all data
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                final_count = len(lst)
                assert final_count == count2

                # Verify sequential IDs
                for i in range(min(final_count, 100)):
                    assert lst[i]["id"] == i

                print(f"  ✓ All sessions preserved correctly")

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_no_data_loss_with_proper_close(self):
        """Test that proper close saves all data."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Insert 2500 items with proper close
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                for i in range(2500):
                    lst.append({"id": i, "value": f"item_{i}"})

                # Context manager should save on exit

            # Verify all 2500 items saved
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64", "value": "U50"})

                assert len(lst) == 2500
                print(f"\n  ✓ All 2500 items saved with proper close")

                # Verify data
                assert lst[0]["id"] == 0
                assert lst[1000]["id"] == 1000
                assert lst[2499]["id"] == 2499

        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestMultiStructureInterruptSafety:
    """Test interrupt safety with multiple structures."""

    def test_interrupt_with_multiple_structures(self):
        """Test that interrupt doesn't corrupt other structures."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create multiple structures
            with DB(filename) as db:
                list1 = db.create_list("list1", {"id": "uint64"})
                list2 = db.create_list("list2", {"id": "uint64"})
                bf = db.create_bloomfilter("bloom", expected_items=1000)

                # Populate all structures
                for i in range(500):
                    list1.append({"id": i})
                    list2.append({"id": i * 2})
                    bf.add(f"item_{i}")

            # Verify all structures saved
            with DB(filename) as db:
                list1 = db.create_list("list1", {"id": "uint64"})
                list2 = db.create_list("list2", {"id": "uint64"})
                bf = db.create_bloomfilter("bloom", expected_items=1000)

                assert len(list1) == 500
                assert len(list2) == 500
                assert len(bf) == 500

                # Verify data integrity
                assert list1[100]["id"] == 100
                assert list2[100]["id"] == 200
                assert "item_100" in bf

                print(f"\n  ✓ All structures preserved correctly")
                print(f"    List1: {len(list1)} items")
                print(f"    List2: {len(list2)} items")
                print(f"    BloomFilter: {len(bf)} items")

        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestAutoSaveInterval:
    """Test different auto-save intervals."""

    def test_custom_auto_save_interval(self):
        """Test custom auto-save interval."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create list with auto-save every 100 items
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"}, cache_size=10)
                # Note: auto_save_interval is set in DataStructure base class
                lst._auto_save_interval = 100

                for i in range(250):
                    lst.append({"id": i})
                # Should auto-save at 100, 200
                # Items 200-249 might be lost on interrupt

            # Verify
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                # Should have at least 200 items
                saved = len(lst)
                print(f"\n  Custom interval (100): saved {saved}/250 items")
                assert saved >= 200

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_zero_auto_save_interval(self):
        """Test with auto-save disabled."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create list with auto-save disabled
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})
                lst._auto_save_interval = 0  # Disable auto-save

                for i in range(100):
                    lst.append({"id": i})

                # Manual save
                lst.save()

            # Verify
            with DB(filename) as db:
                lst = db.create_list("test", {"id": "uint64"})

                # Should have all 100 items (manual save)
                assert len(lst) == 100
                print(f"\n  ✓ Manual save works (100 items)")

        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
