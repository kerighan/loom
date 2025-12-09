"""
Phase 1 Demo: Header Management and Allocation

Demonstrates the new features added in Phase 1:
- Header metadata storage
- Automatic allocation tracking
- Backward compatibility with existing code
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loom.fileio import ByteFileDB


def demo_header_management():
    """Demonstrate header metadata storage."""
    print("=" * 60)
    print("DEMO 1: Header Management")
    print("=" * 60)

    filename = "/tmp/demo_phase1.db"

    # Clean up if exists
    if os.path.exists(filename):
        os.remove(filename)

    # Create database and store metadata
    print("\n1. Creating database and storing metadata...")
    db = ByteFileDB(filename, header_size=4096)
    db.open()

    # Store various types of metadata
    db.set_header_field("database_name", "MyDatabase")
    db.set_header_field("version", "1.0.0")
    db.set_header_field("created_at", "2025-11-28")
    db.set_header_field("datasets", ["users", "posts", "comments"])
    db.set_header_field("record_count", 0)

    print(f"   Database name: {db.get_header_field('database_name')}")
    print(f"   Version: {db.get_header_field('version')}")
    print(f"   Datasets: {db.get_header_field('datasets')}")

    db.close()

    # Reopen and verify persistence
    print("\n2. Reopening database to verify persistence...")
    db = ByteFileDB(filename, header_size=4096)
    db.open()

    print(f"   Database name: {db.get_header_field('database_name')}")
    print(f"   Version: {db.get_header_field('version')}")
    print(f"   Created at: {db.get_header_field('created_at')}")
    print(f"   Datasets: {db.get_header_field('datasets')}")
    print(f"   Record count: {db.get_header_field('record_count')}")

    db.close()
    print("\n✓ Header data persisted successfully!")


def demo_allocation_tracking():
    """Demonstrate automatic allocation tracking."""
    print("\n" + "=" * 60)
    print("DEMO 2: Allocation Tracking")
    print("=" * 60)

    filename = "/tmp/demo_phase1.db"

    # Clean up if exists
    if os.path.exists(filename):
        os.remove(filename)

    db = ByteFileDB(filename, header_size=4096)
    db.open()

    print("\n1. Initial state:")
    print(f"   File size: {db.get_file_size()} bytes")
    print(f"   Used space: {db.get_used_space()} bytes")
    print(f"   Free space: {db.get_free_space()} bytes")
    print(f"   Allocation index: {db.get_allocation_index()}")

    # Allocate some blocks
    print("\n2. Allocating blocks...")
    addr1 = db.allocate(1000)
    print(f"   Allocated 1000 bytes at address: {addr1}")

    addr2 = db.allocate(2000)
    print(f"   Allocated 2000 bytes at address: {addr2}")

    addr3 = db.allocate(500)
    print(f"   Allocated 500 bytes at address: {addr3}")

    print("\n3. After allocations:")
    print(f"   File size: {db.get_file_size()} bytes")
    print(f"   Used space: {db.get_used_space()} bytes")
    print(f"   Free space: {db.get_free_space()} bytes")
    print(f"   Allocation index: {db.get_allocation_index()}")

    # Write data to allocated blocks
    print("\n4. Writing data to allocated blocks...")
    db.write(addr1, b"Block 1 data")
    db.write(addr2, b"Block 2 data")
    db.write(addr3, b"Block 3 data")

    # Read back
    print(f"   Block 1: {db.read(addr1, 12)}")
    print(f"   Block 2: {db.read(addr2, 12)}")
    print(f"   Block 3: {db.read(addr3, 12)}")

    db.close()

    # Reopen and continue allocating
    print("\n5. Reopening and continuing allocation...")
    db = ByteFileDB(filename, header_size=4096)
    db.open()

    addr4 = db.allocate(300)
    print(f"   Allocated 300 bytes at address: {addr4}")
    print(f"   Allocation continues from: {db.get_allocation_index()}")

    db.close()
    print("\n✓ Allocation tracking works across sessions!")


def demo_combined_usage():
    """Demonstrate using header and allocation together."""
    print("\n" + "=" * 60)
    print("DEMO 3: Combined Usage (Simulating Dataset Registry)")
    print("=" * 60)

    filename = "/tmp/demo_phase1.db"

    # Clean up if exists
    if os.path.exists(filename):
        os.remove(filename)

    db = ByteFileDB(filename, header_size=4096)
    db.open()

    # Simulate creating multiple datasets
    print("\n1. Creating dataset registry...")
    datasets = {}

    for name in ["users", "posts", "comments"]:
        # Allocate space for each dataset
        addr = db.allocate(1000)
        datasets[name] = {
            "address": addr,
            "size": 1000,
            "record_count": 0,
            "schema": f"{name}_schema",
        }
        print(f"   Dataset '{name}' allocated at address {addr}")

    # Store registry in header
    db.set_header_field("dataset_registry", datasets)
    db.set_header_field("dataset_count", len(datasets))

    print("\n2. Writing sample data to datasets...")
    for name, info in datasets.items():
        sample_data = f"{name}_sample_data".encode()
        db.write(info["address"], sample_data)
        print(f"   Wrote to '{name}': {sample_data}")

    db.close()

    # Reopen and access datasets
    print("\n3. Reopening and accessing datasets...")
    db = ByteFileDB(filename, header_size=4096)
    db.open()

    registry = db.get_header_field("dataset_registry")
    count = db.get_header_field("dataset_count")

    print(f"   Found {count} datasets in registry")

    for name, info in registry.items():
        data = db.read(info["address"], 20)
        print(f"   Dataset '{name}': {data}")

    db.close()
    print("\n✓ Dataset registry working perfectly!")


def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    print("\n" + "=" * 60)
    print("DEMO 4: Backward Compatibility")
    print("=" * 60)

    filename = "/tmp/demo_phase1.db"

    # Clean up if exists
    if os.path.exists(filename):
        os.remove(filename)

    db = ByteFileDB(filename)
    db.open()

    print("\n1. Using old-style manual addressing (still works!)...")

    # Manual address calculation (old way)
    address = db.header_size + 1000  # Skip header manually
    data = b"Old style write"
    db.write(address, data)

    read_data = db.read(address, len(data))
    print(f"   Wrote and read: {read_data}")

    print("\n2. Using new allocation system (recommended)...")

    # New way with automatic allocation
    addr = db.allocate(100)
    new_data = b"New style write"
    db.write(addr, new_data)

    read_new = db.read(addr, len(new_data))
    print(f"   Wrote and read: {read_new}")

    print("\n3. Using transactions (still works!)...")

    addr1 = db.allocate(50)
    addr2 = db.allocate(50)

    writes = [(addr1, b"Transaction write 1"), (addr2, b"Transaction write 2")]
    db.transaction(writes)

    print(f"   Transaction 1: {db.read(addr1, 19)}")
    print(f"   Transaction 2: {db.read(addr2, 19)}")

    db.close()
    print("\n✓ All existing functionality still works!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 1: ByteFileDB Enhancement Demo")
    print("=" * 60)

    demo_header_management()
    demo_allocation_tracking()
    demo_combined_usage()
    demo_backward_compatibility()

    print("\n" + "=" * 60)
    print("All demos completed successfully! 🎉")
    print("=" * 60)
    print("\nPhase 1 adds:")
    print("  ✓ Header metadata storage (any picklable Python object)")
    print("  ✓ Automatic allocation tracking")
    print("  ✓ Space usage monitoring")
    print("  ✓ Full backward compatibility")
    print("  ✓ Crash recovery with header data")
    print("\nReady for Phase 2: Prefix-based type system!")
    print("=" * 60 + "\n")
