"""
Phase 1 Tests: ByteFileDB Header Management and Allocation

Tests for:
- Header initialization and persistence
- Metadata storage and retrieval
- Allocation tracking
- Backward compatibility with existing functionality
- Crash recovery with header data
"""

import os
import tempfile
import pytest
import numpy as np
from loom.fileio import ByteFileDB


class TestHeaderManagement:
    """Test header metadata storage and retrieval."""

    def test_header_initialization(self):
        """Test that header is properly initialized on first open."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Check that header is initialized
            assert db.has_header_field("_is_initialized")
            assert db.get_header_field("_is_initialized") is True
            assert db.has_header_field("_allocation_index")
            assert db.get_header_field("_allocation_index") == db.header_size

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_header_persistence(self):
        """Test that header data persists across open/close cycles."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Write some metadata
            db = ByteFileDB(filename)
            db.open()
            db.set_header_field("test_string", "hello world")
            db.set_header_field("test_int", 42)
            db.set_header_field("test_list", [1, 2, 3])
            db.set_header_field("test_dict", {"a": 1, "b": 2})
            db.close()

            # Reopen and verify
            db = ByteFileDB(filename)
            db.open()
            assert db.get_header_field("test_string") == "hello world"
            assert db.get_header_field("test_int") == 42
            assert db.get_header_field("test_list") == [1, 2, 3]
            assert db.get_header_field("test_dict") == {"a": 1, "b": 2}
            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_header_field_operations(self):
        """Test get, set, has, and delete operations."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Test set and get
            db.set_header_field("key1", "value1")
            assert db.get_header_field("key1") == "value1"

            # Test has
            assert db.has_header_field("key1")
            assert not db.has_header_field("nonexistent")

            # Test default value
            assert db.get_header_field("nonexistent", "default") == "default"

            # Test delete
            db.delete_header_field("key1")
            assert not db.has_header_field("key1")
            assert db.get_header_field("key1") is None

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_header_with_complex_types(self):
        """Test storing complex Python objects in header."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Store numpy array
            arr = np.array([1, 2, 3, 4, 5])
            db.set_header_field("numpy_array", arr)

            # Store nested structure
            nested = {
                "datasets": ["users", "posts", "comments"],
                "counters": {"users": 100, "posts": 500},
                "metadata": {"version": "1.0", "created": "2025-11-28"},
            }
            db.set_header_field("database_info", nested)

            db.close()

            # Reopen and verify
            db = ByteFileDB(filename)
            db.open()

            retrieved_arr = db.get_header_field("numpy_array")
            assert np.array_equal(retrieved_arr, arr)

            retrieved_nested = db.get_header_field("database_info")
            assert retrieved_nested == nested

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestAllocationManagement:
    """Test memory allocation and tracking."""

    def test_basic_allocation(self):
        """Test basic allocation functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename, header_size=4096)
            db.open()

            # First allocation should start after header
            addr1 = db.allocate(100)
            assert addr1 == 4096

            # Second allocation should be contiguous
            addr2 = db.allocate(200)
            assert addr2 == 4096 + 100

            # Third allocation
            addr3 = db.allocate(50)
            assert addr3 == 4096 + 100 + 200

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_allocation_persistence(self):
        """Test that allocation index persists across sessions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # First session: allocate some space
            db = ByteFileDB(filename, header_size=4096)
            db.open()
            addr1 = db.allocate(1000)
            addr2 = db.allocate(2000)
            db.close()

            # Second session: continue allocating
            db = ByteFileDB(filename, header_size=4096)
            db.open()
            addr3 = db.allocate(500)

            # Should continue from where we left off
            assert addr3 == 4096 + 1000 + 2000

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_allocation_with_file_expansion(self):
        """Test that allocation properly expands the file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename, initial_size=1024, header_size=512)
            db.open()

            initial_size = db.get_file_size()

            # Allocate more than initial size
            large_size = 10000
            addr = db.allocate(large_size)

            # File should have expanded
            assert db.get_file_size() >= addr + large_size

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_space_tracking(self):
        """Test used/free space tracking."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename, header_size=4096)
            db.open()

            # Initially, only header is used
            assert db.get_used_space() == 4096

            # Allocate some space
            db.allocate(1000)
            assert db.get_used_space() == 4096 + 1000

            db.allocate(500)
            assert db.get_used_space() == 4096 + 1000 + 500

            # Free space should be file size - used space
            assert db.get_free_space() == db.get_file_size() - db.get_used_space()

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestBackwardCompatibility:
    """Test that existing functionality still works."""

    def test_basic_read_write(self):
        """Test that basic read/write operations still work."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Write some data (avoiding header region)
            address = db.header_size + 100
            data = b"Hello, World!"
            db.write(address, data)

            # Read it back
            read_data = db.read(address, len(data))
            assert read_data == data

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_transaction_system(self):
        """Test that WAL transactions still work."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Use transaction system
            address = db.header_size + 100
            writes = [
                (address, b"data1"),
                (address + 10, b"data2"),
                (address + 20, b"data3"),
            ]
            db.transaction(writes)

            # Verify writes
            assert db.read(address, 5) == b"data1"
            assert db.read(address + 10, 5) == b"data2"
            assert db.read(address + 20, 5) == b"data3"

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_crash_recovery_with_header(self):
        """Test that crash recovery works with header system."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Create a transaction but don't commit
            db = ByteFileDB(filename)
            db.open()

            # Set some header data
            db.set_header_field("test_key", "test_value")

            address = db.header_size + 100
            # Write a committed WAL entry using the internal API
            with open(db.log_filename, "ab") as log:
                db.log_write(log, address, b"pending_data")
                db.log_commit(log)

            # Simulate crash (don't apply writes, just close)
            db.close()

            # Reopen - should recover from log
            db = ByteFileDB(filename)
            db.open()

            # Header data should be intact
            assert db.get_header_field("test_key") == "test_value"

            # Transaction should have been applied
            assert db.read(address, 12) == b"pending_data"

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)
            log_file = filename + ".log"
            if os.path.exists(log_file):
                os.remove(log_file)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_allocation_before_open(self):
        """Test that allocation fails if DB is not open."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)

            with pytest.raises(AssertionError):
                db.allocate(100)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_header_operations_before_open(self):
        """Test that header operations fail if DB is not open."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)

            with pytest.raises(AssertionError):
                db.set_header_field("key", "value")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_multiple_open_close_cycles(self):
        """Test multiple open/close cycles."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)

            for i in range(5):
                db.open()
                db.set_header_field(f"key_{i}", f"value_{i}")
                addr = db.allocate(100)
                db.write(addr, f"data_{i}".encode())
                db.close()

            # Verify all data persisted
            db.open()
            for i in range(5):
                assert db.get_header_field(f"key_{i}") == f"value_{i}"
            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_custom_header_size(self):
        """Test using different header sizes."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            # Small header
            db = ByteFileDB(filename, header_size=1024)
            db.open()
            addr1 = db.allocate(100)
            assert addr1 == 1024
            db.close()

            # Large header
            db = ByteFileDB(filename, header_size=8192)
            db.open()
            addr2 = db.allocate(100)
            # Should continue from previous allocation
            assert addr2 == 1024 + 100
            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_realistic_usage_scenario(self):
        """Test a realistic usage scenario with datasets."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename, header_size=4096)
            db.open()

            # Simulate creating multiple datasets
            datasets = {}
            for i, name in enumerate(["users", "posts", "comments"]):
                # Allocate space for dataset
                addr = db.allocate(1000)
                datasets[name] = {"address": addr, "size": 1000, "record_count": 0}

            # Store dataset registry in header
            db.set_header_field("datasets", datasets)

            # Write some data to each dataset
            for name, info in datasets.items():
                data = f"{name}_data".encode()
                db.write(info["address"], data)

            db.close()

            # Reopen and verify
            db = ByteFileDB(filename, header_size=4096)
            db.open()

            retrieved_datasets = db.get_header_field("datasets")
            assert retrieved_datasets == datasets

            # Verify data
            for name, info in retrieved_datasets.items():
                expected_data = f"{name}_data".encode()
                actual_data = db.read(info["address"], len(expected_data))
                assert actual_data == expected_data

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
