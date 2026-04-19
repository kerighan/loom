"""
Phase 2 Tests: Dataset with Prefix-Based Type System

Tests for:
- Prefix-based identification
- Multiple datasets in one file
- Type safety
- Soft deletes
- Field operations
- Clean API
"""

import os
import tempfile
import pytest
import numpy as np
from loom.fileio import ByteFileDB
from loom.dataset import Dataset
from loom.errors import (
    InvalidIdentifierError,
    DeletedRecordError,
    WrongDatasetError,
)


class TestDatasetBasics:
    """Test basic dataset functionality."""

    def test_dataset_creation(self):
        """Test creating a dataset with schema."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Create dataset
            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            assert users.name == "users"
            assert users.identifier == 1
            assert "id" in users.user_schema.names
            assert "name" in users.user_schema.names
            assert "age" in users.user_schema.names

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_identifier_validation(self):
        """Test that identifier must be in valid range."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Valid identifiers
            Dataset("test1", db, 1, field="int32")
            Dataset("test127", db, 127, field="int32")

            # Invalid identifiers
            with pytest.raises(InvalidIdentifierError):
                Dataset("test0", db, 0, field="int32")

            with pytest.raises(InvalidIdentifierError):
                Dataset("test128", db, 128, field="int32")

            with pytest.raises(InvalidIdentifierError):
                Dataset("test_neg", db, -1, field="int32")

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestReadWrite:
    """Test reading and writing records."""

    def test_write_and_read(self):
        """Test basic write and read operations."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            # Allocate and write
            addr = users.allocate_block(1)
            users.write(addr, id=1, name="Alice", age=30)

            # Read back
            record = users.read(addr)
            assert record["id"] == 1
            assert record["name"] == "Alice"
            assert record["age"] == 30

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_multiple_records(self):
        """Test writing and reading multiple records."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            # Write multiple records
            records = [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ]

            addresses = []
            for rec in records:
                addr = users.allocate_block(1)
                users.write(addr, **rec)
                addresses.append(addr)

            # Read back and verify
            for addr, expected in zip(addresses, records):
                actual = users.read(addr)
                assert actual == expected

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_default_values(self):
        """Test that missing fields get default values."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            # Write with missing fields
            addr = users.allocate_block(1)
            users.write(addr, id=1)  # name and age will be defaults

            record = users.read(addr)
            assert record["id"] == 1
            assert record["name"] == ""  # Default for string
            assert record["age"] == 0  # Default for int

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestMultipleDatasets:
    """Test multiple datasets in one file."""

    def test_two_datasets_same_file(self):
        """Test that two datasets can coexist."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Create two datasets
            users = Dataset("users", db, 1, id="uint64", name="U50")
            posts = Dataset("posts", db, 2, id="uint64", title="U100")

            # Write to both
            user_addr = users.allocate_block(1)
            users.write(user_addr, id=1, name="Alice")

            post_addr = posts.allocate_block(1)
            posts.write(post_addr, id=100, title="Hello World")

            # Read from both
            user = users.read(user_addr)
            assert user["id"] == 1
            assert user["name"] == "Alice"

            post = posts.read(post_addr)
            assert post["id"] == 100
            assert post["title"] == "Hello World"

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_type_safety_between_datasets(self):
        """Test that reading with wrong dataset raises error."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")
            posts = Dataset("posts", db, 2, id="uint64", title="U100")

            # Write user record
            addr = users.allocate_block(1)
            users.write(addr, id=1, name="Alice")

            # Try to read with wrong dataset
            with pytest.raises(WrongDatasetError):
                posts.read(addr)

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_many_datasets(self):
        """Test creating many datasets."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            # Create 10 datasets
            datasets = []
            for i in range(1, 11):
                ds = Dataset(f"dataset_{i}", db, i, value="int32")
                datasets.append(ds)

            # Write to each
            addresses = []
            for i, ds in enumerate(datasets):
                addr = ds.allocate_block(1)
                ds.write(addr, value=i * 100)
                addresses.append(addr)

            # Read and verify
            for i, (ds, addr) in enumerate(zip(datasets, addresses)):
                record = ds.read(addr)
                assert record["value"] == i * 100

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestSoftDeletes:
    """Test soft delete functionality."""

    def test_delete_record(self):
        """Test deleting a record."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            # Write and delete
            addr = users.allocate_block(1)
            users.write(addr, id=1, name="Alice")

            assert users.exists(addr)
            assert not users.is_deleted(addr)

            users.delete(addr)

            assert not users.exists(addr)
            assert users.is_deleted(addr)

            # Reading deleted record should fail
            with pytest.raises(DeletedRecordError):
                users.read(addr)

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_exists_check(self):
        """Test exists() method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            addr = users.allocate_block(1)

            # Before writing
            assert not users.exists(addr)

            # After writing
            users.write(addr, id=1, name="Alice")
            assert users.exists(addr)

            # After deleting
            users.delete(addr)
            assert not users.exists(addr)

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestFieldOperations:
    """Test field-level operations."""

    def test_read_single_field(self):
        """Test reading a single field."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            addr = users.allocate_block(1)
            users.write(addr, id=1, name="Alice", age=30)

            # Read individual fields
            assert users.read_field(addr, "id") == 1
            assert users.read_field(addr, "name") == "Alice"
            assert users.read_field(addr, "age") == 30

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_write_single_field(self):
        """Test updating a single field."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            addr = users.allocate_block(1)
            users.write(addr, id=1, name="Alice", age=30)

            # Update single field
            users.write_field(addr, "age", 31)

            # Verify
            assert users.read_field(addr, "age") == 31
            assert users.read_field(addr, "name") == "Alice"  # Unchanged

            # Update another field
            users.write_field(addr, "name", "Alicia")
            assert users.read_field(addr, "name") == "Alicia"

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_invalid_field_name(self):
        """Test that invalid field names raise errors."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            addr = users.allocate_block(1)
            users.write(addr, id=1, name="Alice")

            with pytest.raises(ValueError, match="not in schema"):
                users.read_field(addr, "nonexistent")

            with pytest.raises(ValueError, match="not in schema"):
                users.write_field(addr, "nonexistent", "value")

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestAllocation:
    """Test block allocation."""

    def test_allocate_single_record(self):
        """Test allocating space for one record."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            addr = users.allocate_block(1)
            assert addr >= db.header_size  # Should be after header

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_allocate_multiple_records(self):
        """Test allocating space for multiple records."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            # Allocate block for 10 records
            addr = users.allocate_block(10)

            # Write to each position
            for i in range(10):
                offset = addr + (i * users.record_size)
                users.write(offset, id=i, name=f"User{i}")

            # Read back
            for i in range(10):
                offset = addr + (i * users.record_size)
                record = users.read(offset)
                assert record["id"] == i
                assert record["name"] == f"User{i}"

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDataTypes:
    """Test various numpy data types."""

    def test_numeric_types(self):
        """Test various numeric types."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            ds = Dataset(
                "numbers",
                db,
                1,
                u8="uint8",
                u16="uint16",
                u32="uint32",
                u64="uint64",
                i8="int8",
                i16="int16",
                i32="int32",
                i64="int64",
                f32="float32",
                f64="float64",
            )

            addr = ds.allocate_block(1)
            ds.write(
                addr,
                u8=255,
                u16=65535,
                u32=4294967295,
                u64=18446744073709551615,
                i8=-128,
                i16=-32768,
                i32=-2147483648,
                i64=-9223372036854775808,
                f32=3.14,
                f64=2.718281828,
            )

            record = ds.read(addr)
            assert record["u8"] == 255
            assert record["i8"] == -128
            assert abs(record["f32"] - 3.14) < 0.01
            assert abs(record["f64"] - 2.718281828) < 0.000001

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_string_types(self):
        """Test string fields."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            ds = Dataset("strings", db, 1, short="U10", long="U100")

            addr = ds.allocate_block(1)
            ds.write(addr, short="Hello", long="This is a longer string")

            record = ds.read(addr)
            assert record["short"] == "Hello"
            assert record["long"] == "This is a longer string"

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            repr_str = repr(users)
            assert "users" in repr_str
            assert "id=1" in repr_str

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestOperatorOverloading:
    """Test pythonic operator overloading."""

    def test_getitem_setitem(self):
        """Test dict-like read/write with [] operator."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            addr = users.allocate_block(1)

            # Write using []
            users[addr] = {"id": 1, "name": "Alice", "age": 30}

            # Read using []
            record = users[addr]
            assert record["id"] == 1
            assert record["name"] == "Alice"
            assert record["age"] == 30

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_delitem(self):
        """Test deletion with del operator."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            addr = users.allocate_block(1)
            users[addr] = {"id": 1, "name": "Alice"}

            assert addr in users

            # Delete using del
            del users[addr]

            assert addr not in users
            assert users.is_deleted(addr)

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_contains(self):
        """Test 'in' operator for existence check."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            addr = users.allocate_block(1)

            # Before writing
            assert addr not in users

            # After writing
            users[addr] = {"id": 1, "name": "Alice"}
            assert addr in users

            # After deleting
            del users[addr]
            assert addr not in users

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_setitem_type_check(self):
        """Test that setitem requires a dict."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50")

            addr = users.allocate_block(1)

            # Should fail with non-dict
            with pytest.raises(TypeError, match="must be a dict"):
                users[addr] = "not a dict"

            with pytest.raises(TypeError, match="must be a dict"):
                users[addr] = [1, 2, 3]

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_pythonic_workflow(self):
        """Test complete pythonic workflow."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name

        try:
            db = ByteFileDB(filename)
            db.open()

            users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

            # Allocate block for 3 users
            addr = users.allocate_block(3)

            # Write using dict-like syntax
            users[addr] = {"id": 1, "name": "Alice", "age": 30}
            users[addr + users.record_size] = {"id": 2, "name": "Bob", "age": 25}
            users[addr + 2 * users.record_size] = {
                "id": 3,
                "name": "Charlie",
                "age": 35,
            }

            # Read using dict-like syntax
            assert users[addr]["name"] == "Alice"
            assert users[addr + users.record_size]["name"] == "Bob"
            assert users[addr + 2 * users.record_size]["name"] == "Charlie"

            # Check existence
            assert addr in users
            assert (addr + users.record_size) in users

            # Delete one
            del users[addr + users.record_size]
            assert (addr + users.record_size) not in users

            # Others still exist
            assert addr in users
            assert (addr + 2 * users.record_size) in users

            db.close()
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
