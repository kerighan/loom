"""
Tests for blob storage.
"""

import os
import tempfile
import pytest
from loom.database import DB


class TestBlobBasics:
    """Test basic blob operations."""

    def test_write_read_blob(self):
        """Test writing and reading a blob."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                data = b"Hello, World!"
                offset, n_slots = db.write_blob(data)

                assert offset > 0
                assert n_slots >= 1

                result = db.read_blob(offset)
                assert result == data

    def test_write_read_large_blob(self):
        """Test writing and reading a large blob."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            # Use no compression to test multi-slot storage
            with DB(db_path, blob_compression=None) as db:
                # 10KB of data (uncompressed needs ~157 slots at 64 bytes each)
                data = b"x" * 10000
                offset, n_slots = db.write_blob(data)

                result = db.read_blob(offset)
                assert result == data
                assert n_slots > 1  # Should span multiple slots

    def test_compression(self):
        """Test that compression reduces size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, blob_compression="brotli") as db:
                # Highly compressible data
                data = b"a" * 10000
                offset, n_slots = db.write_blob(data)

                result = db.read_blob(offset)
                assert result == data

                # With brotli, 10000 'a's should compress to very few slots
                # (64 bytes per slot, so uncompressed would need ~157 slots)
                assert n_slots < 50  # Should be much smaller due to compression

    def test_no_compression(self):
        """Test blob storage without compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, blob_compression=None) as db:
                data = b"Hello, World!"
                offset, n_slots = db.write_blob(data)

                result = db.read_blob(offset)
                assert result == data


class TestBlobFreelist:
    """Test blob freelist and space reuse."""

    def test_delete_and_reuse(self):
        """Test that deleted blob space is reused."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, blob_compression=None) as db:
                # Write first blob
                data1 = b"First blob data"
                offset1, n_slots1 = db.write_blob(data1)

                # Write second blob
                data2 = b"Second blob data"
                offset2, n_slots2 = db.write_blob(data2)

                # Delete first blob
                db.delete_blob(offset1, n_slots1)

                # Write third blob (should reuse first blob's space)
                data3 = b"Third blob"  # Smaller than first
                offset3, n_slots3 = db.write_blob(data3)

                # Should reuse the freed space
                assert offset3 == offset1

                # Verify data integrity
                assert db.read_blob(offset2) == data2
                assert db.read_blob(offset3) == data3

    def test_freelist_persistence(self):
        """Test that freelist persists across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Session 1: Write and delete
            with DB(db_path, blob_compression=None) as db:
                data1 = b"First blob"
                offset1, n_slots1 = db.write_blob(data1)
                db.delete_blob(offset1, n_slots1)

            # Session 2: New write should reuse space
            with DB(db_path, blob_compression=None) as db:
                data2 = b"Second"  # Smaller
                offset2, n_slots2 = db.write_blob(data2)

                # Should reuse the freed space from session 1
                assert offset2 == offset1


class TestBlobDataset:
    """Test blob fields in datasets."""

    def test_dataset_with_blob_field(self):
        """Test creating a dataset with a blob field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                # Create dataset with blob field
                docs = db.create_dataset(
                    "documents", id="uint64", title="U100", content="blob"
                )

                # Check that blob field is tracked
                assert "content" in docs._blob_fields

                # Write a record with blob
                content_data = b"This is the document content"
                blob_ref = db.write_blob(content_data)

                addr = docs.allocate_block(1)
                docs.write(addr, id=1, title="Test Doc", content=blob_ref)

                # Read back
                record = docs.read(addr)
                assert record["id"] == 1
                assert record["title"] == "Test Doc"

                # content should be (offset, n_slots) tuple
                assert record["content"] is not None
                offset, n_slots = record["content"]

                # Read actual blob data
                actual_content = db.read_blob(offset)
                assert actual_content == content_data

    def test_null_blob(self):
        """Test null blob field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path) as db:
                docs = db.create_dataset("documents", id="uint64", content="blob")

                # Write record without blob
                addr = docs.allocate_block(1)
                docs.write(addr, id=1)  # No content

                record = docs.read(addr)
                assert record["id"] == 1
                assert record["content"] is None  # Null blob


class TestBlobStats:
    """Test blob storage statistics."""

    def test_get_stats(self):
        """Test getting blob store statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with DB(db_path, blob_compression="brotli") as db:
                # Write some blobs
                db.write_blob(b"First")
                offset, n_slots = db.write_blob(b"Second")
                db.write_blob(b"Third")

                # Delete one
                db.delete_blob(offset, n_slots)

                stats = db.blob_store.get_stats()
                assert stats["compression"] == "brotli"
                assert stats["slot_size"] == 64
                assert stats["free_slots"] == n_slots
                assert stats["freelist_entries"] == 1
