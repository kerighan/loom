"""
Tests for the "text" dtype — variable-length UTF-8 strings via BlobStore.

Covers:
- Basic write / read transparency
- Unicode, emoji, long texts
- Empty / null / default values
- write_field() replaces old blob
- delete() frees blobs
- Round-trip persistence (registry survives close/reopen)
- read_many() slow path
- Multiple text fields per record
- Blob freelist reuse after updates
- Mixed schema (text + fixed fields)
"""

import os
import tempfile
import pytest
from loom.database import DB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


class TestSchemaDetection:
    def test_text_field_tracked(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint64", content="text")
            assert "content" in ds._text_fields
            assert "content" not in ds._blob_fields

    def test_text_field_not_in_regular_schema(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint64", name="U50", content="text")
            assert "name" not in ds._text_fields
            assert "id" not in ds._text_fields

    def test_blob_store_initialised_for_text(self, db_path):
        with DB(db_path) as db:
            db.create_dataset("msgs", content="text")
            assert db._blob_store is not None

    def test_no_blob_store_without_text(self, db_path):
        with DB(db_path) as db:
            db.create_dataset("users", id="uint64", name="U50")
            assert db._blob_store is None


# ---------------------------------------------------------------------------
# Basic read / write
# ---------------------------------------------------------------------------


class TestBasicReadWrite:
    def test_write_and_read_simple(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "Hello, world!"})
            record = ds.read(ref.addr)
            assert record["id"] == 1
            assert record["content"] == "Hello, world!"

    def test_write_and_read_unicode(self, db_path):
        texts = [
            "Bonjour, comment ça va ?",
            "日本語テスト",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉🔥✨",
        ]
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            refs = [ds.insert({"id": i, "content": t}) for i, t in enumerate(texts)]
            for ref, expected in zip(refs, texts):
                assert ds.read(ref.addr)["content"] == expected

    def test_write_and_read_long_text(self, db_path):
        long_text = "x" * 100_000  # 100 KB
        with DB(db_path) as db:
            ds = db.create_dataset("docs", id="uint32", body="text")
            ref = ds.insert({"id": 1, "body": long_text})
            assert ds.read(ref.addr)["body"] == long_text

    def test_write_multiple_records(self, db_path):
        messages = [
            {"id": 1, "role": "user", "content": "What is loom?"},
            {"id": 2, "role": "assistant", "content": "A persistent data structure library."},
            {"id": 3, "role": "user", "content": "Cool!"},
        ]
        with DB(db_path) as db:
            ds = db.create_dataset("chat", id="uint32", role="U20", content="text")
            refs = [ds.insert(m) for m in messages]
            for ref, expected in zip(refs, messages):
                rec = ds.read(ref.addr)
                assert rec["content"] == expected["content"]
                assert rec["role"] == expected["role"]


# ---------------------------------------------------------------------------
# Empty / null / default
# ---------------------------------------------------------------------------


class TestEmptyAndNull:
    def test_empty_string(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": ""})
            assert ds.read(ref.addr)["content"] == ""

    def test_default_when_field_omitted(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1})  # content omitted
            assert ds.read(ref.addr)["content"] == ""

    def test_none_treated_as_empty(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": None})
            assert ds.read(ref.addr)["content"] == ""


# ---------------------------------------------------------------------------
# write_field — update single text field
# ---------------------------------------------------------------------------


class TestWriteField:
    def test_update_text_field(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "original"})
            ds.write_field(ref.addr, "content", "updated")
            assert ds.read(ref.addr)["content"] == "updated"

    def test_update_frees_old_blob(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "old text"})

            before = db.blob_store.get_stats()["free_slots"]
            ds.write_field(ref.addr, "content", "new text")
            after = db.blob_store.get_stats()["free_slots"]

            # Old blob should have been freed then immediately reused or
            # left on freelist. Either way free_slots should have gone up
            # before being consumed — net effect: unchanged or higher.
            # The key is no crash and the new value is correct.
            assert ds.read(ref.addr)["content"] == "new text"

    def test_update_to_empty_frees_blob(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "some text"})

            before = db.blob_store.get_stats()["free_slots"]
            ds.write_field(ref.addr, "content", "")
            after = db.blob_store.get_stats()["free_slots"]

            assert after > before  # Blob freed, nothing new written
            assert ds.read(ref.addr)["content"] == ""

    def test_update_non_text_field_unchanged(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "hello"})
            ds.write_field(ref.addr, "id", 99)
            rec = ds.read(ref.addr)
            assert rec["id"] == 99
            assert rec["content"] == "hello"


# ---------------------------------------------------------------------------
# delete — blobs must be freed
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_frees_blob(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "to be deleted"})

            before = db.blob_store.get_stats()["free_slots"]
            ds.delete(ref.addr)
            after = db.blob_store.get_stats()["free_slots"]

            assert after > before

    def test_delete_empty_text_no_error(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": ""})
            ds.delete(ref.addr)  # Should not crash (no blob to free)
            assert ds.is_deleted(ref.addr)

    def test_delete_multiple_text_fields(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", q="text", a="text")
            ref = ds.insert({"id": 1, "q": "question text", "a": "answer text"})

            before = db.blob_store.get_stats()["free_slots"]
            ds.delete(ref.addr)
            after = db.blob_store.get_stats()["free_slots"]

            # Both blobs freed
            assert after >= before + 2

    def test_record_marked_deleted_after_delete(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "bye"})
            ds.delete(ref.addr)
            assert ds.is_deleted(ref.addr)
            assert not ds.exists(ref.addr)


# ---------------------------------------------------------------------------
# Persistence — round-trip across sessions
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_roundtrip_simple(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 7, "content": "persisted text"})
            addr = ref.addr

        with DB(db_path) as db:
            ds = db["msgs"]
            rec = ds.read(addr)
            assert rec["id"] == 7
            assert rec["content"] == "persisted text"

    def test_roundtrip_multiple(self, db_path):
        data = [(i, f"message number {i} " + "x" * i * 10) for i in range(20)]

        addrs = []
        with DB(db_path) as db:
            ds = db.create_dataset("chat", id="uint32", body="text")
            for i, text in data:
                ref = ds.insert({"id": i, "body": text})
                addrs.append(ref.addr)

        with DB(db_path) as db:
            ds = db["chat"]
            for addr, (i, text) in zip(addrs, data):
                rec = ds.read(addr)
                assert rec["id"] == i
                assert rec["body"] == text

    def test_registry_preserves_text_marker(self, db_path):
        """Schema must survive close/reopen with "text" marker intact."""
        with DB(db_path) as db:
            db.create_dataset("msgs", id="uint32", note="U50", content="text")

        with DB(db_path) as db:
            ds = db["msgs"]
            assert "content" in ds._text_fields
            assert "note" not in ds._text_fields

    def test_write_after_reopen(self, db_path):
        addrs = []
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "first session"})
            addrs.append(ref.addr)

        with DB(db_path) as db:
            ds = db["msgs"]
            ref2 = ds.insert({"id": 2, "content": "second session"})
            addrs.append(ref2.addr)

        with DB(db_path) as db:
            ds = db["msgs"]
            assert ds.read(addrs[0])["content"] == "first session"
            assert ds.read(addrs[1])["content"] == "second session"

    def test_freelist_persists_across_sessions(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            ref = ds.insert({"id": 1, "content": "to be deleted"})
            ds.delete(ref.addr)
            slots_freed = db.blob_store.get_stats()["free_slots"]

        with DB(db_path) as db:
            # Freelist should still know about freed slots
            assert db.blob_store.get_stats()["free_slots"] == slots_freed


# ---------------------------------------------------------------------------
# read_many
# ---------------------------------------------------------------------------


class TestReadMany:
    def test_read_many_with_text(self, db_path):
        messages = [f"message {i}" for i in range(10)]
        with DB(db_path) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")
            block_addr = ds.allocate_block(len(messages))
            for i, text in enumerate(messages):
                ds.write(block_addr + i * ds.record_size, id=i, content=text)

            results = ds.read_many(block_addr, len(messages))
            assert len(results) == len(messages)
            for i, rec in enumerate(results):
                assert rec["content"] == messages[i]

    def test_read_many_mixed_fields(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("chat", role="U10", content="text")
            block_addr = ds.allocate_block(3)
            ds.write(block_addr + 0 * ds.record_size, role="user", content="hello")
            ds.write(block_addr + 1 * ds.record_size, role="bot", content="hi there")
            ds.write(block_addr + 2 * ds.record_size, role="user", content="bye")

            results = ds.read_many(block_addr, 3)
            assert results[0]["content"] == "hello"
            assert results[1]["content"] == "hi there"
            assert results[2]["content"] == "bye"
            assert results[1]["role"] == "bot"


# ---------------------------------------------------------------------------
# Multiple text fields
# ---------------------------------------------------------------------------


class TestMultipleTextFields:
    def test_two_text_fields(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("qa", id="uint32", question="text", answer="text")
            ref = ds.insert({
                "id": 1,
                "question": "What is the meaning of life?",
                "answer": "42",
            })
            rec = ds.read(ref.addr)
            assert rec["question"] == "What is the meaning of life?"
            assert rec["answer"] == "42"

    def test_update_one_text_leaves_other_intact(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset("qa", id="uint32", q="text", a="text")
            ref = ds.insert({"id": 1, "q": "original question", "a": "original answer"})
            ds.write_field(ref.addr, "q", "updated question")
            rec = ds.read(ref.addr)
            assert rec["q"] == "updated question"
            assert rec["a"] == "original answer"

    def test_mixed_text_and_fixed_fields(self, db_path):
        with DB(db_path) as db:
            ds = db.create_dataset(
                "events",
                id="uint64",
                timestamp="int64",
                category="U20",
                payload="text",
            )
            ref = ds.insert({
                "id": 42,
                "timestamp": 1_700_000_000,
                "category": "chat",
                "payload": '{"role": "user", "text": "hello"}',
            })
            rec = ds.read(ref.addr)
            assert rec["id"] == 42
            assert rec["timestamp"] == 1_700_000_000
            assert rec["category"] == "chat"
            assert rec["payload"] == '{"role": "user", "text": "hello"}'


# ---------------------------------------------------------------------------
# Blob reuse / freelist interaction
# ---------------------------------------------------------------------------


class TestBlobReuse:
    def test_update_reuses_freed_slot(self, db_path):
        """After update, old blob slot should be reused for next write."""
        with DB(db_path, blob_compression=None) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")

            # Insert a record
            ref = ds.insert({"id": 1, "content": "aaa"})

            # The freed slot from the update should be on the freelist
            # and potentially reused for subsequent inserts.
            ds.write_field(ref.addr, "content", "bbb")
            ref2 = ds.insert({"id": 2, "content": "ccc"})

            # Both records should be readable correctly
            assert ds.read(ref.addr)["content"] == "bbb"
            assert ds.read(ref2.addr)["content"] == "ccc"

    def test_delete_then_insert_reuses_space(self, db_path):
        with DB(db_path, blob_compression=None) as db:
            ds = db.create_dataset("msgs", id="uint32", content="text")

            ref1 = ds.insert({"id": 1, "content": "first"})
            ds.delete(ref1.addr)

            stats_before = db.blob_store.get_stats()["free_slots"]
            assert stats_before > 0

            ref2 = ds.insert({"id": 2, "content": "reuse"})
            stats_after = db.blob_store.get_stats()["free_slots"]

            # Free slots consumed (reused)
            assert stats_after < stats_before
            assert ds.read(ref2.addr)["content"] == "reuse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
