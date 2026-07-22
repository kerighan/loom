"""Crash consistency of the header double-buffer + List metadata.

Regression for the field-mixed header that made a List reopen with `length`
ahead of `p_last`, crashing _calculate_block_and_offset with
"IndexError: Index N out of range" on the next append.
"""

import os
import pickle
import struct
import tempfile
import zlib

import pytest

from loom import Search
from loom.database import DB
from loom.fileio import ByteFileDB


def _make_db(path):
    db = DB(path)
    db.open()
    return db


class TestListReopenAfterTornHeader:
    """B: a List heals a header whose fields span two generations."""

    def test_length_ahead_of_p_last_is_healed_on_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = _make_db(path)
            lst = db.create_list("posts", {"pk": "uint64"})
            # 272 items exactly fill blocks 0,1,2 (57+86+129); the 273rd needs
            # block 3 — the index that crashed in production.
            for i in range(272):
                lst.append({"pk": i})
            db.close()

            # Simulate a crash-recovered header: keep the fresh length but roll
            # p_last back to a stale generation (blocks stay allocated).
            raw = ByteFileDB(path)
            raw.open()
            meta = raw.get_header_field("_ds_posts_metadata")
            assert meta["length"] == 272
            meta["p_last"] = 11  # stale: as if only blocks 0,1 were allocated
            raw.set_header_field("_ds_posts_metadata", meta)
            raw.close()

            # Reopen and append the 273rd item — used to raise IndexError.
            db = _make_db(path)
            lst = db["posts"]
            assert len(lst) == 272
            lst.append({"pk": 272})  # must not raise
            assert len(lst) == 273
            assert lst[272] == {"pk": 272}
            db.close()


class TestHeaderTornSlotRejected:
    """A: a torn slot is rejected via CRC; the newest intact slot wins."""

    def test_partial_slot_write_falls_back_to_intact_generation(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = ByteFileDB(path, header_size=8192)
            db.open()
            db.set_header_field("x", "generation-1")
            db.set_header_field("x", "generation-2")  # flips to the other slot
            active = db.mapped_file[0]
            db.close()

            # Corrupt the *active* slot's payload in place (a torn writeback:
            # the flip byte landed but the slot's pages did not). Its CRC no
            # longer matches, so load must fall back to the intact sibling.
            with open(path, "r+b") as f:
                mm = f.read()
            slot_off = 1 + active * ((8192 - 1) // 2)
            magic = bytes(mm[slot_off : slot_off + 4])
            assert magic == ByteFileDB._MAGIC
            _seq, size, _crc = struct.unpack(
                "<QII", bytes(mm[slot_off + 4 : slot_off + 20])
            )
            payload_off = slot_off + 20
            data = bytearray(open(path, "rb").read())
            data[payload_off + size - 1] ^= 0xFF  # flip a byte in the pickle
            with open(path, "r+b") as f:
                f.write(bytes(data))

            db = ByteFileDB(path, header_size=8192)
            db.open()
            # Never a torn/garbage value: either the intact new gen, or a clean
            # rollback to gen-1 — never a spliced state.
            assert db.get_header_field("x") in ("generation-1", "generation-2")
            db.close()

    def test_higher_seqno_wins_regardless_of_hint_byte(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = ByteFileDB(path, header_size=8192)
            db.open()
            for i in range(5):
                db.set_header_field("x", f"v{i}")
            db.close()

            # Scramble the hint byte; the seqno must still select the newest.
            with open(path, "r+b") as f:
                f.seek(0)
                f.write(bytes([0 if bytes(open(path, "rb").read(1)) == b"\x01" else 1]))

            db = ByteFileDB(path, header_size=8192)
            db.open()
            assert db.get_header_field("x") == "v4"
            db.close()


class TestBulkInsertSurvivesUnsyncedReopen:
    """End-to-end: a Collection bulk insert reopens cleanly without close()."""

    def test_reopen_without_close_is_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = _make_db(path)
            col = db.collection(
                "posts",
                {"pk": "utf8[16]", "body": "text"},
                indexes={"pk": "primary", "content": Search(fields=["body"])},
            )
            for i in range(300):
                col.insert({"pk": f"p{i}", "body": f"post number {i}"})
            db.flush()  # writeback WITHOUT close (the durable checkpoint)
            del col, db

            db = _make_db(path)
            col = db.collection("posts")
            # docid2pk (the List that crashed) must be appendable again.
            col.insert({"pk": "p300", "body": "post number 300"})
            assert col["p300"]["body"] == "post number 300"
            db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
