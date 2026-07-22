"""Opt-in durable transactions: all-or-nothing crash safety.

`db.durable()` (and `insert(durable=True)` / `insert_many(durable=True)`) must
guarantee that a hard crash mid-block leaves the DB at its exact pre-block
state — never a torn record/index — while a committed block persists.
"""

import os
import tempfile

import pytest

from loom import Search
from loom.database import DB
import loom.database as _dbmod
from loom.fileio import ByteFileDB


def _open(path):
    db = DB(path)
    db.open()
    return db


def _simulate_hard_crash(db):
    """Abandon a DB as if the process were killed: leave whatever is on disk
    (including any .txn snapshot) untouched, drop mmap handles without the
    normal saving close(), and unregister from the atexit auto-close."""
    db._db.mapped_file.close()
    db._db.mapped_file = None
    db._db.file_handle.close()
    db._db.file_handle = None
    db._is_open = False
    _dbmod._OPEN_DBS.discard(db)


class TestByteFileDBSnapshot:
    def test_uncommitted_block_rolls_back_on_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = ByteFileDB(path, header_size=8192)
            db.open()
            addr = db.allocate(64)
            db.write(addr, b"BEFORE".ljust(64, b"\x00"))
            db.set_header_field("k", "before")
            db.close()

            db = ByteFileDB(path, header_size=8192)
            db.open()
            db.begin_txn()
            db.write(addr, b"AFTER".ljust(64, b"\x00"))
            db.set_header_field("k", "after")
            db.flush()  # force the partial writes to disk (as the OS might)
            # crash: no commit_txn(), no close() — .txn stays on disk
            db.mapped_file.close()
            db.file_handle.close()

            db = ByteFileDB(path, header_size=8192)
            db.open()  # _recover_txn_snapshot() rolls the file back
            assert db.get_header_field("k") == "before"
            assert bytes(db.read(addr, 64)).rstrip(b"\x00") == b"BEFORE"
            assert not os.path.exists(db.txn_filename)
            db.close()

    def test_committed_block_persists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = ByteFileDB(path, header_size=8192)
            db.open()
            db.begin_txn()
            db.set_header_field("k", "after")
            db.commit_txn()
            assert not os.path.exists(db.txn_filename)
            db.close()

            db = ByteFileDB(path, header_size=8192)
            db.open()
            assert db.get_header_field("k") == "after"
            db.close()

    def test_nested_blocks_single_snapshot_and_commit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = ByteFileDB(path, header_size=8192)
            db.open()
            db.begin_txn()
            db.begin_txn()
            assert os.path.exists(db.txn_filename)
            db.commit_txn()  # inner: still in a block
            assert os.path.exists(db.txn_filename)
            assert db._txn_depth == 1
            db.commit_txn()  # outer: commit point
            assert not os.path.exists(db.txn_filename)
            db.close()


class TestCollectionDurable:
    def _make(self, db):
        return db.collection(
            "posts",
            {"pk": "utf8[16]", "body": "text"},
            indexes={"pk": "primary", "content": Search(fields=["body"])},
        )

    def test_insert_many_durable_hard_crash_rolls_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = _open(path)
            col = self._make(db)
            col.insert_many([{"pk": f"p{i}", "body": f"post {i}"} for i in range(50)])
            db.close()

            # Drive a durable block manually so we can crash inside it, mid
            # insert_many, exactly like the production incident.
            db = _open(path)
            col = db.collection("posts")
            db._db.begin_txn()
            col.insert_many([{"pk": f"q{i}", "body": f"more {i}"} for i in range(50)])
            db._db.flush()  # partial writes reach disk; no commit
            _simulate_hard_crash(db)

            db = _open(path)
            col = db.collection("posts")
            assert len(col) == 50  # the 50 'q' rows rolled back
            assert col["p0"]["body"] == "post 0"
            assert "q0" not in col
            # index/search still consistent → a fresh durable insert works
            col.insert_many([{"pk": "q0", "body": "more 0"}], durable=True)
            assert col["q0"]["body"] == "more 0"
            assert len(col) == 51
            db.close()

    def test_insert_many_durable_commit_persists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = _open(path)
            col = self._make(db)
            col.insert_many(
                [{"pk": f"p{i}", "body": f"post {i}"} for i in range(30)],
                durable=True,
            )
            assert not os.path.exists(db._db.txn_filename)
            db.close()

            db = _open(path)
            col = db.collection("posts")
            assert len(col) == 30
            assert col.search("content", "post")  # search index intact
            db.close()

    def test_durable_context_exception_rolls_back_live(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.loom")
            db = _open(path)
            col = self._make(db)
            col.insert({"pk": "p0", "body": "keep"})

            with pytest.raises(RuntimeError):
                with db.durable():
                    col.insert({"pk": "p1", "body": "gone"})
                    raise RuntimeError("boom")

            # Live rollback: re-fetch the handle, the failed write is gone.
            col = db.collection("posts")
            assert len(col) == 1
            assert col["p0"]["body"] == "keep"
            assert "p1" not in col
            assert not os.path.exists(db._db.txn_filename)
            db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
