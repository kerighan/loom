"""Metadata durability: bulk ops checkpoint, and flush() persists metadata —
so len()/counts stay correct even without an explicit close()."""

import os
import subprocess
import sys
import tempfile

import pytest

from loom import DB


def test_atexit_closes_forgotten_db():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "atexit.db")

    # A child interpreter that opens a DB and appends WITHOUT `with`/close().
    # On clean interpreter exit the atexit safety net must close it and persist
    # the structure metadata.  (subprocess.run → real interpreter exit, so
    # atexit fires — unlike multiprocessing's fork child, which uses os._exit.)
    script = (
        "from loom import DB\n"
        f"db = DB({path!r})\n"
        "ds = db.create_dataset('d', id='int64')\n"
        "lst = db.create_list('feed', ds)\n"
        "[lst.append({'id': i}) for i in range(142)]\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    with DB(path) as db:
        assert len(db["feed"]) == 142   # metadata persisted by atexit


def test_append_many_durable_without_close():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "d.db")

    db = DB(path)
    db.open()
    ds = db.create_dataset("d", id="int64")
    lst = db.create_list("feed", ds)
    lst.append_many([{"id": i} for i in range(142)])   # not a multiple of 100
    del db   # NO close()/flush() — rely on the bulk checkpoint + OS mmap flush

    db2 = DB(path)
    db2.open()
    feed = db2["feed"]
    assert len(feed) == 142
    assert sum(1 for _ in feed) == 142   # len() matches the actual record count
    db2.close()


def test_flush_persists_metadata():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "d.db")

    db = DB(path)
    db.open()
    ds = db.create_dataset("d", id="int64")
    lst = db.create_list("feed", ds)
    for i in range(142):       # individual appends (tail past the auto-save boundary)
        lst.append({"id": i})
    db.flush()                 # flush must persist structure metadata, not just data
    del db

    db2 = DB(path)
    db2.open()
    assert len(db2["feed"]) == 142
    db2.close()


def test_queue_push_many_durable_without_close():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "d.db")

    db = DB(path)
    db.open()
    db.create_queue("q", {"id": "int64"})
    db["q"].push_many([{"id": i} for i in range(142)])
    del db

    db2 = DB(path)
    db2.open()
    assert len(db2["q"]) == 142
    db2.close()


class TestStaleReaderRemap:
    """A long-lived read-only handle whose file grew under it (the API-cache +
    separate-writer pattern): mmap pages are shared, so the reader follows
    FRESH in-place-updated pointers into regions beyond its frozen map —
    slicing there used to return silently truncated bytes (struct.error deep
    in blob decoding).  read() now remaps on demand; a genuinely dangling
    reference raises a clear error instead of returning short bytes."""

    def test_reader_survives_file_growth(self, tmp_path):
        from loom import DB
        path = str(tmp_path / "grow.db")
        w = DB(path)
        w.open()
        col_w = w.collection("articles", {"id": "utf8[16]", "text": "text"},
                             indexes={"id": "primary"})
        col_w.insert_many([{"id": f"a{i}", "text": "court"} for i in range(50)])
        w.flush()

        r = DB(path, flag="r")
        r.open()
        col_r = r.collection("articles")
        assert col_r["a1"]["text"] == "court"

        import os as _os
        before = _os.path.getsize(path)
        big = "un article très long " * 20000          # forces file doubling
        col_w.insert({"id": "a1", "text": big})        # upsert seen via page cache
        w.flush()
        assert _os.path.getsize(path) > before

        assert col_r["a1"]["text"] == big              # remap-on-demand
        r.close()
        w.close()

    def test_read_past_eof_raises_clearly(self, tmp_path):
        import os as _os
        from loom import DB
        path = str(tmp_path / "eof.db")
        db = DB(path)
        db.open()
        with pytest.raises(ValueError, match="read past end of file"):
            db._db.read(_os.path.getsize(path) + 10_000, 8)
        db.close()
