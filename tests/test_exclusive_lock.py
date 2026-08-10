"""DB(exclusive=True): OS-enforced single writer per file.

A second WRITER opening the same file fails fast (DatabaseLockedError) instead
of corrupting it — the enforcement behind any hash/router that only *attributes*
a file to one writer. Readers (flag="r") never lock; the lock is auto-released
on close (and by the OS if the writer crashes). POSIX only.
"""

import os
import tempfile
import multiprocessing as mp

import pytest

from loom import DB, DatabaseLockedError


def _child_open_exclusive(path, q):
    try:
        db = DB(path, exclusive=True)
        q.put("acquired")
        db.close()
    except DatabaseLockedError:
        q.put("locked")
    except Exception as e:  # pragma: no cover - surfaces unexpected failures
        q.put(f"err:{type(e).__name__}:{e}")


def _seed(path):
    with DB(path) as db:
        db.create_dict("d", {"v": "int64"})["a"] = {"v": 1}


@pytest.fixture
def ctx():
    return mp.get_context("spawn")   # clean process, no inherited fds


class TestExclusiveLock:
    def test_second_writer_fails_fast(self, ctx):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            holder = DB(path, exclusive=True)
            try:
                holder["d"]["b"] = {"v": 2}         # holder can write
                q = ctx.Queue()
                p = ctx.Process(target=_child_open_exclusive, args=(path, q))
                p.start(); p.join()
                assert q.get() == "locked"
            finally:
                holder.close()

    def test_lock_released_on_close(self, ctx):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            DB(path, exclusive=True).close()        # acquire then release
            q = ctx.Queue()
            p = ctx.Process(target=_child_open_exclusive, args=(path, q))
            p.start(); p.join()
            assert q.get() == "acquired"

    def test_reader_not_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            holder = DB(path, exclusive=True)
            try:
                r = DB(path, flag="r")              # reader takes no lock
                assert r["d"]["a"]["v"] == 1
                r.close()
                # exclusive + read_only also takes no lock
                ro = DB(path, exclusive=True, flag="r")
                assert ro["d"]["a"]["v"] == 1
                ro.close()
            finally:
                holder.close()

    def test_single_writer_normal_use(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            with DB(path, exclusive=True) as db:
                dd = db.create_dict("d", {"v": "int64"})
                dd["a"] = {"v": 1}
                assert dd["a"]["v"] == 1
            with DB(path, exclusive=True) as db:    # reopen after close
                assert db["d"]["a"]["v"] == 1


class TestLockfileFdLifecycle:
    """close() must close the lock fd, not just release the flock — otherwise
    it leaks until GC (DB holds reference cycles, so __del__ may never run) and
    a service cycling projects climbs toward EMFILE.
    """

    def _nfd(self):
        return len(os.listdir("/proc/self/fd"))

    @pytest.mark.parametrize("kw", [{"exclusive": True},
                                    {"multiprocess_safe": True}])
    def test_close_does_not_leak_fd(self, kw):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            base = self._nfd()
            for _ in range(20):
                db = DB(path, **kw)
                db["d"]["a"]
                db.close()                          # must free the lock fd
            assert self._nfd() - base <= 1          # no per-close leak

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            db = DB(path, exclusive=True)
            db.close()
            db.close()                              # second close() is a no-op
            assert db._lockfile is None

    def test_reopen_after_close_reacquires_lock(self):
        # vacuum() does close()+open() on one object; the fd must re-open.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            db = DB(path, exclusive=True)
            db.close()
            assert db._lockfile is None
            db.open()
            assert db._lockfile is not None
            assert db["d"]["a"]["v"] == 1
            db.close()

    def test_reader_opens_no_lock_fd(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.loom")
            _seed(path)
            with DB(path, flag="r") as r:           # readers never lock
                assert r._lockfile is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
