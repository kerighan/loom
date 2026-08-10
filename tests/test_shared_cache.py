"""Shared LRU cache across multiple DB files.

Two files that share a schema have homonymous structures at identical table
offsets.  A single LRUCache handed to several DBs must namespace each file's
entries by a STABLE per-file identity, or file B's lookup would hit an address
that file A cached and read the wrong record — silent corruption.

Also checks that the budget is genuinely shared (one LRU, not N) and that
closing one borrower does not wipe the others' warm entries.
"""

import os
import tempfile

import pytest

from loom import DB, LRUCache


@pytest.fixture
def two_paths():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "a.loom"), os.path.join(d, "b.loom")


def _seed(path, pairs):
    """Create a dict "d" and insert (key, value) pairs in order."""
    with DB(path) as db:
        dd = db.create_dict("d", {"v": "int64"})
        for k, v in pairs:
            dd[k] = {"v": v}


def _seed_btree(path, items):
    """Create an int-keyed btree "b" and insert (key, value) items."""
    with DB(path) as db:
        bt = db.create_btree("b", {"v": "int64"}, int_keys=True)
        for k, v in items:
            bt[k] = {"v": v}


class TestSharedCacheIsolation:
    def test_shared_btree_reads_are_isolated(self, two_paths):
        # BTree's node cache maps addr -> the *deserialised node* (keys,
        # children, values), so a namespace collision would let file B read
        # file A's tree nodes wholesale.  Divergent shapes (deep vs flat) make
        # the same node address hold different content in each file, so a
        # mis-namespaced hit returns the wrong record — the real hazard.
        pa, pb = two_paths
        _seed_btree(pa, [(i, 1000 + i) for i in range(400)])   # deep tree
        _seed_btree(pb, [(k, 2000 + k) for k in (5, 7, 9)])    # flat tree

        shared = LRUCache(100_000)
        a = DB(pa, flag="r", cache=shared)
        b = DB(pb, flag="r", cache=shared)
        try:
            for k in range(400):            # warm every node of A into cache
                assert a["b"][k]["v"] == 1000 + k
            for k in (5, 7, 9):             # B must read its OWN nodes
                assert b["b"][k]["v"] == 2000 + k
        finally:
            a.close()
            b.close()

    def test_namespacing_is_load_bearing(self, two_paths):
        # Prove the per-file prefix is what prevents corruption: force both
        # DBs to the SAME identity (as if the prefix were absent) and the
        # shared node cache DOES serve A's nodes for B's reads.  This guards
        # against a future refactor silently dropping the prefix.
        pa, pb = two_paths
        _seed_btree(pa, [(i, 1000 + i) for i in range(400)])
        _seed_btree(pb, [(k, 2000 + k) for k in (5, 7, 9)])

        shared = LRUCache(100_000)
        # Set the shared identity BEFORE open() materialises the structures
        # (which capture the identity), simulating the missing prefix.
        a = DB(pa, flag="r", cache=shared, auto_open=False)
        a._cache_id = "SAME"
        a.open()
        b = DB(pb, flag="r", cache=shared, auto_open=False)
        b._cache_id = "SAME"
        b.open()
        try:
            for k in range(400):
                a["b"][k]
            corrupted = 0
            for k in (5, 7, 9):
                try:
                    if b["b"][k]["v"] != 2000 + k:
                        corrupted += 1
                except Exception:
                    corrupted += 1        # a stale node can also raise
            assert corrupted > 0            # collision => wrong reads
        finally:
            a.close()
            b.close()

    def test_distinct_files_get_distinct_ids(self, two_paths):
        pa, pb = two_paths
        _seed(pa, [("x", 1)])
        _seed(pb, [("x", 2)])
        a, b = DB(pa, flag="r"), DB(pb, flag="r")
        try:
            assert a._cache_id != b._cache_id
        finally:
            a.close()
            b.close()

    def test_cache_id_is_stable_across_reopen(self, two_paths):
        pa, _ = two_paths
        _seed(pa, [("x", 1)])
        first = DB(pa, flag="r")._cache_id
        second = DB(pa, flag="r")._cache_id
        assert first == second               # realpath is stable per path
        assert first.startswith("path:")

    def test_budget_is_shared_not_multiplied(self, two_paths):
        pa, pb = two_paths
        _seed(pa, [("x", 1)])
        _seed(pb, [("y", 2)])
        shared = LRUCache(100_000)
        a = DB(pa, flag="r", cache=shared)
        b = DB(pb, flag="r", cache=shared)
        try:
            a["d"]["x"]
            b["d"]["y"]
            # Both borrowers point at the very same LRU object.
            assert a._shared_cache is shared
            assert b._shared_cache is shared
            assert a._owns_cache is False and b._owns_cache is False
        finally:
            a.close()
            b.close()

    def test_close_does_not_wipe_shared_cache(self, two_paths):
        pa, pb = two_paths
        _seed(pa, [("x", 1)])
        _seed(pb, [("y", 2)])
        shared = LRUCache(100_000)
        a = DB(pa, flag="r", cache=shared)
        b = DB(pb, flag="r", cache=shared)
        try:
            a["d"]["x"]
            b["d"]["y"]
            populated = len(shared)
            assert populated > 0
            a.close()                        # borrower closes...
            assert len(shared) == populated  # ...cache untouched for b
        finally:
            if a._is_open:
                a.close()
            b.close()


class TestCacheParamValidation:
    def test_rejects_non_lrucache(self, two_paths):
        pa, _ = two_paths
        _seed(pa, [("x", 1)])
        with pytest.raises(TypeError):
            DB(pa, flag="r", cache={})       # a plain dict is not an LRUCache

    def test_default_still_owns_private_cache(self, two_paths):
        pa, _ = two_paths
        _seed(pa, [("x", 1)])
        with DB(pa, flag="r") as db:
            assert db._owns_cache is True
            assert db._shared_cache is not None

    def test_cache_size_zero_disables(self, two_paths):
        pa, _ = two_paths
        _seed(pa, [("x", 1)])
        with DB(pa, flag="r", cache_size=0) as db:
            assert db._shared_cache is None
            assert db._owns_cache is False


class TestPathIdentity:
    def test_symlinked_path_resolves_to_same_identity(self, two_paths):
        # realpath resolves symlinks, so a file opened via a symlink shares
        # the underlying file's cache namespace (no accidental double-cache).
        pa, _ = two_paths
        _seed(pa, [("x", 1)])
        link = pa + ".link"
        os.symlink(pa, link)
        direct = DB(pa, flag="r")._cache_id
        via_link = DB(link, flag="r")._cache_id
        assert direct == via_link


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
