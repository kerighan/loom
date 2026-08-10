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


def _seed_collection(path, n, base=0):
    """Create a collection "c" with a primary + Many index, n records."""
    from loom import Many

    with DB(path) as db:
        col = db.collection("c", {"id": "utf8[8]", "grp": "utf8[4]",
                                  "v": "int64"},
                            indexes={"id": "primary", "grp": Many()})
        col.insert_many([{"id": f"r{i}", "grp": f"g{i % 4}", "v": base + i}
                         for i in range(n)])


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


class TestStalenessInvalidation:
    """A shared cache outlives the DB handle, so a reader reopened on a path
    another writer has since modified would keep serving pre-write nodes.
    cache_id versioning (scan-free) and invalidate_prefix (explicit) fix it.
    """

    def _count(self, db, hi=300):
        return sum(1 for k in range(hi) if db["b"].get(k) is not None)

    def test_reopen_with_default_id_serves_stale(self, two_paths):
        pa, _ = two_paths
        _seed_btree(pa, [(k, k) for k in range(100)])
        shared = LRUCache(200_000)

        r1 = DB(pa, flag="r", cache=shared)
        assert self._count(r1) == 100          # warm every node into cache
        r1.close()

        with DB(pa) as w:                       # external writer adds 200 more
            bt = w["b"]
            for k in range(100, 300):
                bt[k] = {"v": k}

        r2 = DB(pa, flag="r", cache=shared)     # same path -> same namespace
        try:
            assert self._count(r2) < 300        # STALE: pre-write view survives
        finally:
            r2.close()

    def test_versioned_cache_id_reads_fresh(self, two_paths):
        pa, _ = two_paths
        _seed_btree(pa, [(k, k) for k in range(100)])
        shared = LRUCache(200_000)

        r1 = DB(pa, flag="r", cache=shared)
        self._count(r1)
        r1.close()
        with DB(pa) as w:
            bt = w["b"]
            for k in range(100, 300):
                bt[k] = {"v": k}

        rp = "path:" + os.path.realpath(pa)
        r2 = DB(pa, flag="r", cache=shared, cache_id=rp + "#2")
        try:
            assert self._count(r2) == 300       # fresh namespace, no stale hits
        finally:
            r2.close()

    def test_invalidate_prefix_reads_fresh_and_spares_siblings(self, two_paths):
        pa, pb = two_paths
        _seed_btree(pa, [(k, k) for k in range(100)])
        _seed_btree(pb, [(k, 1000 + k) for k in range(50)])
        shared = LRUCache(200_000)

        def n_entries(prefix):
            return sum(
                1 for k in shared._cache.keys()
                if isinstance(k, tuple) and k and isinstance(k[0], str)
                and k[0].startswith(prefix)
            )

        a1 = DB(pa, flag="r", cache=shared)
        b = DB(pb, flag="r", cache=shared)
        self._count(a1)
        for k in range(50):
            b["b"].get(k)                        # warm B's nodes
        a1.close()
        with DB(pa) as w:
            bt = w["b"]
            for k in range(100, 300):
                bt[k] = {"v": k}

        rp_a = "path:" + os.path.realpath(pa)
        rp_b = "path:" + os.path.realpath(pb)
        b_before = n_entries(rp_b)
        assert b_before > 0
        # A's entries live under rp_a; B's under rp_b — evict only A's.
        n = shared.invalidate_prefix(rp_a)
        assert n > 0
        assert n_entries(rp_a) == 0               # A fully evicted...
        assert n_entries(rp_b) == b_before        # ...B untouched
        try:
            a2 = DB(pa, flag="r", cache=shared)   # default id, but A was evicted
            assert self._count(a2) == 300         # fresh
            a2.close()
            for k in range(50):                   # B still correct
                assert b["b"].get(k)["v"] == 1000 + k
        finally:
            b.close()

    def test_cache_id_overrides_realpath(self, two_paths):
        pa, _ = two_paths
        _seed_btree(pa, [(1, 1)])
        with DB(pa, flag="r", cache_id="custom#7") as db:
            assert db._cache_id == "custom#7"

    def _warm_collection(self, db, n):
        for i in range(n):
            db["c"][f"r{i}"]

    def test_drop_collection_spares_siblings_in_shared_cache(self, two_paths):
        # drop_collection must invalidate only THIS file's namespace when the
        # cache is borrowed — not clear() the whole shared budget and cold-
        # start every other project.
        pa, pb = two_paths
        _seed_collection(pa, 50)
        _seed_collection(pb, 50, base=1000)
        shared = LRUCache(200_000)

        def n_entries(db):
            pref = db._cache_id + "\x1f"
            return sum(
                1 for k in shared._cache.keys()
                if isinstance(k, tuple) and k and isinstance(k[0], str)
                and k[0].startswith(pref)
            )

        wa = DB(pa, cache=shared)                 # writer on A (borrows cache)
        b = DB(pb, flag="r", cache=shared)        # reader on B
        try:
            self._warm_collection(wa, 50)
            self._warm_collection(b, 50)
            b_before = n_entries(b)
            assert n_entries(wa) > 0 and b_before > 0

            wa.drop_collection("c")               # borrowed cache: spare B
            assert n_entries(wa) == 0             # A's entries gone
            assert n_entries(b) == b_before       # B's entries untouched
            for i in range(50):                   # B still reads correctly
                assert b["c"][f"r{i}"]["v"] == 1000 + i
        finally:
            wa.close()
            b.close()

    def test_owned_cache_drop_still_clears(self, two_paths):
        # With a private cache there are no siblings; clear() is fine.
        pa, _ = two_paths
        _seed_collection(pa, 50)
        with DB(pa) as db:                        # owns a private cache
            self._warm_collection(db, 50)
            assert len(db._shared_cache) > 0
            db.drop_collection("c")
            assert len(db._shared_cache) == 0


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
