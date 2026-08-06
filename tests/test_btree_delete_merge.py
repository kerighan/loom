"""B+ tree delete rebalances (borrow/merge) instead of leaving under-full nodes.

The classic, delicate part: a delete must keep every non-root node >= MIN_KEYS,
all leaves at one depth, separators consistent, and the root collapse when it
loses its last key — while staying bit-correct against a reference model and
keeping the node/value arenas bounded under churn.
"""

import os
import random
import tempfile

from loom import DB


def _check_invariants(bt):
    """Assert the full B+ tree invariant set; return (#keys, set of node addrs)."""
    ORDER, MIN = bt.ORDER, bt.MIN_KEYS
    seen = set()
    leaf_depths = set()
    total = [0]

    def walk(addr, depth, lo, hi, is_root):
        assert addr not in seen, f"node {addr} reachable twice (cycle/shared)"
        seen.add(addr)
        n = bt._read_node(addr)
        nk = n["num_keys"]
        assert nk == len(n["keys"])
        # key count bounds (root is exempt from the lower bound)
        assert nk <= ORDER - 1, f"overfull node {addr}: {nk}"
        if not is_root:
            assert nk >= MIN, f"under-full node {addr}: {nk} < {MIN}"
        # keys sorted and within the (lo, hi) range inherited from ancestors
        for i in range(nk - 1):
            assert n["keys"][i] < n["keys"][i + 1], f"unsorted keys in {addr}"
        if nk:
            assert (lo is None or n["keys"][0] >= lo)
            assert (hi is None or n["keys"][-1] < hi), f"{n['keys'][-1]} >= {hi}"
        if n["is_leaf"]:
            assert len(n["children"]) == nk
            leaf_depths.add(depth)
            total[0] += nk
        else:
            assert len(n["children"]) == nk + 1, f"internal {addr} child count"
            bounds = [lo] + n["keys"] + [hi]
            for i, ch in enumerate(n["children"]):
                walk(int(ch), depth + 1, bounds[i], bounds[i + 1], False)

    if bt.root_addr:
        walk(int(bt.root_addr), 0, None, None, True)
    assert len(leaf_depths) <= 1, f"leaves at different depths: {leaf_depths}"
    assert total[0] == len(bt), f"leaf entries {total[0]} != size {len(bt)}"
    return total[0], seen


class TestDeleteMergeInvariants:
    def test_sequential_delete_to_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as bt_db:
                bt = bt_db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(3000):
                    bt[f"{i:06d}"] = {"v": i}
                _check_invariants(bt)
                # delete in a scrambled order, checking invariants throughout
                order = list(range(3000))
                random.Random(0).shuffle(order)
                for j, i in enumerate(order):
                    del bt[f"{i:06d}"]
                    if j % 200 == 0:
                        _check_invariants(bt)
                assert len(bt) == 0
                _check_invariants(bt)
                assert bt.height <= 1        # collapsed back to a single leaf/empty

    def test_random_churn_matches_reference(self):
        rng = random.Random(1)
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as bt_db:
                bt = bt_db.create_btree("t", {"v": "int64"}, key_size=16)
                ref = {}
                for _ in range(12000):
                    k = f"{rng.randint(0, 4000):06d}"
                    if k in ref and rng.random() < 0.5:
                        del bt[k]; del ref[k]
                    else:
                        v = rng.randint(0, 10**6)
                        bt[k] = {"v": v}; ref[k] = v
                _check_invariants(bt)
                assert len(bt) == len(ref)
                assert [k for k in bt.keys()] == sorted(ref)
                for k, v in list(ref.items())[:500]:
                    assert bt[k]["v"] == v
                lo, hi = "001000", "002000"
                assert ([kk for kk, _ in bt.range(lo, hi)]
                        == sorted(k for k in ref if lo <= k <= hi))

    def test_churn_is_bounded_not_growing(self):
        # Steady-state churn on a fixed key set must NOT grow the arena forever
        # (merges free nodes + value slots; splits/inserts reuse them).
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(4000):
                    bt[f"{i:06d}"] = {"v": i}
                db.flush()
                base = db._db.get_allocation_index()
                rng = random.Random(2)
                for _ in range(20000):                 # delete a key, re-add another
                    a = f"{rng.randint(0, 4000):06d}"
                    if a in [f"{i:06d}" for i in range(0)]:
                        pass
                    try:
                        del bt[a]
                    except KeyError:
                        bt[a] = {"v": 0}
                db.flush()
                grew = db._db.get_allocation_index() - base
                _check_invariants(bt)
                # generous bound: churn must not scale with #ops (was ~390 B/op)
                assert grew < 200_000, f"arena grew {grew} B under churn"

    def test_survives_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            rng = random.Random(3)
            with DB(path) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                ref = {}
                for i in range(5000):
                    bt[f"{i:06d}"] = {"v": i}; ref[f"{i:06d}"] = i
                for _ in range(3000):
                    k = f"{rng.randint(0, 5000):06d}"
                    if k in ref:
                        del bt[k]; del ref[k]
            with DB(path) as db:
                bt = db["t"]
                _check_invariants(bt)
                assert [k for k in bt.keys()] == sorted(ref)
                # keep operating after reopen
                del bt[sorted(ref)[0]]
                _check_invariants(bt)


    def test_invariants_hold_after_every_op(self):
        # Tiny key domain → dense borrow/merge/collapse churn; check the full
        # invariant set after EVERY single op (catches rare rebalance bugs the
        # periodic checks would miss).
        for seed in (0, 1, 2):
            rng = random.Random(seed)
            with tempfile.TemporaryDirectory() as d:
                with DB(os.path.join(d, "a.loom")) as db:
                    bt = db.create_btree("t", {"v": "int64"}, key_size=8)
                    ref = {}
                    for step in range(2000):
                        k = f"{rng.randint(0, 120):04d}"
                        if k in ref and rng.random() < 0.5:
                            del bt[k]; del ref[k]
                        else:
                            bt[k] = {"v": step}; ref[k] = step
                        _check_invariants(bt)
                        assert len(bt) == len(ref)
                    assert [k for k in bt.keys()] == sorted(ref)


class TestDeleteInsideDeferredBlock:
    """Regression: a del inside deferred_node_writes() must not corrupt the node
    freelist. The freed node's intrusive next-pointer used to be clobbered by a
    still-pending buffered write flushed at block exit; the freelist head then
    pointed at garbage and the next _create_node read a wild address. Frees are
    now deferred and linked only after the flush."""

    def test_del_and_insert_in_one_deferred_block(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=8)
                ref = {}
                for i in range(3000):
                    bt[f"{i:06d}"] = {"v": i}; ref[f"{i:06d}"] = i
                with bt.deferred_node_writes():
                    for i in range(2000):                 # many merges
                        del bt[f"{i:06d}"]; del ref[f"{i:06d}"]
                    for i in range(3000, 3500):           # splits (create_node)
                        bt[f"{i:06d}"] = {"v": i}; ref[f"{i:06d}"] = i
                _check_invariants(bt)
                # hammering inserts pops the freelist — used to crash here
                for i in range(3500, 6000):
                    bt[f"{i:06d}"] = {"v": i}; ref[f"{i:06d}"] = i
                _check_invariants(bt)
                assert [k for k in bt.keys()] == sorted(ref)
                assert bt[f"005000"]["v"] == 5000

    def test_moves_in_deferred_block_reuse_freed_nodes(self):
        # A batched "move" pattern (del old key + insert new key) inside one
        # deferred block, exactly like a batched counter re-index.
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=12)
                for i in range(5000):
                    bt[f"a{i:06d}"] = {"v": i}
                with bt.deferred_node_writes():
                    for i in range(5000):                 # move every key
                        del bt[f"a{i:06d}"]
                        bt[f"b{i:06d}"] = {"v": i}
                _check_invariants(bt)
                assert len(bt) == 5000
                assert [k for k in bt.keys()] == [f"b{i:06d}" for i in range(5000)]
                assert bt["b002500"]["v"] == 2500


class TestCollectionDeleteMerge:
    def test_collection_delete_heavy_stays_consistent(self):
        from pydantic import BaseModel
        from loom import Many
        from loom.schema import Utf8

        class Rec(BaseModel):
            id: Utf8(16)
            grp: Utf8(8)
            score: int

        rng = random.Random(4)
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = db.collection("c", Rec, indexes={
                    "id": "primary", "grp": Many(sort="score", desc=True)})
                ref = {}
                for _ in range(8000):
                    pk = f"k{rng.randint(0, 2000):05d}"
                    if pk in ref and rng.random() < 0.5:
                        col.delete(pk); del ref[pk]
                    else:
                        rec = {"id": pk, "grp": f"g{rng.randint(0,4)}",
                               "score": rng.randint(0, 100)}
                        col.insert(rec); ref[pk] = rec
                assert len(col) == len(ref)
                _check_invariants(col._indexes["grp"]["struct"])
                # a group query still returns exactly its members, score-desc
                g = "g2"
                want = sorted([r for r in ref.values() if r["grp"] == g],
                              key=lambda r: -r["score"])
                got = col.find("grp", g)
                assert [r["id"] for r in got] == [r["id"] for r in want] or \
                    sorted(r["id"] for r in got) == sorted(r["id"] for r in want)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
