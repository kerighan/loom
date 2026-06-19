"""Randomised whole-database integrity fuzz test.

Hammers a DB holding several datasets + every structure kind (Dict, BTree,
Set, Graph, List) with anarchic interleaved operations — inserts, updates,
random deletes — and random close/reopen cycles, then checks every structure
against an in-memory shadow model that mirrors the expected state exactly.

This is the strongest guard for the shared address cache: small key/node
pools make updates, deletes and re-inserts of the *same* key collide
constantly (stressing cache invalidation and freed-slot reuse), and the
random close/reopen verifies the on-disk state is correct independently of
the cache (a fresh DB starts with an empty cache).
"""

import os
import random
import tempfile

import pytest

from loom.database import DB

# Small pools → frequent collisions (update / delete-then-readd of same key).
DICT_NAMES = ["users", "posts", "tags"]
KEY_POOL = [f"k{i}" for i in range(40)]
NODE_POOL = [f"n{i}" for i in range(25)]
LIST_TAGS = 0  # running counter for unique list item ids (see fixture-less state)

VALUE_SCHEMA = dict(v="int64", tag="U16")
NODE_SCHEMA = {"label": "U16"}
EDGE_SCHEMA = {"w": "int32"}


def _strip(rec):
    """Compare only the user fields we wrote (ignore hidden/aux fields)."""
    return {"v": int(rec["v"]), "tag": str(rec["tag"])}


class Model:
    """Shadow state mirrored alongside the real DB."""

    def __init__(self):
        self.dicts = {n: {} for n in DICT_NAMES}   # name -> {key: record}
        self.bt = {}                                # key -> record
        self.st = set()                             # set members
        self.nodes = {}                             # node_id -> attrs
        self.out = {}                               # src -> {dst: attrs}
        self.lst = []                               # list of records (order matters)


def _open(path, create):
    """Open the DB and return (db, handles).  On first creation build the
    structures; on reopen resolve them from the auto-loaded registry."""
    db = DB(path)
    if create:
        h = {}
        for n in DICT_NAMES:
            ds = db.create_dataset(f"ds_{n}", **VALUE_SCHEMA)
            h[n] = db.create_dict(n, ds)
        bt_ds = db.create_dataset("ds_bt", **VALUE_SCHEMA)
        h["bt"] = db.create_btree("bt", bt_ds)
        h["st"] = db.create_set("st", key_size=20)
        h["g"] = db.create_graph("g", NODE_SCHEMA, EDGE_SCHEMA, directed=True)
        list_ds = db.create_dataset("ds_list", **dict(v="int64", tag="U24"))
        h["lst"] = db.create_list("lst", list_ds)
    else:
        h = {name: db._datastructures[name] for name in (*DICT_NAMES, "bt", "st", "g", "lst")}
    return db, h


def _verify(h, m):
    """Assert every real structure matches the shadow model exactly."""
    # Dicts
    for n in DICT_NAMES:
        d = h[n]
        assert set(d.keys()) == set(m.dicts[n]), f"dict {n} keyset mismatch"
        assert "" not in set(d.keys()), f"dict {n} leaked empty key"
        for k, rec in m.dicts[n].items():
            assert _strip(d[k]) == rec, f"dict {n}[{k}] = {d[k]} != {rec}"

    # BTree (+ sorted order)
    bt = h["bt"]
    assert set(bt.keys()) == set(m.bt), "btree keyset mismatch"
    assert list(bt.keys()) == sorted(m.bt), "btree keys not sorted"
    for k, rec in m.bt.items():
        assert _strip(bt[k]) == rec, f"btree[{k}] mismatch"

    # Set
    st = h["st"]
    assert set(st) == m.st, "set members mismatch"
    assert len(st) == len(m.st), "set len mismatch"
    for x in list(m.st)[:10]:
        assert x in st
    # a few negatives
    for k in KEY_POOL:
        if k not in m.st:
            assert k not in st

    # Graph
    g = h["g"]
    for src, adj in m.out.items():
        assert set(g.neighbors(src)) == set(adj), f"neighbors({src}) mismatch"
        for dst, attrs in adj.items():
            assert g.has_edge(src, dst), f"missing edge {src}->{dst}"
            assert int(g.get_edge(src, dst)["w"]) == attrs["w"], f"edge {src}->{dst} attr"
    for nid, attrs in m.nodes.items():
        assert str(g.get_node(nid)["label"]) == attrs["label"], f"node {nid} attr"

    # List (order-sensitive)
    got = [(int(r["v"]), str(r["tag"])) for r in h["lst"]]
    exp = [(r["v"], r["tag"]) for r in m.lst]
    assert got == exp, f"list mismatch len {len(got)} vs {len(exp)}"


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 1234, 2024, 31337])
def test_fuzz_whole_db_integrity(seed):
    rng = random.Random(seed)
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    db, h = _open(path, create=True)
    m = Model()
    uid = [0]  # unique value generator

    def nv():
        uid[0] += 1
        return uid[0]

    N_OPS = 2500
    try:
        for step in range(N_OPS):
            op = rng.random()

            if op < 0.22:  # dict put (insert or update)
                n = rng.choice(DICT_NAMES)
                k = rng.choice(KEY_POOL)
                rec = {"v": nv(), "tag": f"t{rng.randint(0, 999)}"}
                h[n][k] = rec
                m.dicts[n][k] = rec

            elif op < 0.32:  # dict delete
                n = rng.choice(DICT_NAMES)
                if m.dicts[n]:
                    k = rng.choice(list(m.dicts[n]))
                    del h[n][k]
                    del m.dicts[n][k]

            elif op < 0.45:  # btree put
                k = rng.choice(KEY_POOL)
                rec = {"v": nv(), "tag": f"b{rng.randint(0, 999)}"}
                h["bt"][k] = rec
                m.bt[k] = rec

            elif op < 0.52:  # btree delete
                if m.bt:
                    k = rng.choice(list(m.bt))
                    del h["bt"][k]
                    del m.bt[k]

            elif op < 0.60:  # set add
                k = rng.choice(KEY_POOL)
                h["st"].add(k)
                m.st.add(k)

            elif op < 0.66:  # set remove
                if m.st:
                    k = rng.choice(list(m.st))
                    h["st"].remove(k)
                    m.st.discard(k)

            elif op < 0.74:  # graph add_edge
                s, d = rng.choice(NODE_POOL), rng.choice(NODE_POOL)
                w = nv()
                h["g"].add_edge(s, d, w=w)
                m.out.setdefault(s, {})[d] = {"w": w}

            elif op < 0.78:  # graph add_node
                nid = rng.choice(NODE_POOL)
                lab = f"L{rng.randint(0, 50)}"
                h["g"].add_node(nid, label=lab)
                m.nodes[nid] = {"label": lab}

            elif op < 0.82:  # graph remove_edge
                if m.out:
                    s = rng.choice(list(m.out))
                    if m.out[s]:
                        d = rng.choice(list(m.out[s]))
                        h["g"].remove_edge(s, d)
                        del m.out[s][d]

            elif op < 0.85:  # graph remove_node (cascades to incident edges)
                nid = rng.choice(NODE_POOL)
                h["g"].remove_node(nid)
                m.nodes.pop(nid, None)
                m.out.pop(nid, None)
                for adj in m.out.values():
                    adj.pop(nid, None)

            elif op < 0.92:  # list append
                rec = {"v": nv(), "tag": f"l{rng.randint(0, 9999)}"}
                h["lst"].append(rec)
                m.lst.append(rec)

            elif op < 0.96:  # list delete by valid index
                if m.lst:
                    i = rng.randrange(len(m.lst))
                    del h["lst"][i]
                    del m.lst[i]

            elif op < 0.975:  # list compact (no visible change)
                h["lst"].compact()

            else:  # random close + reopen (verify disk independent of cache)
                db.close()
                db, h = _open(path, create=False)

            # Occasional mid-run spot verification (exercises warm cache reads)
            if step % 250 == 0:
                _verify(h, m)

        # Final: verify, then reopen from cold and verify again.
        _verify(h, m)
        db.close()
        db, h = _open(path, create=False)
        _verify(h, m)
    finally:
        db.close()
        os.unlink(path)
