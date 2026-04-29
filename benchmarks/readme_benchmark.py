"""
README performance benchmark.

Reproduces the figures published in README.md so the table can be refreshed
in one go.  Tries to keep allocations / DB lifetime identical to a typical
user workflow (single open(), close on context exit).

Run with:
    PYTHONPATH=. python benchmarks/readme_benchmark.py
"""

from __future__ import annotations

import gc
import os
import random
import statistics
import string
import sys
import tempfile
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB

try:
    from sqlitedict import SqliteDict

    HAS_SQLITEDICT = True
except ImportError:
    HAS_SQLITEDICT = False


N = 10_000  # ops for the comparison table
N_BLOB = 2_000  # smaller for blob-compression tests
KEY_LEN = 20

# Graph benchmark config
GRAPH_N = 20_000  # nodes in Barabási-Albert (m=2) graph
GRAPH_QUERIES = 10_000  # point queries (has_edge, neighbors, etc.)


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────
@contextmanager
def tmp_path(suffix=".loom"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M ops/s"
    if v >= 1_000:
        return f"{v/1_000:.1f}k ops/s"
    return f"{v:.0f} ops/s"


def run(label: str, fn, n: int):
    gc.collect()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    rate = n / elapsed
    us = elapsed / n * 1e6
    print(f"  {label:<28} {n/elapsed:>12,.0f} ops/s   ({us:6.2f} µs/op)")
    return rate


# ─────────────────────────────────────────────────────────────────────
# loom Dict
# ─────────────────────────────────────────────────────────────────────
def bench_loom_dict():
    print("\n[loom Dict]  fixed schema, N =", N)
    keys = [f"user_{i:0{KEY_LEN}d}" for i in range(N)]
    values = [
        {"id": i, "name": f"User {i}", "email": f"u{i}@x.io", "score": i * 1.5}
        for i in range(N)
    ]
    miss_keys = [f"missing_{i:0{KEY_LEN}d}" for i in range(N)]

    res = {}
    with tmp_path() as path, DB(path) as db:
        ds = db.create_dataset(
            "users", id="uint64", name="U50", email="U100", score="float64"
        )
        d = db.create_dict("users_dict", ds, cache_size=0)

        def _insert():
            for k, v in zip(keys, values):
                d[k] = v

        res["insert"] = run("insert", _insert, N)

        def _read():
            for k in keys:
                _ = d[k]

        res["read"] = run("read", _read, N)

        def _contains_hit():
            for k in keys:
                _ = k in d

        res["contains_hit"] = run("contains (hit)", _contains_hit, N)

        def _contains_miss():
            for k in miss_keys:
                _ = k in d

        res["contains_miss"] = run("contains (miss)", _contains_miss, N)

        def _keys():
            list(d.keys())

        res["keys"] = run("keys()", _keys, N)

        def _items():
            for _ in d.items():
                pass

        res["items"] = run("items()", _items, N)

    # batch insert (separate DB, fresh state)
    with tmp_path() as path, DB(path) as db:
        ds = db.create_dataset(
            "users", id="uint64", name="U50", email="U100", score="float64"
        )
        d = db.create_dict("users_dict", ds, cache_size=0)

        def _batch_insert():
            with db.batch():
                for k, v in zip(keys, values):
                    d[k] = v

        res["insert_batch"] = run("insert (batch)", _batch_insert, N)

    return res


# ─────────────────────────────────────────────────────────────────────
# SqliteDict (for the comparison table)
# ─────────────────────────────────────────────────────────────────────
def bench_sqlitedict():
    if not HAS_SQLITEDICT:
        return None

    print("\n[SqliteDict]  N =", N)
    keys = [f"user_{i:0{KEY_LEN}d}" for i in range(N)]
    values = [
        {"id": i, "name": f"User {i}", "email": f"u{i}@x.io", "score": i * 1.5}
        for i in range(N)
    ]
    miss_keys = [f"missing_{i:0{KEY_LEN}d}" for i in range(N)]
    res = {}

    # autocommit (per-call durability) — comparable to loom default
    with tmp_path(suffix=".sqlite") as path:
        d = SqliteDict(path, autocommit=True)

        def _insert():
            for k, v in zip(keys, values):
                d[k] = v

        res["insert"] = run("insert (autocommit)", _insert, N)

        def _read():
            for k in keys:
                _ = d[k]

        res["read"] = run("read", _read, N)

        def _contains_hit():
            for k in keys:
                _ = k in d

        res["contains_hit"] = run("contains (hit)", _contains_hit, N)

        def _contains_miss():
            for k in miss_keys:
                _ = k in d

        res["contains_miss"] = run("contains (miss)", _contains_miss, N)

        def _keys():
            list(d.keys())

        res["keys"] = run("keys()", _keys, N)

        d.close()

    # batch (single transaction)
    with tmp_path(suffix=".sqlite") as path:
        d = SqliteDict(path, autocommit=False)

        def _batch_insert():
            for k, v in zip(keys, values):
                d[k] = v
            d.commit()

        res["insert_batch"] = run("insert (commit once)", _batch_insert, N)
        d.close()

    return res


# ─────────────────────────────────────────────────────────────────────
# loom List + Queue
# ─────────────────────────────────────────────────────────────────────
def bench_loom_list_queue():
    print("\n[loom List / Queue]  N =", N)
    res = {}
    values = [{"id": i, "score": i * 1.5} for i in range(N)]

    with tmp_path() as path, DB(path) as db:
        ds = db.create_dataset("items", id="uint64", score="float64")
        lst = db.create_list("lst", ds)

        def _append():
            for v in values:
                lst.append(v)

        res["list_append"] = run("List.append", _append, N)

        def _read_idx():
            for i in range(N):
                _ = lst[i]

        res["list_read"] = run("List[i] read", _read_idx, N)

    with tmp_path() as path, DB(path) as db:
        ds = db.create_dataset("items", id="uint64", score="float64")
        q = db.create_queue("q", ds)

        def _push_batch():
            with db.batch():
                for v in values:
                    q.push(v)

        res["queue_push_batch"] = run("Queue.push (batch)", _push_batch, N)

        def _pop():
            for _ in range(N):
                q.pop()

        res["queue_pop"] = run("Queue.pop", _pop, N)

    return res


# ─────────────────────────────────────────────────────────────────────
# blob compression (str body)
# ─────────────────────────────────────────────────────────────────────
def bench_blob_dict(compression):
    print(f"\n[loom Dict + str body]  compression={compression!r}, N={N_BLOB}")
    rng = random.Random(42)
    body_chars = string.ascii_lowercase + " "
    bodies = ["".join(rng.choices(body_chars, k=600)) for _ in range(N_BLOB)]
    keys = [f"k{i:06d}" for i in range(N_BLOB)]
    res = {}

    with tmp_path() as path:
        # NB: DB defaults to blob_compression="brotli" — pass it explicitly
        # so `compression=None` actually disables compression.
        with DB(path, blob_compression=compression) as db:
            ds = db.create_dataset(
                "docs",
                id="uint64",
                body="text",
            )
            d = db.create_dict("docs_dict", ds, cache_size=0)

            def _insert():
                # text fields benefit a lot from batch (single BlobStore flush)
                with db.batch():
                    for k, b in zip(keys, bodies):
                        d[k] = {"id": int(k[1:]), "body": b}

            res["insert"] = run("insert (text body, batch)", _insert, N_BLOB)

            def _read():
                for k in keys:
                    _ = d[k]["body"]

            res["read"] = run("read (text body)", _read, N_BLOB)

    return res


# ─────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────
def _ba_graph(n, m=2, seed=42):
    """Barabási-Albert preferential-attachment edge list."""
    rng = random.Random(seed)
    edges = []
    repeated = list(range(m))
    for i in range(m, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(repeated))
        for j in chosen:
            edges.append((i, j))
        repeated.extend(chosen)
        repeated.append(i)
        repeated.extend([i] * (m - 1))
    return edges


def bench_loom_graph():
    print(f"\n[loom Graph]  Barabási-Albert  N={GRAPH_N}, m=2")
    edges = _ba_graph(GRAPH_N, m=2)
    n_edges = len(edges)
    print(f"  generated {n_edges:,} edges")

    res = {}
    with tmp_path() as path, DB(path) as db:
        g = db.create_graph(
            "g",
            node_schema={"name": "U30", "age": "uint32"},
            edge_schema={"weight": "float32"},
            directed=True,
        )

        def _add_nodes():
            for i in range(GRAPH_N):
                g.add_node(str(i), name=f"U{i}", age=i % 90)

        res["add_node"] = run("add_node", _add_nodes, GRAPH_N)

        def _add_edges():
            for u, v in edges:
                g.add_edge(str(u), str(v), weight=1.0)

        res["add_edge"] = run("add_edge", _add_edges, n_edges)

        rng = random.Random(123)
        sample_nodes = [str(rng.randrange(GRAPH_N)) for _ in range(GRAPH_QUERIES)]
        sample_edges = [
            (str(u), str(v))
            for u, v in (edges[rng.randrange(n_edges)] for _ in range(GRAPH_QUERIES))
        ]
        miss_edges = [
            (str(rng.randrange(GRAPH_N)), str(rng.randrange(GRAPH_N)))
            for _ in range(GRAPH_QUERIES)
        ]

        def _get_node():
            for nid in sample_nodes:
                _ = g[nid]

        res["get_node"] = run("get_node", _get_node, GRAPH_QUERIES)

        def _has_edge_hit():
            for u, v in sample_edges:
                _ = g.has_edge(u, v)

        res["has_edge_hit"] = run("has_edge (hit)", _has_edge_hit, GRAPH_QUERIES)

        def _has_edge_miss():
            for u, v in miss_edges:
                _ = g.has_edge(u, v)

        res["has_edge_miss"] = run("has_edge (miss)", _has_edge_miss, GRAPH_QUERIES)

        def _get_edge():
            for u, v in sample_edges:
                _ = g.get_edge(u, v)

        res["get_edge"] = run("get_edge", _get_edge, GRAPH_QUERIES)

        def _neighbors():
            total = 0
            for nid in sample_nodes:
                for _ in g.neighbors(nid):
                    total += 1
            res["_neighbors_total"] = total

        run("neighbors()", _neighbors, GRAPH_QUERIES)
        # report per-node (outer loop) rate
        # We also report per-edge-visited below for context.

        # Re-time neighbors to get per-node rate cleanly
        gc.collect()
        t0 = time.perf_counter()
        total_visited = 0
        for nid in sample_nodes:
            for _ in g.neighbors(nid):
                total_visited += 1
        elapsed = time.perf_counter() - t0
        res["neighbors_per_node"] = GRAPH_QUERIES / elapsed
        res["neighbors_per_edge"] = total_visited / elapsed if total_visited else 0
        print(
            f"    neighbors visited {total_visited:,} edges "
            f"→ {res['neighbors_per_edge']:,.0f} edges/s"
        )

        def _out_degree():
            for nid in sample_nodes:
                _ = g.out_degree(nid)

        res["out_degree"] = run("out_degree", _out_degree, GRAPH_QUERIES)

        size = os.path.getsize(path)

    res["n_edges"] = n_edges
    res["disk_bytes"] = size
    res["bytes_per_edge"] = size / n_edges
    return res


# ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Python {sys.version.split()[0]}, " f"loom benchmarks (N={N})")

    loom = bench_loom_dict()
    sqld = bench_sqlitedict() if HAS_SQLITEDICT else None
    listq = bench_loom_list_queue()
    blob_none = bench_blob_dict(None)
    try:
        import brotli  # noqa: F401

        blob_br = bench_blob_dict("brotli")
    except ImportError:
        print("\nbrotli not installed → skipping compressed blob bench")
        blob_br = None
    graph = bench_loom_graph()

    # ── summary block (markdown-ready) ───────────────────────────────
    print("\n" + "=" * 70)
    print("MARKDOWN SUMMARY (paste into README)")
    print("=" * 70)

    if sqld:
        print("\n### loom vs SqliteDict — {:,} ops, fixed schema\n".format(N))
        print("| Operation | **loom** | **SqliteDict** | Ratio |")
        print("|---|---:|---:|---:|")
        rows = [
            ("Dict insert", loom["insert"], sqld["insert"]),
            ("Dict insert, batch", loom["insert_batch"], sqld["insert_batch"]),
            ("Dict read", loom["read"], sqld["read"]),
            ("Dict contains", loom["contains_hit"], sqld["contains_hit"]),
            ("Dict keys()", loom["keys"], sqld["keys"]),
        ]
        for name, lv, sv in rows:
            faster, ratio = ("loom", lv / sv) if lv >= sv else ("SQLite", sv / lv)
            print(
                f"| {name} | **{lv:,.0f} ops/s** | "
                f"{sv:,.0f} ops/s | {faster} **{ratio:.1f}×** |"
            )

    print("\n### loom operations — all structures\n")
    print("| Structure | Operation | ops/s | µs/op |")
    print("|---|---|---:|---:|")
    rows = [
        ("Dict", "insert", loom["insert"]),
        ("Dict", "read", loom["read"]),
        ("Dict", "contains", loom["contains_hit"]),
        ("Dict", "keys()", loom["keys"]),
        ("Dict", "items()", loom["items"]),
        ("List", "append", listq["list_append"]),
        ("List", "read[i]", listq["list_read"]),
        ("Queue", "push (batch)", listq["queue_push_batch"]),
        ("Queue", "pop", listq["queue_pop"]),
    ]
    for ds, op, rate in rows:
        print(f"| {ds} | {op} | {rate:,.0f} | {1e6/rate:.0f} |")

    print("\n### Dict with `text` body field\n")
    print("| Schema | Compression | insert | read |")
    print("|---|---|---:|---:|")
    print(
        f"| Fixed fields only | — "
        f"| {loom['insert']:,.0f} ops/s | {loom['read']:,.0f} ops/s |"
    )
    print(
        f"| + `str` body | None "
        f"| {blob_none['insert']:,.0f} ops/s | {blob_none['read']:,.0f} ops/s |"
    )
    if blob_br:
        print(
            f"| + `str` body | brotli "
            f"| {blob_br['insert']:,.0f} ops/s | {blob_br['read']:,.0f} ops/s |"
        )

    # Graph ---------------------------------------------------------------
    print(
        f"\n### Graph — Barabási-Albert (N={GRAPH_N:,} nodes, "
        f"{graph['n_edges']:,} edges, directed)\n"
    )
    print("| Operation | ops/s | µs/op |")
    print("|---|---:|---:|")
    rows = [
        ("add_node", graph["add_node"]),
        ("add_edge", graph["add_edge"]),
        ("get_node", graph["get_node"]),
        ("has_edge (hit)", graph["has_edge_hit"]),
        ("has_edge (miss)", graph["has_edge_miss"]),
        ("get_edge", graph["get_edge"]),
        ("neighbors()", graph["neighbors_per_node"]),
        ("out_degree", graph["out_degree"]),
    ]
    for name, rate in rows:
        print(f"| {name} | {rate:,.0f} | {1e6/rate:.1f} |")
    print(
        f"\nNeighbor iteration visits edges at "
        f"**{graph['neighbors_per_edge']:,.0f} edges/s**.  "
        f"On-disk size: {graph['disk_bytes']/1024/1024:.1f} MB "
        f"({graph['bytes_per_edge']:.0f} bytes/edge, double-indexed)."
    )
    print()


if __name__ == "__main__":
    main()
