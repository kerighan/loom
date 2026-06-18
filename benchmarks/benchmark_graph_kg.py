"""Comprehensive Graph / knowledge-graph performance benchmark on FB15k.

FB15k (Freebase subset): ~592k triples, 14,951 entities, 1,345 relations.
Each entity is given a *type* label (the domain of the relations it appears
in, e.g. ``film`` / ``people`` / ``location``), so this exercises loom's
typed-graph path (``label_field``, ``(a:Type)`` queries, the label index).
Relations are interned to a ``uint16`` id, as a real KG store would.

Data: benchmarks/data/fb15k_{train,valid,test}.txt
  (head \\t relation \\t tail).  Download once with, e.g.:
    base=https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15K
    for s in train valid test; do curl -sf $base/$s.txt -o benchmarks/data/fb15k_$s.txt; done

Run:
    PYTHONPATH=. python benchmarks/benchmark_graph_kg.py            # full train split
    PYTHONPATH=. python benchmarks/benchmark_graph_kg.py --triples 100000
    PYTHONPATH=. python benchmarks/benchmark_graph_kg.py --splits train valid test  # all 592k
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── data loading ─────────────────────────────────────────────────────────


def load_triples(splits, limit=None):
    triples = []
    for split in splits:
        path = os.path.join(DATA_DIR, f"fb15k_{split}.txt")
        if not os.path.exists(path):
            print(f"  (missing {path} — skipping)")
            continue
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                triples.append(tuple(parts))
                if limit and len(triples) >= limit:
                    return triples
    return triples


def derive_types(triples):
    """entity → type = most common relation-domain it participates in."""
    counts = defaultdict(Counter)
    for h, r, t in triples:
        dom = r.split("/")[1] if r.startswith("/") and "/" in r[1:] else "misc"
        counts[h][dom] += 1
        counts[t][dom] += 1
    return {e: c.most_common(1)[0][0] for e, c in counts.items()}


# ── timing helper ──────────────────────────────────────────────────────────


def timed(label, fn, n, unit="ops"):
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    rate = n / dt if dt else float("inf")
    print(f"  {label:<34} {rate:>12,.0f} {unit}/s   ({dt:7.3f}s, n={n:,})")
    return rate, dt, result


def fmt_bytes(b):
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"


# ── benchmark ────────────────────────────────────────────────────────────


def run(args):
    print(f"Loading FB15k {args.splits} (limit={args.triples}) ...")
    triples = load_triples(args.splits, args.triples)
    if not triples:
        print("No triples loaded — download the dataset first (see module docstring).")
        return
    etype = derive_types(triples)
    entities = list(etype.keys())
    rels = sorted({r for _, r, _ in triples})
    rel2id = {r: i for i, r in enumerate(rels)}

    n_nodes, n_edges = len(entities), len(triples)
    print(f"  triples={n_edges:,}  entities={n_nodes:,}  relations={len(rels):,}  "
          f"types={len(set(etype.values()))}  avg_degree={2*n_edges/n_nodes:.1f}\n")

    res = {}
    tmp = f"/tmp/loom_kg_bench_{os.getpid()}.loom"
    if os.path.exists(tmp):
        os.unlink(tmp)

    try:
        with DB(tmp) as db:
            g = db.create_graph(
                "kg",
                node_schema={"type": "U24"},
                edge_schema={"rel": "uint16"},
                directed=True,
                node_id_max_len=32,
                label_field="type",
            )

            # ---------- BUILD ----------
            print("BUILD")
            node_items = [(e, {"type": etype[e]}) for e in entities]
            res["add_nodes"] = timed("add_nodes (bulk, typed)",
                                     lambda: g.add_nodes(node_items), n_nodes, "nodes")[0]

            edge_items = [(h, t, {"rel": rel2id[r]}) for h, r, t in triples]
            res["add_edges"] = timed("add_edges (bulk, interned rel)",
                                     lambda: g.add_edges(edge_items), n_edges, "edges")[0]
            db.flush()

            size = os.path.getsize(tmp)
            print(f"  on-disk: {fmt_bytes(size)}  "
                  f"({size / n_edges:.0f} B/edge, {size / n_nodes:.0f} B/node)\n")
            res["bytes_per_edge"] = size / n_edges

            # ---------- per-call add_edge (small sample, fresh structure) ----------
            sample_e = edge_items[:5000]
            g2 = db.create_graph("kg2", {"type": "U24"}, {"rel": "uint16"},
                                 directed=True, node_id_max_len=32)
            def percall():
                for h, t, a in sample_e:
                    g2.add_edge(h, t, **a)
            res["add_edge_percall"] = timed("add_edge (per-call)", percall, len(sample_e), "edges")[0]

            # ---------- POINT READS ----------
            print("\nPOINT READS (random samples)")
            rng = random.Random(42)
            S = min(args.samples, n_nodes)
            sample_nodes = [entities[rng.randrange(n_nodes)] for _ in range(S)]
            sample_tri = [triples[rng.randrange(n_edges)] for _ in range(S)]
            miss = [(entities[rng.randrange(n_nodes)], entities[rng.randrange(n_nodes)])
                    for _ in range(S)]

            res["get_node"] = timed("get_node",
                lambda: [g[n] for n in sample_nodes], S)[0]
            res["has_edge_hit"] = timed("has_edge (hit)",
                lambda: [g.has_edge(h, t) for h, _, t in sample_tri], S)[0]
            res["has_edge_miss"] = timed("has_edge (miss)",
                lambda: [g.has_edge(a, b) for a, b in miss], S)[0]
            res["get_edge"] = timed("get_edge",
                lambda: [g.get_edge(h, t) for h, _, t in sample_tri], S)[0]
            res["out_degree"] = timed("out_degree",
                lambda: [g.out_degree(n) for n in sample_nodes], S)[0]

            visited = [0]
            def walk():
                tot = 0
                for n in sample_nodes:
                    for _ in g.neighbors(n):
                        tot += 1
                visited[0] = tot
            r_nb, _, _ = timed("neighbors (per source node)", walk, S, "nodes")
            res["neighbors_per_node"] = r_nb
            res["neighbors_edges_s"] = visited[0] / (S / r_nb) if r_nb else 0
            print(f"    └ visited {visited[0]:,} edges → "
                  f"{res['neighbors_edges_s']:,.0f} edges/s")

            # ---------- LABEL INDEX ----------
            print("\nLABEL INDEX")
            r, dt, _ = timed("nodes_with_label (1st call = build)",
                lambda: g.nodes_with_label("film"), 1, "calls")
            res["label_index_build_s"] = dt
            res["nodes_with_label_warm"] = timed("nodes_with_label (warm)",
                lambda: [g.nodes_with_label(l) for l in
                         ("film", "people", "music", "award", "location")], 5, "calls")[0]
            by_label = Counter(etype.values())
            print(f"    └ {len(by_label)} labels; biggest: {by_label.most_common(3)}")

            # ---------- QUERY ENGINE ----------
            print("\nQUERY ENGINE")
            # high-out-degree hubs for traversal queries
            outdeg = Counter()
            for h, _, _ in triples:
                outdeg[h] += 1
            hubs = [h for h, _ in outdeg.most_common(args.queries)]
            mids = [entities[rng.randrange(n_nodes)] for _ in range(args.queries)]

            res["q_1hop_seeded"] = timed("1-hop  id-seeded  (a)->(b)",
                lambda: [g.query(f"MATCH (a)->(b) WHERE id(a)=='{h}' RETURN id(b)")
                         for h in hubs], len(hubs), "queries")[0]
            res["q_2hop_chain"] = timed("2-hop  chain      (a)->(b)->(c)",
                lambda: [g.query(f"MATCH (a)->(b)->(c) WHERE id(a)=='{h}' RETURN id(c) LIMIT 100")
                         for h in mids], len(mids), "queries")[0]
            res["q_varlen2"] = timed("var-len [*2]      (a)-[*2]->(b)",
                lambda: [g.query(f"MATCH (a)-[*2]->(b) WHERE id(a)=='{h}' RETURN id(b) LIMIT 100")
                         for h in mids], len(mids), "queries")[0]
            res["q_label_seeded"] = timed("label-seeded (a:Type)->(b) LIMIT 50",
                lambda: [g.query(f"MATCH (a:{l})->(b) RETURN id(b) LIMIT 50")
                         for l in ("olympics", "education", "tv", "government", "music")],
                5, "queries")[0]

            # ---------- REOPEN ----------
            print("\nREOPEN")
        gc.collect()
        t0 = time.perf_counter()
        db2 = DB(tmp)
        g3 = db2._datastructures["kg"]
        _ = g3.num_nodes
        print(f"  reopen + load: {(time.perf_counter()-t0)*1000:.1f} ms")
        t0 = time.perf_counter()
        _ = g3.query(f"MATCH (a)->(b) WHERE id(a)=='{hubs[0]}' RETURN id(b)")
        print(f"  first query after reopen: {(time.perf_counter()-t0)*1000:.1f} ms")
        db2.close()

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    # ---------- SUMMARY ----------
    print("\n" + "=" * 66)
    print(f"SUMMARY — FB15k  {n_nodes:,} nodes / {n_edges:,} edges (typed, directed)")
    print("=" * 66)
    print("| Phase | Operation | Throughput |")
    print("|---|---|---:|")
    rows = [
        ("Build", "add_nodes (bulk)", res["add_nodes"], "nodes/s"),
        ("Build", "add_edges (bulk)", res["add_edges"], "edges/s"),
        ("Build", "add_edge (per-call)", res["add_edge_percall"], "edges/s"),
        ("Read", "get_node", res["get_node"], "ops/s"),
        ("Read", "has_edge (hit)", res["has_edge_hit"], "ops/s"),
        ("Read", "has_edge (miss)", res["has_edge_miss"], "ops/s"),
        ("Read", "get_edge", res["get_edge"], "ops/s"),
        ("Read", "out_degree", res["out_degree"], "ops/s"),
        ("Read", "neighbors", res["neighbors_edges_s"], "edges/s"),
        ("Label", "nodes_with_label (warm)", res["nodes_with_label_warm"], "calls/s"),
        ("Query", "1-hop id-seeded", res["q_1hop_seeded"], "queries/s"),
        ("Query", "2-hop chain", res["q_2hop_chain"], "queries/s"),
        ("Query", "var-len [*2]", res["q_varlen2"], "queries/s"),
        ("Query", "label-seeded 1-hop", res["q_label_seeded"], "queries/s"),
    ]
    for phase, op, rate, unit in rows:
        print(f"| {phase} | {op} | {rate:,.0f} {unit} |")
    print(f"\nDisk: {res['bytes_per_edge']:.0f} B/edge | "
          f"label-index build: {res['label_index_build_s']*1000:.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=int, default=None,
                    help="cap total triples loaded (default: all of --splits)")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="which FB15k splits to load (default: train)")
    ap.add_argument("--samples", type=int, default=10000,
                    help="random samples for point-read benchmarks")
    ap.add_argument("--queries", type=int, default=200,
                    help="number of source nodes for query benchmarks")
    run(ap.parse_args())
