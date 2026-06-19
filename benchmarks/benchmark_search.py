"""SearchIndex benchmark on the 20 Newsgroups corpus (~18.8k documents).

Measures build throughput + on-disk size of the inverted index, boolean and
BM25 query throughput, and reopen latency.

Data is fetched once via scikit-learn (cached under ~/scikit_learn_data):
    pip install scikit-learn eldar
Run with:
    PYTHONPATH=. python benchmarks/benchmark_search.py [n_docs]
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB


def load_docs(limit=None):
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="all", remove=())
    docs = [str(t) for t in data.data]
    if limit:
        docs = docs[:limit]
    return docs


def timed(label, fn):
    t = time.time()
    out = fn()
    dt = time.time() - t
    return out, dt


def run_query_set(idx, queries, mode=None, limit=10):
    t = time.time()
    total = 0
    for q in queries:
        res = idx.search(q, return_ids=True, mode=mode, limit=limit)
        total += len(res)
    dt = time.time() - t
    return len(queries) / dt, total


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print("Loading 20 Newsgroups ...")
    docs = load_docs(limit)
    n = len(docs)
    print(f"  {n:,} documents")

    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)

    rng = random.Random(0)
    rows = []

    try:
        db = DB(path)
        idx = db.create_search_index("news", scoring="bm25")

        # ── build ────────────────────────────────────────────────────────────
        t = time.time()
        for d in docs:
            idx.add(d)
        build_dt = time.time() - t
        db.flush()

        vocab = len(idx._postings)
        avg_len = int(idx._meta["total_len"]["v"]) / max(n, 1)
        disk_mb = os.path.getsize(path) / (1024 * 1024)

        rows.append(("Build", "add (per-doc, indexed)", f"{n / build_dt:,.0f} docs/s"))
        rows.append(("Build", "total build time", f"{build_dt:.1f} s"))
        rows.append(("Corpus", "documents", f"{n:,}"))
        rows.append(("Corpus", "vocabulary (distinct terms)", f"{vocab:,}"))
        rows.append(("Corpus", "avg doc length (tokens)", f"{avg_len:.0f}"))
        rows.append(("Disk", "index on disk", f"{disk_mb:.1f} MB"))

        # ── pick query terms: mid-frequency terms (skip ultra-rare & ultra-common)
        terms = list(idx._postings.keys())
        rng.shuffle(terms)
        sampled = []
        for term in terms:
            if not term.isalpha() or len(term) < 4:
                continue
            df = idx.document_frequency(term)
            if 20 <= df <= n // 10:
                sampled.append(term)
            if len(sampled) >= 400:
                break
        print(f"  sampled {len(sampled)} mid-frequency query terms")

        def pairs():
            return [(rng.choice(sampled), rng.choice(sampled)) for _ in range(200)]

        single = [rng.choice(sampled) for _ in range(200)]
        q_and = [f"{a} AND {b}" for a, b in pairs()]
        q_or = [f"{a} OR {b}" for a, b in pairs()]
        q_andnot = [f"{a} AND NOT {b}" for a, b in pairs()]

        # ── queries ───────────────────────────────────────────────────────────
        qps, _ = run_query_set(idx, single, mode="boolean")
        rows.append(("Boolean", "single term", f"{qps:,.0f} q/s"))
        qps, _ = run_query_set(idx, q_and, mode="boolean")
        rows.append(("Boolean", "A AND B", f"{qps:,.0f} q/s"))
        qps, _ = run_query_set(idx, q_or, mode="boolean")
        rows.append(("Boolean", "A OR B", f"{qps:,.0f} q/s"))
        qps, _ = run_query_set(idx, q_andnot, mode="boolean")
        rows.append(("Boolean", "A AND NOT B", f"{qps:,.0f} q/s"))

        qps, _ = run_query_set(idx, single, mode="bm25")
        rows.append(("BM25", "single term, top-10", f"{qps:,.0f} q/s"))
        qps, _ = run_query_set(idx, q_and, mode="bm25")
        rows.append(("BM25", "A AND B, top-10", f"{qps:,.0f} q/s"))
        qps, _ = run_query_set(idx, q_or, mode="bm25")
        rows.append(("BM25", "A OR B, top-10", f"{qps:,.0f} q/s"))

        db.close()

        # ── reopen ────────────────────────────────────────────────────────────
        t = time.time()
        db = DB(path)
        idx = db._datastructures["news"]
        reopen_ms = (time.time() - t) * 1000
        t = time.time()
        _ = idx.search(single[0], return_ids=True, mode="bm25", limit=10)
        first_q_ms = (time.time() - t) * 1000
        rows.append(("Reopen", "reopen + load", f"{reopen_ms:.1f} ms"))
        rows.append(("Reopen", "first query after reopen", f"{first_q_ms:.1f} ms"))
        db.close()
    finally:
        os.unlink(path)

    print("\n" + "=" * 60)
    print(f"SUMMARY — 20 Newsgroups  ({n:,} docs, BM25 index)")
    print("=" * 60)
    print("| Phase | Operation | Result |")
    print("|---|---|---:|")
    for phase, op, val in rows:
        print(f"| {phase} | {op} | {val} |")


if __name__ == "__main__":
    main()
