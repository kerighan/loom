"""Collection benchmark on synthetic social-media posts.

Exercises the four index kinds at once: primary (id), many (username, recent
first), range (engagement).  Measures build, point/range/find reads, the
atomic like increment (which re-indexes), on-disk size and reopen.

Run: PYTHONPATH=. python benchmarks/benchmark_collection.py [n_posts]
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB
from loom import Many


SCHEMA = {
    "id": "utf8[16]", "username": "utf8[24]", "created_at": "int64",
    "engagement": "int64", "likes": "int64", "text": "text",
}


def gen_posts(n, rng):
    users = [f"user{u}" for u in range(max(1, n // 50))]   # ~50 posts/user
    return [{
        "id": f"p{i}",
        "username": rng.choice(users),
        "created_at": i,
        "engagement": rng.randint(0, 10000),
        "likes": 0,
        "text": f"post {i} about topic {rng.randint(0, 99)}",
    } for i in range(n)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    rng = random.Random(0)
    posts_data = gen_posts(n, rng)
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    rows = []
    try:
        db = DB(path)
        posts = db.collection("posts", SCHEMA, indexes={
            "id": "primary",
            "username": Many(sort="created_at", desc=True),
            "engagement": "range",
        })

        t = time.time()
        posts.insert_many(posts_data)
        build = time.time() - t
        db.flush()
        rows.append(("Build", "insert_many", f"{n / build:,.0f} posts/s ({build:.1f}s)"))
        rows.append(("Disk", "used", f"{db._db.get_used_space() / 1024 / 1024:.1f} MB"))

        users = list({p["username"] for p in posts_data})
        ids = [p["id"] for p in posts_data]

        def bench(label, fn, k=10_000):
            t = time.time()
            c = 0
            for _ in range(k):
                c += fn()
            dt = time.time() - t
            rows.append(("Read", label, f"{k / dt:,.0f} ops/s ({1e6 * dt / k:.1f} µs)"))

        bench("get by id", lambda: 1 if posts[rng.choice(ids)] else 0)
        bench("find(username) [feed]",
              lambda: 1 if posts.find("username", rng.choice(users), limit=20) else 0)
        bench("range(engagement>=9000)",
              lambda: 1 if posts.range("engagement", 9000, None, limit=20) else 0, k=2_000)
        bench("increment likes (no reindex)",
              lambda: 1 if posts.increment(rng.choice(ids), "likes") else 0)
        bench("increment engagement (reindex)",
              lambda: 1 if posts.increment(rng.choice(ids), "engagement", 1) else 0, k=5_000)

        db.close()
        t = time.time()
        db = DB(path)
        posts = db.collection("posts")
        rows.append(("Reopen", "reopen", f"{(time.time() - t) * 1000:.1f} ms"))
        db.close()
    finally:
        os.unlink(path)

    print("\n" + "=" * 58)
    print(f"SUMMARY — Collection / posts  (n={n:,})")
    print("=" * 58)
    print("| Phase | Operation | Result |")
    print("|---|---|---:|")
    for ph, op, val in rows:
        print(f"| {ph} | {op} | {val} |")


if __name__ == "__main__":
    main()
