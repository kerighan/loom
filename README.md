# loom

**Persistent Python data structures that feel native.**

loom is a file-backed database library that lets you work with `Dict`, `List`, `Queue`, `Set`, `BTree`, `Graph`, and **vector indexes** exactly like their in-memory counterparts — but stored on disk with mmap zero-copy access, crash-safe writes, and automatic space reclamation.

No server. No ORM. No SQL. Just Python objects that persist.

```python
from pydantic import BaseModel, Field
from loom.database import DB

class User(BaseModel):
    id:       int
    username: str = Field(max_length=50)  # fixed-length, fast lookups
    bio:      str                          # variable-length text, compressed
    score:    float

with DB("app.db") as db:
    users = db.create_dataset("users", User)
    dct   = db.create_dict("users_by_name", users)

    dct["alice"] = {"id": 1, "username": "alice", "bio": "Hello!", "score": 9.5}
    print(dct["alice"])          # {'id': 1, 'username': 'alice', ...}
    print("alice" in dct)        # True
```

---

## Installation

```bash
pip install loom-db          # coming soon — for now: clone + pip install -e .
```

Dependencies: `numpy`, `lru-dict`, `mmh3`, `pydantic` (optional), `brotli` (optional).

---

## Core concepts

### Schemas with Pydantic

Define your record schema as a Pydantic model. loom maps types automatically:

| Python / Pydantic type | loom dtype | Notes |
|---|---|---|
| `int` | `int64` | |
| `float` | `float64` | |
| `bool` | `bool` | |
| `str` | `text` | Variable-length, compressed via BlobStore |
| `str = Field(max_length=N)` | `U{N}` | Fixed-length numpy unicode, faster reads |
| `FixedStr(N)` | `U{N}` | loom shorthand |

```python
from pydantic import BaseModel, Field
from loom.schema import FixedStr

class Message(BaseModel):
    id:      int
    role:    str = Field(max_length=20)  # → U20, fast key
    content: str                          # → text, variable-length

class Product(BaseModel):
    sku:    FixedStr(20)   # → U20
    name:   str            # → text
    price:  float
    stock:  int
```

You can also pass plain dicts:

```python
db.create_dataset("events", id="uint32", ts="int64", kind="U20")
```

**Rule of thumb:** use `Field(max_length=N)` or `FixedStr(N)` for short fields used as keys or in frequent filters. Use plain `str` (→ `text`) for anything that can be long or variable — body text, descriptions, JSON payloads, etc.

### The DB object

```python
# Open (creates if not exists)
db = DB("mydata.db")

# Context manager (recommended — auto-saves and flushes on exit)
with DB("mydata.db",
        blob_compression=None,        # None (default, fastest) | "brotli" | "zlib"
        auto_save_interval=100,       # metadata save frequency (default 100)
        cache_size=50_000,            # shared LRU entries across ALL structures
        sync_writes=False,            # False=fast (flush on close), True=safe (flush every write)
) as db:
    ...

# db.batch() is useful mainly for text/blob fields — for fixed schemas the
# lazy flush already makes per-call inserts as fast as batch inserts.
with db.batch():
    for item in large_dataset:
        my_dict[item["key"]] = item

# Long-running servers: call flush() periodically instead of sync_writes=True
db.flush()  # force mmap writeback without closing
```

---

## Data structures

### Dataset — typed record store

```python
users = db.create_dataset("users", User)

# Insert → returns a Ref (address + dataset handle)
ref = users.insert({"id": 1, "username": "alice", "bio": "...", "score": 9.5})

# Read / update / delete
rec = users.read(ref.addr)
ref.update(score=10.0)             # fast field-level update
ref.update(bio="New bio!")         # frees old blob, writes new one
users.delete(ref.addr)             # soft delete
```

### Dict — persistent hash map

```python
ds  = db.create_dataset("users", User)
dct = db.create_dict("users_dict", ds)

dct["alice"] = {"id": 1, "username": "alice", "bio": "...", "score": 9.5}
print(dct["alice"])
print("alice" in dct)              # O(1), bloom-filter accelerated
del dct["alice"]

# Bulk export — one mmap read + sorted blob resolution
snapshot = dct.to_dict()           # {key: record_dict, ...}

# Iteration
for key, val in dct.items(): ...
list(dct.keys())
list(dct.values())
```

### List — dynamic array

```python
lst = db.create_list("events", event_ds)

lst.append({"id": 1, "ts": 1700000000, "kind": "click"})
print(lst[0])
print(lst[-1])
print(lst[10:20])                  # slice
del lst[5]                         # soft delete, auto-compacts at 30% waste

for item in lst: ...
```

### Queue — FIFO with O(1) push/pop and automatic space reclamation

```python
q = db.create_queue("tasks", Task, block_size=64)

q.push({"id": 1, "payload": "process me", "priority": 0.9})
q.push_many([...])                  # bulk push

item = q.peek()                     # look without consuming
item = q.pop()                      # consume — exhausted blocks freed automatically

for item in q: ...                  # non-destructive iteration
print(len(q))
```

When a block is exhausted by `pop()`, it is returned to the shared ByteFileDB freelist. In steady state (push rate ≈ pop rate), **the file does not grow**.

### BTree — ordered key-value with range queries

```python
idx = db.create_btree("docs_by_title", doc_ds, key_size=100)

idx["Alpha post"] = {"id": 1, ...}
idx["Beta post"]  = {"id": 2, ...}

for k in idx.keys():              # sorted order
    print(k)

for k, v in idx.range("A", "C"):  # range query
    print(k, v)

for k, v in idx.prefix("Beta"):   # prefix search
    print(k, v)
```

### Set — unique string collection

```python
tags = db.create_set("popular_tags", key_size=50)

tags.add("python")
tags.add("machine-learning")
print("python" in tags)            # True
tags.remove("python")

for tag in tags: ...
```

### FlatIndex / IVFIndex — vector similarity search

```python
import numpy as np

# Exact search — no training
flat = db.create_flat_index("passages", dim=1536, metric="cosine")
flat.add("doc_1", embedding_array)                       # np.ndarray
flat.add_batch([("doc_1", v1), ("doc_2", v2)])
results = flat.search(query_vec, k=10)
# [("doc_1", 0.95), ("doc_2", 0.87), ...]

# Approximate — requires training
ivf = db.create_ivf_index("passages", dim=1536,
                            n_clusters=256,   # √n_vecs is a good default
                            pq=False)         # True for PQ compression
ivf.train(sample_matrix)                      # np.ndarray (n_sample, dim)
ivf.add_batch([("doc_1", v1), ("doc_2", v2)])
results = ivf.search(query_vec, k=10, nprobe=32)

# With PQ compression (n_sub bytes per vector)
ivf_pq = db.create_ivf_index("passages", dim=1536,
                               n_clusters=256, pq=True, n_sub=16)
# 1536 dims × float32 = 6 144 bytes  →  16 bytes with PQ (384× compression)
```

Metrics: `"cosine"` (default), `"l2"`, `"dot"`.  
Vectors are L2-normalised at insert time for cosine similarity.

### BloomFilter / CountingBloomFilter — probabilistic membership

```python
seen = db.create_bloomfilter("seen_ids", expected_items=1_000_000)
seen.add("user_42")
if "user_42" in seen:
    print("probably seen before")   # no false negatives

# CountingBloomFilter supports removal
cbf = db.create_counting_bloomfilter("cache", expected_items=10_000)
cbf.add("item")
cbf.remove("item")
```

### Graph — persistent directed/undirected graph with Cypher queries

```python
class Person(BaseModel):
    name: str
    age:  int

class Follows(BaseModel):
    weight: float
    since:  int

g = db.create_graph("social", Person, Follows,
                    directed=True, node_id_max_len=50)

g.add_node("alice", name="Alice", age=30)
g.add_node("bob",   name="Bob",   age=25)
g.add_edge("alice", "bob", weight=0.9, since=2022)

# Bulk insert (recommended for large graphs — slightly lower overhead)
g.add_nodes([("alice", {"name": "Alice", "age": 30}),
             ("bob",   {"name": "Bob",   "age": 25})])
g.add_edges([("alice", "bob",   {"weight": 0.9, "since": 2022}),
             ("alice", "carol", {"weight": 0.5, "since": 2021})])

print(g["alice"])                        # node attrs
print(g.has_edge("alice", "bob"))        # O(1)
print(list(g.neighbors("alice")))        # outgoing neighbors
print(list(g.predecessors("bob")))       # incoming neighbors

# Cypher-like queries
results = g.query("""
    MATCH (a)-[r]->(b)
    WHERE a.age > 25 AND r.weight > 0.7
    RETURN a.name, b.name, r.weight
""")

# Path quantifiers
g.query("MATCH (a)-[+]->(b) WHERE id(a) == 'alice' RETURN id(b)")      # 1..∞ hops
g.query("MATCH (a)-[*2..4]->(b) WHERE id(a) == 'alice' RETURN id(b)")  # 2–4 hops

# Filter by node key
g.query("MATCH (a)->(b) WHERE id(a) IN ['alice','bob'] RETURN b.name")
g.query("MATCH (a {name:'Alice'})->(b) RETURN id(b)")   # inline node props

# LIMIT
g.query("MATCH (a)->(b) WHERE a.age > 20 RETURN a.name LIMIT 5")

# Lazy iterator for large results
for row in g.query_iter("MATCH (a)-[+]->(b) WHERE id(a)=='alice' RETURN id(b)"):
    process(row)
```

---

## Nesting

Compose structures freely. loom validates compatibility at creation time and raises `NestingNotSupportedError` for invalid combinations.

```python
from loom.datastructures import List, Dict, Queue, Set

# Dict[List] — user → chronological feed
PostList  = List.template(post_ds)
user_feed = db.create_dict("feeds", PostList)
user_feed["alice"].append({"id": 1, "title": "Hello"})

# Dict[Dict] — user → posts by slug
PostDict   = Dict.template(post_ds)
user_posts = db.create_dict("user_posts", PostDict)
user_posts["alice"]["intro"] = {"id": 1, "title": "Intro"}

# Dict[Queue] — per-user task queues
TaskQueue   = Queue.template(task_ds, block_size=32)
user_queues = db.create_dict("user_queues", TaskQueue)
user_queues["alice"].push({"id": 1, "payload": "run job"})
item = user_queues["alice"].pop()

# List[Queue] — event channels
EventQueue = Queue.template(event_ds)
channels   = db.create_list("channels", EventQueue)
ch0        = channels.append()
ch0.push({"ts": 1700000000, "kind": "click"})

# Dict[Set] — per-user tag sets
TagSet    = Set.template(key_size=50)
user_tags = db.create_dict("user_tags", TagSet)
user_tags["alice"].add("python")
```

### Nesting compatibility matrix

| Inner ↓ \ Outer → | Dict | List | BTree |
|---|:---:|:---:|:---:|
| **List** | ✓ | ✓ | ✗ |
| **Dict** | ✓ | ✓ | ✓ |
| **Queue** | ✓ | ✓ | ✗ |
| **Set** | ✓ | ✓ | ✗ |
| **BTree** | ✓ | ✓ | ✗ |
| **BloomFilter** | ✗ | ✗ | ✗ |

---

## HTTP server (optional)

Any open `DB` can expose itself as a REST API in one line. The server is built on FastAPI, so you get **interactive Swagger UI and ReDoc out of the box** — no extra wiring.

```bash
pip install 'fastapi[standard]'    # adds fastapi + uvicorn
```

```python
from loom import DB

with DB("app.db") as db:
    db.create_dataset("users", id="uint32", username="U50", age="uint32")
    db.create_dict("by_username", db._datasets["users"])
    db.create_set("active")

    db.serve(host="127.0.0.1", port=8000)   # blocking
```

Then point your browser at:

| URL | What you get |
|---|---|
| `http://127.0.0.1:8000/docs` | **Swagger UI** — interactive try-it-out for every endpoint |
| `http://127.0.0.1:8000/redoc` | **ReDoc** — clean reference-style docs |
| `http://127.0.0.1:8000/openapi.json` | Raw OpenAPI 3 schema (feed it to Postman, codegen, etc.) |
| `http://127.0.0.1:8000/` | Index: filename, datasets, structures, links to the docs |

Every dataset, `Dict`, `List`, `Set`, `BTree`, `Queue`, `BloomFilter`, `LRUDict` you create gets a CRUD route family automatically (`/datasets/<name>`, `/dicts/<name>/items/<key>`, `/btrees/<name>/range`, …). Request bodies are validated against your loom schema via Pydantic models generated on the fly.

```python
# Mount inside your own ASGI stack instead of using db.serve()
app = db.fastapi_app()      # standard FastAPI instance
# uvicorn.run(app, host=..., port=...)
```

> **Concurrency**: loom is single-writer / single-reader. The server takes a single process-wide lock so requests are serialised against the underlying mmap. Safe from any number of clients, but throughput is bounded by one writer at a time.

---

## Error handling

```python
import loom

try:
    db["nonexistent"]
except loom.StructureNotFoundError as e:
    print(e.name)

try:
    dataset.read(deleted_addr)
except loom.DeletedRecordError as e:
    print(e.address)

try:
    db.create_dataset("users", User)
    db.create_dataset("users", User)  # duplicate
except loom.DuplicateNameError as e:
    print(e.name, e.kind)

try:
    BloomFilter._check_nesting(Dict)   # unsupported combination
except loom.NestingNotSupportedError as e:
    print(e.outer, e.inner)
```

Full exception hierarchy (`loom.LoomError` is the base):

```
LoomError
├── DatabaseError
│   ├── DatabaseNotOpenError
│   ├── DuplicateNameError
│   ├── StructureNotFoundError   (also KeyError)
│   └── NestingNotSupportedError
├── HeaderError
│   └── HeaderTooLargeError
├── SchemaError
│   ├── InvalidIdentifierError
│   └── UnknownDtypeError
└── RecordError
    ├── DeletedRecordError
    └── WrongDatasetError
```

---

## Performance

All benchmarks: modern laptop (Linux, SSD), `sync_writes=False` (default — durable on `close()`).

All numbers below come from `benchmarks/readme_benchmark.py` (run it yourself with `PYTHONPATH=. python benchmarks/readme_benchmark.py`). Figures are rounded; expect ±10% run-to-run variance from page-cache warmup and CPU frequency.

### loom vs SqliteDict — 10 000 ops, fixed schema

| Operation | **loom** | **SqliteDict** | Ratio |
|---|---:|---:|---:|
| Dict insert (per-call) | **27 000 ops/s** | 4 900 ops/s (autocommit) | loom **5.5×** |
| Dict insert (batch) | **28 000 ops/s** | 22 000 ops/s (single `COMMIT`) | loom **1.3×** |
| Dict read | **55 000 ops/s** | 11 200 ops/s | loom **4.9×** |
| Dict contains | **95 000 ops/s** | 11 800 ops/s | loom **8×** |
| Dict keys() | **180 000 ops/s** | 115 000 ops/s | loom **1.6×** |

loom is faster on every line, including SQLite's most favourable mode (defer all writes to a single transaction). The gap is widest on point ops because loom uses lazy mmap flush — no `msync()` per write, just OS page-cache writeback — while SQLite has to walk a B-tree per call. `contains` benefits from binary murmur128 slots (25 bytes regardless of key length), which makes the inner loop a couple of integer compares.

Per-call inserts already run at batch speed in loom (the flush is already lazy), so wrapping inserts in `db.batch()` mostly matters for `text` / blob fields, where it amortises one BlobStore flush across many writes.

### loom operations — all structures

| Structure | Operation | ops/s | µs/op |
|---|---|---:|---:|
| Dict | insert | 27 000 | 37 |
| Dict | read | 55 000 | 18 |
| Dict | contains | 90 000 | 11 |
| Dict | keys() | 185 000 | 5 |
| Dict | items() | 90 000 | 11 |
| List | append | 105 000 | 10 |
| List | read[i] | 165 000 | 6 |
| Queue | push (batch) | 220 000 | 5 |
| Queue | pop | 260 000 | 4 |

### `str` (text) fields — impact of variable-length blobs

`str` fields bypass the fixed record and write to a separate BlobStore — slower but space-efficient.

| Schema | Compression | insert | read |
|---|---|---:|---:|
| Fixed fields only | — | 27 000 ops/s | 55 000 ops/s |
| + `str` body (≈600 chars) | None (default) | 29 000 ops/s | 54 000 ops/s |
| + `str` body (≈600 chars) | brotli | 1 400 ops/s | 32 000 ops/s |

- `blob_compression=None` (**default**) — fastest writes, larger files.
- `"brotli"` — 3–5× compression on natural language, but ~20× slower inserts; pick when storage > write throughput.
- `Field(max_length=N)` — keeps the field in the fixed record, no BlobStore at all.

### Graph — Barabási-Albert scale-free network (m=2, 20 000 nodes / 39 996 edges, directed)

Build phase (write):

| Operation | ops/s | µs/op |
|---|---:|---:|
| `add_node` | 25 000 | 40 |
| `add_edge` (per-call) | 2 900 | 345 |
| `add_edges` (bulk, in `db.batch()`) | 3 300 | 305 |

Each `add_edge` does two double-indexed nested-Dict writes (`_out[src][dst]` and `_in[dst][src]`); each of those probes the parent's hash table, finds / grows the child's slot block, writes the value record, and rewrites the parent's ref. Bulk insertion via `g.add_edges([...])` is ~15 % faster — most of the difference is one deferred header save per `db.allocate()` (`db.batch()` accumulates them).

Read phase, point queries (10 000 random samples):

| Operation | ops/s | µs/op |
|---|---:|---:|
| `g[node_id]` (`get_node`) | 57 000 | 18 |
| `has_edge` (hit) | 17 000 | 60 |
| `has_edge` (miss) | 16 000 | 60 |
| `get_edge` | 19 000 | 52 |
| `out_degree` | 20 000 | 50 |
| `neighbors(node_id)` (per source) | 12 000 | 80 |

`neighbors()` visits edges at **~25 000 edges/s** when iterating the full neighbour list of each sampled node — useful for BFS / PageRank style passes.

On-disk size: ~10 KB / edge in this configuration (double-indexed `_out` + `_in` plus per-source nested-Dict overhead — each new source allocates an 8-slot hash block + 8-slot values block, which doubles on demand). The dominant cost is the per-source nested table; sparser graphs (low average degree) pay a higher per-edge constant than dense ones.

### Vector search — FlatIndex and IVFIndex

Benchmarks: 10 000 vectors, dim=128, cosine similarity, 50 random queries.
K=100 IVF clusters.  Training on 5 000 samples.

**Insert throughput:**

| Index | insert | train |
|---|---:|---|
| FlatIndex | 10 600 vecs/s | — (no training) |
| IVFFlat | 9 800 vecs/s | 5 s |
| IVF+PQ M=16 | 2 700 vecs/s | 38 s |

**Search — recall@10 vs nprobe (IVFFlat, K=100):**

| nprobe | ms/query | recall@10 |
|---|---:|---:|
| 8 (8%) | 3.2 ms | 32% |
| 16 (16%) | 4.2 ms | 47% |
| 32 (32%) | 5.4 ms | 68% |
| 50 (50%) | 6.5 ms | 82% |
| 99 (all) | 6.3 ms | 100% |

FlatIndex (exact): **6.4 ms/query**.

**IVF+PQ M=16 (32× compression)** — same query times (~2–3 ms) but lower recall (15–21%) due to the V1 approximation in ADC residual computation.

**V1 limitations and roadmap:**
- IVFFlat does not yet have per-cell sequential inverted lists; it loads all records and filters by `cell_id` in numpy. Query time is therefore similar to FlatIndex regardless of nprobe. The speedup comes from the reduced comparison set, not from skipping disk I/O.
- IVF+PQ residual computation uses a single approximate centroid at search time, which reduces recall. V2 will compute per-cell residuals.
- Both will improve in V2 with true inverted list access (per-cell contiguous blocks) and correct per-cell residuals.

---

## Reliability

**Crash safety**
- Double-buffer header: two alternating slots, single-byte atomic flip. If the process crashes mid-write, the previous slot is intact.
- Write-Ahead Log (WAL) for multi-write atomic transactions (`db.apply_writes()`).
- `auto_save_interval=N` controls how often metadata is flushed to disk (default: every 100 ops per structure). Set lower for more durability, higher for more speed.

**Durability modes**

By default (`sync_writes=False`), mmap pages are written by the OS page cache — fast, but a crash between writes and the OS flush could lose recent data. Full durability is guaranteed at `close()`.

```python
# Script / one-shot job (default): flush on exit
with DB("app.db") as db:
    db["key"] = value
# ← fully on disk here

# Long-running server: periodic explicit flush
db = DB("app.db")
db["key"] = value
db.flush()  # call every N seconds, or after critical transactions

# Full sync on every write (safe, 3× slower)
db = DB("app.db", sync_writes=True)
```

**Space reclamation**
- `ByteFileDB` freelist: freed blocks (Queue pop exhaustion, List compaction) are reused by any future allocation — the file does not grow unboundedly in steady state.
- `Dict` tracks freed value slots and reuses them for new inserts.
- `List.compact()` rebuilds without deleted items and returns old blocks to the freelist.

**Session safety**
- Multiple processes reading simultaneously: safe — each has its own independent mmap.
- Single writer + multiple readers: safe for fixed-schema fields. The writer always writes the value record before making the hash table entry visible. Avoid concurrent reads of a record whose `text` field is being updated.

---

## When to use loom

| Scenario | Recommendation |
|---|---|
| Fast persistent key-value, read-heavy | **loom Dict** |
| Ordered data, range / prefix queries | **loom BTree** |
| FIFO task queue, event stream | **loom Queue** |
| Graph data, social / knowledge networks | **loom Graph** |
| Large text content per record | **loom + `str` + brotli** |
| Complex relational queries (JOINs) | SQLite / PostgreSQL |
| Concurrent multi-writer | SQLite / PostgreSQL |
| Pure analytics, columnar scans | Parquet / DuckDB |
