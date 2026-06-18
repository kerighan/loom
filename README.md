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

> **Variable-length hop semantics.** Quantifiers (`[*N]`, `[*lo..hi]`, `[+]`, `[*]`)
> use **shortest-path BFS**: each destination node is reported once, at its
> *minimum* hop distance from the source. So `[*N]` means "nodes whose shortest
> path is exactly N hops" — **not** "every endpoint of an N-step walk". A node
> reachable in both 1 and 2 hops appears only for `[*1]`, and `[*2]` may return
> nothing even when 2-hop paths exist. To get the **whole neighbourhood within N
> hops**, use a range from 1: `MATCH (a)-[*1..N]->(b)`. This is a deliberate
> design choice (cheap, no path explosion); it is not Neo4j's path-enumeration
> semantics.

#### Knowledge graphs — node labels and multi-hop chains

Declare a `label_field` (a node-schema field that holds each node's type) to
unlock label syntax and multi-hop pattern chains — enough to model a typed
knowledge graph like `category → company → employee`:

```python
g = db.create_graph(
    "kg",
    node_schema={"type": "U20", "name": "U40", "sector": "U20"},
    edge_schema={"rel": "U20"},
    directed=True,
    label_field="type",          # ← which field is the node label
)

g.add_nodes([
    ("tech",   {"type": "category", "name": "Tech"}),
    ("acme",   {"type": "company",  "name": "Acme",  "sector": "SaaS"}),
    ("alice",  {"type": "employee", "name": "Alice"}),
])
g.add_edges([("tech", "acme", {"rel": "has"}), ("acme", "alice", {"rel": "emp"})])

# Label filter:  (var:Label)  →  fast, seeded via an in-memory label→nodes index
g.query("MATCH (a:company)->(b:employee) RETURN id(a), id(b)")

# Multi-hop chain with a per-hop constraint on the *intermediate* node
g.query("""
    MATCH (a:category)->(b:company)->(c:employee)
    WHERE b.sector == 'SaaS'
    RETURN id(c), c.name
""")

# label(x) sugar in WHERE / RETURN; direct lookup of all nodes of a class
g.query("MATCH (a)->(b) WHERE label(a) == 'company' RETURN id(b)")
g.nodes_with_label("company")        # ['acme', ...]
```

Notes:
- `(a:Label)` compiles to a filter on `label_field`; when it anchors the first
  node of a pattern, the query is **seeded** from the label index (no full
  scan). The index is in-memory, built lazily on first use, and rebuilt after
  node mutations — it is a cache, never persisted.
- Chains `(a)->(b)->(c)->…` bind and constrain every node and edge along the
  path. They support fixed single-hop edges only; the `[*]`/`[+]`/`[*a..b]`/`[?]`
  quantifiers remain available for two-node patterns.
- Nodes are heterogeneous via the shared `node_schema` (a union of fields);
  unused fields stay empty per node.
- **Use stable node ids.** Edges are keyed by node id (there is no `rename` —
  changing an id would orphan its edges), so give each entity a permanent id
  up front (its natural key, a hash, etc.) rather than a label you might later
  want to change. Mutable attributes belong in the node's fields.

---

## Collection — a record store with attached indexes

Looking records up by more than one field normally means maintaining a second
`Dict` by hand and keeping it in sync. A `Collection` does that for you: the
record lives **once** in a primary index (keyed by your primary key), and you
**attach** secondary indexes that map another field → the primary key. Every
insert / update / delete updates all of them under one lock.

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(max_length=30)
    email:    str = Field(max_length=40)
    city:     str = Field(max_length=20)

users = db.collection(
    "users", User,
    key="username",                       # primary key field
    indexes={
        "email":    "email",              # field name → persisted, auto-restored
        "city":     "city",
        "email_lc": lambda r: r["email"].lower(),  # computed key → re-supply on reopen
    },
)

users.insert({"username": "alice", "email": "Alice@x.com", "city": "NYC"})

users["alice"]                    # primary lookup
users.get("email", "Alice@x.com") # secondary lookup → full record
users.get("email_lc", "alice@x.com")
users.get_pk("city", "NYC")       # → "alice"  (primary key only, no record read)

users.update("alice", email="new@x.com")  # re-indexes the email automatically
users.delete("alice")                      # removed from every index
```

`indexes` can also be a plain list of field names: `indexes=["email", "city"]`.

- **No duplication.** Secondary indexes store the *primary key*, not a copy of
  the record — a secondary lookup is one extra hop (`index → pk → record`).
- **Persistence.** Field-name indexes are saved with the collection, so
  `db.collection("users")` (no model) reopens it and rebuilds them
  automatically. Lambda/computed indexes can't be serialised — re-pass them in
  `indexes=` on reopen.
- **Atomicity.** Writes touch all indexes under `db.write_lock()` + `db.batch()`
  (serialised, grouped). This is not a full crash-atomic WAL across indexes; if
  a crash ever desyncs an index, call `users.reindex()` to rebuild it.

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

Protect the API, docs, and dashboard with a shared token when exposing the server beyond trusted local use:

```python
db.serve(host="0.0.0.0", port=8000, dashboard=True, auth_token="change-me")
```

Clients can authenticate with `Authorization: Bearer <token>` or `X-API-Key: <token>`. Browser users can sign in from `/dashboard`; the server stores the token in an HttpOnly same-site cookie. You can also open `/dashboard?token=<token>` once to set that cookie directly.

> **Concurrency**: Within a single process, all requests are serialised through `db._lock` (threading.RLock). For multi-process deployments, open the DB with `multiprocess_safe=True` and pass `workers=N` to `db.serve()` — cross-process writes are then serialised via `fcntl.flock` on a companion `.lock` file with no additional overhead on reads.

```python
# Single process (default)
db.serve(host="0.0.0.0", port=8000)

# Multiple workers — writes serialised across processes via flock
with DB("app.db", multiprocess_safe=True) as db:
    db.serve(host="0.0.0.0", port=8000, workers=4)
```

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

Dict, List, Queue and BTree numbers come from `benchmarks/readme_benchmark.py` (run it yourself with `PYTHONPATH=. python benchmarks/readme_benchmark.py`); Set and LRUDict from their dedicated `benchmarks/benchmark_*.py`; and **Graph from `benchmarks/benchmark_graph_kg.py` on FB15k** (the reference graph benchmark — see its section below). Figures are rounded; expect ±10% run-to-run variance from page-cache warmup and CPU frequency. The Dict/List/Queue point-op numbers are measured with `cache_size=0` (cold mmap on every op) so those structures are compared on equal footing; BTree uses its default `cache_size=1024` (it is meaningless at zero cache — see the note below) and Graph uses its default cache.

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
| BTree | insert | 13 000 | 77 |
| BTree | read | 131 000 | 8 |
| BTree | contains | 280 000 | 4 |
| BTree | keys() [sorted] | 368 000 | 3 |
| Set | add | 73 000 | 14 |
| Set | contains | 49 000 | 21 |
| Set | remove | 55 000 | 18 |
| LRUDict | set (no eviction) | 145 000 | 7 |
| LRUDict | get (hit) | 115 000 | 9 |
| LRUDict | set (with eviction) | 145 000 | 7 |
| Graph | add_nodes / add_edges (bulk) | 26 000 | 38 |
| Graph | add_edge (per-call) | 1 650 | 606 |
| Graph | get_node | 54 000 | 18 |
| Graph | has_edge | 16 000 | 63 |
| Graph | neighbors | 166 000 edges/s | 6 |

The Dict/List/Queue/Set/LRUDict rows are `cache_size=0` (cold). The **BTree rows use its default `cache_size=1024`** — running a B-tree at `cache_size=0` is pathological (every node a cold mmap read) and unrepresentative; Graph rows are the FB15k reference benchmark below.

**Dict vs BTree — pick the right tool.** The B-tree is far more cache-sensitive than the Dict, because it relies on keeping its handful of internal nodes hot. At its default cache its random `read` (~131k/s) and `contains` (~280k/s) actually *beat* the Dict — the few internal nodes live in RAM, so a lookup is one leaf read; at `cache_size=0` the same `read` collapses to ~6 800/s. The Dict, by contrast, barely changes with caching (~55k read either way). So:

- **Insert**: Dict wins (~27k vs ~13k/s) — O(1) slot vs O(log n) node path.
- **Random read / contains**: BTree wins *with a warm node cache*, Dict wins at zero cache; the Dict is the safe choice when you can't keep a cache hot.
- **Ordered iteration / range / prefix**: BTree only — `keys()` comes out sorted at ~370k/s and `range()`/`prefix()` walk a contiguous key span without scanning everything.

Bottom line: use a `Dict` for pure point access, a `BTree` when you need ordering or range queries (and leave its node cache enabled).

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

### Graph — FB15k knowledge graph (reference benchmark)

The reference graph benchmark is **FB15k** (a Freebase subset): 14,951 typed
entities, 1,345 relation types, **483,142 directed edges** (avg degree ~65).
Each node carries a `type` label (relation domain) and relations are interned
to a `uint16`. Reproduce with `PYTHONPATH=. python benchmarks/benchmark_graph_kg.py`
(downloads the dataset once; see the script header).

Build (write):

| Operation | rate | µs/op |
|---|---:|---:|
| `add_nodes` (bulk) | 26 000 nodes/s | 38 |
| `add_edges` (bulk) | 26 000 edges/s | 38 |
| `add_edge` (per-call) | 1 650 edges/s | 606 |

`add_edges` groups edges by source for `_out` and by target for `_in`, then
bulk-inserts each node's whole adjacency sub-dict in one shot (one allocation
+ one parent-ref update per node) — ~16× faster than per-call `add_edge`.
**Load all edges in a single `add_edges` call** so each node hits this fast
path (re-inserting into an already-populated node falls back to a slower path).

Read (10 000 random samples):

| Operation | rate | µs/op |
|---|---:|---:|
| `g[node_id]` (`get_node`) | 54 000 ops/s | 18 |
| `has_edge` (hit) | 16 000 ops/s | 63 |
| `get_edge` | 18 000 ops/s | 55 |
| `out_degree` | 17 000 ops/s | 59 |
| `neighbors` (iterate) | **166 000 edges/s** | 6 |

Query engine (Cypher):

| Query | rate |
|---|---:|
| label-seeded 1-hop `(a:Type)->(b) LIMIT 50` | 560 queries/s |
| 2-hop chain `(a)->(b)->(c) LIMIT 100` | 300 queries/s |
| 1-hop from a hub `id(a)=='X'` (all neighbours) | 85 queries/s |
| variable-length `(a)-[*2]->(b) LIMIT 100` | 28 queries/s |

The label index (`nodes_with_label`, used to seed `(a:Type)…` queries) builds
in ~90 ms for 15k nodes and then serves ~37 000 lookups/s. Reopen is ~1 ms.

On-disk: **~575 bytes/edge** (277 MB for 483k edges, double-indexed `_out` +
`_in`). What's left is mostly the live edge records (each direction stores the
edge once) plus the per-source nested hash tables; the node-id strings are a
minor part. Graphs with few nodes benefit even more from the small default
values-block (see note) — a sparse graph that used to be ~3× this size is now
close to its live data.

> **Node ids are the adjacency key — pick stable ids.** Edges are keyed by node
> id, so there is no `rename`: changing a node's id would orphan its edges.
> Assign each entity a stable id up front (its own key, a hash, etc.) and keep
> it for the node's lifetime.

### Vector search — FlatIndex and IVFIndex

Benchmarks: **100 000 vectors, dim=384**, cosine similarity, 200 random queries.  
K=256 clusters (≈ √N), training on 20 000 samples.  All timings include full Python round-trip.

**IVFIndex storage layout:** `List[List]` — K outer slots, one inner List per cluster.
`cell_vecs[cell_id]` is a direct O(1) integer index, no hashmap probe.
Per-cell data is a single sequential mmap read via `slice_array()`.

**Insert throughput:**

| Index | vecs/s | note |
|---|---:|---|
| FlatIndex | 6 100 | no training |
| IVFIndex | 1 970 | training: ~200s on 20K samples (once, offline) |

**Search — FlatIndex exact baseline:**

| N | dim | ms/query | QPS |
|---|---|---:|---:|
| 100 000 | 384 | 349 ms | 3 |

At this scale FlatIndex reads ~150 MB of mmap data per query — it becomes the bottleneck.

**IVFIndex — speed and recall@10 vs nprobe (K=256):**

| nprobe | % corpus | QPS | vs FlatIndex | recall@10 † |
|---:|---:|---:|---:|---:|
| 1 | 0.4% | 240 | **84×** | ~2% |
| 4 | 1.6% | 76 | **27×** | ~6% |
| 8 | 3.1% | 40 | **14×** | ~10% |
| 16 | 6.2% | 24 | **8×** | ~17% |
| 32 | 12.5% | 13 | **4×** | ~29% |
| 64 | 25.0% | 7 | **2×** | ~48% |
| 128 | 50.0% | 4 | 1.2× | ~73% |

† Recall figures are measured on **random Gaussian vectors**, which have no cluster structure — every centroid is nearly equidistant from every query point. On real semantic embeddings (text, image, audio), IVF clusters align with the data manifold and recall at nprobe=8–16 is typically **70–95%**.

**When to use which:**

- **FlatIndex** — best for ≤ 50K vectors. One sequential mmap read + numpy matmul. Exact results, no training, no parameters.
- **IVFIndex** — best for > 100K vectors. At 100K the speedup is 4–84× depending on nprobe. The crossover vs FlatIndex is around 50K vectors.
- **Rule of thumb:** `n_clusters ≈ sqrt(n_vecs)`, `nprobe ≈ sqrt(K)` as a starting point; tune nprobe up for higher recall. Training is a one-time offline step.

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

| Scenario | Status | Mechanism |
|---|---|---|
| Single process, single thread | ✅ | nominal |
| Single process, multiple threads | ✅ | `threading.RLock` via `@write_op` on every write method |
| Multiple HTTP clients → `db.serve()` | ✅ | same RLock shared across all request handlers |
| Multiple processes, reads only | ✅ | Linux shared mmap pages, x86 TSO ordering |
| Multiple processes, SWMR | ✅ | `DB(path, multiprocess_safe=True)` + `fcntl.flock(LOCK_EX)` on writes; readers never block |
| Multiple concurrent writers | ❌ | not supported — use a single writer process |

Every public write method (`__setitem__`, `append`, `push`, `add`, …) acquires the lock automatically — no manual wrapping required.  For compound operations that must be atomic together, use `with db.write_lock(): ...`.

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
