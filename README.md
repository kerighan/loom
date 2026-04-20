# loom

**Persistent Python data structures that feel native.**

loom is a file-backed database library that lets you work with `Dict`, `List`, `Queue`, `Set`, `BTree`, and `Graph` exactly like their in-memory counterparts — but stored on disk with mmap zero-copy access, crash-safe writes, and automatic space reclamation.

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

# Context manager (recommended — auto-saves on exit)
with DB("mydata.db",
        blob_compression="brotli",   # "brotli" | "zlib" | None
        auto_save_interval=100,       # metadata flush frequency (default 100)
        cache_size=50_000,            # shared LRU entries across ALL structures
) as db:
    ...

# Bulk inserts — defer header flushes for 10–100x speedup
with db.batch():
    for item in large_dataset:
        my_dict[item["key"]] = item
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

# Bulk insert (75x faster than per-call for large graphs)
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

Benchmarks on a modern laptop (Linux, SSD), 10 000 operations.

### loom vs SqliteDict

Benchmarks on 10 000 operations, fixed schema (no text/blob fields).

| Operation | **loom** | **SqliteDict** | Notes |
|---|---:|---:|---|
| Dict insert, no batch | **23 500 ops/s** | 5 800 ops/s | loom **4×** faster — lazy mmap flush vs. per-op SQLite COMMIT |
| Dict insert, **batch** | **23 500 ops/s** | **45 200 ops/s** | SQLite wins: defers all I/O to one COMMIT. loom batch() mainly helps for blobby text fields. |
| **Dict read** | **46 800 ops/s** | 13 300 ops/s | loom **3.5×** faster — mmap zero-copy |
| **Dict contains** | **66 600 ops/s** | 14 400 ops/s | loom **4.6×** faster — bloom filter |
| **Dict keys()** | **670 000 ops/s** | 120 000 ops/s | loom **5.6×** faster — bulk numpy read |

loom inserts are now **4× faster** than SqliteDict per-call because loom no longer flushes the mmap on every write — the OS page cache handles writeback. Full durability is guaranteed on `close()` (or via `db.flush()` / `sync_writes=True` for server workloads).

**Dict insert batch ≈ no-batch** since the flush is already lazy. `db.batch()` still helps for text/blob fields (defers blob BlobStore saves) and for concurrent reader visibility.

### loom operations reference

All figures with default `sync_writes=False` (lazy flush, durable on `close()`).

| Structure | Operation | ops/s | µs/op |
|---|---|---:|---:|
| Dict | insert | 23 500 | 43 |
| Dict | read | 46 800 | 21 |
| Dict | contains (bloom) | 66 600 | 15 |
| Dict | keys() | 670 000 | 1.5 |
| Dict | items() | 155 800 | 6.4 |
| List | append | 84 100 | 12 |
| List | read[i] | 141 000 | 7 |
| Queue | push (batch) | 258 000 | 3.9 |
| Queue | pop | 273 000 | 3.7 |

Note: `db.batch()` no longer makes a meaningful difference for fixed-schema Dicts and Lists — the lazy flush already removes the sync overhead. `batch()` is still useful for text/blob fields (defers BlobStore metadata flushes).

### Impact of text (`str`) fields

| Schema | Compression | insert | read |
|---|---|---:|---:|
| Fixed fields only | — | 23 500 ops/s | 46 800 ops/s |
| + `str` body field | None | 13 100 ops/s | 39 200 ops/s |
| + `str` body field | brotli | 1 920 ops/s | 26 700 ops/s |

Text/blob inserts are still slower because each `str` field writes compressed data to the BlobStore (CPU-bound for brotli, allocation-bound for uncompressed).

### Graph — Barabási-Albert scale-free network (m=2)

The lazy flush now makes per-call `add_edge()` and batch `add_edges()` nearly equivalent (1.2× difference, down from 75× before). Both are usable; `add_edges()` is still recommended for clarity and slightly lower overhead.

```python
# Both are now fast
g.add_edge("alice", "bob", weight=0.9)        # per-call: OK

g.add_edges([                                  # batch: marginally faster
    ("alice", "bob",   {"weight": 0.9}),
    ("alice", "carol", {"weight": 0.5}),
])
```

| Nodes | Edges | add_node µs | add_edge µs | total | disk used |
|---|---|---:|---:|---:|---:|
| 5 000 | 9 996 | 370 | 246 | 4.3 s | 25 MB |
| 20 000 | 39 996 | 373 | 274 | 18 s | 40 MB |
| 50 000 | 99 996 | 379 | 295 | 48 s | 70 MB |
| 100 000 | 199 996 | 378 | 309 | 100 s | 127 MB |

Storage: ~1.3 KB/edge (double-indexed: outgoing + incoming adjacency).
Bottleneck: hash-table probing in per-node nested Dicts (~38% CPU), not I/O.

**Per-call vs batch** (5k nodes):
- `add_edge()` per-call: **3 278 ops/s** (305 µs)
- `add_edges()` batch: **3 864 ops/s** (259 µs) — 1.2× faster

**Guidelines for `str` fields:**
- Use `blob_compression=None` when write throughput matters more than disk space.
- Use `"brotli"` for text-heavy workloads (natural language compresses 3–5×).
- Use `Field(max_length=N)` for short, frequently-read fields (IDs, roles, tags) to avoid BlobStore entirely.

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
