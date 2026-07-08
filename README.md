# loom

**Persistent Python data structures that feel native.**

loom is a file-backed database library that lets you work with `Dict`, `List`, `Queue`, `Set`, `BTree`, `Graph`, **vector indexes**, and **full-text search** exactly like their in-memory counterparts — but stored on disk with mmap zero-copy access, crash-safe writes, and automatic space reclamation. Higher-level `Collection`s bundle a record store with typed indexes kept in sync for you.

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
| `datetime` / `date` | `datetime` | **Stored inline as int64 epoch-µs**, read/written as Python `datetime` — naturally ordered (range/sort/PriorityQueue work) |
| `Literal["a", "b", …]` | `utf8[N]` | Closed string set → strict utf8 **sized to the longest value** (minimal space, refuses anything bigger). `Literal[1, 2]` → `int64`, `Literal[True, False]` → `bool` |
| `str` | `text` | Variable-length, compressed via BlobStore |
| `dict` / `Json()` | `json` | Arbitrary JSON value (dict/list/nested) via BlobStore — `json.dumps`/`loads` transparently; `None` round-trips |
| `Utf8(N)` | `utf8[N]` | **Fixed-width inline UTF-8, N bytes** — ~4× smaller than `U{N}` for ASCII, same read speed. **Raises** if a value exceeds N bytes; `Utf8(N, truncate=True)` to truncate instead |
| `str = Field(max_length=N)` | `U{N}` | Fixed-length numpy UCS-4 (4 bytes/char) |
| `FixedStr(N)` | `U{N}` | loom shorthand for the above |

```python
from pydantic import BaseModel, Field
from loom.schema import FixedStr, Utf8

from datetime import datetime

class Message(BaseModel):
    id:         int
    role:       Utf8(20)   # → utf8[20], 20 B inline, fast key
    content:    str        # → text, variable-length
    created_at: datetime   # → datetime; write/read datetime objects directly

class Product(BaseModel):
    sku:    Utf8(20)    # → utf8[20]
    name:   str         # → text
    price:  float
    stock:  int
```

You can also pass plain dicts:

```python
db.create_dataset("events", id="uint32", ts="int64", kind="utf8[20]")
```

**Rule of thumb:** use `Utf8(N)` for short bounded strings used as keys or in frequent filters (inline, compact, fast) — it's ~4× smaller than `U{N}` for ASCII. Use plain `str` (→ `text`) for anything long or variable — body text, descriptions, JSON payloads. Reach for `U{N}`/`FixedStr(N)` only when you specifically need fixed UCS-4.

**Datetimes** are first-class: declare a field as `datetime` (or use `Datetime()`) and read/write Python `datetime` objects — loom stores them inline as **int64 epoch-microseconds** (8 bytes, no BlobStore hop) and handles the conversion for you. They're naturally ordered, so a `datetime` field works directly as a Collection `range`/`many` criterion (e.g. `range("created_at", limit=50, desc=True)` for the 50 most recent) or a `PriorityQueue` priority. Naive datetimes are treated as UTC; timezone-aware values are converted to UTC and returned naive.

### The DB object

```python
# Open (creates if not exists)
db = DB("mydata.db")

# Context manager (recommended — auto-saves and flushes on exit)
with DB("mydata.db",
        blob_compression=None,        # None (default, fastest) | "brotli" | "zlib"
        auto_save_interval=100,       # metadata save frequency (default 100)
        cache_size=200_000,           # shared key→address LRU for the whole DB (default; 0 disables)
        sync_writes=False,            # False=fast (flush on close), True=safe (flush every write)
        header_size=32768,            # metadata region; raise it for many structures/indexes
) as db:
    ...

# header_size is stored in the file: after creation you can reopen with just
# DB("mydata.db") and the real value is detected automatically (no need to
# re-pass it). Raise it at creation if you hit HeaderTooLargeError (lots of
# indexes/collections); it can't shrink/grow on an existing file.

# db.batch() is useful mainly for text/blob fields — for fixed schemas the
# lazy flush already makes per-call inserts as fast as batch inserts.
with db.batch():
    for item in large_dataset:
        my_dict[item["key"]] = item

# Long-running servers: call flush() periodically instead of sync_writes=True
db.flush()  # force mmap writeback without closing
```

**Creating vs. retrieving.** Names are unique across the whole DB (a dataset and a
data structure can't share a name — `db[name]` would be ambiguous). `create_*`
**raises `DuplicateNameError`** if the name already exists, so re-running a script
or reopening a DB won't silently re-create or duplicate anything:

```python
ds   = db.create_dataset("posts", Post)
feed = db.create_list("feed", ds)

db["feed"]                       # retrieve an existing structure (or dataset)
"feed" in db                     # membership test
db.create_list("feed", ds)                 # → DuplicateNameError
db.create_list("feed", ds, exist_ok=True)  # idempotent: returns the existing one
```

After reopening a DB, get your structures back with `db["name"]` (they're already
loaded) — or pass `exist_ok=True` to `create_*` for an explicit open-or-create.
This applies to **collections** too: `db["posts"]` (or `"posts" in db`) returns a
reopened `Collection`, `db.collection("posts", Model, ...)` raises if it already
exists, and `db.collection("posts", Model, ..., exist_ok=True)` opens-or-creates.

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

### PriorityQueue — highest-priority-first, O(log n) push/pop

A persistent priority queue backed by a BTree. The highest priority pops first by default (`max_first=False` for lowest-first); items sharing a priority pop in **FIFO** order. Priorities can be **int, float or datetime** — all order-preserving encoded.

```python
pq = db.create_priority_queue("jobs", {"task": "utf8[40]"})

pq.push({"task": "send email"}, priority=5)
pq.push({"task": "reindex"},    priority=9)
pq.push_many([({"task": "warm cache"}, 7.5), ...])   # bulk (bulk_load when empty)

pq.peek()                         # -> {"task": "reindex"}   (priority 9, not removed)
pq.pop()                          # -> {"task": "reindex"}
pq.pop(default=None)              # None instead of raising on empty
len(pq)
```

`push`/`pop`/`peek` are O(log n) (BTree `min()` + delete). Priorities use the same order-preserving encoding as Collection indexes, so a float score or a `datetime` deadline sorts correctly.

### BTree — ordered key-value with range queries

```python
idx = db.create_btree("docs_by_title", doc_ds, key_size=100)

idx["Alpha post"] = {"id": 1, ...}
idx["Beta post"]  = {"id": 2, ...}

for k in idx.keys():              # sorted order
    print(k)

for k, v in idx.range("A", "C"):  # range query
    print(k, v)

for k, v in idx.range(reverse=True):   # descending (seek to end, walk down)
    print(k, v)                        # → great for "latest N" with a limit

for k, v in idx.prefix("Beta"):   # prefix search
    print(k, v)
```

`range()` is O(log n + k) in both directions — it seeks straight to the start (or, with `reverse=True`, the end) leaf rather than scanning from the smallest key.

Keys are strings by default; pass `int_keys=True` for **integer keys ordered numerically** (stored order-preserving, so `2 < 10`, negatives supported) — `keys()`/`items()`/`range()`/`min()`/`max()` then take and return ints:

```python
ticks = db.create_btree("ticks", ohlcv_ds, int_keys=True)
ticks[1700000000] = {...}
ticks.range(1700000000, 1700003600)   # numeric range
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

## Collection — a record store with typed indexes

Looking records up by more than one field normally means maintaining several
structures by hand and keeping them in sync. A `Collection` does that for you:
you **declare each field's index kind** and loom maps it to the right structure,
kept in sync on every insert / update / delete under one lock. The record lives
**once** in the primary index; every other index stores only the primary key.

| Kind | Structure | Use it for |
|------|-----------|------------|
| `"primary"` | Dict | the record store + unique key (exactly one required) |
| `"unique"` | Dict | a second 1:1 lookup key (enforces uniqueness) |
| `"range"` | B+Tree | numeric / ordered range scans (`>=`, between) |
| `Many(sort=, desc=, field=)` | B+Tree | one-to-many groups, returned in sort order. It's a **compound** key, so `find()` can bound the sort field → *equality AND range* in one seek |
| `Search(fields=, scoring=)` | SearchIndex | full-text (boolean or BM25) |
| `Vector(field=, metric=)` | — (vectors live in the records) | exact similarity, pre-filtered through the other indexes (`nearest()`) |

```python
from pydantic import BaseModel, Field
from loom import Many, Search

class Post(BaseModel):
    id:         str = Field(max_length=32)
    username:   str = Field(max_length=30)
    created_at: int
    engagement: int
    text:       str

posts = db.collection("posts", Post, indexes={
    "id":         "primary",                          # record store, by id
    "username":   Many(sort="created_at", desc=True), # a user's feed, recent first
    "engagement": "range",                            # engagement >= x
    "body":       Search(fields=["text"], scoring="bm25"),  # full-text
})

posts.insert({"id": "p1", "username": "alice", "created_at": 170,
              "engagement": 9, "text": "an inverted index in pure python"})
posts.insert_many([...])                       # bulk (bulk-loads empty BTrees)

posts["p1"]                                    # by primary key
posts.find("username", "alice", limit=20)      # alice's 20 most-recent posts
posts.range("engagement", 1000, None)          # engagement >= 1000, ascending
posts.range("created_at", limit=50, desc=True) # 50 most recent — no grouping needed
posts.count("username", "alice")               # group size — key-only scan,
posts.count("username", "alice", start=T)      # never reads a record
posts.find("username", "alice",
           fields=["id", "engagement"])        # projection: row read only, no blobs
posts.search("body", "inverted OR index")      # full-text → records
posts.search("body", "inverted",               # full-text AND structured filter
             where={"username": "alice", "created_at": (date(2026, 1, 1), None)})

posts.increment("p1", "engagement", 1)         # atomic counter bump
posts.update("p1", engagement=5000)            # re-indexes only changed fields
posts["p1"]["engagement"] = 5000               # same thing — write-through record
posts.delete("p1")                             # removed from every index
```

- **Upsert.** `insert`/`insert_many` replace a record whose primary key already
  exists and re-point every index (the old field values' entries are dropped
  first) — so re-loading a batch that contains existing keys stays consistent.
  Within one `insert_many` batch a repeated key keeps the last occurrence.
  `unique` constraints are enforced by both `insert` and `insert_many` (a
  duplicate value raises `ValueError` before anything is written).
- **No duplication.** Secondary indexes store the *primary key*, not a copy of
  the record — a secondary lookup is one extra hop (`index → pk → record`). The
  full-text index keeps only postings (no record copy either).
- **Key sizing is automatic.** Index storage is sized from the declared field
  widths, so a long primary key (e.g. a `Utf8(256)` URL) is never truncated in
  index entries. A `unique`/`many` index on an **unbounded** field (`text`,
  `dict`/`json`) is indexed by a 128-bit **hash** of the value — so you can
  group by a long text (an "argument", an article title): `find("argument", x)`
  works (equality), but `range` on such a field is rejected (no ordering).
- **Ordered & paginated for free.** `Many(sort="created_at", desc=True)` stores
  composite keys in a B+Tree, so `find()` returns a group already in sort order
  with cheap `limit` — ideal for recent-first feeds. The sort/range criterion may
  be an **int, float or datetime** (all order-preserving encoded), with
  `desc=True`/`False`.
- **Latest-N with no grouping.** A `range` index iterates either direction:
  `range("created_at", limit=50, desc=True)` returns the 50 most recent items
  (seek to the high end, walk down — O(log n + k)), no group field needed.
  Use `Many(sort=..., desc=True)` instead when you want a *per-group* ordered
  feed (e.g. each user's timeline); `range(desc=True)` is the single-axis
  "most-relevant/most-recent first" feed, and `range("relevance", 3.0, None)`
  still filters by score.
- **Compound filter — equality AND range, fast at any scale.** A `Many` key is
  `group · sort · pk`, so it *is* a compound index. `find()` can bound the sort
  field, turning `category = X AND created_at >= T` into a single seek + bounded
  scan — **O(log n + matches)**, independent of how much history the group holds
  (a year of data stays instant). Use `field=` to index the same field under
  several names (e.g. one ordering by date, one by engagement):

  ```python
  posts = db.collection("posts", Post, indexes={
      "post_id":  "primary",
      "category": Many(field="category_alias", sort="engagements", desc=True),  # top of a category
      "cat_time": Many(field="category_alias", sort="created_at",  desc=True),  # category over time
  })
  posts.find("cat_time", "politics", start=date(2026, 1, 1))            # category X since a date
  posts.find("cat_time", "politics", start=date(2026, 1, 1), end=date(2026, 6, 1))  # a window
  posts.find("category", "politics", limit=20)                          # top-20 by engagement
  ```
- **Counting & projection.** `count(index, value, start=, end=)` sizes a group
  (or a window of its sort field) with a **key-only** scan — no index values,
  no records, ~18× faster than `len(find(...))`. `find(..., fields=[...])` /
  `range(..., fields=[...])` materialize only the requested fields: one row
  read per hit, and unrequested `text`/`json` fields never touch the blob
  store.
- **Maintained group counters.** `Many(..., counted=True)` keeps a companion
  `group → count` Dict in sync on every write: `count()` becomes **O(1)**
  (~7 µs) and `groups()` lists every group with its size without touching a
  single record — "all narratives ordered by post volume" is one call:

  ```python
  posts = db.collection("posts", Post, indexes={
      "id":        "primary",
      "narrative": Many(sort="created_at", desc=True, counted=True),
  })
  posts.count("narrative", "ukraine")          # O(1)
  posts.groups("narrative")                    # [(value, count), ...] biggest first
  posts.groups("narrative", order_by="value", desc=False, limit=50)
  ```

  Opt-in: the counter costs one extra field write per insert (~10–20% on a
  minimal 3-field schema, a few % on a realistic wide one). Windowed counts
  (`start=`/`end=`) still use the key-only scan.
- **Vector similarity — pre-filtered, exact.** `Vector(metric="cosine"|"l2"|"dot")`
  on an inline `Vec(N)` field costs *nothing on write* (no backing structure —
  the vector lives in the record). `nearest()` narrows candidates through the
  collection's regular indexes first (same `where` spec as `search()`), then
  scores only the survivors' vectors exactly — no ANN recall to worry about,
  and the filter applies *before* scoring, so a selective filter makes the
  query faster, not emptier:

  ```python
  from loom import Vector, Vec

  docs = db.collection("docs", Doc, indexes={
      "id":    "primary",
      "topic": Many(sort="created_at", desc=True),
      "emb":   Vector(metric="cosine"),          # Doc.emb: Vec(384)
  })
  docs.nearest("emb", qvec, k=10)                              # whole collection
  docs.nearest("emb", qvec, k=10, with_scores=True,            # group + window →
               where={"topic": "politics",                     # one index seek,
                      "created_at": (date(2026, 6, 1), None)}) # then exact top-k
  ```

  100k docs (dim 384): full scan 1.7 s, topic group (2.5k candidates) 79 ms,
  group + month window (830 candidates) 28 ms. If an unfiltered corpus
  outgrows the flat scan, the standalone `FlatIndex`/`IVFIndex` are the ANN
  path.
- **Persistence.** The index declaration is saved with the collection, so
  `db.collection("posts")` (no model) reopens it and restores every index
  automatically.
- **Atomicity.** Writes touch all indexes under `db.write_lock()` + `db.batch()`
  (serialised, grouped). This is not a full crash-atomic WAL across indexes; if
  a crash ever desyncs an index, call `posts.reindex()` to rebuild them.
- **Schema migration.** `db.migrate_collection(name, NewModel, transforms={...})`
  rebuilds a collection under a new schema (add / drop / rename fields). Each
  record is rebuilt from `transforms[field](old)` if given, else the old value,
  else the field default; indexes are reused unless you pass new ones. Records
  are read into memory first, so a mid-migration crash can't lose data.
  `db.drop_collection(name)` removes a collection (space reclaimed by `vacuum()`).

---

## SearchIndex — full-text inverted index

A persistent inverted index for boolean and ranked (BM25) full-text search.
Documents live in one of your datasets; the index stores only the postings and
keeps each document's address — text is never duplicated. Boolean queries
(`AND` / `OR` / `AND NOT`, parentheses, `*` wildcards) are parsed by
[`eldar`](https://pypi.org/project/eldar/) (`pip install eldar`).

```python
docs = db.create_dataset("docs", title="utf8[120]", body="text")
idx  = db.create_search_index(
    "idx", docs,
    text_fields=["title", "body"],   # None → all string fields
    scoring="bm25",                  # "boolean" (default) or "bm25"
)

i = idx.add({"title": "Fast search", "body": "an inverted index"})  # → doc-id
idx.add_many([{"title": ..., "body": ...}, ...])

# Boolean — unranked, doc-id order
idx.search("inverted AND NOT slow", mode="boolean")   # → [records]

# Ranked — BM25, best first (needs scoring="bm25"). Combine terms with
# explicit operators; a bare "a b" is treated as a phrase, not "a OR b".
idx.search("inverted OR index")                       # → [records]
idx.search("inverted AND index", limit=10, with_scores=True)  # → [(record, score)]
idx.search("invert*", return_ids=True)                # → [doc-ids] (no fetch)

idx.get_document(i)   # the stored record for a doc-id
idx.delete(i)         # tombstone a document
```

- **Compact postings.** Each term's posting list is stored as a delta + varint
  blob (monotonically increasing doc-ids), rebuilt once per flush — small on
  disk and fast to decode.
- **Buffered build.** `add()` buffers; the index materialises on the next
  `search()` or explicit `flush()` (one blob write per term, grouped). Bulk
  loads stay cheap.
- **Native results.** `search()` returns your records (Python types), or
  `(record, score)` with `with_scores=True`, or raw doc-ids with
  `return_ids=True`.

Inside a `Collection`, declare a full-text index with `Search(...)` (e.g.
`indexes={..., "body": Search(fields=["title", "body"], scoring="bm25")}`) and
query it with `collection.search("body", "inverted index")`. The collection
wires a doc-store-less `SearchIndex` (postings only, no record duplication) and
maps results back to records for you.

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
| `http://127.0.0.1:8000/` | Index: filename, structures, collections, links to the docs |

Every `Dict`, `List`, `Set`, `BTree`, `Queue`, `PriorityQueue`, `BloomFilter`, `LRUDict`, `SearchIndex` and `Collection` you create gets a CRUD route family automatically (`/dicts/<name>/items/<key>`, `/btrees/<name>/range`, …). Request bodies are validated against your loom schema via Pydantic models generated on the fly.

- **`SearchIndex`** → `/search_indexes/<name>`: `GET /search?q=&mode=&limit=&with_scores=` (boolean or BM25), `POST/GET/DELETE /documents[/<doc_id>]`.
- **`Collection`** → `/collections/<name>`: `GET/POST/PUT/DELETE /records[/<pk>]`, `POST /records/<pk>/increment`, and per-index lookups `GET /find/<index>`, `GET /range/<index>`, `GET /search/<index>`. Collections are listed under the `collections` key of the index route.

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

Dict, List, Queue and BTree numbers come from `benchmarks/readme_benchmark.py` (run it yourself with `PYTHONPATH=. python benchmarks/readme_benchmark.py`); Set and LRUDict from their dedicated `benchmarks/benchmark_*.py`; and **Graph from `benchmarks/benchmark_graph_kg.py` on FB15k** (the reference graph benchmark — see its section below). Figures are rounded; expect ±10% run-to-run variance from page-cache warmup and CPU frequency.

All numbers use the **default shared cache** (`DB(cache_size=200_000)`) — one LRU for the whole DB that stores **key→address** (not values), so a lookup skips the per-table slot scan and reads the record directly. It costs nothing until used (~170 B per hot key, so the 200k default tops out around ~35 MB) and is what makes random reads several × faster than the uncached path. `read`/`contains` figures assume the working set fits the cache (the benchmark reads keys it just wrote); with a working set far larger than the cache, expect them to fall back toward the cold-scan rate. Set `DB(cache_size=0)` to disable caching entirely.

### loom vs SqliteDict — 10 000 ops, fixed schema

| Operation | **loom** | **SqliteDict** | Ratio |
|---|---:|---:|---:|
| Dict insert (per-call) | **21 000 ops/s** | 5 000 ops/s (autocommit) | loom **4.2×** |
| Dict insert (batch) | **22 000 ops/s** | 22 000 ops/s (single `COMMIT`) | loom **1.0×** |
| Dict read | **167 000 ops/s** | 11 900 ops/s | loom **14×** |
| Dict contains | **624 000 ops/s** | 12 300 ops/s | loom **51×** |
| Dict keys() | **263 000 ops/s** | 111 000 ops/s | loom **2.4×** |

loom is faster on every line, including SQLite's most favourable mode (defer all writes to a single transaction). The gap is widest on point ops because loom uses lazy mmap flush — no `msync()` per write, just OS page-cache writeback — while SQLite has to walk a B-tree per call. `contains` benefits from binary murmur128 slots (25 bytes regardless of key length), which makes the inner loop a couple of integer compares.

Per-call inserts already run at batch speed in loom (the flush is already lazy), so wrapping inserts in `db.batch()` mostly matters for `text` / blob fields, where it amortises one BlobStore flush across many writes.

### loom operations — all structures

| Structure | Operation | ops/s | µs/op |
|---|---|---:|---:|
| Dict | insert | 21 000 | 48 |
| Dict | read | 167 000 | 6 |
| Dict | contains | 624 000 | 2 |
| Dict | keys() | 263 000 | 4 |
| Dict | items() | 134 000 | 7 |
| List | append | 91 000 | 11 |
| List | read[i] | 190 000 | 5 |
| Queue | push (batch) | 158 000 | 6 |
| Queue | pop | 187 000 | 5 |
| BTree | insert | 12 000 | 81 |
| BTree | read | 104 000 | 10 |
| BTree | contains | 175 000 | 6 |
| BTree | keys() [sorted] | 6 160 000 | 0 |
| Set | add | 27 000 | 37 |
| Set | contains | 441 000 | 2 |
| Set | remove | 25 000 | 40 |
| LRUDict | set | 8 500 | 118 |
| LRUDict | get (hit) | 31 000 | 32 |
| Graph | add_nodes / add_edges (bulk) | 26 000 | 38 |
| Graph | add_edge (per-call) | 1 700 | 583 |
| Graph | get_node | 184 000 | 5 |
| Graph | has_edge | 61 000 | 16 |
| Graph | neighbors | 263 000 edges/s | 4 |

All rows use the default shared cache; Graph rows are the FB15k reference benchmark below. `contains`/`read` on Dict, Set and BTree are dominated by cache hits (key→address), so they assume a working set that fits the cache. `Set add`/`LRUDict set` track `Dict insert` because both wrap a Dict (Set adds a membership record; LRUDict adds eviction bookkeeping), so neither can be faster than the underlying insert.

**Dict vs BTree — pick the right tool.** The B-tree relies on keeping its internal nodes hot. With the shared cache its random `read` (~104k/s) and `contains` (~175k/s) are strong, and `keys()` iterates already-sorted, mostly-cached nodes at millions/s; at `cache_size=0` the same `read` collapses (every node a cold mmap read). The Dict reaches higher read rates from the address cache and inserts ~2× faster. So:

- **Insert**: Dict wins (~21k vs ~12k/s) — O(1) slot vs O(log n) node path. Building a fresh BTree from a batch, use `bulk_load()` (O(n) bottom-up, ~140k keys/s).
- **Random read / contains**: comparable with a warm cache; the Dict is the simpler choice for pure point access.
- **Ordered iteration / range / prefix**: BTree only — `keys()` comes out sorted, and `range()`/`prefix()` seek to the start key then walk the span (O(log n + k)). A range of 101 keys takes ~364 µs (~2 700 ranges/s).

Bottom line: use a `Dict` for pure point access, a `BTree` when you need ordering or range queries.

### `str` (text) fields — impact of variable-length blobs

`str` fields bypass the fixed record and write to a separate BlobStore — slower but space-efficient.

| Schema | Compression | insert | read |
|---|---|---:|---:|
| Fixed fields only | — | 21 000 ops/s | 167 000 ops/s |
| + `str` body (≈600 chars) | None (default) | 22 000 ops/s | 167 000 ops/s |
| + `str` body (≈600 chars) | brotli | 1 300 ops/s | 59 000 ops/s |

- `blob_compression=None` (**default**) — fastest writes, larger files.
- `"brotli"` — 3–5× compression on natural language, but ~20× slower inserts; pick when storage > write throughput.
- `Field(max_length=N)` → `U{N}` — keeps the field in the fixed record (UCS-4, **4 bytes/char**), no BlobStore.
- `Utf8(N)` → `utf8[N]` — fixed-width **inline UTF-8**: N bytes in the record, no BlobStore hop, so ~**4× smaller than `U{N}`** for ASCII at the same read speed. `N` is a byte budget; a value over budget **raises ValueError** by default (use `Utf8(N, truncate=True)` to truncate on a codepoint boundary instead). The sweet spot for short ASCII-ish strings — ids, URLs, codes, enums.

**Choosing a string field**: `Utf8(N)` for bounded ASCII-ish values you read a lot (inline, compact, fast); `str`/`text` for long or unbounded natural language (BlobStore, compressible); `U{N}`/`FixedStr(N)` only when you specifically need fixed UCS-4. loom stores Dict/Graph **keys** (`_key`) as `utf8` by default for exactly this reason.

### Graph — FB15k knowledge graph (reference benchmark)

The reference graph benchmark is **FB15k** (a Freebase subset): 14,951 typed
entities, 1,345 relation types, **483,142 directed edges** (avg degree ~65).
Each node carries a `type` label (relation domain) and relations are interned
to a `uint16`. Reproduce with `PYTHONPATH=. python benchmarks/benchmark_graph_kg.py`
(downloads the dataset once; see the script header).

Build (write):

| Operation | rate | µs/op |
|---|---:|---:|
| `add_nodes` (bulk) | 23 000 nodes/s | 44 |
| `add_edges` (bulk) | 26 000 edges/s | 39 |
| `add_edge` (per-call) | 1 700 edges/s | 583 |

`add_edges` groups edges by source for `_out` and by target for `_in`, then
bulk-inserts each node's whole adjacency sub-dict in one shot (one allocation
+ one parent-ref update per node) — ~16× faster than per-call `add_edge`.
**Load all edges in a single `add_edges` call** so each node hits this fast
path (re-inserting into an already-populated node falls back to a slower path).

Read (10 000 random samples):

| Operation | rate | µs/op |
|---|---:|---:|
| `g[node_id]` (`get_node`) | 184 000 ops/s | 5 |
| `has_edge` (hit) | 61 000 ops/s | 16 |
| `get_edge` | 101 000 ops/s | 10 |
| `out_degree` | 251 000 ops/s | 4 |
| `neighbors` (iterate) | **263 000 edges/s** | 4 |

These are several × the uncached rates: every adjacency lookup now hits the
shared address cache instead of re-scanning the source node's hash tables. The
gain is largest on repeated/hot access (`out_degree` ~15×); `neighbors`
iteration, which streams a whole sub-dict, gains less (~1.6×).

Query engine (Cypher):

| Query | rate |
|---|---:|
| label-seeded 1-hop `(a:Type)->(b) LIMIT 50` | 1 185 queries/s |
| 2-hop chain `(a)->(b)->(c) LIMIT 100` | 590 queries/s |
| 1-hop from a hub `id(a)=='X'` (all neighbours) | 193 queries/s |
| variable-length `(a)-[*2]->(b) LIMIT 100` | 28 queries/s |

The label index (`nodes_with_label`, used to seed `(a:Type)…` queries) builds
in ~110 ms for 15k nodes and then serves ~36 000 lookups/s. Reopen is ~1 ms.

On-disk: **~277 bytes/edge** (127 MB for 483k edges, double-indexed `_out` +
`_in`). Storing the adjacency key (`_key`, the destination/source node id) as
inline `utf8` instead of UCS-4 `U{N}` roughly halved this (277 → 127 MB) at no
read cost — node ids are short ASCII, so 1 byte/char instead of 4. What's left
is mostly the live edge records (each direction stores the edge once) plus the
per-source nested hash tables. Graphs with few nodes benefit even more from the
small default values-block (see note) — a sparse graph that used to be several×
this size is now close to its live data.

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
- `auto_save_interval=N` controls how often structure metadata (lengths, counters) is checkpointed to disk (default: every 100 ops per structure). Bulk ops (`append_many`, `push_many`) also checkpoint at the end, and `flush()`/`close()` persist it immediately. An `atexit` safety net also closes any still-open DB on a clean interpreter exit, so a script that forgets `close()`/`with` still persists its metadata. Prefer `with DB(...)` (or an explicit `close()`/`flush()`) anyway — `atexit` does not run on `kill -9` or a hard crash.

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
- `db.vacuum()` rewrites the whole database into a fresh file — dropping soft-deleted records, fragmentation, and arenas orphaned by drop/migrate/upsert — then atomically swaps it in. Supports collection-based databases (raises rather than risk dropping a standalone structure type it can't yet copy).

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
| Priority scheduling, ranked work | **loom PriorityQueue** |
| Graph data, social / knowledge networks | **loom Graph** |
| Full-text / boolean / BM25 search | **loom SearchIndex** |
| Records looked up by several fields at once | **loom Collection** |
| Large text content per record | **loom + `str` + brotli** |
| Complex relational queries (JOINs) | SQLite / PostgreSQL |
| Concurrent multi-writer | SQLite / PostgreSQL |
| Pure analytics, columnar scans | Parquet / DuckDB |
