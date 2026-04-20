# loom — Roadmap

Items grouped by theme, roughly ordered by impact/feasibility.
Not a commitment to timeline — just a shared backlog.

---

## 1. Multi-indexation / Collection

**Problem:** looking up records by a non-primary field requires maintaining
a second Dict manually and keeping it in sync by hand.

**Design:** a `Collection` coordinator wraps an existing `Dataset` and N
`Dict`/`BTree` indexes, keeping them atomically in sync via WAL transaction.
No new primitives — `create_dict` and `create_btree` stay exactly as-is.

```python
users_ds = db.create_dataset("users", User)
by_name  = db.create_dict("by_name",  users_ds)
by_email = db.create_dict("by_email", users_ds)

users = db.collection(
    dataset=users_ds,
    primary  =(by_name,  lambda r: r["username"]),
    secondary=[(by_email, lambda r: r["email"])],
)

users.insert({"username": "alice", "email": "a@x.com"})
users.delete("alice")                        # removes from all indexes
users["alice"]                               # primary lookup
users.get("by_email", "a@x.com")            # secondary lookup
```

The `Collection` is not persisted — it is reconstructed at session open with
the same lambdas.  The underlying Dicts persist independently.

**Atomicity:** insert/delete batches all index writes in one WAL transaction.

---

## 2. Vector / Embedding support

**Problem:** no vector dtype; storing embeddings requires raw bytes (`blob`)
with manual encode/decode.  No approximate nearest-neighbour search.

**2a. `vec_float32(dim)` dtype — exact storage**

Add a Pydantic-compatible dtype for fixed-length float arrays:

```python
from loom.schema import Vec

class Passage(BaseModel):
    text:      str
    embedding: Vec(1536)      # → stores 1536×float32 = 6144 bytes in BlobStore

passages = db.create_dataset("passages", Passage)
```

Read/write returns a `numpy.ndarray` of shape `(dim,)`.

**2b. VectorIndex — approximate nearest-neighbour (ANN)**

A new top-level structure wrapping an HNSW graph stored in loom blocks:

```python
idx = db.create_vector_index("embeddings", dim=1536, metric="cosine")
idx.add("doc_42", embedding_vector)          # insert by ID
results = idx.search(query_vector, k=10)     # [(id, score), ...]
```

Internal storage: HNSW layers as Datasets + adjacency stored in the
ByteFileDB file.  No external dependencies (pure numpy implementation).

**Use case:** 50K–500K text embeddings with fast approximate lookup.
For exact lookup by text → pre-compute a short ID (SHA-256[:16]) and use
a regular Dict; store the embedding as `Vec(dim)` in the record.

---

## 3. Multi-writer with file locking

**Problem:** only one process can write; multiple pipelines or workers
writing to the same store require external coordination.

**Design:** advisory file lock via `fcntl.flock` (Linux/Mac) +
`msvcrt.locking` (Windows), with a `.lock` companion file.

```python
# Auto-lock mode: each mutation acquires LOCK_EX, readers get LOCK_SH
db = DB("app.db", locking=True)

# Or explicit: batch several writes under one lock acquisition
with db.write_lock():
    db["key1"] = val1
    db["key2"] = val2
```

**Guarantees:**
- Safe for multiple processes writing sequentially (pipelines, batch jobs).
- NOT a substitute for a server under high-concurrency web load.
- Lock is released automatically by the OS on process death (no cleanup needed).

**Non-goals:** row-level locking, high-concurrency OLTP, MVCC.

---

## 4. Schema migration

**Problem:** adding/removing/renaming fields in a Pydantic model does not
migrate existing records.

**Design:** a `db.migrate(dataset_name, new_model, transforms={...})` helper
that reads old records, applies field transforms, and rewrites them.

```python
# Old model had `name: str`, new one has `first_name` + `last_name`
db.migrate("users", UserV2, transforms={
    "first_name": lambda old: old["name"].split()[0],
    "last_name":  lambda old: old["name"].split()[-1],
})
```

Internally: creates a new dataset, rewrites all records, swaps the registry
entry.  Old data kept as backup until `db.drop_backup("users")`.

---

## 5. Global vacuum / compaction

**Problem:** fragmentation accumulates over time in long-lived databases
(soft-deleted records, Dict hash tables with many deleted slots, BTree
nodes after heavy deletion).

**Design:** `db.vacuum()` — rebuild all structures without dead space,
similar to SQLite's VACUUM.

```python
before = db.file_size()
db.vacuum()
after = db.file_size()
print(f"reclaimed {(before - after) / 1e6:.1f} MB")
```

Implementation: for each structure, reads live data and rewrites to a new
file, then atomically swaps.  O(n) but safe.

---

## 6. Read-only mode

**Problem:** opening a DB for reading requires `r+b` (read-write).  No way
to enforce that a session cannot write.

**Design:**

```python
db = DB("app.db", mode="r")  # opens with mmap.ACCESS_READ
# any write attempt raises ReadOnlyError
```

Unlocks safer multi-reader patterns and is a prerequisite for clean
multi-writer semantics (readers hold LOCK_SH in locking mode).

---

## 7. Packaging / distribution

- Publish to PyPI as `loom-db`
- Semantic versioning (MAJOR.MINOR.PATCH)
- CI/CD: GitHub Actions running tests on Linux, macOS, Windows
- Auto-generated API docs (MkDocs or Sphinx)
- `setup.py` / `pyproject.toml` with proper optional deps
  (`pydantic`, `brotli`, `mmh3` as extras)

---

## 8. Quality-of-life / small items

- `db.stats()` — file size, fragmentation ratio, per-structure entry counts
- `db.verify()` — walk all structures and check for corruption
- Integer keys on BTree (currently strings only)
- `dict.get_many(keys)` — bulk lookup with one mmap pass
- `db.collection()` persisted metadata (avoid re-declaring lambdas on reopen)
- Graceful handling of `HeaderTooLargeError` with auto-resize hint
