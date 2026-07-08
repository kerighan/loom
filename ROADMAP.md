# loom — Roadmap

Open items first, roughly ordered by impact/feasibility. Done items are kept
as a dated one-line log at the bottom. Not a commitment to timeline — just a
shared backlog.

---

## Open

### 1. ANN behind the `Vector` spec

`Vector()` + `nearest()` (done) is exact, pre-filtered flat search — the
right tool while candidates are bounded by a filter. If an **unfiltered**
corpus outgrows the flat scan (> ~500k vectors), plug the standalone
`FlatIndex`/`IVFIndex` behind the same spec so `nearest()` transparently uses
ANN when no `where=` narrows the search. Same API, no migration.

### 2. Order-statistics BTree (O(log n) range counts)

Subtree counts stored in BTree nodes would make `count()` O(log n) for *any*
range — including arbitrary time windows on huge groups — instead of
O(matches). Disk-format change + split/merge complexity: only worth it if
windowed counts on very large groups become a hot path (counted indexes cover the
per-group case without it).

### 3. Cross-index crash atomicity for Collections

Collection writes are serialised (`write_lock()` + `batch()`) but not
crash-atomic across indexes: a crash mid-write can desync an index
(`reindex()` repairs). A plan-then-apply WAL on the Dict/BTree write path
would close this — significant plumbing, evaluate against real incident rate
(zero so far).

### 4. Quality-of-life, small

- Windows: `msvcrt.locking` fallback for `multiprocess_safe=True` (fcntl is
  Linux/Mac only) — only if a Windows deployment ever materialises.

---

## Decided against / no longer relevant

- **PyPI / public packaging** — the project stays private by choice. Docs
  exist (Sphinx under `docs/`, built by the pre-commit hook); no public CI.
- **HNSW vector index** — superseded: exact `FlatIndex` + trained
  `IVFIndex` (optional PQ) cover the standalone ANN need; Collections got
  pre-filtered exact `nearest()` instead, which fits the real query shapes
  (filter first, then similarity) better than ANN post-filtering.

---

## Done

- **2026-06-06 — Collection v1** (`db.collection`): record store + attached
  reference indexes, synced under lock+batch.
- **2026-06-20 — Collection redesign**: declarative typed indexes
  (`Primary`/`Unique`/`Range`/`Many(sort=, desc=, field=)`/`Search`),
  order-preserving key encoding, Record write-through wrapper, BTree
  `bulk_load`, seek-based `range()/prefix()`, fast `_deserialize`.
- **2026-06-20 — Vec(N) dtype + vector indexes**: inline float32 arrays;
  `FlatIndex` (exact) and `IVFIndex` (trained, optional PQ compression).
- **2026-06-20/23 — Full-text in Collections**: doc-store-less SearchIndex
  (`Search(fields=, scoring=)`), `search(where=...)` structured filter.
- **2026-06-22 — Compound equality+range**: `find(index, value, start=, end=)`
  in one seek; upsert semantics for `insert`/`insert_many`.
- **2026-06-23 — migrate/drop/vacuum**: `db.migrate_collection`,
  `db.drop_collection`, `db.vacuum()` (atomic file swap); `json` dtype;
  automatic key sizing (long pks never truncated); hashed indexes on
  unbounded fields.
- **2026-06 — Multi-process SWMR**: `threading.RLock` + `fcntl.flock`
  (`multiprocess_safe=True`), read-only open mode.
- **2026-06-23 — Sphinx docs** (furo + myst): API autodoc + use-case
  tutorials, docs-build pre-commit hook.
- **2026-06 — QoL**: `db.stats()`, `db.verify()`, BTree int keys,
  `get_many`, `sample()/describe()` introspection, self-describing
  `header_size`, PriorityQueue + native datetime fields.
- **2026-07-02 — Insert/read perf pass**: single-descent BTree upsert
  (file-identical to master, proven by sha256), header-cadence flush,
  struct-pack serialize plan, C-level bisect; Collection inserts ~2×.
- **2026-07-02 — Search query perf**: vectorised varint decode + doc-length
  cache; BM25 queries 2–4× faster, byte-identical results.
- **2026-07-03 — `count()`**: key-only group/window counting (~18× vs
  `len(find())`); **`fields=` projection** on `find()`/`range()` (skips
  unrequested blobs).
- **2026-07-06 — Silent-data-loss fixes**: `str`/`"str"` dict-schema dtype
  normalisation (was numpy `<U0` → every value stored as `""`);
  `_serialize` rejects unknown fields (was: record of defaults).
- **2026-07-07 — `Vector` index on Collections**: pre-filtered exact
  similarity, `nearest(where=...)` planned through existing indexes; triple
  store + filtered-vector-search tutorials; README refresh.
- **2026-07-08 — Handle re-binding**: caller-held Collection handles survive
  `vacuum()`/`migrate_collection` (weak registry, in-place re-bind).
- **2026-07-08 — Fuzz harness** (`tests/test_fuzz_collection.py`): thousands
  of random ops vs an in-memory model across every index kind, vacuum/
  migrate/reopen interleaved; first soak caught and fixed the dataset-
  identifier leak that let ~15 migrations brick a file.
- **2026-07-08 — Counted group indexes**: `Many(counted=True)` maintains a
  companion group→count Dict; `count()` O(1), `groups()` lists all groups by
  volume with zero record reads. Opt-in; windowed counts keep the key scan.
- **2026-07-08 — QoL**: `col.latest(index)` / `col.first(index)` (the record
  directly, or None); dropped-collection handles are poisoned — any use
  raises `CollectionDroppedError` instead of undefined stale behaviour.
- **2026-07-08 — Insert hotspots closed**: the struct-pack fast plan now
  covers text/json/blob/vector schemas (fallible phase side-effect-free,
  blob writes in a second phase, file byte-identical to the generic path
  and to master — proven by sha256); tokenizer fast path (precompiled
  regex, set-lookup punctuation drop, exact-equivalence guard against
  eldar's constants). Wide-schema inserts +12%, tokenization 1.26×.
