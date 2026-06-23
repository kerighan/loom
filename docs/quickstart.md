# Quickstart

## The `DB` object

Everything lives in one file, opened through a `DB`:

```python
from loom import DB

with DB("app.db") as db:          # context manager: auto-saves + flushes on exit
    ...
```

`DB("app.db")` also works without `with` — but then call `db.close()` (or
`db.flush()`) before relying on a reopened `len()` (an `atexit` net closes
forgotten DBs on a clean exit, but not on a crash).

```{tip}
`header_size` is stored in the file and auto-detected on reopen — you only set
it at creation (raise it if you hit `HeaderTooLargeError` with many indexes).
```

## Schemas

Declare records as a Pydantic model or a plain dict of dtypes:

```python
from datetime import datetime
from pydantic import BaseModel, Field
from loom import Utf8, Json

class Post(BaseModel):
    post_id:    Utf8(32)        # fixed-width inline UTF-8 (raises if over budget)
    username:   Utf8(30)
    text:       str             # variable-length, BlobStore
    created_at: datetime        # stored inline as int64 epoch-µs
    meta:       dict            # arbitrary JSON (json.dumps/loads)

# or, equivalently, a dict:
schema = {"post_id": "utf8[32]", "created_at": "datetime", "text": "text"}
```

| Python type | loom dtype | notes |
|---|---|---|
| `int` / `float` / `bool` | `int64` / `float64` / `bool` | |
| `datetime` / `date` | `datetime` | int64 epoch-µs, naturally ordered |
| `Literal["a", "b"]` | `utf8[N]` | sized to the longest value |
| `str` | `text` | BlobStore, compressible |
| `Utf8(N)` | `utf8[N]` | inline, ~4× smaller than `U{N}`; raises on overflow |
| `dict` / `Json()` | `json` | any JSON value |

## Creating vs retrieving

Names are unique across the whole DB. `create_*` **raises** if the name exists
(re-running a script won't silently duplicate). Retrieve with `db[name]`:

```python
ds   = db.create_dataset("posts", Post)
feed = db.create_list("feed", ds)

db["feed"]                                  # existing structure (or dataset, or collection)
db.create_list("feed", ds, exist_ok=True)   # idempotent open-or-create
```

## Pick a structure

```python
db.create_dict("by_id", ds)            # O(1) key → record
db.create_btree("by_time", ds)         # ordered keys, range queries
db.create_list("log", ds)             # append-only / indexed
db.create_set("seen", key_size=32)     # unique strings
db.create_queue("jobs", Job)           # FIFO
db.create_priority_queue("work", Job)  # highest priority first
db.create_search_index("idx", ds, scoring="bm25")   # full-text
db.create_flat_index("vecs", dim=384)  # vector similarity
db.collection("posts", Post, indexes={...})         # record store + typed indexes
```

The tutorials walk through complete, runnable use cases for each.
