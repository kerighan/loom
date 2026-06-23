# Social media posts — a `Collection` end to end

This is the canonical loom use case: store millions of social-media posts,
look them up by id, by author (recent-first feed), by engagement, filter a
category over a time window, and run full-text search — all kept in sync, all
on disk.

## The model

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel
from loom import DB, Many, Range, Search, Utf8

class Post(BaseModel):
    post_id:           Utf8(32)            # primary key
    username:          Utf8(30)
    category_alias:    Utf8(64)
    political_leaning: Literal["left", "center", "right", ""]   # → utf8 sized to "center"
    engagements:       int
    created_at:        datetime
    text:              str
```

## Declaring the collection

Each field's *index kind* maps to the right structure and is kept in sync on
every write:

```python
db = DB("posts.db", header_size=131072)   # roomy header: many indexes

posts = db.collection("posts", Post, indexes={
    "post_id":   "primary",                                   # the record store
    "username":  Many(sort="created_at", desc=True),          # a user's feed, recent first
    "category":  Many(field="category_alias", sort="engagements", desc=True),  # top of a category
    "cat_time":  Many(field="category_alias", sort="created_at",  desc=True),   # category over time
    "engagements": "range",                                   # engagement >= x
    "body":      Search(fields=["text"], scoring="bm25"),     # full-text
})
```

```{note}
`field=` lets several indexes back the **same** field — here `category_alias`
is indexed both by engagement (`category`) and by date (`cat_time`).
```

## Ingesting (daily, idempotent)

`insert` / `insert_many` are **upserts**: re-running a daily dump with posts
that already exist replaces them and re-points every index. Pass Pydantic
models directly — no `.model_dump()`.

```python
def ingest(rows):
    posts.insert_many(Post(**r) for r in rows)

ingest([
    {"post_id": "p1", "username": "alice", "category_alias": "politics",
     "political_leaning": "left", "engagements": 1500,
     "created_at": datetime(2026, 1, 5, 9, 0), "text": "an inverted index in pure python"},
    {"post_id": "p2", "username": "alice", "category_alias": "tech",
     "political_leaning": "", "engagements": 50,
     "created_at": datetime(2026, 2, 1, 18, 0), "text": "fast disk search"},
    {"post_id": "p3", "username": "bob", "category_alias": "politics",
     "political_leaning": "right", "engagements": 9000,
     "created_at": datetime(2026, 3, 2, 12, 0), "text": "sequential scan postings"},
])
```

## Reading

```python
posts["p1"]                                   # by primary key → record
posts.find("username", "alice", limit=20)     # alice's 20 most-recent posts
posts.find("category", "politics", limit=10)  # top-10 politics posts by engagement
posts.range("engagements", 1000, None)        # engagement >= 1000

# category + time window — a single seek + bounded scan, O(log n + matches),
# fast no matter how much history "politics" holds:
posts.find("cat_time", "politics", start=datetime(2026, 1, 1))
posts.find("cat_time", "politics",
           start=datetime(2026, 1, 1), end=datetime(2026, 2, 28))
```

## Full-text + structured filter

`search(..., where=...)` combines a BM25 query with field constraints:

```python
posts.search("body", "inverted OR index")                      # ranked records
posts.search("body", "search",
             where={"category_alias": "politics",
                    "created_at": (datetime(2026, 1, 1), None)})  # AND filter
posts.search("body", "search", with_scores=True, limit=20)     # (record, score)
```

## Updating engagement (live counters)

```python
posts.increment("p1", "engagements", 1)        # atomic bump, re-indexes
posts["p1"]["engagements"] = 5000              # write-through record → re-index
posts.update("p1", category_alias="news")      # re-indexes only changed fields
```

## Iterating, counting, deleting

```python
len(posts)
for post in posts:          # yields records (use .keys()/.items() for pks)
    ...
posts.delete("p2")          # removed from every index
```

## Maintenance

```python
db.stats()                  # file size, fragmentation, per-collection counts
db.verify()                 # check every index points to a live record

# add / drop / rename fields later:
db.migrate_collection("posts", PostV2, transforms={
    "sentiment": lambda old: classify(old["text"]),
})

db.vacuum()                 # reclaim space from deletes / upserts / migrations
```

## Reopening

```python
with DB("posts.db") as db:        # header_size auto-detected
    posts = db["posts"]           # or db.collection("posts")
    print(len(posts))
```
