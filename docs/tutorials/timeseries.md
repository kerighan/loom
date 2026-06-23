# Time-series & feeds — BTree ranges

Ordered keys with O(log n + k) range scans in both directions — for ticks,
logs, and "latest N" feeds.

## Integer-keyed time-series (epoch ticks)

```python
from loom import DB

db = DB("ticks.db")
ohlcv = db.create_dataset("ohlcv", o="float64", h="float64",
                          l="float64", c="float64", v="int64")
ticks = db.create_btree("ticks", ohlcv, int_keys=True)   # numeric key ordering

# keys are epoch seconds (ints): ordered numerically, not lexicographically
ticks[1700000000] = {"o": 1.0, "h": 1.2, "l": 0.9, "c": 1.1, "v": 1000}
ticks[1700003600] = {"o": 1.1, "h": 1.3, "l": 1.0, "c": 1.25, "v": 1500}

# a window
for ts, bar in ticks.range(1700000000, 1700003600):
    print(ts, bar["c"])

# most recent 50 (seek to the end, walk down)
latest = list(ticks.range(reverse=True))[:50]
ticks.min(), ticks.max()
```

## Datetime keys

Prefer storing time as a `datetime` field (int64 epoch-µs) on records and
indexing it. In a {doc}`Collection <social_posts>` a `range`/`many` index over
a `datetime` field sorts chronologically for free:

```python
from datetime import datetime
from loom import DB, Many

db = DB("events.db")
events = db.collection("events", {"id": "utf8[16]", "created_at": "datetime", "kind": "utf8[16]"},
                       indexes={"id": "primary", "created_at": "range"})
events.insert_many([
    {"id": f"e{i}", "created_at": datetime(2026, 1, 1 + i), "kind": "click"}
    for i in range(100)
])

events.range("created_at", limit=50, desc=True)              # 50 most recent
events.range("created_at", datetime(2026, 2, 1), datetime(2026, 3, 1))  # a window
```

## A relevance inbox (single axis, no grouping)

A plain `range` index iterates either direction, so "top by score" needs no
grouping field:

```python
inbox = db.collection("inbox", {"id": "utf8[16]", "relevance": "float64", "text": "text"},
                      indexes={"id": "primary", "relevance": "range"})
inbox.range("relevance", limit=20, desc=True)     # 20 most relevant
inbox.range("relevance", 0.8, None)               # relevance >= 0.8
```

## Per-group ordered feeds + time filter

For "category X in a time window" use a `Many(sort="created_at")` composite
index and bound the sort field — a single seek + bounded scan, independent of
how much history the group holds:

```python
posts.find("cat_time", "politics", start=datetime(2026, 1, 1))
```

See {doc}`social_posts` for the full feed/compound-filter walkthrough.
