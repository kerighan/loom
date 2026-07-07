# Triple store — a knowledge graph as a `Collection`

Knowledge-graph edges extracted from documents are **facts with metadata**:
who, relation, whom, *when*, and *from which source document*. When your
queries are analytic — edges over a time window, an entity's latest
relations, degree counts, provenance — a `Collection` of triples is a better
fit than the `Graph` structure, because every one of those questions is an
indexed lookup.

```{note}
Rule of thumb: queries that **start from a node and follow edges** (paths,
n-hop patterns, Cypher) belong to the `Graph` structure. Queries that
**start from a filter** (dates, relation types, top-k recent, counts) belong
to a `Collection`. The two compose: keep the collection as the dated source
of truth and derive a `Graph` from it when you need deep traversals.
```

## The model

One record per extracted triple. The primary key is a hash of the whole
fact — re-ingesting the same document is an idempotent upsert. `src_rel` is
a combined field so that *(source, relation)* pairs get their own index.

```python
from datetime import datetime
from hashlib import blake2b
from pydantic import BaseModel
from loom import DB, Many, Utf8

class Triple(BaseModel):
    triple_id:  Utf8(32)      # blake2b(source, relation, target, ref_id)
    source:     Utf8(64)
    relation:   Utf8(32)
    target:     Utf8(64)
    src_rel:    Utf8(97)      # "{source}|{relation}" — compound lookups
    created_at: datetime      # when the fact was stated / published
    ref_id:     Utf8(32)      # document the triple was extracted from

def make_triple(source, relation, target, created_at, ref_id):
    key = blake2b(f"{source}|{relation}|{target}|{ref_id}".encode(),
                  digest_size=16).hexdigest()
    return Triple(triple_id=key, source=source, relation=relation,
                  target=target, src_rel=f"{source}|{relation}",
                  created_at=created_at, ref_id=ref_id)
```

## Declaring the collection

```python
db = DB("kg.db", header_size=131072)

triples = db.collection("triples", Triple, indexes={
    "triple_id":  "primary",
    "source":     Many(sort="created_at", desc=True),   # outgoing edges, recent first
    "target":     Many(sort="created_at", desc=True),   # incoming edges
    "relation":   Many(sort="created_at", desc=True),   # all "criticizes" edges
    "src_rel":    Many(sort="created_at", desc=True),   # (source, relation) pairs
    "ref_id":     Many(sort="created_at", desc=True),   # provenance: doc → triples
    "created_at": "range",                              # global timeline
})
```

## Ingesting

```python
triples.insert_many([
    make_triple("LFI", "criticizes", "FNSEA",
                datetime(2026, 6, 12), "article_001"),
    make_triple("LFI", "criticizes", "gouvernement",
                datetime(2026, 6, 12), "article_001"),
    make_triple("Confédération paysanne", "opposes", "Mercosur",
                datetime(2026, 6, 20), "article_002"),
    make_triple("gouvernement", "supports", "Mercosur",
                datetime(2026, 7, 1), "article_003"),
])
```

## Querying the graph

```python
# Neighborhoods — a single seek + bounded scan each:
triples.find("source", "LFI")                          # outgoing edges, recent first
triples.find("target", "Mercosur")                     # who talks about Mercosur
triples.find("src_rel", "LFI|criticizes", limit=10)    # typed edges of one node

# Time is a first-class filter on every Many index:
triples.find("source", "LFI", start=datetime(2026, 6, 1))
triples.range("created_at", limit=10_000, desc=True)   # the 10k latest edges

# Degrees without loading a single record (key-only scans):
triples.count("source", "LFI")                                  # out-degree
triples.count("target", "Mercosur", start=datetime(2026, 6, 1)) # windowed in-degree

# Projections skip the fields you don't need:
triples.find("source", "LFI", fields=["target", "relation"])
```

## Provenance

`ref_id` makes extraction auditable in both directions:

```python
triples.find("ref_id", "article_001")     # every fact this article produced
triples.count("ref_id", "article_001")    # how many
```

Deleting a retracted document's facts is a lookup, not a scan:

```python
for t in triples.find("ref_id", "article_001", fields=["triple_id"]):
    triples.delete(t["triple_id"])
```

## Bounded traversals

Shallow hops are loops over `find()`. Deep or pattern-shaped traversals are
the `Graph` structure's job.

```python
def neighborhood(node, hops=2, since=None):
    seen, frontier = {node}, [node]
    for _ in range(hops):
        nxt = []
        for n in frontier:
            for e in triples.find("source", n, start=since,
                                  fields=["target"]):
                if e["target"] not in seen:
                    seen.add(e["target"])
                    nxt.append(e["target"])
        frontier = nxt
    return seen - {node}

neighborhood("LFI", hops=2, since=datetime(2026, 6, 1))
```

## Retention — keep the N most recent edges

The `created_at` range index walks oldest-first by default, so trimming is a
bounded scan of exactly the surplus:

```python
surplus = len(triples) - 10_000
if surplus > 0:
    for t in triples.range("created_at", limit=surplus, fields=["triple_id"]):
        triples.delete(t["triple_id"])
    db.vacuum()      # reclaim the space when the churn adds up
```
