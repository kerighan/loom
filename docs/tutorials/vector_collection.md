# Filtered vector search — a `Vector` index on a Collection

"Find the 10 posts most similar to this one **among last month's posts on this
topic**" is the query real semantic features are made of — and it's exactly
where ANN indexes struggle: filter *after* the ANN search and a selective
filter empties your top-k; filter *before* and the ANN index can't help.

A Collection `Vector` index takes the other path: it's an **exact, pre-filtered
flat search**. The collection's regular indexes narrow the candidates first
(a group, a date window), then only the survivors' vectors are read and scored
precisely. No recall to tune — and the more selective the filter, the *faster*
the query.

```{note}
Cost model: nothing is maintained on write (the vector lives inline in the
record — the spec is pure configuration), and a query is one projected row
read per candidate + one numpy matmul. 100k docs (dim 384): full scan 1.7 s,
a 2.5k-candidate group 79 ms, group + month window (830 candidates) 28 ms.
If you need *unfiltered* search over millions of vectors, that's the
standalone `FlatIndex`/`IVFIndex`'s job (see the semantic-search tutorial).
```

## The model

```python
from datetime import datetime
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers
from loom import DB, Many, Vector, Vec, Utf8

model = SentenceTransformer("all-MiniLM-L6-v2")          # 384-dim

class Post(BaseModel):
    id:         Utf8(32)
    topic:      Utf8(32)
    created_at: datetime
    text:       str
    emb:        Vec(384)          # stored inline in the record
```

## Declaring the collection

```python
db = DB("posts.db", header_size=131072)

posts = db.collection("posts", Post, indexes={
    "id":         "primary",
    "topic":      Many(sort="created_at", desc=True),   # the pre-filter axes
    "created_at": "range",
    "emb":        Vector(metric="cosine"),              # or "l2", "dot"
})

def ingest(rows):
    posts.insert_many([Post(**r, emb=model.encode(r["text"])) for r in rows])
```

## Querying

`nearest()` takes the same `where=` spec as `search()`: `{field: value}` for
equality, `{field: (lo, hi)}` for a range (either bound may be `None`), or a
callable.

```python
q = model.encode("farmers protesting against trade agreements")

posts.nearest("emb", q, k=10)                       # whole collection (flat)

posts.nearest("emb", q, k=10,                       # only politics posts
              where={"topic": "politics"})

posts.nearest("emb", q, k=10, with_scores=True,     # politics, June onward →
              where={"topic": "politics",           # ONE composite-key seek,
                     "created_at": (datetime(2026, 6, 1), None)})

posts.nearest("emb", q, k=10, fields=["id", "text"])   # projected results
```

The planner picks the most selective indexed entry of `where` to drive the
scan: a `unique`/`many` equality first — and when the many's *sort field* is
also bounded (the `topic` + `created_at` case above), both fold into a single
composite-key seek — then a `range` index. Whatever remains is applied as a
per-record predicate. Candidates only ever pay a projected row read (their
vector + the predicate fields); full records are materialized for the k
winners alone.

## Similar-to-this-post

Self-similarity makes "related posts" one call — read the anchor's vector with
a projection, exclude the anchor from the results:

```python
def related(post_id, k=5):
    anchor = posts[post_id]
    hits = posts.nearest("emb", anchor["emb"], k=k + 1,
                         where={"topic": anchor["topic"]})
    return [h for h in hits if h["id"] != post_id][:k]
```

## Semantic + full-text, side by side

`Vector` composes with a `Search` index on the same collection — BM25 for
exact wording, `nearest()` for meaning:

```python
posts_ft = db.collection("posts_ft", Post, indexes={
    "id":    "primary",
    "topic": Many(sort="created_at", desc=True),
    "body":  Search(fields=["text"], scoring="bm25"),
    "emb":   Vector(metric="cosine"),
})

posts_ft.search("body", "Mercosur AND farmers")     # lexical
posts_ft.nearest("emb", q, k=10,                    # semantic, same filters
                 where={"topic": "politics"})
```

## Choosing a metric

- `"cosine"` — direction only; the norms are divided out. The default for
  sentence embeddings.
- `"dot"` — inner product; right when your model bakes relevance into the
  norm (e.g. some retrieval models).
- `"l2"` — Euclidean distance; ranks **ascending** and `with_scores=True`
  returns the actual distance.

Scores come back exact — `with_scores=True` yields `(record, score)` pairs
identical to a numpy brute force over the same candidate set.
