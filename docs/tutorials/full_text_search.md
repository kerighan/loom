# Full-text search — 20 Newsgroups

A persistent inverted index with boolean and BM25 ranked search, built on a
real corpus: the **20 Newsgroups** dataset (~18 k documents, via scikit-learn).

```bash
pip install scikit-learn
```

## Load the corpus into a dataset

The `SearchIndex` is built on a user `Dataset` (the document store): `add()`
keeps only the record's address, so documents are never duplicated.

```python
from sklearn.datasets import fetch_20newsgroups
from loom import DB, Utf8

news = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
target_names = news.target_names

db = DB("news.db", header_size=65536)
docs = db.create_dataset("docs", group=Utf8(40), body="text")
idx = db.create_search_index("news_idx", docs,
                             text_fields=["body"], scoring="bm25")
```

## Index the documents

```python
from tqdm import tqdm

for text, label in tqdm(zip(news.data, news.target), total=len(news.data)):
    idx.add({"group": target_names[label], "body": text})

db.flush()
print(len(idx), "documents indexed")
```

`add()` buffers; the index materialises on the next `search()` or an explicit
`flush()`. For bulk loads this keeps per-term postings writes amortised.

## Query

```python
# Boolean (unranked, doc-id order): AND / OR / AND NOT, parentheses, * wildcards
idx.search("encryption AND NOT export", mode="boolean")

# Ranked (BM25, best first)
hits = idx.search("space shuttle launch")
for doc in hits[:5]:
    print(doc["group"], "—", doc["body"][:80].replace("\\n", " "))

# Scores + limit
for doc, score in idx.search("graphics card driver", with_scores=True, limit=10):
    print(round(score, 2), doc["group"])

# Wildcard
idx.search("crypt*")
```

```{note}
Combine terms with explicit operators. A bare `"a b"` is treated as a *phrase*
(needs positional postings, not stored here); use `"a AND b"` / `"a OR b"`.
```

## Read a stored document by id

```python
doc_id = idx.add({"group": "sci.space", "body": "a new post about mars rovers"})
idx.get_document(doc_id)        # the stored record
idx.delete(doc_id)             # tombstone it (drops from future results)
```

## Performance & disk

The reference benchmark (`benchmarks/benchmark_search.py`) reports build
throughput, query latency (well under a millisecond for typical terms), and
on-disk size via `db._db.get_used_space()`. Postings are stored as compact
delta + varint blobs (monotonic doc-ids), rebuilt once per flush.

## Full-text inside a Collection

When documents are records you also query structurally, declare the search
index *inside* a {doc}`Collection <social_posts>` with `Search(...)` and use
`collection.search("body", query, where=...)` — same engine, zero duplication,
and you get the structured filters (category, time) for free.
