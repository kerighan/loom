# Semantic search — embeddings

Store embeddings and run nearest-neighbour search, persisted on disk. loom has
two vector indexes: `FlatIndex` (exact) and `IVFIndex` (approximate, trained,
optional Product-Quantization compression).

```bash
pip install sentence-transformers      # for real embeddings (optional)
```

## Exact search — `FlatIndex`

```python
from loom import DB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")    # 384-dim
passages = [
    "loom stores data structures on disk with mmap",
    "BM25 ranks documents by term frequency",
    "a B-tree keeps keys in sorted order",
    "vector search finds nearest neighbours by cosine",
]

db = DB("vectors.db")
idx = db.create_flat_index("passages", dim=384, metric="cosine")

for i, text in enumerate(passages):
    idx.add(f"p{i}", model.encode(text))

q = model.encode("how does loom persist data?")
for doc_id, score in idx.search(q, k=3):
    print(round(score, 3), doc_id)
```

`FlatIndex` compares against every vector — exact, no training, ideal up to a
few hundred thousand vectors.

## Approximate search — `IVFIndex`

For millions of vectors, an inverted-file index trades a little recall for
speed. It must be trained on a representative sample first.

```python
import numpy as np

ivf = db.create_ivf_index("big", dim=384, metric="cosine",
                          n_clusters=256)          # rule of thumb: ~sqrt(n)
ivf.train(np.random.rand(10_000, 384).astype("float32"))   # representative sample
for i in range(1_000_000):
    ivf.add(f"v{i}", some_vector(i))

ivf.search(query_vec, k=10)
```

### Product Quantization (compression)

```python
ivf_pq = db.create_ivf_index("compact", dim=1536, metric="cosine",
                            n_clusters=256, pq=True, n_sub=16, n_bits=8)
# 1536 × float32 = 6 144 bytes → 16 bytes per vector with PQ (~384× smaller)
```

## Storing embeddings inside records

To keep the vector *with* its record (text, metadata) instead of a separate
index, use a `Vec(dim)` field on a dataset:

```python
from pydantic import BaseModel
from loom import Vec

class Passage(BaseModel):
    text:      str
    embedding: Vec(384)        # 384 × float32 stored inline

ds = db.create_dataset("passages", Passage)
ref = ds.insert({"text": "...", "embedding": model.encode("...")})
ds[ref.addr]["embedding"]      # → numpy array (384,)
```

Pair this with a `FlatIndex`/`IVFIndex` keyed by the same ids to get both the
nearest-neighbour search and the full records in one database.
