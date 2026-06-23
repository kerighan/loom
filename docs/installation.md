# Installation

loom is an internal, source-distributed library — install it from the checkout:

```bash
pip install -e /path/to/loom
```

## Dependencies

| Package | Needed for |
|---|---|
| `numpy` | core (mmap record encoding) |
| `pydantic` | declaring schemas as models (optional — dict schemas work too) |
| `lru-dict` | the shared address cache |
| `eldar` | boolean / BM25 query parsing for `SearchIndex` |
| `mmh3` | fast hashing (Bloom filters, Dict slots) |
| `brotli` | optional `blob_compression="brotli"` |
| `fastapi` + `uvicorn` | optional — `db.serve()` HTTP API |

```bash
pip install numpy pydantic lru-dict eldar mmh3
pip install "fastapi[standard]" brotli      # optional extras
```

## Building the docs

```bash
pip install -r docs/requirements.txt
cd docs && make html        # → docs/_build/html/index.html
```

A pre-commit hook rebuilds the docs (warnings = errors) whenever `loom/` or
`docs/` changes, so the docs stay in sync with the docstrings. Enable it once
per clone:

```bash
git config core.hooksPath .githooks
```

(Skip a one-off check with `git commit --no-verify`.)

## Sanity check

```python
from loom import DB
with DB("smoke.db") as db:
    d = db.create_dict("kv", {"v": "int64"})
    d["x"] = {"v": 1}
    assert d["x"]["v"] == 1
print("loom OK")
```
