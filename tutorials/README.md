# Loom Tutorials

Minimal, focused examples for learning Loom.

## Getting Started

1. **[01_quickstart.py](01_quickstart.py)** - Your first Loom database
2. **[02_lists.py](02_lists.py)** - Working with persistent lists
3. **[03_bloom_filters.py](03_bloom_filters.py)** - Fast membership testing
4. **[04_datasets.py](04_datasets.py)** - Low-level typed records
5. **[05_nested_lists.py](05_nested_lists.py)** - Lists of lists
6. **[06_context_manager.py](06_context_manager.py)** - Automatic cleanup
7. **[07_atomic_writes.py](07_atomic_writes.py)** - Crash-safe operations
8. **[08_real_world_example.py](08_real_world_example.py)** - Web scraper example

## Quick Reference

```python
from loom.database import DB

with DB("data.loom") as db:
    # List (most common)
    items = db.create_list("items", {"id": "uint64", "name": "U50"})
    items.append({"id": 1, "name": "Alice"})
    print(items[0])      # {'id': 1, 'name': 'Alice'}
    print(items[0:10])   # List of dicts
    
    # Bloom filter (deduplication)
    seen = db.create_bloomfilter("seen", expected_items=10000)
    seen.add("key")
    print("key" in seen)  # True
    
    # Counting bloom filter (with counts)
    cache = db.create_counting_bloomfilter("cache", expected_items=1000)
    cache.add("key")
    print(cache.count("key"))  # 1
```

## Schema Types

| Type | Description | Example |
|------|-------------|---------|
| `uint8`, `uint16`, `uint32`, `uint64` | Unsigned integers | `"age": "uint8"` |
| `int8`, `int16`, `int32`, `int64` | Signed integers | `"balance": "int64"` |
| `float32`, `float64` | Floating point | `"price": "float64"` |
| `bool` | Boolean | `"active": "bool"` |
| `U10`, `U50`, `U100`, `U200` | Fixed-length strings | `"name": "U50"` |
