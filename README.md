# Loom

**Fast, persistent data structures for Python**

Loom provides memory-mapped, persistent dictionaries and lists with automatic crash recovery and minimal overhead.

## Features

- 🚀 **Fast**: Memory-mapped I/O with ~1-2 μs operations
- 💾 **Persistent**: All data automatically saved to disk
- 🔒 **Crash-safe**: Optional atomic operations with WAL recovery
- 📦 **Nested structures**: Dictionaries of dictionaries, lists of lists
- 🎯 **Bloom filters**: Fast membership checks for large datasets
- 🔄 **Auto-growth**: Hash tables and blocks grow automatically
- 💨 **Zero-copy**: Direct access to memory-mapped data

## Quick Start

```python
from loom.database import DB

# Create database
with DB("mydata.db") as db:
    # Define schema
    user_ds = db.create_dataset("users", id="uint64", name="U50", email="U100")
    
    # Create persistent dict
    users = db.create_dict("users_dict", user_ds)
    
    # Insert data
    users["alice"] = {"id": 1, "name": "Alice", "email": "alice@example.com"}
    users["bob"] = {"id": 2, "name": "Bob", "email": "bob@example.com"}
    
    # Read data
    print(users["alice"])  # {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}
    
    # Iterate
    for key, value in users.items():
        print(f"{key}: {value['name']}")

# Data persists automatically!
```

## Installation

```bash
git clone <repository-url>
cd loom
pip install -r requirements.txt
```

## Core Data Structures

### Dict

Persistent hash table with automatic growth:

```python
# Create dict
users = db.create_dict("users", user_ds, bloom_size=100000, cache_size=1000)

# Operations
users["alice"] = {"id": 1, "name": "Alice"}  # Insert/update
user = users["alice"]                         # Read
del users["alice"]                            # Delete
"alice" in users                              # Check existence
len(users)                                    # Size

# Iteration
for key in users.keys(): ...
for value in users.values(): ...
for key, value in users.items(): ...

# Atomic operations (crash-safe)
users.set("bob", {"id": 2, "name": "Bob"}, atomic=True)
```

### List

Persistent list with automatic block allocation:

```python
# Create list
logs = db.create_list("logs", log_ds, cache_size=100)

# Operations
logs.append({"timestamp": 123, "message": "Event"})  # Append
logs.append_many(items, atomic=True)                 # Batch append
log = logs[0]                                        # Read by index
logs[0] = {"timestamp": 124, "message": "Updated"}   # Update
del logs[5]                                          # Delete
len(logs)                                            # Size

# Slicing
recent = logs[-100:]      # Last 100 items
subset = logs[10:20]      # Range
every_other = logs[::2]   # Step

# Iteration
for log in logs: ...
for i, log in enumerate(logs): ...
```

### Nested Structures

Dictionaries of dictionaries, lists of lists:

```python
# Nested dicts
user_ds = db.create_dataset("users", id="uint32", name="U50")
UserDict = Dict.template(user_ds)
teams = db.create_dict("teams", UserDict)

# Auto-create nested dict on access
eng_team = teams["engineering"]
eng_team["alice"] = {"id": 1, "name": "Alice"}

# Access nested values
print(teams["engineering"]["alice"]["name"])  # "Alice"

# Nested lists
item_ds = db.create_dataset("items", id="uint32", value="float32")
ItemList = List.template(item_ds)
categories = db.create_list("categories", ItemList)

# Append nested list
electronics = categories.append()
electronics.append({"id": 1, "value": 999.99})
```

## Atomic Operations

Use atomic operations for crash safety via Write-Ahead Logging (WAL):

```python
# Dict atomic insert
users.set("alice", {"id": 1, "name": "Alice"}, atomic=True)

# List atomic append
logs.append({"timestamp": 123, "message": "Event"}, atomic=True)

# Batch atomic operations
logs.append_many([
    {"timestamp": 123, "message": "Event 1"},
    {"timestamp": 124, "message": "Event 2"},
], atomic=True)
```

**Performance**:
- Fast path (default): ~1-2 μs per operation
- Atomic path: ~2-4 μs per operation

**Recovery**: On crash, WAL automatically replays uncommitted transactions on database open.

## Performance

### Benchmarks

**Dict operations** (1M items):
- Insert: ~1.5 μs per item
- Read (cached): ~0.5 μs per item
- Read (uncached): ~2 μs per item
- Contains (with bloom): ~0.3 μs per check

**List operations** (1M items):
- Append: ~1.2 μs per item
- Read by index: ~0.8 μs per item
- Slice (100 items): ~80 μs

### Optimization Tips

1. **Use bloom filters** for large dicts:
   ```python
   users = db.create_dict("users", user_ds, bloom_size=100000)
   ```

2. **Adjust cache sizes**:
   ```python
   hot_data = db.create_dict("hot", ds, cache_size=10000)
   cold_data = db.create_dict("cold", ds, cache_size=10)
   ```

3. **Batch atomic operations**:
   ```python
   logs.append_many(items, atomic=True)  # vs many individual appends
   ```

4. **Use fast path for non-critical data**:
   ```python
   cache[key] = value  # Fast (default)
   critical.__setitem__(key, value, atomic=True)  # Crash-safe
   ```

## Architecture

### Memory-Mapped I/O

All data is memory-mapped for zero-copy access:

```
┌─────────────────────────────────────┐
│         Database File               │
├─────────────────────────────────────┤
│  Header (4KB)                       │
│  - Metadata                         │
│  - Dataset registry                 │
├─────────────────────────────────────┤
│  Datasets                           │
│  - User data (structured records)   │
│  - Hash tables (Dict)               │
│  - Block arrays (List)              │
├─────────────────────────────────────┤
│  Data Structures                    │
│  - Dict metadata                    │
│  - List metadata                    │
│  - Bloom filters                    │
└─────────────────────────────────────┘
```

### Hash Table Growth

Dict uses linear hashing for automatic growth:

```
Initial: 1 table (256 slots)
After growth: 2 tables (256 + 512 slots)
After growth: 3 tables (256 + 512 + 1024 slots)
...
```

### Block Allocation

List uses exponential block growth:

```
Block 0: 256 items
Block 1: 512 items
Block 2: 1024 items
Block 3: 2048 items
...
```

### Write-Ahead Logging

Atomic operations use WAL for crash safety:

```
1. Log operation to WAL file
2. Apply operation to data file
3. Commit WAL entry
4. On crash: replay uncommitted WAL entries
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Tutorial and examples
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [PLAN.md](PLAN.md) - Roadmap and development status

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_dict_atomic.py -v

# Run with coverage
python -m pytest tests/ --cov=loom --cov-report=html
```

**Current status**: ✅ 168/168 tests passing

## Examples

### User Management

```python
with DB("users.db") as db:
    user_ds = db.create_dataset("users", 
        id="uint64", 
        username="U50", 
        email="U100",
        is_active="bool"
    )
    users = db.create_dict("users", user_ds, bloom_size=100000)
    
    users["alice"] = {
        "id": 1,
        "username": "alice",
        "email": "alice@example.com",
        "is_active": True
    }
    
    if "alice" in users:
        print(f"Welcome {users['alice']['username']}!")
```

### Event Log

```python
with DB("events.db") as db:
    event_ds = db.create_dataset("events",
        timestamp="uint64",
        event_type="U50",
        user_id="uint64",
        data="U500"
    )
    events = db.create_list("events", event_ds)
    
    # Atomic append for crash safety
    events.append({
        "timestamp": 1234567890,
        "event_type": "login",
        "user_id": 1,
        "data": "User logged in"
    }, atomic=True)
    
    # Query recent events
    recent = events[-100:]
```

### Hierarchical Data

```python
with DB("hierarchy.db") as db:
    item_ds = db.create_dataset("items", id="uint32", name="U50", value="float32")
    ItemList = List.template(item_ds)
    categories = db.create_list("categories", ItemList)
    
    # Add category with items
    electronics = categories.append()
    electronics.append({"id": 1, "name": "Laptop", "value": 999.99})
    electronics.append({"id": 2, "name": "Phone", "value": 699.99})
    
    # Access nested data
    print(categories[0][0]["name"])  # "Laptop"
```

## Limitations

### Current Limitations

- **No concurrent access**: Single process only (no file locking yet)
- **No multi-operation transactions**: Atomic operations are per-operation
- **No compression**: Data stored uncompressed
- **No encryption**: Data stored in plaintext
- **Fixed schemas**: Cannot modify dataset schema after creation

### Roadmap

See [PLAN.md](PLAN.md) for planned features and improvements.

## Contributing

Contributions welcome! Please:

1. Run tests: `python -m pytest tests/ -v`
2. Follow existing code style
3. Add tests for new features
4. Update documentation

## License

[Add license information]

## Acknowledgments

Built with:
- Memory-mapped I/O via Python's `mmap`
- NumPy for structured arrays
- Linear hashing for Dict growth
- Write-Ahead Logging for crash recovery
