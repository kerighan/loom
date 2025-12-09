# Loom API Reference

## Database

### `DB(filename, header_size=4096)`

Main database class for managing persistent data structures.

**Parameters**:
- `filename` (str): Path to database file
- `header_size` (int, optional): Size of header in bytes (default: 4096)

**Methods**:

#### `open()`
Open the database file and load metadata.

```python
db = DB("mydata.db")
db.open()
```

#### `close()`
Close the database and flush all changes.

```python
db.close()
```

#### Context Manager
Automatically opens and closes the database.

```python
with DB("mydata.db") as db:
    # Use database
    pass
# Automatically closed
```

#### `create_dataset(name, **schema)`
Create a new dataset with specified schema.

**Parameters**:
- `name` (str): Unique dataset name
- `**schema`: Field definitions as numpy dtype strings

**Returns**: `Dataset` instance

**Raises**: `ValueError` if dataset name already exists

**Example**:
```python
users = db.create_dataset("users", 
    id="uint64", 
    name="U50",      # Unicode string, max 50 chars
    age="int32",
    score="float32"
)
```

#### `get_dataset(name)`
Get an existing dataset by name.

**Parameters**:
- `name` (str): Dataset name

**Returns**: `Dataset` instance

**Raises**: `KeyError` if dataset doesn't exist

#### `has_dataset(name)`
Check if dataset exists.

**Parameters**:
- `name` (str): Dataset name

**Returns**: `bool`

#### `list_datasets()`
Get list of all dataset names.

**Returns**: `list[str]`

#### `delete_dataset(name)`
Delete a dataset.

**Parameters**:
- `name` (str): Dataset name

**Raises**: `KeyError` if dataset doesn't exist

#### `create_dict(name, dataset, bloom_size=None, cache_size=100)`
Create a persistent dictionary.

**Parameters**:
- `name` (str): Unique dict name
- `dataset` (Dataset): Dataset defining value schema
- `bloom_size` (int, optional): Bloom filter size for fast lookups
- `cache_size` (int, optional): LRU cache size (default: 100)

**Returns**: `Dict` instance

**Example**:
```python
user_ds = db.create_dataset("users", id="uint64", name="U50")
users = db.create_dict("users_dict", user_ds, cache_size=1000)
```

#### `create_list(name, dataset, cache_size=100)`
Create a persistent list.

**Parameters**:
- `name` (str): Unique list name
- `dataset` (Dataset): Dataset defining item schema
- `cache_size` (int, optional): LRU cache size (default: 100)

**Returns**: `List` instance

---

## Dataset

### `Dataset(name, db, identifier, **schema)`

Low-level dataset for storing structured records.

**Note**: Usually created via `DB.create_dataset()`, not directly.

### Operations

#### Read/Write
```python
# Write record
dataset[address] = {"id": 1, "name": "Alice", "age": 30}

# Read record
record = dataset[address]

# Check if record exists (not deleted)
if address in dataset:
    ...

# Delete record (soft delete)
del dataset[address]
```

#### Allocation
```python
# Allocate single record
addr = dataset.allocate()

# Allocate block of records
addr = dataset.allocate_block(count=100)
```

---

## Dict

### `Dict(name, db, dataset, bloom_size=None, cache_size=100)`

Persistent hash table with automatic growth.

**Note**: Usually created via `DB.create_dict()`, not directly.

### Operations

#### Basic Operations

```python
# Insert/Update (fast path)
users["alice"] = {"id": 1, "name": "Alice", "score": 95.5}

# Insert/Update (atomic - crash-safe)
users.set("alice", {"id": 1, "name": "Alice", "score": 95.5}, atomic=True)

# Read
user = users["alice"]

# Check existence
if "alice" in users:
    ...

# Delete (soft delete)
del users["alice"]

# Length
count = len(users)
```

#### Iteration

```python
# Iterate keys
for key in users.keys():
    print(key)

# Iterate values
for value in users.values():
    print(value)

# Iterate items
for key, value in users.items():
    print(f"{key}: {value}")
```

#### Methods

##### `set(key, value, atomic=False)`
Set value for key (convenience method).

**Parameters**:
- `key` (str): Key to set
- `value` (dict): Value matching dataset schema
- `atomic` (bool, optional): Use WAL for crash safety (default: False)

**Example**:
```python
# Fast path (default)
users.set("alice", {"id": 1, "name": "Alice"})

# Atomic path (crash-safe)
users.set("alice", {"id": 1, "name": "Alice"}, atomic=True)
```

##### `__setitem__(key, value, atomic=False)`
Set value for key (low-level method).

**Note**: Use `.set()` method for cleaner syntax with atomic parameter.

**Parameters**:
- `key` (str): Key to set
- `value` (dict): Value matching dataset schema
- `atomic` (bool, optional): Use WAL for crash safety (default: False)

**Example**:
```python
# Fast path (default) - use bracket notation
users["alice"] = {"id": 1, "name": "Alice"}

# Atomic path - must use __setitem__ directly or .set() method
users.__setitem__("alice", {"id": 1, "name": "Alice"}, atomic=True)
# Better: use .set() method
users.set("alice", {"id": 1, "name": "Alice"}, atomic=True)
```

##### `__getitem__(key)`
Get value for key.

**Raises**: `KeyError` if key not found

##### `__delitem__(key)`
Delete key (soft delete).

**Raises**: `KeyError` if key not found

##### `__contains__(key)`
Check if key exists.

##### `keys()`
Iterate over all keys.

##### `values()`
Iterate over all values.

##### `items()`
Iterate over (key, value) pairs.

##### `save()`
Save metadata to disk.

### Nested Dictionaries

```python
# Create nested dict template
user_ds = db.create_dataset("users", id="uint32", name="U50")
UserDict = Dict.template(user_ds, cache_size=10)

# Create parent dict
teams = db.create_dict("teams", UserDict)

# Access nested dict (auto-created)
eng_team = teams["engineering"]

# Add items to nested dict
eng_team["alice"] = {"id": 1, "name": "Alice"}
eng_team["bob"] = {"id": 2, "name": "Bob"}

# Access nested values
print(teams["engineering"]["alice"]["name"])  # "Alice"
```

### Performance Characteristics

- **Insert**: O(1) average, O(n) worst case (hash collision)
- **Read**: O(1) average with cache, O(log n) without
- **Delete**: O(1) (soft delete)
- **Iteration**: O(n)
- **Space**: O(n) with automatic growth

---

## List

### `List(name, db, dataset, cache_size=100)`

Persistent list with automatic block allocation.

**Note**: Usually created via `DB.create_list()`, not directly.

### Operations

#### Basic Operations

```python
# Append (fast path)
items.append({"id": 1, "name": "Item 1"})

# Append (atomic - crash-safe)
items.append({"id": 1, "name": "Item 1"}, atomic=True)

# Append many (atomic batch)
items.append_many([
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"},
], atomic=True)

# Read by index
item = items[0]

# Update by index
items[0] = {"id": 1, "name": "Updated"}

# Delete by index
del items[5]

# Length
count = len(items)
```

#### Slicing

```python
# Get slice
subset = items[10:20]

# Get slice with step
every_other = items[::2]

# Negative indices
last_10 = items[-10:]
```

#### Iteration

```python
# Iterate items
for item in items:
    print(item)

# Enumerate
for i, item in enumerate(items):
    print(f"{i}: {item}")
```

#### Methods

##### `append(item=None, atomic=False)`
Append item to end of list.

**Parameters**:
- `item` (dict): Item matching dataset schema
- `atomic` (bool, optional): Use WAL for crash safety (default: False)

**Returns**: The appended item

##### `append_many(items, atomic=False)`
Append multiple items.

**Parameters**:
- `items` (list): List of items to append
- `atomic` (bool, optional): Batch all writes in single transaction (default: False)

##### `__getitem__(index)`
Get item by index (supports negative indices).

##### `__setitem__(index, item)`
Update item by index.

##### `__delitem__(index)`
Delete item by index (soft delete).

##### `compact()`
Remove deleted items and rebuild list.

**Note**: Automatically triggered when deleted items exceed threshold.

##### `save()`
Save metadata to disk.

### Nested Lists

```python
# Create nested list template
item_ds = db.create_dataset("items", id="uint32", value="float32")
ItemList = List.template(item_ds)

# Create parent list
groups = db.create_list("groups", ItemList)

# Append nested list (auto-created)
group1 = groups.append()

# Add items to nested list
group1.append({"id": 1, "value": 10.5})
group1.append({"id": 2, "value": 20.3})

# Access nested values
print(groups[0][0]["value"])  # 10.5
```

### Performance Characteristics

- **Append**: O(1) amortized
- **Read by index**: O(1) with cache, O(log n) without
- **Update**: O(1)
- **Delete**: O(1) (soft delete)
- **Iteration**: O(n)
- **Compact**: O(n)
- **Space**: O(n) with automatic block allocation

---

## BloomFilter

### `BloomFilter(name, db, capacity=10000, false_positive_rate=0.01)`

Probabilistic set membership test.

### Operations

```python
# Create bloom filter
bloom = BloomFilter("seen_ids", db, capacity=100000)

# Add item
bloom.add("user_123")

# Check membership
if "user_123" in bloom:
    print("Probably in set")

# Length (approximate)
count = len(bloom)

# Clear all items
bloom.clear()
```

### Methods

##### `add(item)`
Add item to filter.

##### `__contains__(item)`
Check if item is probably in set.

**Returns**: `bool` (False = definitely not in set, True = probably in set)

##### `__len__()`
Get approximate count of items.

##### `clear()`
Remove all items.

### Performance Characteristics

- **Add**: O(k) where k = number of hash functions
- **Check**: O(k)
- **Space**: O(m) where m = bit array size
- **False positive rate**: Configurable (default: 1%)

---

## Data Types

### Supported Types

**Integers**:
- `uint8`, `uint16`, `uint32`, `uint64`
- `int8`, `int16`, `int32`, `int64`

**Floats**:
- `float16`, `float32`, `float64`

**Strings**:
- `U<N>`: Unicode string, max N characters (e.g., `U50`)
- `S<N>`: Byte string, max N bytes

**Example**:
```python
dataset = db.create_dataset("records",
    id="uint64",           # 8 bytes
    count="uint32",        # 4 bytes
    score="float32",       # 4 bytes
    name="U50",            # 50 chars (200 bytes UTF-32)
    data="S100"            # 100 bytes
)
```

---

## Atomic Operations

### Write-Ahead Logging (WAL)

Both `Dict` and `List` support atomic operations via WAL for crash safety.

**When to use**:
- Critical data that must not be lost
- Operations where consistency is required
- Batch operations that must succeed or fail together

**Performance**:
- Fast path (default): ~1-2 μs per operation
- Atomic path: ~2-4 μs per operation

### Dict Atomic Operations

```python
# Single atomic insert (recommended syntax)
users.set("alice", {"id": 1, "name": "Alice"}, atomic=True)

# Batch atomic inserts
for i in range(1000):
    users.set(
        f"user_{i}",
        {"id": i, "name": f"User {i}"},
        atomic=True
    )
```

### List Atomic Operations

```python
# Single atomic append
items.append({"id": 1, "name": "Item 1"}, atomic=True)

# Batch atomic append
items.append_many([
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"},
    {"id": 3, "name": "Item 3"},
], atomic=True)
```

### WAL Recovery

On database open, uncommitted WAL transactions are automatically replayed:

```python
# Write with atomic=True
items.append({"id": 1, "name": "Item"}, atomic=True)

# Crash occurs here...

# On reopen, WAL recovery happens automatically
db = DB("mydata.db")  # WAL replayed, data intact
```

---

## Best Practices

### 1. Use Context Managers

```python
# Good
with DB("mydata.db") as db:
    users = db.create_dict("users", user_ds)
    users["alice"] = {"id": 1, "name": "Alice"}
# Automatically closed

# Avoid
db = DB("mydata.db")
db.open()
# ... might forget to close
```

### 2. Choose Appropriate Cache Sizes

```python
# Small dataset, frequent access
users = db.create_dict("users", user_ds, cache_size=1000)

# Large dataset, infrequent access
logs = db.create_list("logs", log_ds, cache_size=10)
```

### 3. Use Atomic Operations for Critical Data

```python
# Critical: financial transactions
transactions.__setitem__(tx_id, tx_data, atomic=True)

# Non-critical: cache data
cache[key] = value  # Fast path
```

### 4. Use Bloom Filters for Large Dicts

```python
# Large dict with frequent membership checks
users = db.create_dict("users", user_ds, bloom_size=100000)

# Fast negative lookups
if "unknown_user" not in users:  # O(k) via bloom filter
    print("Definitely not in dict")
```

### 5. Compact Lists Periodically

```python
# After many deletes
if items.deleted_count > 1000:
    items.compact()
```

### 6. Use Nested Structures Efficiently

```python
# Good: shared datasets for nested structures
UserDict = Dict.template(user_ds, cache_size=10)
teams = db.create_dict("teams", UserDict)

# Avoid: creating separate datasets for each nested structure
```

---

## Error Handling

### Common Exceptions

**`KeyError`**: Key/dataset not found
```python
try:
    user = users["unknown"]
except KeyError:
    print("User not found")
```

**`ValueError`**: Invalid schema or duplicate name
```python
try:
    db.create_dataset("users", invalid_type="bad")
except ValueError as e:
    print(f"Invalid schema: {e}")
```

**`RuntimeError`**: Database not open
```python
db = DB("mydata.db")
try:
    db.create_dataset("users", id="uint64")
except RuntimeError:
    db.open()
    db.create_dataset("users", id="uint64")
```

**`IndexError`**: List index out of range
```python
try:
    item = items[1000]
except IndexError:
    print("Index out of range")
```

---

## Performance Tips

### 1. Batch Operations

```python
# Slow: many individual atomic operations
for i in range(1000):
    items.append(item, atomic=True)

# Fast: single atomic batch
items.append_many(items_list, atomic=True)
```

### 2. Cache Sizing

```python
# Hot data: large cache
active_users = db.create_dict("active", user_ds, cache_size=10000)

# Cold data: small cache
archived = db.create_dict("archived", user_ds, cache_size=10)
```

### 3. Avoid Unnecessary Saves

```python
# Bad: save after every operation
for i in range(1000):
    users[f"user_{i}"] = data
    users.save()  # Expensive!

# Good: auto-save handles it
for i in range(1000):
    users[f"user_{i}"] = data
# Auto-save triggers periodically
```

### 4. Use Bloom Filters

```python
# Without bloom: O(log n) lookup
users = db.create_dict("users", user_ds)

# With bloom: O(k) negative lookups
users = db.create_dict("users", user_ds, bloom_size=100000)
```

---

## Limitations

### Current Limitations

1. **No concurrent access**: Single process only (no file locking)
2. **No transactions**: Atomic operations are per-operation, not multi-operation
3. **No compression**: Data stored uncompressed
4. **No encryption**: Data stored in plaintext
5. **Fixed schemas**: Cannot modify dataset schema after creation

### Workarounds

**Concurrent access**: Use separate database files per process

**Multi-operation transactions**: Use atomic operations + manual rollback

**Compression**: Compress data before storing in dataset

**Encryption**: Encrypt data at application level

**Schema changes**: Create new dataset with new schema, migrate data

---

## Examples

See `examples/` directory for complete examples:
- `basic_usage.py` - Getting started
- `nested_structures.py` - Nested dicts and lists
- `atomic_operations.py` - Crash-safe operations
- `performance.py` - Benchmarking and optimization
