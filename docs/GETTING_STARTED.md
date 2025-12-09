# Getting Started with Loom

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd loom

# Install dependencies (if any)
pip install -r requirements.txt
```

## Quick Start

### 1. Create a Database

```python
from loom.database import DB

# Create/open database
with DB("mydata.db") as db:
    print("Database created!")
```

### 2. Create a Dataset

Datasets define the schema for your data:

```python
with DB("mydata.db") as db:
    # Define schema: field_name="dtype"
    users = db.create_dataset(
        "users",
        id="uint64",
        name="U50",      # Unicode string, max 50 chars
        email="U100",
        age="int32",
        score="float32"
    )
```

### 3. Create a Dict

```python
with DB("mydata.db") as db:
    user_ds = db.create_dataset("users", id="uint64", name="U50", email="U100")
    
    # Create persistent dictionary
    users = db.create_dict("users_dict", user_ds)
    
    # Insert data
    users["alice"] = {"id": 1, "name": "Alice", "email": "alice@example.com"}
    users["bob"] = {"id": 2, "name": "Bob", "email": "bob@example.com"}
    
    # Read data
    print(users["alice"])  # {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}
    
    # Check existence
    if "alice" in users:
        print("Alice exists!")
    
    # Iterate
    for key, value in users.items():
        print(f"{key}: {value['name']}")
```

### 4. Create a List

```python
with DB("mydata.db") as db:
    log_ds = db.create_dataset("logs", timestamp="uint64", message="U200")
    
    # Create persistent list
    logs = db.create_list("logs_list", log_ds)
    
    # Append data
    logs.append({"timestamp": 1234567890, "message": "Server started"})
    logs.append({"timestamp": 1234567891, "message": "User logged in"})
    
    # Read by index
    print(logs[0])  # {'timestamp': 1234567890, 'message': 'Server started'}
    
    # Iterate
    for log in logs:
        print(log["message"])
    
    # Slice
    recent = logs[-10:]  # Last 10 items
```

## Core Concepts

### Persistence

All data is automatically persisted to disk. No need to manually save:

```python
with DB("mydata.db") as db:
    users = db.create_dict("users", user_ds)
    users["alice"] = {"id": 1, "name": "Alice"}
    # Data is automatically saved!

# Reopen database
with DB("mydata.db") as db:
    from loom.datastructures.dict import Dict
    users = Dict("users_dict", db, None)
    print(users["alice"])  # Data persists!
```

### Atomic Operations

Use atomic operations for crash safety:

```python
with DB("mydata.db") as db:
    users = db.create_dict("users", user_ds)
    
    # Fast path (default) - no crash safety
    users["alice"] = {"id": 1, "name": "Alice"}
    
    # Atomic path - crash-safe via WAL
    users.set("bob", {"id": 2, "name": "Bob"}, atomic=True)
```

### Nested Structures

Create dictionaries of dictionaries or lists of lists:

```python
with DB("mydata.db") as db:
    # Define inner schema
    user_ds = db.create_dataset("users", id="uint32", name="U50")
    
    # Create template for nested dicts
    from loom.datastructures.dict import Dict
    UserDict = Dict.template(user_ds)
    
    # Create parent dict
    teams = db.create_dict("teams", UserDict)
    
    # Access nested dict (auto-created)
    eng_team = teams["engineering"]
    eng_team["alice"] = {"id": 1, "name": "Alice"}
    eng_team["bob"] = {"id": 2, "name": "Bob"}
    
    # Access nested values
    print(teams["engineering"]["alice"]["name"])  # "Alice"
```

## Common Patterns

### User Management System

```python
from loom.database import DB

with DB("users.db") as db:
    # Create schema
    user_ds = db.create_dataset(
        "users",
        id="uint64",
        username="U50",
        email="U100",
        created_at="uint64",
        is_active="bool"
    )
    
    # Create dict with bloom filter for fast lookups
    users = db.create_dict("users", user_ds, bloom_size=100000)
    
    # Add users
    users["alice"] = {
        "id": 1,
        "username": "alice",
        "email": "alice@example.com",
        "created_at": 1234567890,
        "is_active": True
    }
    
    # Fast membership check
    if "alice" in users:
        user = users["alice"]
        print(f"Welcome {user['username']}!")
```

### Event Log

```python
with DB("events.db") as db:
    # Create schema
    event_ds = db.create_dataset(
        "events",
        timestamp="uint64",
        event_type="U50",
        user_id="uint64",
        data="U500"
    )
    
    # Create list
    events = db.create_list("events", event_ds)
    
    # Append events atomically (crash-safe)
    events.append({
        "timestamp": 1234567890,
        "event_type": "login",
        "user_id": 1,
        "data": "User logged in from 192.168.1.1"
    }, atomic=True)
    
    # Query recent events
    recent = events[-100:]  # Last 100 events
    for event in recent:
        print(f"{event['timestamp']}: {event['event_type']}")
```

### Cache System

```python
with DB("cache.db") as db:
    # Create schema
    cache_ds = db.create_dataset(
        "cache",
        key="U100",
        value="U1000",
        expires_at="uint64"
    )
    
    # Create dict with large cache
    cache = db.create_dict("cache", cache_ds, cache_size=10000)
    
    # Fast writes (no atomic overhead for cache)
    cache["user:1:profile"] = {
        "key": "user:1:profile",
        "value": '{"name": "Alice", "age": 30}',
        "expires_at": 1234567890
    }
    
    # Fast reads (from LRU cache)
    if "user:1:profile" in cache:
        data = cache["user:1:profile"]
```

### Hierarchical Data

```python
with DB("hierarchy.db") as db:
    # Define schemas
    item_ds = db.create_dataset("items", id="uint32", name="U50", value="float32")
    
    # Create nested structure
    from loom.datastructures.list import List
    ItemList = List.template(item_ds)
    
    # Parent list
    categories = db.create_list("categories", ItemList)
    
    # Add categories with items
    electronics = categories.append()
    electronics.append({"id": 1, "name": "Laptop", "value": 999.99})
    electronics.append({"id": 2, "name": "Phone", "value": 699.99})
    
    books = categories.append()
    books.append({"id": 3, "name": "Python Guide", "value": 49.99})
    books.append({"id": 4, "name": "Data Structures", "value": 59.99})
    
    # Access nested data
    print(categories[0][0]["name"])  # "Laptop"
    print(categories[1][1]["value"])  # 59.99
```

## Performance Tips

### 1. Use Bloom Filters for Large Dicts

```python
# Without bloom: slower membership checks
users = db.create_dict("users", user_ds)

# With bloom: fast negative lookups
users = db.create_dict("users", user_ds, bloom_size=100000)
```

### 2. Adjust Cache Sizes

```python
# Hot data: large cache
active = db.create_dict("active", ds, cache_size=10000)

# Cold data: small cache
archived = db.create_dict("archived", ds, cache_size=10)
```

### 3. Batch Atomic Operations

```python
# Slow: many individual atomic operations
for item in items:
    logs.append(item, atomic=True)

# Fast: single atomic batch
logs.append_many(items, atomic=True)
```

### 4. Use Fast Path for Non-Critical Data

```python
# Critical data: use atomic
transactions.__setitem__(tx_id, tx_data, atomic=True)

# Cache/temp data: use fast path
cache[key] = value  # Default, no atomic overhead
```

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed documentation
- Check out [examples/](../examples/) for more code samples
- See [PLAN.md](../PLAN.md) for roadmap and known issues

## Troubleshooting

### Database won't open

```python
# Make sure to use context manager or call open()
with DB("mydata.db") as db:  # Automatically opens
    ...

# Or manually
db = DB("mydata.db")
db.open()
# ... use db
db.close()
```

### KeyError when accessing data

```python
# Check if key exists first
if "alice" in users:
    user = users["alice"]
else:
    print("User not found")
```

### Data not persisting

```python
# Data persists automatically, but make sure to close properly
with DB("mydata.db") as db:
    users["alice"] = {"id": 1, "name": "Alice"}
    # Automatically closed and saved

# Or manually
db = DB("mydata.db")
db.open()
users["alice"] = {"id": 1, "name": "Alice"}
db.close()  # Important!
```

### Performance issues

```python
# 1. Increase cache size
users = db.create_dict("users", user_ds, cache_size=10000)

# 2. Use bloom filters
users = db.create_dict("users", user_ds, bloom_size=100000)

# 3. Batch operations
items.append_many(items_list, atomic=True)  # vs many individual appends

# 4. Use fast path for non-critical data
cache[key] = value  # Don't use atomic=True for cache
```
