# loom

**Persistent Python data structures that feel native.**

loom is a file-backed database library: work with `Dict`, `List`, `Queue`,
`Set`, `BTree`, `PriorityQueue`, `Graph`, **vector indexes**, **full-text
search**, and high-level **`Collection`s** exactly like their in-memory
counterparts — but stored on disk with mmap zero-copy access, crash-safe
writes, and automatic space reclamation. No server, no ORM, no SQL.

```python
from pydantic import BaseModel, Field
from loom import DB

class User(BaseModel):
    id:       int
    username: str = Field(max_length=50)
    bio:      str
    score:    float

with DB("app.db") as db:
    users = db.create_dataset("users", User)
    dct   = db.create_dict("users_by_name", users)
    dct["alice"] = {"id": 1, "username": "alice", "bio": "Hello!", "score": 9.5}
    print(dct["alice"])
```

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials (complete use cases)

tutorials/social_posts
tutorials/full_text_search
tutorials/knowledge_graph
tutorials/triple_store
tutorials/semantic_search
tutorials/vector_collection
tutorials/task_scheduling
tutorials/timeseries
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
