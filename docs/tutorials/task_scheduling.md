# Task scheduling — Queue & PriorityQueue

Durable work queues that survive restarts.

## FIFO — `Queue`

`push`/`pop`/`peek` are O(1); exhausted blocks are returned to the freelist, so
a steady push≈pop rate keeps the file from growing.

```python
from pydantic import BaseModel, Field
from loom import DB

class Job(BaseModel):
    id:   int
    task: str = Field(max_length=40)

db = DB("jobs.db")
q = db.create_queue("jobs", Job)

q.push({"id": 1, "task": "send email"})
q.push_many([{"id": 2, "task": "resize image"}, {"id": 3, "task": "reindex"}])

q.peek()      # look without consuming
job = q.pop() # FIFO
len(q)
```

A worker loop:

```python
while len(q):
    job = q.pop()
    handle(job)        # crash-safe: re-open and the un-popped jobs are still there
```

## Priority — `PriorityQueue`

Highest priority pops first (set `max_first=False` for lowest-first); equal
priorities pop FIFO. Priorities can be int, float, **or datetime** (a deadline).

```python
pq = db.create_priority_queue("work", {"task": "utf8[40]"})

pq.push({"task": "send email"}, priority=5)
pq.push({"task": "incident"},   priority=10)
pq.push_many([({"task": "warm cache"}, 7.5), ({"task": "cleanup"}, 1)])

pq.peek()     # → {"task": "incident"}   (priority 10, not removed)
pq.pop()      # → {"task": "incident"}
```

### Deadline scheduler (datetime priorities, earliest first)

```python
from datetime import datetime

deadlines = db.create_priority_queue("deadlines", {"job": "utf8[40]"},
                                    max_first=False)        # earliest deadline first
deadlines.push({"job": "renew cert"},  datetime(2026, 6, 1))
deadlines.push({"job": "send report"}, datetime(2026, 1, 15))

while len(deadlines):
    nxt = deadlines.pop()        # always the soonest deadline
    run(nxt)
```

Backed by a B-tree (`O(log n)` push/pop), so it scales to large backlogs and is
fully persistent.
