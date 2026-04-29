"""Serve a loom database over HTTP.

Run:
    pip install 'fastapi[standard]'
    python examples/server.py

Then open:
    http://localhost:8000/docs        # Swagger UI (interactive)
    http://localhost:8000/redoc       # ReDoc

Try it out (in another shell):
    curl localhost:8000/

    # Insert a record (validated against the User model)
    curl -X POST localhost:8000/datasets/users/records \
         -H 'content-type: application/json' \
         -d '{"id": 1, "username": "alice", "age": 30, "premium": true}'

    # Read it back
    curl localhost:8000/datasets/users/records/0

    # Push a job onto the queue
    curl -X POST localhost:8000/queues/jobs/push \
         -H 'content-type: application/json' \
         -d '{"id": 1, "task": "send-email", "priority": 0.9}'

    # Range query on the BTree
    curl 'localhost:8000/btrees/catalog/range?start=A&end=Z'

    # Probabilistic membership
    curl localhost:8000/bloomfilters/seen/contains/alice

Single-writer / single-reader: every request is serialized through a
process-wide lock, so the underlying mmap stays consistent.
"""

import os
import tempfile

from pydantic import BaseModel, Field

from loom import DB


# ── Schemas (Pydantic — also drive HTTP validation) ────────────────────────

class User(BaseModel):
    id: int
    username: str = Field(max_length=50)
    age: int
    premium: bool


class Job(BaseModel):
    id: int
    task: str = Field(max_length=80)
    priority: float


class Product(BaseModel):
    sku: str = Field(max_length=20)
    price: float
    stock: int


# ── Build the DB ──────────────────────────────────────────────────────────

DB_PATH = os.path.join(tempfile.gettempdir(), "loom_demo_server.db")
fresh = not os.path.exists(DB_PATH)

db = DB(DB_PATH)

if fresh:
    # Datasets
    users_ds = db.create_dataset("users", User)
    prods_ds = db.create_dataset("products", Product)

    # Datastructures
    db.create_list("audit_log", users_ds)
    db.create_queue("jobs", Job)
    db.create_btree("catalog", prods_ds, key_size=20)
    db.create_bloomfilter("seen", expected_items=10_000)

    # Seed a couple of records so the demo is non-empty
    users_ds.insert(
        {"id": 1, "username": "alice", "age": 30, "premium": True}
    )
    db["catalog"]["MacBook Pro"] = {
        "sku": "P001", "price": 2499.0, "stock": 12,
    }
    db["jobs"].push({"id": 1, "task": "boot", "priority": 1.0})
    db["seen"].add("alice")

print(f"Loom DB: {DB_PATH}")
print("Open http://localhost:8000/docs for the interactive API")
print("Press Ctrl-C to stop.\n")

# Blocking call — runs uvicorn under the hood
db.serve(host="127.0.0.1", port=8000)
