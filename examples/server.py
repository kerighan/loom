"""Serve a loom database over HTTP.

Run:
    pip install 'fastapi[standard]'
    python examples/server.py

Optional dashboard:
    db.serve(host="127.0.0.1", port=8000, dashboard=True)

Then open:
    http://localhost:8000/docs        # Swagger UI (interactive)
    http://localhost:8000/redoc       # ReDoc
    http://localhost:8000/dashboard   # Integrated dashboard

Try it out (in another shell):
    curl localhost:8000/

    # Push a job onto the queue
    curl -X POST localhost:8000/queues/jobs/push \
         -H 'content-type: application/json' \
         -d '{"id": 1, "task": "send-email", "priority": 0.9}'

    # Pop it back
    curl -X POST localhost:8000/queues/jobs/pop

    # Set a BTree value
    curl -X PUT localhost:8000/btrees/catalog/items/iPhone \
         -H 'content-type: application/json' \
         -d '{"sku": "P002", "price": 999.0, "stock": 42}'

    # Range query on the BTree
    curl 'localhost:8000/btrees/catalog/range?start=A&end=Z'

    # Probabilistic membership
    curl localhost:8000/bloomfilters/seen/contains/alice

Single-writer / single-reader: every request is serialized through a
process-wide lock, so the underlying mmap stays consistent.
"""

import os
import sys
import tempfile

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
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
API_TOKEN = "secret"
fresh = not os.path.exists(DB_PATH)

try:
    db = DB(DB_PATH)
except Exception:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    fresh = True
    db = DB(DB_PATH)

if fresh:
    # Datasets
    users_ds = db.create_dataset("users", User)
    prods_ds = db.create_dataset("products", Product)

    # Datastructures
    db.create_dict("users_by_username", users_ds)
    db.create_list("audit_log", users_ds)
    db.create_queue("jobs", Job)
    db.create_btree("catalog", prods_ds, key_size=20)
    db.create_bloomfilter("seen", expected_items=10_000)

    # Seed a couple of records so the demo is non-empty
    users_ds.insert({"id": 1, "username": "alice", "age": 30, "premium": True})
    db["users_by_username"]["alice"] = {
        "id": 1,
        "username": "alice",
        "age": 30,
        "premium": True,
    }
    db["users_by_username"]["bob"] = {
        "id": 2,
        "username": "bob",
        "age": 41,
        "premium": False,
    }
    db["catalog"]["MacBook Pro"] = {
        "sku": "P001",
        "price": 2499.0,
        "stock": 12,
    }
    db["jobs"].push({"id": 1, "task": "boot", "priority": 1.0})
    db["seen"].add("alice")

print(f"Loom DB: {DB_PATH}")
print("Open http://localhost:8000/docs for the interactive API")
print("Open http://localhost:8000/dashboard for the integrated dashboard")
print(f"Dashboard/API token: {API_TOKEN}")
print("Press Ctrl-C to stop.\n")

# Blocking call — runs uvicorn under the hood
db.serve(host="127.0.0.1", port=8000, dashboard=True, auth_token=API_TOKEN)
