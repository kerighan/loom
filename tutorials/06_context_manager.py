# Context Manager
# Automatic cleanup with 'with' statement

from loom.database import DB
import os

# Recommended: use context manager
with DB("context_example.loom") as db:
    users = db.create_list("users", {"name": "U50"})
    users.append({"name": "Alice"})
    # db.close() called automatically, even if exception occurs

# Data persists
with DB("context_example.loom") as db:
    users = db.create_list("users", {"name": "U50"})
    print(users[0])  # {'name': 'Alice'}

os.remove("context_example.loom")
