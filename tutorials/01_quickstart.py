# Loom Quickstart
# A persistent database that feels like Python

from loom.database import DB

# Open a database (creates file if needed)
db = DB("my_data.loom")

# Create a list with a schema
users = db.create_list("users", {"name": "U50", "age": "uint8"})

# Append items (just like Python lists)
users.append({"name": "Alice", "age": 30})
users.append({"name": "Bob", "age": 25})

# Access by index
print(users[0])  # {'name': 'Alice', 'age': 30}
print(users[-1]) # {'name': 'Bob', 'age': 25}

# Slice (returns list of dicts)
print(users[0:2])  # [{'name': 'Alice', ...}, {'name': 'Bob', ...}]

# Iterate
for user in users:
    print(user["name"])

# Length
print(len(users))  # 2

# Close when done
db.close()

# Data persists! Reopen later:
db = DB("my_data.loom")
users = db["users"]  # Auto-loaded from registry
print(users[0])  # Still there: {'name': 'Alice', 'age': 30}
db.close()

# Cleanup for this example
import os; os.remove("my_data.loom")
