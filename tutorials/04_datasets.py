# Datasets
# Low-level typed records with direct address access

from loom.database import DB
import os

# Clean up any existing database
try:
    if os.path.exists("datasets_example.loom"):
        os.remove("datasets_example.loom")
except PermissionError:
    pass

db = DB("datasets_example.loom")

# Create a dataset (typed record storage)
users = db.create_dataset("users", 
    id="uint64", 
    name="U50", 
    email="U100",
    score="float64"
)

# Allocate space for records
addr1 = users.allocate_block(1)  # Space for 1 record
addr2 = users.allocate_block(1)

# Write records (dict-like)
users[addr1] = {"id": 1, "name": "Alice", "email": "alice@example.com", "score": 95.5}
users[addr2] = {"id": 2, "name": "Bob", "email": "bob@example.com", "score": 87.0}

# Read records
print(users[addr1])  # {'id': 1, 'name': 'Alice', ...}

# Update single field (efficient - only writes that field)
users.write_field(addr1, "score", 98.0)
print(users.read_field(addr1, "score"))  # 98.0

# Delete (soft delete - marks as deleted)
users.delete(addr2)
print(users.exists(addr2))     # True (address exists)
print(users.is_deleted(addr2)) # True (but marked deleted)

# Datasets are best for:
# - Fixed records you access by address
# - Building higher-level structures
# - When you need field-level updates

# For most use cases, prefer List (see 02_lists.py)

db.close()

# Cleanup
try:
    os.remove("datasets_example.loom")
except PermissionError:
    pass  # File still in use on Windows
