# Working with Lists
# Persistent lists with typed schemas

from loom.database import DB
import os

# Clean up any existing database
try:
    if os.path.exists("lists_example.loom"):
        os.remove("lists_example.loom")
except PermissionError:
    pass

db = DB("lists_example.loom")

# --- Schema Types ---
# Integers: uint8, uint16, uint32, uint64, int8, int16, int32, int64
# Floats: float32, float64
# Strings: U10, U50, U100, U200 (fixed-length, number = max chars)
# Booleans: bool

products = db.create_list("products", {
    "id": "uint64",
    "name": "U100",
    "price": "float64",
    "in_stock": "bool"
})

# --- Appending ---
products.append({"id": 1, "name": "Laptop", "price": 999.99, "in_stock": True})
products.append({"id": 2, "name": "Mouse", "price": 29.99, "in_stock": True})
products.append({"id": 3, "name": "Keyboard", "price": 79.99, "in_stock": False})

# --- Indexing ---
print(products[0])   # First item
print(products[-1])  # Last item

# --- Slicing (returns list of dicts) ---
first_two = products[0:2]
last_two = products[-2:]
every_other = products[::2]

# --- Modify in place ---
products[0] = {"id": 1, "name": "Laptop Pro", "price": 1299.99, "in_stock": True}

# --- Delete ---
del products[1]  # Soft delete (marks as deleted)
print(len(products))  # 2 (only counts valid items)

# --- Batch append (faster for many items) ---
items = [{"id": i, "name": f"Item {i}", "price": float(i), "in_stock": True} 
         for i in range(100)]
products.append_many(items)

# --- Fast array slicing (for analytics) ---
# Returns NumPy array instead of dicts - much faster for large slices
# Note: Only works when there are no deletions
arr = products.slice_array(0, 50)
print(f"Average price of first 50: {arr['price'].mean():.2f}")

# --- Delete ---
del products[1]  # Soft delete (marks as deleted)
print(f"Valid items after deletion: {len(products)}")  # 102 (only counts valid items)

db.close()

# Cleanup
try:
    os.remove("lists_example.loom")
except PermissionError:
    pass  # File still in use on Windows
