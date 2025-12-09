# Atomic Writes
# Crash-safe operations using write-ahead logging

from loom.database import DB
import os

# Clean up any existing database
try:
    if os.path.exists("atomic_example.loom"):
        os.remove("atomic_example.loom")
except PermissionError:
    pass

with DB("atomic_example.loom") as db:
    logs = db.create_list("logs", {"event": "U100", "timestamp": "uint64"})
    
    # --- Default: Fast writes (no WAL) ---
    # Best for: bulk loading, non-critical data
    logs.append({"event": "user_login", "timestamp": 1000})
    
    # --- Atomic: Safe writes (uses WAL) ---
    # Best for: critical data, financial records
    logs.append({"event": "payment_processed", "timestamp": 1001}, atomic=True)
    
    # --- Batch atomic writes ---
    # All items written in single transaction
    events = [
        {"event": f"event_{i}", "timestamp": 2000 + i}
        for i in range(100)
    ]
    logs.append_many(events, atomic=True)
    
    # If crash occurs during atomic write:
    # - Either ALL items are written
    # - Or NONE are written
    # Never partial writes

# Cleanup
try:
    os.remove("atomic_example.loom")
except PermissionError:
    pass  # File still in use on Windows
