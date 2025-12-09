"""
Phase 2 Demo: Prefix-Based Type System

Demonstrates:
- Multiple datasets in one file
- Type safety with prefixes
- Soft deletes
- Clean pythonic API
"""

import os
import tempfile
from loom.fileio import ByteFileDB
from loom.dataset import Dataset


def demo_multiple_datasets():
    """Demo: Multiple datasets in one file."""
    print("\n" + "=" * 70)
    print("DEMO 1: Multiple Datasets in One File")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        db = ByteFileDB(filename)
        db.open()

        # Create three different datasets
        users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")
        posts = Dataset("posts", db, 2, id="uint64", title="U100", likes="uint32")
        comments = Dataset("comments", db, 3, id="uint64", text="U200", author="U50")

        print("1. Created 3 datasets:")
        print(f"   - {users}")
        print(f"   - {posts}")
        print(f"   - {comments}")

        # Write data to each
        print("\n2. Writing data to each dataset...")

        user_addr = users.allocate_block(1)
        users.write(user_addr, id=1, name="Alice", age=30)
        print(f"   User written at address {user_addr}")

        post_addr = posts.allocate_block(1)
        posts.write(post_addr, id=100, title="Hello Loom!", likes=42)
        print(f"   Post written at address {post_addr}")

        comment_addr = comments.allocate_block(1)
        comments.write(comment_addr, id=200, text="Great post!", author="Bob")
        print(f"   Comment written at address {comment_addr}")

        # Read back
        print("\n3. Reading data back:")
        user = users.read(user_addr)
        print(f"   User: {user}")

        post = posts.read(post_addr)
        print(f"   Post: {post}")

        comment = comments.read(comment_addr)
        print(f"   Comment: {comment}")

        # Show file stats
        print(f"\n4. File statistics:")
        print(f"   File size: {db.get_file_size()} bytes")
        print(f"   Used space: {db.get_used_space()} bytes")
        print(f"   Free space: {db.get_free_space()} bytes")

        db.close()
        print("\n✓ Multiple datasets working perfectly!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_type_safety():
    """Demo: Type safety between datasets."""
    print("\n" + "=" * 70)
    print("DEMO 2: Type Safety with Prefixes")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        db = ByteFileDB(filename)
        db.open()

        users = Dataset("users", db, 1, id="uint64", name="U50")
        posts = Dataset("posts", db, 2, id="uint64", title="U100")

        # Write user
        addr = users.allocate_block(1)
        users.write(addr, id=1, name="Alice")
        print(f"1. Wrote user at address {addr}")

        # Read with correct dataset
        user = users.read(addr)
        print(f"   Reading with 'users' dataset: {user} ✓")

        # Try to read with wrong dataset
        print(f"\n2. Trying to read with wrong dataset...")
        try:
            posts.read(addr)
            print("   ERROR: Should have failed!")
        except ValueError as e:
            print(f"   Caught error (expected): {e} ✓")

        db.close()
        print("\n✓ Type safety working!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_soft_deletes():
    """Demo: Soft delete functionality."""
    print("\n" + "=" * 70)
    print("DEMO 3: Soft Deletes")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        db = ByteFileDB(filename)
        db.open()

        users = Dataset("users", db, 1, id="uint64", name="U50", age="int32")

        # Write some users
        print("1. Writing 3 users...")
        addresses = []
        for i, (name, age) in enumerate(
            [("Alice", 30), ("Bob", 25), ("Charlie", 35)], 1
        ):
            addr = users.allocate_block(1)
            users.write(addr, id=i, name=name, age=age)
            addresses.append(addr)
            print(f"   User {i}: {name}, age {age} at {addr}")

        # Check existence
        print("\n2. Checking existence...")
        for addr in addresses:
            print(
                f"   Address {addr}: exists={users.exists(addr)}, deleted={users.is_deleted(addr)}"
            )

        # Delete middle user
        print(f"\n3. Deleting user at {addresses[1]}...")
        users.delete(addresses[1])

        # Check again
        print("\n4. After deletion:")
        for i, addr in enumerate(addresses, 1):
            exists = users.exists(addr)
            deleted = users.is_deleted(addr)
            status = "DELETED" if deleted else "ACTIVE"
            print(f"   User {i} at {addr}: {status}")

        # Try to read deleted record
        print("\n5. Trying to read deleted record...")
        try:
            users.read(addresses[1])
            print("   ERROR: Should have failed!")
        except ValueError as e:
            print(f"   Caught error (expected): {e} ✓")

        # Can still read non-deleted records
        print("\n6. Reading non-deleted records:")
        user1 = users.read(addresses[0])
        user3 = users.read(addresses[2])
        print(f"   User 1: {user1}")
        print(f"   User 3: {user3}")

        db.close()
        print("\n✓ Soft deletes working!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_field_operations():
    """Demo: Field-level read/write operations."""
    print("\n" + "=" * 70)
    print("DEMO 4: Field-Level Operations")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        db = ByteFileDB(filename)
        db.open()

        users = Dataset(
            "users", db, 1, id="uint64", name="U50", age="int32", score="float32"
        )

        # Write a record
        addr = users.allocate_block(1)
        users.write(addr, id=1, name="Alice", age=30, score=95.5)
        print("1. Wrote user:")
        print(f"   {users.read(addr)}")

        # Read individual fields
        print("\n2. Reading individual fields:")
        print(f"   name: {users.read_field(addr, 'name')}")
        print(f"   age: {users.read_field(addr, 'age')}")
        print(f"   score: {users.read_field(addr, 'score')}")

        # Update single field
        print("\n3. Updating age field only...")
        users.write_field(addr, "age", 31)
        print(f"   New age: {users.read_field(addr, 'age')}")
        print(f"   Full record: {users.read(addr)}")

        # Update another field
        print("\n4. Updating score field...")
        users.write_field(addr, "score", 98.7)
        print(f"   New score: {users.read_field(addr, 'score')}")
        print(f"   Full record: {users.read(addr)}")

        db.close()
        print("\n✓ Field operations working!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_clean_api():
    """Demo: Clean, pythonic API."""
    print("\n" + "=" * 70)
    print("DEMO 5: Clean Pythonic API")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        db = ByteFileDB(filename)
        db.open()

        # Simple dataset creation
        print("1. Simple dataset creation:")
        users = Dataset("users", db, 1, id="uint64", name="U50", email="U100")
        print(f"   {users}")

        # Allocate and write in one flow
        print("\n2. Allocate and write:")
        addr = users.allocate_block(3)  # Space for 3 records

        # Write multiple records
        for i, (name, email) in enumerate(
            [
                ("Alice", "alice@example.com"),
                ("Bob", "bob@example.com"),
                ("Charlie", "charlie@example.com"),
            ],
            1,
        ):
            offset = addr + ((i - 1) * users.record_size)
            users.write(offset, id=i, name=name, email=email)

        print(f"   Wrote 3 users starting at {addr}")

        # Read them back
        print("\n3. Reading back:")
        for i in range(3):
            offset = addr + (i * users.record_size)
            user = users.read(offset)
            print(f"   {user}")

        # Simple field access
        print("\n4. Quick field access:")
        first_user_addr = addr
        name = users.read_field(first_user_addr, "name")
        email = users.read_field(first_user_addr, "email")
        print(f"   First user: {name} ({email})")

        db.close()
        print("\n✓ API is clean and pythonic!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("         PHASE 2: Prefix-Based Type System Demo")
    print("=" * 70)

    demo_multiple_datasets()
    demo_type_safety()
    demo_soft_deletes()
    demo_field_operations()
    demo_clean_api()

    print("\n" + "=" * 70)
    print("All demos completed successfully! 🎉")
    print("=" * 70)
    print("\nPhase 2 adds:")
    print("  ✓ Multiple datasets in one file")
    print("  ✓ Type safety with prefix identifiers")
    print("  ✓ Soft deletes (flip prefix sign)")
    print("  ✓ Clean pythonic API")
    print("  ✓ Field-level operations")
    print("\nReady for Phase 3: Database orchestrator!")
    print("=" * 70 + "\n")
