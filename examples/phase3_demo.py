"""
Phase 3 Demo: Database Orchestrator

Demonstrates:
- Automatic schema persistence
- Dataset registry management
- Clean high-level API
- Context manager support
- Multi-session workflows
"""

import os
import tempfile
from loom.database import DB


def demo_basic_usage():
    """Demo: Basic database usage with schema persistence."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage with Schema Persistence")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        # Session 1: Create database and datasets
        print("SESSION 1: Creating database...")
        with DB(filename) as db:
            # Create datasets - schema stored automatically!
            users = db.create_dataset("users", id="uint64", name="U50", age="int32")
            posts = db.create_dataset(
                "posts", id="uint64", title="U100", likes="uint32"
            )

            print(f"Created: {users}")
            print(f"Created: {posts}")

            # Add some data
            user_addr = users.allocate_block(1)
            users[user_addr] = {"id": 1, "name": "Alice", "age": 30}

            post_addr = posts.allocate_block(1)
            posts[post_addr] = {"id": 100, "title": "Hello Loom!", "likes": 42}

            print(f"\nWrote user at {user_addr}")
            print(f"Wrote post at {post_addr}")

        # Session 2: Reopen - schema loaded automatically!
        print("\nSESSION 2: Reopening database...")
        with DB(filename) as db:
            print(f"Available datasets: {db.list_datasets()}")

            # Get datasets - NO need to specify schema!
            users = db["users"]
            posts = db["posts"]

            print(f"\nLoaded: {users}")
            print(f"Loaded: {posts}")

            # Read data
            user = users[user_addr]
            post = posts[post_addr]

            print(f"\nUser: {user}")
            print(f"Post: {post}")

        print("\n✓ Schema persistence working perfectly!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_multi_session_workflow():
    """Demo: Realistic multi-session workflow."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Session Workflow")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        # Day 1: Initial setup
        print("DAY 1: Setting up database...")
        with DB(filename) as db:
            users = db.create_dataset("users", id="uint64", name="U50", email="U100")

            # Add initial users
            addr = users.allocate_block(3)
            users[addr] = {"id": 1, "name": "Alice", "email": "alice@example.com"}
            users[addr + users.record_size] = {
                "id": 2,
                "name": "Bob",
                "email": "bob@example.com",
            }
            users[addr + 2 * users.record_size] = {
                "id": 3,
                "name": "Charlie",
                "email": "charlie@example.com",
            }

            print(f"Added 3 users starting at {addr}")

        # Day 2: Different file, continue work
        print("\nDAY 2: Continuing work (different file)...")
        with DB(filename) as db:
            # Just load it - schema remembered!
            users = db["users"]

            # Read existing data
            print("\nExisting users:")
            for i in range(3):
                user = users[addr + i * users.record_size]
                print(f"  {user}")

            # Add more data
            new_addr = users.allocate_block(1)
            users[new_addr] = {"id": 4, "name": "Diana", "email": "diana@example.com"}
            print(f"\nAdded new user at {new_addr}")

        # Day 3: Add another dataset
        print("\nDAY 3: Adding another dataset...")
        with DB(filename) as db:
            # Create new dataset
            posts = db.create_dataset(
                "posts", id="uint64", title="U100", author_id="uint64"
            )

            # Add posts
            post_addr = posts.allocate_block(2)
            posts[post_addr] = {"id": 100, "title": "First Post", "author_id": 1}
            posts[post_addr + posts.record_size] = {
                "id": 101,
                "title": "Second Post",
                "author_id": 2,
            }

            print(f"Created posts dataset")
            print(f"Available datasets: {db.list_datasets()}")

        # Day 4: Use both datasets
        print("\nDAY 4: Using multiple datasets...")
        with DB(filename) as db:
            users = db["users"]
            posts = db["posts"]

            # Read and display
            print("\nAll users:")
            for i in range(4):
                try:
                    user_addr = addr + i * users.record_size if i < 3 else new_addr
                    user = users[user_addr]
                    print(f"  {user['name']}: {user['email']}")
                except:
                    pass

            print("\nAll posts:")
            for i in range(2):
                post = posts[post_addr + i * posts.record_size]
                print(f"  {post['title']} (by user {post['author_id']})")

        print("\n✓ Multi-session workflow working!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_dict_like_access():
    """Demo: Dict-like access to datasets."""
    print("\n" + "=" * 70)
    print("DEMO 3: Dict-Like Access")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        with DB(filename) as db:
            # Create datasets
            db.create_dataset("users", id="uint64", name="U50")
            db.create_dataset("posts", id="uint64", title="U100")
            db.create_dataset("comments", id="uint64", text="U200")

            print("1. Check existence with 'in':")
            print(f"   'users' in db: {'users' in db}")
            print(f"   'products' in db: {'products' in db}")

            print("\n2. Access with []:")
            users = db["users"]
            print(f"   db['users']: {users}")

            print("\n3. List all datasets:")
            print(f"   db.list_datasets(): {db.list_datasets()}")

            print("\n4. Get all datasets dict:")
            datasets = db.datasets
            for name, dataset in datasets.items():
                print(f"   {name}: {dataset}")

        print("\n✓ Dict-like access working!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_context_manager():
    """Demo: Context manager for automatic resource management."""
    print("\n" + "=" * 70)
    print("DEMO 4: Context Manager")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        print("1. Using context manager (recommended):")
        with DB(filename) as db:
            print(f"   Database: {db}")
            users = db.create_dataset("users", id="uint64", name="U50")
            addr = users.allocate_block(1)
            users[addr] = {"id": 1, "name": "Alice"}
        # Auto-closed here
        print("   ✓ Database auto-closed")

        print("\n2. Manual open/close (also works):")
        db = DB(filename)
        db.open()
        print(f"   Database: {db}")
        users = db["users"]
        print(f"   User: {users[addr]}")
        db.close()
        print("   ✓ Database manually closed")

        print("\n✓ Both methods work!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def demo_real_world_example():
    """Demo: Real-world blog application."""
    print("\n" + "=" * 70)
    print("DEMO 5: Real-World Blog Application")
    print("=" * 70 + "\n")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    try:
        # Setup: Create blog database
        print("SETUP: Creating blog database...")
        with DB(filename) as db:
            users = db.create_dataset(
                "users", id="uint64", username="U50", email="U100", created_at="uint64"
            )

            posts = db.create_dataset(
                "posts",
                id="uint64",
                title="U100",
                author_id="uint64",
                views="uint32",
                created_at="uint64",
            )

            comments = db.create_dataset(
                "comments",
                id="uint64",
                post_id="uint64",
                author_id="uint64",
                text="U500",
                created_at="uint64",
            )

            print(f"Created 3 datasets: {db.list_datasets()}")

        # Day 1: Add users and posts
        print("\nDAY 1: Adding users and posts...")
        with DB(filename) as db:
            users = db["users"]
            posts = db["posts"]

            # Add users
            user1_addr = users.allocate_block(1)
            users[user1_addr] = {
                "id": 1,
                "username": "alice",
                "email": "alice@blog.com",
                "created_at": 1000,
            }

            user2_addr = users.allocate_block(1)
            users[user2_addr] = {
                "id": 2,
                "username": "bob",
                "email": "bob@blog.com",
                "created_at": 1001,
            }

            # Add posts
            post1_addr = posts.allocate_block(1)
            posts[post1_addr] = {
                "id": 100,
                "title": "Welcome to my blog!",
                "author_id": 1,
                "views": 0,
                "created_at": 1100,
            }

            post2_addr = posts.allocate_block(1)
            posts[post2_addr] = {
                "id": 101,
                "title": "Python Tips",
                "author_id": 2,
                "views": 0,
                "created_at": 1101,
            }

            print(f"Added 2 users and 2 posts")

        # Day 2: Add comments and update views
        print("\nDAY 2: Adding comments and updating views...")
        with DB(filename) as db:
            posts = db["posts"]
            comments = db["comments"]

            # Add comments
            comment1_addr = comments.allocate_block(1)
            comments[comment1_addr] = {
                "id": 1000,
                "post_id": 100,
                "author_id": 2,
                "text": "Great post!",
                "created_at": 1200,
            }

            comment2_addr = comments.allocate_block(1)
            comments[comment2_addr] = {
                "id": 1001,
                "post_id": 101,
                "author_id": 1,
                "text": "Very helpful!",
                "created_at": 1201,
            }

            # Update post views
            posts.write_field(post1_addr, "views", 42)
            posts.write_field(post2_addr, "views", 73)

            print(f"Added 2 comments and updated views")

        # Day 3: Generate report
        print("\nDAY 3: Generating report...")
        with DB(filename) as db:
            users = db["users"]
            posts = db["posts"]
            comments = db["comments"]

            print("\n📊 Blog Statistics:")
            print(f"   Datasets: {len(db.list_datasets())}")

            # Show posts with views
            print("\n📝 Posts:")
            post1 = posts[post1_addr]
            post2 = posts[post2_addr]
            print(f"   '{post1['title']}' - {post1['views']} views")
            print(f"   '{post2['title']}' - {post2['views']} views")

            # Show comments
            print("\n💬 Comments:")
            comment1 = comments[comment1_addr]
            comment2 = comments[comment2_addr]
            print(f"   On post {comment1['post_id']}: '{comment1['text']}'")
            print(f"   On post {comment2['post_id']}: '{comment2['text']}'")

        print("\n✓ Blog application working perfectly!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("         PHASE 3: Database Orchestrator Demo")
    print("=" * 70)

    demo_basic_usage()
    demo_multi_session_workflow()
    demo_dict_like_access()
    demo_context_manager()
    demo_real_world_example()

    print("\n" + "=" * 70)
    print("All demos completed successfully! 🎉")
    print("=" * 70)
    print("\nPhase 3 adds:")
    print("  ✓ Automatic schema persistence")
    print("  ✓ Dataset registry management")
    print("  ✓ High-level pythonic API")
    print("  ✓ Context manager support")
    print("  ✓ Multi-session workflows")
    print("\nNow you can:")
    print("  • Create datasets once, use them forever")
    print("  • No need to remember schemas")
    print("  • Work across multiple files/sessions")
    print("  • Clean, minimal code")
    print("\nReady for Phase 4: Dynamic data structures!")
    print("=" * 70 + "\n")
