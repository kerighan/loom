"""Benchmark cross-nested data structures: List of Dicts and Dict of Lists.

Tests performance of:
- List[Dict] - A list where each element is a Dict
- Dict[List] - A dict where each value is a List
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import time
import random
from loom.database import DB
from loom.datastructures import List
from loom.datastructures.dict import Dict


def benchmark_list_of_dicts(num_dicts=100, items_per_dict=100):
    """Benchmark List of Dicts (List[Dict]).

    Args:
        num_dicts: Number of dicts in the list
        items_per_dict: Number of items in each dict
    """
    print(f"\n{'='*60}")
    print(f"List of Dicts Benchmark (List[Dict])")
    print(f"{'='*60}")
    print(f"Creating 1 list containing {num_dicts:,} dicts")
    print(f"Each dict has {items_per_dict:,} items")
    print(f"Total items: {num_dicts * items_per_dict:,}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".loom") as tmp:
        filename = tmp.name

    try:
        # Create database
        db = DB(filename, header_size=1024 * 1024)

        # Create dataset for dict items
        user_dataset = db.create_dataset(
            "users", id="uint64", name="U50", score="float64"
        )

        # Create template for nested dicts
        UserDictTemplate = Dict.template(user_dataset, cache_size=100, use_bloom=False)

        # Create list of dicts
        teams = db.create_list("teams", UserDictTemplate)

        # Benchmark: Create and fill dicts
        print(f"\n--- Creating and filling {num_dicts:,} dicts ---")

        overall_start = time.time()

        for i in range(num_dicts):
            # Create dict by appending to list
            team = teams.append()

            # Add items to dict
            for j in range(items_per_dict):
                user_key = f"user_{j}"
                team[user_key] = {
                    "id": i * items_per_dict + j,
                    "name": f"User {j} of Team {i}",
                    "score": float(i * 100 + j),
                }

            # Update the stored reference
            teams.update_nested_ref(i, team)

            # Progress updates
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - overall_start
                rate = (i + 1) / elapsed
                items_rate = (i + 1) * items_per_dict / elapsed
                print(
                    f"  [{i+1:,}/{num_dicts:,}] {rate:.1f} dicts/sec, {items_rate:,.0f} items/sec, {elapsed:.1f}s elapsed"
                )

        creation_time = time.time() - overall_start
        total_items = num_dicts * items_per_dict

        print(f"\nCreation complete:")
        print(f"  Time: {creation_time:.2f}s")
        print(f"  Dicts/sec: {num_dicts/creation_time:,.1f}")
        print(f"  Items/sec: {total_items/creation_time:,.0f}")

        # Save and get file size
        teams.save(force=True)
        db.close()

        file_size = os.path.getsize(filename)
        print(f"\n--- Storage ---")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"  Per dict: {file_size/num_dicts:.0f} bytes")
        print(f"  Per item: {file_size/total_items:.0f} bytes")

        # Benchmark: Random access
        print(f"\n--- Random Access ---")
        db = DB(filename)
        teams = db._datastructures.get("teams")
        if teams is None:
            user_dataset = db.get_dataset("users")
            UserDictTemplate = Dict.template(
                user_dataset, cache_size=100, use_bloom=False
            )
            teams = db.create_list("teams", UserDictTemplate)

        random.seed(42)
        num_reads = 1000
        start = time.time()

        for _ in range(num_reads):
            dict_idx = random.randint(0, num_dicts - 1)
            item_idx = random.randint(0, items_per_dict - 1)
            user_key = f"user_{item_idx}"
            item = teams[dict_idx][user_key]

        random_access_time = time.time() - start
        print(f"  {num_reads:,} random accesses in {random_access_time:.3f}s")
        print(f"  Access rate: {num_reads/random_access_time:,.0f} items/sec")
        print(f"  Avg latency: {random_access_time/num_reads*1000:.3f}ms")

        # Benchmark: Iteration
        print(f"\n--- Iteration ---")
        start = time.time()

        count = 0
        for team in teams:
            for user_key in team.keys():
                user = team[user_key]
                count += 1

        iteration_time = time.time() - start
        print(f"  Iterated {count:,} items in {iteration_time:.3f}s")
        print(f"  Iteration rate: {count/iteration_time:,.0f} items/sec")

        db.close()

        return {
            "creation_time": creation_time,
            "random_access_time": random_access_time,
            "iteration_time": iteration_time,
            "file_size": file_size,
            "num_dicts": num_dicts,
            "items_per_dict": items_per_dict,
            "total_items": total_items,
        }

    finally:
        try:
            os.unlink(filename)
        except (PermissionError, FileNotFoundError):
            pass


def benchmark_dict_of_lists(num_lists=100, items_per_list=100):
    """Benchmark Dict of Lists (Dict[List]).

    Args:
        num_lists: Number of lists in the dict
        items_per_list: Number of items in each list
    """
    print(f"\n{'='*60}")
    print(f"Dict of Lists Benchmark (Dict[List])")
    print(f"{'='*60}")
    print(f"Creating 1 dict containing {num_lists:,} lists")
    print(f"Each list has {items_per_list:,} items")
    print(f"Total items: {num_lists * items_per_list:,}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".loom") as tmp:
        filename = tmp.name

    try:
        # Create database
        db = DB(filename, header_size=1024 * 1024)

        # Create dataset for list items
        task_dataset = db.create_dataset(
            "tasks", id="uint64", title="U50", priority="uint8"
        )

        # Create template for nested lists
        TaskListTemplate = List.template(task_dataset, cache_size=10)

        # Create dict of lists
        user_tasks = db.create_dict("user_tasks", TaskListTemplate, use_bloom=False)

        # Benchmark: Create and fill lists
        print(f"\n--- Creating and filling {num_lists:,} lists ---")

        overall_start = time.time()

        for i in range(num_lists):
            user_key = f"user_{i}"
            # Access key auto-creates a list
            tasks = user_tasks[user_key]

            # Add items to list
            for j in range(items_per_list):
                tasks.append(
                    {
                        "id": i * items_per_list + j,
                        "title": f"Task {j} for User {i}",
                        "priority": j % 5,
                    }
                )

            # Progress updates
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - overall_start
                rate = (i + 1) / elapsed
                items_rate = (i + 1) * items_per_list / elapsed
                print(
                    f"  [{i+1:,}/{num_lists:,}] {rate:.1f} lists/sec, {items_rate:,.0f} items/sec, {elapsed:.1f}s elapsed"
                )

        creation_time = time.time() - overall_start
        total_items = num_lists * items_per_list

        print(f"\nCreation complete:")
        print(f"  Time: {creation_time:.2f}s")
        print(f"  Lists/sec: {num_lists/creation_time:,.1f}")
        print(f"  Items/sec: {total_items/creation_time:,.0f}")

        # Save and get file size
        user_tasks.save(force=True)
        db.close()

        file_size = os.path.getsize(filename)
        print(f"\n--- Storage ---")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"  Per list: {file_size/num_lists:.0f} bytes")
        print(f"  Per item: {file_size/total_items:.0f} bytes")

        # Benchmark: Random access
        print(f"\n--- Random Access ---")
        db = DB(filename)
        user_tasks = db._datastructures.get("user_tasks")
        if user_tasks is None:
            task_dataset = db.get_dataset("tasks")
            TaskListTemplate = List.template(task_dataset, cache_size=10)
            user_tasks = db.create_dict("user_tasks", TaskListTemplate, use_bloom=False)

        random.seed(42)
        num_reads = 1000
        start = time.time()

        for _ in range(num_reads):
            list_idx = random.randint(0, num_lists - 1)
            item_idx = random.randint(0, items_per_list - 1)
            user_key = f"user_{list_idx}"
            item = user_tasks[user_key][item_idx]

        random_access_time = time.time() - start
        print(f"  {num_reads:,} random accesses in {random_access_time:.3f}s")
        print(f"  Access rate: {num_reads/random_access_time:,.0f} items/sec")
        print(f"  Avg latency: {random_access_time/num_reads*1000:.3f}ms")

        # Benchmark: Iteration
        print(f"\n--- Iteration ---")
        start = time.time()

        count = 0
        for user_key in user_tasks.keys():
            tasks = user_tasks[user_key]
            for task in tasks:
                count += 1

        iteration_time = time.time() - start
        print(f"  Iterated {count:,} items in {iteration_time:.3f}s")
        print(f"  Iteration rate: {count/iteration_time:,.0f} items/sec")

        db.close()

        return {
            "creation_time": creation_time,
            "random_access_time": random_access_time,
            "iteration_time": iteration_time,
            "file_size": file_size,
            "num_lists": num_lists,
            "items_per_list": items_per_list,
            "total_items": total_items,
        }

    finally:
        try:
            os.unlink(filename)
        except (PermissionError, FileNotFoundError):
            pass


def benchmark_stress_test(num_structures=200):
    """Stress test: create many nested structures to verify no limits.

    Args:
        num_structures: Number of nested structures to create
    """
    print(f"\n{'='*60}")
    print(f"Stress Test: {num_structures:,} Nested Structures")
    print(f"{'='*60}")
    print(f"Testing both List[Dict] and Dict[List] at scale...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".loom") as tmp:
        filename = tmp.name

    try:
        db = DB(filename, header_size=2 * 1024 * 1024)

        # Test List[Dict]
        print(f"\n--- List[Dict] Stress Test ---")
        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDictTemplate = Dict.template(user_dataset, cache_size=0, use_bloom=False)
        teams = db.create_list("teams", UserDictTemplate)

        start = time.time()
        for i in range(num_structures):
            team = teams.append()
            team[f"user_{i}"] = {"id": i, "name": f"User {i}"}
            teams.update_nested_ref(i, team)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  Created {i+1:,} dicts in {elapsed:.1f}s")

        list_dict_time = time.time() - start
        print(f"  List[Dict]: {num_structures:,} dicts in {list_dict_time:.2f}s")

        # Test Dict[List]
        print(f"\n--- Dict[List] Stress Test ---")
        task_dataset = db.create_dataset("tasks", id="uint32", title="U50")
        TaskListTemplate = List.template(task_dataset, cache_size=0)
        user_tasks = db.create_dict("user_tasks", TaskListTemplate, use_bloom=False)

        start = time.time()
        for i in range(num_structures):
            user_key = f"user_{i}"
            tasks = user_tasks[user_key]
            tasks.append({"id": i, "title": f"Task {i}"})

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  Created {i+1:,} lists in {elapsed:.1f}s")

        dict_list_time = time.time() - start
        print(f"  Dict[List]: {num_structures:,} lists in {dict_list_time:.2f}s")

        # Verify
        print(f"\n--- Verification ---")
        assert (
            len(teams) == num_structures
        ), f"Expected {num_structures} teams, got {len(teams)}"
        assert (
            len(user_tasks) == num_structures
        ), f"Expected {num_structures} user_tasks, got {len(user_tasks)}"

        # Random access verification
        random.seed(42)
        for _ in range(100):
            idx = random.randint(0, num_structures - 1)
            # List[Dict]
            team = teams[idx]
            assert team[f"user_{idx}"]["id"] == idx
            # Dict[List]
            tasks = user_tasks[f"user_{idx}"]
            assert tasks[0]["id"] == idx

        print(f"  Verification: PASSED")

        db.close()

        file_size = os.path.getsize(filename)
        print(f"\n--- Summary ---")
        print(f"  Total structures: {num_structures * 2:,}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.0f} KB)")
        print(
            f"  List[Dict] time: {list_dict_time:.2f}s ({num_structures/list_dict_time:.0f}/sec)"
        )
        print(
            f"  Dict[List] time: {dict_list_time:.2f}s ({num_structures/dict_list_time:.0f}/sec)"
        )

        return True

    finally:
        try:
            os.unlink(filename)
        except (PermissionError, FileNotFoundError):
            pass


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CROSS-NESTED DATA STRUCTURE BENCHMARKS")
    print("=" * 60)
    print("\nStructures tested:")
    print("  - List[Dict]: A list where each element is a Dict")
    print("  - Dict[List]: A dict where each value is a List")

    # Stress test first
    print("\n" + "-" * 60)
    print("STRESS TEST")
    print("-" * 60)
    benchmark_stress_test(num_structures=200)

    # List[Dict] benchmarks
    print("\n" + "-" * 60)
    print("LIST[DICT] BENCHMARK - SMALL (50 dicts x 50 items = 2,500 items)")
    print("-" * 60)
    list_dict_small = benchmark_list_of_dicts(num_dicts=50, items_per_dict=50)

    print("\n" + "-" * 60)
    print("LIST[DICT] BENCHMARK - MEDIUM (100 dicts x 100 items = 10,000 items)")
    print("-" * 60)
    list_dict_medium = benchmark_list_of_dicts(num_dicts=100, items_per_dict=100)

    # Dict[List] benchmarks
    print("\n" + "-" * 60)
    print("DICT[LIST] BENCHMARK - SMALL (50 lists x 50 items = 2,500 items)")
    print("-" * 60)
    dict_list_small = benchmark_dict_of_lists(num_lists=50, items_per_list=50)

    print("\n" + "-" * 60)
    print("DICT[LIST] BENCHMARK - MEDIUM (100 lists x 100 items = 10,000 items)")
    print("-" * 60)
    dict_list_medium = benchmark_dict_of_lists(num_lists=100, items_per_list=100)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nList[Dict] Small (2.5K items):")
    print(
        f"  Creation: {list_dict_small['creation_time']:.2f}s ({list_dict_small['total_items']/list_dict_small['creation_time']:,.0f} items/sec)"
    )
    print(
        f"  Random:   {list_dict_small['random_access_time']:.3f}s ({1000/list_dict_small['random_access_time']:,.0f} accesses/sec)"
    )
    print(f"  Storage:  {list_dict_small['file_size']/1024:.0f} KB")

    print(f"\nList[Dict] Medium (10K items):")
    print(
        f"  Creation: {list_dict_medium['creation_time']:.2f}s ({list_dict_medium['total_items']/list_dict_medium['creation_time']:,.0f} items/sec)"
    )
    print(
        f"  Random:   {list_dict_medium['random_access_time']:.3f}s ({1000/list_dict_medium['random_access_time']:,.0f} accesses/sec)"
    )
    print(f"  Storage:  {list_dict_medium['file_size']/1024:.0f} KB")

    print(f"\nDict[List] Small (2.5K items):")
    print(
        f"  Creation: {dict_list_small['creation_time']:.2f}s ({dict_list_small['total_items']/dict_list_small['creation_time']:,.0f} items/sec)"
    )
    print(
        f"  Random:   {dict_list_small['random_access_time']:.3f}s ({1000/dict_list_small['random_access_time']:,.0f} accesses/sec)"
    )
    print(f"  Storage:  {dict_list_small['file_size']/1024:.0f} KB")

    print(f"\nDict[List] Medium (10K items):")
    print(
        f"  Creation: {dict_list_medium['creation_time']:.2f}s ({dict_list_medium['total_items']/dict_list_medium['creation_time']:,.0f} items/sec)"
    )
    print(
        f"  Random:   {dict_list_medium['random_access_time']:.3f}s ({1000/dict_list_medium['random_access_time']:,.0f} accesses/sec)"
    )
    print(f"  Storage:  {dict_list_medium['file_size']/1024:.0f} KB")

    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE!")
    print("=" * 60 + "\n")
