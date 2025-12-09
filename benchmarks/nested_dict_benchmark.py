"""Benchmark nested Dict structures: Dict[Dict].

Tests the optimized nested Dict implementation with:
- Shared datasets (no 127 dataset limit)
- Compact binary references (~84 bytes per nested dict)
- No bloom filters for nested dicts
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import time
import random
from loom.database import DB
from loom.datastructures.dict import Dict


def benchmark_nested_dict_creation(num_inner_dicts=100, items_per_dict=100):
    """Benchmark creating nested dicts (Dict of Dicts).
    
    Args:
        num_inner_dicts: Number of inner dicts to create
        items_per_dict: Number of items in each inner dict
    """
    print(f"\n{'='*60}")
    print(f"Nested Dict Creation Benchmark")
    print(f"{'='*60}")
    print(f"Creating 1 dict containing {num_inner_dicts:,} inner dicts")
    print(f"Each inner dict has {items_per_dict:,} items")
    print(f"Total items: {num_inner_dicts * items_per_dict:,}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.loom') as tmp:
        filename = tmp.name
    
    try:
        # Create database
        db = DB(filename)
        
        # Create dataset for inner dict items
        user_dataset = db.create_dataset(
            "users", id="uint64", name="U50", score="float64"
        )
        
        # Create template for nested dicts
        UserDictTemplate = Dict.template(user_dataset, cache_size=100, use_bloom=False)
        
        # Create outer dict (dict of dicts)
        teams = db.create_dict("teams", UserDictTemplate, use_bloom=False)
        
        # Benchmark: Create and fill inner dicts
        print(f"\n--- Creating and filling {num_inner_dicts:,} inner dicts ---")
        
        overall_start = time.time()
        
        for i in range(num_inner_dicts):
            team_name = f"team_{i}"
            # Auto-creates nested dict on access
            team = teams[team_name]
            
            # Add items to inner dict
            for j in range(items_per_dict):
                user_key = f"user_{j}"
                team[user_key] = {
                    "id": i * items_per_dict + j,
                    "name": f"User {j} of Team {i}",
                    "score": float(i * 100 + j)
                }
            
            # Progress updates
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - overall_start
                rate = (i + 1) / elapsed
                items_rate = (i + 1) * items_per_dict / elapsed
                print(f"  [{i+1:,}/{num_inner_dicts:,}] {rate:.1f} dicts/sec, {items_rate:,.0f} items/sec, {elapsed:.1f}s elapsed")
        
        creation_time = time.time() - overall_start
        total_items = num_inner_dicts * items_per_dict
        
        print(f"\nCreation complete:")
        print(f"  Time: {creation_time:.2f}s")
        print(f"  Dicts/sec: {num_inner_dicts/creation_time:,.1f}")
        print(f"  Items/sec: {total_items/creation_time:,.0f}")
        
        # Save and get file size before random access
        teams.save(force=True)
        db.close()
        
        file_size = os.path.getsize(filename)
        print(f"\n--- Storage ---")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"  Per inner dict: {file_size/num_inner_dicts:.0f} bytes")
        print(f"  Per item: {file_size/total_items:.0f} bytes")
        
        # Benchmark: Random access
        print(f"\n--- Random Access ---")
        db = DB(filename)
        teams = db._datastructures.get("teams")
        if teams is None:
            user_dataset = db.get_dataset("users")
            UserDictTemplate = Dict.template(user_dataset, cache_size=100, use_bloom=False)
            teams = db.create_dict("teams", UserDictTemplate, use_bloom=False)
        
        random.seed(42)
        num_reads = 1000
        start = time.time()
        
        for _ in range(num_reads):
            dict_idx = random.randint(0, num_inner_dicts - 1)
            item_idx = random.randint(0, items_per_dict - 1)
            team_name = f"team_{dict_idx}"
            user_key = f"user_{item_idx}"
            item = teams[team_name][user_key]
        
        random_access_time = time.time() - start
        print(f"  {num_reads:,} random accesses in {random_access_time:.3f}s")
        print(f"  Access rate: {num_reads/random_access_time:,.0f} items/sec")
        print(f"  Avg latency: {random_access_time/num_reads*1000:.3f}ms")
        
        # Benchmark: Iteration
        print(f"\n--- Iteration ---")
        start = time.time()
        
        count = 0
        for team_name in teams.keys():
            team = teams[team_name]
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
            "num_inner_dicts": num_inner_dicts,
            "items_per_dict": items_per_dict,
            "total_items": total_items,
        }
        
    finally:
        try:
            os.unlink(filename)
        except (PermissionError, FileNotFoundError):
            pass


def benchmark_stress_test(num_inner_dicts=1000):
    """Stress test: create many nested dicts to verify no dataset limit.
    
    Previously this would fail at ~60 nested dicts due to 127 dataset limit.
    With shared datasets, we can create thousands.
    """
    print(f"\n{'='*60}")
    print(f"Stress Test: {num_inner_dicts:,} Nested Dicts")
    print(f"{'='*60}")
    print(f"Testing that we can exceed the old 127 dataset limit...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.loom') as tmp:
        filename = tmp.name
    
    try:
        db = DB(filename)
        
        user_dataset = db.create_dataset("users", id="uint32", name="U50")
        UserDictTemplate = Dict.template(user_dataset, cache_size=0, use_bloom=False)
        departments = db.create_dict("departments", UserDictTemplate, use_bloom=False)
        
        start = time.time()
        
        for i in range(num_inner_dicts):
            dept = departments[f"dept_{i}"]
            dept[f"user_{i}"] = {"id": i, "name": f"User {i}"}
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  Created {i+1:,} nested dicts in {elapsed:.1f}s")
        
        elapsed = time.time() - start
        
        print(f"\nStress test PASSED!")
        print(f"  Created {num_inner_dicts:,} nested dicts")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {num_inner_dicts/elapsed:,.0f} dicts/sec")
        
        # Verify random access works
        random.seed(42)
        for _ in range(100):
            idx = random.randint(0, num_inner_dicts - 1)
            dept = departments[f"dept_{idx}"]
            user = dept[f"user_{idx}"]
            assert user["id"] == idx
        
        print(f"  Random access verification: PASSED")
        
        db.close()
        
        file_size = os.path.getsize(filename)
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.0f} KB)")
        print(f"  Per nested dict: {file_size/num_inner_dicts:.0f} bytes")
        
        return True
        
    finally:
        try:
            os.unlink(filename)
        except (PermissionError, FileNotFoundError):
            pass


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NESTED DICT BENCHMARKS")
    print("=" * 60)
    print("\nOptimizations tested:")
    print("  - Shared datasets (no 127 dataset limit)")
    print("  - Compact binary references (~84 bytes per nested dict)")
    print("  - No bloom filters for nested dicts")
    print("  - Parent updates nested refs on modification")
    
    # Stress test first - verify the optimization works
    print("\n" + "-" * 60)
    print("STRESS TEST")
    print("-" * 60)
    benchmark_stress_test(num_inner_dicts=500)
    
    # Small benchmark
    print("\n" + "-" * 60)
    print("SMALL SCALE (100 dicts x 100 items = 10,000 items)")
    print("-" * 60)
    small = benchmark_nested_dict_creation(num_inner_dicts=100, items_per_dict=100)
    
    # Medium benchmark  
    print("\n" + "-" * 60)
    print("MEDIUM SCALE (500 dicts x 100 items = 50,000 items)")
    print("-" * 60)
    medium = benchmark_nested_dict_creation(num_inner_dicts=500, items_per_dict=100)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nSmall scale (10K items):")
    print(f"  Creation: {small['creation_time']:.2f}s ({small['total_items']/small['creation_time']:,.0f} items/sec)")
    print(f"  Random:   {small['random_access_time']:.3f}s ({1000/small['random_access_time']:,.0f} accesses/sec)")
    print(f"  Iterate:  {small['iteration_time']:.3f}s ({small['total_items']/small['iteration_time']:,.0f} items/sec)")
    print(f"  Storage:  {small['file_size']/1024:.0f} KB ({small['file_size']/small['total_items']:.0f} bytes/item)")
    
    print(f"\nMedium scale (50K items):")
    print(f"  Creation: {medium['creation_time']:.2f}s ({medium['total_items']/medium['creation_time']:,.0f} items/sec)")
    print(f"  Random:   {medium['random_access_time']:.3f}s ({1000/medium['random_access_time']:,.0f} accesses/sec)")
    print(f"  Iterate:  {medium['iteration_time']:.3f}s ({medium['total_items']/medium['iteration_time']:,.0f} items/sec)")
    print(f"  Storage:  {medium['file_size']/1024:.0f} KB ({medium['file_size']/medium['total_items']:.0f} bytes/item)")
    
    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE!")
    print("=" * 60 + "\n")
