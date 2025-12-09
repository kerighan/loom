"""
Benchmark for nested list performance.

Tests creation and access performance for lists of lists.
"""

import os
import sys
import tempfile
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.database import DB
from loom.datastructures import List


def benchmark_nested_list_creation(num_inner_lists=1000, items_per_list=1000):
    """Benchmark creating a single list of lists.
    
    Args:
        num_inner_lists: Number of inner lists to create
        items_per_list: Number of items in each inner list
    """
    print(f"\n{'='*60}")
    print(f"Nested List Creation Benchmark")
    print(f"{'='*60}")
    print(f"Creating 1 list containing {num_inner_lists:,} inner lists")
    print(f"Each inner list has {items_per_list:,} items")
    print(f"Total items: {num_inner_lists * items_per_list:,}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.loom') as tmp:
        filename = tmp.name
    
    try:
        # Create database with larger header for nested list metadata
        db = DB(filename, header_size=1024*1024)  # 1MB header for metadata
        
        # Create dataset for inner list items
        item_dataset = db.create_dataset("items", id="uint64", value="float64")
        
        # Create template for nested lists
        ItemListTemplate = List.template(item_dataset)
        
        # Create outer list (single list of lists)
        outer_list = db.create_list("teams", ItemListTemplate)
        
        # Benchmark: Create and fill inner lists
        print(f"\n--- Creating and filling {num_inner_lists:,} inner lists ---")
        print(f"Progress: [Lists created] (rate, cumulative time)")
        
        overall_start = time.time()
        
        for i in range(num_inner_lists):
            # Create inner list
            inner_list = outer_list.append()
            
            # Add items to inner list
            items = [{"id": j, "value": float(i * items_per_list + j)} 
                     for j in range(items_per_list)]
            inner_list.append_many(items)
            
            # Update the stored reference with current metadata
            # (The reference was stored at creation time with length=0)
            outer_list.update_nested_ref(i, inner_list)
            
            # Progress updates
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - overall_start
                rate = (i + 1) / elapsed
                items_rate = (i + 1) * items_per_list / elapsed
                print(f"  [{i+1:,}/{num_inner_lists:,}] {rate:.1f} lists/sec, {items_rate:,.0f} items/sec, {elapsed:.1f}s elapsed")
        
        elapsed = time.time() - overall_start
        total_items = num_inner_lists * items_per_list
        
        print(f"\nCreation complete:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Lists/sec: {num_inner_lists/elapsed:,.1f}")
        print(f"  Items/sec: {total_items/elapsed:,.0f}")
        print(f"  Avg time per list: {elapsed/num_inner_lists*1000:.2f}ms")
        
        # Benchmark: Random access
        print(f"\n--- Random Access ---")
        import random
        random.seed(42)
        
        
        num_reads = 1000
        start = time.time()
        
        for _ in range(num_reads):
            list_idx = random.randint(0, num_inner_lists - 1)
            item_idx = random.randint(0, items_per_list - 1)
            item = outer_list[list_idx][item_idx]
        
        elapsed = time.time() - start
        print(f"  {num_reads:,} random accesses in {elapsed:.3f}s")
        print(f"  Access rate: {num_reads/elapsed:,.0f} items/sec")
        
        # Benchmark: Sequential iteration
        print(f"\n--- Sequential Iteration ---")
        start = time.time()
        
        count = 0
        for inner_list in outer_list:
            for item in inner_list:
                count += 1
        
        elapsed = time.time() - start
        print(f"  Iterated {count:,} items in {elapsed:.2f}s")
        print(f"  Iteration rate: {count/elapsed:,.0f} items/sec")
        
        # File size
        file_size = os.path.getsize(filename)
        print(f"\n--- Storage ---")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"  Bytes per item: {file_size/total_items:.1f}")
        
        db.close()
        
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except PermissionError:
            pass  # File still in use on Windows


def benchmark_comparison(num_outer=100, num_inner=100):
    """Compare nested lists vs flat list performance."""
    print(f"\n{'='*60}")
    print(f"Nested vs Flat List Comparison")
    print(f"{'='*60}")
    
    total_items = num_outer * num_inner
    print(f"Total items: {total_items:,} ({num_outer} x {num_inner})")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.loom') as tmp:
        filename = tmp.name
    
    try:
        # Test 1: Nested lists
        print(f"\n--- Nested List (list of lists) ---")
        db = DB(filename, header_size=1024*1024)  # 1MB header
        
        item_dataset = db.create_dataset("items", id="uint64", value="float64")
        ItemListTemplate = List.template(item_dataset)
        nested = db.create_list("nested", ItemListTemplate)
        
        start = time.time()
        for i in range(num_outer):
            inner = nested.append()
            items = [{"id": j, "value": float(i * num_inner + j)} 
                     for j in range(num_inner)]
            inner.append_many(items)
        nested_time = time.time() - start
        
        print(f"  Creation: {nested_time:.3f}s ({total_items/nested_time:,.0f} items/sec)")
        
        db.close()
        os.remove(filename)
        
        # Test 2: Flat list
        print(f"\n--- Flat List (single list) ---")
        db = DB(filename)
        
        flat = db.create_list("flat", {"id": "uint64", "value": "float64", "group": "uint64"})
        
        start = time.time()
        all_items = []
        for i in range(num_outer):
            for j in range(num_inner):
                all_items.append({
                    "id": j, 
                    "value": float(i * num_inner + j),
                    "group": i
                })
        flat.append_many(all_items)
        flat_time = time.time() - start
        
        print(f"  Creation: {flat_time:.3f}s ({total_items/flat_time:,.0f} items/sec)")
        
        print(f"\n--- Comparison ---")
        print(f"  Nested/Flat ratio: {nested_time/flat_time:.2f}x")
        if nested_time < flat_time:
            print(f"  Nested is {flat_time/nested_time:.1f}x faster")
        else:
            print(f"  Flat is {nested_time/flat_time:.1f}x faster")
        
        db.close()
        
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except PermissionError:
            pass  # File still in use on Windows


if __name__ == "__main__":
    # Quick test with smaller dataset
    print("Running quick test (100 inner lists x 100 items each)...")
    benchmark_nested_list_creation(num_inner_lists=100, items_per_list=100)
    
    # Comparison
    benchmark_comparison(num_outer=100, num_inner=100)
    
    # Full benchmark (uncomment for full test)
    print("\n\nRunning full benchmark (1000 inner lists x 1000 items each)...")
    print("This will create 1 million items total...")
    benchmark_nested_list_creation(num_inner_lists=1000, items_per_list=1000)
