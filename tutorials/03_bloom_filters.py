# Bloom Filters
# Fast membership testing: "Have I seen this before?"

from loom.database import DB
import os

# Clean up any existing database
try:
    if os.path.exists("bloom_example.loom"):
        os.remove("bloom_example.loom")
except PermissionError:
    pass

db = DB("bloom_example.loom")

# --- Basic Bloom Filter ---
# Perfect for: deduplication, caching, "seen" tracking
seen_urls = db.create_bloomfilter("seen_urls", expected_items=10000)

# Add items
seen_urls.add("https://example.com/page1")
seen_urls.add("https://example.com/page2")

# Check membership (set-like syntax)
print("https://example.com/page1" in seen_urls)  # True (probably)
print("https://example.com/page3" in seen_urls)  # False (definitely)

# Note: False positives possible, false negatives impossible
# If it says "not in", it's definitely not in
# If it says "in", it's probably in (small chance of false positive)

print(len(seen_urls))  # 2


# --- Counting Bloom Filter ---
# Like Bloom filter, but supports removal (uses counters internally)
cache = db.create_counting_bloomfilter("cache", expected_items=1000)

# Add items
cache.add("user:123")
cache.add("user:456")
cache.add("user:789")

print("user:123" in cache)  # True
print("user:999" in cache)  # False

# Remove items (this is what makes it "counting")
cache.remove("user:123")
print("user:123" in cache)  # False (removed)

# Can add and remove multiple times
cache.add("session:abc")
cache.add("session:abc")  # Adding again is safe
cache.remove("session:abc")
print("session:abc" in cache)  # False (removed)

print(f"Approximate items: {len(cache)}")  # Tracks add/remove count

db.close()

# Cleanup
try:
    os.remove("bloom_example.loom")
except PermissionError:
    pass  # File still in use on Windows
