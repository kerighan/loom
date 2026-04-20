"""
Counting Bloom Filter with removal support.

Same space as regular Bloom filter (uint8 per bucket) but supports removal!
- Counters instead of bits (0-255 per bucket)
- Can remove items by decrementing counters
- Same false positive guarantees
- Max 255 items per bucket (configurable)

Perfect for hash tables and caches where items are added/removed frequently.
"""

import mmh3
from loom.datastructures.base import DataStructure


class CounterOverflowError(Exception):
    """Raised when a bucket counter would overflow.

    This indicates too many hash collisions. Solutions:
    - Increase expected_items parameter (more buckets)
    - Increase max_count parameter (higher limit per bucket)
    - Use a different data structure
    """

    pass


class CountingBloomFilter(DataStructure):
    # Top-level only — cannot be nested
    _outer_types_supported = ()
    _inner_types_supported = ()
    """Counting Bloom filter with removal support.

    Uses uint8 counters (0-255) instead of bits. Since NumPy bool uses 8 bits anyway,
    this costs NO extra space but enables removal!

    Usage:
        # Create
        cbf = CountingBloomFilter('cache', db, expected_items=10000)

        # Add items
        cbf.add("user123")
        cbf.add("user456")

        # Check membership
        if "user123" in cbf:
            print("Probably in set")

        # Remove items (unlike regular Bloom filter!)
        cbf.remove("user123")

        # Now it's gone
        assert "user123" not in cbf

    Performance:
        - Add: O(k) where k = number of hash functions
        - Remove: O(k)
        - Check: O(k)
        - Space: 8 bits per bucket (same as regular Bloom filter in NumPy!)

    Limitations:
        - Max 255 items can hash to same bucket (configurable with max_count)
        - Counter overflow if exceeded (wraps to 0, breaking the filter)
        - Still has false positives (like regular Bloom filter)
    """

    def __init__(
        self, name, db, expected_items=10000, false_positive_rate=0.01, max_count=255
    ):
        """Initialize Counting Bloom filter.

        Args:
            name: Unique name for this filter
            db: DB instance
            expected_items: Expected number of items to store
            false_positive_rate: Desired false positive rate (0.0 to 1.0)
            max_count: Maximum count per bucket (default 255 for uint8)
        """
        super().__init__(name, db)

        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        self.max_count = max_count

        # Calculate optimal parameters
        self.num_buckets = self._optimal_num_bits(expected_items, false_positive_rate)
        self.num_hashes = self._optimal_num_hashes(self.num_buckets, expected_items)

        # Load or initialize
        metadata = self._load_metadata()
        if metadata:
            self._load()
        else:
            self._initialize()

    @staticmethod
    def _optimal_num_bits(n, p):
        """Calculate optimal number of buckets (same formula as Bloom filter)."""
        import math

        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_num_hashes(m, n):
        """Calculate optimal number of hash functions."""
        import math

        k = (m / n) * math.log(2)
        return int(math.ceil(k))

    def _initialize(self):
        """Initialize new Counting Bloom filter."""
        # Create dataset to store counters
        # uint8 = 8 bits = same as NumPy bool, but can count 0-255!
        dataset_name = f"_cbf_{self.name}_counters"
        self._counters_dataset = self._db.create_dataset(
            dataset_name, count="uint8"  # 0-255 counter per bucket
        )

        # Allocate counter array and bulk-init with valid prefix in one write
        self._counters_addr = self._counters_dataset.allocate_block(self.num_buckets)
        zero_record = self._counters_dataset._serialize(count=0)
        self._counters_dataset.db.write(
            self._counters_addr, zero_record * self.num_buckets
        )

        # Save metadata
        self._save_metadata(
            {
                "num_buckets": self.num_buckets,
                "num_hashes": self.num_hashes,
                "expected_items": self.expected_items,
                "false_positive_rate": self.false_positive_rate,
                "max_count": self.max_count,
                "counters_dataset": dataset_name,
                "counters_addr": self._counters_addr,
                "num_items": 0,
            }
        )

        self.num_items = 0
        self._cache_raw_fields()

    def _cache_raw_fields(self):
        """Cache record size and raw DB handle for direct mmap access."""
        self._record_size = self._counters_dataset.record_size
        self._db_raw = self._counters_dataset.db

    def _load(self):
        """Load existing Counting Bloom filter."""
        metadata = self._load_metadata()

        self.num_buckets = metadata["num_buckets"]
        self.num_hashes = metadata["num_hashes"]
        self.expected_items = metadata["expected_items"]
        self.false_positive_rate = metadata["false_positive_rate"]
        self.max_count = metadata["max_count"]
        self.num_items = metadata["num_items"]

        self._counters_dataset = self._get_dataset(metadata["counters_dataset"])
        self._counters_addr = metadata["counters_addr"]
        self._cache_raw_fields()

    def _get_hashes(self, item):
        """Generate k hash values using double hashing."""
        if not isinstance(item, (str, bytes)):
            item = str(item)
        if isinstance(item, str):
            item = item.encode("utf-8")

        # Two hash values from MurmurHash3
        h1 = mmh3.hash(item, seed=0) % self.num_buckets
        h2 = mmh3.hash(item, seed=1) % self.num_buckets

        # Generate k hashes using double hashing
        hashes = []
        for i in range(self.num_hashes):
            hash_val = (h1 + i * h2) % self.num_buckets
            hashes.append(hash_val)

        return hashes

    def _get_counter(self, bucket_index):
        """Get counter value for a bucket (raw mmap, no numpy)."""
        # Record layout: [_prefix: 1 byte][count: 1 byte]
        addr = self._counters_addr + bucket_index * self._record_size
        return self._db_raw.read(addr + 1, 1)[0]

    def _set_counter(self, bucket_index, value):
        """Set counter value for a bucket (raw mmap, no numpy)."""
        addr = self._counters_addr + bucket_index * self._record_size
        self._db_raw.write(addr + 1, bytes([min(value, self.max_count)]))

    def _increment_counter(self, bucket_index):
        """Increment counter for a bucket.

        Raises:
            CounterOverflowError: If counter is already at max_count
        """
        addr = self._counters_addr + bucket_index * self._record_size
        current = self._db_raw.read(addr + 1, 1)[0]
        if current >= self.max_count:
            raise CounterOverflowError(
                f"Bucket {bucket_index} counter at maximum ({self.max_count}). "
                f"Too many hash collisions. Consider increasing expected_items "
                f"or max_count parameter."
            )
        self._db_raw.write(addr + 1, bytes([current + 1]))

    def _decrement_counter(self, bucket_index):
        """Decrement counter for a bucket."""
        current = self._get_counter(bucket_index)
        if current > 0:
            self._set_counter(bucket_index, current - 1)

    def add(self, item):
        """Add an item to the filter.

        Args:
            item: Item to add (will be hashed)
        """
        hashes = self._get_hashes(item)

        for hash_val in hashes:
            self._increment_counter(hash_val)

        # Update count and auto-save periodically
        self.num_items += 1
        self._auto_save_check()

    def remove(self, item):
        """Remove an item from the filter.

        This is the key feature that regular Bloom filters don't have!

        Args:
            item: Item to remove (will be hashed)

        Note:
            - If item wasn't actually in the filter, counters may be decremented incorrectly
            - This can lead to false negatives if you remove items that weren't added
            - Always pair add() and remove() calls correctly
        """
        hashes = self._get_hashes(item)

        for hash_val in hashes:
            self._decrement_counter(hash_val)

        # Update count and auto-save periodically
        if self.num_items > 0:
            self.num_items -= 1
            self._auto_save_check()

    def __contains__(self, item):
        """Check if item might be in the filter.

        Args:
            item: Item to check

        Returns:
            True if item might be in set (or false positive)
            False if item is definitely not in set
        """
        hashes = self._get_hashes(item)

        # OPTIMIZATION: Read all counters in one bulk read
        # Instead of k separate reads, read the entire counter array once
        record_size = self._counters_dataset.record_size

        for hash_val in hashes:
            # Direct byte read - record is [_prefix (1 byte), count (1 byte)]
            addr = self._counters_addr + hash_val * record_size
            data = self._counters_dataset.db.read(addr, record_size)
            count = data[1] if len(data) > 1 else 0  # count is at offset 1
            if count == 0:
                return False  # Definitely not in set

        return True  # Probably in set

    def __delitem__(self, item):
        """Remove an item using del operator.

        Args:
            item: Item to remove

        Example:
            del cbf["item"]  # Same as cbf.remove("item")
        """
        self.remove(item)

    def __len__(self):
        """Get approximate number of items added (minus removed)."""
        return self.num_items

    def save(self):
        """Save metadata to disk.

        Call this periodically or before closing to persist num_items count.
        Automatically called by DB context manager on exit.
        """
        metadata = self._load_metadata()
        metadata["num_items"] = self.num_items
        self._save_metadata(metadata)

    def clear(self):
        """Clear all items from the filter."""
        for i in range(self.num_buckets):
            addr = self._counters_addr + i * self._counters_dataset.record_size
            self._counters_dataset[addr] = {"count": 0}

        self.num_items = 0
        metadata = self._load_metadata()
        metadata["num_items"] = 0
        self._save_metadata(metadata)

    @property
    def current_false_positive_rate(self):
        """Calculate current false positive rate based on items added."""
        import math

        if self.num_items == 0:
            return 0.0

        exponent = -self.num_hashes * self.num_items / self.num_buckets
        p = (1 - math.exp(exponent)) ** self.num_hashes
        return p

    # ---- Registry protocol ----

    def _get_registry_params(self):
        return {
            "expected_items": self.expected_items,
            "false_positive_rate": self.false_positive_rate,
            "max_count": self.max_count,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db,
            params["expected_items"],
            params["false_positive_rate"],
            params.get("max_count", 255),
        )

    def __repr__(self):
        """String representation."""
        return (
            f"CountingBloomFilter('{self.name}', "
            f"items={self.num_items}/{self.expected_items}, "
            f"fp_rate={self.current_false_positive_rate:.4f})"
        )
