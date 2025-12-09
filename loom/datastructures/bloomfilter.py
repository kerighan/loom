"""
Optimized Bloom Filter implementation.

A space-efficient probabilistic data structure for membership testing.
- False positives possible (says "maybe in set")
- False negatives impossible (says "definitely not in set")
- Extremely fast lookups
- Minimal memory usage

Modern optimizations:
- Multiple hash functions from single hash (double hashing)
- Optimal bit array sizing
- Efficient bit manipulation
- Cache-friendly access patterns
"""

import mmh3
import numpy as np
from loom.datastructures.base import DataStructure


class BloomFilter(DataStructure):
    """Optimized Bloom filter for fast membership testing.

    Usage:
        # Create
        bf = BloomFilter('seen_users', db, expected_items=10000, false_positive_rate=0.01)

        # Add items
        bf.add(12345)
        bf.add("user@example.com")

        # Check membership
        if 12345 in bf:
            print("Probably seen before")

        if 99999 in bf:
            print("Definitely not seen")

    Performance:
        - Add: O(k) where k = number of hash functions (typically 3-10)
        - Check: O(k)
        - Space: ~10 bits per item for 1% false positive rate
    """

    def __init__(self, name, db, expected_items=10000, false_positive_rate=0.01):
        """Initialize Bloom filter.

        Args:
            name: Unique name for this filter
            db: DB instance
            expected_items: Expected number of items to store
            false_positive_rate: Desired false positive rate (0.0 to 1.0)
        """
        super().__init__(name, db)

        # Load or initialize
        metadata = self._load_metadata()
        if metadata:
            self._load()
        else:
            # Only calculate parameters for new filter
            self.expected_items = expected_items
            self.false_positive_rate = false_positive_rate
            self.num_bits = self._optimal_num_bits(expected_items, false_positive_rate)
            self.num_hashes = self._optimal_num_hashes(self.num_bits, expected_items)
            self._initialize()

    @staticmethod
    def _optimal_num_bits(n, p):
        """Calculate optimal number of bits.

        Formula: m = -n * ln(p) / (ln(2)^2)

        Args:
            n: Expected number of items
            p: Desired false positive rate

        Returns:
            Optimal number of bits
        """
        import math

        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_num_hashes(m, n):
        """Calculate optimal number of hash functions.

        Formula: k = (m/n) * ln(2)

        Args:
            m: Number of bits
            n: Expected number of items

        Returns:
            Optimal number of hash functions
        """
        import math

        k = (m / n) * math.log(2)
        return int(math.ceil(k))

    def _initialize(self):
        """Initialize new Bloom filter."""
        # Create dataset to store bit array
        # NumPy bool uses 8 bits anyway, so we use uint8 (same space)
        # This allows us to easily upgrade to counting later
        num_buckets = self.num_bits

        dataset_name = f"_bf_{self.name}_bits"
        self._bits_dataset = self._db.create_dataset(
            dataset_name, bit="uint8"  # 8 bits per bucket (same as bool in numpy)
        )

        # Allocate bit array and initialize to zero
        self._bits_addr = self._bits_dataset.allocate_block(num_buckets)
        for i in range(num_buckets):
            addr = self._bits_addr + i * self._bits_dataset.record_size
            self._bits_dataset[addr] = {"bit": 0}

        # Save metadata
        self._save_metadata(
            {
                "num_bits": self.num_bits,
                "num_hashes": self.num_hashes,
                "expected_items": self.expected_items,
                "false_positive_rate": self.false_positive_rate,
                "bits_dataset": dataset_name,
                "bits_addr": self._bits_addr,
                "num_items": 0,
            }
        )

        self.num_items = 0

    def _load(self):
        """Load existing Bloom filter."""
        metadata = self._load_metadata()

        self.num_bits = metadata["num_bits"]
        self.num_hashes = metadata["num_hashes"]
        self.expected_items = metadata["expected_items"]
        self.false_positive_rate = metadata["false_positive_rate"]
        self.num_items = metadata["num_items"]

        self._bits_dataset = self._get_dataset(metadata["bits_dataset"])
        self._bits_addr = metadata["bits_addr"]

    def _get_hashes(self, item):
        """Generate k hash values using double hashing.

        Uses MurmurHash3 with two seeds to generate k independent hashes.
        This is faster than calling k different hash functions.

        Formula: h_i(x) = (h1(x) + i * h2(x)) mod m

        Args:
            item: Item to hash (will be converted to string)

        Returns:
            List of k hash values
        """
        if not isinstance(item, (str, bytes)):
            item = str(item)
        if isinstance(item, str):
            item = item.encode("utf-8")

        # Two hash values from MurmurHash3
        h1 = mmh3.hash(item, seed=0) % self.num_bits
        h2 = mmh3.hash(item, seed=1) % self.num_bits

        # Generate k hashes using double hashing
        hashes = []
        for i in range(self.num_hashes):
            hash_val = (h1 + i * h2) % self.num_bits
            hashes.append(hash_val)

        return hashes

    def _get_bit(self, bit_index):
        """Get value of a specific bit.

        Args:
            bit_index: Index of bit to get

        Returns:
            1 if bit is set, 0 otherwise
        """
        addr = self._bits_addr + bit_index * self._bits_dataset.record_size
        bit_val = self._bits_dataset[addr]["bit"]

        return 1 if bit_val > 0 else 0

    def _set_bit(self, bit_index):
        """Set a specific bit to 1.

        Args:
            bit_index: Index of bit to set
        """
        addr = self._bits_addr + bit_index * self._bits_dataset.record_size
        self._bits_dataset[addr] = {"bit": 1}

    def add(self, item):
        """Add an item to the Bloom filter.

        Args:
            item: Item to add (will be hashed)
        """
        hashes = self._get_hashes(item)

        for hash_val in hashes:
            self._set_bit(hash_val)

        # Update count and auto-save periodically
        self.num_items += 1
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

        for hash_val in hashes:
            if self._get_bit(hash_val) == 0:
                return False  # Definitely not in set

        return True  # Probably in set

    def __len__(self):
        """Get approximate number of items added.

        Note: This is a count of add() calls, not unique items.
        """
        return self.num_items

    def save(self):
        """Save metadata to disk.

        Call this periodically or before closing to persist num_items count.
        Automatically called by DB context manager on exit.
        """
        self._save_metadata(
            {
                "num_bits": self.num_bits,
                "num_hashes": self.num_hashes,
                "expected_items": self.expected_items,
                "false_positive_rate": self.false_positive_rate,
                "bits_dataset": self._bits_dataset.name,
                "bits_addr": self._bits_addr,
                "num_items": self.num_items,
            }
        )

    def clear(self):
        """Clear all items from the filter."""
        for i in range(self.num_bits):
            addr = self._bits_addr + i * self._bits_dataset.record_size
            self._bits_dataset[addr] = {"bit": 0}

        self.num_items = 0
        metadata = self._load_metadata()
        metadata["num_items"] = 0
        self._save_metadata(metadata)

    @property
    def current_false_positive_rate(self):
        """Calculate current false positive rate based on items added.

        Formula: p ≈ (1 - e^(-kn/m))^k
        where k = num_hashes, n = num_items, m = num_bits
        """
        import math

        if self.num_items == 0:
            return 0.0

        exponent = -self.num_hashes * self.num_items / self.num_bits
        p = (1 - math.exp(exponent)) ** self.num_hashes
        return p

    def __repr__(self):
        """String representation."""
        return (
            f"BloomFilter('{self.name}', "
            f"items={self.num_items}/{self.expected_items}, "
            f"fp_rate={self.current_false_positive_rate:.4f})"
        )
