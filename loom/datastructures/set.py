"""
Persistent Set implementation.

A Set is a collection of unique items with O(1) add, remove, and membership testing.
Implemented as a thin wrapper around Dict.
"""

from .dict import Dict
from .base import DataStructure


class Set(DataStructure):
    """Persistent set with O(1) operations.

    Stores unique items with fast membership testing via bloom filters.
    Backed by Dict internally.

    Usage:
        with DB("mydata.db") as db:
            users = db.create_set("active_users", key_size=50)

            # Add items
            users.add("alice")
            users.add("bob")

            # Check membership
            if "alice" in users:
                print("Alice is active")

            # Remove items
            users.remove("alice")
            users.discard("charlie")  # No error if missing

            # Iterate
            for user in users:
                print(user)

            # Bulk operations
            users.update(["charlie", "diana"])

    Performance:
        - Add: O(1) average
        - Remove: O(1) average
        - Contains: O(1) average (bloom filter accelerated)
        - Iteration: O(n)
    """

    # Dummy schema for values (minimal storage)
    _DUMMY_SCHEMA = {"_v": "bool"}
    _DUMMY_VALUE = {"_v": True}

    def __init__(
        self,
        name: str,
        db,
        key_size: int = 50,
        use_bloom: bool = True,
        cache_size: int = 0,  # No cache needed for sets
        _load_existing: bool = False,
    ):
        """Initialize a Set.

        Args:
            name: Unique name for this set
            db: Database instance
            key_size: Maximum length of string keys (default: 50)
            use_bloom: Whether to use bloom filter for fast lookups (default: True)
            cache_size: LRU cache size (default: 0, disabled for sets)
            _load_existing: If True, load existing set from disk
        """
        super().__init__(name, db)

        self._key_size = key_size
        self._use_bloom = use_bloom
        self._cache_size = cache_size

        if _load_existing:
            self._load()
        else:
            self._initialize()

    def _initialize(self):
        """Initialize a new set."""
        # Create a dataset for dummy values
        dummy_ds_name = f"_set_{self.name}_values"
        if self._db.has_dataset(dummy_ds_name):
            dummy_ds = self._db.get_dataset(dummy_ds_name)
        else:
            dummy_ds = self._db.create_dataset(dummy_ds_name, **self._DUMMY_SCHEMA)

        # Create internal dict with dummy value schema
        self._dict = Dict(
            name=f"_set_{self.name}_dict",
            db=self._db,
            dataset_or_template=dummy_ds,
            key_size=self._key_size,
            use_bloom=self._use_bloom,
            cache_size=self._cache_size,
        )
        self.save()

    def _load(self):
        """Load existing set from disk."""
        metadata = self._load_metadata() or {}
        self._key_size = metadata.get("key_size", 50)
        self._use_bloom = metadata.get("use_bloom", True)
        self._cache_size = metadata.get("cache_size", 0)

        # Load internal dict
        self._dict = Dict(
            name=f"_set_{self.name}_dict",
            db=self._db,
            dataset_or_template=None,  # Will load from existing
        )

    def save(self):
        """Save set metadata."""
        self._save_metadata(
            {
                "type": "Set",
                "key_size": self._key_size,
                "use_bloom": self._use_bloom,
                "cache_size": self._cache_size,
                "dict_name": f"_set_{self.name}_dict",
            }
        )

    # ========== Core Set Operations ==========

    def add(self, item, atomic: bool = False):
        """Add an item to the set.

        Args:
            item: Item to add (must be string)
            atomic: If True, use WAL for crash safety

        No effect if item already present.
        """
        if atomic:
            self._dict.set(item, self._DUMMY_VALUE, atomic=True)
        else:
            self._dict[item] = self._DUMMY_VALUE

    def remove(self, item):
        """Remove an item from the set.

        Args:
            item: Item to remove

        Raises:
            KeyError: If item not in set
        """
        del self._dict[item]

    def discard(self, item):
        """Remove an item if present.

        Args:
            item: Item to remove

        No error if item not present.
        """
        try:
            del self._dict[item]
        except KeyError:
            pass

    def pop(self):
        """Remove and return an arbitrary item.

        Returns:
            An item from the set

        Raises:
            KeyError: If set is empty
        """
        for item in self:
            self.remove(item)
            return item
        raise KeyError("pop from an empty set")

    def clear(self):
        """Remove all items from the set."""
        # Delete all items
        for item in list(self.keys()):
            self.discard(item)

    # ========== Membership & Lookup ==========

    def __contains__(self, item) -> bool:
        """Check if item is in set."""
        return item in self._dict

    def __len__(self) -> int:
        """Return number of items in set."""
        return len(self._dict)

    def __bool__(self) -> bool:
        """Return True if set is non-empty."""
        return len(self) > 0

    # ========== Iteration ==========

    def __iter__(self):
        """Iterate over items in set."""
        return iter(self._dict.keys())

    def keys(self):
        """Return iterator over items (alias for __iter__)."""
        return self._dict.keys()

    # ========== Bulk Operations ==========

    def update(self, items, atomic: bool = False):
        """Add multiple items to the set.

        Args:
            items: Iterable of items to add
            atomic: If True, use WAL for crash safety
        """
        for item in items:
            self.add(item, atomic=atomic)

    # ========== Set Operations ==========

    def issubset(self, other) -> bool:
        """Test if every item in this set is in other."""
        for item in self:
            if item not in other:
                return False
        return True

    def issuperset(self, other) -> bool:
        """Test if every item in other is in this set."""
        for item in other:
            if item not in self:
                return False
        return True

    def isdisjoint(self, other) -> bool:
        """Test if sets have no items in common."""
        for item in self:
            if item in other:
                return False
        return True

    # ========== Representation ==========

    def __repr__(self) -> str:
        """String representation."""
        return f"Set(name='{self.name}', size={len(self)})"

    # ========== DataStructure Interface ==========

    def close(self):
        """Close the set and save metadata."""
        self._dict.close()
        self.save()
