"""
Persistent Set implementation.

A Set is a collection of unique items with O(1) add, remove, and membership testing.
Implemented as a thin wrapper around Dict.
"""

from .dict import Dict
from .base import DataStructure
from .template import DataStructureTemplate
from loom.cache import LRUCache


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

    # Nested set parameters
    MAX_TABLES_NESTED = 8  # Same as Dict for nested

    @classmethod
    def template(cls, key_size=50, use_bloom=False, cache_size=0):
        """Create a template for nested Sets.

        Unlike List and Dict, Set doesn't need a dataset since it only stores keys.

        Args:
            key_size: Maximum length of string keys (default: 50)
            use_bloom: Whether to use bloom filter (default: False for nested)
            cache_size: LRU cache size (default: 0)

        Returns:
            SetTemplate instance

        Example:
            TagSet = Set.template(key_size=50)
            user_tags = db.create_dict('user_tags', TagSet)
        """
        return SetTemplate(
            cls, key_size=key_size, use_bloom=use_bloom, cache_size=cache_size
        )

    @classmethod
    def _get_nested_ref_schema(cls):
        """Compact binary schema for nested Set references.

        Since Set wraps Dict internally, we store the Dict's state.
        """
        return {
            # Core metadata
            "size": "uint32",  # 4 bytes
            "p_last": "uint8",  # 1 byte
            "p_init": "uint8",  # 1 byte
            "next_data_offset": "uint32",  # 4 bytes
            "values_block_addr": "uint64",  # 8 bytes
            # 8 table addresses × 8 bytes = 64 bytes
            "table_0": "uint64",
            "table_1": "uint64",
            "table_2": "uint64",
            "table_3": "uint64",
            "table_4": "uint64",
            "table_5": "uint64",
            "table_6": "uint64",
            "table_7": "uint64",
            # Config
            "key_size": "uint16",  # 2 bytes
            "cache_size": "uint16",  # 2 bytes
        }

    @classmethod
    def get_shared_dataset_specs(cls, parent_name, inner_schema=None):
        """Get specifications for shared datasets needed when Set is nested.

        Set uses a Dict internally, which needs hash table and values datasets.
        The inner_schema is ignored since Set uses a fixed dummy schema.

        Args:
            parent_name: Name of the parent container
            inner_schema: Ignored (Set uses fixed dummy schema)

        Returns:
            Dict with shared dataset specifications
        """
        return {
            "_shared_hash_table": {
                "name": f"_{parent_name}_shared_hash",
                "schema": {
                    "hash": "uint64",
                    "key": "U100",  # Fixed key size for shared table
                    "value_addr": "uint64",
                    "valid": "bool",
                },
            },
            "_shared_values_dataset": {
                "name": f"_{parent_name}_shared_values",
                "schema": cls._DUMMY_SCHEMA,
            },
        }

    def __init__(
        self,
        name: str,
        db,
        key_size: int = 50,
        use_bloom: bool = True,
        cache_size: int = 0,  # No cache needed for sets
        _load_existing: bool = False,
        _parent=None,
    ):
        """Initialize a Set.

        Args:
            name: Unique name for this set
            db: Database instance
            key_size: Maximum length of string keys (default: 50)
            use_bloom: Whether to use bloom filter for fast lookups (default: True)
            cache_size: LRU cache size (default: 0, disabled for sets)
            _load_existing: If True, load existing set from disk
            _parent: Parent data structure if this is nested (internal use)
        """
        self._key_size = key_size
        self._use_bloom = use_bloom
        self._cache_size = cache_size
        self._parent_key = None  # Key in parent dict (set by parent for Dict[Set])
        self._dict = None  # Will be set by _initialize or _initialize_nested

        super().__init__(name, db, _parent=_parent)

        if _load_existing:
            self._load()
        elif _parent is not None and _parent != "__nested__":
            # Nested set: use parent's shared datasets
            self._initialize_nested()
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

    def _initialize_nested(self):
        """Initialize a nested set using parent's shared datasets."""
        # For nested sets, we create an internal Dict that uses the parent's shared datasets
        # The Dict will be initialized with the shared datasets from the parent
        self._shared_hash_table = self._parent._shared_hash_table
        self._shared_values_dataset = self._parent._shared_values_dataset

        # Create internal dict that uses parent's shared datasets
        self._dict = Dict(
            name=f"_set_{self.name}_dict",
            db=self._db,
            dataset_or_template=self._shared_values_dataset,
            key_size=self._key_size,
            use_bloom=self._use_bloom,
            cache_size=self._cache_size,
            _parent=self._parent,  # Pass parent so Dict uses shared datasets
        )

    # ========== Nested Structure Support ==========

    def set_shared_datasets(self, shared_datasets):
        """Set shared datasets on this nested Set instance."""
        if "_shared_hash_table" in shared_datasets:
            self._shared_hash_table = shared_datasets["_shared_hash_table"]
        if "_shared_values_dataset" in shared_datasets:
            self._shared_values_dataset = shared_datasets["_shared_values_dataset"]

        # Also set on internal dict if it exists
        if self._dict is not None:
            self._dict.set_shared_datasets(shared_datasets)

    def needs_shared_datasets(self):
        """Check if this nested Set needs shared datasets."""
        return (
            not hasattr(self, "_shared_hash_table") or self._shared_hash_table is None
        )

    def to_ref(self):
        """Get reference to this set for storage in parent.

        For nested sets, uses compact binary format.
        """
        if self._parent is not None and self._parent != "__nested__":
            # Nested set: use compact binary format (delegate to internal dict)
            ref = {"valid": True}

            # Get internal dict's state
            ref["size"] = self._dict.size
            ref["p_last"] = self._dict.p_last
            ref["p_init"] = self._dict._p_init
            ref["next_data_offset"] = self._dict.next_data_offset
            ref["values_block_addr"] = self._dict.values_block_addr

            # Table addresses (up to 8 for nested)
            for i in range(self.MAX_TABLES_NESTED):
                ref[f"table_{i}"] = (
                    self._dict.table_addrs[i] if i < len(self._dict.table_addrs) else 0
                )

            # Config
            ref["key_size"] = self._key_size
            ref["cache_size"] = self._cache_size

            return ref
        else:
            # Top-level: use base class implementation
            return super().to_ref()

    @classmethod
    def _from_ref_impl(cls, db, ref):
        """Reconstruct Set from reference."""
        # Check if this is binary format (nested set) or standard format (top-level)
        is_binary_format = "size" in ref and "table_0" in ref

        if is_binary_format:
            # Binary format: reconstruct from compact fields
            instance = object.__new__(cls)

            # Set basic attributes
            instance.name = f"_nested_set_{id(instance)}"
            instance._db = db
            instance._parent = "__nested__"
            instance._key_size = int(ref.get("key_size", 50))
            instance._use_bloom = False  # No bloom for nested
            instance._cache_size = int(ref.get("cache_size", 0))
            instance._parent_key = None
            instance._auto_save_interval = 0
            instance._ops_since_save = 0
            instance._inline_metadata = None
            instance._metadata_key = None

            # Create internal Dict from the reference
            # We need to reconstruct the Dict's state
            dict_instance = object.__new__(Dict)
            dict_instance.name = f"_set_{instance.name}_dict"
            dict_instance._db = db
            dict_instance._parent = "__nested__"
            dict_instance._is_nested = False
            dict_instance._template = None
            dict_instance.size = int(ref["size"])
            dict_instance.p_last = int(ref["p_last"])
            dict_instance._p_init = int(ref.get("p_init", Dict.P_INIT))
            dict_instance.next_data_offset = int(ref["next_data_offset"])
            dict_instance.values_block_addr = int(ref["values_block_addr"])
            dict_instance.values_capacity = 10000  # Default

            # Reconstruct table_addrs
            dict_instance.table_addrs = []
            for i in range(cls.MAX_TABLES_NESTED):
                addr = int(ref[f"table_{i}"])
                if addr > 0 or i == 0:
                    dict_instance.table_addrs.append(addr)
                else:
                    break

            dict_instance.cache_size = instance._cache_size
            dict_instance._cache = (
                LRUCache(dict_instance.cache_size)
                if dict_instance.cache_size > 0
                else None
            )
            dict_instance._blooms = []
            dict_instance.use_bloom = False
            dict_instance._auto_save_interval = 0
            dict_instance._ops_since_save = 0
            dict_instance._inline_metadata = None
            dict_instance._metadata_key = None
            dict_instance._shared_datasets = {}

            # These will be set by set_shared_datasets
            dict_instance._hash_table = None
            dict_instance._values_dataset = None
            dict_instance.item_schema = None

            instance._dict = dict_instance
            instance._shared_hash_table = None
            instance._shared_values_dataset = None

            return instance

        # Standard format (top-level)
        ds_name = ref["ds_name"]
        if ds_name in db._datastructures:
            return db._datastructures[ds_name]

        return cls(
            ds_name,
            db,
            key_size=ref.get("key_size", 50),
            use_bloom=ref.get("use_bloom", True),
            cache_size=ref.get("cache_size", 0),
            _load_existing=True,
        )

    # ========== Core Set Operations ==========

    def _update_parent_ref(self):
        """Update our reference in parent container after modification."""
        if (
            self._parent is not None
            and self._parent != "__nested__"
            and self._parent_key is not None
        ):
            self._parent.update_nested_ref(self._parent_key, self)

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

        # Update reference in parent if nested
        self._update_parent_ref()

    def remove(self, item):
        """Remove an item from the set.

        Args:
            item: Item to remove

        Raises:
            KeyError: If item not in set
        """
        del self._dict[item]
        self._update_parent_ref()

    def discard(self, item):
        """Remove an item if present.

        Args:
            item: Item to remove

        No error if item not present.
        """
        try:
            del self._dict[item]
            self._update_parent_ref()
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
            self.remove(item)  # remove already calls _update_parent_ref
            return item
        raise KeyError("pop from an empty set")

    def clear(self):
        """Remove all items from the set."""
        # Delete all items
        for item in list(self.keys()):
            self.discard(item)  # discard already calls _update_parent_ref

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

    # ---- Registry protocol ----

    def _get_registry_params(self):
        return {
            "key_size": self._key_size,
            "use_bloom": self._use_bloom,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db,
            key_size=params.get("key_size", 50),
            use_bloom=params.get("use_bloom", True),
            _load_existing=True,
        )

    # ---- Template reconstruction protocol ----

    @classmethod
    def _reconstruct_template(cls, db, template_config, template_class_name):
        """Set uses SetTemplate which doesn't need a real dataset."""
        return SetTemplate(
            cls,
            key_size=template_config.get("key_size", 50),
            use_bloom=template_config.get("use_bloom", False),
            cache_size=template_config.get("cache_size", 0),
        )


class SetTemplate(DataStructureTemplate):
    """Template for creating nested Sets.

    Inherits from DataStructureTemplate but doesn't require a dataset
    since Sets only store keys (strings), not structured data.

    Example:
        TagSet = Set.template(key_size=50)
        user_tags = db.create_dict('user_tags', TagSet)
    """

    def __init__(self, ds_class, key_size=50, use_bloom=False, cache_size=0):
        """Initialize template.

        Args:
            ds_class: Set class
            key_size: Maximum length of string keys
            use_bloom: Whether to use bloom filter
            cache_size: LRU cache size
        """
        # Create a dummy dataset for compatibility with parent class
        dummy_dataset = _DummyDataset(Set._DUMMY_SCHEMA)

        config = {
            "key_size": key_size,
            "use_bloom": use_bloom,
            "cache_size": cache_size,
        }

        # Call parent constructor
        super().__init__(ds_class, dummy_dataset, config)

    def new(self, db, name=None, **kwargs):
        """Create new Set instance from this template.

        Args:
            db: Database instance
            name: Optional name (auto-generated if None)
            **kwargs: Additional arguments (e.g., _parent)

        Returns:
            New Set instance
        """
        if name is None:
            name = f"_Set_{id(self)}_{self._counter}"
            self._counter += 1

        all_config = {**self.config, **kwargs}
        return self.ds_class(name, db, **all_config)

    def __repr__(self):
        return f"SetTemplate(key_size={self.config['key_size']})"


class _DummyDataset:
    """Dummy dataset placeholder for SetTemplate compatibility.

    The template system expects a dataset attribute, but Set doesn't
    need one since it only stores keys. This provides the minimal
    interface needed.
    """

    def __init__(self, schema):
        self.name = "_set_dummy"
        self._schema = schema

    @property
    def user_schema(self):
        """Return a minimal schema-like object."""
        return _DummySchema(self._schema)


class _DummySchema:
    """Minimal schema interface for _DummyDataset."""

    def __init__(self, schema_dict):
        self._schema = schema_dict
        self.names = list(schema_dict.keys())
        self.fields = {name: (dtype, 0) for name, dtype in schema_dict.items()}
