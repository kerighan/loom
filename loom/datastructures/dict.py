"""Persistent dictionary with multi-table exponential growth."""

import struct
import mmh3
from loom.datastructures.base import DataStructure
from loom.datastructures.template import DataStructureTemplate
from loom.datastructures.counting_bloomfilter import CountingBloomFilter
from loom.cache import LRUCache  # kept for Set.template usage in this module
from loom.ref import Ref


class Dict(DataStructure):
    """Persistent dictionary with exponential multi-table growth."""

    GROWTH_FACTOR = 2
    P_INIT = 10
    MAX_TABLES = 32
    MAX_TABLES_NESTED = 8  # Nested dicts use fewer tables (still ~65K items)
    PROBE_FACTOR = 0.5

    # Nesting compatibility
    _outer_types_supported = ("Dict", "List", "BTree")  # Dict[Dict], List[Dict], BTree[Dict]
    _inner_types_supported = ("List", "Dict", "Set", "BTree", "Queue")

    @classmethod
    def _get_ref_config_schema(cls):
        return {"cache_size": "uint32", "use_bloom": "bool", "p_init": "uint32"}

    @classmethod
    def get_shared_dataset_specs(cls, parent_name, inner_schema, key_size=50):
        """Get specifications for shared datasets needed when Dict is nested.

        When Dict is used as inner type (e.g., List[Dict] or Dict[Dict]),
        the parent needs shared hash table and values datasets.

        Args:
            parent_name: Name of the parent container
            inner_schema: Schema dict for dict values
            key_size: Max key length for inner dict keys (default 50)

        Returns:
            Dict with shared dataset specifications
        """
        return {
            "_shared_hash_table": {
                "name": f"_{parent_name}_shared_hash",
                "schema": {
                    "hash": "uint64",
                    "key": f"U{key_size}",
                    "value_addr": "uint64",
                    "valid": "bool",
                },
            },
            "_shared_values_dataset": {
                "name": f"_{parent_name}_shared_values",
                "schema": inner_schema,
            },
        }

    def set_shared_datasets(self, shared_datasets):
        """Set shared datasets on this nested Dict instance."""
        if "_shared_hash_table" in shared_datasets:
            self._hash_table = shared_datasets["_shared_hash_table"]
        if "_shared_values_dataset" in shared_datasets:
            self._values_dataset = shared_datasets["_shared_values_dataset"]
            self.item_schema = self._extract_schema(self._values_dataset)

    def needs_shared_datasets(self):
        """Check if this nested Dict needs shared datasets."""
        return self._hash_table is None

    @classmethod
    def _get_nested_ref_schema(cls):
        """Compact binary schema for nested Dict references.

        Total size: ~150 bytes per reference (vs ~2000+ bytes with JSON)
        """
        return {
            # Core metadata
            "size": "uint32",  # 4 bytes
            "p_last": "uint8",  # 1 byte
            "p_init": "uint8",  # 1 byte - initial p value (may differ for nested)
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
            "cache_size": "uint16",  # 2 bytes
        }
        # Total: 1(valid) + 4 + 1 + 1 + 4 + 8 + 64 + 2 = 85 bytes

    # Default capacities - can be tuned for storage vs performance
    DEFAULT_KEY_SIZE = 50  # Max key length in characters (was 100)
    DEFAULT_INITIAL_CAPACITY = 100000  # Initial values block size

    def __init__(
        self,
        name,
        db,
        dataset_or_template,
        cache_size=1000,
        use_bloom=True,
        auto_save_interval=None,
        key_size=None,
        initial_capacity=None,
        hash_keys=False,
        hash_bits=128,
        _parent=None,
    ):
        self.cache_size = cache_size
        self.use_bloom = use_bloom
        self._key_size = key_size or self.DEFAULT_KEY_SIZE
        self._hash_keys = hash_keys
        self._hash_bits = hash_bits
        if hash_keys:
            # Each hash stored as hex → bits/4 hex chars
            hex_chars = hash_bits // 4
            self._key_size = hex_chars
            self._hash_key_fn = self._make_hash_fn(hex_chars)
        else:
            self._hash_key_fn = None
        self._initial_capacity = initial_capacity or self.DEFAULT_INITIAL_CAPACITY
        self._parent = _parent  # Parent Dict if this is nested
        self._parent_key = None  # Key in parent dict (set by parent)

        if isinstance(dataset_or_template, DataStructureTemplate):
            self._template = dataset_or_template
            self._is_nested = True
            self.item_schema = None
        elif dataset_or_template is not None:
            self._template = None
            self._is_nested = False
            self._user_dataset = dataset_or_template
            self.item_schema = (
                dataset_or_template.schema
                if hasattr(dataset_or_template, "schema")
                else dataset_or_template
            )
        else:
            # Loading existing dict
            self._template = None
            self._is_nested = False
            self.item_schema = None

        super().__init__(name, db, auto_save_interval, _parent=_parent)

        metadata = self._load_metadata()
        if metadata:
            self._load()
        else:
            self._initialize()

        self._cache = self._make_cache("values", cache_size)

    def _alloc_value_addr(self):
        """Allocate a value address, reusing freed slots first."""
        if hasattr(self, "_value_freelist") and self._value_freelist:
            return self._value_freelist.pop()
        addr = (
            self.values_block_addr
            + self.next_data_offset * self._values_dataset.record_size
        )
        self.next_data_offset += 1
        return addr

    @staticmethod
    def _make_hash_fn(hex_chars):
        import hashlib
        def _hk(key):
            return hashlib.sha256(str(key).encode()).hexdigest()[:hex_chars]
        return _hk

    def _to_internal_key(self, key):
        """Apply hash function if hash_keys=True, else return key unchanged."""
        if self._hash_key_fn is not None:
            return self._hash_key_fn(key)
        return key

    def _get_capacity(self, p):
        return self.GROWTH_FACTOR**p

    def _get_probe_range(self, p):
        return int(round(p * self.PROBE_FACTOR * self.GROWTH_FACTOR))

    def _hash(self, key, seed=0):
        if not isinstance(key, str):
            key = str(key)
        return mmh3.hash(key, seed=seed, signed=False)

    def _load_table_addresses(self):
        """Load table addresses from metadata (already loaded in _load)."""
        # Table addresses are already loaded in _load() from metadata
        # This method exists for compatibility with base class
        pass

    def _initialize(self):
        # Check if this is a nested dict that should share parent's datasets
        if self._parent is not None and self._parent != "__nested__":
            # Nested dict: use parent's shared datasets
            # Parent can be either a Dict (Dict[Dict]) or a List (List[Dict])
            self._hash_table = self._parent._shared_hash_table
            self._values_dataset = self._parent._shared_values_dataset
            self.item_schema = self._extract_schema(self._values_dataset)

            # Allocate our own block in the shared hash table.
            # Start very small (8 slots, 2^3) — most nodes in real-world graphs
            # have low degree (power-law). Grows on demand via _create_new_table.
            self._p_init = 3  # 2^3 = 8 slots
            capacity = self._get_capacity(self._p_init)
            first_table_addr = self._hash_table.allocate_block(capacity)

            # Same for the values block: start small, grow on demand
            initial_values_capacity = 8
            self.values_block_addr = self._values_dataset.allocate_block(
                initial_values_capacity
            )
            self.values_capacity = initial_values_capacity

            # No bloom filters for nested dicts
            self._blooms = []
            self.use_bloom = False

            # No shared datasets (we're a child, not a container)
            self._shared_datasets = {}
        else:
            # Top-level dict: create our own datasets
            self._hash_table = self._db.create_dataset(
                f"_dict_{self.name}_hashtable",
                hash="uint64",
                key=f"U{self._key_size}",  # Configurable key size
                value_addr="uint64",
                valid="bool",
            )

            # For nested dict containers, create shared datasets for children
            # For regular dicts, use the user's dataset directly (compact!)
            if self._is_nested:
                # Validate nesting compatibility
                inner_class = self._template.ds_class
                inner_class._check_nesting(type(self))

                ref_schema = self._template.get_ref_schema()
                self._values_dataset = self._db.create_dataset(
                    f"_dict_{self.name}_values", **ref_schema
                )
                self.item_schema = ref_schema

                # Use modular approach: ask inner type what shared datasets it needs
                inner_schema = self._extract_schema(self._template.dataset)
                shared_specs = inner_class.get_shared_dataset_specs(
                    f"dict_{self.name}", inner_schema,
                    key_size=self._key_size,
                )

                # Create the shared datasets
                self._shared_datasets = {}
                for attr_name, spec in shared_specs.items():
                    dataset = self._db.create_dataset(spec["name"], **spec["schema"])
                    setattr(self, attr_name, dataset)
                    self._shared_datasets[attr_name] = dataset
            else:
                # Compact version: use user's dataset directly!
                self._values_dataset = self._user_dataset
                self.item_schema = self._extract_schema(self._user_dataset)
                self._shared_datasets = {}  # No shared datasets for regular dicts

            self._p_init = self.P_INIT  # Use class constant for top-level
            capacity = self._get_capacity(self._p_init)
            first_table_addr = self._hash_table.allocate_block(capacity)

            # Allocate initial block in values dataset for compact storage
            self.values_block_addr = self._values_dataset.allocate_block(
                self._initial_capacity
            )
            self.values_capacity = self._initial_capacity

        # Track next free slot within the block
        self.next_data_offset = 0
        self._value_freelist = []

        self.p_last = self._p_init
        self.size = 0
        self.table_addrs = [int(first_table_addr)]

        # Per-table bloom filters for fast lookups
        # Each table has its own bloom filter to quickly check if a key might be there
        self._blooms = []
        if self.use_bloom and not self._parent:
            # Create counting bloom filter for the first table (supports delete!)
            bloom = CountingBloomFilter(
                f"{self.name}_bloom_0",
                self._db,
                expected_items=capacity,
                false_positive_rate=0.01,
            )
            self._blooms.append(bloom)

        # Save metadata on initialization (force=True to save even if nested)
        # Nested dicts need their metadata saved so they can be reloaded
        self.save(force=True)

    def _load(self):
        metadata = self._load_metadata()
        self.p_last = metadata["p_last"]
        self._p_init = metadata.get("p_init", self.P_INIT)  # Fallback for old data
        self.size = metadata["size"]
        self.values_block_addr = metadata.get("values_block_addr", 0)
        self.values_capacity = metadata.get("values_capacity", 0)
        self.next_data_offset = metadata.get("next_data_offset", 0)
        self._value_freelist = metadata.get("value_freelist", [])
        self.use_bloom = metadata["use_bloom"]
        self._is_nested = metadata.get("is_nested", False)
        self._key_size = metadata.get("key_size", self.DEFAULT_KEY_SIZE)
        self._hash_keys = metadata.get("hash_keys", False)
        self._hash_bits = metadata.get("hash_bits", 128)
        if self._hash_keys:
            self._hash_key_fn = self._make_hash_fn(self._hash_bits // 4)
        else:
            self._hash_key_fn = None
        self._initial_capacity = metadata.get(
            "initial_capacity", self.DEFAULT_INITIAL_CAPACITY
        )
        self.table_addrs = metadata.get(
            "table_addrs", []
        )  # Load early to prevent AttributeError

        self._hash_table = self._get_dataset(metadata["hash_table_name"])
        self._values_dataset = self._get_dataset(metadata["values_dataset_name"])

        # Load per-table counting bloom filters
        self._blooms = []
        if self.use_bloom and "bloom_names" in metadata:
            for bloom_name in metadata["bloom_names"]:
                bloom = self._db._datastructures.get(bloom_name) or CountingBloomFilter(
                    bloom_name, self._db
                )
                self._blooms.append(bloom)
        elif self.use_bloom and "bloom_name" in metadata:
            # Legacy: single bloom filter - convert to list
            bloom_name = metadata["bloom_name"]
            bloom = self._db._datastructures.get(bloom_name) or CountingBloomFilter(
                bloom_name, self._db
            )
            self._blooms.append(bloom)
        elif not self.use_bloom:
            self._blooms = []

        if self._is_nested:
            from loom.datastructures.base import _DS_REGISTRY

            template_config = metadata["template_config"]
            template_class_name = metadata.get("template_class", "Dict")

            # Get template class from registry (modular approach)
            template_class = _DS_REGISTRY.get(template_class_name, Dict)

            # Reconstruct template via class protocol (no per-type switch)
            full_config = {**template_config, "_template_dataset": metadata.get("template_dataset")}
            self._template = template_class._reconstruct_template(
                self._db, full_config, template_class_name
            )
            self.item_schema = self._template.get_ref_schema()

            # Load shared datasets (modular approach)
            self._shared_datasets = {}
            shared_datasets_meta = metadata.get("shared_datasets", {})

            # Handle legacy format for backward compatibility
            if not shared_datasets_meta:
                if "shared_items_dataset_name" in metadata:
                    shared_datasets_meta["_shared_items_dataset"] = metadata[
                        "shared_items_dataset_name"
                    ]
                elif "shared_hash_table_name" in metadata:
                    shared_datasets_meta["_shared_hash_table"] = metadata[
                        "shared_hash_table_name"
                    ]
                    shared_datasets_meta["_shared_values_dataset"] = metadata[
                        "shared_values_dataset_name"
                    ]

            for attr_name, dataset_name in shared_datasets_meta.items():
                dataset = self._get_dataset(dataset_name)
                setattr(self, attr_name, dataset)
                self._shared_datasets[attr_name] = dataset
        else:
            self.item_schema = {
                name: str(self._values_dataset.user_schema.fields[name][0])
                for name in self._values_dataset.user_schema.names
            }
            self._shared_datasets = {}

        self._load_table_addresses()

    def _get_ref_fields(self):
        """Get Dict-specific reference fields (for top-level dicts)."""
        return {
            "ref_dataset_name": self._values_dataset.name,
            "cache_size": self.cache_size,
            "use_bloom": self.use_bloom,
            "p_init": self.P_INIT,
        }

    def to_ref(self):
        """Get reference to this dict for storage in parent.

        For nested dicts, uses compact binary format (~84 bytes).
        For top-level dicts, uses standard format with metadata key.
        """
        if self._parent is not None and self._parent != "__nested__":
            # Nested dict: use compact binary format
            ref = {"valid": True}

            # Core metadata as binary fields
            ref["size"] = self.size
            ref["p_last"] = self.p_last
            ref["p_init"] = self._p_init
            ref["next_data_offset"] = self.next_data_offset
            ref["values_block_addr"] = self.values_block_addr

            # Table addresses (up to 8 for nested)
            for i in range(self.MAX_TABLES_NESTED):
                ref[f"table_{i}"] = (
                    self.table_addrs[i] if i < len(self.table_addrs) else 0
                )

            # Config
            ref["cache_size"] = self.cache_size

            return ref
        else:
            # Top-level: use base class implementation
            return super().to_ref()

    @classmethod
    def _from_ref_impl(cls, db, ref):
        """Reconstruct Dict from reference."""
        # Check if this is binary format (nested dict) or JSON format (top-level)
        is_binary_format = "size" in ref and "table_0" in ref

        if is_binary_format:
            # Binary format: reconstruct from compact fields
            instance = object.__new__(cls)

            # Extract metadata from binary fields
            instance.size = int(ref["size"])
            instance.p_last = int(ref["p_last"])
            instance._p_init = int(
                ref.get("p_init", cls.P_INIT)
            )  # Fallback for old refs
            instance.next_data_offset = int(ref["next_data_offset"])
            instance.values_block_addr = int(ref["values_block_addr"])

            # Reconstruct table_addrs array
            instance.table_addrs = []
            for i in range(cls.MAX_TABLES_NESTED):
                addr = int(ref[f"table_{i}"])
                if addr > 0 or i == 0:  # Always include first table
                    instance.table_addrs.append(addr)
                else:
                    break

            # Set other required attributes
            instance.name = f"_nested_dict_{id(instance)}"
            instance._db = db
            instance._parent = "__nested__"
            instance._is_nested = False  # This nested dict stores data, not refs
            instance._template = None
            instance._shared_hash_table = None
            instance._shared_values_dataset = None
            instance.cache_size = int(ref["cache_size"])
            instance._cache = instance._make_cache("values", instance.cache_size)
            instance._blooms = []
            instance.use_bloom = False
            instance._auto_save_interval = 0
            instance._ops_since_save = 0
            instance._inline_metadata = None
            instance._metadata_key = None
            instance.values_capacity = 10000  # Default for nested
            # hash_keys not supported for nested dicts
            instance._hash_keys = False
            instance._hash_bits = 128
            instance._hash_key_fn = None

            # These will be set by the parent when reconstructing
            # For now, they need to be passed somehow - we'll get them from parent
            instance._hash_table = None
            instance._values_dataset = None
            instance.item_schema = None

            return instance

        # JSON format (top-level)
        ds_name = ref["ds_name"]
        if ds_name in db._datastructures:
            return db._datastructures[ds_name]

        return cls(
            ds_name,
            db,
            None,  # None triggers load from metadata
            cache_size=ref["cache_size"],
            use_bloom=ref["use_bloom"],
        )

    def save(self, force=False):
        """Save Dict metadata.

        Args:
            force: If True, save even during parent operations
        """
        # Skip auto-save during parent operations (but allow explicit save)
        if self._parent and not force:
            return

        metadata = {
            "p_last": self.p_last,
            "p_init": self._p_init,
            "size": self.size,
            "values_block_addr": self.values_block_addr,
            "values_capacity": self.values_capacity,
            "next_data_offset": self.next_data_offset,
            "table_addrs": self.table_addrs,
            "hash_table_name": self._hash_table.name,
            "values_dataset_name": self._values_dataset.name,
            "use_bloom": self.use_bloom,
            "is_nested": self._is_nested,
            "key_size": self._key_size,
            "initial_capacity": self._initial_capacity,
            "value_freelist": getattr(self, "_value_freelist", []),
            "hash_keys": getattr(self, "_hash_keys", False),
            "hash_bits": getattr(self, "_hash_bits", 128),
        }
        # Save per-table bloom filter names
        if self.use_bloom and self._blooms:
            metadata["bloom_names"] = [b.name for b in self._blooms]
        if self._is_nested and self._template:
            metadata["template_dataset"] = self._template.dataset.name
            metadata["template_config"] = self._template.config
            metadata["template_class"] = self._template.ds_class.__name__

            # Save shared dataset names (modular approach)
            if hasattr(self, "_shared_datasets") and self._shared_datasets:
                metadata["shared_datasets"] = {
                    attr_name: dataset.name
                    for attr_name, dataset in self._shared_datasets.items()
                }
        self._save_metadata(metadata)

    def update_nested_ref(self, key, nested_item):
        """Update the stored reference for a nested structure after modification.

        Args:
            key: The key in this dict where the nested structure is stored
            nested_item: The nested Dict or List instance with updated state
        """
        if not self._is_nested:
            return

        # Find the entry for this key
        key_hash = self._hash(key)
        try:
            p, table_addr, position, entry = self._find_slot(
                key, key_hash, for_insert=False
            )
            value_addr = int(entry["value_addr"])

            # Update the stored reference with current state
            self._values_dataset[value_addr] = nested_item.to_ref()

            # Invalidate cache to ensure fresh data on next access
            if self._cache:
                self._cache.invalidate(key)
        except KeyError:
            pass  # Key not found, nothing to update

    def _find_slot(self, key, key_hash, for_insert=False):
        """Find slot for key using per-table bloom filters.

        For lookups: use bloom filters to skip tables that don't have the key.
        For inserts: first check if key exists (for update), then find free slot in last table.
        """
        p_init = getattr(self, "_p_init", self.P_INIT)

        # Search from largest table (p_last) to smallest (p_init)
        # Use per-table bloom filters to skip tables that definitely don't have the key
        for p in range(self.p_last, p_init - 1, -1):
            table_idx = p - p_init

            # Use bloom filter to skip tables
            if self._blooms and table_idx < len(self._blooms):
                if key not in self._blooms[table_idx]:
                    # Key definitely not in this table - skip it
                    continue

            result = self._find_slot_in_table(key, key_hash, p, for_insert=False)
            if result is not None:
                p_res, table_addr, position, entry = result
                if entry.get("valid", False) and entry.get("key") == key:
                    # Found the key - return it
                    return result

        # Key not found in any table
        if for_insert:
            # Try all tables from newest to oldest — reuse deleted slots
            for p in range(self.p_last, p_init - 1, -1):
                result = self._find_slot_in_table(key, key_hash, p, for_insert=True)
                if result is not None:
                    return result
            return None  # All tables full → caller creates a new table
        else:
            raise KeyError(key)

    def _find_slot_in_table(self, key, key_hash, p, for_insert):
        """Find slot in a specific table."""
        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        probe_range = self._get_probe_range(p)
        p_init = getattr(self, "_p_init", self.P_INIT)
        table_idx = p - p_init
        table_addr = self.table_addrs[table_idx]

        record_size = self._hash_table.record_size
        first_free = None
        slots_to_read = min(probe_range, capacity)

        # Field offsets in hash table record:
        # _prefix: 1 byte at offset 0
        # hash: 8 bytes at offset 1
        # key: key_size*4 bytes at offset 9
        # value_addr: 8 bytes at offset 9 + key_size*4
        # valid: 1 byte at offset 9 + key_size*4 + 8
        # Get actual key size from hash table schema (more reliable than _key_size)
        # Schema format: [('_prefix', 'i1'), ('hash', '<u8'), ('key', '<U50'), ...]
        key_dtype = str(self._hash_table.schema[2])  # e.g., '<U50' or 'U100'
        key_chars = int(key_dtype.replace("<U", "").replace("U", ""))
        valid_offset = 9 + key_chars * 4 + 8  # offset of valid field

        # Check if we wrap around
        if bucket + slots_to_read <= capacity:
            # No wraparound - read contiguous block
            start_addr = table_addr + bucket * record_size
            chunk_size = slots_to_read * record_size
            try:
                chunk_data = self._hash_table.db.read(start_addr, chunk_size)

                # OPTIMIZATION: Check hash and valid first before full deserialize
                for i in range(slots_to_read):
                    position = bucket + i
                    offset = i * record_size

                    # Quick check: valid byte (0 = invalid/empty)
                    valid = chunk_data[offset + valid_offset]
                    if not valid:
                        if for_insert and first_free is None:
                            first_free = (p, table_addr, position, {})
                        continue

                    # Quick check: hash match (8 bytes at offset 1)
                    entry_hash = struct.unpack_from("<Q", chunk_data, offset + 1)[0]
                    if entry_hash != key_hash:
                        continue

                    # Hash matches - now deserialize full entry to check key
                    entry_data = chunk_data[offset : offset + record_size]
                    try:
                        entry = self._hash_table._deserialize(entry_data)
                    except Exception:
                        continue

                    if entry["key"] == key:
                        return (p, table_addr, position, entry)

            except Exception:
                # Fallback to individual reads if batch read fails
                pass
        else:
            # Wraparound case - fall back to individual reads
            # (Could optimize this too, but wraparound is rare)
            for i in range(slots_to_read):
                position = (bucket + i) % capacity
                entry_addr = table_addr + position * record_size

                try:
                    entry = self._hash_table[entry_addr]
                except Exception:
                    if for_insert and first_free is None:
                        first_free = (p, table_addr, position, {})
                    continue

                if not entry.get("valid", False):
                    if for_insert and first_free is None:
                        first_free = (p, table_addr, position, entry)
                    continue

                if entry["hash"] == key_hash and entry["key"] == key:
                    return (p, table_addr, position, entry)

        if for_insert and first_free is not None:
            return first_free
        return None

    def _create_new_table(self):
        self.p_last += 1
        capacity = self._get_capacity(self.p_last)
        new_table_addr = self._hash_table.allocate_block(capacity)
        self.table_addrs.append(int(new_table_addr))

        # Create counting bloom filter for the new table (supports delete!)
        if self.use_bloom and not self._parent:
            table_idx = len(self.table_addrs) - 1
            bloom = CountingBloomFilter(
                f"{self.name}_bloom_{table_idx}",
                self._db,
                expected_items=capacity,
                false_positive_rate=0.01,
            )
            self._blooms.append(bloom)

        self.save()

    def __setitem__(self, key, value, atomic=False):
        """Set value for key.

        Args:
            key: Key to set
            value: Value to store (dict for regular, Dict for nested)
            atomic: If True, use WAL for crash safety (default: False)
        """
        if atomic:
            self._setitem_atomic(key, value)
        else:
            self._setitem_fast(key, value)

    def _setitem_fast(self, key, value):
        """Fast path for non-atomic insert/update (current implementation)."""
        key = self._to_internal_key(key)
        key_hash = self._hash(key)
        slot = self._find_slot(key, key_hash, for_insert=True)

        if slot is None:
            self._create_new_table()
            slot = self._find_slot(key, key_hash, for_insert=True)

        p, table_addr, position, entry = slot
        entry_addr = table_addr + position * self._hash_table.record_size
        is_update = entry.get("valid", False) and entry.get("key") == key

        is_ref_value = isinstance(value, Ref)
        if is_ref_value and self._is_nested:
            raise TypeError("Cannot store Ref values in nested Dict")
        if is_ref_value and value.dataset is not self._values_dataset:
            raise TypeError(
                "Ref dataset mismatch: Ref must point to this Dict's values dataset"
            )

        if is_update:
            # Update existing item
            value_addr = int(entry["value_addr"])
            if self._is_nested:
                # Modular: use template's ds_class for type checking
                expected_type = self._template.ds_class
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Expected {expected_type.__name__}, got {type(value)}"
                    )
                self._values_dataset[value_addr] = value.to_ref()
            else:
                if is_ref_value:
                    # Re-point existing entry to Ref address (no copy)
                    value_addr = int(value.addr)
                    entry["value_addr"] = value_addr
                    self._hash_table[entry_addr] = entry
                else:
                    # Update in place in user's dataset
                    self._values_dataset[value_addr] = value
        else:
            # Insert new item - use append() to get next free address
            if self._is_nested:
                # Modular: use template's ds_class for type checking and creation
                expected_type = self._template.ds_class
                if value is None:
                    value = self._template.new(self._db, _parent=self)
                    value._parent_key = key  # Track key for update_nested_ref
                    # NOT registered in _datastructures — nested structures
                    # are managed by their parent, cached in LRU
                elif not isinstance(value, expected_type):
                    raise TypeError(
                        f"Expected {expected_type.__name__}, got {type(value)}"
                    )
                ref = value.to_ref()
                value_addr = self._alloc_value_addr()
                self._values_dataset[value_addr] = ref
            else:
                if is_ref_value:
                    # Store pointer directly (no allocation/copy)
                    value_addr = int(value.addr)
                else:
                    value_addr = self._alloc_value_addr()
                    self._values_dataset[value_addr] = value

            self._hash_table[entry_addr] = {
                "hash": key_hash,
                "key": key,
                "value_addr": value_addr,
                "valid": True,
            }
            # Add to the correct table's bloom filter
            p_init = getattr(self, "_p_init", self.P_INIT)
            table_idx = p - p_init
            if self._blooms and table_idx < len(self._blooms):
                self._blooms[table_idx].add(key)
            self.size += 1

        # Cache the actual value (not the address!)
        if self._cache:
            self._cache[key] = value if self._is_nested else int(value_addr)

        self._auto_save_check()

        # If this is a nested dict, update our reference in parent
        if (
            self._parent is not None
            and self._parent != "__nested__"
            and self._parent_key is not None
        ):
            self._parent.update_nested_ref(self._parent_key, self)

        return value if self._is_nested else None

    def get_ref(self, key):
        """Return a Ref handle to the underlying record for a key.

        Only supported for non-nested dicts.
        """
        if self._is_nested:
            raise TypeError("get_ref is not supported for nested Dict")

        key_hash = self._hash(key)
        p, table_addr, position, entry = self._find_slot(
            key, key_hash, for_insert=False
        )
        value_addr = int(entry["value_addr"])
        return Ref(self._values_dataset, value_addr)

    def _setitem_atomic(self, key, value):
        """Atomic insert/update using WAL for crash safety."""
        key = self._to_internal_key(key)
        key_hash = self._hash(key)
        slot = self._find_slot(key, key_hash, for_insert=True)

        if slot is None:
            self._create_new_table()
            slot = self._find_slot(key, key_hash, for_insert=True)

        p, table_addr, position, entry = slot
        entry_addr = table_addr + position * self._hash_table.record_size
        is_update = entry.get("valid", False) and entry.get("key") == key
        expected_type = self._template.ds_class if self._is_nested else None

        if is_update:
            # Update existing item - single write to values dataset
            value_addr = int(entry["value_addr"])
            if self._is_nested:
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type.__name__}, got {type(value)}")
                ref_data = self._values_dataset._serialize(**value.to_ref())
            else:
                ref_data = self._values_dataset._serialize(**value)

            # Atomic update: single write
            with self._db.write_batch() as writes:
                writes.append((value_addr, ref_data))
        else:
            # Insert new item - batch hash table + values writes
            if self._is_nested:
                if value is None:
                    value = self._template.new(self._db, _parent=self)
                    value._parent_key = key
                elif not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type.__name__}, got {type(value)}")
                ref = value.to_ref()
                value_addr = self._alloc_value_addr()
                value_data = self._values_dataset._serialize(**ref)
            else:
                value_addr = self._alloc_value_addr()
                value_data = self._values_dataset._serialize(**value)

            # Hash table entry
            hash_entry = {
                "hash": key_hash,
                "key": key,
                "value_addr": value_addr,
                "valid": True,
            }
            hash_data = self._hash_table._serialize(**hash_entry)

            # Atomic insert: batch both writes
            with self._db.write_batch() as writes:
                writes.append((value_addr, value_data))
                writes.append((entry_addr, hash_data))

            # Update in-memory state after successful transaction
            self.size += 1
            # Add to the correct table's bloom filter
            p_init = getattr(self, "_p_init", self.P_INIT)
            table_idx = p - p_init
            if self._blooms and table_idx < len(self._blooms):
                self._blooms[table_idx].add(key)

        # Cache the actual value (not the address!)
        if self._cache:
            self._cache[key] = value if self._is_nested else int(value_addr)

        self._auto_save_check()

        # If this is a nested dict, update our reference in parent
        if (
            self._parent is not None
            and self._parent != "__nested__"
            and self._parent_key is not None
        ):
            self._parent.update_nested_ref(self._parent_key, self)

        return value if self._is_nested else None

    def set(self, key, value, atomic=False):
        """Convenience method for setting values with optional atomic flag.

        This is an alias for __setitem__ that allows using the atomic parameter
        with cleaner syntax.

        Args:
            key: Key to set
            value: Value to store (dict for regular, Dict for nested)
            atomic: If True, use WAL for crash safety (default: False)

        Returns:
            The value (for nested dicts) or None

        Example:
            # Fast path
            users.set("alice", {"id": 1, "name": "Alice"})

            # Atomic path
            users.set("alice", {"id": 1, "name": "Alice"}, atomic=True)
        """
        return self.__setitem__(key, value, atomic=atomic)

    def __getitem__(self, key):
        """Get value for key.

        Args:
            key: Key to look up

        Returns:
            Value associated with key

        Raises:
            KeyError: If key not found
        """
        # Check cache first (stores actual values!)
        if self._cache:
            cached_value = self._cache.get(key)
            if cached_value is not None:
                if self._is_nested:
                    return cached_value
                return self._values_dataset[int(cached_value)]

        # Cache miss - do full lookup
        key = self._to_internal_key(key)
        key_hash = self._hash(key)

        try:
            p, table_addr, position, entry = self._find_slot(
                key, key_hash, for_insert=False
            )
            value_addr = int(entry["value_addr"])
            value_data = self._values_dataset[value_addr]

            if self._is_nested:
                # Use modular approach: let the inner class reconstruct itself
                inner_class = self._template.ds_class
                result = inner_class.from_ref(self._db, value_data)

                # Set shared datasets if needed (modular approach)
                if result.needs_shared_datasets():
                    result.set_shared_datasets(self._shared_datasets)

                # Set parent reference so nested structure can update us
                result._parent = self
                result._parent_key = key
            else:
                # Return data directly from user's dataset
                result = value_data

            # Cache the actual value for next time!
            if self._cache:
                self._cache[key] = result if self._is_nested else int(value_addr)

            return result
        except KeyError:
            # For nested dicts, auto-create on access (like List.append)
            if self._is_nested:
                # Use fast path for auto-creation
                return self._setitem_fast(key, None)
            else:
                raise

    def __delitem__(self, key):
        """Delete key from dict (soft delete).

        Args:
            key: Key to delete

        Raises:
            KeyError: If key not found
        """
        key = self._to_internal_key(key)
        key_hash = self._hash(key)
        p, table_addr, position, entry = self._find_slot(
            key, key_hash, for_insert=False
        )
        entry_addr = table_addr + position * self._hash_table.record_size
        entry["valid"] = False
        self._hash_table[entry_addr] = entry

        # Remove from counting bloom filter (reduces false positives over time)
        p_init = getattr(self, "_p_init", self.P_INIT)
        table_idx = p - p_init
        if self._blooms and table_idx < len(self._blooms):
            self._blooms[table_idx].remove(key)

        # Return value slot to internal freelist for reuse
        value_addr = int(entry["value_addr"])
        if not hasattr(self, "_value_freelist"):
            self._value_freelist = []
        self._value_freelist.append(value_addr)

        self.size -= 1
        if self._cache and key in self._cache:
            del self._cache[key]
        self._auto_save_check()

    def __contains__(self, key):
        key = self._to_internal_key(key)
        # Check cache first
        if self._cache and self._cache.get(key) is not None:
            return True
        # Direct hash table probe
        key_hash = self._hash(key)
        try:
            self._find_slot(key, key_hash, for_insert=False)
            return True
        except KeyError:
            return False

    def __len__(self):
        return self.size

    def get(self, key, default=None):
        """Get value for key with default fallback.

        Args:
            key: Key to look up
            default: Value to return if key not found

        Returns:
            Value for key, or default if not found
        """
        try:
            return self[key]
        except KeyError:
            return default

    def _read_table_entries(self, table_addr, capacity):
        """Bulk-read a hash table and yield valid (key, value_addr) pairs.

        Unlike read_many(), this does NOT stop at the first uninitialized
        slot — hash tables are sparse and may have gaps.
        """
        import numpy as np

        record_size = self._hash_table.record_size
        total_size = capacity * record_size
        raw_data = self._hash_table.db.read(table_addr, total_size)
        arr = np.frombuffer(raw_data, dtype=self._hash_table.schema)

        identifier = self._hash_table.identifier
        mask = (arr["_prefix"] == identifier) & (arr["valid"])
        for rec in arr[mask]:
            yield str(rec["key"]), int(rec["value_addr"])

    def keys(self):
        """Iterate over all keys in dict.

        Yields:
            Keys in arbitrary order
        """
        p_init = getattr(self, "_p_init", self.P_INIT)
        for p in range(p_init, self.p_last + 1):
            table_idx = p - p_init
            table_addr = self.table_addrs[table_idx]
            capacity = self._get_capacity(p)
            try:
                for key, _ in self._read_table_entries(table_addr, capacity):
                    yield key
            except Exception:
                pass

    def values(self):
        """Iterate over all values in dict.

        Yields:
            Values in arbitrary order
        """
        for _, value in self._iter_entries():
            yield value

    def items(self):
        """Iterate over all (key, value) pairs in dict.

        Yields:
            Tuples of (key, value) in arbitrary order
        """
        yield from self._iter_entries()

    def _iter_entries(self):
        """Iterate over all (key, value) pairs efficiently.

        Reads hash tables in bulk and resolves values directly
        from stored addresses, avoiding redundant key lookups.
        """
        p_init = getattr(self, "_p_init", self.P_INIT)
        for p in range(p_init, self.p_last + 1):
            table_idx = p - p_init
            table_addr = self.table_addrs[table_idx]
            capacity = self._get_capacity(p)

            try:
                for key, value_addr in self._read_table_entries(table_addr, capacity):
                    value_data = self._values_dataset[value_addr]

                    if self._is_nested:
                        inner_class = self._template.ds_class
                        result = inner_class.from_ref(self._db, value_data)
                        if result.needs_shared_datasets():
                            result.set_shared_datasets(self._shared_datasets)
                        result._parent = self
                        result._parent_key = key
                        yield key, result
                    else:
                        yield key, value_data
            except Exception:
                pass

    def __iter__(self):
        return self.keys()

    # ---- Registry protocol ----

    def _get_registry_params(self):
        # Nested child Dicts are managed by their parent, not the registry
        if self._parent is not None:
            return None
        return {
            "schema": self.item_schema,
            "cache_size": self.cache_size,
            "use_bloom": self.use_bloom,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db, params["schema"],
            cache_size=params.get("cache_size", 1000),
            use_bloom=params.get("use_bloom", True),
        )

    def to_dict(self):
        """Bulk-export all entries as a plain Python dict.

        Reads the entire values block in one mmap slice, then resolves
        each entry — much faster than iterating via items() for large dicts.

        Returns:
            dict mapping keys to value dicts
        """
        import numpy as np

        # 1) Collect all (key, value_addr) from hash tables
        entries = []
        p_init = getattr(self, "_p_init", self.P_INIT)
        for p in range(p_init, self.p_last + 1):
            table_idx = p - p_init
            table_addr = self.table_addrs[table_idx]
            capacity = self._get_capacity(p)
            try:
                for key, value_addr in self._read_table_entries(table_addr, capacity):
                    entries.append((key, value_addr))
            except Exception:
                pass

        if not entries:
            return {}

        # 2) Bulk-read the values block if all addresses are in it
        record_size = self._values_dataset.record_size
        block_start = self.values_block_addr
        block_end = block_start + self.next_data_offset * record_size

        # Check if all value addrs fall in the contiguous block
        all_contiguous = all(
            block_start <= addr < block_end for _, addr in entries
        )

        if all_contiguous and self.next_data_offset > 0 and not self._is_nested:
            # Single bulk read of the entire values block
            total = self.next_data_offset * record_size
            raw = self._values_dataset.db.read(block_start, total)
            arr = np.frombuffer(raw, dtype=self._values_dataset.schema)

            ds = self._values_dataset
            has_variable = bool(ds._text_fields or ds._blob_fields)

            if not has_variable:
                # Pure fast path: all fields are fixed-size numpy, no blobs
                result = {}
                for key, addr in entries:
                    idx = (addr - block_start) // record_size
                    rec = arr[idx]
                    result[key] = {f: rec[f] for f in ds.user_schema.names}
                return result

            # Pass 1 — extract fixed fields from numpy, collect blob refs
            # pending_blobs: list of (file_offset, key, field_name, is_text)
            #   sorted by file_offset in pass 2 for sequential I/O
            result = {}
            pending_blobs = []

            for key, addr in entries:
                idx = (addr - block_start) // record_size
                rec = arr[idx]
                d = {}
                for field in ds.user_schema.names:
                    if field in ds._text_fields:
                        val = rec[field]
                        off, ns = int(val["offset"]), int(val["n_slots"])
                        if off == 0 and ns == 0:
                            d[field] = ""
                        else:
                            d[field] = None          # placeholder
                            pending_blobs.append((off, key, field, True))
                    elif field in ds._blob_fields:
                        val = rec[field]
                        off, ns = int(val["offset"]), int(val["n_slots"])
                        if off == 0 and ns == 0:
                            d[field] = None
                        else:
                            d[field] = None          # placeholder
                            pending_blobs.append((off, key, field, False))
                    else:
                        d[field] = rec[field]
                result[key] = d

            # Pass 2 — resolve blobs in file-offset order (sequential I/O,
            # better page-cache and TLB behaviour on large datasets)
            if pending_blobs:
                pending_blobs.sort(key=lambda x: x[0])
                for off, key, field, is_text in pending_blobs:
                    raw_blob = ds.blob_store.read(off)
                    result[key][field] = raw_blob.decode("utf-8") if is_text else raw_blob

            return result

        # 3) Fallback: individual reads (nested dicts, or scattered addresses)
        return dict(self._iter_entries())

    def __repr__(self):
        return f"Dict('{self.name}', size={self.size}, tables={self.p_last - self.P_INIT + 1})"
