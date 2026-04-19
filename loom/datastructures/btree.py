"""
Persistent B-tree implementation.

A B-tree provides ordered key-value storage with O(log n) operations
and efficient range queries. Keys are stored in sorted order, enabling:
- Ordered iteration
- Range queries (all keys between A and B)
- Prefix searches (all keys starting with X)
- Min/max operations

This implementation stores addresses (like Dict) rather than inline objects,
allowing flexible value sizes and nested structures.
"""

from .base import DataStructure
from loom.datastructures.template import DataStructureTemplate
from loom.cache import LRUCache
from loom.ref import Ref


class BTree(DataStructure):
    MAX_VALUE_BLOCKS = 8
    """Persistent B-tree with O(log n) operations and range queries.

    Usage:
        with DB("mydata.db") as db:
            user_ds = db.create_dataset("users", id="uint32", name="U50")
            users = db.create_btree("users_btree", user_ds)

            # Basic operations (like Dict)
            users["alice"] = {"id": 1, "name": "Alice"}
            user = users["alice"]
            del users["alice"]

            # Ordered iteration
            for key in users.keys():  # Sorted order!
                print(key)

            # Range queries
            for key, value in users.range("a", "m"):
                print(key, value)

            # Prefix search
            for key, value in users.prefix("admin_"):
                print(key, value)

    Performance:
        - Insert: O(log n)
        - Lookup: O(log n)
        - Delete: O(log n)
        - Range query: O(log n + k) where k is result size
        - Min/Max: O(log n)
    """

    # B+ tree parameters
    # Lower ORDER = smaller nodes = faster read/write but taller tree
    # Higher ORDER = larger nodes = slower read/write but shorter tree
    ORDER = 32  # Maximum children per node (keys = ORDER - 1)
    MIN_KEYS = ORDER // 2 - 1  # Minimum keys in non-root node

    def __init__(
        self,
        name: str,
        db,
        dataset_or_template=None,
        key_size: int = 50,
        cache_size: int = 100,
        auto_save_interval: int = None,
        _parent=None,
    ):
        """Initialize a BTree.

        Args:
            name: Unique name for this BTree
            db: Database instance
            dataset_or_template: Dataset for values, or None to load existing
            key_size: Maximum length of string keys (default: 50)
            cache_size: Number of nodes to cache (default: 100)
            auto_save_interval: Save metadata every N operations
        """
        self._key_size = key_size
        self.cache_size = cache_size
        self._parent_key = None

        if isinstance(dataset_or_template, DataStructureTemplate):
            self._template = dataset_or_template
            self._is_nested = True
            self._user_dataset = None
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
            self._template = None
            self._is_nested = False
            self._user_dataset = None
            self.item_schema = None

        super().__init__(name, db, auto_save_interval, _parent=_parent)

        # Check if already exists
        metadata = self._load_metadata()

        if metadata:
            self._load()
        else:
            self._initialize()

        # Node cache for performance
        self._node_cache = self._make_cache("nodes", cache_size)

    @classmethod
    def _get_nested_ref_schema(cls):
        return {
            "root_addr": "uint64",
            "size": "uint32",
            "height": "uint16",
            "values_block_addr": "uint64",
            "values_capacity": "uint32",
            "next_data_offset": "uint32",
            "values_blocks_head": "uint64",
            "values_blocks_tail": "uint64",
            "current_values_block_addr": "uint64",
            "current_values_block_capacity": "uint32",
            "current_values_block_offset": "uint32",
            "key_size": "uint16",
            "cache_size": "uint16",
        }

    def _allocate_value_addr(self):
        """Allocate an address for a new value record."""
        if not getattr(self, "values_blocks_head", 0):
            raise RuntimeError("BTree values blocks are not initialized")

        if self.current_values_block_offset >= self.current_values_block_capacity:
            new_capacity = int(self.current_values_block_capacity) * 2
            new_block_addr = int(self._values_dataset.allocate_block(new_capacity))
            new_node_addr = int(self._blocks_dataset.allocate_block(1))
            self._blocks_dataset[new_node_addr] = {
                "block_addr": int(new_block_addr),
                "capacity": int(new_capacity),
                "next": 0,
            }

            tail_node = self._blocks_dataset[int(self.values_blocks_tail)]
            tail_node["next"] = int(new_node_addr)
            self._blocks_dataset[int(self.values_blocks_tail)] = tail_node

            self.values_blocks_tail = int(new_node_addr)
            self.current_values_block_addr = int(new_block_addr)
            self.current_values_block_capacity = int(new_capacity)
            self.current_values_block_offset = 0

        value_addr = int(self.current_values_block_addr) + int(
            self.current_values_block_offset
        ) * int(self._values_dataset.record_size)
        self.current_values_block_offset += 1
        self.next_data_offset += 1
        return int(value_addr)

    @classmethod
    def get_shared_dataset_specs(cls, parent_name, inner_schema, **kwargs):
        node_schema = {
            "is_leaf": "bool",
            "num_keys": "uint16",
        }
        for i in range(cls.ORDER - 1):
            node_schema[f"key_{i}"] = "U50"
        for i in range(cls.ORDER):
            node_schema[f"child_{i}"] = "uint64"

        return {
            "_shared_btree_nodes": {
                "name": f"_{parent_name}_shared_btree_nodes",
                "schema": node_schema,
            },
            "_shared_values_dataset": {
                "name": f"_{parent_name}_shared_btree_values",
                "schema": inner_schema,
            },
            "_shared_btree_value_blocks": {
                "name": f"_{parent_name}_shared_btree_value_blocks",
                "schema": {
                    "block_addr": "uint64",
                    "capacity": "uint32",
                    "next": "uint64",
                },
            },
        }

    def set_shared_datasets(self, shared_datasets):
        if "_shared_btree_nodes" in shared_datasets:
            self._node_dataset = shared_datasets["_shared_btree_nodes"]
        if "_shared_values_dataset" in shared_datasets:
            self._values_dataset = shared_datasets["_shared_values_dataset"]
            self.item_schema = self._extract_schema(self._values_dataset)
        if "_shared_btree_value_blocks" in shared_datasets:
            self._blocks_dataset = shared_datasets["_shared_btree_value_blocks"]

        # Backward compat: nested instances created from legacy refs can be
        # migrated once the shared blocks dataset is available.
        if (
            getattr(self, "values_blocks_head", 0) == 0
            and getattr(self, "_legacy_values_block_addrs", None)
            and getattr(self, "_blocks_dataset", None) is not None
        ):
            self._migrate_legacy_blocks_to_dataset(self._legacy_values_block_addrs)
            self._legacy_values_block_addrs = None

    def _migrate_legacy_blocks_to_dataset(self, block_addrs):
        if not block_addrs:
            return
        head = 0
        prev = 0
        for addr in block_addrs:
            node_addr = int(self._blocks_dataset.allocate_block(1))
            self._blocks_dataset[node_addr] = {
                "block_addr": int(addr),
                "capacity": int(self.values_capacity),
                "next": 0,
            }
            if head == 0:
                head = int(node_addr)
            if prev != 0:
                prev_node = self._blocks_dataset[int(prev)]
                prev_node["next"] = int(node_addr)
                self._blocks_dataset[int(prev)] = prev_node
            prev = int(node_addr)

        self.values_blocks_head = int(head)
        self.values_blocks_tail = int(prev)
        self.current_values_block_addr = int(block_addrs[-1])
        self.current_values_block_capacity = int(self.values_capacity)
        self.current_values_block_offset = int(self.next_data_offset) % int(
            self.values_capacity
        )

    def needs_shared_datasets(self):
        return (
            getattr(self, "_node_dataset", None) is None
            or getattr(self, "_values_dataset", None) is None
            or getattr(self, "_blocks_dataset", None) is None
        )

    def _initialize(self):
        """Initialize a new BTree."""
        # Create node dataset for B-tree structure
        # Each node stores: is_leaf, num_keys, keys[], children[] or value_addrs[]
        node_schema = {
            "is_leaf": "bool",
            "num_keys": "uint16",
        }
        # Add key slots
        for i in range(self.ORDER - 1):
            node_schema[f"key_{i}"] = f"U{self._key_size}"
        # Add child/value address slots (ORDER for children, ORDER-1 for values)
        for i in range(self.ORDER):
            node_schema[f"child_{i}"] = "uint64"  # Child node address or value address

        if self._parent is not None and self._parent != "__nested__":
            self._node_dataset = self._parent._shared_btree_nodes
            self._values_dataset = self._parent._shared_values_dataset
            self._blocks_dataset = self._parent._shared_btree_value_blocks
            self.item_schema = self._extract_schema(self._values_dataset)
            self._shared_datasets = {}
        else:
            self._node_dataset = self._db.create_dataset(
                f"_btree_{self.name}_nodes", **node_schema
            )

            self._blocks_dataset = self._db.create_dataset(
                f"_btree_{self.name}_value_blocks",
                block_addr="uint64",
                capacity="uint32",
                next="uint64",
            )

            if self._is_nested:
                ref_schema = self._template.get_ref_schema()
                self._values_dataset = self._db.create_dataset(
                    f"_btree_{self.name}_values", **ref_schema
                )
                self.item_schema = ref_schema

                inner_schema = self._extract_schema(self._template.dataset)
                inner_class = self._template.ds_class
                shared_specs = inner_class.get_shared_dataset_specs(
                    f"btree_{self.name}", inner_schema
                )

                self._shared_datasets = {}
                for attr_name, spec in shared_specs.items():
                    dataset = self._db.create_dataset(spec["name"], **spec["schema"])
                    setattr(self, attr_name, dataset)
                    self._shared_datasets[attr_name] = dataset
            else:
                self._values_dataset = self._user_dataset
                self.item_schema = self._extract_schema(self._user_dataset)
                self._shared_datasets = {}

        self._initial_capacity = 10000
        self.values_block_addr = int(
            self._values_dataset.allocate_block(self._initial_capacity)
        )
        self.values_capacity = int(self._initial_capacity)
        self.next_data_offset = 0

        head_addr = int(self._blocks_dataset.allocate_block(1))
        self._blocks_dataset[head_addr] = {
            "block_addr": int(self.values_block_addr),
            "capacity": int(self.values_capacity),
            "next": 0,
        }
        self.values_blocks_head = int(head_addr)
        self.values_blocks_tail = int(head_addr)
        self.current_values_block_addr = int(self.values_block_addr)
        self.current_values_block_capacity = int(self.values_capacity)
        self.current_values_block_offset = 0

        # Initialize empty tree (no root yet)
        self.root_addr = 0  # 0 means empty tree
        self.size = 0
        self.height = 0

        self.save()

    def _load(self):
        """Load existing BTree from disk."""
        metadata = self._load_metadata()

        self._key_size = metadata.get("key_size", 50)
        self.cache_size = metadata.get("cache_size", self.cache_size)
        self.root_addr = metadata.get("root_addr", 0)
        self.size = metadata.get("size", 0)
        self.height = metadata.get("height", 0)
        self.values_block_addr = metadata.get("values_block_addr", 0)
        self.values_capacity = metadata.get("values_capacity", 10000)
        self.next_data_offset = metadata.get("next_data_offset", 0)

        self.values_blocks_head = int(metadata.get("values_blocks_head", 0))
        self.values_blocks_tail = int(metadata.get("values_blocks_tail", 0))
        self.current_values_block_addr = int(
            metadata.get("current_values_block_addr", 0)
        )
        self.current_values_block_capacity = int(
            metadata.get("current_values_block_capacity", 0)
        )
        self.current_values_block_offset = int(
            metadata.get("current_values_block_offset", 0)
        )

        self._is_nested = metadata.get("is_nested", False)

        self._node_dataset = self._get_dataset(metadata["node_dataset_name"])
        self._values_dataset = self._get_dataset(metadata["values_dataset_name"])

        # Backward compat: older BTrees didn't have a blocks dataset.
        blocks_dataset_name = metadata.get("blocks_dataset_name")
        if blocks_dataset_name:
            self._blocks_dataset = self._get_dataset(blocks_dataset_name)
        else:
            self._blocks_dataset = self._db.create_dataset(
                f"_btree_{self.name}_value_blocks",
                block_addr="uint64",
                capacity="uint32",
                next="uint64",
            )
            legacy_addrs = metadata.get("values_block_addrs")
            if not legacy_addrs and self.values_block_addr:
                legacy_addrs = [int(self.values_block_addr)]
            self._migrate_legacy_blocks_to_dataset([int(a) for a in legacy_addrs])
            self.save()
        if self._is_nested:
            from loom.datastructures.base import _DS_REGISTRY

            template_config = metadata["template_config"]
            template_class_name = metadata.get("template_class", "Dict")
            template_class = _DS_REGISTRY[template_class_name]
            full_config = {**template_config, "_template_dataset": metadata.get("template_dataset")}
            self._template = template_class._reconstruct_template(
                self._db, full_config, template_class_name
            )
            self.item_schema = self._template.get_ref_schema()

            self._shared_datasets = {}
            shared_datasets_meta = metadata.get("shared_datasets", {})
            for attr_name, dataset_name in shared_datasets_meta.items():
                dataset = self._get_dataset(dataset_name)
                setattr(self, attr_name, dataset)
                self._shared_datasets[attr_name] = dataset
        else:
            self._template = None
            self.item_schema = self._extract_schema(self._values_dataset)
            self._shared_datasets = {}

    def save(self):
        """Save BTree metadata."""
        metadata = {
            "key_size": self._key_size,
            "cache_size": self.cache_size,
            "root_addr": self.root_addr,
            "size": self.size,
            "height": self.height,
            "values_block_addr": self.values_block_addr,
            "values_capacity": self.values_capacity,
            "next_data_offset": self.next_data_offset,
            "node_dataset_name": self._node_dataset.name,
            "values_dataset_name": self._values_dataset.name,
            "blocks_dataset_name": self._blocks_dataset.name,
            "is_nested": self._is_nested,
            "values_blocks_head": int(getattr(self, "values_blocks_head", 0)),
            "values_blocks_tail": int(getattr(self, "values_blocks_tail", 0)),
            "current_values_block_addr": int(
                getattr(self, "current_values_block_addr", 0)
            ),
            "current_values_block_capacity": int(
                getattr(self, "current_values_block_capacity", 0)
            ),
            "current_values_block_offset": int(
                getattr(self, "current_values_block_offset", 0)
            ),
        }

        if self._is_nested:
            metadata["template_dataset"] = self._template.dataset.name
            metadata["template_config"] = self._template.config
            metadata["template_class"] = self._template.ds_class.__name__
            metadata["shared_datasets"] = {
                attr_name: dataset.name
                for attr_name, dataset in getattr(self, "_shared_datasets", {}).items()
            }

        self._save_metadata(metadata)

    def to_ref(self):
        if self._parent is not None and self._parent != "__nested__":
            return {
                "valid": True,
                "root_addr": int(self.root_addr),
                "size": int(self.size),
                "height": int(self.height),
                "values_block_addr": int(self.values_block_addr),
                "values_capacity": int(self.values_capacity),
                "next_data_offset": int(self.next_data_offset),
                "values_blocks_head": int(getattr(self, "values_blocks_head", 0)),
                "values_blocks_tail": int(getattr(self, "values_blocks_tail", 0)),
                "current_values_block_addr": int(
                    getattr(self, "current_values_block_addr", 0)
                ),
                "current_values_block_capacity": int(
                    getattr(self, "current_values_block_capacity", 0)
                ),
                "current_values_block_offset": int(
                    getattr(self, "current_values_block_offset", 0)
                ),
                "key_size": int(self._key_size),
                "cache_size": int(self.cache_size),
            }
        return super().to_ref()

    @classmethod
    def _from_ref_impl(cls, db, ref):
        is_binary_format = "root_addr" in ref and "values_block_addr" in ref
        if is_binary_format:
            instance = object.__new__(cls)

            instance.root_addr = int(ref["root_addr"])
            instance.size = int(ref["size"])
            instance.height = int(ref["height"])
            instance.values_block_addr = int(ref["values_block_addr"])
            instance.values_capacity = int(ref["values_capacity"])
            instance.next_data_offset = int(ref["next_data_offset"])
            instance.values_blocks_head = int(ref.get("values_blocks_head", 0))
            instance.values_blocks_tail = int(ref.get("values_blocks_tail", 0))
            instance.current_values_block_addr = int(
                ref.get("current_values_block_addr", 0)
            )
            instance.current_values_block_capacity = int(
                ref.get("current_values_block_capacity", 0)
            )
            instance.current_values_block_offset = int(
                ref.get("current_values_block_offset", 0)
            )
            instance._key_size = int(ref.get("key_size", 50))
            instance.cache_size = int(ref.get("cache_size", 0))

            instance.name = f"_nested_btree_{id(instance)}"
            instance._db = db
            instance._parent = "__nested__"
            instance._parent_key = None
            instance._is_nested = False
            instance._template = None
            instance._shared_datasets = {}
            instance._auto_save_interval = 0
            instance._ops_since_save = 0
            instance._inline_metadata = None
            instance._metadata_key = None

            instance._node_dataset = None
            instance._values_dataset = None
            instance._blocks_dataset = None
            instance.item_schema = None

            # Backward compat: legacy nested refs stored a fixed list of block addrs
            blocks_count = int(ref.get("values_blocks_count", 0))
            if blocks_count:
                addrs = (
                    [int(instance.values_block_addr)]
                    if instance.values_block_addr
                    else []
                )
                for i in range(1, blocks_count):
                    addrs.append(int(ref.get(f"values_block_{i}", 0)))
                instance._legacy_values_block_addrs = [a for a in addrs if a]
            else:
                instance._legacy_values_block_addrs = None

            instance._initial_capacity = 0
            instance._node_cache = instance._make_cache("nodes", instance.cache_size)

            return instance

        return cls(
            ref["ds_name"],
            db,
            None,
            cache_size=ref["cache_size"],
            key_size=ref["key_size"],
        )

    # ========== Node Operations ==========

    def _create_node(self, is_leaf=True):
        """Create a new node and return its address."""
        # Allocate space for one node
        addr = self._node_dataset.allocate_block(1)

        # Initialize empty node
        node_data = {"is_leaf": is_leaf, "num_keys": 0}
        for i in range(self.ORDER - 1):
            node_data[f"key_{i}"] = ""
        for i in range(self.ORDER):
            node_data[f"child_{i}"] = 0

        self._node_dataset[addr] = node_data
        return addr

    def _read_node(self, addr):
        """Read a node from disk (with caching)."""
        if addr == 0:
            return None

        # Check cache
        if self._node_cache:
            cached = self._node_cache.get(addr)
            if cached is not None:
                return cached

        # Read from disk
        node_data = self._node_dataset[addr]

        # Parse into structured format
        node = {
            "addr": addr,
            "is_leaf": bool(node_data["is_leaf"]),
            "num_keys": int(node_data["num_keys"]),
            "keys": [],
            "children": [],
        }

        for i in range(node["num_keys"]):
            key = str(node_data[f"key_{i}"]).rstrip("\x00")
            node["keys"].append(key)

        # For leaf nodes, children are value addresses
        # For internal nodes, children are node addresses
        num_children = node["num_keys"] + 1 if not node["is_leaf"] else node["num_keys"]
        for i in range(num_children):
            node["children"].append(int(node_data[f"child_{i}"]))

        # Cache it
        if self._node_cache:
            self._node_cache[addr] = node

        return node

    def _write_node(self, node):
        """Write a node to disk."""
        addr = node["addr"]

        node_data = {
            "is_leaf": node["is_leaf"],
            "num_keys": node["num_keys"],
        }

        for i in range(self.ORDER - 1):
            if i < len(node["keys"]):
                node_data[f"key_{i}"] = node["keys"][i]
            else:
                node_data[f"key_{i}"] = ""

        for i in range(self.ORDER):
            if i < len(node["children"]):
                node_data[f"child_{i}"] = node["children"][i]
            else:
                node_data[f"child_{i}"] = 0

        self._node_dataset[addr] = node_data

        # Update cache
        if self._node_cache:
            self._node_cache[addr] = node

    def _invalidate_cache(self, addr):
        """Invalidate a node in the cache."""
        if self._node_cache and addr in self._node_cache:
            del self._node_cache[addr]

    def _update_parent_ref(self):
        if (
            self._parent is not None
            and self._parent != "__nested__"
            and self._parent_key is not None
        ):
            self._parent.update_nested_ref(self._parent_key, self)

    # ========== Search Operations ==========

    def _search(self, key):
        """Search for a key in the tree (B+ tree style).

        In a B+ tree, all data is in leaf nodes. Internal nodes only
        contain separator keys for navigation.

        Returns:
            (node, index, found) where:
            - node: The leaf node where key is or should be
            - index: Position in node's keys
            - found: True if key exists
        """
        if self.root_addr == 0:
            return None, 0, False

        node = self._read_node(self.root_addr)

        # Always descend to leaf (B+ tree: data only in leaves)
        while not node["is_leaf"]:
            i = self._find_key_index(node, key)
            # In B+ tree, if key equals separator, go right
            if i < node["num_keys"] and node["keys"][i] == key:
                i += 1
            node = self._read_node(node["children"][i])

        # Now in leaf - search for key
        i = self._find_key_index(node, key)
        found = i < node["num_keys"] and node["keys"][i] == key
        return node, i, found

    def _find_key_index(self, node, key):
        """Find index where key should be in node (binary search)."""
        keys = node["keys"]
        lo, hi = 0, node["num_keys"]

        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid

        return lo

    # ========== Insert Operations ==========

    def _insert(self, key, value_addr):
        """Insert a key-value pair into the tree."""
        if self.root_addr == 0:
            # Create root node
            self.root_addr = self._create_node(is_leaf=True)
            self.height = 1

        # Insert into tree, potentially splitting nodes
        result = self._insert_non_full(self.root_addr, key, value_addr)

        if result is not None:
            # Root was split, create new root
            new_key, new_child = result
            new_root_addr = self._create_node(is_leaf=False)
            new_root = self._read_node(new_root_addr)

            new_root["keys"] = [new_key]
            new_root["children"] = [self.root_addr, new_child]
            new_root["num_keys"] = 1

            self._write_node(new_root)
            self.root_addr = new_root_addr
            self.height += 1

    def _insert_non_full(self, node_addr, key, value_addr):
        """Insert into a node, returning split info if node overflows.

        B+ tree style: all data goes to leaf nodes.

        Returns:
            None if no split needed
            (median_key, new_node_addr) if split occurred
        """
        node = self._read_node(node_addr)
        i = self._find_key_index(node, key)

        if node["is_leaf"]:
            # Check if key already exists (update)
            if i < node["num_keys"] and node["keys"][i] == key:
                # Update existing - just change the value address
                node["children"][i] = value_addr
                self._write_node(node)
                return None

            # Insert into leaf
            node["keys"].insert(i, key)
            node["children"].insert(i, value_addr)
            node["num_keys"] += 1
            self._write_node(node)

            # Check if split needed
            if node["num_keys"] >= self.ORDER:
                return self._split_node(node)
            return None
        else:
            # Internal node: find correct child to descend
            # In B+ tree, if key equals separator, go right
            if i < node["num_keys"] and node["keys"][i] == key:
                i += 1

            # Recurse into child
            result = self._insert_non_full(node["children"][i], key, value_addr)

            if result is not None:
                # Child was split, insert median key
                median_key, new_child = result

                node["keys"].insert(i, median_key)
                node["children"].insert(i + 1, new_child)
                node["num_keys"] += 1
                self._write_node(node)

                # Check if this node needs splitting
                if node["num_keys"] >= self.ORDER:
                    return self._split_node(node)

            return None

    def _split_node(self, node):
        """Split a full node into two nodes.

        Returns:
            (median_key, new_node_addr)
        """
        mid = node["num_keys"] // 2

        # Create new node for right half
        new_addr = self._create_node(is_leaf=node["is_leaf"])
        new_node = self._read_node(new_addr)

        if node["is_leaf"]:
            # For leaf: split keys and values
            median_key = node["keys"][mid]

            new_node["keys"] = node["keys"][mid:]
            new_node["children"] = node["children"][mid:]
            new_node["num_keys"] = len(new_node["keys"])

            node["keys"] = node["keys"][:mid]
            node["children"] = node["children"][:mid]
            node["num_keys"] = mid
        else:
            # For internal: median goes up, split rest
            median_key = node["keys"][mid]

            new_node["keys"] = node["keys"][mid + 1 :]
            new_node["children"] = node["children"][mid + 1 :]
            new_node["num_keys"] = len(new_node["keys"])

            node["keys"] = node["keys"][:mid]
            node["children"] = node["children"][: mid + 1]
            node["num_keys"] = mid

        self._write_node(node)
        self._write_node(new_node)

        return median_key, new_addr

    # ========== Delete Operations ==========

    def _delete(self, key):
        """Delete a key from the tree (B+ tree style).

        In a B+ tree, all data is in leaves. We simply find the leaf
        and delete from there. Internal node separators can stay as-is
        (they're just for navigation, not actual data).
        """
        if self.root_addr == 0:
            raise KeyError(key)

        # Find the leaf containing this key
        node, idx, found = self._search(key)

        if not found:
            raise KeyError(key)

        # Delete from leaf
        node["keys"].pop(idx)
        node["children"].pop(idx)
        node["num_keys"] -= 1
        self._write_node(node)
        self._invalidate_cache(node["addr"])

        self._update_parent_ref()

    # ========== Public API ==========

    def __setitem__(self, key, value):
        """Insert or update a key-value pair."""
        if not isinstance(key, str):
            key = str(key)

        if self._is_nested:
            expected_type = self._template.ds_class
            if value is None:
                value = self._template.new(self._db, _parent=self)
                value._parent_key = key
            elif not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected {expected_type.__name__} or None, got {type(value)}"
                )

            ref = value.to_ref()

            node, idx, found = self._search(key)
            if found:
                value_addr = node["children"][idx]
                self._values_dataset[value_addr] = ref
            else:
                value_addr = self._allocate_value_addr()
                self._values_dataset[value_addr] = ref
                self._insert(key, value_addr)
                self.size += 1

            self._auto_save_check()
            self._update_parent_ref()
            return value

        is_ref_value = isinstance(value, Ref)
        if is_ref_value and value.dataset is not self._values_dataset:
            raise TypeError(
                "Ref dataset mismatch: Ref must point to this BTree's values dataset"
            )

        # Check if key exists (for update vs insert)
        node, idx, found = self._search(key)

        if found:
            # Update existing - get current value address and update
            if is_ref_value:
                node["children"][idx] = int(value.addr)
                self._write_node(node)
                self._invalidate_cache(node["addr"])
            else:
                value_addr = node["children"][idx]
                self._values_dataset[value_addr] = value
        else:
            if is_ref_value:
                value_addr = int(value.addr)
            else:
                # Insert new - allocate value storage
                value_addr = self._allocate_value_addr()
                self._values_dataset[value_addr] = value

            self._insert(key, value_addr)
            self.size += 1

        self._auto_save_check()
        self._update_parent_ref()

    def get_ref(self, key):
        """Return a Ref handle to the underlying record for a key."""
        if not isinstance(key, str):
            key = str(key)

        node, idx, found = self._search(key)
        if not found:
            raise KeyError(key)

        value_addr = int(node["children"][idx])
        return Ref(self._values_dataset, value_addr)

    def __getitem__(self, key):
        """Get value for key."""
        if not isinstance(key, str):
            key = str(key)

        node, idx, found = self._search(key)
        if not found:
            if self._is_nested:
                return self.__setitem__(key, None)
            raise KeyError(key)

        value_addr = node["children"][idx]
        value_data = self._values_dataset[value_addr]

        if self._is_nested:
            inner_class = self._template.ds_class
            result = inner_class.from_ref(self._db, value_data)
            if result.needs_shared_datasets():
                result.set_shared_datasets(self._shared_datasets)
            result._parent = self
            result._parent_key = key
            return result

        return value_data

    def update_nested_ref(self, key, nested_item):
        if not self._is_nested:
            return

        node, idx, found = self._search(key)
        if not found:
            return

        value_addr = node["children"][idx]
        self._values_dataset[value_addr] = nested_item.to_ref()
        if self._node_cache:
            self._node_cache.invalidate(node["addr"])

    def __delitem__(self, key):
        """Delete a key."""
        if not isinstance(key, str):
            key = str(key)

        self._delete(key)
        self.size -= 1
        self._auto_save_check()
        self._update_parent_ref()

    def __contains__(self, key):
        """Check if key exists."""
        if not isinstance(key, str):
            key = str(key)

        _, _, found = self._search(key)
        return found

    def __len__(self):
        """Return number of items."""
        return self.size

    def get(self, key, default=None):
        """Get value for key with default."""
        try:
            return self[key]
        except KeyError:
            return default

    # ========== Ordered Operations ==========

    def keys(self):
        """Iterate over keys in sorted order."""
        if self.root_addr == 0:
            return

        yield from self._inorder_keys(self.root_addr)

    def _inorder_keys(self, node_addr):
        """In-order traversal yielding keys (B+ tree: only from leaves)."""
        node = self._read_node(node_addr)

        if node["is_leaf"]:
            for key in node["keys"]:
                yield key
        else:
            # B+ tree: internal keys are separators, not data.
            # Only yield from leaf children.
            for i in range(node["num_keys"] + 1):
                yield from self._inorder_keys(node["children"][i])

    def _inorder_entries(self, node_addr):
        """In-order traversal yielding (key, value_addr) from leaves only."""
        node = self._read_node(node_addr)
        if node["is_leaf"]:
            for i in range(node["num_keys"]):
                yield node["keys"][i], node["children"][i]
        else:
            for i in range(node["num_keys"] + 1):
                yield from self._inorder_entries(node["children"][i])

    def values(self):
        """Iterate over values in key-sorted order."""
        if self.root_addr == 0:
            return
        for key, value_addr in self._inorder_entries(self.root_addr):
            value_data = self._values_dataset[value_addr]
            if self._is_nested:
                inner_class = self._template.ds_class
                result = inner_class.from_ref(self._db, value_data)
                if result.needs_shared_datasets():
                    result.set_shared_datasets(self._shared_datasets)
                result._parent = self
                result._parent_key = key
                yield result
            else:
                yield value_data

    def items(self):
        """Iterate over (key, value) pairs in sorted order."""
        if self.root_addr == 0:
            return
        for key, value_addr in self._inorder_entries(self.root_addr):
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

    def __iter__(self):
        """Iterate over keys."""
        return self.keys()

    def min(self):
        """Return the minimum key, or None if empty."""
        if self.root_addr == 0:
            return None

        node = self._read_node(self.root_addr)
        while not node["is_leaf"]:
            node = self._read_node(node["children"][0])

        return node["keys"][0] if node["num_keys"] > 0 else None

    def max(self):
        """Return the maximum key, or None if empty."""
        if self.root_addr == 0:
            return None

        node = self._read_node(self.root_addr)
        while not node["is_leaf"]:
            node = self._read_node(node["children"][node["num_keys"]])

        return node["keys"][node["num_keys"] - 1] if node["num_keys"] > 0 else None

    # ========== Range Queries ==========

    def range(self, start=None, end=None, inclusive=(True, True)):
        """Iterate over (key, value) pairs in a key range.

        Args:
            start: Lower bound (None for no lower bound)
            end: Upper bound (None for no upper bound)
            inclusive: Tuple of (start_inclusive, end_inclusive)

        Yields:
            (key, value) pairs in sorted order
        """
        if self.root_addr == 0:
            return

        start_inc, end_inc = inclusive

        for key, value_addr in self._inorder_entries(self.root_addr):
            # Check lower bound
            if start is not None:
                if start_inc:
                    if key < start:
                        continue
                else:
                    if key <= start:
                        continue

            # Check upper bound — can break early since keys are sorted
            if end is not None:
                if end_inc:
                    if key > end:
                        break
                else:
                    if key >= end:
                        break

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

    def prefix(self, prefix_str):
        """Iterate over (key, value) pairs with keys starting with prefix.

        Args:
            prefix_str: The prefix to search for

        Yields:
            (key, value) pairs in sorted order
        """
        if self.root_addr == 0:
            return

        for key, value_addr in self._inorder_entries(self.root_addr):
            if key.startswith(prefix_str):
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
            elif key > prefix_str and not key.startswith(prefix_str):
                break

    # ========== Representation ==========

    def __repr__(self):
        return f"BTree('{self.name}', size={self.size}, height={self.height})"

    def close(self):
        """Close the BTree and save metadata."""
        self.save()

    # ---- Registry protocol ----

    def _get_registry_params(self):
        return {
            "schema": self.item_schema,
            "key_size": self._key_size,
            "cache_size": self.cache_size,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db, None,
            key_size=params.get("key_size", 50),
            cache_size=params.get("cache_size", 100),
        )
