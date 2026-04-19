"""
Persistent List with exponential block growth.

Features:
- O(1) append (no reallocation)
- O(1) indexing without deletions
- O(log n) indexing with deletions
- Exponential block growth (1.5x)
- Soft deletion with auto-compaction
- LRU cache for hot blocks
"""

import json

from loom.datastructures.base import DataStructure
from loom.datastructures.template import DataStructureTemplate
from loom.cache import LRUCache


class List(DataStructure):
    """Persistent list with exponential block growth and soft deletion.

    Uses multiple blocks that grow exponentially (1.5x) to avoid
    reallocation and copying. Supports soft deletion with automatic
    compaction when waste exceeds threshold.

    Usage:
        lst = List('tasks', db, item_schema={'id': 'uint64', 'task': 'U200'})

        # Append items
        lst.append({'id': 1, 'task': 'Write code'})
        lst.append({'id': 2, 'task': 'Test code'})

        # Access by index
        item = lst[0]
        lst[1] = {'id': 2, 'task': 'Test thoroughly'}

        # Delete items
        del lst[0]

        # Iterate
        for item in lst:
            print(item)

        # Length
        print(len(lst))

    Performance:
        - append: O(1)
        - get[i]: O(1) without deletions, O(log n) with deletions
        - set[i]: O(1) without deletions, O(log n) with deletions
        - del[i]: O(log n) soft delete
        - len: O(1)
        - iteration: O(n)

    Storage:
        - Overhead: 536 bytes per list
        - Block sizes: 57, 86, 129, 194, ... (1.5^p)
        - Max capacity: ~2.7 billion items (32 blocks)
    """

    # Exponential growth parameters
    GROWTH_FACTOR = 1.5
    P_INIT = 10  # First block: 1.5^10 ≈ 57 items
    MAX_BLOCKS = 32
    MAX_BLOCKS_NESTED = 16  # Nested lists use fewer blocks (still ~1B items)

    # Auto-compaction threshold
    COMPACT_THRESHOLD = 0.3  # Compact at 30% waste

    @classmethod
    def _get_ref_config_schema(cls):
        """Schema for List config fields in references."""
        return {
            "growth_factor": "float64",
            "cache_size": "uint32",
            "p_init": "uint32",
        }

    @classmethod
    def get_shared_dataset_specs(cls, parent_name, inner_schema, **kwargs):
        """Get specifications for shared datasets needed when List is nested.

        When List is used as inner type (e.g., Dict[List] or List[List]),
        the parent needs a shared items dataset for all nested lists.

        Args:
            parent_name: Name of the parent container
            inner_schema: Schema dict for list items

        Returns:
            Dict with shared dataset specifications
        """
        full_schema = {"valid": "bool", **inner_schema}
        return {
            "_shared_items_dataset": {
                "name": f"_{parent_name}_shared_items",
                "schema": full_schema,
            }
        }

    def set_shared_datasets(self, shared_datasets):
        """Set shared datasets on this nested List instance."""
        if "_shared_items_dataset" in shared_datasets:
            self._items_dataset = shared_datasets["_shared_items_dataset"]
            self.item_schema = self._extract_schema(self._items_dataset)

    def needs_shared_datasets(self):
        """Check if this nested List needs shared datasets."""
        return self._items_dataset is None

    @classmethod
    def _get_nested_ref_schema(cls):
        """Compact binary schema for nested List references.

        Total size: ~280 bytes per reference (vs ~3400 bytes with JSON)
        - 1M nested lists: ~280 MB (vs ~3.4 GB)
        """
        return {
            # Dataset name for reconstruction
            "items_dataset": "U50",  # 200 bytes (50 chars)
            # Metadata fields (binary, not JSON)
            "length": "uint32",  # 4 bytes
            "valid_count": "uint32",  # 4 bytes
            "p_last": "uint8",  # 1 byte
            # 16 blocks × 8 bytes = 128 bytes (supports ~1 billion items)
            "block_0": "uint64",
            "block_1": "uint64",
            "block_2": "uint64",
            "block_3": "uint64",
            "block_4": "uint64",
            "block_5": "uint64",
            "block_6": "uint64",
            "block_7": "uint64",
            "block_8": "uint64",
            "block_9": "uint64",
            "block_10": "uint64",
            "block_11": "uint64",
            "block_12": "uint64",
            "block_13": "uint64",
            "block_14": "uint64",
            "block_15": "uint64",
            # 16 valid counts × 4 bytes = 64 bytes
            "bvc_0": "uint32",
            "bvc_1": "uint32",
            "bvc_2": "uint32",
            "bvc_3": "uint32",
            "bvc_4": "uint32",
            "bvc_5": "uint32",
            "bvc_6": "uint32",
            "bvc_7": "uint32",
            "bvc_8": "uint32",
            "bvc_9": "uint32",
            "bvc_10": "uint32",
            "bvc_11": "uint32",
            "bvc_12": "uint32",
            "bvc_13": "uint32",
            "bvc_14": "uint32",
            "bvc_15": "uint32",
            # Config
            "cache_size": "uint16",  # 2 bytes
        }
        # Total: 1(valid) + 200 + 4 + 4 + 1 + 128 + 64 + 2 = 404 bytes

    def __init__(
        self,
        name,
        db,
        dataset_or_template,
        cache_size=10,
        auto_save_interval=None,
        _parent=None,
    ):
        """Initialize a List.

        Args:
            name: Unique name for this list
            db: Database instance
            dataset_or_template: Either a Dataset or DataStructureTemplate
            cache_size: Number of blocks to cache (0 to disable)
            auto_save_interval: Save metadata every N operations
            _parent: Parent data structure if this is nested (internal use)
        """
        self.cache_size = cache_size
        self._parent_key = None  # Key in parent dict (set by parent for Dict[List])

        # Detect if this is a nested list
        if isinstance(dataset_or_template, DataStructureTemplate):
            self._template = dataset_or_template
            self._is_nested = True
            self.item_schema = None
        elif dataset_or_template is not None:
            self._template = None
            self._is_nested = False
            self.item_schema = (
                dataset_or_template.schema
                if hasattr(dataset_or_template, "schema")
                else dataset_or_template
            )
            self._user_dataset = dataset_or_template
        else:
            # Loading existing list
            self._template = None
            self._is_nested = False
            self.item_schema = None

        super().__init__(name, db, auto_save_interval, _parent=_parent)

        # Check if already exists
        metadata = self._load_metadata()

        if metadata:
            self._load()
        else:
            self._initialize()

        # Initialize block cache
        self._block_cache = self._make_cache("blocks", cache_size)

    def _initialize(self):
        """Initialize new list."""
        # Check if this is a nested list that should share parent's dataset
        if self._parent is not None and self._parent != "__nested__":
            # Nested list: use parent's shared items dataset
            # The parent should have created _shared_items_dataset for us
            self._items_dataset = self._parent._shared_items_dataset
            self.item_schema = self._extract_schema(self._user_dataset)
            dataset_name = self._items_dataset.name
            self._shared_datasets = {}  # No shared datasets for child lists
        else:
            # Top-level list: create our own dataset
            dataset_name = f"_list_{self.name}_items"

            if self._is_nested:
                # Nested list container: store references AND create shared datasets for inner structures
                ref_schema = self._template.get_ref_schema()
                self._items_dataset = self._db.create_dataset(
                    dataset_name, **ref_schema
                )
                self.item_schema = ref_schema

                # Use modular approach: ask inner type what shared datasets it needs
                inner_schema = self._extract_schema(self._template.dataset)
                inner_class = self._template.ds_class
                shared_specs = inner_class.get_shared_dataset_specs(
                    f"list_{self.name}", inner_schema
                )

                # Create the shared datasets
                self._shared_datasets = {}
                for attr_name, spec in shared_specs.items():
                    dataset = self._db.create_dataset(spec["name"], **spec["schema"])
                    setattr(self, attr_name, dataset)
                    self._shared_datasets[attr_name] = dataset
            else:
                # Regular list: store data from user dataset
                schema_dict = self._extract_schema(self._user_dataset)
                full_schema = {"valid": "bool", **schema_dict}
                self._items_dataset = self._db.create_dataset(
                    dataset_name, **full_schema
                )
                self.item_schema = schema_dict
                self._shared_datasets = {}  # No shared datasets for regular lists

        # Initialize metadata
        self.length = 0
        self.valid_count = 0
        self.p_last = self.P_INIT - 1  # No blocks allocated yet
        self.blocks = [0] * self.MAX_BLOCKS  # 0 = not allocated
        self.block_valid_counts = [0] * self.MAX_BLOCKS

        # Save initial metadata
        metadata = {
            "length": self.length,
            "valid_count": self.valid_count,
            "p_last": self.p_last,
            "blocks": self.blocks,
            "block_valid_counts": self.block_valid_counts,
            "dataset_name": dataset_name,
            "is_nested": self._is_nested,
        }

        # Add template info if nested (only for top-level nested containers)
        if self._is_nested and (self._parent is None or self._parent == "__nested__"):
            metadata["template_dataset"] = self._template.dataset.name
            metadata["template_config"] = self._template.config
            metadata["template_class"] = self._template.ds_class.__name__

            # Save shared dataset names (modular approach)
            metadata["shared_datasets"] = {
                attr_name: dataset.name
                for attr_name, dataset in self._shared_datasets.items()
            }

        self._save_metadata(metadata)

    def _load(self):
        """Load existing list."""
        from loom.datastructures.base import _DS_REGISTRY

        metadata = self._load_metadata()

        self.length = metadata["length"]
        self.valid_count = metadata["valid_count"]
        self.p_last = metadata["p_last"]
        self.blocks = metadata["blocks"]
        self.block_valid_counts = metadata["block_valid_counts"]

        self._is_nested = metadata.get("is_nested", False)

        # Reconstruct template if nested
        if self._is_nested:
            template_config = metadata["template_config"]
            template_class_name = metadata.get("template_class", "List")

            # Get template class from registry (modular approach)
            template_class = _DS_REGISTRY.get(template_class_name, List)

            # Reconstruct template via class protocol (no per-type switch)
            full_config = {**template_config, "_template_dataset": metadata.get("template_dataset")}
            self._template = template_class._reconstruct_template(
                self._db, full_config, template_class_name
            )
            # Set item_schema from template
            self.item_schema = self._template.get_ref_schema()

            # Load shared datasets (modular approach)
            self._shared_datasets = {}
            shared_datasets_meta = metadata.get("shared_datasets", {})

            # Handle legacy format for backward compatibility
            if not shared_datasets_meta:
                # Legacy: check for old-style keys
                if "shared_hash_table_name" in metadata:
                    shared_datasets_meta["_shared_hash_table"] = metadata[
                        "shared_hash_table_name"
                    ]
                    shared_datasets_meta["_shared_values_dataset"] = metadata[
                        "shared_values_dataset_name"
                    ]
                elif "shared_dataset_name" in metadata:
                    shared_datasets_meta["_shared_items_dataset"] = metadata[
                        "shared_dataset_name"
                    ]

            for attr_name, dataset_name in shared_datasets_meta.items():
                dataset = self._get_dataset(dataset_name)
                setattr(self, attr_name, dataset)
                self._shared_datasets[attr_name] = dataset
        else:
            # Regular list - extract schema from items dataset
            self._items_dataset = self._get_dataset(metadata["dataset_name"])
            # Extract user schema (without 'valid' field)
            self.item_schema = {
                name: str(self._items_dataset.user_schema.fields[name][0])
                for name in self._items_dataset.user_schema.names
            }
            self._shared_datasets = {}
            return  # Early return since we already set _items_dataset

        self._items_dataset = self._get_dataset(metadata["dataset_name"])

    def _get_capacity(self, p):
        """Get capacity for block at power p."""
        return int(self.GROWTH_FACTOR**p)

    def _calculate_block_and_offset(self, index):
        """Calculate which block and offset for a given index.

        Args:
            index: Absolute index (slot number, not logical position)

        Returns:
            (block_idx, offset): Block index and offset within block
        """
        cumulative = 0

        for p in range(self.P_INIT, self.p_last + 2):  # +2 to check next block
            capacity = self._get_capacity(p)

            if index < cumulative + capacity:
                block_idx = p - self.P_INIT
                offset = index - cumulative
                return block_idx, offset

            cumulative += capacity

        raise IndexError(f"Index {index} out of range")

    def _allocate_block(self, block_idx):
        """Allocate a new block.

        Args:
            block_idx: Index in self.blocks array
        """
        p = self.P_INIT + block_idx
        capacity = self._get_capacity(p)

        # Allocate block in dataset
        block_addr = self._items_dataset.allocate_block(capacity)

        # Store block address
        self.blocks[block_idx] = block_addr
        self.block_valid_counts[block_idx] = 0
        self.p_last = max(self.p_last, p)

        # Invalidate cache for this block
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            if cache_key in self._block_cache:
                del self._block_cache[cache_key]

    def _read_block(self, block_idx):
        """Read entire block into memory.

        Args:
            block_idx: Index in self.blocks array

        Returns:
            List of items in block
        """
        # Check cache first
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            cached = self._block_cache.get(cache_key)
            if cached is not None:
                return cached

        block_addr = self.blocks[block_idx]
        if block_addr == 0:
            return []

        p = self.P_INIT + block_idx
        capacity = self._get_capacity(p)

        # Bulk read: one mmap slice, parse all records at once
        raw_items = self._items_dataset.read_many(block_addr, capacity)

        # Add 'valid' flag for items that don't have it (valid records)
        items = []
        for item in raw_items:
            if "valid" not in item:
                item["valid"] = True
            items.append(item)

        # Cache the block
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            self._block_cache[cache_key] = items

        return items

    def append(self, item=None, atomic=False):
        """Append item to end of list.

        Args:
            item: For regular list: dict matching item_schema
                  For nested list: List instance or None (creates new)

        Returns:
            The appended item (or newly created List for nested lists)

        Performance: O(1)
        """
        if self._is_nested:
            # Nested list: append reference
            expected_type = self._template.ds_class
            if item is None:
                # Create new nested structure from template
                # Pass _parent=self to prevent header pollution
                item = self._template.new(self._db, _parent=self)
                # Note: nested structures are NOT registered in _datastructures
                # to avoid polluting the registry with potentially millions of instances
            elif not isinstance(item, expected_type):
                raise TypeError(
                    f"Expected {expected_type.__name__} or None, got {type(item)}"
                )

            # Get reference dict
            ref = item.to_ref()

            # Append reference
            self._append_item(ref)

            return item
        else:
            # Regular list: append data
            if item is None:
                raise ValueError("Cannot append None to data list")
            if atomic:
                # Use atomic one-item batch
                self.append_many([item], atomic=True)
            else:
                # Fast path: direct mmap write
                self._append_item(item)

            return item

    def append_many(self, items, atomic=False):
        """Append multiple items.

        Args:
            items: Iterable of items (same type as for append).
            atomic: If True, commit all writes in a single WAL-backed batch.
                    If False (default), behave like repeated append() calls
                    for maximum speed.

        Notes:
            - For regular lists with atomic=True, this uses the DB-level
              write_batch context to commit all record writes in a single
              WAL transaction.
            - For atomic=False, this simply loops over append(), matching
              the behavior and performance characteristics of individual
              appends.
            - For nested lists, behavior is delegated to append() so that
              semantics remain unchanged.
        """
        if self._is_nested:
            for item in items:
                self.append(item, atomic=atomic)
            return

        items = list(items)
        if not items:
            return

        if not atomic:
            # Non-atomic fast path: mirror repeated append() calls
            for item in items:
                self.append(item)
            return

        # Atomic path: single WAL-backed batch
        with self._db.write_batch() as writes:
            index = self.length
            for item in items:
                # Find block and offset for next slot
                block_idx, offset = self._calculate_block_and_offset(index)

                # Allocate block if needed
                if self.blocks[block_idx] == 0:
                    self._allocate_block(block_idx)

                # Compute address for this item
                block_addr = self.blocks[block_idx]
                item_addr = block_addr + offset * self._items_dataset.record_size

                # Serialize item with valid flag
                item_with_valid = {"valid": True, **item}
                data = self._items_dataset._serialize(**item_with_valid)
                writes.append((item_addr, data))

                # Update metadata counters
                index += 1
                self.valid_count += 1
                self.block_valid_counts[block_idx] += 1

                # Invalidate block cache for this block
                if self._block_cache:
                    cache_key = f"block_{block_idx}"
                    if cache_key in self._block_cache:
                        del self._block_cache[cache_key]

            # Update overall length after planning all items
            self.length = index

        # Auto-save metadata if needed
        self._auto_save_check()

    def _append_item(self, item):
        """Internal method to append item dict to storage.

        Args:
            item: Item dict (data or reference)
        """
        # Find block and offset for next slot
        block_idx, offset = self._calculate_block_and_offset(self.length)

        # Allocate block if needed
        if self.blocks[block_idx] == 0:
            self._allocate_block(block_idx)

        # Write item with valid flag
        block_addr = self.blocks[block_idx]
        item_addr = block_addr + offset * self._items_dataset.record_size

        item_with_valid = {"valid": True, **item}
        self._items_dataset[item_addr] = item_with_valid

        # Update metadata
        self.length += 1
        self.valid_count += 1
        self.block_valid_counts[block_idx] += 1

        # Invalidate block cache
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            if cache_key in self._block_cache:
                del self._block_cache[cache_key]

        self._auto_save_check()

        # If this is a nested list in a Dict, update our reference in parent
        if (
            self._parent is not None
            and self._parent != "__nested__"
            and self._parent_key is not None
        ):
            self._parent.update_nested_ref(self._parent_key, self)

    def _find_nth_valid_item(self, n):
        """Find the nth valid item (for indexing with deletions).

        Args:
            n: Logical index (0-based, counting only valid items)

        Returns:
            Item dict

        Performance: O(log n) - scans through blocks
        """
        if n < 0 or n >= self.valid_count:
            raise IndexError("list index out of range")

        cumulative_valid = 0

        for block_idx in range(len(self.blocks)):
            if self.blocks[block_idx] == 0:
                continue

            block_valid = self.block_valid_counts[block_idx]

            if n < cumulative_valid + block_valid:
                # Found the block! Now find nth valid item within block
                offset_in_block = n - cumulative_valid
                return self._get_nth_valid_in_block(block_idx, offset_in_block)

            cumulative_valid += block_valid

        raise IndexError("list index out of range")

    def _get_nth_valid_in_block(self, block_idx, n):
        """Get nth valid item within a block.

        Args:
            block_idx: Block index
            n: Index within valid items in this block

        Returns:
            Item dict
        """
        items = self._read_block(block_idx)

        valid_count = 0
        for item in items:
            if item.get("valid", False):
                if valid_count == n:
                    # Remove 'valid' flag before returning
                    return {k: v for k, v in item.items() if k != "valid"}
                valid_count += 1

        raise IndexError("Item not found in block")

    def _get_nth_valid_address_in_block(self, block_idx, n):
        """Get address of nth valid item within a block.

        Args:
            block_idx: Block index
            n: Index within valid items in this block

        Returns:
            Item address
        """
        block_addr = self.blocks[block_idx]
        p = self.P_INIT + block_idx
        capacity = self._get_capacity(p)

        valid_count = 0
        for i in range(capacity):
            item_addr = block_addr + i * self._items_dataset.record_size
            try:
                item = self._items_dataset[item_addr]
                if item.get("valid", False):
                    if valid_count == n:
                        return item_addr
                    valid_count += 1
            except Exception:
                # Uninitialized slot
                break

        raise IndexError("Item not found in block")

    def slice_array(self, start, stop):
        """Get a slice as a NumPy structured array (fast, no dict overhead).

        Args:
            start: Start index
            stop: Stop index

        Returns:
            NumPy structured array with user schema fields

        Note:
            Only works when there are no deletions (length == valid_count).
            For lists with deletions, use regular slicing.
        """
        if self.length != self.valid_count:
            raise ValueError("slice_array only works on lists without deletions")

        if start < 0:
            start = self.valid_count + start
        if stop < 0:
            stop = self.valid_count + stop

        start = max(0, min(start, self.valid_count))
        stop = max(0, min(stop, self.valid_count))

        if start >= stop:
            return None

        remaining = stop - start
        arrays = []

        block_idx, offset = self._calculate_block_and_offset(start)

        while remaining > 0 and block_idx < len(self.blocks):
            block_addr = self.blocks[block_idx]
            if block_addr == 0:
                break

            p = self.P_INIT + block_idx
            capacity = self._get_capacity(p)

            available = capacity - offset
            take = min(remaining, available)
            if take <= 0:
                break

            item_addr = block_addr + offset * self._items_dataset.record_size
            arr = self._items_dataset.read_many(item_addr, take, as_array=True)
            arrays.append(arr)

            remaining -= len(arr)
            block_idx += 1
            offset = 0

        if not arrays:
            return None

        import numpy as np

        return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]

    def __getitem__(self, index):
        """Get item(s) by index or slice.

        Args:
            index: Index (int) or slice object

        Returns:
            Item dict (for int) or list of items (for slice)

        Performance:
            - Single item: O(1) if no deletions, O(log n) if deletions
            - Slice: O(k) where k is slice length
        """
        # Handle slicing
        if isinstance(index, slice):
            return self._getslice(index)

        # Handle negative indices
        if index < 0:
            index = self.valid_count + index

        if index < 0 or index >= self.valid_count:
            raise IndexError("list index out of range")

        # Fast path: no deletions
        if self.length == self.valid_count:
            # Direct calculation
            block_idx, offset = self._calculate_block_and_offset(index)
            block_addr = self.blocks[block_idx]
            item_addr = block_addr + offset * self._items_dataset.record_size
            item = self._items_dataset[item_addr]

            # Remove 'valid' flag
            item = {k: v for k, v in item.items() if k != "valid"}

            # Resolve reference if nested
            if self._is_nested:
                return self._resolve_nested_ref(item)

            return item

        # Slow path: has deletions, need to find nth valid item
        item = self._find_nth_valid_item(index)

        # Resolve reference if nested
        if self._is_nested:
            return self._resolve_nested_ref(item)

        return item

    def _resolve_nested_ref(self, item):
        """Resolve a nested reference to the appropriate data structure.

        Args:
            item: Reference dict from the items dataset

        Returns:
            Reconstructed List or Dict instance
        """
        # Use modular approach: let the inner class reconstruct itself
        inner_class = self._template.ds_class
        result = inner_class.from_ref(self._db, item)

        # Set shared datasets if needed (modular approach)
        if result.needs_shared_datasets():
            result.set_shared_datasets(self._shared_datasets)

        return result

    def _getslice(self, slice_obj):
        """Get slice of items.

        Args:
            slice_obj: Slice object

        Returns:
            List of items
        """
        start, stop, step = slice_obj.indices(self.valid_count)

        # Fast path: no deletions and unit step -> contiguous logical indices
        if step == 1 and self.length == self.valid_count:
            if start >= stop:
                return []

            remaining = stop - start
            items = []

            # Find starting block and offset for the first index
            block_idx, offset = self._calculate_block_and_offset(start)

            while remaining > 0 and block_idx < len(self.blocks):
                block_addr = self.blocks[block_idx]
                if block_addr == 0:
                    break

                p = self.P_INIT + block_idx
                capacity = self._get_capacity(p)

                # How many items to take from this block
                available = capacity - offset
                take = min(remaining, available)
                if take <= 0:
                    break

                item_addr = block_addr + offset * self._items_dataset.record_size

                # If dataset has text/blob fields, use slow path so _deserialize
                # resolves blob references → actual strings.
                # Otherwise use the fast numpy bulk-read path.
                has_variable = bool(
                    getattr(self._items_dataset, "_text_fields", None) or
                    getattr(self._items_dataset, "_blob_fields", None)
                )

                if has_variable:
                    raw_items = self._items_dataset.read_many(item_addr, take)
                    for item in raw_items:
                        d = {k: v for k, v in item.items() if k != "valid"}
                        if self._is_nested:
                            items.append(self._resolve_nested_ref(d))
                        else:
                            items.append(d)
                    remaining -= len(raw_items)
                    block_idx += 1
                    offset = 0
                    continue

                # Fast numpy path (no text/blob fields)
                arr = self._items_dataset.read_many(item_addr, take, as_array=True)

                # Convert to dicts only at the end
                user_fields = self._items_dataset.user_schema.names
                for rec in arr:
                    if self._is_nested:
                        d = {
                            field: rec[field]
                            for field in user_fields
                            if field != "valid"
                        }
                        items.append(List.from_ref(self._db, d))
                    else:
                        items.append(
                            {
                                field: rec[field]
                                for field in user_fields
                                if field != "valid"
                            }
                        )

                remaining -= len(arr)
                block_idx += 1
                offset = 0

            return items

        # Fallback: general case (deletions or non-unit step)
        items = []
        for i in range(start, stop, step):
            items.append(self[i])

        return items

    def __setitem__(self, index, item):
        """Set item at index.

        Args:
            index: Index (negative indices supported)
            item: Item dict matching item_schema

        Performance:
            - O(1) if no deletions
            - O(log n) if deletions present
        """
        # Handle negative indices
        if index < 0:
            index = self.valid_count + index

        if index < 0 or index >= self.valid_count:
            raise IndexError("list index out of range")

        # Fast path: no deletions
        if self.length == self.valid_count:
            block_idx, offset = self._calculate_block_and_offset(index)
            block_addr = self.blocks[block_idx]
            item_addr = block_addr + offset * self._items_dataset.record_size
            item_with_valid = {"valid": True, **item}
            self._items_dataset[item_addr] = item_with_valid

            # Invalidate cache
            if self._block_cache:
                cache_key = f"block_{block_idx}"
                if cache_key in self._block_cache:
                    del self._block_cache[cache_key]

            self._auto_save_check()
            return

        # Slow path: has deletions, need to find nth valid item
        block_idx, item_addr = self._find_nth_valid_item_address(index)

        item_with_valid = {"valid": True, **item}
        self._items_dataset[item_addr] = item_with_valid

        # Invalidate cache
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            if cache_key in self._block_cache:
                del self._block_cache[cache_key]

        self._auto_save_check()

    def update_nested_ref(self, index, nested_item):
        """Update the stored reference for a nested structure.

        This is needed because when a nested structure is created via append(),
        its reference is stored with the initial metadata (length=0, size=0).
        After adding items to the nested structure, call this to update the
        stored reference with the current metadata.

        Args:
            index: Index of the nested structure in this list
            nested_item: The nested List or Dict instance with updated data

        Raises:
            ValueError: If this is not a nested list
        """
        if not self._is_nested:
            raise ValueError("update_nested_ref only works on nested lists")

        # Get the updated reference
        ref = nested_item.to_ref()

        # Use __setitem__ logic to update the stored reference
        # Handle negative indices
        if index < 0:
            index = self.valid_count + index

        if index < 0 or index >= self.valid_count:
            raise IndexError("list index out of range")

        # Fast path: no deletions
        if self.length == self.valid_count:
            block_idx, offset = self._calculate_block_and_offset(index)
            block_addr = self.blocks[block_idx]
            item_addr = block_addr + offset * self._items_dataset.record_size
            ref_with_valid = {"valid": True, **ref}
            self._items_dataset[item_addr] = ref_with_valid

            # Invalidate cache for the affected block
            if self._block_cache:
                cache_key = f"block_{block_idx}"
                if cache_key in self._block_cache:
                    del self._block_cache[cache_key]
            return

        # Slow path: has deletions
        block_idx, item_addr = self._find_nth_valid_item_address(index)
        ref_with_valid = {"valid": True, **ref}
        self._items_dataset[item_addr] = ref_with_valid

        # Invalidate cache for the affected block
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            if cache_key in self._block_cache:
                del self._block_cache[cache_key]

    def __delitem__(self, index):
        """Delete item at index (soft delete).

        Args:
            index: Index (negative indices supported)

        Performance: O(log n)
        """
        # Handle negative indices
        if index < 0:
            index = self.valid_count + index

        if index < 0 or index >= self.valid_count:
            raise IndexError("list index out of range")

        # Find the nth valid item and mark as deleted
        if self.length == self.valid_count:
            # Fast path: no deletions yet, direct calculation
            block_idx, offset = self._calculate_block_and_offset(index)
            block_addr = self.blocks[block_idx]
            item_addr = block_addr + offset * self._items_dataset.record_size

            item = self._items_dataset[item_addr]
            item["valid"] = False
            self._items_dataset[item_addr] = item

            self.valid_count -= 1
            self.block_valid_counts[block_idx] -= 1
        else:
            # Slow path: has deletions, need to find nth valid item
            block_idx, item_addr = self._find_nth_valid_item_address(index)

            item = self._items_dataset[item_addr]
            item["valid"] = False
            self._items_dataset[item_addr] = item

            self.valid_count -= 1
            self.block_valid_counts[block_idx] -= 1

        # Invalidate cache
        if self._block_cache:
            cache_key = f"block_{block_idx}"
            if cache_key in self._block_cache:
                del self._block_cache[cache_key]

        # Auto-compact if too much waste
        deletion_ratio = 1 - (self.valid_count / self.length) if self.length > 0 else 0
        if deletion_ratio > self.COMPACT_THRESHOLD:
            self.compact()

        self._auto_save_check()

    def _find_nth_valid_item_address(self, n):
        """Find the address of the nth valid item.

        Args:
            n: Logical index (0-based, counting only valid items)

        Returns:
            (block_idx, item_addr): Block index and item address
        """
        if n < 0 or n >= self.valid_count:
            raise IndexError("list index out of range")

        cumulative_valid = 0

        for block_idx in range(len(self.blocks)):
            if self.blocks[block_idx] == 0:
                continue

            block_valid = self.block_valid_counts[block_idx]

            if n < cumulative_valid + block_valid:
                # Found the block! Now find nth valid item within block
                offset_in_block = n - cumulative_valid
                item_addr = self._get_nth_valid_address_in_block(
                    block_idx, offset_in_block
                )
                return block_idx, item_addr

            cumulative_valid += block_valid

        raise IndexError("list index out of range")

    def __len__(self):
        """Get number of valid items.

        Returns:
            Number of valid (non-deleted) items

        Performance: O(1)
        """
        return self.valid_count

    def __iter__(self):
        """Iterate over valid items.

        Yields:
            Item dicts (for regular list) or List instances (for nested list)

        Performance: O(n)
        """
        for block_idx in range(len(self.blocks)):
            if self.blocks[block_idx] == 0:
                continue

            items = self._read_block(block_idx)

            for item in items:
                if item.get("valid", False):
                    # Remove 'valid' flag
                    clean_item = {k: v for k, v in item.items() if k != "valid"}

                    # Resolve reference if nested
                    if self._is_nested:
                        yield self._resolve_nested_ref(clean_item)
                    else:
                        yield clean_item

    def compact(self):
        """Compact list by removing deleted items.

        Rebuilds list without deleted items, resetting to optimal
        block structure. This is an O(n) operation but restores
        O(1) indexing performance.

        Performance: O(n)
        """
        if self.length == self.valid_count:
            # No deletions, nothing to compact
            return

        # Collect all valid items
        valid_items = list(self)

        # Reset metadata
        old_blocks = self.blocks[:]
        self.length = 0
        self.valid_count = 0
        self.p_last = self.P_INIT - 1
        self.blocks = [0] * self.MAX_BLOCKS
        self.block_valid_counts = [0] * self.MAX_BLOCKS

        # Clear cache
        if self._block_cache:
            self._block_cache.clear()

        # Re-append all valid items atomically to ensure crash-safe rebuild
        self.append_many(valid_items, atomic=True)

        # Free old blocks back to the file-level freelist
        record_size = self._items_dataset.record_size
        for block_idx, block_addr in enumerate(old_blocks):
            if block_addr == 0:
                continue
            # Skip blocks that are reused by the new layout
            if block_addr in self.blocks:
                continue
            p = self.P_INIT + block_idx
            capacity = self._get_capacity(p)
            self._items_dataset.db.free(block_addr, capacity * record_size)

        self.save()

    def _get_ref_fields(self):
        """Get List-specific reference fields (for top-level lists)."""
        return {
            "ref_dataset_name": self._items_dataset.name,
            "growth_factor": self.GROWTH_FACTOR,
            "cache_size": self.cache_size,
            "p_init": self.P_INIT,
        }

    def to_ref(self):
        """Get reference to this list for storage in parent.

        For nested lists, uses compact binary format (~400 bytes).
        For top-level lists, uses standard format with metadata key.
        """
        if self._parent is not None and self._parent != "__nested__":
            # Nested list: use compact binary format
            ref = {"valid": True}

            # Dataset name for reconstruction
            ref["items_dataset"] = self._items_dataset.name

            # Core metadata as binary fields
            ref["length"] = self.length
            ref["valid_count"] = self.valid_count
            ref["p_last"] = self.p_last

            # Block addresses (up to 16 for nested)
            for i in range(self.MAX_BLOCKS_NESTED):
                ref[f"block_{i}"] = self.blocks[i] if i < len(self.blocks) else 0

            # Block valid counts
            for i in range(self.MAX_BLOCKS_NESTED):
                ref[f"bvc_{i}"] = (
                    self.block_valid_counts[i]
                    if i < len(self.block_valid_counts)
                    else 0
                )

            # Config
            ref["cache_size"] = self.cache_size

            return ref
        else:
            # Top-level: use base class implementation
            return super().to_ref()

    @classmethod
    def _from_ref_impl(cls, db, ref):
        """Reconstruct List from reference.

        Args:
            db: Database instance
            ref: Reference dict (binary format for nested, JSON for top-level)

        Returns:
            List instance
        """
        # Check if this is binary format (nested list) or JSON format (top-level)
        is_binary_format = "length" in ref and "block_0" in ref

        if is_binary_format:
            # Binary format: reconstruct from compact fields
            # Create instance and manually set all fields
            instance = object.__new__(cls)

            # Get the dataset
            dataset_name = str(ref["items_dataset"]).strip()
            instance._items_dataset = db.get_dataset(dataset_name)
            instance._db = db

            # Extract metadata from binary fields
            instance.length = int(ref["length"])
            instance.valid_count = int(ref["valid_count"])
            instance.p_last = int(ref["p_last"])

            # Reconstruct blocks array
            instance.blocks = [0] * cls.MAX_BLOCKS
            for i in range(cls.MAX_BLOCKS_NESTED):
                instance.blocks[i] = int(ref[f"block_{i}"])

            # Reconstruct block_valid_counts
            instance.block_valid_counts = [0] * cls.MAX_BLOCKS
            for i in range(cls.MAX_BLOCKS_NESTED):
                instance.block_valid_counts[i] = int(ref[f"bvc_{i}"])

            # Set other required attributes
            instance.name = f"_nested_{id(instance)}"
            instance._parent = "__nested__"
            instance._is_nested = False  # This nested list stores data, not refs
            instance._template = None
            instance._shared_items_dataset = None
            instance.cache_size = int(ref["cache_size"])
            instance._block_cache = instance._make_cache("blocks", instance.cache_size)
            instance._auto_save_interval = 0  # Don't auto-save nested lists
            instance._ops_since_save = 0
            instance._inline_metadata = None
            instance._metadata_key = None

            # Extract item schema from dataset
            instance.item_schema = {
                name: str(instance._items_dataset.user_schema.fields[name][0])
                for name in instance._items_dataset.user_schema.names
            }

            return instance

        # JSON format (legacy or top-level)
        instance = object.__new__(cls)

        # Set inline metadata if present (before __init__)
        if "inline_metadata" in ref and ref["inline_metadata"]:
            # Deserialize JSON string to dict
            metadata = json.loads(ref["inline_metadata"])
            instance._inline_metadata = metadata
        else:
            instance._inline_metadata = None
            metadata = None

        # Determine if this is a nested list (has inline_metadata) or top-level
        is_nested_ref = metadata is not None

        if metadata and metadata.get("is_nested"):
            # Nested list that contains other nested lists: reconstruct template
            template_dataset = db.get_dataset(metadata["template_dataset"])
            template_config = metadata["template_config"]
            template = DataStructureTemplate(cls, template_dataset, template_config)

            instance.__init__(
                ref["ds_name"],
                db,
                template,
                cache_size=ref["cache_size"],
                _parent="__nested__",
            )
        else:
            # Regular list (may be nested or top-level)
            dataset = db.get_dataset(ref["ref_dataset_name"])

            if is_nested_ref:
                instance.__init__(
                    ref["ds_name"],
                    db,
                    dataset,
                    cache_size=ref["cache_size"],
                    _parent="__nested__",
                )
            else:
                instance.__init__(
                    ref["ds_name"], db, dataset, cache_size=ref["cache_size"]
                )

        return instance

    def _get_current_metadata(self):
        """Get current metadata dict without saving.

        Used by to_ref() to include metadata inline for nested structures.

        Returns:
            Metadata dict
        """
        metadata = {
            "length": self.length,
            "valid_count": self.valid_count,
            "p_last": self.p_last,
            "blocks": self.blocks,
            "block_valid_counts": self.block_valid_counts,
            "dataset_name": self._items_dataset.name,  # Use actual dataset name
            "is_nested": self._is_nested,
        }

        # Add template info if nested (for list containers)
        if self._is_nested:
            metadata["template_dataset"] = self._template.dataset.name
            metadata["template_config"] = self._template.config
            metadata["template_class"] = self._template.ds_class.__name__

            # Save shared dataset names (modular approach)
            if hasattr(self, "_shared_datasets") and self._shared_datasets:
                metadata["shared_datasets"] = {
                    attr_name: dataset.name
                    for attr_name, dataset in self._shared_datasets.items()
                }

        return metadata

    def save(self, force=False):
        """Save metadata to disk.

        Args:
            force: If True, save even during parent operations (for nested structures)
        """
        # Skip auto-save during parent operations (but allow explicit save)
        if self._parent and not force:
            return

        metadata = self._get_current_metadata()
        self._save_metadata(metadata)

    # ---- Registry protocol ----

    def _get_registry_params(self):
        return {"schema": self.item_schema, "cache_size": self.cache_size}

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(name, db, params["schema"], params.get("cache_size", 10))

    def __repr__(self):
        """String representation."""
        deletion_ratio = 1 - (self.valid_count / self.length) if self.length > 0 else 0
        return (
            f"List('{self.name}', "
            f"length={self.valid_count}, "
            f"blocks={self.p_last - self.P_INIT + 1}, "
            f"waste={deletion_ratio:.1%})"
        )
