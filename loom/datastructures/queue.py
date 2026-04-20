"""
Persistent Queue (FIFO) with O(1) push/pop and automatic space reclamation.

Design
------
Items are stored in fixed-size *blocks* of `block_size` records each.
The queue maintains a list of active blocks: the head block is consumed
by pop(), the tail block is filled by push().

When pop() exhausts a block it calls ByteFileDB.free() — the block is
returned to the shared freelist and can immediately be reused by any
structure in the same DB.  Because block_size is constant for a given
queue, blocks freed by pop() have exactly the right size for the next
push() allocation: in steady state, the file does not grow.

Layout
------
    blocks = [addr_head, addr_1, ..., addr_tail]   <- only live blocks
    head_offset  = index of next item to pop   in blocks[0]
    tail_offset  = index of next slot to push  in blocks[-1]
    size         = total items currently queued

Metadata is stored in the DB header (like List/Dict).

Usage
-----
    from pydantic import BaseModel, Field

    class Task(BaseModel):
        id:      int
        payload: str = Field(max_length=100)
        priority: float

    with DB("work.db") as db:
        q = db.create_queue("tasks", Task, block_size=64)

        q.push({"id": 1, "payload": "hello", "priority": 0.9})
        q.push({"id": 2, "payload": "world", "priority": 0.5})

        item = q.peek()          # {'id': 1, ...}  — doesn't remove
        item = q.pop()           # {'id': 1, ...}  — removes
        print(len(q))            # 1

        for item in q:           # non-destructive iteration
            print(item)

Performance
-----------
    push : O(1) — sequential mmap write
    pop  : O(1) — sequential mmap read + occasional ByteFileDB.free()
    peek : O(1)
    len  : O(1)
    iter : O(n) sequential scan
"""

import numpy as np

from loom.datastructures.base import DataStructure


class Queue(DataStructure):
    """Persistent FIFO queue with block-based storage and auto reclamation."""

    DEFAULT_BLOCK_SIZE = 64

    def __init__(
        self,
        name: str,
        db,
        dataset_or_schema=None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        auto_save_interval=None,
        _parent=None,
    ):
        """
        Args:
            name:              Unique name for this queue
            db:                DB instance
            dataset_or_schema: Dataset, schema dict, Pydantic model, or None
                               to load an existing queue.
            block_size:        Records per block (default 64).  A larger value
                               reduces block allocation overhead; smaller
                               reduces wasted space for tiny queues.
        """
        self.block_size = block_size
        self._resolve_schema(dataset_or_schema)

        super().__init__(name, db, auto_save_interval, _parent=_parent)

        metadata = self._load_metadata()
        has_schema = self._items_dataset is not None or getattr(self, "_schema_dict", None) is not None
        if metadata:
            self._load()
        elif has_schema:
            self._initialize()
        else:
            raise ValueError("dataset_or_schema required for a new Queue")

    # ── Schema resolution ─────────────────────────────────────────────────────

    def _resolve_schema(self, dataset_or_schema):
        """Accept Dataset, dict schema, Pydantic model, or None (load)."""
        self._items_dataset = None
        self.item_schema = None

        if dataset_or_schema is None:
            return  # will load from metadata

        # Pydantic model class
        if hasattr(dataset_or_schema, "model_fields"):
            from loom.schema import schema_from_model
            self._schema_dict = schema_from_model(dataset_or_schema)
            return

        # Plain dict schema
        if isinstance(dataset_or_schema, dict):
            self._schema_dict = dataset_or_schema
            return

        # Dataset object
        self._items_dataset = dataset_or_schema
        self.item_schema = self._extract_schema(dataset_or_schema)
        self._schema_dict = None

    # ── Init / load ───────────────────────────────────────────────────────────

    def _initialize(self):
        """First-time setup: create the items dataset and first block."""
        # If schema was given as dict/Pydantic rather than Dataset, create it
        if self._items_dataset is None:
            schema = getattr(self, "_schema_dict", None)
            if schema is None:
                raise ValueError("No schema available to create Queue dataset")
            self._items_dataset = self._db.create_dataset(
                f"_queue_{self.name}_items", **schema
            )

        self.item_schema = self._extract_schema(self._items_dataset)

        # Allocate first block
        first_addr = self._alloc_block()

        self._blocks = [int(first_addr)]
        self._head_offset = 0   # next pop reads here (in blocks[0])
        self._tail_offset = 0   # next push writes here (in blocks[-1])
        self._size = 0

        self._save_state()

    def _load(self):
        """Restore from persisted metadata."""
        meta = self._load_metadata()
        self._items_dataset = self._db.get_dataset(meta["dataset_name"])
        self.item_schema = self._extract_schema(self._items_dataset)
        self.block_size = meta["block_size"]
        self._blocks = meta["blocks"]
        self._head_offset = meta["head_offset"]
        self._tail_offset = meta["tail_offset"]
        self._size = meta["size"]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_state(self):
        self._save_metadata({
            "dataset_name": self._items_dataset.name,
            "block_size": self.block_size,
            "blocks": self._blocks,
            "head_offset": self._head_offset,
            "tail_offset": self._tail_offset,
            "size": self._size,
        })

    def save(self, force=False):
        self._save_state()

    # ── Block management ──────────────────────────────────────────────────────

    def _alloc_block(self) -> int:
        """Allocate a new block from ByteFileDB (or freelist)."""
        size = self.block_size * self._items_dataset.record_size
        return int(self._items_dataset.db.allocate(size))

    def _free_head_block(self):
        """Return the exhausted head block to the ByteFileDB freelist."""
        addr = self._blocks.pop(0)
        size = self.block_size * self._items_dataset.record_size
        self._items_dataset.db.free(addr, size)

    def _item_addr(self, block_idx: int, offset: int) -> int:
        """Compute mmap address of a record."""
        return (
            self._blocks[block_idx]
            + offset * self._items_dataset.record_size
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def push(self, item: dict):
        """Enqueue an item at the tail.

        Args:
            item: Dict matching the queue's schema.

        Performance: O(1) amortized (occasional block allocation)
        """
        if not isinstance(item, dict):
            raise TypeError(f"item must be a dict, got {type(item).__name__}")

        # If tail block is full, allocate a new one
        if self._tail_offset >= self.block_size:
            new_addr = self._alloc_block()
            self._blocks.append(int(new_addr))
            self._tail_offset = 0

        addr = self._item_addr(-1, self._tail_offset)
        self._items_dataset[addr] = item

        self._tail_offset += 1
        self._size += 1
        self._auto_save_check()

    def pop(self) -> dict:
        """Dequeue and return the front item.

        Returns:
            Item dict

        Raises:
            IndexError: if the queue is empty
        """
        if self._size == 0:
            raise IndexError("pop from empty queue")

        addr = self._item_addr(0, self._head_offset)
        item = self._items_dataset[addr]
        # Strip the 'valid' prefix byte residue from Dataset read
        item = {k: v for k, v in item.items()}

        self._head_offset += 1
        self._size -= 1

        # If we've consumed the entire head block, free it
        if self._head_offset >= self.block_size and self._size > 0:
            self._free_head_block()
            self._head_offset = 0
        elif self._head_offset >= self.block_size and self._size == 0:
            # Queue is now empty — keep one block (the tail) to avoid
            # immediately re-allocating on the next push
            self._free_head_block()
            self._head_offset = 0
            self._tail_offset = 0
            # Allocate a fresh block so push() always has somewhere to write
            new_addr = self._alloc_block()
            self._blocks = [int(new_addr)]

        self._auto_save_check()
        return item

    def peek(self) -> dict:
        """Return the front item without removing it.

        Raises:
            IndexError: if the queue is empty
        """
        if self._size == 0:
            raise IndexError("peek at empty queue")

        addr = self._item_addr(0, self._head_offset)
        return self._items_dataset[addr]

    # ── Batch push ────────────────────────────────────────────────────────────

    def push_many(self, items):
        """Push multiple items efficiently using numpy bulk writes.

        Fills the current tail block, then allocates new blocks as needed.
        Significantly faster than repeated push() for bulk ingestion.

        Args:
            items: Iterable of item dicts
        """
        items = list(items)
        if not items:
            return

        ds = self._items_dataset
        record_size = ds.record_size

        for item in items:
            if self._tail_offset >= self.block_size:
                new_addr = self._alloc_block()
                self._blocks.append(int(new_addr))
                self._tail_offset = 0

            addr = self._item_addr(-1, self._tail_offset)
            ds[addr] = item
            self._tail_offset += 1
            self._size += 1

        self._auto_save_check()

    # ── Iteration (non-destructive) ───────────────────────────────────────────

    def __iter__(self):
        """Iterate over all items without consuming them.

        Yields items from head to tail in FIFO order.
        """
        ds = self._items_dataset
        for block_idx, block_addr in enumerate(self._blocks):
            start = self._head_offset if block_idx == 0 else 0
            end = (
                self._tail_offset
                if block_idx == len(self._blocks) - 1
                else self.block_size
            )
            for offset in range(start, end):
                addr = block_addr + offset * ds.record_size
                yield ds[addr]

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __repr__(self) -> str:
        return (
            f"Queue('{self.name}', size={self._size}, "
            f"blocks={len(self._blocks)}, block_size={self.block_size})"
        )

    # ── Registry protocol ─────────────────────────────────────────────────────

    def _get_registry_params(self):
        return {
            "schema": self.item_schema,
            "block_size": self.block_size,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(name, db, None, block_size=params.get("block_size", 64))

    # _initialize and _load are already defined above — they satisfy the ABC.
