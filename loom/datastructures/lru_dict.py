"""
Persistent LRU Dict — fixed-capacity key-value store that evicts the
least recently used entry when full.

Storage layout
--------------
Two internal structures, fully self-contained:

  _lru_{name}_items  — Dataset with user fields + 3 hidden LRU fields:
      _lru_prev : uint64   address of the more-recently-used record
      _lru_next : uint64   address of the less-recently-used record
      _lru_key  : U{N}     stored key (needed to remove from index on evict)
      <user fields>

  _lru_{name}_index — Dict mapping key → {"record_addr": uint64}
      Pre-seeded with initial_capacity = 2 × capacity so it never grows.

  A contiguous block of exactly `capacity` records is pre-allocated in
  the items dataset.  On eviction, the freed tail slot is immediately
  reused for the new head entry — zero ByteFileDB allocation during
  steady-state operation.

Complexity
----------
  get   O(1) — index lookup + 4 field writes (move to head)
  set   O(1) — index lookup + 1 record write + 4 field writes
  del   O(1) — index delete + 4 field writes (unlink)
  evict O(1) — read tail key + index delete + reuse slot

Note on schema
--------------
The items dataset adds three hidden fields to the user schema.
A record stored in LRUDict is NOT schema-identical to the same record
in a plain List or Dict — consistent with how List adds a `valid` field.
"""

import numpy as np

from loom.datastructures.base import DataStructure
from loom.datastructures.dict import Dict


_NULL = 0   # sentinel: "no pointer" (address 0 is the header, never a record)


class LRUDict(DataStructure):
    """Persistent LRU Dict with O(1) get/set/evict."""

    # Nesting compatibility
    _outer_types_supported = ("Dict", "List")
    _inner_types_supported = ()

    def __init__(
        self,
        name: str,
        db,
        dataset_or_schema=None,
        capacity: int = 1000,
        key_size: int = 50,
        hash_keys: bool = False,
        hash_bits: int = 128,
        auto_save_interval=None,
        _parent=None,
    ):
        self.capacity   = capacity
        self._key_size  = key_size if not hash_keys else hash_bits // 4
        self._hash_keys = hash_keys
        self._hash_bits = hash_bits
        self._hash_key_fn = (
            Dict._make_hash_fn(hash_bits // 4) if hash_keys else None
        )
        self._resolve_schema(dataset_or_schema)

        super().__init__(name, db, auto_save_interval, _parent=_parent)

        meta = self._load_metadata()
        if meta:
            self._load()
        elif getattr(self, "_schema_dict", None) is not None:
            self._initialize()
        else:
            raise ValueError(
                "dataset_or_schema required for a new LRUDict"
            )

    # ── Schema resolution ─────────────────────────────────────────────────

    def _resolve_schema(self, src):
        self._schema_dict = None
        if src is None:
            return
        if hasattr(src, "model_fields"):          # Pydantic model
            from loom.schema import schema_from_model
            self._schema_dict = schema_from_model(src)
        elif isinstance(src, dict):
            self._schema_dict = src
        else:                                       # Dataset object
            raw = self._extract_schema(src)
            # strip any hidden fields from a re-used dataset
            self._schema_dict = {
                k: v for k, v in raw.items()
                if not k.startswith("_lru_")
            }

    def _user_schema(self):
        """Schema without the three hidden LRU fields."""
        return {
            k: v for k, v in self._extract_schema(self._items_ds).items()
            if not k.startswith("_lru_")
        }

    # ── Init / Load ───────────────────────────────────────────────────────

    def _initialize(self):
        ks = self._key_size

        # Items dataset: hidden LRU pointers + user fields
        items_schema = {
            "_lru_prev": "uint64",
            "_lru_next": "uint64",
            "_lru_key":  f"U{ks}",
            **self._schema_dict,
        }
        self._items_ds = self._db.create_dataset(
            f"_lru_{self.name}_items", **items_schema
        )

        # Pre-allocate exactly `capacity` record slots
        self._block_addr = int(
            self._items_ds.allocate_block(self.capacity)
        )

        # Index: key → {"record_addr": uint64}
        addr_ds = self._db.create_dataset(
            f"_lru_{self.name}_addrs", record_addr="uint64"
        )
        self._index = Dict(
            f"_lru_{self.name}_index",
            self._db,
            addr_ds,
            key_size=ks,
            initial_capacity=max(self.capacity * 2, 64),
            use_bloom=False,
            cache_size=min(self.capacity, 2000),
        )

        self._head      = _NULL   # most recently used record addr
        self._tail      = _NULL   # least recently used record addr
        self._size      = 0
        self._next_slot = 0       # bump pointer during fill phase
        self._del_slots = []      # freed slots from explicit deletes

        self._save_state()

    def _load(self):
        meta = self._load_metadata()

        self._items_ds   = self._db.get_dataset(meta["items_ds_name"])
        self._block_addr = meta["block_addr"]
        self.capacity    = meta["capacity"]
        self._key_size   = meta["key_size"]
        self._hash_keys  = meta.get("hash_keys", False)
        self._hash_bits  = meta.get("hash_bits", 128)
        self._hash_key_fn = (
            Dict._make_hash_fn(self._hash_bits // 4)
            if self._hash_keys else None
        )
        self._head      = meta["head"]
        self._tail      = meta["tail"]
        self._size      = meta["size"]
        self._next_slot = meta["next_slot"]
        self._del_slots = meta.get("del_slots", [])

        # Index resolved lazily — may not be in _datastructures yet
        # if the registry loop hasn't reached it.
        self._index = None
        self._index_name = f"_lru_{self.name}_index"

        # Build user schema dict (strip LRU fields)
        self._schema_dict = self._user_schema()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_state(self):
        self._save_metadata({
            "items_ds_name": self._items_ds.name,
            "block_addr":    self._block_addr,
            "capacity":      self.capacity,
            "key_size":      self._key_size,
            "hash_keys":     self._hash_keys,
            "hash_bits":     self._hash_bits,
            "head":          self._head,
            "tail":          self._tail,
            "size":          self._size,
            "next_slot":     self._next_slot,
            "del_slots":     self._del_slots,
        })

    def save(self, force=False):
        self._save_state()

    # ── Key normalisation ─────────────────────────────────────────────────

    def _ensure_index(self):
        """Resolve _index lazily (may not exist at _load() time)."""
        if self._index is None:
            idx_name = getattr(self, "_index_name",
                               f"_lru_{self.name}_index")
            self._index = self._db._datastructures[idx_name]

    def _ikey(self, key) -> str:
        """Internal key (hashed if hash_keys=True)."""
        if self._hash_key_fn:
            return self._hash_key_fn(key)
        return key if isinstance(key, str) else str(key)

    # ── Direct mmap field helpers (bypass Dataset prefix check) ───────────

    def _foffset(self, field: str) -> int:
        return self._items_ds.schema.fields[field][1]

    def _read_u64(self, addr: int, field: str) -> int:
        off = self._foffset(field)
        return int(np.frombuffer(
            self._items_ds.db.read(addr + off, 8), dtype="uint64"
        )[0])

    def _write_u64(self, addr: int, field: str, val: int):
        off = self._foffset(field)
        self._items_ds.db.write(addr + off, np.uint64(val).tobytes())

    def _read_prev(self, addr): return self._read_u64(addr, "_lru_prev")
    def _read_next(self, addr): return self._read_u64(addr, "_lru_next")
    def _write_prev(self, addr, v): self._write_u64(addr, "_lru_prev", v)
    def _write_next(self, addr, v): self._write_u64(addr, "_lru_next", v)

    def _read_key(self, addr) -> str:
        """Read the stored internal key from a record."""
        off   = self._foffset("_lru_key")
        ks    = self._key_size
        raw   = self._items_ds.db.read(addr + off, ks * 4)
        chars = np.frombuffer(raw, dtype=f"U{ks}")
        return str(chars[0])

    # ── Doubly-linked list ────────────────────────────────────────────────

    def _unlink(self, addr: int):
        """Remove addr from its current position without touching its data."""
        prev = self._read_prev(addr)
        nxt  = self._read_next(addr)

        if prev != _NULL:
            self._write_next(prev, nxt)
        else:
            self._head = nxt          # addr was head

        if nxt != _NULL:
            self._write_prev(nxt, prev)
        else:
            self._tail = prev         # addr was tail

    def _prepend(self, addr: int):
        """Attach addr at the head (most recently used position)."""
        self._write_prev(addr, _NULL)
        self._write_next(addr, self._head)

        if self._head != _NULL:
            self._write_prev(self._head, addr)
        else:
            self._tail = addr         # first entry

        self._head = addr

    def _move_to_head(self, addr: int):
        if addr == self._head:
            return
        self._unlink(addr)
        self._prepend(addr)

    # ── Slot management ───────────────────────────────────────────────────

    def _alloc_slot(self) -> int:
        """Return address of a free slot (fill phase or from deleted list)."""
        if self._del_slots:
            return self._del_slots.pop()
        addr = self._block_addr + self._next_slot * self._items_ds.record_size
        self._next_slot += 1
        return addr

    def _evict_tail(self) -> int:
        """Evict LRU entry, return its slot address for immediate reuse."""
        if self._tail == _NULL:
            raise RuntimeError("LRUDict is empty, cannot evict")

        tail_addr = self._tail
        tail_key  = self._read_key(tail_addr)
        del self._index[tail_key]
        self._unlink(tail_addr)
        self._size -= 1
        return tail_addr

    # ── Public API ────────────────────────────────────────────────────────

    def __setitem__(self, key, value: dict):
        if not isinstance(value, dict):
            raise TypeError(
                f"value must be a dict, got {type(value).__name__}"
            )
        self._ensure_index()
        ikey = self._ikey(key)

        existing = self._index.get(ikey)
        if existing is not None:
            # Update existing record in place, then move to head
            addr = int(existing["record_addr"])
            for field, val in value.items():
                if field in self._items_ds.user_schema.names:
                    self._items_ds.write_field(addr, field, val)
            self._move_to_head(addr)
        else:
            # New entry: evict if full, else bump-allocate
            if self._size >= self.capacity:
                addr = self._evict_tail()
            else:
                addr = self._alloc_slot()

            # Write record (LRU fields will be overwritten by _prepend)
            record = {
                "_lru_prev": _NULL,
                "_lru_next": _NULL,
                "_lru_key":  ikey,
                **value,
            }
            self._items_ds.write(addr, **record)

            self._index[ikey] = {"record_addr": addr}
            self._prepend(addr)
            self._size += 1

        self._auto_save_check()

    def __getitem__(self, key) -> dict:
        self._ensure_index()
        ikey = self._ikey(key)
        ptr  = self._index.get(ikey)
        if ptr is None:
            raise KeyError(key)

        addr = int(ptr["record_addr"])
        rec  = self._items_ds.read(addr)
        self._move_to_head(addr)
        self._auto_save_check()

        return {k: v for k, v in rec.items() if not k.startswith("_lru_")}

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __delitem__(self, key):
        self._ensure_index()
        ikey = self._ikey(key)
        ptr  = self._index.get(ikey)
        if ptr is None:
            raise KeyError(key)

        addr = int(ptr["record_addr"])
        self._unlink(addr)
        del self._index[ikey]
        self._del_slots.append(addr)   # make slot available for reuse
        self._size -= 1
        self._auto_save_check()

    def __contains__(self, key) -> bool:
        self._ensure_index()
        return self._index.get(self._ikey(key)) is not None

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def __iter__(self):
        """Iterate (key, value) from most recent to least recent."""
        addr = self._head
        while addr != _NULL:
            rec   = self._items_ds.read(addr)
            ikey  = str(rec.get("_lru_key", ""))
            value = {k: v for k, v in rec.items() if not k.startswith("_lru_")}
            nxt   = int(rec.get("_lru_next", _NULL))
            yield ikey, value
            addr  = nxt

    def items(self):
        return self.__iter__()

    def keys(self):
        for k, _ in self:
            yield k

    def values(self):
        for _, v in self:
            yield v

    def __repr__(self) -> str:
        return (
            f"LRUDict('{self.name}', "
            f"size={self._size}/{self.capacity}, "
            f"hash_keys={self._hash_keys})"
        )

    # ── Registry protocol ─────────────────────────────────────────────────

    def _get_registry_params(self):
        if self._parent:
            return None
        return {
            "schema":    self._schema_dict,
            "capacity":  self.capacity,
            "key_size":  self._key_size,
            "hash_keys": self._hash_keys,
            "hash_bits": self._hash_bits,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db, None,
            capacity  = params.get("capacity",  1000),
            key_size  = params.get("key_size",  50),
            hash_keys = params.get("hash_keys", False),
            hash_bits = params.get("hash_bits", 128),
        )

    # _initialize and _load are defined above — they satisfy the DataStructure ABC.
