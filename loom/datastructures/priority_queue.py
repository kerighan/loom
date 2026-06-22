"""PriorityQueue — a persistent priority queue backed by a BTree.

``push(item, priority)`` and ``pop()`` / ``peek()`` run in O(log n).  The
highest-priority item pops first by default (a max-priority queue); set
``max_first=False`` for a min-priority queue.  Items sharing a priority pop in
FIFO order — a monotonic sequence number breaks ties.

Priorities are encoded with the same order-preserving scheme as Collection
indexes (``encode_value``), so int, float and datetime priorities all sort
correctly.  Use one consistent priority type per queue.

    pq = db.create_priority_queue("jobs", {"task": "utf8[40]"})
    pq.push({"task": "send email"}, priority=5)
    pq.push({"task": "reindex"},    priority=9)
    pq.pop()        # -> {"task": "reindex"}   (priority 9 first)
"""

from __future__ import annotations

from loom.collection import _SEP, encode_value
from loom.dataset import as_record
from loom.datastructures.base import DataStructure


class PriorityQueueEmpty(Exception):
    """Raised by pop()/peek() on an empty queue when no default is given."""


_RAISE = object()


class PriorityQueue(DataStructure):
    def __init__(self, name, db, schema=None, max_first=True, _parent=None):
        self._schema = schema
        self._schema_name = None
        self._max_first = max_first
        self._bt = None
        self._meta = None
        self._seq = 0
        super().__init__(name, db, _parent=_parent)
        if self._load_metadata():
            self._load()
        else:
            self._initialize()

    # ── names of the internal structures ──────────────────────────────────────
    def _bt_name(self):
        return f"_pq_{self.name}_bt"

    def _meta_name(self):
        return f"_pq_{self.name}_meta"

    # ── construction / persistence ────────────────────────────────────────────
    def _initialize(self):
        from loom.dataset import Dataset

        if self._schema is None:
            raise ValueError("create_priority_queue requires a value schema")
        if isinstance(self._schema, Dataset):
            ds = self._schema
        elif isinstance(self._schema, dict):
            ds = self._db.create_dataset(f"_pq_{self.name}_vals", **self._schema)
        else:
            ds = self._db.create_dataset(f"_pq_{self.name}_vals", self._schema)
        self._schema_name = ds.name
        # composite key = encode_value(priority) + SEP + 20-digit sequence
        self._bt = self._db.create_btree(self._bt_name(), ds, key_size=64)
        mds = self._db.create_dataset(f"_pq_{self.name}_metads", v="int64")
        self._meta = self._db.create_dict(self._meta_name(), mds)
        self._meta["seq"] = {"v": 0}
        self._seq = 0
        self.save()

    def _load(self):
        m = self._load_metadata()
        self._schema_name = m["schema_name"]
        self._max_first = m.get("max_first", True)
        self._bt = None
        self._meta = None

    def _ensure_loaded(self):
        if self._bt is not None:
            return
        self._bt = self._db._datastructures[self._bt_name()]
        self._meta = self._db._datastructures[self._meta_name()]
        self._seq = int(self._meta["seq"]["v"])

    def save(self, force=False):
        self._save_metadata(self._config())

    def _config(self):
        return {"schema_name": self._schema_name, "max_first": self._max_first}

    def _get_registry_params(self):
        return self._config()

    @classmethod
    def _from_registry_params(cls, name, db, params):
        # schema is recovered from metadata in _load(); only max_first matters here
        return cls(name, db, max_first=params.get("max_first", True))

    # ── key construction ──────────────────────────────────────────────────────
    def _key(self, priority, seq):
        return encode_value(priority, desc=self._max_first) + _SEP + f"{seq:020d}"

    # ── operations ────────────────────────────────────────────────────────────
    def push(self, item, priority):
        """Enqueue ``item`` with the given ``priority`` (O(log n))."""
        self._ensure_loaded()
        self._bt[self._key(priority, self._seq)] = as_record(item)
        self._seq += 1
        self._meta["seq"] = {"v": self._seq}

    def push_many(self, items):
        """Bulk-enqueue an iterable of ``(item, priority)`` pairs.

        On an empty queue this uses the BTree's O(n) bulk_load; otherwise it
        falls back to individual inserts."""
        self._ensure_loaded()
        batch = []
        for item, priority in items:
            batch.append((self._key(priority, self._seq), as_record(item)))
            self._seq += 1
        if not batch:
            return
        if len(self._bt) == 0:
            self._bt.bulk_load(batch)
        else:
            for key, item in batch:
                self._bt[key] = item
        self._meta["seq"] = {"v": self._seq}

    def pop(self, default=_RAISE):
        """Remove and return the highest-priority item (FIFO on ties)."""
        self._ensure_loaded()
        key = self._bt.min()
        if key is None:
            if default is _RAISE:
                raise PriorityQueueEmpty(f"priority queue {self.name!r} is empty")
            return default
        value = self._bt[key]
        del self._bt[key]
        return value

    def peek(self, default=_RAISE):
        """Return the highest-priority item without removing it."""
        self._ensure_loaded()
        key = self._bt.min()
        if key is None:
            if default is _RAISE:
                raise PriorityQueueEmpty(f"priority queue {self.name!r} is empty")
            return default
        return self._bt[key]

    def __len__(self):
        self._ensure_loaded()
        return len(self._bt)

    def __repr__(self):
        order = "max-first" if self._max_first else "min-first"
        try:
            n = len(self)
        except Exception:
            n = "?"
        return f"PriorityQueue('{self.name}', {order}, n={n})"
