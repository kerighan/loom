"""
Database orchestrator for managing multiple datasets and data structures.

Provides high-level API with:
- Automatic schema persistence
- Dataset registry management
- Data structure factory methods
- Context manager support
- Dict-like access
"""

from contextlib import contextmanager

from loom.fileio import ByteFileDB
from loom.dataset import Dataset
from loom.datastructures import BloomFilter, CountingBloomFilter, List, Set
from loom.blob import BlobStore
from loom.errors import (
    DatabaseNotOpenError,
    DuplicateNameError,
    StructureNotFoundError,
)


class DB:
    """High-level database orchestrator.

    Manages multiple datasets with automatic schema persistence.

    Usage:
        # Create and use
        db = DB('app.db')
        users = db.create_dataset('users', id='uint64', name='U50')
        users[addr] = {'id': 1, 'name': 'Alice'}
        db.close()

        # Later, in another file
        db = DB('app.db')
        users = db['users']  # Schema loaded automatically!
        record = users[addr]

        # Or with context manager
        with DB('app.db') as db:
            users = db.create_dataset('users', id='uint64', name='U50')
            users[addr] = {'id': 1, 'name': 'Alice'}
    """

    REGISTRY_KEY = "_dataset_registry"
    DATASTRUCTURES_REGISTRY_KEY = "_datastructures_registry"
    NEXT_ID_KEY = "_next_identifier"
    BLOB_COMPRESSION_KEY = "_blob_compression"

    def __init__(
        self,
        filename,
        initial_size=1024,
        header_size=32768,
        auto_open=True,
        blob_compression="brotli",
        auto_save_interval=100,
        cache_size=0,
        sync_writes=False,
    ):
        """Initialize database.

        Args:
            filename: Path to database file
            initial_size: Initial file size in bytes
            header_size: Header region size in bytes
            auto_open: Automatically open database (default: True)
            blob_compression: Compression for blobs ("brotli", "zlib", or None)
            auto_save_interval: Auto-save metadata every N operations per
                data structure.  Lower = safer (less data lost on crash),
                higher = faster.  Use 0 to disable (manual save only).
                Default: 100.
            cache_size: Total number of entries shared across ALL data
                structures in this DB.  0 (default) = per-structure caches
                with their own individual sizes (backward-compatible).
                Set to a positive integer to activate the shared cache:
                    DB(path, cache_size=50_000)  # 50K entries total
                This lets hot structures grow their share naturally via LRU
                without pre-allocating per-structure budgets.
            sync_writes: If True, flush mmap to disk after every header write
                (slow but fully durable — use for long-running servers).
                If False (default), flush only on close() — fast, but data
                may not be on disk immediately after each write.
                For servers, prefer calling db.flush() periodically, or use
                sync_writes=True for full durability.
        """
        self.filename = filename
        self.auto_save_interval = auto_save_interval

        # Shared cache: one LRU for the entire DB, namespaced per-structure
        if cache_size > 0:
            from loom.cache import LRUCache
            self._shared_cache = LRUCache(cache_size)
        else:
            self._shared_cache = None
        self._db = ByteFileDB(filename, initial_size, header_size,
                              sync_writes=sync_writes)
        self._datasets = {}  # name -> Dataset instance
        self._datastructures = {}  # name -> DataStructure instance
        self._is_open = False
        self._loading_registry = False  # Flag to prevent saving during load
        self._blob_compression = blob_compression
        self._blob_store = None  # Lazy initialized

        # Auto-open by default for convenience
        if auto_open:
            self.open()

    def open(self):
        """Open database and load dataset registry.

        Note: Usually not needed - database opens automatically on __init__.
        Only use if you created with auto_open=False.
        """
        if self._is_open:
            return self  # Already open

        self._db.open()
        self._is_open = True
        self._load_registry()

        # Load blob compression setting from header (for existing DBs)
        saved_compression = self._db.get_header_field(self.BLOB_COMPRESSION_KEY)
        if saved_compression is not None:
            self._blob_compression = saved_compression

        return self

    @property
    def blob_store(self):
        """Get the blob store (lazy initialized)."""
        if self._blob_store is None:
            self._blob_store = BlobStore(self._db, compression=self._blob_compression)
            # Save compression setting
            self._db.set_header_field(self.BLOB_COMPRESSION_KEY, self._blob_compression)
        return self._blob_store

    def close(self):
        """Close database and save all data structures."""
        if self._is_open:
            # Auto-save all data structures before closing
            for ds in self._datastructures.values():
                if hasattr(ds, "save"):
                    try:
                        ds.save(force=True)
                    except TypeError:
                        ds.save()

            # Save blob freelist before closing
            if self._blob_store is not None:
                self._blob_store._save_freelist()

            self._db.close()
            self._is_open = False
            self._datasets.clear()
            self._datastructures.clear()
            self._blob_store = None

    # -------------------------------------------------------------------------
    # Blob methods
    # -------------------------------------------------------------------------

    def write_blob(self, data: bytes) -> tuple[int, int]:
        """Write blob data and return reference.

        Args:
            data: Raw bytes to store

        Returns:
            Tuple of (offset, n_slots) to store in dataset record
        """
        return self.blob_store.write(data)

    def read_blob(self, offset: int) -> bytes:
        """Read blob data from offset.

        Args:
            offset: Blob offset (from write_blob)

        Returns:
            Original uncompressed data
        """
        return self.blob_store.read(offset)

    def delete_blob(self, offset: int, n_slots: int):
        """Delete blob and return space to freelist.

        Args:
            offset: Blob offset
            n_slots: Number of slots (from write_blob)
        """
        self.blob_store.delete(offset, n_slots)

    def _load_registry(self):
        """Load dataset registry from header and recreate Dataset instances."""
        registry = self._db.get_header_field(self.REGISTRY_KEY, {})

        for name, info in registry.items():
            identifier = info["identifier"]
            schema = info["schema"]

            # Pass blob_store if the schema uses variable-length fields
            needs_blobs = any(v in ("blob", "text") for v in schema.values())
            blob_store = self.blob_store if needs_blobs else None

            # Recreate Dataset instance
            dataset = Dataset(name, self._db, identifier, blob_store=blob_store, **schema)
            self._datasets[name] = dataset

        # Load data structures registry (generic — no per-type switch)
        from loom.datastructures.base import _DS_REGISTRY

        ds_registry = self._db.get_header_field(self.DATASTRUCTURES_REGISTRY_KEY, {})

        self._loading_registry = True
        try:
            for name, info in ds_registry.items():
                ds_type = info["type"]
                params = info["params"]

                ds_class = _DS_REGISTRY.get(ds_type)
                if ds_class is None:
                    continue  # Unknown type — skip gracefully

                if name not in self._datastructures:
                    ds_class._from_registry_params(name, self, params)
        finally:
            self._loading_registry = False

    def _save_registry(self):
        """Save dataset registry to header."""
        registry = {}
        for name, dataset in self._datasets.items():
            # Store schema as field_name -> dtype_string mapping.
            # Preserve the original "blob"/"text" markers so they survive
            # round-trips (numpy's .str would give '|V10' for BLOB_DTYPE).
            schema = {}
            for field_name in dataset.user_schema.names:
                schema[field_name] = self._dtype_to_registry_str(dataset, field_name)

            registry[name] = {"identifier": dataset.identifier, "schema": schema}

        self._db.set_header_field(self.REGISTRY_KEY, registry)

    @staticmethod
    def _dtype_to_registry_str(dataset, field_name):
        """Serialize one field's dtype for the registry, preserving array shapes."""
        from loom.dataset import dtype_to_str
        if field_name in dataset._text_fields:
            return "text"
        if field_name in dataset._blob_fields:
            return "blob"
        dtype = dataset.user_schema.fields[field_name][0]
        return dtype_to_str(dtype)

    def _save_datastructures_registry(self):
        """Save data structures registry to header (generic — no per-type switch)."""
        if self._loading_registry:
            return

        ds_registry = {}
        for name, ds in self._datastructures.items():
            params = ds._get_registry_params()
            if params is not None:
                ds_registry[name] = {
                    "type": type(ds).__name__,
                    "params": params,
                }

        self._db.set_header_field(self.DATASTRUCTURES_REGISTRY_KEY, ds_registry)

    def _get_next_identifier(self):
        """Get next available identifier and increment."""
        next_id = self._db.get_header_field(self.NEXT_ID_KEY, 1)
        if next_id > 127:
            raise ValueError("Maximum number of datasets (127) reached")

        self._db.set_header_field(self.NEXT_ID_KEY, next_id + 1)
        return next_id

    def flush(self):
        """Flush all pending mmap writes to disk immediately.

        In the default sync_writes=False mode, writes accumulate in the
        OS page cache and are only guaranteed on disk after close() or
        this method.

        For long-running servers, call this periodically:
            import schedule
            schedule.every(30).seconds.do(db.flush)

        Or after critical transactions:
            db["key"] = value
            db.flush()   # now on disk regardless of OS writeback schedule
        """
        if not self._is_open:
            return
        if self._blob_store is not None:
            self._blob_store._save_freelist()
        self._db.flush()

    @contextmanager
    def batch(self):
        """Batch mode: defer all header flushes until the block exits.

        Use this for bulk inserts to avoid per-operation header writes.
        Data is still written to mmap immediately, only the header
        (allocation index, metadata) flush is deferred.

        Example:
            with db.batch():
                for i in range(10000):
                    dataset.insert({'id': i, 'content': 'hello'})
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._db.begin_batch()
        try:
            yield
        finally:
            self._db.end_batch()

    def apply_writes(self, writes):
        if not self._is_open:
            raise DatabaseNotOpenError()

        if not writes:
            return

        self._db.transaction(writes)

    @contextmanager
    def write_batch(self):
        if not self._is_open:
            raise DatabaseNotOpenError()

        writes = []
        try:
            yield writes
            if writes:
                self._db.transaction(writes)
        finally:
            pass

    def create_dataset(self, dataset_name, model=None, **schema):
        """Create a new dataset with automatic identifier assignment.

        Args:
            dataset_name: Dataset name (must be unique)
            model: Optional Pydantic BaseModel class — schema is derived
                   from its fields (int→int64, float→float64, str→text, etc.)
            **schema: Field definitions as numpy dtypes (ignored if model given)

        Returns:
            Dataset instance

        Examples:
            # Classic kwargs
            users = db.create_dataset('users', id='uint64', name='U50', age='int32')

            # Pydantic model
            class User(BaseModel):
                id: int
                name: str
                score: float
            users = db.create_dataset('users', User)
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        # Accept Pydantic model as positional arg
        if model is not None and not schema:
            from loom.schema import schema_from_model
            schema = schema_from_model(model)

        if dataset_name in self._datasets:
            raise DuplicateNameError(dataset_name)

        # Get next identifier
        identifier = self._get_next_identifier()

        # Pass blob_store if schema uses variable-length fields
        needs_blobs = any(v in ("blob", "text") for v in schema.values())
        blob_store = self.blob_store if needs_blobs else None

        # Create dataset
        dataset = Dataset(dataset_name, self._db, identifier, blob_store=blob_store, **schema)
        self._datasets[dataset_name] = dataset

        # Save registry
        self._save_registry()

        return dataset

    def get_dataset(self, name):
        """Get an existing dataset by name.

        Args:
            name: Dataset name

        Returns:
            Dataset instance

        Raises:
            KeyError: If dataset doesn't exist
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        if name not in self._datasets:
            raise StructureNotFoundError(name)

        return self._datasets[name]

    def __getitem__(self, name):
        """Get a dataset or data structure by name.

        Args:
            name: Name of the dataset or data structure

        Returns:
            Dataset or data structure instance

        Raises:
            KeyError: If name not found

        Example:
            users = db["users"]  # Get existing list/dataset by name
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        # Check data structures first (List, BloomFilter, etc.)
        if name in self._datastructures:
            return self._datastructures[name]

        # Then check datasets
        if name in self._datasets:
            return self._datasets[name]

        raise StructureNotFoundError(name)

    def __contains__(self, name):
        """Check if a name exists in datasets or data structures."""
        return name in self._datasets or name in self._datastructures

    def has_dataset(self, name):
        """Check if a dataset exists.

        Args:
            name: Dataset name

        Returns:
            True if dataset exists
        """
        return name in self._datasets

    def list_datasets(self):
        """List all dataset names.

        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())

    def delete_dataset(self, name):
        """Delete a dataset from the registry.

        Note: This only removes the dataset from the registry.
        The actual data remains in the file.

        Args:
            name: Dataset name

        Raises:
            KeyError: If dataset doesn't exist
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        if name not in self._datasets:
            raise StructureNotFoundError(name)

        del self._datasets[name]
        self._save_registry()

    @property
    def datasets(self):
        """Get dict of all datasets.

        Returns:
            Dict mapping name -> Dataset
        """
        return self._datasets.copy()

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self):
        status = "open" if self._is_open else "closed"
        dataset_count = len(self._datasets)
        return f"DB('{self.filename}', {status}, {dataset_count} datasets)"

    # Data Structure Factory Methods

    def create_bloomfilter(self, name, expected_items=10000, false_positive_rate=0.01):
        """Create a Bloom filter.

        Args:
            name: Unique name for this filter
            expected_items: Expected number of items
            false_positive_rate: Desired false positive rate

        Returns:
            BloomFilter instance

        Example:
            bf = db.create_bloomfilter('seen_users', expected_items=100000)
            bf.add("user123")
            if "user123" in bf:
                print("Seen before")
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        # Return existing if already loaded
        if name in self._datastructures:
            return self._datastructures[name]

        bf = BloomFilter(name, self, expected_items, false_positive_rate)
        self._datastructures[name] = bf  # Register for auto-save on close
        self._save_datastructures_registry()  # Persist registry
        return bf

    def create_counting_bloomfilter(
        self, name, expected_items=10000, false_positive_rate=0.01, max_count=255
    ):
        """Create a Counting Bloom filter (supports removal).

        Args:
            name: Unique name for this filter
            expected_items: Expected number of items
            false_positive_rate: Desired false positive rate
            max_count: Maximum count per bucket (default 255)

        Returns:
            CountingBloomFilter instance

        Example:
            cbf = db.create_counting_bloomfilter('cache', expected_items=10000)
            cbf.add("item")
            cbf.remove("item")  # Can remove!
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        # Return existing if already loaded
        if name in self._datastructures:
            return self._datastructures[name]

        cbf = CountingBloomFilter(
            name, self, expected_items, false_positive_rate, max_count
        )
        self._datastructures[name] = cbf  # Register for auto-save on close
        self._save_datastructures_registry()  # Persist registry
        return cbf

    def create_list(self, name, dataset_or_template, cache_size=10):
        """Create a persistent List.

        Args:
            name: Unique name for this list
            dataset_or_template: Dataset (for regular list) or DataStructureTemplate (for nested list)
            cache_size: Number of blocks to cache (0 to disable)

        Returns:
            List instance

        Example:
            # Regular list
            user_ds = db.create_dataset('users', id='uint64', name='U50')
            users = db.create_list('users_list', user_ds)
            users.append({'id': 1, 'name': 'Alice'})

            # Nested list (list of lists)
            UserList = List.template(user_ds)
            teams = db.create_list('teams', UserList)
            eng = teams.append()  # Creates nested list
            eng.append({'id': 1, 'name': 'Alice'})
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        # Return existing if already loaded
        if name in self._datastructures:
            return self._datastructures[name]

        lst = List(name, self, dataset_or_template, cache_size)
        self._datastructures[name] = lst  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return lst

    def create_dict(
        self,
        name,
        dataset_or_template,
        cache_size=1000,
        use_bloom=True,
        key_size=None,
        initial_capacity=None,
        hash_keys=False,
        hash_bits=128,
    ):
        """Create a persistent Dict.

        Args:
            name: Unique name for this dict
            dataset_or_template: Dataset (for regular dict) or DataStructureTemplate (for nested dict)
            cache_size: Number of keys to cache (0 to disable)
            use_bloom: Whether to use bloom filters for acceleration

        Returns:
            Dict instance

        Example:
            # Regular dict
            user_ds = db.create_dataset('users', id='uint64', name='U50')
            users = db.create_dict('users_dict', user_ds)
            users['alice'] = {'id': 1, 'name': 'Alice'}

            # Nested dict (dict of dicts)
            UserDict = Dict.template(user_ds)
            teams = db.create_dict('teams', UserDict)
            eng = teams['engineering']  # Creates nested dict
            eng['alice'] = {'id': 1, 'name': 'Alice'}
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        from loom.datastructures.dict import Dict

        # Return existing if already loaded
        if name in self._datastructures:
            return self._datastructures[name]

        dct = Dict(
            name,
            self,
            dataset_or_template,
            cache_size=cache_size,
            use_bloom=use_bloom,
            key_size=key_size,
            initial_capacity=initial_capacity,
            hash_keys=hash_keys,
            hash_bits=hash_bits,
        )
        self._datastructures[name] = dct  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return dct

    def create_set(self, name, key_size=50, use_bloom=True):
        """Create a persistent Set.

        Args:
            name: Unique name for this set
            key_size: Maximum length of string keys (default: 50)
            use_bloom: Whether to use bloom filters for acceleration

        Returns:
            Set instance

        Example:
            active_users = db.create_set('active_users')
            active_users.add('alice')
            active_users.add('bob')

            if 'alice' in active_users:
                print("Alice is active")
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        # Return existing if already loaded
        if name in self._datastructures:
            return self._datastructures[name]

        s = Set(name, self, key_size=key_size, use_bloom=use_bloom)
        self._datastructures[name] = s  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return s

    def create_btree(self, name, dataset, key_size=50, cache_size=100):
        """Create a persistent BTree with ordered keys.

        BTree provides O(log n) operations with ordered iteration
        and efficient range queries.

        Args:
            name: Unique name for this BTree
            dataset: Dataset for storing values
            key_size: Maximum length of string keys (default: 50)
            cache_size: Number of nodes to cache (default: 100)

        Returns:
            BTree instance

        Example:
            user_ds = db.create_dataset('users', id='uint32', name='U50')
            users = db.create_btree('users_btree', user_ds)

            users['alice'] = {'id': 1, 'name': 'Alice'}
            users['bob'] = {'id': 2, 'name': 'Bob'}

            # Ordered iteration
            for key in users.keys():
                print(key)  # alice, bob (sorted)

            # Range queries
            for key, value in users.range('a', 'm'):
                print(key, value)
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        from loom.datastructures.btree import BTree

        # Return existing if already loaded
        if name in self._datastructures:
            return self._datastructures[name]

        btree = BTree(name, self, dataset, key_size=key_size, cache_size=cache_size)
        self._datastructures[name] = btree  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return btree

    def create_graph(self, name, node_schema, edge_schema, directed=True, node_id_max_len=50):
        """Create a persistent Graph.

        Args:
            name: Unique name for this graph
            node_schema: Pydantic BaseModel or dict for node attributes
            edge_schema: Pydantic BaseModel or dict for edge attributes
            directed: If True (default), edges are directed

        Returns:
            Graph instance

        Example:
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            class Knows(BaseModel):
                weight: float

            g = db.create_graph("social", Person, Knows)
            g.add_node("alice", name="Alice", age=30)
            g.add_edge("alice", "bob", weight=0.9)
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        from loom.datastructures.graph import Graph

        if name in self._datastructures:
            return self._datastructures[name]

        g = Graph(name, self, node_schema, edge_schema, directed=directed, node_id_max_len=node_id_max_len)
        self._datastructures[name] = g
        self._save_datastructures_registry()
        return g

    def create_queue(self, name, schema, block_size=64):
        """Create a persistent FIFO Queue.

        Args:
            name:       Unique name
            schema:     Pydantic BaseModel class, or dict of dtype strings
            block_size: Records per block (default 64).  Larger = fewer
                        allocations; smaller = less wasted space when small.

        Returns:
            Queue instance

        Example:
            from pydantic import BaseModel, Field

            class Task(BaseModel):
                id:      int
                payload: str = Field(max_length=100)

            q = db.create_queue("tasks", Task, block_size=128)
            q.push({"id": 1, "payload": "hello"})
            item = q.pop()
        """
        if not self._is_open:
            raise DatabaseNotOpenError()

        from loom.datastructures.queue import Queue

        if name in self._datastructures:
            return self._datastructures[name]

        q = Queue(name, self, schema, block_size=block_size)
        self._datastructures[name] = q
        self._save_datastructures_registry()
        return q
