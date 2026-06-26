"""
Database orchestrator for managing multiple datasets and data structures.

Provides high-level API with:
- Automatic schema persistence
- Dataset registry management
- Data structure factory methods
- Context manager support
- Dict-like access
"""

import atexit
import threading
import weakref
from contextlib import contextmanager

from loom.fileio import ByteFileDB
from loom.dataset import Dataset
from loom.datastructures import BloomFilter, CountingBloomFilter, List, Set
from loom.blob import BlobStore
from loom.errors import (
    DatabaseNotOpenError,
    DuplicateNameError,
    ReadOnlyError,
    StructureNotFoundError,
)


# Best-effort safety net: close any still-open writable DBs at interpreter exit,
# so a script that forgets close()/`with` still persists structure metadata
# (lengths, counters).  Data pages are flushed by the OS regardless, but the
# in-memory metadata is only written to the header by close()/flush()/save().
# atexit is far more reliable than __del__ at shutdown; everything is wrapped so
# a teardown error never escapes.
_OPEN_DBS = weakref.WeakSet()


def _close_open_dbs_at_exit():
    for db in list(_OPEN_DBS):
        try:
            if getattr(db, "_is_open", False) and not getattr(db, "read_only", False):
                db.close()
        except Exception:
            pass


atexit.register(_close_open_dbs_at_exit)


class DB:
    """High-level database orchestrator.

    Manages multiple datasets with automatic schema persistence.

    Usage::

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
        blob_compression=None,
        auto_save_interval=100,
        cache_size=200_000,
        sync_writes=False,
        multiprocess_safe=False,
        flag="r+",
    ):
        """Initialize database.

        Args:
            filename: Path to database file
            initial_size: Initial file size in bytes
            header_size: Header region size in bytes
            auto_open: Automatically open database (default: True)
            blob_compression: Compression for blobs ("brotli", "zlib", or None).
                Default is None — compression typically costs ~20× on insert
                throughput (CPU-bound).  Pass "brotli" to trade speed for
                ~3–5× space savings on natural language.
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
            multiprocess_safe: If True, acquire an exclusive fcntl.flock on a
                companion <filename>.lock file for the duration of every write
                operation.  Prevents concurrent writes from separate processes
                and makes SWMR (single-writer / multiple-reader) safe on Linux.
                Readers never block each other — Linux shared mmap pages give
                immediate inter-process visibility without msync.
                Cost: ~1-2 µs per write (flock syscall, uncontested).
        """
        self.filename = filename
        self.flag = flag
        self.read_only = flag == "r"
        self.auto_save_interval = auto_save_interval

        # Thread-safety: RLock allows re-entrant locking from the same thread
        # (e.g. save() called inside a write operation).  Uncontested cost ~50 ns.
        self._lock = threading.RLock()

        # Process-safety (opt-in): exclusive flock on a companion lock file.
        self._multiprocess_safe = multiprocess_safe
        self._lockfile = None
        if multiprocess_safe:
            import fcntl as _fcntl  # noqa: F401 — validate availability early

            lockpath = filename + ".lock"
            self._lockfile = open(lockpath, "w")
            self._fcntl = _fcntl

        # Shared cache: one LRU for the entire DB, namespaced per-structure
        if cache_size > 0:
            from loom.cache import LRUCache

            self._shared_cache = LRUCache(cache_size)
        else:
            self._shared_cache = None
        self._db = ByteFileDB(
            filename,
            initial_size,
            header_size,
            sync_writes=sync_writes,
            flag=flag,
        )
        self._datasets = {}  # name -> Dataset instance
        self._datastructures = {}  # name -> DataStructure instance
        self._is_open = False
        self._loading_registry = False  # Flag to prevent saving during load
        self._blob_compression = blob_compression
        self._blob_store = None  # Lazy initialized

        # Auto-open by default for convenience
        if auto_open:
            self.open()

    @contextmanager
    def write_lock(self):
        """Acquire the write lock for a compound operation.

        Use this to make multi-step writes atomic:

            with db.write_lock():
                ds.insert(...)
                bt[key] = val

        Thread-safe (threading.RLock) in all modes.
        Process-safe (fcntl.flock LOCK_EX) when multiprocess_safe=True.
        Readers are never blocked — safe to read without the lock.
        """
        if self.read_only:
            raise ReadOnlyError()
        if self._multiprocess_safe:
            self._fcntl.flock(self._lockfile, self._fcntl.LOCK_EX)
        try:
            with self._lock:
                yield
        finally:
            if self._multiprocess_safe:
                self._fcntl.flock(self._lockfile, self._fcntl.LOCK_UN)

    def open(self):
        """Open database and load dataset registry.

        Note: Usually not needed - database opens automatically on __init__.
        Only use if you created with auto_open=False.
        """
        if self._is_open:
            return self  # Already open

        self._db.open()
        self._is_open = True
        _OPEN_DBS.add(self)   # auto-close at interpreter exit if not closed
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
            if not self.read_only:
                self._db.set_header_field(
                    self.BLOB_COMPRESSION_KEY, self._blob_compression
                )
        return self._blob_store

    def close(self):
        """Close database and save all data structures."""
        if self._is_open:
            if not self.read_only:
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
        self._ensure_writable()
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
        self._ensure_writable()
        self.blob_store.delete(offset, n_slots)

    def _load_registry(self):
        """Load dataset registry from header and recreate Dataset instances."""
        registry = self._db.get_header_field(self.REGISTRY_KEY, {})

        for name, info in registry.items():
            identifier = info["identifier"]
            schema = info["schema"]

            # Pass blob_store if the schema uses variable-length fields
            needs_blobs = any(v in ("blob", "text", "json") for v in schema.values())
            blob_store = self.blob_store if needs_blobs else None

            # Recreate Dataset instance
            dataset = Dataset(
                name, self._db, identifier, blob_store=blob_store, **schema
            )
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
        self._ensure_writable()
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
        if field_name in getattr(dataset, "_json_fields", set()):
            return "json"
        if field_name in dataset._blob_fields:
            return "blob"
        if field_name in getattr(dataset, "_utf8_fields", {}):
            strict = field_name in getattr(dataset, "_utf8_strict", set())
            return f"utf8[{dataset._utf8_fields[field_name]}{'!' if strict else ''}]"
        if field_name in getattr(dataset, "_datetime_fields", set()):
            return "datetime"
        dtype = dataset.user_schema.fields[field_name][0]
        return dtype_to_str(dtype)

    def _save_datastructures_registry(self):
        """Save data structures registry to header (generic — no per-type switch)."""
        if self._loading_registry:
            return
        self._ensure_writable()

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
        self._ensure_writable()
        next_id = self._db.get_header_field(self.NEXT_ID_KEY, 1)
        if next_id > 127:
            raise ValueError("Maximum number of datasets (127) reached")

        self._db.set_header_field(self.NEXT_ID_KEY, next_id + 1)
        return next_id

    def flush(self):
        """Persist everything to disk immediately: in-memory structure metadata
        (lengths, counters …) AND the mmap data pages.

        In the default sync_writes=False mode, writes accumulate in the
        OS page cache and are only guaranteed on disk after close() or
        this method.  Structure metadata is otherwise only checkpointed every
        ``auto_save_interval`` ops and at the end of bulk ops (append_many …),
        so call flush()/close() (or use ``with DB(...)``) before relying on a
        reopened len()/count.

        For long-running servers, call this periodically:
            import schedule
            schedule.every(30).seconds.do(db.flush)

        Or after critical transactions:
            db["key"] = value
            db.flush()   # now on disk regardless of OS writeback schedule
        """
        if not self._is_open:
            return
        if not self.read_only:
            # Persist in-memory structure metadata (lengths, counters …) too,
            # not just the mmap data pages — otherwise len()/counts can read
            # stale values after a flush without close.
            for ds in self._datastructures.values():
                if hasattr(ds, "save"):
                    try:
                        ds.save(force=True)
                    except TypeError:
                        ds.save()
            if self._blob_store is not None:
                self._blob_store._save_freelist()
        self._db.flush()

    def _ensure_writable(self):
        if self.read_only:
            raise ReadOnlyError()

    @contextmanager
    def batch(self):
        """Batch mode: defer all header flushes until the block exits.

        Use this for bulk inserts to avoid per-operation header writes.
        Data is still written to mmap immediately, only the header
        (allocation index, metadata) flush is deferred.

        Example::

            with db.batch():
                for i in range(10000):
                    dataset.insert({'id': i, 'content': 'hello'})
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()
        self._db.begin_batch()
        try:
            yield
        finally:
            self._db.end_batch()

    def apply_writes(self, writes):
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        if not writes:
            return

        self._db.transaction(writes)

    @contextmanager
    def write_batch(self):
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        writes = []
        try:
            yield writes
            if writes:
                self._db.transaction(writes)
        finally:
            pass

    def _reserve_name(self, name, in_dataset, exist_ok):
        """Validate a new name across BOTH the dataset and datastructure
        namespaces (so ``db[name]`` is never ambiguous).

        Returns the existing object when it already exists in its OWN namespace
        and ``exist_ok`` is True; raises DuplicateNameError on any other clash.
        """
        own = self._datasets if in_dataset else self._datastructures
        other = self._datastructures if in_dataset else self._datasets
        if name in own:
            if exist_ok:
                return own[name]
            raise DuplicateNameError(name, "Dataset" if in_dataset else "Data structure")
        if name in other:
            # name taken by the other kind — db[name] would be ambiguous
            raise DuplicateNameError(name, "Data structure" if in_dataset else "Dataset")
        if self._is_collection(name):
            raise DuplicateNameError(name, "Collection")
        return None

    def create_dataset(self, dataset_name, model=None, exist_ok=False, **schema):
        """Create a new dataset with automatic identifier assignment.

        Args:
            dataset_name: Dataset name (must be unique)
            model: Optional Pydantic BaseModel class — schema is derived
                   from its fields (int→int64, float→float64, str→text, etc.)
            **schema: Field definitions as numpy dtypes (ignored if model given)

        Returns:
            Dataset instance

        Examples::

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
        self._ensure_writable()

        # Convert Pydantic model to schema if provided
        if model is not None and not schema:
            from loom.schema import schema_from_model

            schema = schema_from_model(model)

        existing = self._reserve_name(dataset_name, in_dataset=True, exist_ok=exist_ok)
        if existing is not None:
            return existing

        # Get next identifier
        identifier = self._get_next_identifier()

        # Pass blob_store if schema uses variable-length fields
        needs_blobs = any(v in ("blob", "text", "json") for v in schema.values())
        blob_store = self.blob_store if needs_blobs else None

        # Create dataset
        dataset = Dataset(
            dataset_name, self._db, identifier, blob_store=blob_store, **schema
        )
        self._datasets[dataset_name] = dataset

        # Save registry
        self._save_registry()

        return dataset

    def _dataset_for(self, struct_name, spec):
        """Coerce a dict's/list's/btree's value-store spec into a Dataset.

        Accepts what these structures have always accepted — a Dataset (passed
        through) or a DataStructureTemplate (nested, passed through) — and, as a
        convenience, a Pydantic model class or a plain schema dict, in which
        case a backing dataset named ``_{struct_name}_ds`` is auto-created.
        This lets ``db.create_dict("metadata", Metadata)`` replace the explicit
        ``ds = db.create_dataset("metadata_ds", Metadata)`` + ``create_dict``.
        """
        from loom.datastructures.template import DataStructureTemplate

        if spec is None or isinstance(spec, (Dataset, DataStructureTemplate)):
            return spec
        auto_name = f"_{struct_name}_ds"
        if isinstance(spec, dict):
            return self.create_dataset(auto_name, exist_ok=True, **spec)
        if hasattr(spec, "model_fields"):  # Pydantic BaseModel subclass
            return self.create_dataset(auto_name, spec, exist_ok=True)
        return spec

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

        Example::

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

        # Then collections (stored as header config, reopened on demand)
        if self._is_collection(name):
            return self.collection(name)

        raise StructureNotFoundError(name)

    def _is_collection(self, name):
        """True if `name` is a Collection (stored as a header config entry)."""
        return self._db.has_header_field(f"__collection__::{name}")

    def __contains__(self, name):
        """Check if a name exists as a dataset, data structure, or collection."""
        return (name in self._datasets or name in self._datastructures
                or self._is_collection(name))

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

    # -------------------------------------------------------------------------
    # HTTP server
    # -------------------------------------------------------------------------

    def fastapi_app(self, *, title=None, dashboard=False, auth_token=None):
        """Return a FastAPI app exposing this DB over HTTP.

        Datastructures are auto-mounted with proper OpenAPI docs and
        Pydantic validation.  Mount this app in your own ASGI stack, or
        use `db.serve()` to run it directly.

        Requires `fastapi` (`pip install 'fastapi[standard]'`).
        Pass `auth_token="..."` to protect the API, docs, and dashboard.
        """
        from loom.server import build_app

        return build_app(
            self,
            title=title,
            dashboard=dashboard,
            auth_token=auth_token,
        )

    def serve(
        self,
        host="127.0.0.1",
        port=8000,
        dashboard=False,
        auth_token=None,
        **uvicorn_kwargs,
    ):
        """Run an HTTP server exposing this DB.

        Single-writer / single-reader: requests are serialized through
        a process-wide lock so the underlying mmap stays consistent.

        Blocking call.  Requires `fastapi` and `uvicorn`.
        Pass `dashboard=True` to also mount an optional integrated dashboard
        at `/dashboard` on the same server.
        Pass `auth_token="..."` to protect the API, docs, and dashboard.

        Example::

            with DB("app.db") as db:
                db.create_dataset("users", User)
                db.serve(port=8000)

        Visit http://localhost:8000/docs for the auto-generated API.
        """
        from loom.server import serve

        serve(
            self,
            host=host,
            port=port,
            dashboard=dashboard,
            auth_token=auth_token,
            **uvicorn_kwargs,
        )

    # Data Structure Factory Methods

    def create_bloomfilter(self, name, expected_items=10000, false_positive_rate=0.01, exist_ok=False):
        """Create a Bloom filter.

        Args:
            name: Unique name for this filter
            expected_items: Expected number of items
            false_positive_rate: Desired false positive rate

        Returns:
            BloomFilter instance

        Example::

            bf = db.create_bloomfilter('seen_users', expected_items=100000)
            bf.add("user123")
            if "user123" in bf:
                print("Seen before")
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        # Return existing if already loaded
        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        bf = BloomFilter(name, self, expected_items, false_positive_rate)
        self._datastructures[name] = bf  # Register for auto-save on close
        self._save_datastructures_registry()  # Persist registry
        return bf

    def create_counting_bloomfilter(
        self, name, expected_items=10000, false_positive_rate=0.01, max_count=255,
        exist_ok=False,
    ):
        """Create a Counting Bloom filter (supports removal).

        Args:
            name: Unique name for this filter
            expected_items: Expected number of items
            false_positive_rate: Desired false positive rate
            max_count: Maximum count per bucket (default 255)

        Returns:
            CountingBloomFilter instance

        Example::

            cbf = db.create_counting_bloomfilter('cache', expected_items=10000)
            cbf.add("item")
            cbf.remove("item")  # Can remove!
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        # Return existing if already loaded
        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        cbf = CountingBloomFilter(
            name, self, expected_items, false_positive_rate, max_count
        )
        self._datastructures[name] = cbf  # Register for auto-save on close
        self._save_datastructures_registry()  # Persist registry
        return cbf

    def create_list(self, name, dataset_or_template, exist_ok=False):
        """Create a persistent List.

        Args:
            name: Unique name for this list
            dataset_or_template: A Dataset, a Pydantic model class or a schema
                dict (a backing dataset ``_{name}_ds`` is then auto-created), or
                a DataStructureTemplate (for a nested list).

        Returns:
            List instance

        Example::

            # Pydantic model — backing dataset created automatically
            users = db.create_list('users_list', User)
            users.append({'id': 1, 'name': 'Alice'})

            # Or pass an explicit Dataset
            user_ds = db.create_dataset('users', id='uint64', name='U50')
            users = db.create_list('users_list', user_ds)

            # Nested list (list of lists)
            UserList = List.template(user_ds)
            teams = db.create_list('teams', UserList)
            eng = teams.append()  # Creates nested list
            eng.append({'id': 1, 'name': 'Alice'})
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        # Return existing if already loaded
        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        dataset_or_template = self._dataset_for(name, dataset_or_template)
        lst = List(name, self, dataset_or_template)
        self._datastructures[name] = lst  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return lst

    def create_dict(
        self,
        name,
        dataset_or_template,
        use_bloom=True,
        key_size=None,
        initial_capacity=None,
        hash_keys=False,
        hash_bits=128,
        store_key=True,
        max_key_len=100,
        key_dtype=None,
        exist_ok=False,
    ):
        """Create a persistent Dict.

        Args:
            name: Unique name for this dict
            dataset_or_template: A Dataset, a Pydantic model class or a schema
                dict (a backing dataset ``_{name}_ds`` is then auto-created), or
                a DataStructureTemplate (for a nested dict).
            use_bloom: Whether to use bloom filters for acceleration
            max_key_len: Byte budget for the stored key (used for iteration
                recovery via keys()/items()).  Default 100.  Smaller = smaller
                value records when keys are short.
            key_dtype: How the recovered key is stored. Default None →
                "utf8[max_key_len]" (inline UTF-8, ~4× smaller than UCS-4 for
                ASCII). Pass "U{N}" for fixed UCS-4 or "text" for unbounded.

        Returns:
            Dict instance

        Example::

            # Pydantic model — backing dataset created automatically
            users = db.create_dict('users_dict', User)
            users['alice'] = {'id': 1, 'name': 'Alice'}

            # Or pass an explicit Dataset
            user_ds = db.create_dataset('users', id='uint64', name='U50')
            users = db.create_dict('users_dict', user_ds)

            # Nested dict (dict of dicts)
            UserDict = Dict.template(user_ds)
            teams = db.create_dict('teams', UserDict)
            eng = teams['engineering']  # Creates nested dict
            eng['alice'] = {'id': 1, 'name': 'Alice'}
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        from loom.datastructures.dict import Dict

        # Return existing if already loaded
        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        dataset_or_template = self._dataset_for(name, dataset_or_template)
        dct = Dict(
            name,
            self,
            dataset_or_template,
            use_bloom=use_bloom,
            key_size=key_size,
            initial_capacity=initial_capacity,
            hash_keys=hash_keys,
            hash_bits=hash_bits,
            store_key=store_key,
            max_key_len=max_key_len,
            key_dtype=key_dtype,
        )
        self._datastructures[name] = dct  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return dct

    def create_set(self, name, key_size=50, use_bloom=True, exist_ok=False):
        """Create a persistent Set.

        Args:
            name: Unique name for this set
            key_size: Maximum length of string keys (default: 50)
            use_bloom: Whether to use bloom filters for acceleration

        Returns:
            Set instance

        Example::

            active_users = db.create_set('active_users')
            active_users.add('alice')
            active_users.add('bob')

            if 'alice' in active_users:
                print("Alice is active")
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        # Return existing if already loaded
        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        s = Set(name, self, key_size=key_size, use_bloom=use_bloom)
        self._datastructures[name] = s  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return s

    def create_btree(self, name, dataset, key_size=50, int_keys=False, exist_ok=False):
        """Create a persistent BTree with ordered keys.

        BTree provides O(log n) operations with ordered iteration
        and efficient range queries.

        Args:
            name: Unique name for this BTree
            dataset: A Dataset, a Pydantic model class or a schema dict for the
                stored values (a backing dataset ``_{name}_ds`` is auto-created
                for the latter two).
            key_size: Maximum length of string keys (default: 50)
            int_keys: If True, keys are integers ordered numerically (stored
                order-preserving); keys()/items()/range() take and return ints.

        Returns:
            BTree instance

        Example::

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
        self._ensure_writable()

        from loom.datastructures.btree import BTree

        # Return existing if already loaded
        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        dataset = self._dataset_for(name, dataset)
        btree = BTree(name, self, dataset, key_size=key_size, int_keys=int_keys)
        self._datastructures[name] = btree  # Register for caching and auto-save
        self._save_datastructures_registry()  # Persist registry
        return btree

    def create_graph(
        self, name, node_schema, edge_schema, directed=True, node_id_max_len=50,
        label_field=None, exist_ok=False,
    ):
        """Create a persistent Graph.

        Args:
            name: Unique name for this graph
            node_schema: Pydantic BaseModel or dict for node attributes
            edge_schema: Pydantic BaseModel or dict for edge attributes
            directed: If True (default), edges are directed

        Returns:
            Graph instance

        Example::

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
        self._ensure_writable()

        from loom.datastructures.graph import Graph

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        g = Graph(
            name,
            self,
            node_schema,
            edge_schema,
            directed=directed,
            node_id_max_len=node_id_max_len,
            label_field=label_field,
        )
        self._datastructures[name] = g
        self._save_datastructures_registry()
        return g

    def create_search_index(
        self,
        name,
        dataset=None,
        text_fields=None,
        ignore_case=True,
        ignore_accent=True,
        ignore_punctuation=True,
        doc_id_dtype="uint32",
        scoring="boolean",
        bm25_k1=1.5,
        bm25_b=0.75,
        store_documents=True,
        exist_ok=False,
    ):
        """Create or reopen a full-text SearchIndex over a documents dataset.

        The index is built on a user Dataset (the document store), like the
        other structures.  add(record) inserts the record into that dataset and
        indexes its text fields, keeping only the record's address — documents
        are never duplicated.  Boolean queries (AND / OR / AND NOT, parens, `*`)
        are parsed by eldar (`pip install eldar`); loom stores the postings.

        Args:
            name: Unique name for this index.
            dataset: Dataset holding the documents (its fields are the schema
                of the records you add()).
            text_fields: Dataset fields to index (None → all string fields).
            ignore_case / ignore_accent / ignore_punctuation: normalisation
                applied identically at index and query time.
            doc_id_dtype: stored doc-id width ("uint32" = up to 4 G documents).
            scoring: "boolean" (default, postings store only doc-ids) or "bm25"
                (also store term frequencies + doc lengths to enable ranked
                search). A "bm25" index still answers mode="boolean" queries.
            bm25_k1, bm25_b: BM25 tuning parameters.

        Returns:
            SearchIndex instance.

        Example::

            docs = db.create_dataset("docs", title="utf8[120]", body="text")
            idx = db.create_search_index("idx", docs, text_fields=["title", "body"],
                                         scoring="bm25")
            i = idx.add({"title": "Fast search", "body": "inverted index"})
            idx.search("search AND NOT slow")              # ranked (bm25)
            idx.search("search", mode="boolean")           # unranked
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        from loom.datastructures.search import SearchIndex

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        si = SearchIndex(
            name,
            self,
            dataset,
            text_fields=text_fields,
            ignore_case=ignore_case,
            ignore_accent=ignore_accent,
            ignore_punctuation=ignore_punctuation,
            doc_id_dtype=doc_id_dtype,
            scoring=scoring,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            store_documents=store_documents,
        )
        self._datastructures[name] = si
        self._save_datastructures_registry()
        return si

    def create_queue(self, name, schema, block_size=64, exist_ok=False):
        """Create a persistent FIFO Queue.

        Args:
            name:       Unique name
            schema:     Pydantic BaseModel class, or dict of dtype strings
            block_size: Records per block (default 64).  Larger = fewer
                        allocations; smaller = less wasted space when small.

        Returns:
            Queue instance

        Example::

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
        self._ensure_writable()

        from loom.datastructures.queue import Queue

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        q = Queue(name, self, schema, block_size=block_size)
        self._datastructures[name] = q
        self._save_datastructures_registry()
        return q

    def create_priority_queue(self, name, schema=None, max_first=True, exist_ok=False):
        """Create or reopen a persistent PriorityQueue (BTree-backed).

        push(item, priority) / pop() / peek() in O(log n).  The highest
        priority pops first by default (max_first=True); set max_first=False
        for a min-priority queue.  Equal priorities pop in FIFO order.
        Priorities may be int, float or datetime (order-preserving encoded).

        Args:
            name:      Unique name.
            schema:    Pydantic model / dict of dtypes / Dataset for the items.
                       Omit to reopen an existing queue.
            max_first: True (default) → highest priority first; False → lowest.

        Returns:
            PriorityQueue instance.

        Example::

            pq = db.create_priority_queue("jobs", {"task": "utf8[40]"})
            pq.push({"task": "send email"}, priority=5)
            pq.push({"task": "reindex"},    priority=9)
            pq.pop()        # -> {"task": "reindex"}   (priority 9 first)
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        from loom.datastructures.priority_queue import PriorityQueue

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        pq = PriorityQueue(name, self, schema, max_first=max_first)
        self._datastructures[name] = pq
        self._save_datastructures_registry()
        return pq

    def create_lru_dict(
        self,
        name,
        schema,
        capacity=1000,
        key_size=50,
        hash_keys=False,
        hash_bits=128,
        exist_ok=False,
    ):
        """Create a persistent LRU Dict (fixed-capacity, evicts LRU on insert).

        Args:
            name:      Unique name
            schema:    Pydantic BaseModel, dict of dtypes, or existing Dataset
            capacity:  Maximum number of entries (evicts oldest on overflow)
            key_size:  Max key length (ignored when hash_keys=True)
            hash_keys: Hash keys before storage (for long/arbitrary keys)
            hash_bits: Hash length in bits (64 or 128 recommended)

        Example::

            from pydantic import BaseModel

            class Profile(BaseModel):
                name:  str
                score: float

            cache = db.create_lru_dict("profiles", Profile, capacity=10_000)
            cache["alice"] = {"name": "Alice", "score": 9.5}
            val = cache["alice"]    # O(1), moves alice to most-recent
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()

        from loom.datastructures.lru_dict import LRUDict

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing

        lru = LRUDict(
            name,
            self,
            schema,
            capacity=capacity,
            key_size=key_size,
            hash_keys=hash_keys,
            hash_bits=hash_bits,
        )
        self._datastructures[name] = lru
        self._save_datastructures_registry()
        return lru

    def create_flat_index(self, name, dim, metric="cosine", exist_ok=False):
        """Create a FlatIndex for exact vector similarity search.

        Args:
            name:   Unique name
            dim:    Vector dimensionality (e.g. 1536 for text-embedding-3-small)
            metric: "cosine" (default), "l2", or "dot"

        Example::

            idx = db.create_flat_index("passages", dim=1536)
            idx.add("doc_1", embedding)
            results = idx.search(query, k=10)
            # [("doc_1", 0.95), ("doc_2", 0.87), ...]
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()
        from loom.datastructures.vector_index import FlatIndex

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing
        idx = FlatIndex(name, self, dim=dim, metric=metric)
        self._datastructures[name] = idx
        self._save_datastructures_registry()
        return idx

    def create_ivf_index(
        self,
        name,
        dim,
        metric="cosine",
        n_clusters=256,
        pq=False,
        n_sub=16,
        n_bits=8,
        exist_ok=False,
    ):
        """Create an IVFIndex for approximate vector similarity search.

        Requires calling .train(sample_vectors) before inserting.

        Args:
            name:       Unique name
            dim:        Vector dimensionality
            metric:     "cosine" (default), "l2", or "dot"
            n_clusters: Number of IVF cells (default 256; rule of thumb: sqrt(n))
            pq:         Enable Product Quantization compression (default False)
            n_sub:      Number of PQ sub-vectors (default 16; higher → less compression)
            n_bits:     Bits per sub-quantizer (default 8 → 256 centroids each)

        Example::

            ivf = db.create_ivf_index("passages", dim=1536,
                                       n_clusters=256, pq=True, n_sub=16)
            ivf.train(sample_matrix)           # train on a representative sample
            ivf.add_batch([("id", vec), ...])  # bulk insert
            results = ivf.search(query, k=10, nprobe=32)
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()
        from loom.datastructures.vector_index import IVFIndex

        existing = self._reserve_name(name, in_dataset=False, exist_ok=exist_ok)
        if existing is not None:
            return existing
        idx = IVFIndex(
            name,
            self,
            dim=dim,
            metric=metric,
            n_clusters=n_clusters,
            pq=pq,
            n_sub=n_sub,
            n_bits=n_bits,
        )
        self._datastructures[name] = idx
        self._save_datastructures_registry()
        return idx

    def collection(self, name, model=None, indexes=None, key_size=64,
                   index_key_size=192, exist_ok=False):
        """Create or reopen a Collection: a record store with typed indexes.

        Each field's index *kind* is declared and mapped to the right loom
        structure, kept in sync automatically.  Exactly one index must be the
        ``primary`` (it stores the records and is the unique key).

        Args:
            name:    Unique collection name.
            model:   Pydantic model / dict schema / Dataset for the records.
                     Omit to reopen an existing collection.
            indexes: ``{field: kind}`` where kind is a string ("primary",
                     "unique", "range", "many") or a spec object
                     (``Unique()``, ``Range()``, ``Many(sort=..., desc=...)``).
            key_size:       max primary-key length (default 64).
            index_key_size: max composite-key length for range/many BTrees.

        Kinds → structures: primary/unique → Dict; range/many → BTree.

        Example::

            from loom import Many
            posts = db.collection("posts", Post, indexes={
                "id":         "primary",
                "username":   Many(sort="created_at", desc=True),
                "engagement": "range",
            })
            posts.insert({"id": "p1", "username": "alice",
                          "created_at": 170, "engagement": 9})
            posts["p1"]                              # by id
            posts.find("username", "alice", limit=20)   # user's recent posts
            posts.range("engagement", 1000, None)       # engagement >= 1000
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        from loom.collection import Collection, Many, Range, Unique, Primary, _as_spec

        cfg_key = f"__collection__::{name}"
        cfg = self._db.get_header_field(cfg_key)

        def _spec_from_cfg(ic):
            k = ic["kind"]
            if k == "unique":
                return Unique()
            if k == "range":
                return Range()
            if k == "many":
                return Many(sort=ic.get("sort"), desc=ic.get("desc", False))
            return Primary()

        def _build_index_objs(cfg):
            objs = {}
            for idx_name, ic in cfg["indexes"].items():
                spec = _spec_from_cfg(ic)
                field = ic.get("field", idx_name)   # field may differ from name
                sources = [field]
                if spec.kind == "many" and spec.sort is not None:
                    sources.append(spec.sort)
                objs[idx_name] = {
                    "name": idx_name, "spec": spec, "field": field,
                    "struct": self._datastructures[ic["struct"]],
                    "sources": sources,
                    "hashed": ic.get("hashed", False),
                }
            return objs

        def _build_search_objs(cfg):
            objs = {}
            for field, sc in cfg.get("search", {}).items():
                objs[field] = {
                    "fields": sc["fields"],
                    "index": self._datastructures[sc["search"]],
                    "pk2docid": self._datastructures[sc["pk2docid"]],
                    "docid2pk": self._datastructures[sc["docid2pk"]],
                }
            return objs

        if model is None:
            # ── Reopen ──────────────────────────────────────────────────
            if not cfg:
                raise StructureNotFoundError(name)
            return Collection(
                self, name, self.get_dataset(cfg["dataset"]),
                cfg["primary_field"], self._datastructures[cfg["primary"]],
                _build_index_objs(cfg), cfg["key_size"],
                search=_build_search_objs(cfg),
            )

        # ── Create ──────────────────────────────────────────────────────
        self._ensure_writable()
        if cfg:
            # collection already exists — reopen it (exist_ok) or fail clearly
            # (rather than the confusing internal "Dataset 'name__data' exists").
            if exist_ok:
                return self.collection(name)
            raise DuplicateNameError(name, "Collection")
        # the collection's public name must be free across all namespaces
        if name in self._datasets or name in self._datastructures:
            raise DuplicateNameError(name, "Dataset" if name in self._datasets
                                     else "Data structure")
        specs = {f: _as_spec(s) for f, s in (indexes or {}).items()}
        primaries = [f for f, s in specs.items() if s.kind == "primary"]
        if len(primaries) != 1:
            raise ValueError(
                "exactly one index must be 'primary' "
                f"(got {len(primaries)}: {primaries})"
            )
        primary_field = primaries[0]

        ds_before = set(self._datasets)
        struct_before = set(self._datastructures)
        try:
            return self._build_collection(
                name, model, specs, primary_field, key_size, index_key_size, cfg_key,
            )
        except Exception:
            # Roll back everything this call created, so a failed creation (e.g.
            # header overflow on the final config save) leaves no orphan
            # structures that would block a retry.  Disk is reclaimed by vacuum.
            self._rollback_collection(ds_before, struct_before, cfg_key)
            raise

    def drop_collection(self, name):
        """Delete a collection and unregister all its internal structures.

        Disk space is reclaimed lazily by ``db.vacuum()`` (the arenas are
        unlinked from the registry, not yet freed)."""
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()
        cfg_key = f"__collection__::{name}"
        if not self._db.has_header_field(cfg_key):
            raise StructureNotFoundError(name)
        pat = f"{name}__"   # the collection's reserved namespace (incl. wrapped internals)
        hd = self._db._header_data
        for n in [n for n in list(self._datastructures) if pat in n]:
            self._datastructures.pop(n, None)
            hd.pop(f"_ds_{n}_metadata", None)
        for n in [n for n in list(self._datasets) if pat in n]:
            self._datasets.pop(n, None)
        self._db.delete_header_field(cfg_key)
        # Drop cached addresses: a recreated structure of the same name would
        # otherwise read this collection's stale value addresses (the shared
        # cache namespaces by name, and a fresh structure resets its gen to 0).
        if self._shared_cache is not None:
            self._shared_cache.clear()
        self._save_datastructures_registry()
        self._save_registry()

    def migrate_collection(self, name, new_model, transforms=None, indexes=None,
                           key_size=64, index_key_size=192):
        """Migrate a collection to a new record schema (add / drop / rename
        fields), rebuilding it in place.

        For each existing record the new record is built field-by-field: a
        ``transforms[field](old_record)`` callable if given, else the old value
        if the field still exists, else the schema default.  Indexes are reused
        unless `indexes` is supplied.  All records are read into memory first,
        so a crash mid-migration cannot lose data already on disk.

        Example — split ``name`` into first/last::

            db.migrate_collection("users", UserV2, transforms={
                "first_name": lambda r: r["name"].split()[0],
                "last_name":  lambda r: r["name"].split()[-1],
            })
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()
        from loom.dataset import Dataset

        cfg_key = f"__collection__::{name}"
        cfg = self._db.get_header_field(cfg_key)
        if not cfg:
            raise StructureNotFoundError(name)
        old = self.collection(name)
        transforms = transforms or {}

        # New schema's field names (to know what to keep / default).
        if isinstance(new_model, Dataset):
            new_fields = list(new_model.user_schema.names)
        elif isinstance(new_model, dict):
            new_fields = list(new_model)
        else:
            from loom.schema import schema_from_model
            new_fields = list(schema_from_model(new_model))

        # Reconstruct the index declaration from the stored config unless given.
        if indexes is None:
            indexes = self._collection_index_specs(cfg, old)

        # Read + transform every record FIRST (so the drop below can't lose data).
        migrated = []
        for pk, rec in old._primary.items():
            rec = dict(rec)
            new_rec = {}
            for f in new_fields:
                if f in transforms:
                    new_rec[f] = transforms[f](rec)
                elif f in rec:
                    new_rec[f] = rec[f]
                # else: omitted → dataset writes the field default
            migrated.append(new_rec)

        self.drop_collection(name)
        col = self.collection(name, new_model, indexes=indexes,
                              key_size=key_size, index_key_size=index_key_size)
        if migrated:
            col.insert_many(migrated)
        return col

    def _collection_index_specs(self, cfg, col):
        """Rebuild the `indexes=` declaration from a collection's stored config
        (used by migrate / vacuum to recreate it identically)."""
        from loom.collection import Many, Range, Unique, Search
        indexes = {cfg["primary_field"]: "primary"}
        for idx_name, ic in cfg["indexes"].items():
            field = ic.get("field", idx_name)
            fld = None if field == idx_name else field
            if ic["kind"] == "unique":
                indexes[idx_name] = Unique(field=fld)
            elif ic["kind"] == "range":
                indexes[idx_name] = Range(field=fld)
            else:
                indexes[idx_name] = Many(sort=ic.get("sort"),
                                         desc=ic.get("desc", False), field=fld)
        for field, sc in cfg.get("search", {}).items():
            si = col._search[field]["index"]
            indexes[field] = Search(fields=sc["fields"], scoring=si._scoring,
                                    bm25_k1=si._k1, bm25_b=si._b)
        return indexes

    def vacuum(self):
        """Reclaim dead space by rewriting the database into a fresh file.

        Soft-deleted records, fragmentation, and arenas orphaned by
        drop/migrate/upsert are all dropped: every collection (and any
        standalone dataset) is copied compactly into a new file, which then
        atomically replaces the original.

        Supports databases made of collections (+ their internals) and plain
        datasets.  Raises NotImplementedError if it finds a standalone
        structure type it can't safely copy yet — so it is never lossy.
        """
        if not self._is_open:
            raise DatabaseNotOpenError()
        self._ensure_writable()
        import os as _os

        coll_names = sorted(
            k[len("__collection__::"):]
            for k in self._db._header_data
            if k.startswith("__collection__::")
        )
        owned_pats = [f"{c}__" for c in coll_names]

        def _owned(n):
            return any(p in n for p in owned_pats)

        # Refuse rather than risk dropping anything we can't safely copy.
        orphan = ([n for n in self._datastructures if not _owned(n)]
                  + [n for n in self._datasets if not _owned(n)])
        if orphan:
            raise NotImplementedError(
                "vacuum() currently supports databases made of collections only; "
                f"found standalone structures/datasets: {orphan[:5]}"
            )

        # Snapshot each collection (schema dict, index specs, live records).
        snapshots = []
        for name in coll_names:
            cfg = self._db.get_header_field(f"__collection__::{name}")
            col = self.collection(name)
            ds = col.dataset
            schema = {f: self._dtype_to_registry_str(ds, f) for f in ds.user_schema.names}
            indexes = self._collection_index_specs(cfg, col)
            records = [dict(rec) for _pk, rec in col._primary.items()]
            snapshots.append((name, schema, indexes, cfg["key_size"],
                              cfg["index_key_size"], records))

        orig_header = self._db.header_size
        orig_sync = self._db.sync_writes
        tmp_path = self.filename + ".vacuum.tmp"
        for p in (tmp_path, tmp_path + ".log", tmp_path + ".lock"):
            if _os.path.exists(p):
                _os.remove(p)

        fresh = DB(tmp_path, header_size=orig_header, cache_size=0, sync_writes=orig_sync)
        try:
            for name, schema, indexes, ks, iks, records in snapshots:
                ncol = fresh.collection(name, schema, indexes=indexes,
                                        key_size=ks, index_key_size=iks)
                if records:
                    ncol.insert_many(records)
        finally:
            fresh.close()

        # Atomically swap the compacted file in, then reopen on it.
        self.close()
        _os.replace(tmp_path, self.filename)
        for ext in (".log", ".lock"):
            p = tmp_path + ext
            if _os.path.exists(p):
                _os.remove(p)
        self._db = ByteFileDB(self.filename, header_size=orig_header,
                              sync_writes=orig_sync, flag=self.flag)
        self._datasets = {}
        self._datastructures = {}
        self._blob_store = None
        self._is_open = False
        if self._shared_cache is not None:
            self._shared_cache.clear()
        self.open()

    def _collection_names(self):
        return sorted(
            k[len("__collection__::"):]
            for k in self._db._header_data
            if k.startswith("__collection__::")
        )

    def stats(self):
        """Return a snapshot of disk usage and per-structure sizes.

        Keys: filename, file_size, header_size, allocated_bytes (high-water
        mark), free_bytes / free_blocks (reclaimable by vacuum),
        fragmentation (free/allocated), collections, structures (top-level,
        non-collection)."""
        import os as _os

        fio = self._db
        file_size = _os.path.getsize(self.filename) if _os.path.exists(self.filename) else 0
        allocated = fio._header_data.get(fio._allocation_index_key, fio.header_size)
        free_bytes = sum(sz for _addr, sz in getattr(fio, "_freelist", []))
        coll_names = self._collection_names()
        owned_pats = [f"{c}__" for c in coll_names]

        def _owned(n):
            return any(p in n for p in owned_pats)

        def _safe_len(s):
            try:
                return len(s)
            except Exception:
                return None

        collections = {}
        for name in coll_names:
            col = self.collection(name)
            collections[name] = {
                "records": len(col),
                "indexes": list(col._indexes.keys()),
                "search": list(col._search.keys()),
            }
        structures = {
            n: {"type": type(s).__name__, "length": _safe_len(s)}
            for n, s in self._datastructures.items() if not _owned(n)
        }
        return {
            "filename": self.filename,
            "file_size": file_size,
            "header_size": fio.header_size,
            "allocated_bytes": int(allocated),
            "free_bytes": int(free_bytes),
            "free_blocks": len(getattr(fio, "_freelist", [])),
            "fragmentation": (free_bytes / allocated) if allocated else 0.0,
            "collections": collections,
            "structures": structures,
        }

    def verify(self):
        """Walk every structure and collection, returning a report of any
        inconsistency found (dangling secondary-index pks, unreadable
        structures).  O(n) — for occasional integrity checks.

        Returns {"ok": bool, "issues": [str, ...]}."""
        issues = []
        for name, s in self._datastructures.items():
            try:
                if hasattr(s, "__len__"):
                    len(s)
            except Exception as e:                       # pragma: no cover
                issues.append(f"structure {name!r}: not readable ({e})")
        for cname in self._collection_names():
            try:
                col = self.collection(cname)
            except Exception as e:                       # pragma: no cover
                issues.append(f"collection {cname!r}: cannot open ({e})")
                continue
            primary = col._primary
            for idx_name, ix in col._indexes.items():
                struct = ix["struct"]
                try:
                    pairs = struct.items()
                except Exception as e:                   # pragma: no cover
                    issues.append(f"collection {cname!r} index {idx_name!r}: unreadable ({e})")
                    continue
                for _key, entry in pairs:
                    pk = str(entry["pk"])
                    if pk not in primary:
                        issues.append(
                            f"collection {cname!r} index {idx_name!r}: "
                            f"dangling pk {pk!r}"
                        )
        return {"ok": not issues, "issues": issues}

    def _rollback_collection(self, ds_before, struct_before, cfg_key):
        """Undo a partially-created collection: unregister every dataset /
        datastructure created since the snapshot, and drop the config key."""
        hd = self._db._header_data
        for n in list(set(self._datastructures) - struct_before):
            self._datastructures.pop(n, None)
            hd.pop(f"_ds_{n}_metadata", None)
        for n in list(set(self._datasets) - ds_before):
            self._datasets.pop(n, None)
        hd.pop(cfg_key, None)
        self._save_datastructures_registry()
        self._save_registry()

    @staticmethod
    def _field_enc_width(ds, field):
        """Max bytes encode_value() produces for a field (None if unbounded).

        Used to size index keys so a long primary key / value is never
        truncated in an index entry."""
        if field in getattr(ds, "_datetime_fields", set()):
            return 22
        if field in getattr(ds, "_utf8_fields", {}):
            return ds._utf8_fields[field]
        if (field in getattr(ds, "_text_fields", set())
                or field in getattr(ds, "_json_fields", set())
                or field in getattr(ds, "_blob_fields", set())):
            return None
        dt = ds.user_schema.fields[field][0]
        if dt.kind in ("i", "u", "f"):
            return 20
        if dt.kind == "b":
            return 1
        if dt.kind in ("U", "S"):
            return dt.itemsize          # bytes; the value's utf8 never exceeds this
        return 20

    def _build_collection(self, name, model, specs, primary_field,
                          key_size, index_key_size, cfg_key):
        from loom.collection import Collection
        from loom.dataset import Dataset
        if isinstance(model, Dataset):
            dataset = model
        elif isinstance(model, dict):
            dataset = self.create_dataset(f"{name}__data", **model)
        else:
            dataset = self.create_dataset(f"{name}__data", model)

        # Auto-size key_size / index_key_size from the declared field widths so
        # a long primary key (e.g. a URL) is never truncated in index entries —
        # otherwise find()/range()/search() silently drop rows.  An index on an
        # unbounded field (text/blob/json) is *hashed* (fixed 32-byte group key,
        # equality only) so possibly-long text values can be indexed too.
        _HASH_W = 32
        pk_w = self._field_enc_width(dataset, primary_field)
        if pk_w is not None:
            key_size = max(key_size, pk_w)
        eff_pk = pk_w if pk_w is not None else key_size   # text pk: caller's budget
        need_ik = eff_pk + 4
        hashed_index = {}
        for idx_name, spec in specs.items():
            if spec.kind in ("primary", "search"):
                continue
            fld = getattr(spec, "field", None) or idx_name
            vw = self._field_enc_width(dataset, fld)
            hashed = vw is None                      # text/blob/json → hash the group
            if hashed and spec.kind == "range":
                raise ValueError(
                    f"collection {name!r}: range index {idx_name!r} needs an "
                    f"orderable (bounded) field, not {fld!r} (text/blob/json)"
                )
            hashed_index[idx_name] = hashed
            vw = _HASH_W if hashed else vw
            sw = 0
            if spec.kind == "many" and spec.sort is not None:
                sw = self._field_enc_width(dataset, spec.sort)
                if sw is None:
                    raise ValueError(
                        f"collection {name!r}: many index {idx_name!r} sort field "
                        f"{spec.sort!r} must be orderable (bounded), not text/blob/json"
                    )
            need_ik = max(need_ik, vw + sw + eff_pk + 4)
        index_key_size = max(index_key_size, need_ik)

        primary = self.create_dict(
            f"{name}__primary", dataset, key_size=key_size, max_key_len=key_size
        )

        idx_objs = {}
        cfg_indexes = {}
        for idx_name, spec in specs.items():
            if spec.kind in ("primary", "search"):
                continue
            # the index NAME labels the structures (unique); the FIELD is what
            # gets indexed (defaults to the name, but can be shared across
            # several indexes — e.g. one Many by date, one by engagement).
            field = getattr(spec, "field", None) or idx_name
            ix_ds = self.create_dataset(f"{name}__ix_{idx_name}", pk=f"utf8[{key_size}]")
            if spec.kind == "unique":
                struct = self.create_dict(
                    f"{name}__ix_{idx_name}__d", ix_ds,
                    key_size=key_size, max_key_len=key_size,
                )
            else:  # range / many → BTree composite
                struct = self.create_btree(
                    f"{name}__bt_{idx_name}", ix_ds, key_size=index_key_size
                )
            sources = [field]
            if spec.kind == "many" and spec.sort is not None:
                sources.append(spec.sort)
            idx_objs[idx_name] = {
                "name": idx_name, "spec": spec, "field": field,
                "struct": struct, "sources": sources,
                "hashed": hashed_index[idx_name],
            }
            cfg_indexes[idx_name] = {
                "kind": spec.kind, "field": field,
                "sort": getattr(spec, "sort", None),
                "desc": getattr(spec, "desc", False),
                "struct": struct.name,
                "hashed": hashed_index[idx_name],
            }

        # Full-text (search) indexes: a doc-store-less SearchIndex (postings +
        # doc lengths only, no record duplication) + the collection's own
        # doc_id↔pk maps (docid2pk List for results, pk2docid Dict for delete).
        search_objs = {}
        cfg_search = {}
        for field, spec in specs.items():
            if spec.kind != "search":
                continue
            fields = spec.fields or [field]
            si = self.create_search_index(
                f"{name}__search_{field}", store_documents=False,
                scoring=spec.scoring, bm25_k1=spec.bm25_k1, bm25_b=spec.bm25_b,
            )
            d2p_ds = self.create_dataset(f"{name}__d2p_{field}", pk=f"utf8[{key_size}]")
            docid2pk = self.create_list(f"{name}__d2p_{field}__l", d2p_ds)
            p2d_ds = self.create_dataset(f"{name}__s2d_{field}", doc_id="int64")
            p2d = self.create_dict(
                f"{name}__s2d_{field}__d", p2d_ds, key_size=key_size, max_key_len=key_size
            )
            search_objs[field] = {"fields": fields, "index": si,
                                  "pk2docid": p2d, "docid2pk": docid2pk}
            cfg_search[field] = {"fields": fields, "search": si.name,
                                 "pk2docid": p2d.name, "docid2pk": docid2pk.name}

        self._db.set_header_field(cfg_key, {
            "dataset": dataset.name,
            "primary": primary.name,
            "primary_field": primary_field,
            "key_size": key_size,
            "index_key_size": index_key_size,
            "indexes": cfg_indexes,
            "search": cfg_search,
        })
        return Collection(
            self, name, dataset, primary_field, primary, idx_objs, key_size,
            search=search_objs,
        )
