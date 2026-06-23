"""
Base class for all data structures.

Provides common interface and utilities for building high-level
data structures on top of Datasets.
"""

import functools
import json
from abc import ABC, abstractmethod
from loom.datastructures.template import DataStructureTemplate


def write_op(fn):
    """Decorate a DataStructure write method to acquire the DB write lock.

    Ensures every public mutation (insert, delete, append, …) is atomic
    at the operation level.  Re-entrant via threading.RLock, so nested
    write calls (e.g. Dict.__setitem__ → nested List.append) never
    deadlock.  Also acquires fcntl.flock when DB was opened with
    multiprocess_safe=True.

    Overhead: ~50-100 ns uncontested — negligible vs µs-scale mmap writes.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._db.write_lock():
            return fn(self, *args, **kwargs)
    return wrapper

# Global registry for DataStructure types (for reference reconstruction)
_DS_REGISTRY = {}


class DataStructure(ABC):
    """Base class for all Loom data structures.

    Data structures are high-level abstractions built on top of Datasets.
    They provide familiar interfaces (list, dict, set, etc.) while
    maintaining persistence and type safety.

    Design principles:
    - Each data structure is independent and optimized
    - Top-level metadata stored in DB header; nested metadata stored inline
    - Can compose other data structures (e.g., Dict uses BloomFilter)
    - Clean separation from Dataset layer
    """

    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses for reference reconstruction."""
        super().__init_subclass__(**kwargs)
        _DS_REGISTRY[cls.__name__] = cls

    def __init__(self, name, db, auto_save_interval=None, _parent=None):
        """Initialize data structure.

        Args:
            name: Unique name for this data structure
            db: DB instance (not ByteFileDB - the high-level orchestrator)
            auto_save_interval: Auto-save metadata every N operations.
                If None (default), inherits from db.auto_save_interval.
                Set to 0 to disable.
            _parent: Parent data structure if this is nested (internal use)
        """
        self.name = name
        self._db = db
        self._parent = _parent  # Parent structure if nested
        self._metadata_key = f"_ds_{name}_metadata"
        if auto_save_interval is not None:
            self._auto_save_interval = auto_save_interval
        else:
            self._auto_save_interval = getattr(db, "auto_save_interval", 100)
        self._ops_since_save = 0
        # For nested structures, metadata stored inline
        # Only set to None if not already set (e.g., by from_ref)
        if not hasattr(self, "_inline_metadata"):
            self._inline_metadata = None

        # Auto-register for auto-save on DB close (only top-level structures)
        if _parent is None:
            db._datastructures[name] = self

    # ---- Registry protocol (generic save/load for DB persistence) ----

    def _get_registry_params(self):
        """Return the params dict needed to reconstruct this structure.

        Override in subclasses. The dict is stored in the DB header
        registry and passed back to _from_registry_params on reload.

        Returns:
            Dict of picklable params, or None to skip registry.
        """
        return None

    @classmethod
    def _from_registry_params(cls, name, db, params):
        """Reconstruct an instance from saved registry params.

        Override in subclasses.  Called by DB._load_registry() to
        recreate a data structure that was previously persisted.

        Args:
            name: Structure name
            db: DB instance
            params: The dict returned by _get_registry_params()

        Returns:
            DataStructure instance
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _from_registry_params()"
        )

    # ---- Template reconstruction protocol ----

    @classmethod
    def _reconstruct_template(cls, db, template_config, template_class_name):
        """Reconstruct a DataStructureTemplate from saved metadata.

        Default implementation handles the common case where the template
        wraps a real Dataset. Override in subclasses that use a custom
        template class (e.g., Set uses SetTemplate with no real dataset).

        Args:
            db: DB instance
            template_config: The config dict saved in metadata
            template_class_name: Ignored (cls is already the right class)

        Returns:
            DataStructureTemplate instance
        """
        # Default: standard DataStructureTemplate with a dataset
        template_dataset = db.get_dataset(template_config["_template_dataset"])
        config = {k: v for k, v in template_config.items() if k != "_template_dataset"}
        return DataStructureTemplate(cls, template_dataset, config)

    @abstractmethod
    def _initialize(self):
        """Initialize the data structure (first time setup).

        Called when creating a new data structure.
        Should create necessary datasets and set initial metadata.
        """
        pass

    @abstractmethod
    def _load(self):
        """Load existing data structure from DB.

        Called when opening an existing data structure.
        Should load metadata and reconnect to datasets.
        """
        pass

    def _save_metadata(self, metadata):
        """Save metadata.

        For top-level structures: saves to DB header
        For nested structures: stores inline (will be saved with parent's reference)

        Args:
            metadata: Dict of metadata to save
        """
        if self._parent is None:
            # Top-level: save to header
            self._db._db.set_header_field(self._metadata_key, metadata)
        else:
            # Nested: store inline (parent will save it)
            self._inline_metadata = metadata

    def _load_metadata(self, default=None):
        """Load metadata.

        For top-level structures: loads from DB header
        For nested structures: loads from inline storage

        Args:
            default: Default value if metadata doesn't exist

        Returns:
            Metadata dict or default
        """
        if self._parent is None:
            # Top-level: load from header
            return self._db._db.get_header_field(self._metadata_key, default or {})
        else:
            # Nested: load from inline storage
            return self._inline_metadata or default or {}

    def _get_dataset(self, dataset_name):
        """Get a dataset by name.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dataset instance
        """
        return self._db.get_dataset(dataset_name)

    def _make_cache(self, suffix: str, hint_size: int = 0):
        """Create a cache for this structure.

        Caching is governed SOLELY by the DB-level shared cache (one LRU for
        the whole DB, sized by ``DB(cache_size=…)``).  If it exists, every
        structure namespaces into it; otherwise nothing is cached.

        There are no per-structure caches any more.  The old model gave each
        structure its own standalone LRU — unworkable for a Graph, whose
        thousands of inner adjacency dicts would each want a cache.  A single
        shared budget bounds total RAM and lets hot structures claim their
        share via LRU.  ``hint_size`` is accepted for backward compatibility
        but ignored.

        Caching addresses is always safe: records have stable absolute
        addresses (arena redesign — no migration), so a cached value_addr
        never goes stale.  This is what lets lookups skip the per-table scan.

        Args:
            suffix:    Short label appended to the namespace key,
                       e.g. "values", "nodes", "blocks".
            hint_size: Ignored (legacy per-structure cache size).
        """
        from loom.cache import NamespacedCache, NullCache

        shared = getattr(self._db, "_shared_cache", None)
        if shared is not None and self._should_cache():
            # Namespace by a STABLE persistent identity, resolved lazily on
            # every access (see NamespacedCache).  NOT by id(self): nested
            # structures are re-materialised on every parent lookup and their
            # ids get reused after GC, so an id()-based namespace would let a
            # fresh nested structure read a dead sibling's cached entries —
            # silent data corruption.
            return NamespacedCache(shared, lambda: f"{self._cache_namespace()}:{suffix}")
        return NullCache()

    def _should_cache(self):
        """Whether this structure may use the shared cache.

        Caching a value *address* across re-materialisations is always safe —
        addresses are stable (arena redesign, no migration).  Caching mutable
        *contents* (List blocks, BTree nodes) is only safe for a long-lived
        top-level object: a nested List/BTree is re-materialised on every
        parent lookup, so a cached block could be read back stale after a
        sibling materialisation mutated it.  List/BTree therefore override
        this to disable caching when nested; Dict (address cache) does not.
        """
        return True

    def _cache_namespace(self):
        """Stable, collision-free identity for this structure's cache entries.

        Defaults to the (unique, persistent) structure name.  Nested
        structures whose name is id-derived override this to return a stable
        persistent address (e.g. first table address) so the namespace is the
        same every time the structure is re-materialised.
        """
        return self.name

    def _auto_save_check(self):
        """Check if auto-save should trigger and save if needed.

        Call this after each mutating operation (add, remove, etc.).
        Automatically saves metadata every N operations.
        """
        if self._auto_save_interval > 0:
            self._ops_since_save += 1
            if self._ops_since_save >= self._auto_save_interval:
                if hasattr(self, "save"):
                    self.save()
                self._ops_since_save = 0

    @classmethod
    def _check_nesting(cls, outer_cls):
        """Raise NestingNotSupportedError if cls cannot be nested in outer_cls.

        Called by container constructors before creating shared datasets.

        Args:
            outer_cls: The container class (List, Dict, BTree, …)
        """
        from loom.errors import NestingNotSupportedError
        supported = getattr(cls, "_outer_types_supported", None)
        if supported is not None and outer_cls.__name__ not in supported:
            reason = (
                f"{cls.__name__} supports outer containers: "
                + (", ".join(supported) if supported else "none")
            )
            raise NestingNotSupportedError(outer_cls.__name__, cls.__name__, reason)

    @classmethod
    def template(cls, dataset, **config):
        """Create a template for nested data structures.

        Templates enable creating structures that contain other structures,
        e.g., List[List[User]], Dict[str, List[Task]], etc.

        Args:
            dataset: Dataset to use for storage
            **config: DataStructure-specific configuration

        Returns:
            DataStructureTemplate instance

        Example::

            UserList = List.template(user_dataset)
            teams = db.create_list('teams', UserList)
            eng = teams.append()  # Creates nested List
        """
        return DataStructureTemplate(cls, dataset, config)

    @staticmethod
    def _extract_schema(dataset_or_dict):
        """Extract schema dict from Dataset object or dict.

        Args:
            dataset_or_dict: Dataset instance or schema dict

        Returns:
            Schema dict mapping field names to dtype strings
        """
        if hasattr(dataset_or_dict, "user_schema"):
            # It's a Dataset object — extract schema, preserving all markers
            from loom.dataset import dtype_to_str
            ds = dataset_or_dict
            result = {}
            for name in ds.user_schema.names:
                if hasattr(ds, "_text_fields") and name in ds._text_fields:
                    result[name] = "text"
                elif name in getattr(ds, "_json_fields", set()):
                    result[name] = "json"
                elif hasattr(ds, "_blob_fields") and name in ds._blob_fields:
                    result[name] = "blob"
                elif name in getattr(ds, "_utf8_fields", {}):
                    strict = name in getattr(ds, "_utf8_strict", set())
                    result[name] = f"utf8[{ds._utf8_fields[name]}{'!' if strict else ''}]"
                elif name in getattr(ds, "_datetime_fields", set()):
                    result[name] = "datetime"
                else:
                    raw = ds.user_schema.fields[name][0]
                    # _DummySchema stores plain strings, real Dataset stores np.dtype
                    if isinstance(raw, str):
                        result[name] = raw
                    else:
                        result[name] = dtype_to_str(raw)
            return result
        else:
            # Already a dict
            return dataset_or_dict

    @classmethod
    def _get_ref_config_schema(cls):
        """Get schema for config fields in references.

        Override in subclasses to define DS-specific config fields
        that need to be stored in references for reconstruction.

        Returns:
            Dict mapping config field names to numpy dtypes

        Example::

            return {
                'growth_factor': 'float64',
                'p_init': 'uint32',
            }
        """
        return {}

    @classmethod
    def _get_nested_ref_schema(cls):
        """Get compact binary schema for nested structure references.

        Override in subclasses to define efficient binary storage for
        nested structures. This replaces JSON-serialized inline_metadata
        with fixed-size binary fields for much smaller storage.

        Returns:
            Dict mapping field names to numpy dtypes
        """
        # Default fallback: use JSON (subclasses should override)
        return {
            "ds_type": "U20",
            "ds_name": "U50",
            "inline_metadata": "U500",
        }

    @classmethod
    def get_shared_dataset_specs(cls, parent_name, inner_schema, **kwargs):
        """Get specifications for shared datasets needed when this type is nested.

        When a data structure is used as the inner type of a container
        (e.g., List[ThisType] or Dict[ThisType]), the container needs to
        create shared datasets for efficient storage. This method returns
        the specifications for those datasets.

        Override in subclasses to define what shared datasets are needed.

        Args:
            parent_name: Name of the parent container (for generating dataset names)
            inner_schema: Schema dict for the innermost data items

        Returns:
            Dict mapping attribute names to dataset specifications:
            {
                '_shared_items_dataset': {'name': '...', 'schema': {...}},
                '_shared_hash_table': {'name': '...', 'schema': {...}},
                ...
            }
        """
        # Default: no shared datasets needed
        return {}

    def set_shared_datasets(self, shared_datasets):
        """Set shared datasets on this nested instance.

        Called by the parent container after reconstructing a nested instance
        from a reference. The parent passes the shared datasets it created.

        Override in subclasses to set the appropriate attributes.

        Args:
            shared_datasets: Dict mapping attribute names to Dataset instances
        """
        for attr_name, dataset in shared_datasets.items():
            setattr(self, attr_name, dataset)

    def needs_shared_datasets(self):
        """Check if this nested instance needs shared datasets to be set.

        Called by the parent container after reconstructing a nested instance
        to determine if set_shared_datasets() should be called.

        Override in subclasses to check the appropriate attributes.

        Returns:
            True if shared datasets need to be set, False otherwise
        """
        return False

    def to_ref(self):
        """Get self-contained reference to this data structure.

        Returns a dict containing all information needed to reconstruct
        this data structure instance. Used for nested data structures.

        For nested structures, includes inline metadata to avoid header pollution.

        Returns:
            Reference dict
        """
        ref = {
            "ds_type": self.__class__.__name__,
            "ds_name": self.name,
            **self._get_ref_fields(),
        }

        # For nested structures, include metadata inline (serialized as JSON)
        if self._parent is not None:
            # Get current metadata (either from inline storage or by calling save logic)
            if hasattr(self, "_get_current_metadata"):
                metadata = self._get_current_metadata()
            elif self._inline_metadata is not None:
                metadata = self._inline_metadata
            else:
                metadata = None

            if metadata is not None:
                # Serialize metadata to JSON string
                ref["inline_metadata"] = json.dumps(metadata)
            else:
                # Fallback: reference metadata key (shouldn't happen for nested)
                ref["metadata_key"] = ""
                ref["inline_metadata"] = ""
        else:
            # Top-level: reference metadata key in header
            ref["metadata_key"] = self._metadata_key
            ref["inline_metadata"] = ""  # Empty for top-level

        return ref

    def _get_ref_fields(self):
        """Get DS-specific reference fields.

        Override in subclasses to add custom fields to references.

        Returns:
            Dict of additional reference fields
        """
        return {}

    @classmethod
    def from_ref(cls, db, ref):
        """Reconstruct data structure from reference.

        Generic implementation with caching. Subclasses should implement
        _from_ref_impl() for actual reconstruction logic.

        Args:
            db: Database instance
            ref: Reference dict from to_ref()

        Returns:
            DataStructure instance
        """
        # Check if this is binary format (no ds_name/ds_type) or standard format
        if "ds_name" not in ref:
            # Binary format - call the class's _from_ref_impl directly
            # (cls is the calling class, e.g., List)
            return cls._from_ref_impl(db, ref)

        # Standard format with ds_name/ds_type
        # Check cache first (avoid loading same DS multiple times)
        if ref["ds_name"] in db._datastructures:
            return db._datastructures[ref["ds_name"]]

        # Get the actual class from registry
        ds_class = _DS_REGISTRY[ref["ds_type"]]

        # Reconstruct using subclass implementation
        # Pass the full ref so subclass can access inline_metadata if needed
        instance = ds_class._from_ref_impl(db, ref)

        # Cache it (only if top-level - nested structures handled by parent)
        # Top-level refs have non-empty metadata_key; nested refs have empty string
        if ref.get("metadata_key"):
            db._datastructures[ref["ds_name"]] = instance

        return instance

    @classmethod
    def _from_ref_impl(cls, db, ref):
        """Implement reconstruction logic.

        Override in subclasses to define how to reconstruct from reference.

        Args:
            db: Database instance
            ref: Reference dict

        Returns:
            DataStructure instance
        """
        raise NotImplementedError(f"{cls.__name__} must implement _from_ref_impl()")

    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}('{self.name}')"
