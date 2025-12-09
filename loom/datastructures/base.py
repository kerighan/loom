"""
Base class for all data structures.

Provides common interface and utilities for building high-level
data structures on top of Datasets.
"""

import json
from abc import ABC, abstractmethod
from loom.datastructures.template import DataStructureTemplate

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

    def __init__(self, name, db, auto_save_interval=1000, _parent=None):
        """Initialize data structure.

        Args:
            name: Unique name for this data structure
            db: DB instance (not ByteFileDB - the high-level orchestrator)
            auto_save_interval: Auto-save metadata every N operations (0 to disable)
            _parent: Parent data structure if this is nested (internal use)
        """
        self.name = name
        self._db = db
        self._parent = _parent  # Parent structure if nested
        self._metadata_key = f"_ds_{name}_metadata"
        self._auto_save_interval = auto_save_interval
        self._ops_since_save = 0
        # For nested structures, metadata stored inline
        # Only set to None if not already set (e.g., by from_ref)
        if not hasattr(self, "_inline_metadata"):
            self._inline_metadata = None

        # Auto-register for auto-save on DB close (only top-level structures)
        if _parent is None:
            db._datastructures[name] = self

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
    def template(cls, dataset, **config):
        """Create a template for nested data structures.

        Templates enable creating structures that contain other structures,
        e.g., List[List[User]], Dict[str, List[Task]], etc.

        Args:
            dataset: Dataset to use for storage
            **config: DataStructure-specific configuration

        Returns:
            DataStructureTemplate instance

        Example:
            UserList = List.template(user_dataset, cache_size=10)
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
            # It's a Dataset object - extract schema
            return {
                name: str(dataset_or_dict.user_schema.fields[name][0])
                for name in dataset_or_dict.user_schema.names
            }
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

        Example:
            return {
                'cache_size': 'uint32',
                'growth_factor': 'float64',
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
    def get_shared_dataset_specs(cls, parent_name, inner_schema):
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
