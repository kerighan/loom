import numpy as np

from loom.ref import Ref
from loom.errors import (
    InvalidIdentifierError,
    DeletedRecordError,
    WrongDatasetError,
)

# Blob reference dtype: (offset: uint64, n_slots: uint16)
# Total 10 bytes per blob field
BLOB_DTYPE = np.dtype([("offset", "uint64"), ("n_slots", "uint16")])

# Sentinel for null/empty blob references
_NULL_BLOB = (0, 0)


class Dataset:
    """A typed dataset with prefix-based identification.

    Each record is prefixed with a unique identifier byte, enabling:
    - Multiple datasets in one file
    - Type safety at byte level
    - Soft deletes (negative prefix)

    User API:
        dataset.write(address, **record)  # Write a record
        dataset.read(address)              # Read a record
        dataset.read_field(address, field) # Read single field
        dataset.delete(address)            # Soft delete
        dataset.exists(address)            # Check if valid
    """

    def __init__(self, dataset_name, db, identifier, blob_store=None, **schema):
        """
        Args:
            dataset_name: Dataset name (string)
            db: ByteFileDB instance
            identifier: Unique ID (1-127, positive int8)
            blob_store: BlobStore instance (required for "blob" or "text" fields)
            **schema: Field definitions as numpy dtypes.
                      Use "blob" for raw variable-length bytes,
                      "text" for variable-length UTF-8 strings.

        Example:
            Dataset('messages', db, 1, id='uint64', content='text')
        """
        if not (1 <= identifier <= 127):
            raise InvalidIdentifierError(identifier)

        self.name = dataset_name
        self.db = db
        self.identifier = identifier
        self.blob_store = blob_store

        # Track special variable-length fields
        self._blob_fields = set()   # raw bytes via BlobStore
        self._text_fields = set()   # UTF-8 strings via BlobStore (transparent)

        # Convert "blob"/"text" strings to actual blob dtype
        processed_schema = []
        for field, dtype in schema.items():
            if dtype == "blob":
                self._blob_fields.add(field)
                processed_schema.append((field, BLOB_DTYPE))
            elif dtype == "text":
                self._text_fields.add(field)
                processed_schema.append((field, BLOB_DTYPE))
            else:
                processed_schema.append((field, dtype))

        # Build schema with prefix
        self.schema = np.dtype([("_prefix", "int8")] + processed_schema)
        self.user_schema = np.dtype(processed_schema)  # Schema without prefix
        self.record_size = self.schema.itemsize

        # Prefix bytes for valid/deleted records
        self._valid_prefix = np.int8(identifier).tobytes()
        self._deleted_prefix = np.int8(-identifier).tobytes()

    def allocate_block(self, n_records):
        """Allocate space for n records and return the address.

        Args:
            n_records: Number of records to allocate space for

        Returns:
            Address of the allocated block
        """
        size = n_records * self.record_size
        return self.db.allocate(size)

    def _serialize(self, **record):
        """Convert record dict to bytes with prefix.

        For blob fields, expects (offset, n_slots) tuple.
        For text fields, expects a str — stored transparently via BlobStore.
        """
        # Build full record with prefix
        values = [self.identifier]
        for field in self.user_schema.names:
            if field in record:
                value = record[field]
                if field in self._text_fields:
                    # Encode string and store in BlobStore
                    if value is None or value == "":
                        values.append(_NULL_BLOB)
                    else:
                        encoded = value.encode("utf-8") if isinstance(value, str) else value
                        offset, n_slots = self.blob_store.write(encoded)
                        values.append((offset, n_slots))
                elif field in self._blob_fields:
                    if value is None:
                        value = _NULL_BLOB
                    values.append(value)
                else:
                    values.append(value)
            else:
                # Use proper default for field type
                dtype = self.user_schema.fields[field][0]
                if dtype.kind == "U":  # Unicode string
                    values.append("")
                elif field in self._blob_fields or field in self._text_fields:
                    values.append(_NULL_BLOB)
                else:
                    values.append(0)
        arr = np.array(tuple(values), dtype=self.schema)
        return arr.tobytes()

    def _deserialize(self, data):
        """Convert bytes to record dict (without prefix).

        For blob fields, returns (offset, n_slots) tuple.
        For text fields, returns the decoded UTF-8 string transparently.
        Actual blob data must be fetched separately via BlobStore.
        """
        arr = np.frombuffer(data, dtype=self.schema)[0]
        result = {}
        for field in self.user_schema.names:
            value = arr[field]
            if field in self._text_fields:
                offset = int(value["offset"])
                n_slots = int(value["n_slots"])
                if offset == 0 and n_slots == 0:
                    result[field] = ""
                else:
                    result[field] = self.blob_store.read(offset).decode("utf-8")
            elif field in self._blob_fields:
                # Convert structured array to tuple
                offset = int(value["offset"])
                n_slots = int(value["n_slots"])
                if offset == 0 and n_slots == 0:
                    result[field] = None  # Null blob
                else:
                    result[field] = (offset, n_slots)
            else:
                result[field] = value
        return result

    def write(self, address, **record):
        """Write a record at the given address.

        Args:
            address: Where to write
            **record: Field values

        Example:
            dataset.write(addr, id=1, name='Alice', age=30)
        """
        data = self._serialize(**record)
        self.db.write(address, data)

    def read(self, address):
        """Read a record from the given address.

        Args:
            address: Where to read from

        Returns:
            Dict with field values

        Raises:
            ValueError: If record is deleted or wrong type
        """
        # Single read for both prefix check and data
        data = self.db.read(address, self.record_size)
        prefix = data[0:1]

        if prefix == self._deleted_prefix:
            raise DeletedRecordError(address)

        if prefix != self._valid_prefix:
            prefix_val = int(np.frombuffer(prefix, dtype="int8")[0])
            raise WrongDatasetError(address, self.identifier, prefix_val)

        return self._deserialize(data)

    def read_many(self, address, count, as_array=False):
        """Read multiple contiguous records starting at address.

        Args:
            address: Starting address
            count: Number of records to read
            as_array: If True, return raw NumPy structured array (fast).
                      If False, return list of dicts (convenient).

        Returns:
            NumPy array (if as_array=True) or list of record dicts
        """
        if count <= 0:
            return np.array([], dtype=self.schema) if as_array else []

        # Read entire block as one slice
        total_size = count * self.record_size
        data = self.db.read(address, total_size)

        # Parse all records at once with NumPy
        arr = np.frombuffer(data, dtype=self.schema)

        if as_array:
            # Fast path: return NumPy array directly
            # Find where valid records end
            prefixes = arr["_prefix"]
            valid_mask = (prefixes == self.identifier) | (prefixes == -self.identifier)
            # Find first invalid (uninitialized) record
            invalid_indices = np.where(~valid_mask)[0]
            if len(invalid_indices) > 0:
                arr = arr[: invalid_indices[0]]
            return arr

        # Slow path: convert to list of dicts
        results = []
        for rec in arr:
            prefix = rec["_prefix"]
            if prefix == self.identifier or prefix == -self.identifier:
                d = {}
                for field in self.user_schema.names:
                    if field in self._text_fields:
                        value = rec[field]
                        offset = int(value["offset"])
                        n_slots = int(value["n_slots"])
                        d[field] = "" if (offset == 0 and n_slots == 0) else self.blob_store.read(offset).decode("utf-8")
                    elif field in self._blob_fields:
                        value = rec[field]
                        offset = int(value["offset"])
                        n_slots = int(value["n_slots"])
                        d[field] = None if (offset == 0 and n_slots == 0) else (offset, n_slots)
                    else:
                        d[field] = rec[field]
                if prefix == -self.identifier:
                    d["valid"] = False
                results.append(d)
            else:
                # Uninitialized or wrong type - stop here
                break

        return results

    def delete(self, address):
        """Soft delete a record by flipping its prefix.

        For text fields, the associated blobs are freed before deletion.

        Args:
            address: Address of record to delete
        """
        if self._text_fields:
            data = self.db.read(address, self.record_size)
            arr = np.frombuffer(data, dtype=self.schema)[0]
            for field in self._text_fields:
                value = arr[field]
                offset = int(value["offset"])
                n_slots = int(value["n_slots"])
                if offset != 0 or n_slots != 0:
                    self.blob_store.delete(offset, n_slots)
        self.db.write(address, self._deleted_prefix)

    def write_field(self, address, field_name, value):
        """Update a single field in a record.

        For text fields, the old blob is freed and a new one is written.

        Args:
            address: Record address
            field_name: Field to update
            value: New value (str for text fields)
        """
        if field_name not in self.user_schema.names:
            raise ValueError(f"Field '{field_name}' not in schema")

        field_offset = self.schema.fields[field_name][1]

        if field_name in self._text_fields:
            # Free old blob
            old_data = self.db.read(address + field_offset, BLOB_DTYPE.itemsize)
            old_ref = np.frombuffer(old_data, dtype=BLOB_DTYPE)[0]
            old_offset = int(old_ref["offset"])
            old_n_slots = int(old_ref["n_slots"])
            if old_offset != 0 or old_n_slots != 0:
                self.blob_store.delete(old_offset, old_n_slots)

            # Write new blob
            if value is None or value == "":
                new_ref = np.array([_NULL_BLOB], dtype=BLOB_DTYPE)
            else:
                encoded = value.encode("utf-8") if isinstance(value, str) else value
                new_offset, new_n_slots = self.blob_store.write(encoded)
                new_ref = np.array([(new_offset, new_n_slots)], dtype=BLOB_DTYPE)

            self.db.write(address + field_offset, new_ref.tobytes())
            return

        # Standard field update
        field_dtype = self.user_schema.fields[field_name][0]
        data = np.array([value], dtype=field_dtype).tobytes()
        self.db.write(address + field_offset, data)

    def read_field(self, address, field_name):
        """Read a single field from a record.

        Args:
            address: Record address
            field_name: Field to read

        Returns:
            Field value
        """
        if field_name not in self.user_schema.names:
            raise ValueError(f"Field '{field_name}' not in schema")

        # Verify prefix
        prefix = self.db.read(address, 1)
        if prefix != self._valid_prefix:
            raise ValueError(f"Invalid record at {address}")

        # Get field offset and dtype from schema
        field_offset = self.schema.fields[field_name][1]  # (dtype, offset)
        field_dtype = self.user_schema.fields[field_name][0]
        field_size = field_dtype.itemsize

        # Read field data
        data = self.db.read(address + field_offset, field_size)
        return np.frombuffer(data, dtype=field_dtype)[0]

    def exists(self, address):
        """Check if a valid record exists at address.

        Args:
            address: Address to check

        Returns:
            True if valid record exists, False otherwise
        """
        prefix = self.db.read(address, 1)
        return prefix == self._valid_prefix

    def is_deleted(self, address):
        """Check if a record is soft-deleted.

        Args:
            address: Address to check

        Returns:
            True if deleted, False otherwise
        """
        prefix = self.db.read(address, 1)
        return prefix == self._deleted_prefix

    def __getitem__(self, address):
        """Read a record using dict-like syntax.

        Args:
            address: Record address

        Returns:
            Dict with field values

        Example:
            record = dataset[addr]
        """
        if isinstance(address, Ref):
            address = address.addr
        return self.read(address)

    def __setitem__(self, address, record):
        """Write a record using dict-like syntax.

        Args:
            address: Record address
            record: Dict with field values

        Example:
            dataset[addr] = {'id': 1, 'name': 'Alice'}
        """
        if isinstance(address, Ref):
            address = address.addr
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict, got {type(record).__name__}")
        self.write(address, **record)

    def insert(self, record):
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict, got {type(record).__name__}")
        addr = self.allocate_block(1)
        self[addr] = record
        return Ref(self, int(addr))

    def __delitem__(self, address):
        """Delete a record using dict-like syntax.

        Args:
            address: Record address

        Example:
            del dataset[addr]
        """
        self.delete(address)

    def __contains__(self, address):
        """Check if a valid record exists using 'in' operator.

        Args:
            address: Record address

        Returns:
            True if valid record exists

        Example:
            if addr in dataset: ...
        """
        return self.exists(address)

    def __repr__(self):
        fields = ", ".join(
            f"{name}={dtype}"
            for name, dtype in zip(self.user_schema.names, self.user_schema.descr)
        )
        return f"Dataset('{self.name}', id={self.identifier}, {fields})"
