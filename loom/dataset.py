import numpy as np

# Blob reference dtype: (offset: uint64, n_slots: uint16)
# Total 10 bytes per blob field
BLOB_DTYPE = np.dtype([("offset", "uint64"), ("n_slots", "uint16")])


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

    def __init__(self, dataset_name, db, identifier, **schema):
        """
        Args:
            dataset_name: Dataset name (string)
            db: ByteFileDB instance
            identifier: Unique ID (1-127, positive int8)
            **schema: Field definitions as numpy dtypes

        Example:
            Dataset('users', db, 1, id='uint64', name='U50', age='int32')
        """
        if not (1 <= identifier <= 127):
            raise ValueError(f"Identifier must be 1-127, got {identifier}")

        self.name = dataset_name
        self.db = db
        self.identifier = identifier

        # Track blob fields (special handling)
        self._blob_fields = set()

        # Convert "blob" string to actual blob dtype
        processed_schema = []
        for field, dtype in schema.items():
            if dtype == "blob":
                self._blob_fields.add(field)
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
        """
        # Build full record with prefix
        values = [self.identifier]
        for field in self.user_schema.names:
            if field in record:
                value = record[field]
                # Blob fields store (offset, n_slots) tuple
                if field in self._blob_fields:
                    if value is None:
                        value = (0, 0)  # Null blob
                    # value should be (offset, n_slots) tuple
                values.append(value)
            else:
                # Use proper default for field type
                dtype = self.user_schema.fields[field][0]
                if dtype.kind == "U":  # Unicode string
                    values.append("")
                elif field in self._blob_fields:
                    values.append((0, 0))  # Null blob reference
                else:
                    values.append(0)
        arr = np.array(tuple(values), dtype=self.schema)
        return arr.tobytes()

    def _deserialize(self, data):
        """Convert bytes to record dict (without prefix).

        For blob fields, returns (offset, n_slots) tuple.
        Actual blob data must be fetched separately via BlobStore.
        """
        arr = np.frombuffer(data, dtype=self.schema)[0]
        result = {}
        for field in self.user_schema.names:
            value = arr[field]
            if field in self._blob_fields:
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
        # Read prefix first
        prefix = self.db.read(address, 1)

        if prefix == self._deleted_prefix:
            raise ValueError(f"Record at {address} is deleted")

        if prefix != self._valid_prefix:
            prefix_val = np.frombuffer(prefix, dtype="int8")[0]
            raise ValueError(
                f"Wrong dataset at {address}: expected {self.identifier}, got {prefix_val}"
            )

        # Read full record
        data = self.db.read(address, self.record_size)
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
            if prefix == self.identifier:
                # Valid record
                results.append({field: rec[field] for field in self.user_schema.names})
            elif prefix == -self.identifier:
                # Deleted record - still include but mark as invalid
                d = {field: rec[field] for field in self.user_schema.names}
                d["valid"] = False
                results.append(d)
            else:
                # Uninitialized or wrong type - stop here
                break

        return results

    def delete(self, address):
        """Soft delete a record by flipping its prefix.

        Args:
            address: Address of record to delete
        """
        self.db.write(address, self._deleted_prefix)

    def write_field(self, address, field_name, value):
        """Update a single field in a record.

        Args:
            address: Record address
            field_name: Field to update
            value: New value
        """
        if field_name not in self.user_schema.names:
            raise ValueError(f"Field '{field_name}' not in schema")

        # Get field offset from schema
        field_offset = self.schema.fields[field_name][1]  # (dtype, offset)
        field_dtype = self.user_schema.fields[field_name][0]

        # Serialize value
        data = np.array([value], dtype=field_dtype).tobytes()

        # Write at field location
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
        return self.read(address)

    def __setitem__(self, address, record):
        """Write a record using dict-like syntax.

        Args:
            address: Record address
            record: Dict with field values

        Example:
            dataset[addr] = {'id': 1, 'name': 'Alice'}
        """
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict, got {type(record).__name__}")
        self.write(address, **record)

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
