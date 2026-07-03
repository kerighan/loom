import json
import re
import struct
import sys

import numpy as np

from loom.ref import Ref
from loom.errors import (
    InvalidIdentifierError,
    DeletedRecordError,
    WrongDatasetError,
)

# ── Array dtype helpers ────────────────────────────────────────────────────

_ARRAY_DTYPE_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*)\[(\d+(?:,\d+)*)\]$")


def parse_dtype(dtype_str):
    """Parse a dtype string, supporting array notation like 'float32[1536]'.

    Returns a numpy dtype object.

    Examples:
        'uint32'           → np.dtype('uint32')
        'float32[1536]'    → np.dtype(('float32', (1536,)))
        'float32[8,8]'     → np.dtype(('float32', (8, 8)))
    """
    m = _ARRAY_DTYPE_RE.match(str(dtype_str))
    if m:
        base   = m.group(1)
        shape  = tuple(int(d) for d in m.group(2).split(","))
        return np.dtype((base, shape))
    return np.dtype(dtype_str)


def dtype_to_str(dtype):
    """Serialize a numpy dtype back to a string, preserving array shapes.

    Inverse of parse_dtype().

    Examples:
        np.dtype('uint32')                    → 'uint32'   (via .str)
        np.dtype(('float32', (1536,)))        → 'float32[1536]'
        np.dtype(('float32', (8, 8)))         → 'float32[8,8]'
    """
    if dtype.shape:
        base_str = dtype.base.str          # e.g. '<f4'
        # Use the named form if available, else fall back to .str
        try:
            base_str = str(dtype.base)     # e.g. 'float32'
        except Exception:
            pass
        shape_str = ",".join(str(d) for d in dtype.shape)
        return f"{base_str}[{shape_str}]"
    # Friendly named form ('int64', 'uint32', 'float64', 'bool') rather than the
    # raw numpy code ('<i8'), as the docstring promises. Both round-trip through
    # np.dtype() / parse_dtype(), so this stays backward-compatible for reading
    # older registries that stored the '<i8' form.
    named = str(dtype)
    try:
        if np.dtype(named) == dtype:
            return named
    except TypeError:
        pass
    return dtype.str

# Blob reference dtype: (offset: uint64, n_slots: uint16)
# Total 10 bytes per blob field
BLOB_DTYPE = np.dtype([("offset", "uint64"), ("n_slots", "uint16")])

# Sentinel for null/empty blob references
_NULL_BLOB = (0, 0)


def _to_native(value):
    """Convert a numpy scalar (np.int64/float64/bool_/str_) to its native
    Python type on read, so records are clean Python dicts (JSON-serialisable,
    `isinstance(_, int)` works).  Array/vector fields (np.ndarray) are NOT
    numpy scalars and pass through unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def as_record(value):
    """Coerce a record argument to a plain dict.

    Accepts a dict (returned unchanged) or a Pydantic model — so you can do
    ``posts.append(Post(...))`` instead of ``posts.append(post.model_dump())``.
    Pydantic v2 (`model_dump`) and v1 (`dict`) are both supported; anything
    else is passed through untouched."""
    if value is None or isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)   # pydantic v2
    if callable(dump):
        return dump()
    legacy = getattr(value, "dict", None)        # pydantic v1
    if callable(legacy) and type(value).__module__.startswith("pydantic"):
        return legacy()
    return value


# ── Datetime ↔ int64 (epoch microseconds) ──────────────────────────────────
# "datetime" fields are stored inline as int64 microseconds since 1970-01-01,
# encoded/decoded transparently.  8 bytes, naturally ordered (range/sort work),
# no BlobStore hop.  Naive datetimes are treated as UTC; aware ones are
# converted to UTC; on read you get a naive (UTC) datetime back.
from datetime import date as _date, datetime as _datetime, timedelta as _timedelta, timezone as _timezone

_EPOCH = _datetime(1970, 1, 1)


def _dt_to_micros(value):
    """Encode a datetime/date (or raw int micros) to int64 epoch-microseconds."""
    if value is None:
        return 0
    if isinstance(value, _datetime):
        dt = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(_timezone.utc).replace(tzinfo=None)
    elif isinstance(value, _date):
        dt = _datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, np.integer)):
        return int(value)            # already microseconds — pass through
    elif isinstance(value, str):
        dt = _datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            dt = dt.astimezone(_timezone.utc).replace(tzinfo=None)
    else:
        raise TypeError(f"datetime field expects datetime/date/int/ISO str, got {type(value)!r}")
    delta = dt - _EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _micros_to_dt(micros):
    """Decode int64 epoch-microseconds back to a naive (UTC) datetime."""
    return _EPOCH + _timedelta(microseconds=int(micros))


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
        self._json_fields = set()   # arbitrary JSON values via BlobStore (json.dumps/loads)
        # Fixed-width UTF-8 string fields: field_name → byte budget N.
        # Stored inline as numpy S{N} (raw bytes), encoded/decoded transparently
        # — ~4× smaller than U{N} (UCS-4, 4 bytes/char) for ASCII, with no
        # BlobStore indirection.  N is a BYTE budget, not a char count.
        self._utf8_fields = {}
        # utf8 fields declared "utf8[N!]" raise on overflow instead of truncating
        self._utf8_strict = set()
        # Datetime fields: stored inline as int64 epoch-microseconds, encoded/
        # decoded to Python datetime transparently (see _dt_to_micros).
        self._datetime_fields = set()

        # Track array (vector) fields for proper serialization
        self._array_fields = {}   # field_name → shape tuple

        # Convert dtype strings, handling "blob", "text", "utf8[N]", "datetime", "float32[N]"
        processed_schema = []
        for field, dtype in schema.items():
            if dtype == "blob":
                self._blob_fields.add(field)
                processed_schema.append((field, BLOB_DTYPE))
            elif dtype == "text":
                self._text_fields.add(field)
                processed_schema.append((field, BLOB_DTYPE))
            elif dtype == "json":
                self._json_fields.add(field)
                processed_schema.append((field, BLOB_DTYPE))
            elif dtype == "datetime":
                self._datetime_fields.add(field)
                processed_schema.append((field, np.dtype("int64")))
            elif isinstance(dtype, str) and dtype.startswith("utf8[") and dtype.endswith("]"):
                inner = dtype[5:-1]
                strict = inner.endswith("!")
                n = int(inner[:-1] if strict else inner)
                self._utf8_fields[field] = n
                if strict:
                    self._utf8_strict.add(field)
                processed_schema.append((field, np.dtype(f"S{n}")))
            else:
                np_dtype = parse_dtype(dtype)
                if np_dtype.shape:
                    self._array_fields[field] = np_dtype.shape
                processed_schema.append((field, np_dtype))

        # Build schema with prefix
        self.schema = np.dtype([("_prefix", "int8")] + processed_schema)
        self.user_schema = np.dtype(processed_schema)  # Schema without prefix
        self.record_size = self.schema.itemsize

        # Prefix bytes for valid/deleted records
        self._valid_prefix = np.int8(identifier).tobytes()
        self._deleted_prefix = np.int8(-identifier).tobytes()

        # Pre-built zero record for fast bulk-init (one write instead of N)
        self._zero_record = self._serialize_zero()

        # Precompiled struct-pack serialize plan (blob-free scalar schemas)
        self._fast_plan = self._build_fast_plan()

    @staticmethod
    def _utf8_pack(value, n, strict=False, field=None):
        """Encode a str to at most n UTF-8 bytes.

        Truncates on a codepoint boundary (never splits a multi-byte char) by
        default; if ``strict``, raises ValueError when the value exceeds the
        byte budget instead of silently truncating."""
        if value is None:
            return b""
        enc = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        if len(enc) > n:
            if strict:
                raise ValueError(
                    f"utf8 field {field!r}: value is {len(enc)} bytes, exceeds the "
                    f"{n}-byte budget (declared truncate=False)"
                )
            # decode(errors="ignore") drops the trailing partial sequence
            enc = enc[:n].decode("utf-8", "ignore").encode("utf-8")
        return enc

    def _serialize_zero(self):
        """Return a zero/empty record as bytes (for bulk block init)."""
        values = [self.identifier]
        for field in self.user_schema.names:
            dtype = self.user_schema.fields[field][0]
            if field in self._text_fields or field in self._blob_fields or field in self._json_fields:
                values.append((0, 0))
            elif field in self._utf8_fields:
                values.append(b"")
            elif dtype.kind == "U":
                values.append("")
            else:
                values.append(0)
        return np.array(tuple(values), dtype=self.schema).tobytes()

    def allocate_block(self, n_records):
        """Allocate space for n records and return the address.

        Args:
            n_records: Number of records to allocate space for

        Returns:
            Address of the allocated block
        """
        size = n_records * self.record_size
        return self.db.allocate(size)

    _INT_FMT = {
        ("i", 1): "<b", ("i", 2): "<h", ("i", 4): "<i", ("i", 8): "<q",
        ("u", 1): "<B", ("u", 2): "<H", ("u", 4): "<I", ("u", 8): "<Q",
    }

    def _build_fast_plan(self):
        """Precompiled serialize plan: [(kind, field, offset, arg, strict)].

        For schemas made only of inline scalar fields (utf8/int/uint/float/
        bool/datetime — no blob/text/json/array/unicode), records can be
        packed straight into a bytearray with struct, skipping the generic
        per-field numpy path (~an order of magnitude cheaper for the small
        internal datasets: BTree values, hash-table entries, doc mappings).
        Returns None when the schema isn't eligible — generic path only.
        """
        if (
            self._blob_fields or self._text_fields or self._json_fields
            or self._array_fields or sys.byteorder != "little"
        ):
            return None
        plan = []
        for field in self.user_schema.names:
            dtype, offset = self.schema.fields[field]
            if field in self._utf8_fields:
                entry = ("u", field, offset, self._utf8_fields[field],
                         field in self._utf8_strict)
            elif field in self._datetime_fields:
                entry = ("d", field, offset, "<q", False)
            elif dtype.kind in "iu":
                fmt = self._INT_FMT.get((dtype.kind, dtype.itemsize))
                if fmt is None:
                    return None
                entry = ("n", field, offset, fmt, False)
            elif dtype.kind == "f" and dtype.itemsize in (4, 8):
                entry = ("n", field, offset,
                         "<f" if dtype.itemsize == 4 else "<d", False)
            elif dtype.kind == "b":
                entry = ("b", field, offset, None, False)
            else:  # unicode 'U', raw 'S' outside utf8[], datetime64 … → generic
                return None
            plan.append(entry)
        return plan

    def _serialize_fast(self, record):
        """struct-pack a record following the precompiled plan.

        Bytes are identical to the generic numpy path.  Any coercion the plan
        can't reproduce exactly (float into an int field, str into a numeric
        field, strict-utf8 overflow, None …) raises, and the caller falls back
        to the generic path — which then produces the numpy result or the
        proper error.  Raising here has no side effects (no blobs involved).
        """
        buf = bytearray(self._zero_record)
        pack_into = struct.pack_into
        for kind, field, off, arg, strict in self._fast_plan:
            if field not in record:
                continue  # zero record already encodes the default
            value = record[field]
            if kind == "u":
                if value is None:
                    continue  # generic packs b"" — same bytes as the zeros
                enc = (
                    value.encode("utf-8") if isinstance(value, str)
                    else bytes(value)
                )
                if len(enc) > arg:
                    if strict:
                        raise ValueError  # generic path raises the full message
                    enc = enc[:arg].decode("utf-8", "ignore").encode("utf-8")
                buf[off : off + len(enc)] = enc
            elif kind == "n":
                pack_into(arg, buf, off, value)
            elif kind == "d":
                pack_into("<q", buf, off, _dt_to_micros(value))
            else:  # bool — numpy rejects e.g. strings, don't pack truthiness
                if not isinstance(value, (bool, int, np.bool_, np.integer)):
                    raise TypeError
                buf[off] = 1 if value else 0
        return bytes(buf)

    def _serialize(self, **record):
        """Convert record dict to bytes with prefix.

        For blob fields, expects (offset, n_slots) tuple.
        For text fields, expects a str — stored transparently via BlobStore.
        """
        if self._fast_plan is not None:
            try:
                return self._serialize_fast(record)
            except Exception:
                pass  # generic path below reproduces the result or the error
        # Use zeros + field assignment to avoid numpy dtype __str__ overhead
        # that np.array(tuple, dtype=...) triggers on every call.
        arr = np.zeros(1, dtype=self.schema)
        arr["_prefix"] = self.identifier

        for field in self.user_schema.names:
            if field in record:
                value = record[field]
                if field in self._text_fields:
                    if value is None or value == "":
                        pass  # already zero = NULL_BLOB
                    else:
                        encoded = value.encode("utf-8") if isinstance(value, str) else value
                        offset, n_slots = self.blob_store.write(encoded)
                        arr[field]["offset"] = offset
                        arr[field]["n_slots"] = n_slots
                elif field in self._json_fields:
                    if value is None:
                        pass  # NULL_BLOB → None on read
                    else:
                        offset, n_slots = self.blob_store.write(
                            json.dumps(value, separators=(",", ":")).encode("utf-8")
                        )
                        arr[field]["offset"] = offset
                        arr[field]["n_slots"] = n_slots
                elif field in self._blob_fields:
                    if value is not None:
                        arr[field]["offset"] = value[0]
                        arr[field]["n_slots"] = value[1]
                elif field in self._utf8_fields:
                    arr[field] = self._utf8_pack(
                        value, self._utf8_fields[field],
                        field in self._utf8_strict, field,
                    )
                elif field in self._datetime_fields:
                    arr[field] = _dt_to_micros(value)
                elif field in self._array_fields:
                    # numpy array field — assign directly
                    arr[field] = np.asarray(value, dtype=self.user_schema.fields[field][0].base)
                else:
                    arr[field] = value
            # else: field absent — np.zeros already encodes the correct
            # default (empty string for U, 0 for numeric, NULL for blob/text),
            # so no assignment is needed.  Skipping it makes sparse records
            # (e.g. half-full BTree nodes) much cheaper to serialize.
        return arr.tobytes()

    def _deserialize(self, data):
        """Convert bytes to record dict (without prefix).

        For blob fields, returns (offset, n_slots) tuple.
        For text fields, returns the decoded UTF-8 string transparently.
        Actual blob data must be fetched separately via BlobStore.
        """
        arr = np.frombuffer(data, dtype=self.schema)[0]
        if not self._array_fields:
            # Fast path (no vector fields): convert the whole record to native
            # Python in one C call (arr.item()), then fix up only the special
            # scalar fields.  Much cheaper than a per-field isinstance + .item()
            # loop — and most records (incl. store_key Dicts, whose only special
            # field is the utf8 _key) land here.
            result = dict(zip(self.schema.names, arr.item()))
            del result["_prefix"]
            for field in self._utf8_fields:
                v = result[field]
                result[field] = v.rstrip(b"\x00").decode("utf-8") if isinstance(v, bytes) else str(v)
            for field in self._text_fields:
                off, ns = result[field]
                result[field] = "" if (off == 0 and ns == 0) else self.blob_store.read(int(off)).decode("utf-8")
            for field in self._json_fields:
                off, ns = result[field]
                result[field] = None if (off == 0 and ns == 0) else json.loads(self.blob_store.read(int(off)).decode("utf-8"))
            for field in self._blob_fields:
                off, ns = result[field]
                result[field] = None if (off == 0 and ns == 0) else (int(off), int(ns))
            for field in self._datetime_fields:
                result[field] = _micros_to_dt(result[field])
            return result

        # Vector path: keep arrays as numpy (never item() a whole vector).
        result = {}
        for field in self.user_schema.names:
            value = arr[field]
            if field in self._text_fields:
                offset, n_slots = int(value["offset"]), int(value["n_slots"])
                result[field] = "" if (offset == 0 and n_slots == 0) else self.blob_store.read(offset).decode("utf-8")
            elif field in self._json_fields:
                offset, n_slots = int(value["offset"]), int(value["n_slots"])
                result[field] = None if (offset == 0 and n_slots == 0) else json.loads(self.blob_store.read(offset).decode("utf-8"))
            elif field in self._blob_fields:
                offset, n_slots = int(value["offset"]), int(value["n_slots"])
                result[field] = None if (offset == 0 and n_slots == 0) else (offset, n_slots)
            elif field in self._utf8_fields:
                result[field] = bytes(value).rstrip(b"\x00").decode("utf-8")
            elif field in self._datetime_fields:
                result[field] = _micros_to_dt(int(value))
            elif field in self._array_fields:
                result[field] = np.array(value)
            else:
                result[field] = _to_native(value)
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
                    elif field in self._json_fields:
                        value = rec[field]
                        offset = int(value["offset"])
                        n_slots = int(value["n_slots"])
                        d[field] = None if (offset == 0 and n_slots == 0) else json.loads(self.blob_store.read(offset).decode("utf-8"))
                    elif field in self._blob_fields:
                        value = rec[field]
                        offset = int(value["offset"])
                        n_slots = int(value["n_slots"])
                        d[field] = None if (offset == 0 and n_slots == 0) else (offset, n_slots)
                    elif field in self._utf8_fields:
                        d[field] = bytes(rec[field]).rstrip(b"\x00").decode("utf-8")
                    elif field in self._datetime_fields:
                        d[field] = _micros_to_dt(int(rec[field]))
                    else:
                        d[field] = _to_native(rec[field])
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
        if self._text_fields or self._json_fields:
            data = self.db.read(address, self.record_size)
            arr = np.frombuffer(data, dtype=self.schema)[0]
            for field in (self._text_fields | self._json_fields):
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

        if field_name in self._json_fields:
            old_data = self.db.read(address + field_offset, BLOB_DTYPE.itemsize)
            old_ref = np.frombuffer(old_data, dtype=BLOB_DTYPE)[0]
            o, n = int(old_ref["offset"]), int(old_ref["n_slots"])
            if o != 0 or n != 0:
                self.blob_store.delete(o, n)
            if value is None:
                new_ref = np.array([_NULL_BLOB], dtype=BLOB_DTYPE)
            else:
                no, nn = self.blob_store.write(
                    json.dumps(value, separators=(",", ":")).encode("utf-8")
                )
                new_ref = np.array([(no, nn)], dtype=BLOB_DTYPE)
            self.db.write(address + field_offset, new_ref.tobytes())
            return

        if field_name in self._utf8_fields:
            field_dtype = self.user_schema.fields[field_name][0]
            packed = self._utf8_pack(
                value, self._utf8_fields[field_name],
                field_name in self._utf8_strict, field_name,
            )
            data = np.array([packed], dtype=field_dtype).tobytes()
            self.db.write(address + field_offset, data)
            return

        if field_name in self._datetime_fields:
            value = _dt_to_micros(value)

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
        value = np.frombuffer(data, dtype=field_dtype)[0]
        if field_name in self._utf8_fields:
            return bytes(value).rstrip(b"\x00").decode("utf-8")
        if field_name in self._text_fields:
            off, ns = int(value["offset"]), int(value["n_slots"])
            return "" if (off == 0 and ns == 0) else self.blob_store.read(off).decode("utf-8")
        if field_name in self._json_fields:
            off, ns = int(value["offset"]), int(value["n_slots"])
            return None if (off == 0 and ns == 0) else json.loads(self.blob_store.read(off).decode("utf-8"))
        if field_name in self._blob_fields:
            off, ns = int(value["offset"]), int(value["n_slots"])
            return None if (off == 0 and ns == 0) else self.blob_store.read(off)
        if field_name in self._datetime_fields:
            return _micros_to_dt(int(value))
        return _to_native(value)

    def read_fields(self, address, fields):
        """Read a subset of a record's fields in one row read.

        One db.read of the fixed-size row, then only the requested fields
        are materialized: unrequested text/json/blob fields never touch the
        blob store and unrequested scalars are never converted — that's
        where the cost of a full read() lives.

        Args:
            address: Record address
            fields: Iterable of field names to materialize

        Returns:
            Dict with just those fields
        """
        for field in fields:
            if field not in self.user_schema.names:
                raise ValueError(f"Field '{field}' not in schema")

        data = self.db.read(address, self.record_size)
        prefix = data[0:1]
        if prefix == self._deleted_prefix:
            raise DeletedRecordError(address)
        if prefix != self._valid_prefix:
            prefix_val = int(np.frombuffer(prefix, dtype="int8")[0])
            raise WrongDatasetError(address, self.identifier, prefix_val)

        arr = np.frombuffer(data, dtype=self.schema)[0]
        result = {}
        for field in fields:
            value = arr[field]
            if field in self._text_fields:
                off, ns = int(value["offset"]), int(value["n_slots"])
                result[field] = "" if (off == 0 and ns == 0) else self.blob_store.read(off).decode("utf-8")
            elif field in self._json_fields:
                off, ns = int(value["offset"]), int(value["n_slots"])
                result[field] = None if (off == 0 and ns == 0) else json.loads(self.blob_store.read(off).decode("utf-8"))
            elif field in self._blob_fields:
                off, ns = int(value["offset"]), int(value["n_slots"])
                result[field] = None if (off == 0 and ns == 0) else self.blob_store.read(off)
            elif field in self._utf8_fields:
                result[field] = bytes(value).rstrip(b"\x00").decode("utf-8")
            elif field in self._datetime_fields:
                result[field] = _micros_to_dt(int(value))
            elif field in self._array_fields:
                result[field] = np.array(value)
            else:
                result[field] = _to_native(value)
        return result

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
        """Read a record (``dataset[addr]``) or a single field
        (``dataset[addr, field]`` → just that field's value).
        """
        if isinstance(address, tuple) and len(address) == 2:
            addr, field = address
            if isinstance(addr, Ref):
                addr = addr.addr
            return self.read_field(addr, field)
        if isinstance(address, Ref):
            address = address.addr
        return self.read(address)

    def __setitem__(self, address, record):
        """Write a record (``dataset[addr] = {...}``) or a single field in place
        (``dataset[addr, field] = value``).
        """
        if isinstance(address, tuple) and len(address) == 2:
            addr, field = address
            if isinstance(addr, Ref):
                addr = addr.addr
            self.write_field(addr, field, record)
            return
        if isinstance(address, Ref):
            address = address.addr
        record = as_record(record)
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict or Pydantic model, got {type(record).__name__}")
        self.write(address, **record)

    def insert(self, record):
        record = as_record(record)
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict or Pydantic model, got {type(record).__name__}")
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
