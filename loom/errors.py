"""
Custom exceptions for loom.

Hierarchy:
    LoomError (base)
    ├── DatabaseError
    │   ├── DatabaseNotOpenError   — operation on closed DB
    │   ├── DuplicateNameError     — name already registered
    │   └── StructureNotFoundError — unknown dataset / structure name
    ├── HeaderError
    │   └── HeaderTooLargeError    — pickle exceeds slot size
    ├── SchemaError
    │   ├── InvalidIdentifierError — Dataset identifier outside 1..127
    │   └── UnknownDtypeError      — unrecognised field type
    └── RecordError
        ├── DeletedRecordError     — accessing a soft-deleted record
        └── WrongDatasetError      — prefix mismatch (wrong schema)
"""


class LoomError(Exception):
    """Base class for all loom exceptions."""


# ── Database-level ────────────────────────────────────────────


class DatabaseError(LoomError):
    """Errors related to the DB lifecycle or registry."""


class DatabaseNotOpenError(DatabaseError):
    """Operation attempted on a closed database.

    Example::

        db.close()
        db["users"]   # raises DatabaseNotOpenError
    """

    def __init__(self, msg="Database is not open. Call open() first."):
        super().__init__(msg)


class DuplicateNameError(DatabaseError):
    """A dataset or data structure with this name already exists.

    Example::

        db.create_dataset("users", id="uint32")
        db.create_dataset("users", id="uint32")   # raises DuplicateNameError
    """

    def __init__(self, name: str, kind: str = "Dataset"):
        super().__init__(f"{kind} '{name}' already exists")
        self.name = name
        self.kind = kind


class StructureNotFoundError(DatabaseError, KeyError):
    """No dataset or data structure with the requested name.

    Inherits from KeyError so existing ``except KeyError`` handlers
    keep working while new code can be more specific.

    Example::

        db["typo_name"]   # raises StructureNotFoundError
    """

    def __init__(self, name: str):
        msg = f"'{name}' not found in datasets or data structures"
        super().__init__(msg)
        KeyError.__init__(self, msg)
        self.name = name


# ── Header ────────────────────────────────────────────────────


class HeaderError(LoomError):
    """Errors related to the file header."""


class HeaderTooLargeError(HeaderError):
    """Pickled header data exceeds the allocated slot size.

    Increase ``header_size`` when opening the DB, or reduce the number
    of datasets / structures stored.
    """

    def __init__(self, actual: int, max_bytes: int):
        super().__init__(
            f"Header data too large: {actual} bytes (slot max {max_bytes}). "
            f"Increase DB(header_size=...) — current value must at least double."
        )
        self.actual = actual
        self.max_bytes = max_bytes


# ── Schema ────────────────────────────────────────────────────


class SchemaError(LoomError):
    """Errors related to field schemas or identifiers."""


class InvalidIdentifierError(SchemaError):
    """Dataset identifier is outside the valid range 1..127."""

    def __init__(self, identifier: int):
        super().__init__(
            f"Dataset identifier must be in 1..127, got {identifier}"
        )
        self.identifier = identifier


class UnknownDtypeError(SchemaError):
    """A field type cannot be mapped to a loom / numpy dtype.

    Supported Python/Pydantic types: int, float, bool, str.
    Supported loom strings: 'text', 'blob', numpy dtype strings (e.g. 'uint32').
    """

    def __init__(self, field: str, dtype):
        super().__init__(
            f"Cannot map field '{field}' type {dtype!r} to a loom dtype. "
            f"Supported: int, float, bool, str (+ Annotated with max_length)."
        )
        self.field = field
        self.dtype = dtype


# ── Record ────────────────────────────────────────────────────


class RecordError(LoomError):
    """Errors when reading or writing individual records."""


class DeletedRecordError(RecordError):
    """The record at this address was soft-deleted."""

    def __init__(self, address: int):
        super().__init__(f"Record at address {address} is deleted")
        self.address = address


class WrongDatasetError(RecordError):
    """The prefix byte does not match this dataset's identifier.

    Usually indicates reading with the wrong Dataset object, or
    file corruption.
    """

    def __init__(self, address: int, expected: int, got: int):
        super().__init__(
            f"Wrong dataset at address {address}: "
            f"expected identifier {expected}, got {got}"
        )
        self.address = address
        self.expected = expected
        self.got = got
