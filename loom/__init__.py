"""Loom - A persistent database that feels like Python."""

from loom.database import DB
from loom.collection import Collection, Primary, Unique, Range, Many, Search, Vector
from loom.datastructures.search import SearchIndex
from loom.datastructures.priority_queue import PriorityQueue, PriorityQueueEmpty
from loom.client import (
    LoomClient,
    LoomClientError,
    LoomHTTPError,
    LoomNotFoundError,
    LoomValidationError,
    LoomConflictError,
)
from loom.dataset import Dataset
from loom.datastructures import List, BloomFilter, CountingBloomFilter
from loom.schema import dt_key, key_dt, dt_key_size, FixedStr, Utf8, Datetime, Json, Vec, schema_from_model
from loom.errors import (
    LoomError,
    DatabaseError,
    DatabaseNotOpenError,
    DuplicateNameError,
    StructureNotFoundError,
    HeaderError,
    HeaderTooLargeError,
    SchemaError,
    InvalidIdentifierError,
    UnknownDtypeError,
    RecordError,
    DeletedRecordError,
    WrongDatasetError,
    NestingNotSupportedError,
)

__version__ = "0.1.0"
__all__ = [
    "DB",
    "Collection",
    "SearchIndex",
    "PriorityQueue",
    "PriorityQueueEmpty",
    "Primary",
    "Unique",
    "Range",
    "Many",
    "Search",
    "Vector",
    "Dataset",
    "List",
    "BloomFilter",
    "CountingBloomFilter",
    "LoomClient",
    "LoomClientError",
    "LoomHTTPError",
    "LoomNotFoundError",
    "LoomValidationError",
    "LoomConflictError",
    # Exceptions
    "LoomError",
    "DatabaseError",
    "DatabaseNotOpenError",
    "DuplicateNameError",
    "StructureNotFoundError",
    "HeaderError",
    "HeaderTooLargeError",
    "SchemaError",
    "InvalidIdentifierError",
    "UnknownDtypeError",
    "RecordError",
    "DeletedRecordError",
    "WrongDatasetError",
    "NestingNotSupportedError",
    # Datetime helpers
    "dt_key",
    "key_dt",
    "dt_key_size",
    "FixedStr",
    "Utf8",
    "Datetime",
    "Json",
    "Vec",
    "schema_from_model",
]
