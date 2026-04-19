"""Loom - A persistent database that feels like Python."""

from loom.database import DB
from loom.dataset import Dataset
from loom.datastructures import List, BloomFilter, CountingBloomFilter
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
)

__version__ = "0.1.0"
__all__ = [
    "DB", "Dataset", "List", "BloomFilter", "CountingBloomFilter",
    # Exceptions
    "LoomError",
    "DatabaseError", "DatabaseNotOpenError", "DuplicateNameError", "StructureNotFoundError",
    "HeaderError", "HeaderTooLargeError",
    "SchemaError", "InvalidIdentifierError", "UnknownDtypeError",
    "RecordError", "DeletedRecordError", "WrongDatasetError",
]