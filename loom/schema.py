"""
Pydantic model → loom schema conversion.

Maps Python/Pydantic types to numpy dtypes used by Dataset:
    int                         → int64
    float                       → float64
    bool                        → bool
    str                         → text   (variable-length via BlobStore)
    str with max_length         → U{N}   (fixed-length numpy unicode)

Three equivalent ways to declare a fixed-length string field:

    # 1. Field(max_length=N)  — most concise, standard Pydantic v2
    from pydantic import Field
    class M(BaseModel):
        name: str = Field(max_length=20)

    # 2. Annotated + StringConstraints
    from pydantic import StringConstraints
    from typing import Annotated
    class M(BaseModel):
        name: Annotated[str, StringConstraints(max_length=20)]

    # 3. FixedStr(N) helper — shortest
    from loom.schema import FixedStr
    class M(BaseModel):
        name: FixedStr(20)

All three produce U20 (80 bytes per record, fast lookups).
Use plain `str` for variable-length content (→ text, compressed BlobStore).

Usage:
    from pydantic import BaseModel
    from loom.schema import schema_from_model

    class User(BaseModel):
        id: int
        name: str
        score: float

    schema = schema_from_model(User)
    # {'id': 'int64', 'name': 'text', 'score': 'float64'}
"""

from __future__ import annotations

from typing import Any, get_args, get_origin, Union

# numpy dtype strings for common Python types
_TYPE_MAP: dict[type, str] = {
    int: "int64",
    float: "float64",
    bool: "bool",
}


def _resolve_str_field(field_info) -> str:
    """Determine whether a str field is fixed-length (U{N}) or variable (text).

    Checks Pydantic v2 field metadata for max_length constraint.
    """
    # Pydantic v2: metadata is a list of constraint objects
    for meta in getattr(field_info, "metadata", []):
        max_len = getattr(meta, "max_length", None)
        if max_len is not None:
            return f"U{max_len}"
    return "text"


def _resolve_type(annotation: Any, field_info: Any = None) -> str:
    """Map a Python type annotation to a loom dtype string."""
    # Unwrap Optional[X] → X
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]

    if annotation is str:
        return _resolve_str_field(field_info) if field_info else "text"

    dtype = _TYPE_MAP.get(annotation)
    if dtype:
        return dtype

    raise TypeError(
        f"Cannot map type {annotation!r} to a loom dtype. "
        f"Supported: int, float, bool, str (+ Optional variants)."
    )


def schema_from_model(model_class) -> dict[str, str]:
    """Convert a Pydantic BaseModel class to a loom schema dict.

    Args:
        model_class: A Pydantic BaseModel subclass

    Returns:
        Dict mapping field names to loom dtype strings

    Example:
        class User(BaseModel):
            id: int
            name: str
            score: float

        schema_from_model(User)
        # {'id': 'int64', 'name': 'text', 'score': 'float64'}
    """
    schema: dict[str, str] = {}
    for name, field_info in model_class.model_fields.items():
        schema[name] = _resolve_type(field_info.annotation, field_info)
    return schema


def FixedStr(max_length: int):
    """Shorthand for a fixed-length string field (→ U{max_length}).

    Use inside Pydantic models for fields that need fast lookups or
    are used as keys, where a fixed max length is known upfront.
    Plain `str` gives variable-length text instead.

    Example:
        from loom.schema import FixedStr

        class User(BaseModel):
            username: FixedStr(50)   # → U50, fast dict key
            email:    FixedStr(100)  # → U100
            bio:      str            # → text, variable-length
    """
    from pydantic import StringConstraints
    from typing import Annotated
    return Annotated[str, StringConstraints(max_length=max_length)]


# ── Datetime ↔ BTree key helpers ─────────────────────────────────────────────

_DT_FORMATS = {
    "microsecond": "%Y%m%dT%H%M%S%f",   # 20240115T143000123456  (22 chars)
    "second":      "%Y%m%dT%H%M%S",     # 20240115T143000        (15 chars)
    "minute":      "%Y%m%dT%H%M",       # 20240115T1430          (12 chars)
    "hour":        "%Y%m%dT%H",         # 20240115T14            (10 chars)
    "day":         "%Y%m%d",            # 20240115               ( 8 chars)
    "month":       "%Y%m",              # 202401                 ( 6 chars)
    "year":        "%Y",                # 2024                   ( 4 chars)
}

# Key sizes: how many chars the string takes (for key_size= on BTree)
_DT_KEY_SIZES = {k: len(v.replace("%Y", "2024").replace("%m", "01")
                           .replace("%d", "01").replace("%H", "00")
                           .replace("%M", "00").replace("%S", "00")
                           .replace("%f", "000000"))
                 for k, v in _DT_FORMATS.items()}


def dt_key(dt, precision: str = "second") -> str:
    """Convert a datetime to a BTree-compatible sort key.

    The returned string sorts lexicographically in chronological order,
    so it can be used directly as a BTree key for efficient range queries.

    Args:
        dt:        datetime.datetime (or date) to convert
        precision: granularity of the key — one of:
                   "microsecond"  → "20240115T143000123456"  (22 chars)
                   "second"       → "20240115T143000"         (15 chars, default)
                   "minute"       → "20240115T1430"           (12 chars)
                   "hour"         → "20240115T14"             (10 chars)
                   "day"          → "20240115"                ( 8 chars)
                   "month"        → "202401"                  ( 6 chars)
                   "year"         → "2024"                    ( 4 chars)

    Returns:
        str — lexicographically sortable datetime key

    Example:
        from loom.schema import dt_key, key_dt
        from datetime import datetime

        ts = datetime(2024, 1, 15, 14, 30, 0)
        k = dt_key(ts)                    # "20240115T143000"
        k = dt_key(ts, "minute")          # "20240115T1430"

        # BTree range: all ticks on Jan 15
        for ts_str, row in btree.range(dt_key(datetime(2024,1,15), "day"),
                                       dt_key(datetime(2024,1,15,23,59,59))):
            ...
    """
    fmt = _DT_FORMATS.get(precision)
    if fmt is None:
        raise ValueError(
            f"Unknown precision {precision!r}. "
            f"Choose from: {list(_DT_FORMATS)}"
        )
    return dt.strftime(fmt)


def key_dt(key: str, precision: str = "second"):
    """Convert a BTree key back to a datetime.datetime.

    Args:
        key:       key previously produced by dt_key()
        precision: must match the precision used in dt_key()

    Returns:
        datetime.datetime

    Example:
        from loom.schema import dt_key, key_dt
        k = "20240115T143000"
        dt = key_dt(k)   # datetime(2024, 1, 15, 14, 30, 0)
    """
    from datetime import datetime
    fmt = _DT_FORMATS.get(precision)
    if fmt is None:
        raise ValueError(f"Unknown precision {precision!r}.")
    return datetime.strptime(key, fmt)


def dt_key_size(precision: str = "second") -> int:
    """Return the key_size value to pass to BTree for a given precision.

    Example:
        btree = db.create_btree("ticks", ohlcv_ds,
                                key_size=dt_key_size("second"))
    """
    size = _DT_KEY_SIZES.get(precision)
    if size is None:
        raise ValueError(f"Unknown precision {precision!r}.")
    return size
