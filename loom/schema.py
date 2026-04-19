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
