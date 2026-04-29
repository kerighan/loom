"""HTTP server for loom databases.

Exposes every dataset and datastructure registered on a `DB` as a clean
REST API, with auto-generated OpenAPI docs and Pydantic-based request
validation.

Concurrency model
-----------------
loom is currently single-writer / single-reader.  All HTTP handlers
acquire a single, process-wide lock so requests are fully serialized
against the underlying mmap.  This keeps the server safe to use from
any number of HTTP clients without changing the database internals.

Usage
-----
    from loom import DB

    with DB("app.db") as db:
        db.create_dataset("users", User)
        ...
        db.serve(host="127.0.0.1", port=8000)

Or, to mount the API in your own ASGI stack:

    app = db.fastapi_app()           # returns a FastAPI instance
    # uvicorn.run(app, ...)
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Schema → Pydantic helpers
# ---------------------------------------------------------------------------

_INT_DTYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}
_FLOAT_DTYPES = {"float16", "float32", "float64"}
_ARRAY_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*)\[(\d+(?:,\d+)*)\]$")


def _py_type_for_dtype(dtype_str: str):
    """Map a loom dtype string to (python_type, pydantic Field or None)."""
    from pydantic import Field

    s = str(dtype_str)

    if _ARRAY_RE.match(s):
        return list[float], None
    if s in _INT_DTYPES:
        return int, None
    if s in _FLOAT_DTYPES:
        return float, None
    if s == "bool":
        return bool, None
    # Numpy unicode dtype: "U50", "<U50", ">U50"
    m = re.match(r"^[<>|]?U(\d+)$", s)
    if m:
        return str, Field(..., max_length=int(m.group(1)))
    if s in ("text", "blob"):
        return str, None
    return Any, None


def _make_model(name: str, schema: dict[str, str]):
    """Build a Pydantic model class from a loom schema dict."""
    from pydantic import create_model

    fields: dict[str, Any] = {}
    for fname, dt in schema.items():
        py_type, field_info = _py_type_for_dtype(dt)
        fields[fname] = (py_type, field_info if field_info is not None else ...)
    return create_model(name, **fields)


# ---------------------------------------------------------------------------
# Schema introspection — works on both Datasets and datastructures
# ---------------------------------------------------------------------------


def _dataset_schema(ds) -> dict[str, str]:
    """Extract a {field: dtype_str} schema dict from a Dataset."""
    from loom.dataset import dtype_to_str

    out = {}
    for name in ds.user_schema.names:
        if name in getattr(ds, "_text_fields", set()):
            out[name] = "text"
        elif name in getattr(ds, "_blob_fields", set()):
            out[name] = "blob"
        else:
            raw = ds.user_schema.fields[name][0]
            out[name] = raw if isinstance(raw, str) else dtype_to_str(raw)
    return out


def _structure_schema(obj) -> Optional[dict[str, str]]:
    """Return value-schema for a datastructure, or None if nested/unknown."""
    if getattr(obj, "_is_nested", False):
        return None

    # Prefer the underlying user dataset
    for attr in ("_user_dataset", "_values_dataset", "_items_dataset"):
        ds = getattr(obj, attr, None)
        if ds is not None and hasattr(ds, "user_schema"):
            return _dataset_schema(ds)

    sch = getattr(obj, "item_schema", None)
    if isinstance(sch, dict):
        return sch
    return None


# ---------------------------------------------------------------------------
# JSON conversion (numpy → native python)
# ---------------------------------------------------------------------------


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items() if k != "valid"}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def build_app(db, *, title: Optional[str] = None):
    """Build a FastAPI app exposing every dataset/structure in `db`.

    The app is built once and resolves names lazily on every request, so
    structures created after `build_app()` are still reachable.

    Args:
        db: a `loom.DB` instance (must be open).
        title: OpenAPI title (default: "loom: <filename>").

    Returns:
        A FastAPI application.
    """
    try:
        from fastapi import FastAPI, HTTPException, Body, Path, Query
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is required for `loom.server`. "
            "Install with: pip install 'fastapi[standard]'"
        ) from e

    from loom.dataset import Dataset
    from loom.datastructures import (
        BloomFilter,
        CountingBloomFilter,
        List,
        Set,
    )
    from loom.datastructures.dict import Dict as LoomDict
    from loom.datastructures.btree import BTree
    from loom.datastructures.queue import Queue
    from loom.datastructures.lru_dict import LRUDict

    app = FastAPI(
        title=title or f"loom: {db.filename}",
        description=(
            "Auto-generated REST API for a loom database.\n\n"
            "**Interactive docs**\n"
            "* Swagger UI — [`/docs`](/docs) (try requests live)\n"
            "* ReDoc — [`/redoc`](/redoc)\n"
            "* OpenAPI 3 schema — [`/openapi.json`](/openapi.json)\n\n"
            "Single-writer / single-reader: every request is serialized "
            "through a process-wide lock."
        ),
        version="1.0.0",
    )
    lock = threading.Lock()
    _model_cache: dict[tuple[str, str], Any] = {}

    # ------------------------------------------------------------------ utils
    def _model_for(prefix: str, name: str, schema: dict[str, str]):
        key = (prefix, name)
        m = _model_cache.get(key)
        if m is None:
            m = _make_model(f"{prefix}_{name}", schema)
            _model_cache[key] = m
        return m

    def _validate(
        prefix: str, name: str, payload: dict, schema: dict[str, str]
    ) -> dict:
        Model = _model_for(prefix, name, schema)
        try:
            return Model(**payload).model_dump()
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    def _get_dataset(name: str) -> Dataset:
        ds = db._datasets.get(name)
        if ds is None:
            raise HTTPException(404, f"Dataset {name!r} not found")
        return ds

    def _get_structure(name: str, expected_type):
        obj = db._datastructures.get(name)
        if obj is None or not isinstance(obj, expected_type):
            raise HTTPException(
                404,
                f"{expected_type.__name__} {name!r} not found",
            )
        if getattr(obj, "_is_nested", False):
            raise HTTPException(
                422,
                f"{name!r} is a nested {expected_type.__name__}; "
                "nested structures are not supported via the HTTP API yet.",
            )
        return obj

    # ------------------------------------------------------------------ root
    @app.get(
        "/",
        tags=["meta"],
        summary="List datasets, structures, and links to interactive docs",
    )
    def index():
        with lock:
            return {
                "filename": db.filename,
                "datasets": list(db._datasets),
                "structures": {
                    n: type(s).__name__ for n, s in db._datastructures.items()
                },
                "docs": {
                    "swagger_ui": "/docs",
                    "redoc": "/redoc",
                    "openapi_schema": "/openapi.json",
                },
            }

    # ============================================================== datasets
    @app.get("/datasets/{name}", tags=["datasets"], summary="Dataset schema")
    def dataset_info(name: str):
        with lock:
            ds = _get_dataset(name)
            return {
                "name": ds.name,
                "identifier": ds.identifier,
                "schema": _dataset_schema(ds),
                "record_size": ds.record_size,
            }

    @app.post(
        "/datasets/{name}/records",
        tags=["datasets"],
        summary="Insert a record (allocates a new address)",
        status_code=201,
    )
    def dataset_insert(name: str, record: dict = Body(...)):
        with lock:
            ds = _get_dataset(name)
            data = _validate("ds", name, record, _dataset_schema(ds))
            ref = ds.insert(data)
            return {"address": int(ref.addr)}

    @app.get(
        "/datasets/{name}/records/{address}",
        tags=["datasets"],
        summary="Read a record",
    )
    def dataset_read(name: str, address: int):
        with lock:
            ds = _get_dataset(name)
            try:
                return _to_jsonable(ds.read(address))
            except Exception as e:
                raise HTTPException(404, str(e))

    @app.put(
        "/datasets/{name}/records/{address}",
        tags=["datasets"],
        summary="Overwrite a record",
    )
    def dataset_write(name: str, address: int, record: dict = Body(...)):
        with lock:
            ds = _get_dataset(name)
            data = _validate("ds", name, record, _dataset_schema(ds))
            ds.write(address, **data)
            return {"address": int(address)}

    @app.delete(
        "/datasets/{name}/records/{address}",
        tags=["datasets"],
        summary="Soft-delete a record",
        status_code=204,
    )
    def dataset_delete(name: str, address: int):
        with lock:
            ds = _get_dataset(name)
            ds.delete(address)
            return None

    # ================================================================== list
    @app.get("/lists/{name}", tags=["lists"], summary="List metadata")
    def list_info(name: str):
        with lock:
            lst = _get_structure(name, List)
            return {
                "name": lst.name,
                "length": len(lst),
                "schema": _structure_schema(lst),
            }

    @app.get(
        "/lists/{name}/items",
        tags=["lists"],
        summary="Slice items: ?start=&end=",
    )
    def list_slice(
        name: str,
        start: int = Query(0),
        end: Optional[int] = Query(None),
    ):
        with lock:
            lst = _get_structure(name, List)
            if end is None:
                end = len(lst)
            return [_to_jsonable(it) for it in lst[start:end]]

    @app.get("/lists/{name}/items/{index}", tags=["lists"], summary="Read item")
    def list_get(name: str, index: int):
        with lock:
            lst = _get_structure(name, List)
            try:
                return _to_jsonable(lst[index])
            except IndexError:
                raise HTTPException(404, f"index {index} out of range")

    @app.post(
        "/lists/{name}/items",
        tags=["lists"],
        summary="Append an item",
        status_code=201,
    )
    def list_append(name: str, item: dict = Body(...)):
        with lock:
            lst = _get_structure(name, List)
            schema = _structure_schema(lst) or {}
            data = _validate("list", name, item, schema)
            lst.append(data)
            return {"index": len(lst) - 1}

    @app.put("/lists/{name}/items/{index}", tags=["lists"], summary="Replace item")
    def list_set(name: str, index: int, item: dict = Body(...)):
        with lock:
            lst = _get_structure(name, List)
            schema = _structure_schema(lst) or {}
            data = _validate("list", name, item, schema)
            try:
                lst[index] = data
            except IndexError:
                raise HTTPException(404, f"index {index} out of range")
            return {"index": index}

    @app.delete(
        "/lists/{name}/items/{index}",
        tags=["lists"],
        summary="Soft-delete item",
        status_code=204,
    )
    def list_delete(name: str, index: int):
        with lock:
            lst = _get_structure(name, List)
            try:
                del lst[index]
            except IndexError:
                raise HTTPException(404, f"index {index} out of range")
            return None

    # ================================================================== dict
    @app.get("/dicts/{name}", tags=["dicts"], summary="Dict metadata")
    def dict_info(name: str):
        with lock:
            d = _get_structure(name, LoomDict)
            return {
                "name": d.name,
                "length": len(d),
                "schema": _structure_schema(d),
            }

    @app.get("/dicts/{name}/keys", tags=["dicts"], summary="List keys")
    def dict_keys(name: str, limit: int = Query(1000, ge=1, le=100_000)):
        with lock:
            d = _get_structure(name, LoomDict)
            out = []
            for i, k in enumerate(d.keys()):
                if i >= limit:
                    break
                out.append(k)
            return out

    @app.get(
        "/dicts/{name}/items/{key}",
        tags=["dicts"],
        summary="Read value at key",
    )
    def dict_get(name: str, key: str):
        with lock:
            d = _get_structure(name, LoomDict)
            try:
                return _to_jsonable(d[key])
            except KeyError:
                raise HTTPException(404, f"key {key!r} not found")

    @app.put(
        "/dicts/{name}/items/{key}",
        tags=["dicts"],
        summary="Set value at key",
    )
    def dict_set(name: str, key: str, value: dict = Body(...)):
        with lock:
            d = _get_structure(name, LoomDict)
            schema = _structure_schema(d) or {}
            data = _validate("dict", name, value, schema)
            d[key] = data
            return {"key": key}

    @app.delete(
        "/dicts/{name}/items/{key}",
        tags=["dicts"],
        summary="Delete key",
        status_code=204,
    )
    def dict_delete(name: str, key: str):
        with lock:
            d = _get_structure(name, LoomDict)
            try:
                del d[key]
            except KeyError:
                raise HTTPException(404, f"key {key!r} not found")
            return None

    # =================================================================== set
    @app.get("/sets/{name}", tags=["sets"], summary="Set metadata")
    def set_info(name: str):
        with lock:
            s = _get_structure(name, Set)
            return {"name": s.name, "length": len(s)}

    @app.get("/sets/{name}/members", tags=["sets"], summary="List members")
    def set_members(name: str, limit: int = Query(1000, ge=1, le=100_000)):
        with lock:
            s = _get_structure(name, Set)
            out = []
            for i, item in enumerate(s):
                if i >= limit:
                    break
                out.append(item)
            return out

    @app.post(
        "/sets/{name}/members",
        tags=["sets"],
        summary="Add a member",
        status_code=201,
    )
    def set_add(name: str, body: dict = Body(..., examples=[{"item": "alice"}])):
        with lock:
            s = _get_structure(name, Set)
            item = body.get("item")
            if not isinstance(item, str):
                raise HTTPException(422, "body must be {'item': <str>}")
            s.add(item)
            return {"item": item}

    @app.get(
        "/sets/{name}/contains/{item}",
        tags=["sets"],
        summary="Membership test",
    )
    def set_contains(name: str, item: str):
        with lock:
            s = _get_structure(name, Set)
            return {"item": item, "contains": item in s}

    @app.delete(
        "/sets/{name}/members/{item}",
        tags=["sets"],
        summary="Remove member",
        status_code=204,
    )
    def set_remove(name: str, item: str):
        with lock:
            s = _get_structure(name, Set)
            s.discard(item)
            return None

    # ================================================================= btree
    @app.get("/btrees/{name}", tags=["btrees"], summary="BTree metadata")
    def btree_info(name: str):
        with lock:
            t = _get_structure(name, BTree)
            return {
                "name": t.name,
                "length": len(t),
                "schema": _structure_schema(t),
            }

    @app.get("/btrees/{name}/items/{key}", tags=["btrees"], summary="Read by key")
    def btree_get(name: str, key: str):
        with lock:
            t = _get_structure(name, BTree)
            try:
                return _to_jsonable(t[key])
            except KeyError:
                raise HTTPException(404, f"key {key!r} not found")

    @app.put("/btrees/{name}/items/{key}", tags=["btrees"], summary="Set by key")
    def btree_set(name: str, key: str, value: dict = Body(...)):
        with lock:
            t = _get_structure(name, BTree)
            schema = _structure_schema(t) or {}
            data = _validate("btree", name, value, schema)
            t[key] = data
            return {"key": key}

    @app.delete(
        "/btrees/{name}/items/{key}",
        tags=["btrees"],
        summary="Delete by key",
        status_code=204,
    )
    def btree_delete(name: str, key: str):
        with lock:
            t = _get_structure(name, BTree)
            try:
                del t[key]
            except KeyError:
                raise HTTPException(404, f"key {key!r} not found")
            return None

    @app.get(
        "/btrees/{name}/range",
        tags=["btrees"],
        summary="Range query: ?start=&end=",
    )
    def btree_range(
        name: str,
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        limit: int = Query(1000, ge=1, le=100_000),
    ):
        with lock:
            t = _get_structure(name, BTree)
            out = []
            for i, (k, v) in enumerate(t.range(start, end)):
                if i >= limit:
                    break
                out.append({"key": k, "value": _to_jsonable(v)})
            return out

    @app.get(
        "/btrees/{name}/prefix/{prefix}",
        tags=["btrees"],
        summary="Prefix query",
    )
    def btree_prefix(
        name: str,
        prefix: str,
        limit: int = Query(1000, ge=1, le=100_000),
    ):
        with lock:
            t = _get_structure(name, BTree)
            out = []
            for i, (k, v) in enumerate(t.prefix(prefix)):
                if i >= limit:
                    break
                out.append({"key": k, "value": _to_jsonable(v)})
            return out

    # ================================================================ queues
    @app.get("/queues/{name}", tags=["queues"], summary="Queue metadata")
    def queue_info(name: str):
        with lock:
            q = _get_structure(name, Queue)
            return {
                "name": q.name,
                "length": len(q),
                "schema": _structure_schema(q),
            }

    @app.get("/queues/{name}/peek", tags=["queues"], summary="Peek front")
    def queue_peek(name: str):
        with lock:
            q = _get_structure(name, Queue)
            if len(q) == 0:
                raise HTTPException(404, "queue empty")
            return _to_jsonable(q.peek())

    @app.post(
        "/queues/{name}/push",
        tags=["queues"],
        summary="Enqueue an item",
        status_code=201,
    )
    def queue_push(name: str, item: dict = Body(...)):
        with lock:
            q = _get_structure(name, Queue)
            schema = _structure_schema(q) or {}
            data = _validate("queue", name, item, schema)
            q.push(data)
            return {"length": len(q)}

    @app.post(
        "/queues/{name}/pop",
        tags=["queues"],
        summary="Dequeue an item",
    )
    def queue_pop(name: str):
        with lock:
            q = _get_structure(name, Queue)
            if len(q) == 0:
                raise HTTPException(404, "queue empty")
            return _to_jsonable(q.pop())

    # =========================================================== bloom & cbf
    def _bloom_routes(prefix: str, expected_cls):
        @app.post(
            f"/{prefix}/{{name}}/items",
            tags=[prefix],
            summary="Add an item",
            status_code=201,
            name=f"{prefix}_add",
        )
        def _add(name: str, body: dict = Body(..., examples=[{"item": "alice"}])):
            with lock:
                bf = _get_structure(name, expected_cls)
                item = body.get("item")
                if item is None:
                    raise HTTPException(422, "body must be {'item': <value>}")
                bf.add(item)
                return {"item": item}

        @app.get(
            f"/{prefix}/{{name}}/contains/{{item}}",
            tags=[prefix],
            summary="Probabilistic membership test",
            name=f"{prefix}_contains",
        )
        def _contains(name: str, item: str):
            with lock:
                bf = _get_structure(name, expected_cls)
                return {"item": item, "contains": item in bf}

        if expected_cls is CountingBloomFilter:

            @app.delete(
                f"/{prefix}/{{name}}/items/{{item}}",
                tags=[prefix],
                summary="Remove an item (counting bloom only)",
                status_code=204,
                name=f"{prefix}_remove",
            )
            def _remove(name: str, item: str):
                with lock:
                    bf = _get_structure(name, expected_cls)
                    bf.remove(item)
                    return None

    _bloom_routes("bloomfilters", BloomFilter)
    _bloom_routes("counting_bloomfilters", CountingBloomFilter)

    # ============================================================= lru_dicts
    @app.get("/lru_dicts/{name}", tags=["lru_dicts"], summary="LRUDict metadata")
    def lru_info(name: str):
        with lock:
            d = _get_structure(name, LRUDict)
            return {
                "name": d.name,
                "length": len(d),
                "capacity": d.capacity,
                "schema": _structure_schema(d),
            }

    @app.get(
        "/lru_dicts/{name}/items/{key}",
        tags=["lru_dicts"],
        summary="Read by key",
    )
    def lru_get(name: str, key: str):
        with lock:
            d = _get_structure(name, LRUDict)
            try:
                return _to_jsonable(d[key])
            except KeyError:
                raise HTTPException(404, f"key {key!r} not found")

    @app.put(
        "/lru_dicts/{name}/items/{key}",
        tags=["lru_dicts"],
        summary="Set by key",
    )
    def lru_set(name: str, key: str, value: dict = Body(...)):
        with lock:
            d = _get_structure(name, LRUDict)
            schema = _structure_schema(d) or {}
            data = _validate("lru", name, value, schema)
            d[key] = data
            return {"key": key}

    @app.delete(
        "/lru_dicts/{name}/items/{key}",
        tags=["lru_dicts"],
        summary="Delete by key",
        status_code=204,
    )
    def lru_delete(name: str, key: str):
        with lock:
            d = _get_structure(name, LRUDict)
            try:
                del d[key]
            except KeyError:
                raise HTTPException(404, f"key {key!r} not found")
            return None

    return app


def serve(db, host: str = "127.0.0.1", port: int = 8000, **uvicorn_kwargs):
    """Run a uvicorn server exposing `db` over HTTP.

    Blocking call.  Single-writer / single-reader: all requests are
    serialized through a process-wide lock.

    Args:
        db: an open `loom.DB` instance.
        host, port: bind address.
        **uvicorn_kwargs: forwarded to `uvicorn.run`.
    """
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is required for `db.serve()`. "
            "Install with: pip install 'fastapi[standard]'"
        ) from e

    app = build_app(db)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
