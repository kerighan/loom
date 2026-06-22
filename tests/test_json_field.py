"""json field: a BlobStore field transparently json.dumps'd / json.loads'd."""

import os
import tempfile

import pytest
from pydantic import BaseModel

from loom import DB, Json
from loom.schema import schema_from_model


class Event(BaseModel):
    id: int
    meta: dict          # → json
    tags: Json()        # → json (a list)


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = DB(os.path.join(d, "j.db"))
    database.open()
    yield database
    database.close()


def test_pydantic_maps_dict_and_Json_to_json():
    assert schema_from_model(Event) == {"id": "int64", "meta": "json", "tags": "json"}


def test_dataset_roundtrip(db):
    ds = db.create_dataset("ev", Event)
    meta = {"a": 1, "b": [2, 3], "nested": {"x": True, "y": None}}
    ref = ds.insert({"id": 1, "meta": meta, "tags": ["ml", "nlp"]})
    row = ds[ref.addr]
    assert row["meta"] == meta and isinstance(row["meta"], dict)
    assert row["tags"] == ["ml", "nlp"]
    assert ds[ref.addr, "meta"]["nested"]["x"] is True   # single-field read


def test_none_and_empty(db):
    ds = db.create_dataset("ev", Event)
    ref = ds.insert({"id": 1, "meta": None, "tags": []})
    assert ds[ref.addr]["meta"] is None
    assert ds[ref.addr]["tags"] == []


def test_write_field_updates_json(db):
    ds = db.create_dataset("ev", Event)
    ref = ds.insert({"id": 1, "meta": {"v": 1}, "tags": []})
    ds[ref.addr, "meta"] = {"v": 2, "extra": "x"}
    assert ds[ref.addr]["meta"] == {"v": 2, "extra": "x"}


def test_json_in_collection_and_reopen():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "j.db")
    with DB(path) as db:
        posts = db.collection("posts", {"pid": "utf8[8]", "attrs": "json", "text": "text"},
                              indexes={"pid": "primary"})
        posts.insert({"pid": "p1", "attrs": {"score": 9, "labels": ["a", "b"]}, "text": "hi"})
    with DB(path) as db:
        rec = db["posts"]["p1"]
        assert rec["attrs"] == {"score": 9, "labels": ["a", "b"]}
