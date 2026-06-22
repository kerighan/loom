"""Tests for the first-class `datetime` field type (stored inline as int64 µs)."""

import os
import tempfile
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from loom import DB, Datetime, Many


class Event(BaseModel):
    id: int
    created_at: datetime
    note: str


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "dt.db")
    database = DB(path)
    database.open()
    yield database
    database.close()


def test_pydantic_datetime_maps_to_datetime_dtype():
    from loom.schema import schema_from_model

    assert schema_from_model(Event) == {
        "id": "int64", "created_at": "datetime", "note": "text",
    }


def test_dataset_roundtrip_microsecond_precision(db):
    ds = db.create_dataset("events", Event)
    ts = datetime(2026, 6, 22, 14, 30, 5, 123456)
    ref = ds.insert({"id": 1, "created_at": ts, "note": "hi"})
    row = ds[ref.addr]
    assert isinstance(row["created_at"], datetime)
    assert row["created_at"] == ts
    # single-field read + write
    assert ds[ref.addr, "created_at"] == ts
    ds[ref.addr, "created_at"] = datetime(2030, 1, 1)
    assert ds[ref.addr, "created_at"] == datetime(2030, 1, 1)


def test_date_and_tzaware_inputs(db):
    ds = db.create_dataset("events", Event)
    # a date → midnight
    r1 = ds.insert({"id": 1, "created_at": date(2026, 1, 2), "note": ""})
    assert ds[r1.addr]["created_at"] == datetime(2026, 1, 2, 0, 0, 0)
    # tz-aware → stored UTC, returned naive-UTC
    aware = datetime(2026, 1, 2, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    r2 = ds.insert({"id": 2, "created_at": aware, "note": ""})
    assert ds[r2.addr]["created_at"] == datetime(2026, 1, 2, 10, 0, 0)


def test_datetime_reopen():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "dt.db")
    ts = datetime(2026, 6, 22, 9, 0, 0)
    with DB(path) as db:
        ds = db.create_dataset("events", Event)
        ref = ds.insert({"id": 1, "created_at": ts, "note": "x"})
        addr = ref.addr
    with DB(path) as db:
        ds = db.get_dataset("events")
        assert "created_at" in ds._datetime_fields
        assert ds[addr]["created_at"] == ts


def test_datetime_helper_annotation(db):
    from typing import Optional

    class M(BaseModel):
        id: int
        when: Datetime()

    from loom.schema import schema_from_model
    assert schema_from_model(M)["when"] == "datetime"


def test_collection_datetime_range_and_many(db):
    class Msg(BaseModel):
        id: str
        created_at: datetime
        box: str

    inbox = db.collection("inbox", Msg, indexes={
        "id": "primary",
        "created_at": "range",
        "box": Many(sort="created_at", desc=True),
    })
    for i in range(5):
        inbox.insert({"id": f"m{i}", "created_at": datetime(2026, 1, 1 + i), "box": "main"})

    # most recent 3, no grouping — pass datetime objects directly
    latest = inbox.range("created_at", limit=3, desc=True)
    assert [r["id"] for r in latest] == ["m4", "m3", "m2"]
    # range bound with a datetime object
    assert sorted(r["id"] for r in
                  inbox.range("created_at", datetime(2026, 1, 3), None)) == ["m2", "m3", "m4"]
    # per-group ordered feed
    assert [r["id"] for r in inbox.find("box", "main", limit=2)] == ["m4", "m3"]
    # records carry real datetime objects
    assert isinstance(inbox["m0"]["created_at"], datetime)
