"""A field named `model` (or `exist_ok`) must not be silently dropped.

These names collide with create_dataset(dataset_name, model=None, exist_ok=False,
**schema) under Python's keyword dispatch — directly and through the internal
`**schema` spreads (collection → primary Dict values dataset). loom restores the
field instead of dropping it (only a real Pydantic `model=` / bool `exist_ok=`
are treated as the parameters).
"""

import os
import tempfile

import pytest
from pydantic import BaseModel

from loom import DB
from loom.schema import Utf8


class Rec(BaseModel):
    id: Utf8(24)
    model: Utf8(48)          # would collide with create_dataset(model=)
    exist_ok: Utf8(8)        # would collide with create_dataset(exist_ok=)
    provider: Utf8(16)


REC = {"id": "x1", "model": "brightdata-perplexity",
       "exist_ok": "yes", "provider": "bd"}


def _check(rec):
    assert rec["model"] == "brightdata-perplexity"
    assert rec["exist_ok"] == "yes"
    assert rec["provider"] == "bd"


class TestReservedFieldNames:
    def test_pydantic_model_via_collection(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = db.collection("exports", Rec, indexes={"id": "primary"})
                col.insert(REC)
                _check(col["x1"])

    def test_pydantic_model_via_create_dataset_positional(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                ds = db.create_dataset("exports", Rec)
                ref = ds.insert(REC)
                _check(ds.read(ref.addr))

    def test_pydantic_model_keyword_still_works(self):
        # `model=<PydanticClass>` must remain a valid way to pass the schema.
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                ds = db.create_dataset("exports", model=Rec)
                assert set(ds.user_schema.names) >= {"id", "model", "exist_ok", "provider"}

    def test_raw_kwargs_with_model_and_exist_ok_fields(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                ds = db.create_dataset(
                    "exports",
                    id="utf8[24]", model="utf8[48]",
                    exist_ok="utf8[8]", provider="utf8[16]",
                )
                assert "model" in ds.user_schema.names
                assert "exist_ok" in ds.user_schema.names
                ref = ds.insert(REC)
                _check(ds.read(ref.addr))

    def test_dict_schema_as_model_arg(self):
        # A schema dict passed positionally as `model` keeps a `model` field.
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                ds = db.create_dataset(
                    "exports",
                    {"id": "utf8[24]", "model": "utf8[48]", "provider": "utf8[16]"},
                )
                assert "model" in ds.user_schema.names

    def test_exist_ok_still_functions_as_a_flag(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                db.create_dataset("t", id="utf8[8]")
                # real bool exist_ok=True → returns the existing dataset, no raise
                ds = db.create_dataset("t", id="utf8[8]", exist_ok=True)
                assert ds is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
