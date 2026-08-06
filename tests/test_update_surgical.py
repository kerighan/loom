"""Collection.update / increment write only the changed fields.

An unchanged blob (`text`/`json`) field keeps its stored reference — it is not
decompressed then recompressed — so a one-field bump (e.g. a counter) does not
re-encode every blob in the row. Also covers a latent write_field bug: a
single-field write of a vector (`Vec`/array) field silently broadcast the first
element; it must round-trip exactly.
"""

import os
import tempfile

import numpy as np
from datetime import datetime
from pydantic import BaseModel

from loom import DB, Many
from loom.schema import Utf8, Text, Json, Vec


class Doc(BaseModel):
    id: Utf8(16)
    title: Text(compression="brotli")
    meta: Json()
    n: int
    emb: Vec(8)


def _make(db):
    return db.collection("d", Doc, indexes={
        "id": "primary",
        "n": "range",          # top-N by n across the collection
    })


BODY = "Lorem ipsum dolor sit amet " * 200


class TestSurgicalUpdate:
    def test_increment_keeps_unchanged_blob_reference(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = _make(db)
                col.insert({"id": "k", "title": BODY, "meta": {"a": 1},
                            "n": 1, "emb": np.arange(8, dtype=np.float32)})
                # raw blob ref (offset) of `title` in the stored record
                vds = col._primary._values_dataset
                addr = col._primary._resolve_value_addr("k")
                toff = vds.schema.fields["title"][1]

                def title_blob_offset():
                    return int(np.frombuffer(vds.db.read(addr + toff, 8), dtype="uint64")[0])

                before = title_blob_offset()
                col.increment("k", "n", 5)
                assert title_blob_offset() == before      # blob NOT rewritten/moved
                r = col["k"]
                assert r["n"] == 6 and r["title"] == BODY  # value still correct
                assert r["meta"] == {"a": 1}

    def test_update_changed_field_only(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = _make(db)
                col.insert({"id": "k", "title": BODY, "meta": {"a": 1},
                            "n": 1, "emb": np.zeros(8, dtype=np.float32)})
                new = col.update("k", n=99, meta={"b": 2})
                assert new["n"] == 99 and new["meta"] == {"b": 2}
                assert new["title"] == BODY                # unchanged, returned intact
                got = col["k"]
                assert got["n"] == 99 and got["meta"] == {"b": 2} and got["title"] == BODY

    def test_update_reindexes_moving_sort_key(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                col = _make(db)
                for i in range(5):
                    col.insert({"id": f"k{i}", "title": f"t{i}", "meta": {}, "n": i,
                                "emb": np.zeros(8, dtype=np.float32)})
                col.increment("k0", "n", 100)              # k0 jumps to the top
                top = [r["id"] for r in col.range("n", None, None, desc=True, limit=3)]
                assert top[0] == "k0"

    def test_vector_field_survives_update_and_reopen(self):
        # Regression: single-field write of a Vec used to broadcast the first
        # element. update() writes it via that path, so it must round-trip.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            with DB(path) as db:
                col = _make(db)
                col.insert({"id": "k", "title": "t", "meta": {}, "n": 1,
                            "emb": np.zeros(8, dtype=np.float32)})
                v = np.arange(8, 16, dtype=np.float32)
                col.update("k", emb=v)
                assert np.array_equal(col["k"]["emb"], v)
            with DB(path) as db:                            # survives reopen
                assert np.array_equal(db.collection("d")["k"]["emb"], v)


class TestWriteFieldArray:
    def test_single_field_vector_write_roundtrips(self):
        from loom.dataset import Dataset

        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                ds = db.create_dataset("v", id="utf8[8]", vec="float32[8]")
                ref = ds.insert({"id": "a", "vec": np.zeros(8, dtype=np.float32)})
                ds.write_field(ref.addr, "vec", np.arange(8, 16, dtype=np.float32))
                assert np.array_equal(ds.read(ref.addr)["vec"],
                                      np.arange(8, 16, dtype=np.float32))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
