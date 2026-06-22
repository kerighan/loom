"""Fixed-width inline UTF-8 string field (`utf8[N]`).

Stored as numpy S{N} (N raw bytes) with transparent encode/decode — the
space-efficient middle ground between U{N} (UCS-4, 4 bytes/char) and `text`
(variable-length BlobStore). N is a BYTE budget; over-long values truncate on
a UTF-8 codepoint boundary (never splitting a character).
"""

import os
import tempfile

import pytest
from pydantic import BaseModel

from loom.database import DB
from loom.schema import Utf8, schema_from_model


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


# ── Core round-trip ──────────────────────────────────────────────────────────


class TestUtf8RoundTrip:
    def test_ascii_and_unicode(self, db):
        ds = db.create_dataset("d", id="uint32", s="utf8[32]")
        d = db.create_dict("dd", ds)
        d["a"] = {"id": 1, "s": "hello world"}
        d["b"] = {"id": 2, "s": "café münchen 日本"}
        d["c"] = {"id": 3, "s": ""}
        assert d["a"]["s"] == "hello world"
        assert d["b"]["s"] == "café münchen 日本"
        assert d["c"]["s"] == ""

    def test_record_size_smaller_than_ucs4(self, db):
        u = db.create_dataset("u", s="utf8[32]")
        w = db.create_dataset("w", s="U32")
        # utf8[32] = 32 bytes; U32 = 128 bytes (4×) — minus the 1-byte prefix.
        assert (u.record_size - 1) == 32
        assert (w.record_size - 1) == 128

    def test_truncation_is_codepoint_safe(self, db):
        ds = db.create_dataset("d", id="uint32", s="utf8[8]")
        d = db.create_dict("dd", ds)
        # 'é' is 2 bytes in UTF-8: 5×'é' = 10 bytes > 8 → must truncate to a
        # whole number of 'é' (8 bytes = 4 chars), never a half char.
        d["x"] = {"id": 1, "s": "é" * 5}
        got = d["x"]["s"]
        assert len(got.encode("utf-8")) <= 8
        assert got == "é" * 4              # clean boundary, no replacement char
        got.encode("utf-8")               # must be valid (no surrogate/partial)

    def test_survives_reopen(self, db):
        ds = db.create_dataset("d", id="uint32", s="utf8[40]")
        d = db.create_dict("dd", ds)
        d["k"] = {"id": 7, "s": "réopen-test-日本語"}
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            d2 = db2._datastructures["dd"]
            assert d2["k"]["s"] == "réopen-test-日本語"
            # the field must still be recognised as utf8 after reload, not raw S
            assert "s" in db2.get_dataset("d")._utf8_fields
        finally:
            db2.close()

    def test_to_dict_decodes(self, db):
        ds = db.create_dataset("d", id="uint32", s="utf8[20]")
        d = db.create_dict("dd", ds)
        for i in range(10):
            d[f"k{i}"] = {"id": i, "s": f"val_{i}_café"}
        dumped = d.to_dict()
        assert dumped["k3"]["s"] == "val_3_café"
        assert all(isinstance(v["s"], str) for v in dumped.values())

    def test_write_and_read_field(self, db):
        ds = db.create_dataset("d", s="utf8[16]")
        ref = ds.insert({"s": "first"})
        assert ds.read_field(ref.addr, "s") == "first"
        ds.write_field(ref.addr, "s", "updated café")
        assert ds.read_field(ref.addr, "s") == "updated café"
        assert ds[ref.addr]["s"] == "updated café"


# ── As a Dict / Graph key (_key recovery) ────────────────────────────────────


class TestUtf8Keys:
    def test_dict_keys_roundtrip_ascii_and_unicode(self, db):
        # store_key=True now stores _key as utf8[max_key_len] by default.
        ds = db.create_dataset("u", v="int64")
        d = db.create_dict("d", ds, max_key_len=64)
        keys = ["alice", "bob", "café", "naïve", "日本語キー", "x/y/z:1"]
        for i, k in enumerate(keys):
            d[k] = {"v": i}
        assert set(d.keys()) == set(keys)
        assert "" not in set(d.keys())
        for i, k in enumerate(keys):
            assert d[k]["v"] == i

    def test_graph_default_keys_utf8(self, db):
        g = db.create_graph("g", {"label": "U10"}, {"w": "int32"}, directed=True)
        edges = [("/m/abc", "/m/def", {"w": 1}),
                 ("/m/abc", "/m/ghi", {"w": 2}),
                 ("naïve", "café", {"w": 3})]
        g.add_edges(edges)
        assert sorted(g.neighbors("/m/abc")) == ["/m/def", "/m/ghi"]
        assert list(g.neighbors("naïve")) == ["café"]
        assert g.get_edge("/m/abc", "/m/def")["w"] == 1
        # reopen
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            g2 = db2._datastructures["g"]
            assert sorted(g2.neighbors("/m/abc")) == ["/m/def", "/m/ghi"]
            assert list(g2.neighbors("naïve")) == ["café"]
        finally:
            db2.close()


# ── Pydantic integration ─────────────────────────────────────────────────────


class TestUtf8Pydantic:
    def test_schema_from_model(self):
        class Page(BaseModel):
            url: Utf8(200)                 # raise by default → utf8[200!]
            slug: Utf8(64, truncate=True)  # opt into truncation → utf8[64]
            title: str

        assert schema_from_model(Page) == {
            "url": "utf8[200!]", "slug": "utf8[64]", "title": "text",
        }

    def test_overflow_raises_by_default(self, db):
        class Tag(BaseModel):
            name: Utf8(8)                  # raise by default

        ds = db.create_dataset("tags", model=Tag)
        assert ds._utf8_strict == {"name"}
        ds.insert({"name": "short"})       # fits → OK
        with pytest.raises(ValueError):
            ds.insert({"name": "way_too_long_value"})

    def test_truncate_true_still_truncates(self, db):
        class Tag(BaseModel):
            name: Utf8(8, truncate=True)

        ds = db.create_dataset("tags", model=Tag)
        ref = ds.insert({"name": "way_too_long_value"})
        assert ds[ref.addr]["name"] == "way_too_"   # 8 bytes, no error

    def test_strict_survives_reopen(self):
        import os, tempfile
        path = os.path.join(tempfile.mkdtemp(), "u.db")

        class Tag(BaseModel):
            name: Utf8(8)

        with DB(path) as db:
            db.create_dataset("tags", model=Tag)
        with DB(path) as db:
            ds = db.get_dataset("tags")
            assert ds._utf8_strict == {"name"}
            with pytest.raises(ValueError):
                ds.insert({"name": "way_too_long_value"})

    def test_create_dataset_from_model(self, db):
        class Page(BaseModel):
            id: int
            url: Utf8(64)

        ds = db.create_dataset("pages", model=Page)
        assert "url" in ds._utf8_fields and ds._utf8_fields["url"] == 64
        ref = ds.insert({"id": 1, "url": "https://example.com/héllo"})
        assert ds[ref.addr]["url"] == "https://example.com/héllo"
