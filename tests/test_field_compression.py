"""Per-field blob compression — selectively compress one text/json field
(e.g. a big article body) while the rest of the DB stays uncompressed.

Declared via Text(compression=...) / Json(compression=...) in a Pydantic model,
or the raw dtype tag "text[brotli]" / "json[zlib]" / "text[none]".
"""

import os
import tempfile

import pytest

from pydantic import BaseModel

from loom import DB
from loom.schema import Utf8, Text, Json

BOOK = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 1500  # ~84 KB


def _alloc_for_body(path, model):
    """Bytes the arena grows by when inserting one record — i.e. the blob size."""
    with DB(path) as db:
        c = db.collection("c", model, indexes={"id": "primary"})
        base = db._db.get_allocation_index()
        c.insert({"id": "x", "body": BOOK})
        return db._db.get_allocation_index() - base


class TestBlobStoreCodecOverride:
    def test_per_call_codec_overrides_store_default(self):
        from loom.blob import BlobStore

        with tempfile.TemporaryDirectory() as d:
            from loom.fileio import ByteFileDB

            raw = ByteFileDB(os.path.join(d, "b.loom"))
            raw.open()
            store = BlobStore(raw, compression=None)  # default: uncompressed
            data = BOOK.encode()

            off_plain, _ = store.write(data)                       # store default
            off_br, _ = store.write(data, compression="brotli")    # override

            # brotli blob is far smaller than the uncompressed one
            plain_hdr = raw.read(off_plain, 8)
            br_hdr = raw.read(off_br, 8)
            import struct

            plain_sz = struct.unpack("<II", plain_hdr)[0]
            br_sz = struct.unpack("<II", br_hdr)[0]
            assert br_sz < plain_sz / 20

            # each reads back correctly with its own codec
            assert store.read(off_plain) == data
            assert store.read(off_br, compression="brotli") == data
            raw.close()


class TestPydanticDeclaration:
    def test_text_brotli_only_compresses_that_field(self):
        with tempfile.TemporaryDirectory() as d:
            class Compressed(BaseModel):
                id: Utf8(16)
                body: Text(compression="brotli")

            class Plain(BaseModel):
                id: Utf8(16)
                body: str

            comp = _alloc_for_body(os.path.join(d, "c.loom"), Compressed)
            plain = _alloc_for_body(os.path.join(d, "p.loom"), Plain)
            assert plain > len(BOOK) / 2          # plain ≈ raw size
            assert comp < plain / 20              # brotli shrinks it hugely

    def test_roundtrip_and_reopen_no_model(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")

            class Article(BaseModel):
                id: Utf8(16)
                title: str                          # DB default (none)
                body: Text(compression="brotli")
                meta: Json(compression="zlib")

            with DB(path) as db:
                arts = db.collection("articles", Article, indexes={"id": "primary"})
                arts.insert_many([
                    {"id": f"a{i}", "title": f"t{i}", "body": BOOK,
                     "meta": {"k": list(range(30))}}
                    for i in range(10)
                ])
                assert arts["a3"]["body"] == BOOK
                assert arts["a3"]["meta"] == {"k": list(range(30))}
                assert arts["a3", "body"] == BOOK       # single-field read path

            # reopen with no model: codec must round-trip from the registry
            with DB(path) as db:
                arts = db.collection("articles")
                assert arts["a0"]["body"] == BOOK
                assert arts["a0"]["meta"] == {"k": list(range(30))}
                arts.update("a0", body=BOOK + " END")   # rewrite compressed field
                assert arts["a0"]["body"] == BOOK + " END"

    def test_bulk_read_path_decodes_compressed_field(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")

            class Doc(BaseModel):
                id: Utf8(16)
                body: Text(compression="brotli")

            with DB(path) as db:
                docs = db.collection("docs", Doc, indexes={"id": "primary"})
                docs.insert_many([{"id": f"d{i}", "body": f"{BOOK}{i}"} for i in range(25)])
                # .values() / items() drive the Dict bulk-read path
                bodies = {r["id"]: r["body"] for r in docs.values()}
                assert bodies["d10"] == f"{BOOK}10"
                assert len(bodies) == 25


class TestRawDtypeTags:
    def test_text_none_forces_uncompressed_against_db_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            # DB default = brotli; one field opts OUT via "text[none]".
            with DB(path, blob_compression="brotli") as db:
                ds = db.create_dataset("t", id="utf8[16]",
                                       comp="text", raw="text[none]")
                base = db._db.get_allocation_index()
                ref = ds.insert({"id": "x", "comp": BOOK, "raw": BOOK})
                grew = db._db.get_allocation_index() - base
                # 'raw' is stored uncompressed → total grows by at least raw size;
                # 'comp' (DB-default brotli) shrinks to almost nothing, so the
                # whole growth is dominated by the uncompressed 'raw' field.
                assert grew > len(BOOK)
                # both fields still read back correctly with their own codec
                rec = ds.read(ref.addr)
                assert rec["comp"] == BOOK
                assert rec["raw"] == BOOK

    def test_blob_codec_tag_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                with pytest.raises(ValueError, match="only supported on 'text'/'json'"):
                    db.create_dataset("t", id="utf8[16]", data="blob[brotli]")

    def test_unknown_codec_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                with pytest.raises(ValueError, match="unknown compression"):
                    db.create_dataset("t", id="utf8[16]", body="text[lz4]")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
