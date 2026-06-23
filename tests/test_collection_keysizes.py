"""Long primary keys auto-size index storage; text fields are indexed by hash."""

import os
import tempfile

import pytest
from loom import DB, Many, Search, Utf8


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = DB(os.path.join(d, "c.db"), header_size=131072)
    database.open()
    yield database
    database.close()


def test_long_primary_key_not_truncated(db):
    # URL primary up to 256 bytes — index entries must keep the full key
    col = db.collection("articles", {"url": "utf8[256]", "media_type": "utf8[16]", "pert": "int64"},
                        indexes={"url": "primary",
                                 "media_type": Many(sort="pert", desc=True),
                                 "pert": "range"})
    base = "https://news.example.com/2026/an-article-with-a-very-long-slug-padding-" + "x" * 80 + "-"
    for i in range(10):
        col.insert({"url": base + str(i), "media_type": "pure_player", "pert": i * 10})

    assert len(col) == 10
    assert len(col.find("media_type", "pure_player")) == 10        # was 0 before the fix
    assert len(col.range("pert", 0, None)) == 10
    assert col[base + "5"]["pert"] == 50                            # full-key lookup
    assert len(list(col.keys())[0]) > 64                           # keys not truncated


def test_many_index_on_long_text_field_is_hashed(db):
    col = db.collection("posts", {"pid": "utf8[16]", "argument": "text", "eng": "int64"},
                        indexes={"pid": "primary",
                                 "argument": Many(sort="eng", desc=True)})
    arg = "A fairly long argument that reads like an article title " * 3
    for i in range(6):
        col.insert({"pid": f"p{i}", "argument": arg if i % 2 else "other", "eng": i * 100})

    hits = col.find("argument", arg)            # grouped by the long text, sorted by eng desc
    assert [r["pid"] for r in hits] == ["p5", "p3", "p1"]
    assert col.find("argument", "never seen") == []
    # the index was created as hashed
    assert col._indexes["argument"]["hashed"] is True


def test_unique_index_on_text_field(db):
    col = db.collection("docs", {"id": "utf8[8]", "slug": "text"},
                        indexes={"id": "primary", "slug": "unique"})
    long_slug = "this-is-a-very-long-slug-" + "y" * 100
    col.insert({"id": "d1", "slug": long_slug})
    assert col.get("slug", long_slug)["id"] == "d1"
    assert col.get("slug", "missing") is None


def test_range_on_text_field_rejected(db):
    with pytest.raises(ValueError):
        db.collection("x", {"id": "utf8[8]", "body": "text"},
                      indexes={"id": "primary", "body": "range"})
