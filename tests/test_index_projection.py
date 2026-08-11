"""Index-only projection on find().

When every field requested via ``fields=`` already lives in the Many index —
the primary key (in the entry), the group field (== the query value), and the
sort field (decodable from the composite key) — find() serves the hit straight
from the BTree traversal: no primary-store lookup, no record read.

The result must be byte-for-byte what a record read would return, including
Python scalar types.  These tests pin that parity against the record-read path
(forced on by monkeypatching the projector off) and check the fallback cases.
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

import loom.collection as C
from loom import DB, Many
from loom.collection import (encode_value, _decode_int, _decode_float,
                             _decode_str, _decode_datetime)


@pytest.fixture
def col():
    d = tempfile.TemporaryDirectory()
    db = DB(os.path.join(d.name, "x.loom"))
    c = db.collection(
        "v",
        {"id": "utf8[32]", "grp": "utf8[16]", "ts": "datetime",
         "score": "int64", "ratio": "float64", "note": "utf8[8]",
         "answer": "text"},
        indexes={"id": "primary",
                 "grp": Many(sort="ts", desc=True, counted=True)},
    )
    base = datetime(2026, 1, 1)
    c.insert_many([
        {"id": f"r{i}", "grp": "DEI" if i % 2 else "OTHER",
         "ts": base + timedelta(seconds=i, microseconds=i),
         "score": i - 50, "ratio": i * 0.25, "note": f"n{i}",
         "answer": ("lorem ipsum " * 20) + str(i)}
        for i in range(60)
    ])
    db.flush()
    c._td = d      # keep the tempdir alive
    yield c
    db.close()
    d.cleanup()


def _record_read(col, *args, **kw):
    """find() with the index-only projector forced off."""
    saved = C.Collection._index_projection
    C.Collection._index_projection = lambda *_a, **_k: None
    try:
        return col.find(*args, **kw)
    finally:
        C.Collection._index_projection = saved


def _assert_parity(col, value, fields, **kw):
    base = _record_read(col, "grp", value, fields=fields, **kw)
    fast = col.find("grp", value, fields=fields, **kw)
    assert len(base) == len(fast)
    for b, f in zip(base, fast):
        assert dict(b) == dict(f)
        for k in fields:
            assert type(b[k]) is type(f[k]), (k, type(b[k]), type(f[k]))
    return fast


class TestDecoderRoundTrip:
    def test_int(self):
        for desc in (False, True):
            for v in (0, 1, -1, 2 ** 62, -(2 ** 62),
                      9223372036854775807, -9223372036854775808):
                assert _decode_int(encode_value(v, desc), desc) == v

    def test_float(self):
        for desc in (False, True):
            for v in (0.0, -0.0, 1.5, -1.5, 1e-300, 1e300, -1e300, 3.141592653589793):
                assert _decode_float(encode_value(v, desc), desc) == v

    def test_str_and_datetime(self):
        for desc in (False, True):
            for v in ("", "a", "DEI", "héllo", "  x  "):
                assert _decode_str(encode_value(v, desc), desc) == v
            for v in (datetime(1970, 1, 1), datetime(2026, 6, 15, 12, 34, 56, 789012)):
                assert _decode_datetime(encode_value(v, desc), desc) == v

    def test_int_float_cache_no_collision(self):
        # 5 == 5.0 and hash the same, but int and float encode differently.
        # The sort-encode LRU must not let one shadow the other (would build a
        # wrong index key → mis-sort, and a wrong index-only projection).
        from loom.collection import _encode_sort
        assert _encode_sort(5, False) == encode_value(5, False)
        assert _encode_sort(5.0, False) == encode_value(5.0, False)
        assert _encode_sort(5, False) != _encode_sort(5.0, False)


class TestProjectionParity:
    def test_pk_only(self, col):
        assert col._index_projection(col._indexes["grp"], "DEI", ["id"]) is not None
        _assert_parity(col, "DEI", ["id"])

    def test_group_field(self, col):
        _assert_parity(col, "DEI", ["grp"])
        _assert_parity(col, "DEI", ["id", "grp"])

    def test_sort_field_desc(self, col):
        _assert_parity(col, "DEI", ["id", "ts"])
        _assert_parity(col, "DEI", ["ts"])
        _assert_parity(col, "DEI", ["id", "grp", "ts"])

    def test_windowed_and_limited(self, col):
        base = datetime(2026, 1, 1)
        _assert_parity(col, "DEI", ["id", "ts"],
                       start=base, end=base + timedelta(seconds=30))
        got = _assert_parity(col, "DEI", ["id"], limit=5)
        assert len(got) == 5


class TestFallback:
    def test_non_index_field_falls_back(self, col):
        # score is not in this index → must read the record
        assert col._index_projection(col._indexes["grp"], "DEI", ["id", "score"]) is None
        _assert_parity(col, "DEI", ["id", "score"])

    def test_blob_field_falls_back(self, col):
        assert col._index_projection(col._indexes["grp"], "DEI", ["id", "answer"]) is None
        _assert_parity(col, "DEI", ["id", "answer"])

    def test_full_record_when_fields_none(self, col):
        # fields=None never projects; returns whole records
        recs = col.find("grp", "DEI")
        assert recs and "answer" in recs[0] and "score" in recs[0]


class TestOtherSortDtypes:
    def _make(self, sort_field, dtype, desc):
        d = tempfile.mkdtemp()
        db = DB(os.path.join(d, "x.loom"))
        c = db.collection(
            "v", {"id": "utf8[16]", "grp": "utf8[8]", sort_field: dtype},
            indexes={"id": "primary", "grp": Many(sort=sort_field, desc=desc)})
        return db, c

    @pytest.mark.parametrize("desc", [False, True])
    def test_int_sort(self, desc):
        db, c = self._make("k", "int64", desc)
        c.insert_many([{"id": f"r{i}", "grp": "g", "k": i * 7 - 100} for i in range(20)])
        base = _record_read(c, "grp", "g", fields=["id", "k"])
        fast = c.find("grp", "g", fields=["id", "k"])
        assert [dict(x) for x in base] == [dict(x) for x in fast]
        assert all(type(x["k"]) is int for x in fast)
        db.close()

    @pytest.mark.parametrize("desc", [False, True])
    def test_float_sort(self, desc):
        db, c = self._make("k", "float64", desc)
        c.insert_many([{"id": f"r{i}", "grp": "g", "k": i * 1.5 - 10} for i in range(20)])
        base = _record_read(c, "grp", "g", fields=["id", "k"])
        fast = c.find("grp", "g", fields=["id", "k"])
        assert [dict(x) for x in base] == [dict(x) for x in fast]
        assert all(type(x["k"]) is float for x in fast)
        db.close()

    @pytest.mark.parametrize("desc", [False, True])
    def test_str_sort(self, desc):
        db, c = self._make("k", "utf8[16]", desc)
        c.insert_many([{"id": f"r{i}", "grp": "g", "k": f"tag{i:03d}"} for i in range(20)])
        base = _record_read(c, "grp", "g", fields=["id", "k"])
        fast = c.find("grp", "g", fields=["id", "k"])
        assert [dict(x) for x in base] == [dict(x) for x in fast]
        db.close()


class TestHashedGroup:
    def test_hashed_group_serves_pk_but_not_group(self):
        # A Many on an unbounded (text) field is hashed: the group value is a
        # lossy hash, so the group field can't be served index-only, but the pk
        # still can.
        d = tempfile.mkdtemp()
        db = DB(os.path.join(d, "x.loom"))
        c = db.collection(
            "v", {"id": "utf8[16]", "topic": "text", "ts": "datetime"},
            indexes={"id": "primary", "topic": Many(sort="ts")})
        base = datetime(2026, 1, 1)
        long_topic = "a very long topic string " * 5
        c.insert_many([{"id": f"r{i}", "topic": long_topic,
                        "ts": base + timedelta(seconds=i)} for i in range(10)])
        ix = c._indexes["topic"]
        assert ix.get("hashed")
        assert c._index_projection(ix, long_topic, ["id"]) is not None
        assert c._index_projection(ix, long_topic, ["topic"]) is None   # hashed
        got = c.find("topic", long_topic, fields=["id"])
        assert sorted(r["id"] for r in got) == [f"r{i}" for i in range(10)]
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
