"""Auto-sized index_key_size + close-time file truncation.

index_key_size defaults to None → each collection sizes its BTree composite
keys to exactly what the declared fields need (group + sort + pk + margin),
instead of a fat 192-byte floor that made every node ~2x larger than typical
keys warrant.  And close() trims the exponential-growth slack so the on-disk
file ≈ the live data.  Both are read back-compatible (a BTree's key_size is
persisted per structure, so existing files keep their layout on reopen).
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from loom import DB, Many, Range


@pytest.fixture
def tmp():
    d = tempfile.TemporaryDirectory()
    yield d.name
    d.cleanup()


class TestAutoKeySize:
    def test_narrow_keys_below_old_default(self, tmp):
        db = DB(os.path.join(tmp, "x.loom"))
        col = db.collection(
            "v", {"id": "utf8[16]", "grp": "utf8[8]", "ts": "datetime"},
            indexes={"id": "primary", "grp": Many(sort="ts")})
        ks = col._indexes["grp"]["struct"]._key_size
        # group(8) + sort(22) + pk(16) + margin(4) = 50, well under the old 192
        assert ks == 8 + 22 + 16 + 4
        assert ks < 192
        db.close()

    def test_wide_field_grows_key(self, tmp):
        db = DB(os.path.join(tmp, "x.loom"))
        col = db.collection(
            "v", {"id": "utf8[128]", "grp": "utf8[64]"},
            indexes={"id": "primary", "grp": Many()})
        ks = col._indexes["grp"]["struct"]._key_size
        assert ks == 64 + 0 + 128 + 4          # grows to fit wide fields
        db.close()

    def test_explicit_floor_honoured_and_grows(self, tmp):
        db = DB(os.path.join(tmp, "x.loom"))
        col = db.collection(
            "v", {"id": "utf8[16]", "grp": "utf8[8]"},
            indexes={"id": "primary", "grp": Many()}, index_key_size=300)
        assert col._indexes["grp"]["struct"]._key_size == 300   # floor wins
        db.close()

    def test_reopen_preserves_stored_key_size(self, tmp):
        p = os.path.join(tmp, "x.loom")
        db = DB(p)
        col = db.collection(
            "v", {"id": "utf8[16]", "grp": "utf8[8]"},
            indexes={"id": "primary", "grp": Many()})
        created = col._indexes["grp"]["struct"]._key_size
        db.close()
        db = DB(p)
        assert db["v"]._indexes["grp"]["struct"]._key_size == created
        db.close()

    def test_functional_with_narrow_keys(self, tmp):
        db = DB(os.path.join(tmp, "x.loom"))
        col = db.collection(
            "v", {"id": "utf8[16]", "grp": "utf8[8]", "ts": "datetime"},
            indexes={"id": "primary", "grp": Many(sort="ts", desc=True, counted=True)})
        base = datetime(2026, 1, 1)
        col.insert_many([{"id": f"r{i}", "grp": f"g{i % 3}",
                          "ts": base + timedelta(seconds=i)} for i in range(60)])
        assert col.count("grp", "g0") == 20
        got = [r["id"] for r in col.find("grp", "g1", fields=["id"])]
        assert len(got) == 20
        assert col["r5"]["grp"] == "g2"
        db.close()

    def test_long_value_not_truncated(self, tmp):
        # A pk/group at the declared width must round-trip through the index
        # (auto-size must never under-size and drop rows).
        db = DB(os.path.join(tmp, "x.loom"))
        col = db.collection(
            "v", {"id": "utf8[40]", "grp": "utf8[40]"},
            indexes={"id": "primary", "grp": Many()})
        pk = "z" * 40
        grp = "g" * 40
        col.insert({"id": pk, "grp": grp})
        assert [r["id"] for r in col.find("grp", grp)] == [pk]
        db.close()


class TestCloseTruncatesSlack:
    def _build(self, p, n):
        db = DB(p)
        col = db.collection(
            "v", {"id": "utf8[16]", "grp": "utf8[8]", "blob": "text"},
            indexes={"id": "primary", "grp": Many(counted=True)})
        col.insert_many([{"id": f"r{i}", "grp": f"g{i % 5}",
                          "blob": "x" * 500} for i in range(n)])
        return db, col

    def test_file_trimmed_to_high_water(self, tmp):
        p = os.path.join(tmp, "x.loom")
        db, _ = self._build(p, 3000)
        s = db.stats()
        open_file = s["file_size"]
        allocated = s["allocated_bytes"]
        assert open_file > allocated            # growth slack while open
        db.close()
        closed = os.path.getsize(p)
        assert closed <= allocated + 65536       # trimmed to ~high-water
        assert closed < open_file

    def test_reopen_after_trim_reads_all(self, tmp):
        p = os.path.join(tmp, "x.loom")
        db, _ = self._build(p, 2000)
        db.close()
        db = DB(p)
        assert len(db["v"]) == 2000
        assert db["v"]["r1234"]["blob"] == "x" * 500
        assert db["v"].count("grp", "g3") == 400
        db.close()

    def test_readonly_open_does_not_fail(self, tmp):
        p = os.path.join(tmp, "x.loom")
        db, _ = self._build(p, 100)
        db.close()
        r = DB(p, flag="r")                      # read-only never truncates
        assert len(r["v"]) == 100
        r.close()

    def test_idempotent_reopen_close_cycles(self, tmp):
        p = os.path.join(tmp, "x.loom")
        db, _ = self._build(p, 500)
        db.close()
        sizes = []
        for _ in range(3):
            db = DB(p)
            db["v"].insert({"id": "extra", "grp": "g0", "blob": "y"})
            db["v"].delete("extra")
            db.close()
            sizes.append(os.path.getsize(p))
        # no unbounded growth across open/close cycles
        assert max(sizes) - min(sizes) < 65536


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
