import os, tempfile
import pytest
from loom import DB, Many, Search


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = DB(os.path.join(d, "m.db"))
    database.open()
    yield database
    database.close()


def _seed(db):
    col = db.collection("users", {"uid": "utf8[8]", "name": "utf8[32]",
                                  "cat": "utf8[8]", "bio": "text"},
                        indexes={"uid": "primary", "cat": Many(sort="uid"),
                                 "body": Search(fields=["name", "bio"], scoring="bm25")})
    col.insert_many([
        {"uid": "u1", "name": "Alice Martin", "cat": "a", "bio": "python dev"},
        {"uid": "u2", "name": "Bob Durand", "cat": "b", "bio": "java dev"},
    ])
    return col


def test_migrate_rename_add_drop(db):
    _seed(db)
    col = db.migrate_collection("users",
        {"uid": "utf8[8]", "first": "utf8[16]", "last": "utf8[16]",
         "cat": "utf8[8]", "bio": "text", "score": "int64"},
        transforms={"first": lambda r: r["name"].split()[0],
                    "last": lambda r: r["name"].split()[-1]})
    assert len(col) == 2
    assert dict(col["u1"]) == {"uid": "u1", "first": "Alice", "last": "Martin",
                               "cat": "a", "bio": "python dev", "score": 0}
    assert "name" not in col["u1"]                     # dropped
    assert col["u1"]["score"] == 0                     # new field default
    # indexes + full-text rebuilt
    assert [r["uid"] for r in col.find("cat", "a")] == ["u1"]
    assert [r["uid"] for r in col.search("body", "python")] == ["u1"]


def test_migrate_persists(db):
    _seed(db)
    db.migrate_collection("users",
        {"uid": "utf8[8]", "name": "utf8[32]", "cat": "utf8[8]", "bio": "text", "v": "int64"})
    path = db.filename
    db.close()
    with DB(path) as db2:
        c = db2["users"]
        assert len(c) == 2 and c["u2"]["v"] == 0
        assert [r["uid"] for r in c.search("body", "java")] == ["u2"]


def test_drop_collection(db):
    _seed(db)
    assert "users" in db
    db.drop_collection("users")
    assert "users" not in db
    # name is free again → can recreate
    col = db.collection("users", {"uid": "utf8[8]"}, indexes={"uid": "primary"})
    assert len(col) == 0


def test_vacuum_reclaims_space_and_preserves_data(db):
    col = db.collection("posts", {"pid": "utf8[8]", "cat": "utf8[8]", "text": "text"},
                        indexes={"pid": "primary", "cat": Many(sort="pid"),
                                 "body": Search(fields=["text"], scoring="bm25")})
    col.insert_many([{"pid": f"p{i}", "cat": "a",
                      "text": f"hello world number {i} python " * 8}
                     for i in range(3000)])
    for i in range(3000):
        if i % 3:                       # delete ~2/3 → lots of dead space
            col.delete(f"p{i}")
    db.flush()
    path = db.filename
    size_before = os.path.getsize(path)
    live = len(col)

    db.vacuum()

    assert os.path.getsize(path) < size_before          # space reclaimed
    col2 = db["posts"]
    assert len(col2) == live                            # data intact
    assert len(col2.find("cat", "a")) == live           # index rebuilt
    assert len(col2.search("body", "python")) == live   # full-text rebuilt

    # survives reopen
    db.close()
    with DB(path) as db2:
        assert len(db2["posts"]) == live


def test_vacuum_refuses_standalone_structures(db):
    ds = db.create_dataset("docs", id="int64")
    db.create_dict("d", ds)   # standalone (non-collection) structure
    with pytest.raises(NotImplementedError):
        db.vacuum()
