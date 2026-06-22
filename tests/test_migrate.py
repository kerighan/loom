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
