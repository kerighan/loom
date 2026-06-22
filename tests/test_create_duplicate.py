"""create_* throws on duplicate names (like create_dataset); exist_ok opts back
into idempotent open-or-create; names are unique across both namespaces."""

import os
import tempfile

import pytest

from loom import DB
from loom.errors import DuplicateNameError


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    database = DB(os.path.join(tmp, "dup.db"))
    database.open()
    yield database
    database.close()


def test_create_datastructure_twice_raises(db):
    ds = db.create_dataset("posts_data", id="uint32")
    db.create_list("feed", ds)
    with pytest.raises(DuplicateNameError):
        db.create_list("feed", ds)


def test_exist_ok_returns_same_instance(db):
    ds = db.create_dataset("posts_data", id="uint32")
    a = db.create_list("feed", ds)
    b = db.create_list("feed", ds, exist_ok=True)
    assert a is b
    # retrieval via the getter returns the same object too
    assert db["feed"] is a


def test_dataset_exist_ok(db):
    a = db.create_dataset("posts_data", id="uint32")
    b = db.create_dataset("posts_data", id="uint32", exist_ok=True)
    assert a is b
    with pytest.raises(DuplicateNameError):
        db.create_dataset("posts_data", id="uint32")


def test_name_unique_across_namespaces(db):
    ds = db.create_dataset("posts", id="uint32")
    # a datastructure cannot reuse a dataset's name (db["posts"] would be ambiguous)
    with pytest.raises(DuplicateNameError):
        db.create_list("posts", ds)
    # and vice-versa
    ds2 = db.create_dataset("events_data", id="uint32")
    db.create_list("events", ds2)
    with pytest.raises(DuplicateNameError):
        db.create_dataset("events", id="uint32")


def test_collection_iteration_yields_records(db):
    col = db.collection("posts", {"post_id": "utf8[32]", "n": "int64"},
                        indexes={"post_id": "primary"})
    for i in range(5):
        col.insert({"post_id": f"p{i}", "n": i})

    # `for x in collection` yields records (not integer-indexed → no KeyError)
    records = list(col)
    assert len(records) == 5
    assert sum(1 for _ in col) == 5
    assert all(isinstance(r, dict) and "post_id" in r for r in records)
    assert {r["post_id"] for r in records} == {f"p{i}" for i in range(5)}
    # keys()/items() remain available for pks / pairs
    assert set(col.keys()) == {f"p{i}" for i in range(5)}
    assert all(isinstance(pk, str) for pk, _ in col.items())


def test_collection_duplicate_and_retrieval(db):
    from loom import Many

    model = {"post_id": "utf8[32]", "name": "utf8[64]", "created_at": "int64"}
    idx = {"post_id": "primary", "name": Many(sort="created_at", desc=True)}
    posts = db.collection("posts", model, indexes=idx)
    posts.insert({"post_id": "p1", "name": "alice", "created_at": 1})

    # db[name] / `in` work for collections (not just datasets/structures)
    assert "posts" in db
    assert db["posts"] is not None
    assert len(db["posts"]) == 1

    # re-creating an existing collection raises a clear "Collection" error
    with pytest.raises(DuplicateNameError, match="Collection"):
        db.collection("posts", model, indexes=idx)
    # exist_ok reopens it
    again = db.collection("posts", model, indexes=idx, exist_ok=True)
    assert len(again) == 1

    # a dataset / structure cannot reuse a collection's name (db[name] ambiguity)
    with pytest.raises(DuplicateNameError):
        db.create_dataset("posts", id="int64")
    tmp_ds = db.create_dataset("tmp_ds", id="int64")
    with pytest.raises(DuplicateNameError):
        db.create_list("posts", tmp_ds)


def test_unique_enforced_in_insert_many(db):
    from loom import Unique

    users = db.collection("users", {"uid": "utf8[8]", "email": "utf8[24]"},
                          indexes={"uid": "primary", "email": "unique"})
    # duplicate email within the batch → raises, nothing written
    with pytest.raises(ValueError):
        users.insert_many([{"uid": "u1", "email": "a@x"}, {"uid": "u2", "email": "a@x"}])
    assert len(users) == 0

    users.insert_many([{"uid": "u1", "email": "a@x"}, {"uid": "u2", "email": "b@x"}])
    assert len(users) == 2
    # reusing an existing email under a new pk → raises, length unchanged
    with pytest.raises(ValueError):
        users.insert_many([{"uid": "u3", "email": "a@x"}])
    assert len(users) == 2
    # upserting the same pk with an unchanged unique value is fine
    users.insert_many([{"uid": "u1", "email": "a@x"}])
    assert users.get("email", "a@x")["uid"] == "u1"


def test_failed_collection_creation_rolls_back():
    # A creation that overflows the header must not leave orphan internal
    # structures that block a retry.
    from loom import Many, Search
    from loom.errors import HeaderTooLargeError, DuplicateNameError

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "rb.db")
    with DB(path, header_size=2048) as db:   # tiny header → cfg save overflows
        schema = {"id": "utf8[16]", "a": "utf8[16]", "b": "utf8[16]", "text": "text"}
        idx = {"id": "primary", "a": Many(sort="id"), "b": Many(sort="id"),
               "body": Search(fields=["text"], scoring="bm25")}
        with pytest.raises(HeaderTooLargeError):
            db.collection("posts", schema, indexes=idx)
        assert "posts" not in db
        assert not [n for n in db._datastructures if n.startswith("posts")]
        assert not [n for n in db._datasets if n.startswith("posts")]
        # the retry is not blocked by a DuplicateNameError from orphans
        with pytest.raises(HeaderTooLargeError):
            db.collection("posts", schema, indexes=idx)


def test_reopen_retrieves_via_getitem():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "dup.db")
    with DB(path) as db:
        ds = db.create_dataset("posts_data", id="uint32", n="int64")
        feed = db.create_list("feed", ds)
        feed.append({"id": 1, "n": 10})

    with DB(path) as db:
        # re-calling create_* on reopen now raises (consistent with create_dataset)
        with pytest.raises(DuplicateNameError):
            db.create_list("feed", db["posts_data"])
        # the clean way to get an existing structure back:
        feed = db["feed"]
        assert len(feed) == 1
        assert feed[0]["n"] == 10
        # or, explicitly idempotent:
        feed2 = db.create_list("feed", db["posts_data"], exist_ok=True)
        assert feed2 is feed
