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
