"""Tests for Collection — a record store with attached, auto-synced indexes.

The record lives once in the primary Dict (keyed by the primary key); each
attached index maps a field (or a computed/lambda key) → the primary key and
is kept in sync on insert / update / delete.  Field-name indexes persist and
auto-restore on reopen; lambda indexes must be re-supplied.
"""

import os
import tempfile

import pytest

from loom.database import DB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


SCHEMA = {"username": "U30", "email": "U40", "city": "U20", "age": "int64"}


def _users(db, **kw):
    return db.collection(
        "users", SCHEMA, key="username",
        indexes={"email": "email", "city": "city"}, **kw
    )


class TestBasics:
    def test_insert_and_primary_lookup(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        assert u["alice"]["email"] == "a@x.com"
        assert "alice" in u
        assert len(u) == 1

    def test_secondary_lookup(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        assert u.get("email", "a@x.com")["username"] == "alice"
        assert u.get("city", "NYC")["username"] == "alice"

    def test_get_pk(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        assert u.get_pk("email", "a@x.com") == "alice"
        assert u.get_pk("email", "missing") is None

    def test_missing_returns_default(self, db):
        u = _users(db)
        assert u.get("email", "nope") is None
        assert u.get("email", "nope", default="x") == "x"

    def test_unknown_index_raises(self, db):
        u = _users(db)
        with pytest.raises(KeyError):
            u.get("phone", "x")

    def test_insert_missing_pk_raises(self, db):
        u = _users(db)
        with pytest.raises(KeyError):
            u.insert({"email": "a@x.com"})  # no username

    def test_insert_many(self, db):
        u = _users(db)
        u.insert_many([
            {"username": "a", "email": "a@x.com", "city": "NYC", "age": 1},
            {"username": "b", "email": "b@x.com", "city": "LA", "age": 2},
        ])
        assert len(u) == 2
        assert u.get("city", "LA")["username"] == "b"


class TestUpdate:
    def test_update_reindexes_changed_key(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        u.update("alice", email="alice@new.com")
        assert u.get("email", "a@x.com") is None          # old key removed
        assert u.get("email", "alice@new.com")["username"] == "alice"
        assert u["alice"]["email"] == "alice@new.com"

    def test_update_leaves_unchanged_index_alone(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        u.update("alice", age=31)
        assert u.get("email", "a@x.com")["age"] == 31
        assert u.get("city", "NYC")["username"] == "alice"

    def test_update_missing_raises(self, db):
        u = _users(db)
        with pytest.raises(KeyError):
            u.update("ghost", age=1)

    def test_cannot_change_primary_key(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        with pytest.raises(ValueError):
            u.update("alice", username="alice2")


class TestDelete:
    def test_delete_removes_from_all_indexes(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        u.delete("alice")
        assert "alice" not in u
        assert u.get("email", "a@x.com") is None
        assert u.get("city", "NYC") is None
        assert len(u) == 0

    def test_delete_missing_raises(self, db):
        u = _users(db)
        with pytest.raises(KeyError):
            u.delete("ghost")


class TestLambdaIndex:
    def test_computed_key(self, db):
        u = db.collection(
            "u", SCHEMA, key="username",
            indexes={"email_lc": lambda r: r["email"].lower()},
        )
        u.insert({"username": "alice", "email": "Alice@X.com", "city": "NYC", "age": 30})
        assert u.get("email_lc", "alice@x.com")["username"] == "alice"
        assert u.get("email_lc", "Alice@X.com") is None  # only the lowercased key

    def test_lambda_reindexed_on_update(self, db):
        u = db.collection(
            "u", SCHEMA, key="username",
            indexes={"email_lc": lambda r: r["email"].lower()},
        )
        u.insert({"username": "alice", "email": "A@x.com", "city": "NYC", "age": 30})
        u.update("alice", email="B@y.com")
        assert u.get("email_lc", "a@x.com") is None
        assert u.get("email_lc", "b@y.com")["username"] == "alice"


class TestListIndexesSpec:
    def test_list_of_field_names(self, db):
        u = db.collection("u", SCHEMA, key="username", indexes=["email", "city"])
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        assert u.get("email", "a@x.com")["username"] == "alice"
        assert u.get("city", "NYC")["username"] == "alice"

    def test_no_indexes(self, db):
        u = db.collection("u", SCHEMA, key="username")
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        assert u["alice"]["email"] == "a@x.com"
        assert u.index_names == []


class TestPersistence:
    def test_field_indexes_auto_restore(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            u2 = db2.collection("users")  # reopen, no model — auto-restore
            assert u2["alice"]["email"] == "a@x.com"
            assert u2.get("email", "a@x.com")["username"] == "alice"
            assert set(u2.index_names) == {"email", "city"}
            # writes still keep indexes in sync after reopen
            u2.insert({"username": "bob", "email": "b@x.com", "city": "LA", "age": 2})
            assert u2.get("city", "LA")["username"] == "bob"
        finally:
            db2.close()

    def test_reopen_unknown_collection_raises(self, db):
        import loom
        with pytest.raises((loom.StructureNotFoundError, KeyError)):
            db.collection("nonexistent")

    def test_lambda_index_requires_resupply_on_reopen(self, db):
        u = db.collection(
            "u", SCHEMA, key="username",
            indexes={"email_lc": lambda r: r["email"].lower()},
        )
        u.insert({"username": "alice", "email": "A@x.com", "city": "NYC", "age": 30})
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            with pytest.raises(ValueError):
                db2.collection("u")  # lambda index not re-supplied
            # re-supplying the lambda restores it
            u2 = db2.collection("u", indexes={"email_lc": lambda r: r["email"].lower()})
            assert u2.get("email_lc", "a@x.com")["username"] == "alice"
        finally:
            db2.close()


class TestReindex:
    def test_reindex_rebuilds(self, db):
        u = _users(db)
        u.insert({"username": "alice", "email": "a@x.com", "city": "NYC", "age": 30})
        u.reindex()  # idempotent — indexes already correct
        assert u.get("email", "a@x.com")["username"] == "alice"
