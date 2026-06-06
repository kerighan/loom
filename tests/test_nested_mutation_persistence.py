"""Systematic guard against the 'nested mutation not propagated' bug class.

A nested data structure (List/Dict/Set/Queue inside a Dict or List) stores its
state in a *ref record* held by the parent.  Every mutating method must push
the updated ref back to the parent (``update_nested_ref``) — otherwise the
mutation is lost the next time the parent re-materialises the child from a
stale ref, both in-session and across a close/reopen.

This was a real bug: ``List.__delitem__`` and ``Dict.__delitem__`` updated
their own in-memory metadata but never propagated it, so a delete on a nested
List silently reappeared (and a nested Dict's len() went stale).  These tests
exercise *every* (outer, inner) combination and the mutating op of each inner
type, checking the result (a) on a fresh re-fetch from the parent and (b) after
a close/reopen — which is exactly where the propagation gap shows up.
"""

import os
import tempfile

import pytest

from loom.database import DB
from loom.datastructures import List, Dict, Set, Queue


@pytest.fixture
def path():
    fd, p = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    yield p
    try:
        os.unlink(p)
    except FileNotFoundError:
        pass


# ── outer-container helpers ─────────────────────────────────────────────────


def _make_outer(db, outer_kind, template):
    if outer_kind == "dict":
        return db.create_dict("outer", template)
    return db.create_list("outer", template)


def _inner(outer, outer_kind, *, create=False):
    """Fetch (a fresh handle to) the single inner structure under the outer."""
    if outer_kind == "dict":
        return outer["slot"]            # auto-creates on first access
    if create and len(outer) == 0:
        outer.append()
    return outer[0]


def _reopen_outer(path, outer_kind):
    db = DB(path)
    return db, db._datastructures["outer"]


OUTERS = ["dict", "list"]


# ── inner = List  (mutation: __delitem__) ───────────────────────────────────


@pytest.mark.parametrize("outer_kind", OUTERS)
def test_nested_list_delete(path, outer_kind):
    with DB(path) as db:
        tmpl = List.template(db.create_dataset("inner_ds", n="int64"))
        outer = _make_outer(db, outer_kind, tmpl)
        inner = _inner(outer, outer_kind, create=True)
        for i in range(5):
            inner.append({"n": i})
        del inner[2]                                  # remove n==2
        # fresh handle from the parent must already reflect the delete
        refetched = _inner(outer, outer_kind)
        assert len(refetched) == 4
        assert [r["n"] for r in refetched] == [0, 1, 3, 4]

    db, outer = _reopen_outer(path, outer_kind)
    try:
        inner = _inner(outer, outer_kind)
        assert len(inner) == 4
        assert [r["n"] for r in inner] == [0, 1, 3, 4]
    finally:
        db.close()


# ── inner = Dict  (mutation: __delitem__) ───────────────────────────────────


@pytest.mark.parametrize("outer_kind", OUTERS)
def test_nested_dict_delete(path, outer_kind):
    with DB(path) as db:
        tmpl = Dict.template(db.create_dataset("inner_ds", n="int64"))
        outer = _make_outer(db, outer_kind, tmpl)
        inner = _inner(outer, outer_kind, create=True)
        for i in range(5):
            inner[f"k{i}"] = {"n": i}
        del inner["k2"]
        refetched = _inner(outer, outer_kind)
        assert len(refetched) == 4
        assert "k2" not in refetched
        assert set(refetched.keys()) == {"k0", "k1", "k3", "k4"}

    db, outer = _reopen_outer(path, outer_kind)
    try:
        inner = _inner(outer, outer_kind)
        assert len(inner) == 4
        assert "k2" not in inner
        assert "k3" in inner
    finally:
        db.close()


# ── inner = Set  (mutation: remove) ─────────────────────────────────────────


@pytest.mark.parametrize("outer_kind", OUTERS)
def test_nested_set_remove(path, outer_kind):
    with DB(path) as db:
        tmpl = Set.template(key_size=20)
        outer = _make_outer(db, outer_kind, tmpl)
        inner = _inner(outer, outer_kind, create=True)
        for i in range(5):
            inner.add(f"v{i}")
        inner.remove("v2")
        refetched = _inner(outer, outer_kind)
        assert len(refetched) == 4
        assert "v2" not in refetched
        assert set(refetched) == {"v0", "v1", "v3", "v4"}

    db, outer = _reopen_outer(path, outer_kind)
    try:
        inner = _inner(outer, outer_kind)
        assert len(inner) == 4
        assert "v2" not in inner
    finally:
        db.close()


# ── inner = Queue  (mutation: pop) ──────────────────────────────────────────


@pytest.mark.parametrize("outer_kind", OUTERS)
def test_nested_queue_pop(path, outer_kind):
    with DB(path) as db:
        tmpl = Queue.template(db.create_dataset("inner_ds", n="int64"), block_size=8)
        outer = _make_outer(db, outer_kind, tmpl)
        inner = _inner(outer, outer_kind, create=True)
        for i in range(5):
            inner.push({"n": i})
        assert inner.pop()["n"] == 0          # FIFO
        refetched = _inner(outer, outer_kind)
        assert len(refetched) == 4
        assert refetched.peek()["n"] == 1     # next to come out

    db, outer = _reopen_outer(path, outer_kind)
    try:
        inner = _inner(outer, outer_kind)
        assert len(inner) == 4
        assert inner.pop()["n"] == 1
    finally:
        db.close()


# ── repeated delete then re-add (catches index/freelist drift) ──────────────


@pytest.mark.parametrize("outer_kind", OUTERS)
def test_nested_list_delete_then_append(path, outer_kind):
    with DB(path) as db:
        tmpl = List.template(db.create_dataset("inner_ds", n="int64"))
        outer = _make_outer(db, outer_kind, tmpl)
        inner = _inner(outer, outer_kind, create=True)
        for i in range(4):
            inner.append({"n": i})
        del inner[1]
        del inner[1]                                   # was n==2 after first del
        _inner(outer, outer_kind).append({"n": 99})
        vals = [r["n"] for r in _inner(outer, outer_kind)]
        assert vals == [0, 3, 99]

    db, outer = _reopen_outer(path, outer_kind)
    try:
        assert [r["n"] for r in _inner(outer, outer_kind)] == [0, 3, 99]
    finally:
        db.close()


@pytest.mark.parametrize("outer_kind", OUTERS)
def test_nested_list_explicit_compact(path, outer_kind):
    """compact() rebuilds the inner list — its new length/blocks must
    propagate to the parent too (same gap as __delitem__)."""
    with DB(path) as db:
        tmpl = List.template(db.create_dataset("inner_ds", n="int64"))
        outer = _make_outer(db, outer_kind, tmpl)
        inner = _inner(outer, outer_kind, create=True)
        for i in range(10):
            inner.append({"n": i})
        for idx in (7, 5, 3, 1):          # delete a few (no auto-compact yet)
            del inner[idx]
        _inner(outer, outer_kind).compact()
        survivors = [r["n"] for r in _inner(outer, outer_kind)]
        assert survivors == [0, 2, 4, 6, 8, 9]
        assert len(_inner(outer, outer_kind)) == 6

    db, outer = _reopen_outer(path, outer_kind)
    try:
        inner = _inner(outer, outer_kind)
        assert len(inner) == 6
        assert [r["n"] for r in inner] == [0, 2, 4, 6, 8, 9]
    finally:
        db.close()
