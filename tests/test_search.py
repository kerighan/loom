"""SearchIndex — persistent inverted index with boolean queries (via eldar).

The fuzz test uses an in-memory `eldar.Index` built on the same documents as
the oracle: loom delegates parsing + set-algebra to eldar and stores the same
postings eldar would build, so `SearchIndex.search(q)` must return *exactly*
the same doc-ids as `eldar.Index.search(q)` for any boolean query — including
across a close/reopen.
"""

import os
import random
import tempfile

import pytest

from loom.database import DB

eldar = pytest.importorskip("eldar")  # SearchIndex needs eldar


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


DOCS = [
    {"title": "Fast search engine", "body": "an inverted index for full text"},
    {"title": "Slow database", "body": "a slow full table scan approach"},
    {"title": "Python tips", "body": "python search and indexing techniques"},
    {"title": "Java guide", "body": "java programming, no python here"},
]


class TestSearchBasics:
    def test_boolean_ops(self, db):
        idx = db.create_search_index("docs", text_fields=["title", "body"])
        idx.add_many(DOCS)
        s = lambda q: idx.search(q, return_ids=True)
        assert s("search") == [0, 2]
        assert s("python AND search") == [2]
        assert s("java OR slow") == [1, 3]
        assert s("full AND NOT slow") == [0]
        assert s("(python OR java) AND NOT search") == [3]
        assert s("index*") == [0, 2]            # inverted / index / indexing
        assert s("missingword") == []

    def test_returns_documents(self, db):
        idx = db.create_search_index("docs", text_fields=["title", "body"])
        idx.add_many(DOCS)
        res = idx.search("python AND search")
        assert res == [{"title": "Python tips",
                        "body": "python search and indexing techniques"}]

    def test_string_documents(self, db):
        idx = db.create_search_index("docs")
        idx.add("the quick brown fox")
        idx.add("the lazy dog")
        assert idx.search("quick", return_ids=True) == [0]
        assert idx.search("the", return_ids=True) == [0, 1]
        assert idx.get_document(0) == "the quick brown fox"

    def test_delete_tombstone(self, db):
        idx = db.create_search_index("docs", text_fields=["title", "body"])
        idx.add_many(DOCS)
        assert idx.search("search", return_ids=True) == [0, 2]
        idx.delete(0)
        assert idx.search("search", return_ids=True) == [2]
        assert 0 not in idx
        assert len(idx) == 3

    def test_document_frequency_and_limit(self, db):
        idx = db.create_search_index("docs", text_fields=["title", "body"])
        idx.add_many(DOCS)
        assert idx.document_frequency("full") == 2
        assert idx.document_frequency("python") == 2
        assert idx.document_frequency("absent") == 0
        assert idx.search("python OR search OR full", return_ids=True, limit=2) == [0, 1]

    def test_survives_reopen(self, db):
        idx = db.create_search_index("docs", text_fields=["title", "body"])
        idx.add_many(DOCS)
        idx.delete(1)
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            idx2 = db2._datastructures["docs"]
            assert idx2.search("full", return_ids=True) == [0]   # 1 tombstoned
            assert idx2.search("python", return_ids=True) == [2, 3]
            assert idx2.get_document(2)["title"] == "Python tips"
            # new docs keep increasing ids after reopen
            new_id = idx2.add({"title": "extra", "body": "python addendum"})
            assert new_id == 4
            assert idx2.search("python", return_ids=True) == [2, 3, 4]
        finally:
            db2.close()


# ── Fuzz against an in-memory eldar.Index oracle ─────────────────────────────

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "search", "python"]


def _rand_doc(rng):
    n = rng.randint(0, 6)
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def _rand_query(rng, depth=0):
    if depth >= 2 or rng.random() < 0.45:
        term = rng.choice(VOCAB)
        if rng.random() < 0.15:                       # sometimes a prefix wildcard
            term = term[: rng.randint(1, len(term))] + "*"
        return term
    op = rng.choice([" AND ", " OR ", " AND NOT "])
    expr = f"{_rand_query(rng, depth + 1)}{op}{_rand_query(rng, depth + 1)}"
    return f"({expr})" if rng.random() < 0.5 else expr


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_fuzz_matches_eldar_oracle(seed):
    rng = random.Random(seed)
    texts = [_rand_doc(rng) for _ in range(60)]

    # Oracle: eldar's own in-memory inverted index, same docs, same order → ids align.
    oracle = eldar.Index()
    oracle.build(texts)

    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)

    def check(idx, after=""):
        for _ in range(60):
            q = _rand_query(rng)
            try:
                expected = set(oracle.search(q, return_ids=True))
            except Exception:
                continue  # malformed query → skip (both would raise the same way)
            got = set(idx.search(q, return_ids=True))
            assert got == expected, f"{after}query {q!r}: loom {got} != eldar {expected}"

    try:
        db = DB(path)
        idx = db.create_search_index("fuzz")
        for t in texts:
            idx.add(t)
        check(idx)
        db.close()

        # Same must hold from a cold reopen (postings read back from disk).
        db = DB(path)
        try:
            check(db._datastructures["fuzz"], after="(reopened) ")
        finally:
            db.close()
    finally:
        os.unlink(path)
