"""SearchIndex — persistent inverted index over a user Dataset.

The index is built on a documents Dataset; add(record) inserts the record and
indexes its text fields, keeping only the record's address.  Query parsing +
set-algebra come from loom's vendored copy of eldar's machinery — eldar is a
TEST-ONLY dependency here.

The fuzz tests use an in-memory `eldar.Index` built on the same texts as the
oracle (skipped if eldar isn't installed): loom stores the same postings and
vendors the same algebra, so `SearchIndex.search(q)` must return exactly the
same doc-ids as `eldar.Index.search(q)` for any boolean query — including
after a reopen.
"""

import os
import random
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


DOCS = [
    {"title": "Fast search engine", "body": "an inverted index for full text"},
    {"title": "Slow database", "body": "a slow full table scan approach"},
    {"title": "Python tips", "body": "python search and indexing techniques"},
    {"title": "Java guide", "body": "java programming, no python here"},
]


def make_index(db, name="docs", scoring="boolean"):
    ds = db.create_dataset(f"_ds_{name}", title="utf8[120]", body="text")
    return db.create_search_index(
        name, ds, text_fields=["title", "body"], scoring=scoring
    )


def make_body_index(db, name="docs", scoring="boolean"):
    ds = db.create_dataset(f"_ds_{name}", body="text")
    return db.create_search_index(name, ds, scoring=scoring)  # text_fields → ["body"]


class TestSearchBasics:
    def test_boolean_ops(self, db):
        idx = make_index(db)
        idx.add_many(DOCS)
        s = lambda q: idx.search(q, return_ids=True)
        assert s("search") == [0, 2]
        assert s("python AND search") == [2]
        assert s("java OR slow") == [1, 3]
        assert s("full AND NOT slow") == [0]
        assert s("(python OR java) AND NOT search") == [3]
        assert s("index*") == [0, 2]
        assert s("missingword") == []

    def test_returns_documents(self, db):
        idx = make_index(db)
        idx.add_many(DOCS)
        res = idx.search("python AND search")
        assert res == [{"title": "Python tips",
                        "body": "python search and indexing techniques"}]

    def test_indexes_a_user_dataset(self, db):
        idx = make_index(db)
        idx.add_many(DOCS)
        # documents live in the user dataset; the index only stores addresses
        assert idx._dataset.name == "_ds_docs"
        assert idx.get_document(2)["title"] == "Python tips"

    def test_body_only(self, db):
        idx = make_body_index(db)
        idx.add({"body": "the quick brown fox"})
        idx.add({"body": "the lazy dog"})
        assert idx.search("quick", return_ids=True) == [0]
        assert idx.search("the", return_ids=True) == [0, 1]
        assert idx.get_document(0) == {"body": "the quick brown fox"}

    def test_delete_tombstone(self, db):
        idx = make_index(db)
        idx.add_many(DOCS)
        assert idx.search("search", return_ids=True) == [0, 2]
        idx.delete(0)
        assert idx.search("search", return_ids=True) == [2]
        assert 0 not in idx
        assert len(idx) == 3

    def test_document_frequency_and_limit(self, db):
        idx = make_index(db)
        idx.add_many(DOCS)
        assert idx.document_frequency("full") == 2
        assert idx.document_frequency("python") == 2
        assert idx.document_frequency("absent") == 0
        assert idx.search("python OR search OR full", return_ids=True, limit=2) == [0, 1]

    def test_survives_reopen(self, db):
        idx = make_index(db)
        idx.add_many(DOCS)
        idx.delete(1)
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            idx2 = db2._datastructures["docs"]
            assert idx2.search("full", return_ids=True) == [0]
            assert idx2.search("python", return_ids=True) == [2, 3]
            assert idx2.get_document(2)["title"] == "Python tips"
            new_id = idx2.add({"title": "extra", "body": "python addendum"})
            assert new_id == 4
            assert idx2.search("python", return_ids=True) == [2, 3, 4]
        finally:
            db2.close()


class TestBM25:
    def test_ranking_by_tf_and_length(self, db):
        idx = make_body_index(db, scoring="bm25")
        idx.add_many([{"body": t} for t in [
            "python python python",
            "python and some other words here",
            "python python in a medium doc ok",
            "java only no snake here at all",
        ]])
        assert idx.search("python", return_ids=True) == [0, 2, 1]

    def test_boolean_mode_on_scored_index(self, db):
        idx = make_body_index(db, scoring="bm25")
        idx.add_many([{"body": t} for t in ["python python", "a python here", "no snake"]])
        assert idx.search("python", return_ids=True, mode="boolean") == [0, 1]

    def test_with_scores(self, db):
        idx = make_body_index(db, scoring="bm25")
        idx.add_many([{"body": t} for t in ["python python python", "python once"]])
        res = idx.search("python", return_ids=True, with_scores=True)
        assert [i for i, _ in res] == [0, 1]
        assert res[0][1] >= res[1][1] > 0

    def test_tfidf_mode(self, db):
        idx = make_body_index(db, scoring="bm25")
        idx.add_many([{"body": t} for t in ["python python python", "python x y z w", "java"]])
        assert idx.search("python", return_ids=True, mode="tfidf") == [0, 1]

    def test_boolean_index_rejects_ranked(self, db):
        idx = make_body_index(db)  # boolean
        idx.add({"body": "hello world"})
        with pytest.raises(ValueError):
            idx.search("hello", mode="bm25")

    def test_scored_survives_reopen(self, db):
        idx = make_body_index(db, scoring="bm25")
        idx.add_many([{"body": t} for t in
                      ["python python python", "python here too", "java world"]])
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            idx2 = db2._datastructures["docs"]
            assert idx2._scored is True
            assert idx2.search("python", return_ids=True) == [0, 1]
            assert idx2.search("python", return_ids=True, with_scores=True)[0][1] > 0
        finally:
            db2.close()


# ── Fuzz against an in-memory eldar.Index oracle ─────────────────────────────

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
         "golf", "hotel", "india", "juliet", "search", "python"]


def _rand_doc(rng):
    return " ".join(rng.choice(VOCAB) for _ in range(rng.randint(0, 6)))


def _rand_query(rng, depth=0):
    if depth >= 2 or rng.random() < 0.45:
        term = rng.choice(VOCAB)
        if rng.random() < 0.15:
            term = term[: rng.randint(1, len(term))] + "*"
        return term
    op = rng.choice([" AND ", " OR ", " AND NOT "])
    expr = f"{_rand_query(rng, depth + 1)}{op}{_rand_query(rng, depth + 1)}"
    return f"({expr})" if rng.random() < 0.5 else expr


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_fuzz_matches_eldar_oracle(seed):
    eldar = pytest.importorskip("eldar")   # oracle only — loom itself needs no eldar
    rng = random.Random(seed)
    texts = [_rand_doc(rng) for _ in range(60)]
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
                continue
            got = set(idx.search(q, return_ids=True))
            assert got == expected, f"{after}query {q!r}: loom {got} != eldar {expected}"

    try:
        db = DB(path)
        ds = db.create_dataset("d", body="text")
        idx = db.create_search_index("fuzz", ds)
        idx.add_many([{"body": t} for t in texts])
        check(idx)
        db.close()
        db = DB(path)
        try:
            check(db._datastructures["fuzz"], after="(reopened) ")
        finally:
            db.close()
    finally:
        os.unlink(path)


def _bm25_oracle(texts, query_terms, candidates, k1=1.5, b=0.75):
    import math

    docs = [t.split() for t in texts]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N
    terms = sorted(set(query_terms))
    df = {t: sum(1 for d in docs if t in d) for t in terms}

    scores = {i: 0.0 for i in candidates}
    for t in terms:
        if df[t] == 0:
            continue
        idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
        for i in candidates:
            f = docs[i].count(t)
            if f == 0:
                continue
            dl = len(docs[i])
            scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


@pytest.mark.parametrize("seed", [0, 3, 11, 99])
def test_fuzz_bm25_matches_brute_force(seed):
    eldar = pytest.importorskip("eldar")   # candidate-set oracle only
    rng = random.Random(seed)
    texts = [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 8)))
             for _ in range(80)]
    oracle = eldar.Index()
    oracle.build(texts)

    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    try:
        db = DB(path)
        ds = db.create_dataset("d", body="text")
        idx = db.create_search_index("fuzz", ds, scoring="bm25")
        idx.add_many([{"body": t} for t in texts])

        for _ in range(80):
            terms = [rng.choice(VOCAB) for _ in range(rng.randint(1, 3))]
            q = rng.choice([" AND ", " OR "]).join(terms)
            candidates = set(oracle.search(q, return_ids=True))
            expected = _bm25_oracle(texts, terms, candidates)
            got = idx.search(q, mode="bm25", return_ids=True)
            assert got == expected, f"query {q!r}: loom {got} != brute {expected}"
        db.close()
    finally:
        os.unlink(path)
