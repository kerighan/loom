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


class TestBM25:
    def test_ranking_by_tf_and_length(self, db):
        idx = db.create_search_index("docs", scoring="bm25")
        idx.add("python python python")             # 0: tf=3, short
        idx.add("python and some other words here")  # 1: tf=1, longer
        idx.add("python python in a medium doc ok")  # 2: tf=2
        idx.add("java only no snake here at all")     # 3: no python
        ranked = idx.search("python", return_ids=True)
        assert ranked == [0, 2, 1]                    # tf 3 > 2 > 1; 3 excluded

    def test_boolean_mode_on_scored_index(self, db):
        idx = db.create_search_index("docs", scoring="bm25")
        idx.add_many(["python python", "a python here", "no snake"])
        assert idx.search("python", return_ids=True, mode="boolean") == [0, 1]

    def test_with_scores(self, db):
        idx = db.create_search_index("docs", scoring="bm25")
        idx.add_many(["python python python", "python once"])
        res = idx.search("python", return_ids=True, with_scores=True)
        assert [i for i, _ in res] == [0, 1]
        assert res[0][1] >= res[1][1] > 0             # scores descending, positive

    def test_tfidf_mode(self, db):
        idx = db.create_search_index("docs", scoring="bm25")
        idx.add_many(["python python python", "python x y z w", "java"])
        assert idx.search("python", return_ids=True, mode="tfidf") == [0, 1]

    def test_boolean_index_rejects_ranked(self, db):
        idx = db.create_search_index("docs")  # boolean default
        idx.add("hello world")
        with pytest.raises(ValueError):
            idx.search("hello", mode="bm25")

    def test_scored_survives_reopen(self, db):
        idx = db.create_search_index("docs", scoring="bm25")
        idx.add_many(["python python python", "python here too", "java world"])
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            idx2 = db2._datastructures["docs"]
            assert idx2._scored is True
            assert idx2.search("python", return_ids=True) == [0, 1]
            # ranking still works after reopen (tf + lengths read from disk)
            assert idx2.search("python", return_ids=True, with_scores=True)[0][1] > 0
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


def _bm25_oracle(texts, query_terms, candidates, k1=1.5, b=0.75):
    """Brute-force BM25 ranking of `candidates` for `query_terms`, matching
    SearchIndex._rank exactly (same idf, same formula, same tie-break)."""
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
    rng = random.Random(seed)
    # docs of clean lowercase tokens → tokenisation is a plain split.
    texts = [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 8)))
             for _ in range(80)]
    oracle = eldar.Index()
    oracle.build(texts)  # boolean candidate selection (proven correct)

    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    try:
        db = DB(path)
        idx = db.create_search_index("fuzz", scoring="bm25")
        for t in texts:
            idx.add(t)

        for _ in range(80):
            k = rng.randint(1, 3)
            terms = [rng.choice(VOCAB) for _ in range(k)]
            op = rng.choice([" AND ", " OR "])
            q = op.join(terms)
            candidates = set(oracle.search(q, return_ids=True))
            expected = _bm25_oracle(texts, terms, candidates)
            got = idx.search(q, mode="bm25", return_ids=True)
            assert got == expected, f"query {q!r}: loom {got} != brute {expected}"
        db.close()
    finally:
        os.unlink(path)
