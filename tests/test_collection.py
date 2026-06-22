"""Collection — declarative typed indexes, driven by the social-media posts
use case: post by id (primary), a user's posts recent-first (many), posts with
engagement >= x (range), atomic like increment, and the Record wrapper.
"""

import os
import tempfile

import pytest

from loom.database import DB
from loom import Many, Range, Unique


SCHEMA = {
    "id": "utf8[16]",
    "username": "utf8[32]",
    "created_at": "int64",
    "engagement": "int64",
    "likes": "int64",
    "text": "text",
}


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


def make_posts(db):
    return db.collection("posts", SCHEMA, indexes={
        "id": "primary",
        "username": Many(sort="created_at", desc=True),
        "engagement": "range",
    })


def seed(posts):
    posts.insert_many([
        {"id": "p1", "username": "alice", "created_at": 100, "engagement": 5,    "likes": 0, "text": "hello python"},
        {"id": "p2", "username": "alice", "created_at": 300, "engagement": 50,   "likes": 0, "text": "more python"},
        {"id": "p3", "username": "alice", "created_at": 200, "engagement": 999,  "likes": 0, "text": "viral post"},
        {"id": "p4", "username": "bob",   "created_at": 150, "engagement": 1200, "likes": 0, "text": "bob speaks"},
        {"id": "p5", "username": "bob",   "created_at": 400, "engagement": 30,   "likes": 0, "text": "bob again"},
    ])
    return posts


class TestPrimary:
    def test_get_by_id(self, db):
        posts = seed(make_posts(db))
        assert posts["p3"]["text"] == "viral post"
        assert posts["p3"]["username"] == "alice"
        assert "p3" in posts and "px" not in posts
        assert len(posts) == 5

    def test_missing_id_raises(self, db):
        posts = seed(make_posts(db))
        with pytest.raises(KeyError):
            posts["nope"]


class TestMany:
    def test_users_posts_recent_first(self, db):
        posts = seed(make_posts(db))
        assert [p["id"] for p in posts.find("username", "alice")] == ["p2", "p3", "p1"]
        assert [p["id"] for p in posts.find("username", "bob")] == ["p5", "p4"]

    def test_limit(self, db):
        posts = seed(make_posts(db))
        assert [p["id"] for p in posts.find("username", "alice", limit=2)] == ["p2", "p3"]

    def test_unknown_value_empty(self, db):
        posts = seed(make_posts(db))
        assert posts.find("username", "carol") == []


class TestRange:
    def test_engagement_threshold(self, db):
        posts = seed(make_posts(db))
        assert [p["id"] for p in posts.range("engagement", 100, None)] == ["p3", "p4"]

    def test_bounded_range(self, db):
        posts = seed(make_posts(db))
        assert [p["id"] for p in posts.range("engagement", 30, 999)] == ["p5", "p2", "p3"]

    def test_open_low(self, db):
        posts = seed(make_posts(db))
        assert [p["id"] for p in posts.range("engagement", None, 50)] == ["p1", "p5", "p2"]


class TestUnique:
    def test_unique_lookup_and_enforcement(self, db):
        users = db.collection("users", {"id": "utf8[16]", "email": "utf8[40]"},
                              indexes={"id": "primary", "email": "unique"})
        users.insert({"id": "u1", "email": "a@x.com"})
        users.insert({"id": "u2", "email": "b@x.com"})
        assert users.get("email", "a@x.com")["id"] == "u1"
        assert users.get("email", "missing@x.com") is None
        with pytest.raises(ValueError):
            users.insert({"id": "u3", "email": "a@x.com"})


class TestMutations:
    def test_increment_indexed_field_reindexes(self, db):
        posts = seed(make_posts(db))
        posts.increment("p1", "engagement", 2000)
        assert [p["id"] for p in posts.range("engagement", 2000, None)] == ["p1"]
        assert posts["p1"]["engagement"] == 2005

    def test_increment_unindexed_field_fast(self, db):
        posts = seed(make_posts(db))
        for _ in range(3):
            posts.increment("p1", "likes")
        assert posts["p1"]["likes"] == 3

    def test_record_wrapper_setitem_reindexes(self, db):
        posts = seed(make_posts(db))
        p = posts["p2"]
        p["engagement"] = 5000
        assert [r["id"] for r in posts.range("engagement", 5000, None)] == ["p2"]
        assert posts["p2"]["engagement"] == 5000

    def test_update_moves_in_many_index(self, db):
        posts = seed(make_posts(db))
        posts.update("p1", username="bob", created_at=500)
        assert "p1" not in [p["id"] for p in posts.find("username", "alice")]
        assert [p["id"] for p in posts.find("username", "bob")] == ["p1", "p5", "p4"]

    def test_delete_removes_from_all_indexes(self, db):
        posts = seed(make_posts(db))
        posts.delete("p3")
        assert "p3" not in posts
        assert [p["id"] for p in posts.find("username", "alice")] == ["p2", "p1"]
        assert "p3" not in [p["id"] for p in posts.range("engagement", 100, None)]

    def test_cannot_change_primary_key(self, db):
        posts = seed(make_posts(db))
        with pytest.raises(ValueError):
            posts.update("p1", id="pX")

    def test_insert_existing_pk_upserts_and_reindexes(self, db):
        posts = seed(make_posts(db))
        # re-insert p1 with changed indexed fields → record replaced, indexes clean
        posts.insert({"id": "p1", "username": "bob", "created_at": 500,
                      "engagement": 7777, "likes": 0, "text": "moved"})
        assert len(posts) == 5                                  # no duplicate record
        assert posts["p1"]["username"] == "bob"
        assert "p1" not in [p["id"] for p in posts.find("username", "alice")]
        assert "p1" in [p["id"] for p in posts.find("username", "bob")]
        # range: p1 appears exactly once, at its new engagement
        eng = [(p["id"], p["engagement"]) for p in posts.range("engagement", 5000, None)]
        assert eng == [("p1", 7777)]

    def test_insert_many_upserts_and_dedups_batch(self, db):
        posts = seed(make_posts(db))
        posts.insert_many([
            {"id": "p2", "username": "carol", "created_at": 9, "engagement": 10, "likes": 0, "text": "x"},
            {"id": "p9", "username": "dave",  "created_at": 9, "engagement": 20, "likes": 0, "text": "y"},
            {"id": "p9", "username": "dave2", "created_at": 9, "engagement": 25, "likes": 0, "text": "z"},  # dup → last wins
        ])
        assert len(posts) == 6                                  # p2 updated, p9 added once
        assert posts["p2"]["username"] == "carol"
        assert posts["p9"]["username"] == "dave2"
        assert "p2" not in [p["id"] for p in posts.find("username", "alice")]  # old value gone
        assert [p["id"] for p in posts.find("username", "dave")] == []          # overwritten in-batch
        # no duplicate range entries for the upserted/added pks
        ids = [p["id"] for p in posts.range("engagement", None, None)]
        assert len(ids) == len(set(ids)) == 6


class TestPersistence:
    def test_reopen(self, db):
        seed(make_posts(db))
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            posts = db2.collection("posts")
            assert posts["p3"]["text"] == "viral post"
            assert [p["id"] for p in posts.find("username", "alice")] == ["p2", "p3", "p1"]
            assert [p["id"] for p in posts.range("engagement", 100, None)] == ["p3", "p4"]
            posts.insert({"id": "p6", "username": "alice", "created_at": 500,
                          "engagement": 7, "likes": 0, "text": "new"})
            assert [p["id"] for p in posts.find("username", "alice")][0] == "p6"
        finally:
            db2.close()

    def test_compound_equality_and_range(self, db):
        # category (equality) + created_at (range) via a Many(field=, sort=) index
        posts = db.collection("p", {"post_id": "utf8[8]", "cat": "utf8[8]", "ts": "int64"},
                              indexes={
                                  "post_id": "primary",
                                  "by_eng": Many(field="cat", sort="post_id"),       # same field, another name
                                  "cat_time": Many(field="cat", sort="ts", desc=True),
                              })
        for i in range(20):
            posts.insert({"post_id": f"p{i:02d}", "cat": "a" if i % 2 else "b", "ts": i})
        # cat == "a" AND ts >= 10, recent first
        res = posts.find("cat_time", "a", start=10)
        assert all(r["cat"] == "a" and r["ts"] >= 10 for r in res)
        assert [r["ts"] for r in res] == sorted([r["ts"] for r in res], reverse=True)
        # closed window
        res2 = posts.find("cat_time", "a", start=4, end=12)
        assert all(r["cat"] == "a" and 4 <= r["ts"] <= 12 for r in res2)
        # two indexes on the same field coexist
        assert {r["post_id"] for r in posts.find("by_eng", "b")} == \
               {f"p{i:02d}" for i in range(20) if i % 2 == 0}

    def test_find_bounds_require_sort(self, db):
        posts = db.collection("p", {"id": "utf8[8]", "cat": "utf8[8]"},
                              indexes={"id": "primary", "cat": Many()})  # no sort
        posts.insert({"id": "p1", "cat": "a"})
        with pytest.raises(ValueError):
            posts.find("cat", "a", start="x")

    def test_reindex_rebuilds(self, db):
        posts = seed(make_posts(db))
        posts.reindex()
        assert [p["id"] for p in posts.find("username", "alice")] == ["p2", "p3", "p1"]
        assert len(posts) == 5


from loom import Search


def make_articles(db, scoring="bm25"):
    schema = {"id": "utf8[16]", "author": "utf8[24]",
              "title": "text", "body": "text"}
    return db.collection("articles", schema, indexes={
        "id": "primary",
        "author": Many(),
        "content": Search(fields=["title", "body"], scoring=scoring),
    })


ARTICLES = [
    {"id": "a1", "author": "alice", "title": "Fast search", "body": "an inverted index for python"},
    {"id": "a2", "author": "bob",   "title": "Slow database", "body": "a full table scan in java"},
    {"id": "a3", "author": "alice", "title": "Python tips", "body": "python search and indexing"},
]


class TestSearch:
    def test_multifield_boolean(self, db):
        c = make_articles(db)
        c.insert_many(ARTICLES)
        assert sorted(r["id"] for r in c.search("content", "python")) == ["a1", "a3"]
        assert sorted(r["id"] for r in c.search("content", "fast")) == ["a1"]        # title field
        assert sorted(r["id"] for r in c.search("content", "python AND search")) == ["a1", "a3"]
        assert sorted(r["id"] for r in c.search("content", "index AND NOT java")) == ["a1"]
        assert sorted(r["id"] for r in c.search("content", "index* AND NOT java")) == ["a1", "a3"]
        assert c.search("content", "nonexistent") == []

    def test_bm25_ranked_with_scores(self, db):
        c = make_articles(db, scoring="bm25")
        c.insert_many(ARTICLES)
        res = c.search("content", "python", with_scores=True)
        assert [r["id"] for r, _ in res] == ["a3", "a1"]   # a3 mentions python twice
        assert res[0][1] >= res[1][1] > 0
        assert all(hasattr(r, "pk") for r, _ in res)        # records are Collection Records

    def test_results_are_records(self, db):
        c = make_articles(db)
        c.insert_many(ARTICLES)
        r = c.search("content", "java")[0]
        assert r["id"] == "a2" and r["author"] == "bob"     # full record via primary

    def test_update_reindexes_text(self, db):
        c = make_articles(db)
        c.insert_many(ARTICLES)
        c.update("a2", body="now mentions python too")
        assert sorted(r["id"] for r in c.search("content", "python")) == ["a1", "a2", "a3"]

    def test_delete_removes_from_search(self, db):
        c = make_articles(db)
        c.insert_many(ARTICLES)
        c.delete("a1")
        assert sorted(r["id"] for r in c.search("content", "python")) == ["a3"]

    def test_survives_reopen(self, db):
        make_articles(db).insert_many(ARTICLES)
        path = db.filename
        db.close()
        db2 = DB(path)
        try:
            c = db2.collection("articles")
            assert sorted(r["id"] for r in c.search("content", "python")) == ["a1", "a3"]
            # writes after reopen stay indexed
            c.insert({"id": "a4", "author": "carol", "title": "More python", "body": "x"})
            assert sorted(r["id"] for r in c.search("content", "python")) == ["a1", "a3", "a4"]
        finally:
            db2.close()

    def test_survives_repeated_reopen(self):
        # Regression: SearchIndex._from_registry_params used to pop("dataset_name")
        # from the shared registry dict; close() then persisted a params dict
        # missing it → the *second* reopen raised KeyError('dataset_name').
        import os, tempfile
        path = os.path.join(tempfile.mkdtemp(), "art.db")
        with DB(path) as db:
            make_articles(db).insert_many(ARTICLES)
        for _ in range(3):
            with DB(path) as db:
                c = db["articles"]
                assert sorted(r["id"] for r in c.search("content", "python")) == ["a1", "a3"]
