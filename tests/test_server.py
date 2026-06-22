"""Smoke tests for loom.server (FastAPI HTTP layer)."""

import os
import tempfile
from unittest.mock import Mock
import sys

import pytest
from pydantic import BaseModel, Field

from loom import DB

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


class User(BaseModel):
    id: int
    username: str = Field(max_length=50)
    age: int
    premium: bool


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "srv.db")
    db = DB(path)
    yield db
    db.close()


def test_index_lists_resources(db):
    db.create_dataset("users", User)
    db.create_set("active")
    client = TestClient(db.fastapi_app())

    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["structures"]["active"] == "Set"
    # Index should advertise the interactive docs.
    assert body["docs"]["swagger_ui"] == "/docs"
    assert body["docs"]["redoc"] == "/redoc"
    assert body["docs"]["openapi_schema"] == "/openapi.json"


def test_index_lists_dashboard_doc_when_enabled(db):
    db.create_dataset("users", User)
    client = TestClient(db.fastapi_app(dashboard=True))

    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"]["dashboard"] == "/dashboard"


def test_interactive_docs_are_served(db):
    """FastAPI exposes Swagger UI, ReDoc and the OpenAPI schema by default."""
    db.create_dataset("users", User)
    client = TestClient(db.fastapi_app())

    # Swagger UI HTML
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower()

    # ReDoc HTML
    r = client.get("/redoc")
    assert r.status_code == 200
    assert "redoc" in r.text.lower()

    # OpenAPI 3 schema
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["openapi"].startswith("3.")
    assert "/datasets/{name}/records" not in spec["paths"]
    assert "/dicts/{name}/items/{key}" in spec["paths"]


def test_dict_crud(db):
    users_ds = db.create_dataset("users", User)
    db.create_dict("by_username", users_ds)

    client = TestClient(db.fastapi_app())

    r = client.put(
        "/dicts/by_username/items/alice",
        json={"id": 1, "username": "alice", "age": 30, "premium": True},
    )
    assert r.status_code == 200

    r = client.get("/dicts/by_username/items/alice")
    assert r.status_code == 200
    assert r.json()["age"] == 30

    r = client.get("/dicts/by_username/keys")
    assert "alice" in r.json()

    r = client.delete("/dicts/by_username/items/alice")
    assert r.status_code == 204

    r = client.get("/dicts/by_username/items/alice")
    assert r.status_code == 404


def test_read_only_api_hides_write_endpoints_and_rejects_mutations():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "srv_ro.db")

    with DB(path) as writable:
        users_ds = writable.create_dataset("users", User)
        by_username = writable.create_dict("by_username", users_ds)
        by_username["alice"] = {
            "id": 1,
            "username": "alice",
            "age": 30,
            "premium": True,
        }
        writable.create_set("active")

    db_ro = DB(path, flag="r")
    try:
        client = TestClient(db_ro.fastapi_app())

        index = client.get("/")
        assert index.status_code == 200
        assert index.json()["read_only"] is True

        spec = client.get("/openapi.json").json()
        dict_item_path = spec["paths"]["/dicts/{name}/items/{key}"]
        assert "get" in dict_item_path
        assert "put" not in dict_item_path
        assert "delete" not in dict_item_path

        r = client.get("/dicts/by_username/items/alice")
        assert r.status_code == 200
        assert r.json()["username"] == "alice"

        r = client.put(
            "/dicts/by_username/items/bob",
            json={"id": 2, "username": "bob", "age": 31, "premium": False},
        )
        assert r.status_code == 403
        assert "read-only" in r.json()["detail"]

        assert client.post("/sets/active/members", json={"item": "alice"}).status_code == 403
    finally:
        db_ro.close()


def test_bloom(db):
    db.create_bloomfilter("seen", expected_items=100)
    client = TestClient(db.fastapi_app())

    meta = client.get("/bloomfilters/seen")
    assert meta.status_code == 200
    assert meta.json()["expected_items"] == 100

    assert (
        client.post("/bloomfilters/seen/items", json={"item": "u1"}).status_code == 201
    )
    assert client.get("/bloomfilters/seen/contains/u1").json()["contains"] is True


def test_set(db):
    db.create_set("tags")
    client = TestClient(db.fastapi_app())

    assert client.post("/sets/tags/members", json={"item": "python"}).status_code == 201
    assert client.get("/sets/tags/contains/python").json()["contains"] is True
    assert client.get("/sets/tags/contains/rust").json()["contains"] is False
    assert "python" in client.get("/sets/tags/members").json()
    assert client.delete("/sets/tags/members/python").status_code == 204


def test_queue(db):
    class Job(BaseModel):
        id: int
        task: str = Field(max_length=20)

    db.create_queue("jobs", Job)

    client = TestClient(db.fastapi_app())

    for i in range(3):
        r = client.post("/queues/jobs/push", json={"id": i, "task": f"t{i}"})
        assert r.status_code == 201

    assert client.get("/queues/jobs").json()["length"] == 3
    assert client.get("/queues/jobs/peek").json()["id"] == 0
    assert client.post("/queues/jobs/pop").json()["id"] == 0
    assert client.get("/queues/jobs").json()["length"] == 2


def test_btree_range(db):
    class Item(BaseModel):
        price: float

    item_ds = db.create_dataset("items", Item)
    db.create_btree("by_name", item_ds, key_size=20)

    client = TestClient(db.fastapi_app())

    for k, p in [("apple", 1.0), ("banana", 0.5), ("cherry", 2.0)]:
        client.put(f"/btrees/by_name/items/{k}", json={"price": p})

    r = client.get("/btrees/by_name/range", params={"start": "a", "end": "c"})
    keys = [row["key"] for row in r.json()]
    assert keys == ["apple", "banana"]


def test_nested_returns_422(db):
    from loom.datastructures import List

    user_ds = db.create_dataset("users", User)
    PostFeed = List.template(user_ds)
    db.create_dict("feeds", PostFeed)

    client = TestClient(db.fastapi_app())
    r = client.get("/dicts/feeds/items/alice")
    assert r.status_code == 422


def test_dashboard_route_is_served_when_enabled(db):
    db.create_dataset("users", User)
    client = TestClient(db.fastapi_app(dashboard=True))

    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "loom dashboard" in r.text.lower()


def test_auth_token_protects_api_docs_and_dashboard(db):
    db.create_dataset("users", User)
    db.create_set("active")
    client = TestClient(db.fastapi_app(dashboard=True, auth_token="secret-token"))

    assert client.get("/").status_code == 401
    assert client.get("/docs").status_code == 401
    assert client.get("/dashboard").status_code == 401
    assert "api token" in client.get("/dashboard").text.lower()

    r = client.get("/", headers={"Authorization": "Bearer secret-token"})
    assert r.status_code == 200
    assert r.json()["structures"]["active"] == "Set"

    r = client.get("/dashboard", headers={"X-API-Key": "secret-token"})
    assert r.status_code == 200
    assert "loom dashboard" in r.text.lower()


def test_auth_token_query_sets_dashboard_cookie(db):
    db.create_dataset("users", User)
    client = TestClient(db.fastapi_app(dashboard=True, auth_token="secret-token"))

    r = client.get("/dashboard?token=secret-token")
    assert r.status_code == 200
    assert "loom_auth" in r.cookies

    r = client.get("/")
    assert r.status_code == 200


def test_dashboard_login_sets_auth_cookie(db):
    db.create_dataset("users", User)
    client = TestClient(db.fastapi_app(dashboard=True, auth_token="secret-token"))

    r = client.post(
        "/dashboard/login",
        content="token=wrong",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 401

    r = client.post(
        "/dashboard/login",
        content="token=secret-token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "loom_auth" in r.cookies

    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "loom dashboard" in r.text.lower()


def test_empty_auth_token_is_rejected(db):
    with pytest.raises(ValueError):
        db.fastapi_app(auth_token="")


def test_search_index_crud_and_query(db):
    docs = db.create_dataset("docs", title="utf8[60]", body="text")
    db.create_search_index("idx", docs, text_fields=["title", "body"], scoring="bm25")

    client = TestClient(db.fastapi_app())

    meta = client.get("/search_indexes/idx")
    assert meta.status_code == 200
    assert meta.json()["scoring"] == "bm25"
    assert meta.json()["store_documents"] is True

    r = client.post("/search_indexes/idx/documents",
                    json={"title": "Fast search", "body": "an inverted index"})
    assert r.status_code == 201
    i = r.json()["doc_id"]

    client.post("/search_indexes/idx/documents",
                json={"title": "Slow", "body": "sequential disk scan"})
    client.post("/search_indexes/idx/documents",
                json={"title": "Postings", "body": "inverted index postings"})

    # boolean
    r = client.get("/search_indexes/idx/search",
                   params={"q": "inverted AND NOT slow", "mode": "boolean"})
    assert r.status_code == 200
    titles = [d["title"] for d in r.json()]
    assert "Fast search" in titles and "Postings" in titles

    # ranked with scores
    r = client.get("/search_indexes/idx/search",
                   params={"q": "inverted OR index", "with_scores": "true"})
    assert r.status_code == 200
    rows = r.json()
    assert rows and "score" in rows[0] and "document" in rows[0]

    # read a stored document
    assert client.get(f"/search_indexes/idx/documents/{i}").json()["title"] == "Fast search"

    # delete (tombstone) it
    assert client.delete(f"/search_indexes/idx/documents/{i}").status_code == 204
    assert client.get(f"/search_indexes/idx/documents/{i}").status_code == 404

    # appears in the OpenAPI schema
    spec = client.get("/openapi.json").json()
    assert "/search_indexes/{name}/search" in spec["paths"]


def test_collection_crud_find_range_search(db):
    from loom import Many, Search

    posts = db.collection("posts", {
        "id": "utf8[32]", "username": "utf8[30]",
        "created_at": "int64", "engagement": "int64", "text": "text",
    }, indexes={
        "id": "primary",
        "username": Many(sort="created_at", desc=True),
        "engagement": "range",
        "body": Search(fields=["text"], scoring="bm25"),
    })

    client = TestClient(db.fastapi_app())

    # listed at the root
    assert "posts" in client.get("/").json()["collections"]

    meta = client.get("/collections/posts")
    assert meta.status_code == 200
    body = meta.json()
    assert body["primary_field"] == "id"
    assert body["indexes"]["username"] == "many"
    assert body["indexes"]["engagement"] == "range"
    assert body["indexes"]["body"] == "search"

    # insert
    def mk(i, user, ts, eng, text):
        return {"id": i, "username": user, "created_at": ts,
                "engagement": eng, "text": text}

    for rec in [mk("p1", "alice", 170, 9, "an inverted index"),
                mk("p2", "alice", 200, 1500, "fast disk index"),
                mk("p3", "bob", 50, 9000, "sequential scan postings")]:
        r = client.post("/collections/posts/records", json=rec)
        assert r.status_code == 201

    # by primary key
    assert client.get("/collections/posts/records/p1").json()["text"] == "an inverted index"
    assert client.get("/collections/posts/records/missing").status_code == 404

    # find (many → recent first)
    ids = [r["id"] for r in client.get("/collections/posts/find/username",
                                       params={"value": "alice"}).json()]
    assert ids == ["p2", "p1"]

    # range (engagement >= 1000)
    ids = {r["id"] for r in client.get("/collections/posts/range/engagement",
                                       params={"low": 1000}).json()}
    assert ids == {"p2", "p3"}

    # descending range — highest engagement first, no grouping field
    desc = [r["id"] for r in client.get("/collections/posts/range/engagement",
                                        params={"desc": "true", "limit": 2}).json()]
    assert desc == ["p3", "p2"]

    # full-text search
    ids = {r["id"] for r in client.get("/collections/posts/search/body",
                                       params={"q": "inverted OR index"}).json()}
    assert ids == {"p1", "p2"}

    # increment + update
    assert client.post("/collections/posts/records/p1/increment",
                       json={"field": "engagement", "amount": 5}).json()["value"] == 14
    assert client.put("/collections/posts/records/p1",
                      json={"engagement": 7777}).json()["engagement"] == 7777

    # delete
    assert client.delete("/collections/posts/records/p1").status_code == 204
    assert client.get("/collections/posts/records/p1").status_code == 404


def test_collection_read_only_lookup_after_reopen():
    import os, tempfile
    from loom import Many

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "coll_ro.db")
    with DB(path) as w:
        posts = w.collection("posts", {"id": "utf8[16]", "user": "utf8[16]", "ts": "int64"},
                             indexes={"id": "primary", "user": Many(sort="ts")})
        posts.insert({"id": "p1", "user": "alice", "ts": 1})

    ro = DB(path, flag="r")
    try:
        client = TestClient(ro.fastapi_app())
        assert "posts" in client.get("/").json()["collections"]
        assert client.get("/collections/posts/records/p1").json()["user"] == "alice"
        # mutations rejected in read-only mode
        r = client.post("/collections/posts/records",
                        json={"id": "p2", "user": "bob", "ts": 2})
        assert r.status_code == 403
    finally:
        ro.close()


def test_collection_list_endpoint(db):
    from loom import Many

    posts = db.collection("posts", {"id": "utf8[16]", "user": "utf8[16]", "ts": "int64"},
                          indexes={"id": "primary", "user": Many(sort="ts")})
    client = TestClient(db.fastapi_app())
    for i in range(3):
        client.post("/collections/posts/records",
                    json={"id": f"p{i}", "user": "alice", "ts": i})

    rows = client.get("/collections/posts/records", params={"limit": 2}).json()
    assert len(rows) == 2
    assert all("pk" in r and "record" in r for r in rows)


def test_dashboard_supports_search_and_collection(db):
    from loom import Search

    db.create_dataset("docs", title="utf8[40]", body="text")
    db.create_search_index("idx", db.get_dataset("docs"), scoring="bm25")
    db.collection("posts", {"id": "utf8[16]", "text": "text"},
                  indexes={"id": "primary", "body": Search(fields=["text"])})

    client = TestClient(db.fastapi_app(dashboard=True))

    # the dashboard JS knows how to render the new kinds
    html = client.get("/dashboard").text
    assert "renderSearch" in html and "renderCollection" in html

    # the root advertises the SearchIndex (a structure) and the collection
    root = client.get("/").json()
    assert root["structures"]["idx"] == "SearchIndex"
    assert "posts" in root["collections"]


def test_priority_queue_api(db):
    db.create_priority_queue("jobs", {"task": "utf8[40]"})
    client = TestClient(db.fastapi_app())

    meta = client.get("/priority_queues/jobs")
    assert meta.status_code == 200
    assert meta.json()["max_first"] is True

    assert client.post("/priority_queues/jobs/push",
                       json={"item": {"task": "low"}, "priority": 1}).status_code == 201
    client.post("/priority_queues/jobs/push", json={"item": {"task": "high"}, "priority": 9})
    client.post("/priority_queues/jobs/push", json={"item": {"task": "mid"}, "priority": 5})

    assert client.get("/priority_queues/jobs").json()["length"] == 3
    assert client.get("/priority_queues/jobs/peek").json()["task"] == "high"
    assert client.post("/priority_queues/jobs/pop").json()["task"] == "high"
    assert client.get("/priority_queues/jobs").json()["length"] == 2

    # bad body is rejected
    assert client.post("/priority_queues/jobs/push", json={"task": "x"}).status_code == 422

    spec = client.get("/openapi.json").json()
    assert "/priority_queues/{name}/push" in spec["paths"]


def test_priority_queue_dashboard_support(db):
    db.create_priority_queue("jobs", {"task": "utf8[40]"})
    client = TestClient(db.fastapi_app(dashboard=True))
    html = client.get("/dashboard").text
    assert "renderPQ" in html and "priority_queues" in html
    assert client.get("/").json()["structures"]["jobs"] == "PriorityQueue"


def test_collection_datetime_field_over_http(db):
    from datetime import datetime
    from loom import Many

    db.collection("events", {"id": "utf8[16]", "created_at": "datetime", "box": "utf8[8]"},
                  indexes={"id": "primary", "created_at": "range",
                           "box": Many(sort="created_at", desc=True)})
    client = TestClient(db.fastapi_app())

    # schema introspection reports the datetime kind
    assert client.get("/collections/events").json()["schema"]["created_at"] == "datetime"

    for i in range(3):
        r = client.post("/collections/events/records",
                        json={"id": f"e{i}", "created_at": f"2026-01-0{i+1}T00:00:00", "box": "main"})
        assert r.status_code == 201

    # round-trips as ISO 8601
    assert client.get("/collections/events/records/e0").json()["created_at"] == "2026-01-01T00:00:00"

    # descending range (most recent first), bounds parsed from ISO strings
    ids = [r["id"] for r in client.get("/collections/events/range/created_at",
                                       params={"desc": "true", "limit": 2}).json()]
    assert ids == ["e2", "e1"]
    ids = [r["id"] for r in client.get("/collections/events/range/created_at",
                                       params={"low": "2026-01-02T00:00:00"}).json()]
    assert sorted(ids) == ["e1", "e2"]


def test_serve_mounts_optional_dashboard(monkeypatch, db):
    import loom.server as server

    app = Mock()
    uvicorn = Mock()
    build_app = Mock(return_value=app)
    monkeypatch.setattr(server, "build_app", build_app)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    server.serve(
        db,
        host="127.0.0.1",
        port=8123,
        dashboard=True,
        auth_token="secret-token",
    )

    build_app.assert_called_once_with(db, dashboard=True, auth_token="secret-token",
                                       _reopen_lock_on_startup=False)
    uvicorn.run.assert_called_once_with(app, host="127.0.0.1", port=8123, workers=None)
