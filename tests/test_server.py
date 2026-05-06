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
