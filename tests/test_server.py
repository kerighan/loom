"""Smoke tests for loom.server (FastAPI HTTP layer)."""

import os
import tempfile

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
    assert "users" in body["datasets"]
    assert body["structures"]["active"] == "Set"
    # Index should advertise the interactive docs.
    assert body["docs"]["swagger_ui"] == "/docs"
    assert body["docs"]["redoc"] == "/redoc"
    assert body["docs"]["openapi_schema"] == "/openapi.json"


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
    # Every dataset / dict / set / queue / btree route should be in there.
    assert "/datasets/{name}/records" in spec["paths"]


def test_dataset_crud(db):
    users_ds = db.create_dataset("users", User)

    client = TestClient(db.fastapi_app())

    # info
    r = client.get("/datasets/users")
    assert r.status_code == 200
    assert r.json()["schema"]["username"].endswith("U50")

    # insert
    r = client.post(
        "/datasets/users/records",
        json={"id": 1, "username": "alice", "age": 30, "premium": True},
    )
    assert r.status_code == 201
    addr = r.json()["address"]

    # read
    r = client.get(f"/datasets/users/records/{addr}")
    assert r.status_code == 200
    rec = r.json()
    assert rec["username"] == "alice"
    assert rec["premium"] is True

    # invalid: too long username (validation error)
    r = client.post(
        "/datasets/users/records",
        json={"id": 2, "username": "x" * 100, "age": 1, "premium": False},
    )
    assert r.status_code == 422

    # update
    r = client.put(
        f"/datasets/users/records/{addr}",
        json={"id": 1, "username": "alice", "age": 31, "premium": False},
    )
    assert r.status_code == 200

    # delete
    r = client.delete(f"/datasets/users/records/{addr}")
    assert r.status_code == 204


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


def test_bloom(db):
    db.create_bloomfilter("seen", expected_items=100)
    client = TestClient(db.fastapi_app())

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
