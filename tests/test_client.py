import os
import tempfile

import pytest
from pydantic import BaseModel, Field

from loom import (
    DB,
    LoomClient,
    LoomNotFoundError,
    LoomValidationError,
)

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


class User(BaseModel):
    id: int
    username: str = Field(max_length=50)
    age: int
    premium: bool


class Job(BaseModel):
    id: int
    task: str = Field(max_length=20)


@pytest.fixture
def remote_client():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "client.db")
    db = DB(path)
    users_ds = db.create_dataset("users", User)
    items_ds = db.create_dataset("items", User)
    jobs_ds_model = Job
    db.create_dict("by_username", users_ds)
    db.create_list("audit_log", users_ds)
    db.create_set("tags")
    db.create_queue("jobs", jobs_ds_model)
    db.create_btree("by_name", items_ds, key_size=20)
    db.create_bloomfilter("seen", expected_items=100)
    db.create_counting_bloomfilter("seen_counting", expected_items=100)

    app_client = TestClient(db.fastapi_app())
    client = LoomClient("http://testserver", session=app_client, timeout=None)
    try:
        yield client
    finally:
        client.close()
        db.close()


def test_client_exposes_structures_but_not_datasets(remote_client):
    assert hasattr(remote_client, "dicts")
    assert hasattr(remote_client, "queues")
    assert not hasattr(remote_client, "datasets")


def test_remote_dict(remote_client):
    remote_client.dicts["by_username"]["alice"] = {
        "id": 1,
        "username": "alice",
        "age": 30,
        "premium": True,
    }
    assert remote_client.dicts["by_username"]["alice"]["age"] == 30
    assert "alice" in remote_client.dicts["by_username"].keys()


def test_remote_list_and_queue(remote_client):
    remote_client.lists["audit_log"].append(
        {"id": 1, "username": "alice", "age": 30, "premium": True}
    )
    assert remote_client.lists["audit_log"][0]["username"] == "alice"

    remote_client.queues["jobs"].push({"id": 1, "task": "boot"})
    assert remote_client.queues["jobs"].peek()["id"] == 1
    assert remote_client.queues["jobs"].pop()["task"] == "boot"


def test_remote_set_and_bloom_filters(remote_client):
    remote_client.sets["tags"].add("python")
    assert "python" in remote_client.sets["tags"]
    assert "python" in remote_client.sets["tags"].members()

    remote_client.bloomfilters["seen"].add("alice")
    assert "alice" in remote_client.bloomfilters["seen"]

    remote_client.counting_bloomfilters["seen_counting"].add("alice")
    assert (
        remote_client.counting_bloomfilters["seen_counting"].contains("alice") is True
    )
    remote_client.counting_bloomfilters["seen_counting"].remove("alice")


def test_remote_btree_queries(remote_client):
    remote_client.btrees["by_name"]["apple"] = {
        "id": 1,
        "username": "apple",
        "age": 1,
        "premium": False,
    }
    remote_client.btrees["by_name"]["banana"] = {
        "id": 2,
        "username": "banana",
        "age": 2,
        "premium": True,
    }
    rows = remote_client.btrees["by_name"].range(start="a", end="c")
    assert [row["key"] for row in rows] == ["apple", "banana"]
    assert remote_client.btrees["by_name"].prefix("app")[0]["key"] == "apple"


def test_remote_client_maps_http_errors(remote_client):
    with pytest.raises(LoomNotFoundError):
        remote_client.dicts["by_username"]["missing"]

    with pytest.raises(LoomValidationError):
        remote_client.dicts["by_username"]["bob"] = {
            "id": 2,
            "username": "x" * 100,
            "age": 10,
            "premium": False,
        }
