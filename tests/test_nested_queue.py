"""Tests for nested Queue (Dict[Queue], List[Queue]) and nesting validation."""

import tempfile
import pytest
from pydantic import BaseModel, Field
from loom.database import DB
from loom.datastructures import Dict, List, Queue, Set, BloomFilter
from loom.errors import NestingNotSupportedError


class Task(BaseModel):
    id: int
    name: str = Field(max_length=50)


class Event(BaseModel):
    ts: int
    kind: str = Field(max_length=20)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


# ── Dict[Queue] ──────────────────────────────────────────────────────────────

class TestDictQueue:
    def test_dict_queue_basic(self, db_path):
        with DB(db_path) as db:
            task_ds = db.create_dataset("tasks", Task)
            QueueTpl = Queue.template(task_ds, block_size=8)
            user_queues = db.create_dict("user_queues", QueueTpl)

            alice_q = user_queues["alice"]
            alice_q.push({"id": 1, "name": "design"})
            alice_q.push({"id": 2, "name": "review"})

            bob_q = user_queues["bob"]
            bob_q.push({"id": 3, "name": "deploy"})

            assert len(user_queues["alice"]) == 2
            assert len(user_queues["bob"]) == 1

    def test_dict_queue_fifo_order(self, db_path):
        with DB(db_path) as db:
            task_ds = db.create_dataset("tasks", Task)
            QueueTpl = Queue.template(task_ds, block_size=4)
            user_queues = db.create_dict("user_queues", QueueTpl)

            q = user_queues["alice"]
            for i in range(10):
                q.push({"id": i, "name": f"task_{i}"})

            for expected in range(10):
                item = user_queues["alice"].pop()
                assert int(item["id"]) == expected

    def test_dict_queue_multiple_users(self, db_path):
        with DB(db_path) as db:
            task_ds = db.create_dataset("tasks", Task)
            QueueTpl = Queue.template(task_ds)
            user_queues = db.create_dict("user_queues", QueueTpl)

            for user in ["alice", "bob", "carol"]:
                q = user_queues[user]
                for i in range(5):
                    q.push({"id": i, "name": f"{user}_{i}"})

            for user in ["alice", "bob", "carol"]:
                assert len(user_queues[user]) == 5

    def test_dict_queue_block_freed_on_pop(self, db_path):
        """For nested Queues, exhausted blocks are dropped from the block list
        but not returned to ByteFileDB (shared dataset; parent controls lifetime).
        The block list should shrink when a block is exhausted."""
        block_size = 4
        with DB(db_path) as db:
            task_ds = db.create_dataset("tasks", Task)
            QueueTpl = Queue.template(task_ds, block_size=block_size)
            user_queues = db.create_dict("uq", QueueTpl)

            q = user_queues["alice"]
            for i in range(block_size * 3):
                q.push({"id": i, "name": "x"})

            q2 = user_queues["alice"]
            blocks_before = len(q2._blocks)
            # Pop one full block worth
            for _ in range(block_size):
                user_queues["alice"].pop()
            blocks_after = len(user_queues["alice"]._blocks)
            assert blocks_after < blocks_before, \
                "exhausted block should be dropped from block list"

    def test_dict_queue_roundtrip(self, db_path):
        with DB(db_path) as db:
            task_ds = db.create_dataset("tasks", Task)
            QueueTpl = Queue.template(task_ds, block_size=4)
            user_queues = db.create_dict("user_queues", QueueTpl)
            for i in range(8):
                user_queues["alice"].push({"id": i, "name": f"t{i}"})
            user_queues["alice"].pop()  # consume one

        with DB(db_path) as db:
            user_queues = db["user_queues"]
            q = user_queues["alice"]
            assert len(q) == 7
            assert int(q.peek()["id"]) == 1


# ── List[Queue] ───────────────────────────────────────────────────────────────

class TestListQueue:
    def test_list_queue_basic(self, db_path):
        with DB(db_path) as db:
            ev_ds = db.create_dataset("events", Event)
            QueueTpl = Queue.template(ev_ds, block_size=8)
            channels = db.create_list("channels", QueueTpl)

            ch0 = channels.append()
            ch0.push({"ts": 1, "kind": "click"})
            ch0.push({"ts": 2, "kind": "scroll"})

            ch1 = channels.append()
            ch1.push({"ts": 3, "kind": "load"})

            assert len(channels[0]) == 2
            assert len(channels[1]) == 1

    def test_list_queue_fifo(self, db_path):
        with DB(db_path) as db:
            ev_ds = db.create_dataset("events", Event)
            QueueTpl = Queue.template(ev_ds)
            channels = db.create_list("channels", QueueTpl)

            ch = channels.append()
            for i in range(6):
                ch.push({"ts": i, "kind": "evt"})

            for expected_ts in range(6):
                item = channels[0].pop()
                assert int(item["ts"]) == expected_ts

    def test_list_queue_roundtrip(self, db_path):
        with DB(db_path) as db:
            ev_ds = db.create_dataset("events", Event)
            QueueTpl = Queue.template(ev_ds)
            channels = db.create_list("channels", QueueTpl)

            ch = channels.append()
            for i in range(5):
                ch.push({"ts": i, "kind": "x"})
            channels[0].pop()

        with DB(db_path) as db:
            channels = db["channels"]
            q = channels[0]
            assert len(q) == 4
            assert int(q.peek()["ts"]) == 1


# ── NestingNotSupportedError ──────────────────────────────────────────────────

class TestNestingValidation:
    def test_bloom_cannot_be_nested(self, db_path):
        """BloomFilter has _outer_types_supported = () so nesting is forbidden."""
        with pytest.raises(NestingNotSupportedError):
            BloomFilter._check_nesting(Dict)

    def test_queue_in_set_not_supported(self, db_path):
        """Set cannot contain nested structures."""
        with pytest.raises(NestingNotSupportedError):
            Queue._check_nesting(Set)

    def test_valid_nesting_no_error(self, db_path):
        """Valid combinations don't raise."""
        Queue._check_nesting(Dict)   # OK
        Queue._check_nesting(List)   # OK
        List._check_nesting(Dict)    # OK
        Dict._check_nesting(List)    # OK
        Set._check_nesting(Dict)     # OK
        Set._check_nesting(List)     # OK

    def test_create_list_queue_validates(self):
        """The error is raised at _check_nesting time if the inner type is invalid."""
        with pytest.raises(NestingNotSupportedError):
            BloomFilter._check_nesting(List)

    def test_error_message_informative(self):
        try:
            BloomFilter._check_nesting(Dict)
        except NestingNotSupportedError as e:
            assert "BloomFilter" in str(e)
            assert "Dict" in str(e)
            assert e.outer == "Dict"
            assert e.inner == "BloomFilter"
