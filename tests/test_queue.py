"""Tests for persistent Queue."""

import os
import tempfile
import time
import pytest
from pydantic import BaseModel, Field
from loom.database import DB
from loom.errors import LoomError


class Job(BaseModel):
    id: int
    name: str = Field(max_length=50)
    priority: float


class Msg(BaseModel):
    seq: int
    body: str  # text — variable-length


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


# ── Basic ops ─────────────────────────────────────────────────────────────────

class TestBasicOps:
    def test_push_pop_fifo(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            for i in range(5):
                q.push({"id": i, "name": f"job_{i}", "priority": float(i)})
            for expected_id in range(5):
                item = q.pop()
                assert item["id"] == expected_id, "FIFO order violated"

    def test_peek_non_destructive(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            q.push({"id": 1, "name": "a", "priority": 0.5})
            first = q.peek()
            assert first["id"] == 1
            assert len(q) == 1  # not consumed
            assert q.pop()["id"] == 1

    def test_len(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            assert len(q) == 0
            q.push({"id": 1, "name": "a", "priority": 0.0})
            assert len(q) == 1
            q.push({"id": 2, "name": "b", "priority": 0.0})
            assert len(q) == 2
            q.pop()
            assert len(q) == 1

    def test_bool(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            assert not q
            q.push({"id": 1, "name": "x", "priority": 0.0})
            assert q

    def test_pop_empty_raises(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            with pytest.raises(IndexError):
                q.pop()

    def test_peek_empty_raises(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            with pytest.raises(IndexError):
                q.peek()

    def test_push_wrong_type_raises(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            with pytest.raises(TypeError):
                q.push("not a dict")

    def test_iterate_non_destructive(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            for i in range(5):
                q.push({"id": i, "name": f"j{i}", "priority": 0.0})
            ids = [it["id"] for it in q]
            assert ids == [0, 1, 2, 3, 4]
            assert len(q) == 5  # unchanged

    def test_text_field(self, db_path):
        """Variable-length text field works in Queue."""
        with DB(db_path) as db:
            q = db.create_queue("msgs", Msg)
            long_body = "Hello world! " * 50
            q.push({"seq": 1, "body": "short"})
            q.push({"seq": 2, "body": long_body})
            assert q.pop()["body"] == "short"
            assert q.pop()["body"] == long_body

    def test_dict_schema(self, db_path):
        """Plain dict schema (no Pydantic) works."""
        with DB(db_path) as db:
            q = db.create_queue("q", {"x": "uint32", "y": "float64"})
            q.push({"x": 7, "y": 3.14})
            item = q.pop()
            assert int(item["x"]) == 7


# ── Block lifecycle ───────────────────────────────────────────────────────────

class TestBlockLifecycle:
    def test_blocks_freed_on_exhaust(self, db_path):
        """Exhausted head blocks are returned to the ByteFileDB freelist."""
        block_size = 4
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job, block_size=block_size)
            for i in range(12):
                q.push({"id": i, "name": f"j{i}", "priority": 0.0})

            freelist_before = len(db._db._freelist)

            # Pop one full block
            for _ in range(block_size):
                q.pop()

            freelist_after = len(db._db._freelist)
            assert freelist_after > freelist_before, \
                "freed block should appear in ByteFileDB freelist"

    def test_freelist_reused_on_push(self, db_path):
        """After freeing a block, a subsequent push reuses it."""
        block_size = 4
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job, block_size=block_size)
            for i in range(block_size):
                q.push({"id": i, "name": f"j{i}", "priority": 0.0})

            alloc_after_first_block = db._db.get_allocation_index()

            # Exhaust the block → freed
            for _ in range(block_size):
                q.pop()

            # Push again — should reuse the freed block
            for i in range(block_size):
                q.push({"id": 100 + i, "name": "r", "priority": 0.0})

            alloc_after_reuse = db._db.get_allocation_index()
            assert alloc_after_reuse == alloc_after_first_block, \
                "file should not have grown (freed block reused)"

    def test_blocks_count_stable_steady_state(self, db_path):
        """Push N, pop N repeatedly: block count stays low."""
        block_size = 8
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job, block_size=block_size)
            for i in range(block_size * 5):
                q.push({"id": i, "name": "x", "priority": 0.0})
                if i % block_size == block_size - 1:
                    for _ in range(block_size):
                        q.pop()
            assert len(q._blocks) <= 3, "block count should stay low in steady state"

    def test_empty_then_push_no_crash(self, db_path):
        """Pop to empty then push again works correctly."""
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job, block_size=4)
            for i in range(4):
                q.push({"id": i, "name": "x", "priority": 0.0})
            for _ in range(4):
                q.pop()
            assert len(q) == 0
            q.push({"id": 99, "name": "back", "priority": 1.0})
            assert len(q) == 1
            assert q.pop()["id"] == 99


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_roundtrip_basic(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            for i in range(10):
                q.push({"id": i, "name": f"j{i}", "priority": float(i)})
            q.pop()  # consume one

        with DB(db_path) as db:
            q = db["jobs"]
            assert len(q) == 9
            assert q.peek()["id"] == 1

    def test_roundtrip_after_partial_pop(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job, block_size=4)
            for i in range(20):
                q.push({"id": i, "name": f"j{i}", "priority": 0.0})
            for _ in range(7):
                q.pop()

        with DB(db_path) as db:
            q = db["jobs"]
            assert len(q) == 13
            ids = [it["id"] for it in q]
            assert ids == list(range(7, 20))

    def test_multiple_sessions(self, db_path):
        """Open, push, close, open, push, close, open — all items readable."""
        addrs = []
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            for i in range(5):
                q.push({"id": i, "name": f"s1_{i}", "priority": 0.0})

        with DB(db_path) as db:
            q = db["jobs"]
            for i in range(5, 10):
                q.push({"id": i, "name": f"s2_{i}", "priority": 0.0})

        with DB(db_path) as db:
            q = db["jobs"]
            assert len(q) == 10
            ids = [it["id"] for it in q]
            assert ids == list(range(10))

    def test_block_size_persisted(self, db_path):
        with DB(db_path) as db:
            db.create_queue("jobs", Job, block_size=32)

        with DB(db_path) as db:
            q = db["jobs"]
            assert q.block_size == 32


# ── push_many ─────────────────────────────────────────────────────────────────

class TestPushMany:
    def test_push_many_fifo(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job, block_size=4)
            items = [{"id": i, "name": f"j{i}", "priority": 0.0} for i in range(20)]
            q.push_many(items)
            assert len(q) == 20
            for expected_id in range(20):
                assert q.pop()["id"] == expected_id

    def test_push_many_empty(self, db_path):
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            q.push_many([])
            assert len(q) == 0

    def test_push_many_batch_mode(self, db_path):
        """push_many inside db.batch() is safe."""
        with DB(db_path) as db:
            q = db.create_queue("jobs", Job)
            with db.batch():
                q.push_many([{"id": i, "name": "x", "priority": 0.0} for i in range(100)])
            assert len(q) == 100


# ── Performance sanity ────────────────────────────────────────────────────────

class TestPerformance:
    def test_push_pop_speed(self, db_path):
        """10K push + 10K pop should complete in < 5s on any machine."""
        N = 10_000
        with DB(db_path) as db:
            q = db.create_queue("jobs", {"id": "uint32", "score": "float64"})

            t0 = time.perf_counter()
            with db.batch():
                q.push_many([{"id": i, "score": float(i)} for i in range(N)])
            t_push = time.perf_counter() - t0
            assert t_push < 5.0, f"push_many too slow: {t_push:.2f}s"

            t0 = time.perf_counter()
            for _ in range(N):
                q.pop()
            t_pop = time.perf_counter() - t0
            assert t_pop < 5.0, f"pop loop too slow: {t_pop:.2f}s"
            assert len(q) == 0

    def test_steady_state_file_size(self, db_path):
        """File size stays bounded in a push/pop steady state."""
        block_size = 64
        N = block_size * 20  # 20 full blocks
        with DB(db_path) as db:
            q = db.create_queue("jobs", {"id": "uint32"}, block_size=block_size)
            # Fill
            q.push_many([{"id": i} for i in range(N)])
            # Consume all
            for _ in range(N):
                q.pop()
            size_after_drain = db._db.get_allocation_index()

            # Fill again — should reuse freed blocks
            q.push_many([{"id": i} for i in range(N)])
            size_after_refill = db._db.get_allocation_index()

        assert size_after_refill == size_after_drain, \
            "file grew despite freelist reuse"
