"""Tests for LRUDict — persistent O(1) LRU cache."""

import time
import tempfile
import random
import string
import pytest
from pydantic import BaseModel, Field
from loom.database import DB


class Item(BaseModel):
    id:    int
    value: float


class Profile(BaseModel):
    name:  str
    score: float


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


# ── Core correctness ─────────────────────────────────────────────────────


class TestLRUDictBasics:
    def test_insert_and_get(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=10)
            lru["a"] = {"id": 1, "value": 1.0}
            lru["b"] = {"id": 2, "value": 2.0}
            assert int(lru["a"]["id"]) == 1
            assert int(lru["b"]["id"]) == 2

    def test_pydantic_schema(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Profile, capacity=5)
            lru["alice"] = {"name": "Alice", "score": 9.5}
            rec = lru["alice"]
            assert rec["name"] == "Alice"
            assert abs(float(rec["score"]) - 9.5) < 0.01

    def test_contains(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            lru["x"] = {"id": 1, "value": 0.0}
            assert "x" in lru
            assert "y" not in lru

    def test_len_and_is_full(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            assert len(lru) == 0
            assert not lru.is_full
            for i in range(3):
                lru[f"k{i}"] = {"id": i, "value": 0.0}
            assert len(lru) == 3
            assert lru.is_full

    def test_update_existing(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            lru["a"] = {"id": 1, "value": 1.0}
            lru["a"] = {"id": 1, "value": 99.0}   # update
            assert abs(float(lru["a"]["value"]) - 99.0) < 0.01
            assert len(lru) == 1              # no duplicate

    def test_delete(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            lru["a"] = {"id": 1, "value": 1.0}
            lru["b"] = {"id": 2, "value": 2.0}
            del lru["a"]
            assert "a" not in lru
            assert len(lru) == 1
            assert "b" in lru

    def test_delete_then_reinsert(self, db_path):
        """Deleted slot should be reused for new entries."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            lru["a"] = {"id": 1, "value": 1.0}
            lru["b"] = {"id": 2, "value": 2.0}
            lru["c"] = {"id": 3, "value": 3.0}
            del lru["b"]
            lru["d"] = {"id": 4, "value": 4.0}   # should reuse slot
            assert "d" in lru
            assert "b" not in lru
            assert len(lru) == 3

    def test_get_default(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            assert lru.get("missing", None) is None
            lru["x"] = {"id": 1, "value": 0.0}
            assert lru.get("x") is not None

    def test_key_error(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            with pytest.raises(KeyError):
                _ = lru["nonexistent"]

    def test_type_error_value(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            with pytest.raises(TypeError):
                lru["x"] = "not a dict"


# ── LRU eviction order ────────────────────────────────────────────────────


class TestEvictionOrder:
    def test_evicts_lru_on_insert(self, db_path):
        """Inserting when full evicts the least recently used entry."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            lru["a"] = {"id": 1, "value": 0.0}
            lru["b"] = {"id": 2, "value": 0.0}
            lru["c"] = {"id": 3, "value": 0.0}   # order: c(head)→b→a(tail)
            lru["d"] = {"id": 4, "value": 0.0}   # evicts a
            assert "a" not in lru
            assert "b" in lru
            assert "c" in lru
            assert "d" in lru

    def test_get_promotes_to_head(self, db_path):
        """get() moves the accessed entry to most-recent position."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            lru["a"] = {"id": 1, "value": 0.0}
            lru["b"] = {"id": 2, "value": 0.0}
            lru["c"] = {"id": 3, "value": 0.0}   # c→b→a (a=LRU)
            _ = lru["a"]                           # promotes a: a→c→b (b=LRU)
            lru["d"] = {"id": 4, "value": 0.0}   # evicts b
            assert "b" not in lru
            assert "a" in lru

    def test_set_existing_promotes_to_head(self, db_path):
        """set() on existing key promotes it to most-recent."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            lru["a"] = {"id": 1, "value": 0.0}
            lru["b"] = {"id": 2, "value": 0.0}
            lru["c"] = {"id": 3, "value": 0.0}   # c→b→a (a=LRU)
            lru["a"] = {"id": 1, "value": 99.0}  # update a → a→c→b (b=LRU)
            lru["d"] = {"id": 4, "value": 0.0}   # evicts b
            assert "b" not in lru
            assert "a" in lru
            assert abs(float(lru["a"]["value"]) - 99.0) < 0.01

    def test_eviction_sequence(self, db_path):
        """Multiple evictions in the right order."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            lru["a"] = {"id": 1, "value": 0.0}
            lru["b"] = {"id": 2, "value": 0.0}
            lru["c"] = {"id": 3, "value": 0.0}   # c→b→a
            _ = lru["a"]                           # a→c→b  (b=LRU)
            lru["d"] = {"id": 4, "value": 0.0}   # evicts b → d→a→c
            assert "b" not in lru
            _ = lru["c"]                           # c→d→a  (a=LRU)
            lru["e"] = {"id": 5, "value": 0.0}   # evicts a → e→c→d
            assert "a" not in lru
            assert list(lru.keys()) == ["e", "c", "d"]

    def test_iteration_mru_to_lru(self, db_path):
        """items() yields entries from most-recent to least-recent."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            for i, k in enumerate(["a", "b", "c", "d", "e"]):
                lru[k] = {"id": i, "value": float(i)}
            # e is MRU, a is LRU
            keys = list(lru.keys())
            assert keys == ["e", "d", "c", "b", "a"]


# ── Persistence (round-trip) ──────────────────────────────────────────────


class TestPersistence:
    def test_basic_roundtrip(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Profile, capacity=10)
            lru["alice"] = {"name": "Alice", "score": 9.5}
            lru["bob"]   = {"name": "Bob",   "score": 7.0}

        with DB(db_path) as db:
            lru = db["lru"]
            assert len(lru) == 2
            assert lru["alice"]["name"] == "Alice"
            assert abs(float(lru["bob"]["score"]) - 7.0) < 0.01

    def test_lru_order_preserved(self, db_path):
        """MRU/LRU order is restored exactly after reload."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5)
            for i, k in enumerate(["a", "b", "c", "d", "e"]):
                lru[k] = {"id": i, "value": 0.0}
            _ = lru["a"]   # a→e→d→c→b

        with DB(db_path) as db:
            lru = db["lru"]
            keys = list(lru.keys())
            assert keys[0] == "a"   # still MRU
            assert keys[-1] == "b"  # still LRU

    def test_eviction_after_reload(self, db_path):
        """Eviction works correctly after reloading."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3)
            lru["a"] = {"id": 1, "value": 0.0}
            lru["b"] = {"id": 2, "value": 0.0}
            lru["c"] = {"id": 3, "value": 0.0}

        with DB(db_path) as db:
            lru = db["lru"]
            lru["d"] = {"id": 4, "value": 0.0}   # should evict a (LRU)
            assert "a" not in lru
            assert "d" in lru

    def test_capacity_preserved(self, db_path):
        with DB(db_path) as db:
            db.create_lru_dict("lru", Item, capacity=42)

        with DB(db_path) as db:
            lru = db["lru"]
            assert lru.capacity == 42


# ── hash_keys mode ────────────────────────────────────────────────────────


class TestHashKeys:
    def test_hash_keys_basic(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=10,
                                     hash_keys=True, hash_bits=128)
            long_key = "A very long document that would blow up key_size " * 5
            lru[long_key] = {"id": 42, "value": 3.14}
            assert long_key in lru
            assert int(lru[long_key]["id"]) == 42

    def test_hash_keys_eviction(self, db_path):
        """Eviction works correctly with hashed keys."""
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=3,
                                     hash_keys=True, hash_bits=64)
            lru["alpha"] = {"id": 1, "value": 0.0}
            lru["beta"]  = {"id": 2, "value": 0.0}
            lru["gamma"] = {"id": 3, "value": 0.0}
            lru["delta"] = {"id": 4, "value": 0.0}   # evicts alpha
            assert "alpha" not in lru
            assert "delta" in lru

    def test_hash_keys_roundtrip(self, db_path):
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=5,
                                     hash_keys=True, hash_bits=128)
            lru["key_one"] = {"id": 1, "value": 1.0}
            lru["key_two"] = {"id": 2, "value": 2.0}

        with DB(db_path) as db:
            lru = db["lru"]
            assert int(lru["key_one"]["id"]) == 1
            assert int(lru["key_two"]["id"]) == 2

    def test_hash_keys_slot_size(self, db_path):
        """With hash_keys, hash table slots are much smaller than key_size."""
        with DB(db_path) as db:
            lru_hashed = db.create_lru_dict("lru_h", Item, capacity=100,
                                             hash_keys=True, hash_bits=64)
            lru_plain  = db.create_lru_dict("lru_p", Item, capacity=100,
                                             key_size=100)
            slot_hashed = lru_hashed._items_ds.record_size
            slot_plain  = lru_plain._items_ds.record_size
            # Hashed slots should be significantly smaller
            assert slot_hashed < slot_plain


# ── Performance sanity ────────────────────────────────────────────────────


class TestPerformance:
    def test_insert_evict_throughput(self, db_path):
        """Steady-state (always full) insert+evict must be reasonably fast."""
        CAP = 1_000
        N   = 5_000
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=CAP)
            for i in range(CAP):
                lru[f"k{i:06d}"] = {"id": i, "value": float(i)}

            t0 = time.perf_counter()
            for i in range(CAP, CAP + N):
                lru[f"k{i:06d}"] = {"id": i, "value": float(i)}
            elapsed = time.perf_counter() - t0

        # Should manage at least 3 000 evicting inserts/s
        assert N / elapsed > 3_000, f"too slow: {N/elapsed:.0f} ops/s"

    def test_get_throughput(self, db_path):
        """Cache reads must be faster than evicting inserts."""
        N = 5_000
        with DB(db_path) as db:
            lru = db.create_lru_dict("lru", Item, capacity=N)
            for i in range(N):
                lru[f"k{i:06d}"] = {"id": i, "value": float(i)}

            t0 = time.perf_counter()
            for i in range(N):
                _ = lru[f"k{i:06d}"]
            t_get = time.perf_counter() - t0

        assert N / t_get > 10_000, f"read too slow: {N/t_get:.0f} ops/s"

    def test_hash_keys_long_key_slot_size(self, db_path):
        """hash_keys=True reduces slot size by ≥ 5x for 200-char keys."""
        with DB(db_path) as db:
            lru_plain  = db.create_lru_dict("p", Item, capacity=10, key_size=200)
            lru_hashed = db.create_lru_dict("h", Item, capacity=10,
                                             hash_keys=True, hash_bits=64)
            # Index Dict slot sizes
            plain_slot  = lru_plain._items_ds.record_size
            hashed_slot = lru_hashed._items_ds.record_size
            assert plain_slot / hashed_slot >= 5
