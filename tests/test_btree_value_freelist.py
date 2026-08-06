"""BTree reuses freed value slots (like Dict already did).

Before, the BTree value allocator was bump-only: a delete freed nothing, so a
delete-heavy or moving-key (counter) index grew its values arena forever. Now
delete pushes the slot onto an intrusive freelist and a later insert pops it —
reuse survives reopen (the head is persisted). Nodes are a separate story (no
delete-merge), so these tests assert *value-record* reuse specifically.
"""

import os
import tempfile

from loom import DB


def _open(path):
    db = DB(path)
    db.open()
    return db


class TestValueSlotReuse:
    def test_delete_then_insert_reuses_value_records(self):
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(10000):
                    bt[f"{i:06d}"] = {"v": i}
                used = bt.next_data_offset
                for i in range(10000):
                    del bt[f"{i:06d}"]
                # 10k new keys must reuse the 10k freed slots — not allocate more
                for i in range(10000, 20000):
                    bt[f"{i:06d}"] = {"v": i}
                assert bt.next_data_offset == used          # no new value records
                assert len(bt) == 10000
                assert bt["015000"]["v"] == 15000

    def test_reinserting_same_range_barely_grows(self):
        # Re-occupying the same key range reuses BOTH value slots and the
        # emptied leaves (in-range refill) → near-zero arena growth.
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(10000):
                    bt[f"{i:06d}"] = {"v": i}
                db.flush()
                base = db._db.get_allocation_index()
                for _ in range(3):                          # churn the same keys
                    for i in range(10000):
                        del bt[f"{i:06d}"]
                    for i in range(10000):
                        bt[f"{i:06d}"] = {"v": i}
                db.flush()
                grew = db._db.get_allocation_index() - base
                assert grew == 0, grew                      # fully reused
                assert len(bt) == 10000 and bt["005000"]["v"] == 5000

    def test_freelist_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.loom")
            with DB(path) as db:
                bt = db.create_btree("t", {"v": "int64"}, key_size=16)
                for i in range(8000):
                    bt[f"{i:06d}"] = {"v": i}
                for i in range(4000):
                    del bt[f"{i:06d}"]
                head = bt._value_freelist_head
                assert head != 0
            with _open(path) as db:
                bt = db["t"]
                assert bt._value_freelist_head == head      # persisted
                used = bt.next_data_offset
                for i in range(8000, 12000):                # 4000 new → reuse
                    bt[f"{i:06d}"] = {"v": i}
                assert bt.next_data_offset == used          # reused freed slots
                assert len(bt) == 8000 and bt["010000"]["v"] == 10000

    def test_reused_slot_holds_correct_value(self):
        # A reused slot must be fully overwritten — no bleed from the pointer
        # the freelist stored in it, nor from the previous occupant.
        with tempfile.TemporaryDirectory() as d:
            with DB(os.path.join(d, "a.loom")) as db:
                bt = db.create_btree("t", {"v": "int64", "w": "int64"}, key_size=16)
                for i in range(2000):
                    bt[f"{i:06d}"] = {"v": i, "w": i * 2}
                for i in range(2000):
                    del bt[f"{i:06d}"]
                for i in range(2000, 4000):
                    bt[f"{i:06d}"] = {"v": i, "w": i * 2}
                for i in range(2000, 4000):
                    assert bt[f"{i:06d}"] == {"v": i, "w": i * 2}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
