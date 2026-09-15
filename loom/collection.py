"""Collection — a record store with declarative, typed secondary indexes.

A Collection is created from a record schema; each field's index *kind* is
declared, and mapped to the right loom structure, kept in sync automatically:

    posts = db.collection("posts", Post, indexes={
        "id":         "primary",                       # Dict — record store + unique pk
        "username":   Many(sort="created_at", desc=True),  # BTree — posts of a user, recent first
        "engagement": "range",                         # BTree — engagement >= x
        "email":      "unique",                        # Dict — 1:1, enforced
    })

    posts.insert({"id": "p1", "username": "alice", "created_at": 170, "engagement": 9, ...})
    posts["p1"]                          # by primary key
    posts.find("username", "alice", limit=20)   # one-to-many, ordered (recent first)
    posts.range("engagement", 1000, None)       # range scan (engagement >= 1000)
    posts.get("email", "a@x.com")               # unique lookup
    posts.increment("p1", "engagement", 1)      # atomic counter bump (re-indexes)

Index kinds → structures:
    primary / unique  → Dict   (hashmap; primary also stores the records)
    range / many      → BTree  (composite key: range = value+pk; many =
                                group + sort + pk → prefix-scan gives an
                                ordered group for free)

Range/many keys are order-preserving encoded (signed ints zero-padded, floats
via IEEE-754 munging, datetime/date via microsecond keys, strings as-is) so the
BTree's lexicographic order matches the natural order — sort by an int, float or
timestamp criterion with desc=True/False.

Sync: insert/update/delete/increment run under db.write_lock() + db.batch().
Not a crash-atomic cross-index transaction — reindex() rebuilds if needed.
Full-text ("search") indexes are planned (SearchIndex integration).
"""

from __future__ import annotations

import numbers
import struct
from contextlib import nullcontext
from datetime import date, datetime
from functools import lru_cache
from itertools import islice

import mmh3
import numpy as np

from loom.dataset import as_record, _micros_to_dt
from loom.errors import CollectionDroppedError

try:
    from loom_accel import fused_scan as _accel_fused
except ImportError:
    _accel_fused = None

try:
    from loom_accel import sorted_intersect_many as _accel_intersect
except ImportError:
    _accel_intersect = None


def _hash_value(value):
    """Fixed-width 128-bit murmur hash (32 hex chars) of a value — used as the
    group key for indexes on unbounded (text/blob/json) fields, so possibly-long
    values (e.g. an argument or article title) can be indexed without truncation.
    No order is preserved (equality grouping only)."""
    return f"{mmh3.hash128(str(value), signed=False):032x}"

def _values_differ(new, old):
    """True if ``new`` differs from the stored ``old`` (no-op detection for
    update_many).  Handles array/Vec fields whose ``!=`` is elementwise."""
    if new is None or old is None:
        return new is not old                # None vs None → False
    try:
        return bool(new != old)
    except (ValueError, TypeError):          # numpy arrays → ambiguous truth
        return not np.array_equal(new, old)


_SEP = "\x00"          # composite-key separator (numpy U preserves embedded NULs)
_INT_OFFSET = 1 << 63  # map signed int64 → unsigned for zero-padded ordering
_UINT_MAX = (1 << 64) - 1


class _DescTable(dict):
    """str.translate table memoising the codepoint complement (C-speed loop)."""

    def __missing__(self, cp):
        repl = chr(0x10FFFF - cp)
        self[cp] = repl
        return repl


_DESC_TABLE = _DescTable()


def _desc_str(s):
    """Reverse-lexicographic order for a string: complement each codepoint."""
    return s.translate(_DESC_TABLE)


def _float_key(value, desc=False):
    """Order-preserving 20-digit key for a float (IEEE-754 bit munging).

    Flipping the sign bit for positives and all bits for negatives makes the
    raw 64-bit pattern sort in the same order as the float value.  NaN is not
    ordered meaningfully (don't index NaN).
    """
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits & 0x8000000000000000:        # negative → flip everything
        bits ^= 0xFFFFFFFFFFFFFFFF
    else:                                 # positive → flip just the sign bit
        bits ^= 0x8000000000000000
    if desc:
        bits = _UINT_MAX - bits
    return f"{bits:020d}"


# ── index-kind specs ──────────────────────────────────────────────────────────


class Primary:
    kind = "primary"


class Unique:
    kind = "unique"

    def __init__(self, field=None):
        self.field = field   # None → the index's name is the field


class Range:
    kind = "range"

    def __init__(self, field=None):
        self.field = field


class Many:
    kind = "many"

    def __init__(self, sort=None, desc=False, field=None, counted=False):
        self.sort = sort
        self.desc = desc
        # field defaults to the index name — set it to index the SAME field
        # under several indexes (e.g. category by engagement AND by date).
        self.field = field
        # counted=True maintains a companion group→count Dict on every write
        # (~one field write per insert): count() becomes O(1) and groups()
        # lists every group with its size without touching the data.
        self.counted = counted


class Search:
    kind = "search"

    def __init__(self, fields=None, scoring="boolean", bm25_k1=1.5, bm25_b=0.75):
        self.fields = list(fields) if fields else None   # None → [index name]
        self.scoring = scoring
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b


class Vector:
    """Exact (flat) vector similarity over a ``Vec(N)`` field.

    Not an ANN structure: the vectors live inline in the records, nothing is
    maintained on write.  ``nearest()`` narrows candidates through the
    collection's *other* indexes first (``where=``), then scores the
    survivors' vectors exactly — a pre-filtered flat search, which beats ANN
    whenever the filter is selective (a date window, a group, ...).
    """

    kind = "vector"

    def __init__(self, field=None, metric="cosine"):
        self.field = field            # None → the index's name is the field
        self.metric = metric          # "cosine" | "l2" | "dot"


_STRING_KINDS = {"primary": Primary, "unique": Unique, "range": Range,
                 "many": Many, "search": Search, "vector": Vector}


def _as_spec(spec):
    """Normalise a string alias or spec instance to a spec instance."""
    if isinstance(spec, str):
        if spec not in _STRING_KINDS:
            raise ValueError(f"unknown index kind {spec!r}")
        return _STRING_KINDS[spec]()
    return spec


# ── order-preserving value encoding ───────────────────────────────────────────


def encode_value(value, desc=False):
    """Encode a value to a string that sorts in natural (or reverse) order.

    Signed integers → 20-digit zero-padded (offset so negatives sort first);
    floats → 20-digit IEEE-754 munged key; datetime/date → microsecond key
    (chronological); bools → 0/1; everything else → str().  desc=True reverses.

    Index a field with ONE consistent type — int, float, datetime and str keys
    use different encodings, so they must not be mixed within one index.
    """
    if isinstance(value, bool):
        value = int(value)
    # numbers.Integral covers Python int AND numpy integers (np.int64 …),
    # which is what reading a record back yields — both must encode identically
    # or re-indexing on update/delete would miss the original composite key.
    if isinstance(value, numbers.Integral):
        u = int(value) + _INT_OFFSET
        if desc:
            u = _UINT_MAX - u
        return f"{u:020d}"
    # numbers.Real also catches numpy floats (np.float64/np.float32).
    if isinstance(value, numbers.Real):
        return _float_key(value, desc)
    if isinstance(value, (datetime, date)):
        from loom.schema import dt_key
        s = dt_key(value, "microsecond")
        return _desc_str(s) if desc else s
    s = str(value)
    return _desc_str(s) if desc else s


def _decode_int(enc, desc):
    """Inverse of encode_value for a signed integer key (see encode_value)."""
    u = int(enc)
    if desc:
        u = _UINT_MAX - u
    return u - _INT_OFFSET


def _decode_float(enc, desc):
    """Inverse of _float_key: undo desc, unmunge the IEEE-754 bits."""
    bits = int(enc)
    if desc:
        bits = _UINT_MAX - bits
    if bits & 0x8000000000000000:      # sign bit set after munge ⇒ was positive
        bits ^= 0x8000000000000000
    else:                               # clear ⇒ was negative (all bits flipped)
        bits ^= 0xFFFFFFFFFFFFFFFF
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _decode_str(enc, desc):
    """Inverse of the string branch of encode_value (_desc_str is self-inverse)."""
    return _desc_str(enc) if desc else enc


def _decode_datetime(enc, desc):
    """Inverse of the datetime branch of encode_value (microsecond dt_key).

    Parses the fixed "%Y%m%dT%H%M%S%f" layout by slicing rather than via
    strptime — strptime is ~5x slower and would make an index-only timestamp
    projection cost more than just reading the record's inline int64."""
    s = _desc_str(enc) if desc else enc
    return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                    int(s[9:11]), int(s[11:13]), int(s[13:15]), int(s[15:21]))


@lru_cache(maxsize=4096)
def _encode_value_cached(typ, value, desc):
    return encode_value(value, desc)


def _encode_sort(value, desc):
    """encode_value with a small LRU — several Many indexes typically sort on
    the same field (e.g. five indexes × created_at), so within one record's
    index pass the same (value, desc) encodes once instead of N times.

    The cache key carries the value's *type*: encode_value branches on type
    (int → zero-padded, float → IEEE munge, …), yet ``5 == 5.0`` and hash the
    same — so a bare (value, desc) key would let an int and a numerically equal
    float collide and return the wrong encoding.  Keying on type as well keeps
    each branch's encoding distinct."""
    try:
        return _encode_value_cached(type(value), value, desc)
    except TypeError:  # unhashable sort value — encode directly
        return encode_value(value, desc)


class Collection:
    def __init__(self, db, name, dataset, primary_field, primary, indexes,
                 key_size, search=None, vector=None):
        """
        indexes: {idx_name: {"name", "spec", "field", "struct", "sources"}}
        search:  {idx_name: {"fields": [...], "index": SearchIndex,
                             "pk2docid": Dict, "docid2pk": List}}  full-text
        vector:  {idx_name: {"field": ..., "metric": ...}}  flat similarity
                 (no backing structure — the vectors live in the records)
        """
        self._db = db
        self.name = name
        self.dataset = dataset
        self._key_field = primary_field
        self._primary = primary
        self._indexes = indexes
        self._key_size = key_size
        self._search = search or {}
        self._vector = vector or {}
        self._indexed_fields = set()
        for ix in indexes.values():
            # convert sources to a frozenset for fast set intersection
            ix["sources"] = frozenset(ix["sources"])
            self._indexed_fields.update(ix["sources"])
        # fields feeding any full-text index (→ re-index on update)
        self._search_fields = set()
        for si in self._search.values():
            self._search_fields.update(si["fields"])

    # ── key construction ─────────────────────────────────────────────────

    def _pk_of(self, record):
        if self._key_field not in record:
            raise KeyError(f"record missing primary-key field {self._key_field!r}")
        return str(record[self._key_field])

    def _group_key(self, ix, val):
        """Encode the value/group part of an index key — hashed (fixed width)
        for unbounded fields, order-preserving otherwise."""
        return _hash_value(val) if ix.get("hashed") else encode_value(val)

    def _index_key(self, ix, record, pk):
        spec = ix["spec"]
        val = record.get(ix["field"])
        if val is None:
            return None
        if spec.kind == "unique":
            return self._group_key(ix, val)
        if spec.kind == "range":
            return encode_value(val) + _SEP + pk
        if spec.kind == "many":
            parts = [self._group_key(ix, val)]
            if spec.sort is not None:
                parts.append(_encode_sort(record.get(spec.sort), spec.desc))
            parts.append(pk)
            return _SEP.join(parts)
        raise ValueError(f"unsupported index kind {spec.kind!r}")

    # ── writes ────────────────────────────────────────────────────────────

    def _count_add_key(self, ix, group_key, value, delta):
        """Adjust a counted index's group counter by delta (create at first
        member, drop at zero).  The count update is an in-place field write —
        the stored group value (json) is only written once per group."""
        cnt = ix["counter"]
        cur = cnt.get_fields(group_key, ["n"])
        if cur is None:
            if delta > 0:
                if isinstance(value, np.generic):
                    value = value.item()
                cnt[group_key] = {"value": value, "n": delta}
            return
        n = int(cur["n"]) + delta
        if n <= 0:
            del cnt[group_key]
        else:
            cnt[group_key, "n"] = n

    def _count_add(self, ix, value, delta):
        if ix.get("counter") is None or value is None:
            return
        self._count_add_key(ix, self._group_key(ix, value), value, delta)

    def _add_to_indexes(self, record, pk):
        for ix in self._indexes.values():
            key = self._index_key(ix, record, pk)
            if key is None:
                continue
            if ix["spec"].kind == "unique":
                existing = ix["struct"].get(key)
                if existing is not None and str(existing["pk"]) != pk:
                    raise ValueError(
                        f"duplicate value for unique index {ix['name']!r}: "
                        f"{record.get(ix['field'])!r}"
                    )
            ix["struct"][key] = {"pk": pk}
            if ix.get("counter") is not None:
                self._count_add(ix, record.get(ix["field"]), +1)

    def _remove_from_indexes(self, record, pk):
        for ix in self._indexes.values():
            key = self._index_key(ix, record, pk)
            if key is not None and key in ix["struct"]:
                del ix["struct"][key]
                if ix.get("counter") is not None:
                    self._count_add(ix, record.get(ix["field"]), -1)

    def _add_to_search(self, record, pk):
        for si in self._search.values():
            text = " ".join(str(record.get(f, "")) for f in si["fields"])
            doc_id = si["index"].add(None, text=text)
            si["docid2pk"].append({"pk": pk})
            si["pk2docid"][pk] = {"doc_id": doc_id}

    def _remove_from_search(self, pk):
        for si in self._search.values():
            entry = si["pk2docid"].get(pk)
            if entry is not None:
                si["index"].delete(int(entry["doc_id"]))
                del si["pk2docid"][pk]

    def insert(self, record, durable=False):
        """Insert a record, or upsert it if its primary key already exists
        (the stored record is replaced and every index is re-pointed).

        durable=True (default False) runs the write inside ``db.durable()`` so a
        hard crash cannot leave a torn record/index — at the cost of one file
        snapshot. For many rows prefer ``insert_many(..., durable=True)`` (one
        snapshot per batch) or wrap a loop in ``with db.durable():``."""
        record = as_record(record)
        pk = self._pk_of(record)
        guard = self._db.durable() if durable else nullcontext()
        with guard, self._db.write_lock():
            with self._db.batch(defer_save=True):
                old = self._primary.get(pk)
                if old is not None:        # upsert: drop the old index/search entries
                    self._remove_from_indexes(old, pk)
                    self._remove_from_search(pk)
                self._primary[pk] = record
                self._add_to_indexes(record, pk)
                self._add_to_search(record, pk)
        return pk

    def insert_many(self, records, durable=False):
        """Bulk insert, upserting any record whose primary key already exists
        (its old index/search entries are dropped first).  Within one batch a
        repeated primary key keeps the last occurrence.

        durable=True (default False) wraps the whole batch in ``db.durable()``:
        one file snapshot up front, then an all-or-nothing commit, so a hard
        crash (power loss / kill -9 / container shutdown) mid-insert rolls the
        DB back to its pre-batch state instead of leaving a torn index.  Off by
        default to keep the fast path fast; turn it on for the risky bulk
        ingests where a partial write would corrupt the DB."""
        # Dedup within the batch (last write wins), preserving first-seen order.
        dedup = {}
        for r in records:
            r = as_record(r)
            dedup[self._pk_of(r)] = r
        pks = list(dedup.keys())
        records = list(dedup.values())
        guard = self._db.durable() if durable else nullcontext()
        with guard, self._db.write_lock():
            with self._db.batch():
                # Enforce unique constraints up front (before any write), both
                # within the batch and against existing rows — a violation
                # leaves the collection untouched.
                for ix in self._indexes.values():
                    if ix["spec"].kind != "unique":
                        continue
                    seen = {}
                    for record, pk in zip(records, pks):
                        key = self._index_key(ix, record, pk)
                        if key is None:
                            continue
                        if key in seen:
                            raise ValueError(
                                f"duplicate value for unique index {ix['name']!r} "
                                f"within the batch"
                            )
                        existing = ix["struct"].get(key)
                        if existing is not None and str(existing["pk"]) != pk:
                            raise ValueError(
                                f"duplicate value for unique index {ix['name']!r}: "
                                f"{record.get(ix['field'])!r}"
                            )
                        seen[key] = pk
                # Upsert: drop stale index/search entries for pks already stored,
                # so re-loading a batch with existing keys re-indexes cleanly.
                for pk in pks:
                    old = self._primary.get(pk)
                    if old is not None:
                        self._remove_from_indexes(old, pk)
                        self._remove_from_search(pk)
                # Primary + unique (Dict) indexes go through set_batch — one
                # contiguous arena + a single parent-ref update.  Range/many
                # (BTree) indexes have no bulk insert, so they loop.
                self._primary.set_batch(zip(pks, records))
                for si in self._search.values():
                    p2d = []
                    d2p = []
                    for record, pk in zip(records, pks):
                        text = " ".join(str(record.get(f, "")) for f in si["fields"])
                        doc_id = si["index"].add(None, text=text)
                        d2p.append({"pk": pk})
                        p2d.append((pk, {"doc_id": doc_id}))
                    si["docid2pk"].append_many(d2p)
                    si["pk2docid"].set_batch(p2d)
                for ix in self._indexes.values():
                    struct = ix["struct"]
                    if ix["spec"].kind == "unique":
                        entries = []
                        for record, pk in zip(records, pks):
                            key = self._index_key(ix, record, pk)
                            if key is not None:
                                entries.append((key, {"pk": pk}))
                        struct.set_batch(entries)
                    else:  # BTree composite (range / many)
                        entries = []
                        for record, pk in zip(records, pks):
                            key = self._index_key(ix, record, pk)
                            if key is not None:
                                entries.append((key, {"pk": pk}))
                        if struct.size == 0 and struct.root_addr == 0:
                            struct.bulk_load(entries)        # fresh index: O(n) build
                        else:
                            # Non-empty tree: insert in key order under a
                            # deferred-write block, so each touched leaf is
                            # written once (sequential) instead of once per row.
                            # Without this an incremental (chunked) ingest costs
                            # ~2x a single bulk_load; with it, ~parity.
                            entries.sort(key=lambda kv: kv[0])
                            with struct.deferred_node_writes():
                                for key, val in entries:
                                    struct[key] = val
                    if ix.get("counter") is not None:
                        # one counter write per group, not per record;
                        # deduplicate by raw value first so _group_key
                        # (which may hash) is called once per unique value
                        deltas_by_val = {}
                        for record in records:
                            v = record.get(ix["field"])
                            if v is not None:
                                deltas_by_val[v] = deltas_by_val.get(v, 0) + 1
                        for v, delta in deltas_by_val.items():
                            gk = self._group_key(ix, v)
                            self._count_add_key(ix, gk, v, delta)
        return pks

    def update(self, pk, **changes):
        pk = str(pk)
        old = self._primary.get(pk)
        if old is None:
            raise KeyError(pk)
        if self._key_field in changes and str(changes[self._key_field]) != pk:
            raise ValueError("cannot change the primary key via update()")
        new = {**old, **changes}
        changed_fields = set(changes)
        with self._db.write_lock():
            with self._db.batch(defer_save=True):
                for ix in self._indexes.values():
                    if not changed_fields & ix["sources"]:
                        continue
                    old_key = self._index_key(ix, old, pk)
                    new_key = self._index_key(ix, new, pk)
                    if old_key == new_key:
                        continue
                    if old_key is not None and old_key in ix["struct"]:
                        del ix["struct"][old_key]
                    if new_key is not None:
                        if ix["spec"].kind == "unique":
                            ex = ix["struct"].get(new_key)
                            if ex is not None and str(ex["pk"]) != pk:
                                raise ValueError(
                                    f"duplicate value for unique index {ix['name']!r}"
                                )
                        ix["struct"][new_key] = {"pk": pk}
                    if ix.get("counter") is not None:
                        ov, nv = old.get(ix["field"]), new.get(ix["field"])
                        if ov != nv:      # sort-only change keeps the group
                            self._count_add(ix, ov, -1)
                            self._count_add(ix, nv, +1)
                # Re-index full-text fields that changed (delete old doc, add new).
                if self._search and any(f in changes for f in self._search_fields):
                    for si in self._search.values():
                        if not any(f in changes for f in si["fields"]):
                            continue
                        entry = si["pk2docid"].get(pk)
                        if entry is not None:
                            si["index"].delete(int(entry["doc_id"]))
                        text = " ".join(str(new.get(f, "")) for f in si["fields"])
                        doc_id = si["index"].add(None, text=text)
                        si["docid2pk"].append({"pk": pk})
                        si["pk2docid"][pk] = {"doc_id": doc_id}
                # Write ONLY the changed fields, in place — not the whole record.
                # An unchanged field keeps its stored bytes untouched, so a
                # `text`/`json`/`blob` field that didn't change keeps its blob
                # reference (no decompress-then-recompress of data that is
                # identical). Rewriting the whole record made a one-field bump
                # (e.g. a counter) needlessly re-encode every blob in the row.
                for f, v in changes.items():
                    self._primary[pk, f] = v
        return new

    def delete(self, pk):
        """Delete the record with primary key ``pk`` (also ``del col[pk]``).

        Removes it from the primary store and every secondary / full-text index
        in one transaction. Raises ``KeyError`` if no such record exists.
        """
        pk = str(pk)
        record = self._primary.get(pk)
        if record is None:
            raise KeyError(pk)
        with self._db.write_lock():
            with self._db.batch(defer_save=True):
                self._remove_from_indexes(record, pk)
                self._remove_from_search(pk)
                del self._primary[pk]

    def increment(self, pk, field, amount=1):
        """Atomically add `amount` to a numeric field (likes / views / …).

        Re-indexes if `field` feeds an index; otherwise a fast in-place field
        write (no full-record round-trip)."""
        pk = str(pk)
        with self._db.write_lock():
            if field in self._indexed_fields:
                cur = self._primary[pk, field]
                return self.update(pk, **{field: int(cur) + amount})[field]
            if pk not in self._primary:
                raise KeyError(pk)
            new = int(self._primary[pk, field]) + amount
            self._primary[pk, field] = new
            return new

    def increment_many(self, pairs, field, amount=1):
        """Add a per-key amount to a numeric ``field`` across many rows at once.

        ``pairs`` is a mapping ``{pk: amount}`` or an iterable of ``(pk, amount)``
        (a bare pk uses ``amount``); amounts for a repeated pk are summed.
        Returns ``{pk: new_value}``.

        Same result as calling ``increment(pk, field, amount)`` on each key, but
        when ``field`` feeds an ordered index (a ``range`` / ``Many`` sort key)
        the re-index does all the entry MOVES in one deferred-write block per
        index — ~2x cheaper per key than one-at-a-time once the index is
        non-empty — and only the counter field is rewritten in place, so
        unchanged blob fields keep their references (no decompress/recompress).
        Ideal for a batched counter (URL/domain hit counts coalesced per flush).
        """
        deltas = {}
        items = pairs.items() if isinstance(pairs, dict) else pairs
        for it in items:
            pk, amt = it if isinstance(it, tuple) else (it, amount)
            pk = str(pk)
            deltas[pk] = deltas.get(pk, 0) + amt
        if not deltas:
            return {}

        affected = [ix for ix in self._indexes.values() if field in ix["sources"]]
        need = {field}
        for ix in affected:
            need.add(ix["field"])
            if ix["spec"].sort is not None:
                need.add(ix["spec"].sort)
        need = list(need)

        results = {}
        with self._db.write_lock():
            with self._db.batch(defer_save=True):
                olds, news = {}, {}
                for pk, dv in deltas.items():
                    rec = self._primary.get_fields(pk, need)
                    if rec is None:
                        raise KeyError(pk)
                    nv = int(rec[field]) + dv
                    olds[pk] = rec
                    news[pk] = {**rec, field: nv}
                    results[pk] = nv

                for ix in affected:
                    struct = ix["struct"]
                    moves = []
                    for pk in deltas:
                        ok = self._index_key(ix, olds[pk], pk)
                        nk = self._index_key(ix, news[pk], pk)
                        if ok != nk:
                            moves.append((ok, nk, pk))

                    if ix["spec"].kind == "unique":
                        for ok, nk, pk in moves:
                            if ok is not None and ok in struct:
                                del struct[ok]
                            if nk is not None:
                                ex = struct.get(nk)
                                if ex is not None and str(ex["pk"]) != pk:
                                    raise ValueError(
                                        f"duplicate value for unique index "
                                        f"{ix['name']!r}"
                                    )
                                struct[nk] = {"pk": pk}
                    elif moves:  # BTree (range/many): batch the moves
                        moves.sort(key=lambda m: (m[1] is None, m[1]))
                        with struct.deferred_node_writes():
                            for ok, nk, pk in moves:
                                if ok is not None and ok in struct:
                                    del struct[ok]
                                if nk is not None:
                                    struct[nk] = {"pk": pk}

                    # Maintained group counter: only moves if the GROUP field
                    # itself changed (i.e. we incremented the group field, not a
                    # separate sort field).
                    if ix.get("counter") is not None and field == ix["field"]:
                        for pk in deltas:
                            ov, nv = olds[pk].get(field), news[pk].get(field)
                            if ov != nv:
                                self._count_add(ix, ov, -1)
                                self._count_add(ix, nv, +1)

                for pk, nv in results.items():
                    self._primary[pk, field] = nv
        return results

    def update_many(self, updates=None, **changes):
        """Update many rows at once, amortising index maintenance.

        Two call forms:

            col.update_many(["p1", "p2", ...], status="done")   # same change
            col.update_many({"p1": {...}, "p2": {...}})          # per-row change
            col.update_many([("p1", {...}), ("p2", {...})])      # per-row (pairs)

        Same result as calling :meth:`update` on each row, but every affected
        ordered index (``Many`` / ``range``) does all its entry MOVES in one
        deferred-write block, keyed in sort order — one amortised set of BTree
        descents instead of one descent per row (the same trick as
        :meth:`increment_many`; ~2x on a sorted index once it is non-empty).
        Counters are batched per group.

        No-op preserving: a field whose new value equals the stored one is
        never written, and a row whose changes are all no-ops costs nothing —
        so a restore/replay pass that re-applies the current state is cheap.

        Missing primary keys raise ``KeyError`` before any write (all-or-nothing
        validation).  Returns the number of rows actually changed.
        """
        # ── normalise both call forms into {pk: changes_dict} ────────────────
        if changes:
            if updates is None:
                raise TypeError(
                    "update_many(ids, **changes) needs an iterable of ids"
                )
            per = {str(pk): changes for pk in updates}
        else:
            if updates is None:
                return 0
            items = updates.items() if isinstance(updates, dict) else updates
            per = {}
            for pk, c in items:
                per[str(pk)] = c
        if not per:
            return 0

        changed_fields = set()
        for c in per.values():
            changed_fields.update(c.keys())
        if self._key_field in changed_fields:
            for pk, c in per.items():
                if self._key_field in c and str(c[self._key_field]) != pk:
                    raise ValueError(
                        "cannot change the primary key via update_many()"
                    )

        affected = [ix for ix in self._indexes.values()
                    if changed_fields & ix["sources"]]
        need = set(changed_fields)
        for ix in affected:
            need.add(ix["field"])
            sort = getattr(ix["spec"], "sort", None)
            if sort is not None:
                need.add(sort)
        search_touched = bool(self._search) and bool(
            changed_fields & self._search_fields)
        if search_touched:
            for si in self._search.values():
                need.update(si["fields"])
        need = list(need)

        with self._db.write_lock():
            with self._db.batch(defer_save=True):
                # Phase 1 — read olds, compute the *effective* (non-no-op)
                # change per row, and cache the merged new dict.
                olds, effective, news = {}, {}, {}
                for pk, c in per.items():
                    old = self._primary.get_fields(pk, need)
                    if old is None:
                        raise KeyError(pk)
                    olds[pk] = old
                    eff = {f: v for f, v in c.items()
                           if _values_differ(v, old.get(f))}
                    effective[pk] = eff
                    if eff:
                        news[pk] = {**old, **eff}

                # Phase 2 — index moves, batched per index in key order.
                for ix in affected:
                    struct = ix["struct"]
                    srcs = ix["sources"]
                    moves = []
                    for pk, eff in effective.items():
                        if not set(eff) & srcs:
                            continue
                        old = olds[pk]
                        new = news[pk]
                        ok = self._index_key(ix, old, pk)
                        nk = self._index_key(ix, new, pk)
                        if ok != nk:
                            moves.append((ok, nk, pk))
                    if ix["spec"].kind == "unique":
                        for ok, nk, pk in moves:
                            if ok is not None and ok in struct:
                                del struct[ok]
                            if nk is not None:
                                ex = struct.get(nk)
                                if ex is not None and str(ex["pk"]) != pk:
                                    raise ValueError(
                                        f"duplicate value for unique index "
                                        f"{ix['name']!r}"
                                    )
                                struct[nk] = {"pk": pk}
                    elif moves:
                        moves.sort(key=lambda m: (m[1] is None, m[1]))
                        with struct.deferred_node_writes():
                            for ok, nk, pk in moves:
                                if ok is not None and ok in struct:
                                    del struct[ok]
                                if nk is not None:
                                    struct[nk] = {"pk": pk}
                    if ix.get("counter") is not None:
                        for pk, eff in effective.items():
                            if ix["field"] not in eff:
                                continue
                            ov = olds[pk].get(ix["field"])
                            nv = eff[ix["field"]]
                            if ov != nv:
                                self._count_add(ix, ov, -1)
                                self._count_add(ix, nv, +1)

                # Phase 3 — full-text re-index for rows whose search fields moved.
                if search_touched:
                    for si in self._search.values():
                        sf = frozenset(si["fields"])
                        for pk, eff in effective.items():
                            if not set(eff) & sf:
                                continue
                            new = news[pk]
                            entry = si["pk2docid"].get(pk)
                            if entry is not None:
                                si["index"].delete(int(entry["doc_id"]))
                            text = " ".join(str(new.get(f, "")) for f in sf)
                            doc_id = si["index"].add(None, text=text)
                            si["docid2pk"].append({"pk": pk})
                            si["pk2docid"][pk] = {"doc_id": doc_id}

                # Phase 4 — write only the fields that actually changed.
                n = 0
                for pk, eff in effective.items():
                    if not eff:
                        continue
                    for f, v in eff.items():
                        self._primary[pk, f] = v
                    n += 1
        return n

    def reindex(self):
        """Rebuild every secondary index from the primary store (O(n))."""
        with self._db.write_lock():
            with self._db.batch():
                # counters rebuild from scratch (adds below re-count each record)
                for ix in self._indexes.values():
                    cnt = ix.get("counter")
                    if cnt is not None:
                        for gk in list(cnt.keys()):
                            del cnt[gk]
                for pk, record in self._primary.items():
                    self._add_to_indexes(record, str(pk))

    # ── reads ────────────────────────────────────────────────────────────

    def _coerce_field_value(self, field, value):
        """Coerce a query value to the field's numeric type so it encodes the
        same way the stored values do (an int field encodes ints; a float field
        encodes floats — passing 40 to a float index would otherwise mis-sort)."""
        if value is None or field not in self.dataset.user_schema.names:
            return value
        if field in getattr(self.dataset, "_datetime_fields", set()):
            return value   # datetime / ISO str handled by encode_value
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            kind = self.dataset.user_schema.fields[field][0].kind
            if kind == "f":
                return float(value)
            if kind in ("i", "u"):
                return int(value)
        return value

    def _wrap(self, pk, record):
        return Record(self, str(pk), record) if record is not None else None

    def __getitem__(self, pk):
        # col[pk, field] → just that field's value (no full-record read)
        if isinstance(pk, tuple) and len(pk) == 2:
            key, field = pk
            return self._primary[str(key), field]
        return self._wrap(pk, self._primary[str(pk)])

    def __setitem__(self, key, value):
        """``col[pk, field] = value`` — update a single field by primary key.

        Fast in-place field write when the field feeds no index; routed through
        :meth:`update` (which re-indexes) when it backs a secondary or full-text
        index. Raises ``KeyError`` if ``pk`` doesn't exist, ``ValueError`` for
        the primary-key field. (Assigning a whole record — ``col[pk] = {...}`` —
        is not supported; use :meth:`insert` / :meth:`update`.)
        """
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                "assign a single field: col[pk, field] = value "
                "(use insert()/update() for whole records)"
            )
        pk, field = str(key[0]), key[1]
        if field == self._key_field:
            raise ValueError("cannot change the primary key")
        if field in self._indexed_fields or field in self._search_fields:
            self.update(pk, **{field: value})
            return
        with self._db.write_lock():
            if pk not in self._primary:
                raise KeyError(pk)
            self._primary[pk, field] = value

    def __delitem__(self, pk):
        """``del col[pk]`` — alias for :meth:`delete` (raises KeyError if absent)."""
        self.delete(pk)

    def get_primary(self, pk, default=None):
        rec = self._primary.get(str(pk))
        return self._wrap(pk, rec) if rec is not None else default

    def get(self, index_name, value, default=None):
        """Unique lookup → the single record (or default)."""
        ix = self._indexes[index_name]
        if ix["spec"].kind != "unique":
            raise ValueError(
                f"get() needs a 'unique' index; use find()/range() for {index_name!r}"
            )
        value = self._coerce_field_value(ix["field"], value)
        entry = ix["struct"].get(self._group_key(ix, value))
        if entry is None:
            return default
        return self.get_primary(entry["pk"], default)

    def _many_bounds(self, index_name, op, value, start, end):
        """Resolve a Many-index group + optional sort-field window into the
        composite-key interval [low_key, high_key) shared by find()/count()."""
        ix = self._indexes[index_name]
        spec = ix["spec"]
        if spec.kind != "many":
            raise ValueError(f"{op}() needs a 'many' index for {index_name!r}")
        value = self._coerce_field_value(ix["field"], value)
        group = self._group_key(ix, value)

        if start is None and end is None:
            return ix, group + _SEP, group + "\x01"

        if spec.sort is None:
            raise ValueError(
                f"start/end need a Many(sort=...) index; {index_name!r} has no sort"
            )
        es = (encode_value(self._coerce_field_value(spec.sort, start), desc=spec.desc)
              if start is not None else None)
        ee = (encode_value(self._coerce_field_value(spec.sort, end), desc=spec.desc)
              if end is not None else None)
        if es is not None and ee is not None:
            klo, khi = min(es, ee), max(es, ee)   # enc is monotone → value interval
        elif spec.desc:
            klo, khi = ee, es   # value>=start ↔ enc<=es ; value<=end ↔ enc>=ee
        else:
            klo, khi = es, ee
        low_key = group + _SEP + (klo if klo is not None else "")
        high_key = (group + _SEP + khi + "\x01") if khi is not None else group + "\x01"
        return ix, low_key, high_key

    def count(self, index_name, value, start=None, end=None):
        """Number of records in group ``value`` of a 'many' index.

        On a ``Many(counted=True)`` index the unwindowed count is **O(1)**
        (read from the maintained group counter).  Otherwise — or when
        ``start``/``end`` bound the sort field — it is a key-only scan that
        never reads index values or primary records:

            col.count("narrative", "ukraine", start=date(2026, 6, 1))

        O(log n + matches) with only the key walk paid per match."""
        if start is None and end is None:
            ix = self._indexes[index_name]
            if ix["spec"].kind == "many" and ix.get("counter") is not None:
                value = self._coerce_field_value(ix["field"], value)
                rec = ix["counter"].get_fields(self._group_key(ix, value), ["n"])
                return int(rec["n"]) if rec is not None else 0
        ix, low_key, high_key = self._many_bounds(index_name, "count",
                                                  value, start, end)
        it = ix["struct"].range_keys(low_key, high_key, inclusive=(True, False))
        return sum(1 for _ in it)

    def groups(self, index_name, order_by="count", desc=True, limit=None):
        """All groups of a ``Many(counted=True)`` index with their sizes —
        ``[(value, count), ...]`` — read from the maintained counters, without
        touching a single record.

            col.groups("narrative")                       # biggest first
            col.groups("narrative", order_by="value", desc=False)

        ``order_by`` is "count" or "value"."""
        ix = self._indexes[index_name]
        if ix["spec"].kind != "many" or ix.get("counter") is None:
            raise ValueError(
                f"groups() needs a Many(counted=True) index for {index_name!r}"
            )
        out = [(rec["value"], int(rec["n"])) for _k, rec in ix["counter"].items()]
        if order_by == "count":
            out.sort(key=lambda t: (t[1], str(t[0])), reverse=desc)
        elif order_by == "value":
            out.sort(key=lambda t: t[0], reverse=desc)
        else:
            raise ValueError("order_by must be 'count' or 'value'")
        return out[:limit] if limit is not None else out

    def _vector_candidates(self, where):
        """Resolve `where` into (candidate pk iterator, residual predicate).

        Picks the most selective indexed entry of `where` to drive the scan —
        preferring a many/unique equality (and folding a bound on the many's
        sort field into the same seek), then a range index — and returns the
        untouched entries as the residual, applied per-record later.  With no
        usable index (or a callable where), every pk is a candidate."""
        if where is None or callable(where):
            return iter(self._primary.keys()), where
        residual = dict(where)
        best = None
        for idx_name, ix in self._indexes.items():
            spec = ix["spec"]
            if ix["field"] not in residual:
                continue
            is_range = (isinstance(residual[ix["field"]], tuple)
                        and len(residual[ix["field"]]) == 2)
            if spec.kind in ("unique", "many") and not is_range:
                sort_bounded = (spec.kind == "many" and spec.sort is not None
                                and isinstance(residual.get(spec.sort), tuple))
                rank = 3 if sort_bounded else 2
            elif spec.kind == "range" and is_range:
                rank = 1
            else:
                continue
            if best is None or rank > best[0]:
                best = (rank, idx_name)
        if best is None:
            return iter(self._primary.keys()), (residual or None)

        ix = self._indexes[best[1]]
        spec = ix["spec"]
        if spec.kind == "unique":
            value = residual.pop(ix["field"])
            value = self._coerce_field_value(ix["field"], value)
            entry = ix["struct"].get(self._group_key(ix, value))
            pks = [str(entry["pk"])] if entry is not None else []
            return iter(pks), (residual or None)
        if spec.kind == "many":
            value = residual.pop(ix["field"])
            start = end = None
            if spec.sort is not None and isinstance(residual.get(spec.sort), tuple):
                start, end = residual.pop(spec.sort)
            _, low_key, high_key = self._many_bounds(best[1], "nearest",
                                                     value, start, end)
            it = ix["struct"].range(low_key, high_key, inclusive=(True, False))
            return (str(e["pk"]) for _k, e in it), (residual or None)
        # range index
        lo, hi = residual.pop(ix["field"])
        lo = self._coerce_field_value(ix["field"], lo)
        hi = self._coerce_field_value(ix["field"], hi)
        start = None if lo is None else encode_value(lo)
        end = None if hi is None else encode_value(hi) + "\x01"
        it = ix["struct"].range(start, end, inclusive=(True, False))
        return (str(e["pk"]) for _k, e in it), (residual or None)

    def nearest(self, index_name, query, k=10, where=None, fields=None,
                with_scores=False):
        """Exact vector similarity → the k records closest to ``query``.

        A pre-filtered flat search, not ANN: ``where`` narrows candidates
        through the collection's regular indexes first (same spec as
        ``search(where=...)`` — ``{field: value}`` equality, ``{field:
        (lo, hi)}`` range, or a callable), then only the survivors' vectors
        are read (a projected row read each — never the full record) and
        scored exactly with the index's metric.  Full records are
        materialized for the k winners only.

        Example::

            col.nearest("emb", qvec, k=10)                # whole collection
            col.nearest("emb", qvec, k=10,
                        where={"topic": "politics",
                               "created_at": (date(2026, 1, 1), None)})

        cosine / dot rank descending (higher = closer); l2 ranks ascending
        and ``with_scores`` returns the actual distance."""
        vx = self._vector[index_name]
        vec_field, metric = vx["field"], vx["metric"]
        q = np.asarray(query, dtype=np.float32).ravel()

        pk_iter, residual = self._vector_candidates(where)
        pred = self._make_predicate(residual)
        if callable(residual):
            read_cols = None                      # predicate needs full records
        else:
            read_cols = [vec_field] + [f for f in (residual or {})]

        cand_pks, rows = [], []
        for pk in pk_iter:
            rec = (self._primary.get(pk) if read_cols is None
                   else self._primary.get_fields(pk, read_cols))
            if rec is None:
                continue
            if pred is not None and not pred(rec):
                continue
            cand_pks.append(pk)
            rows.append(rec[vec_field])
        if not cand_pks:
            return []

        M = np.stack(rows).astype(np.float32, copy=False)
        if metric == "cosine":
            qn = float(np.linalg.norm(q)) or 1.0
            norms = np.linalg.norm(M, axis=1)
            norms[norms == 0] = 1.0
            scores = (M @ q) / (norms * qn)
            ascending = False
        elif metric == "dot":
            scores = M @ q
            ascending = False
        else:                                     # l2 → distance, lower wins
            d = M - q
            scores = np.sqrt(np.einsum("ij,ij->i", d, d))
            ascending = True

        kk = min(k, len(cand_pks))
        key = scores if ascending else -scores
        top = np.argpartition(key, kk - 1)[:kk]
        top = top[np.argsort(key[top])]

        out = []
        for i in top:
            pk = cand_pks[int(i)]
            rec = (self._primary.get_fields(pk, fields) if fields is not None
                   else self._primary.get(pk))
            if rec is None:
                continue
            wrapped = self._wrap(pk, rec)
            out.append((wrapped, float(scores[int(i)])) if with_scores
                       else wrapped)
        return out

    def _sort_value_decoder(self, field, desc):
        """Return a fn ``enc_str -> value`` inverting encode_value for the sort
        field, matching the Python scalar a record read yields — or None when
        the dtype isn't losslessly recoverable from the key."""
        if field in self.dataset._datetime_fields:
            return lambda enc: _decode_datetime(enc, desc)
        try:
            kind = self.dataset.user_schema.fields[field][0].kind
        except (KeyError, IndexError, AttributeError):
            return None
        if kind in ("i", "u"):
            return lambda enc: _decode_int(enc, desc)
        if kind == "f":
            return lambda enc: _decode_float(enc, desc)
        if kind in ("S", "U"):
            return lambda enc: _decode_str(enc, desc)
        return None

    def _pk_field_coercer(self):
        """How to turn the stored (stringified) pk back into the primary-key
        field's own dtype, so an index-only projection matches a record read —
        or None if that dtype can't be recovered from ``str(pk)``."""
        field = self._key_field
        if field in self.dataset._datetime_fields:
            return None
        try:
            kind = self.dataset.user_schema.fields[field][0].kind
        except (KeyError, IndexError, AttributeError):
            return None
        if kind in ("S", "U"):
            return str
        if kind in ("i", "u"):
            return lambda s: int(s)
        if kind == "f":
            return lambda s: float(s)
        return None

    def _pk_in_key(self, ix):
        """Can this index's stored keys be trusted to still END with the pk?

        ``_index_key`` appends the pk last, so a scan can read it off the key
        and skip the index entry entirely.  That only holds while no key was
        clipped into the BTree's fixed utf8[key_size] slot — and a collection
        created before ``DB._index_key_width`` accounted for desc sort parts (4
        utf8 bytes per character, not 1) may well hold clipped keys.  Reading a
        clipped pk would resolve to a *different* record, so those files keep
        the old path and take the pk from the index entry.

        Cached on the index dict: it is a property of the schema, not of a call.
        """
        cached = ix.get("pk_in_key")
        if cached is None:
            from loom.database import DB
            spec = ix["spec"]
            pk_w = DB._field_enc_width(self.dataset, self._key_field)
            if spec.kind not in ("many", "range") or pk_w is None:
                cached = False
            else:
                need = DB._index_key_width(self.dataset, spec, ix["field"],
                                           pk_w, bool(ix.get("hashed")))
                cached = (need is not None
                          and need <= getattr(ix["struct"], "_key_size", 0))
            ix["pk_in_key"] = cached
        return cached

    def _walk(self, ix, *args, **kwargs):
        """Yield ``(key, pk)`` over an index range.

        Off the key alone when :meth:`_pk_in_key` allows it — the index entry's
        own record is then never read, 2.76 -> 0.46 us per hit on a 12k-row
        group — and off the entry otherwise.
        """
        bt = ix["struct"]
        if self._pk_in_key(ix):
            for key in bt.range_keys(*args, **kwargs):
                yield key, key.rsplit(_SEP, 1)[-1]
        else:
            for key, entry in bt.range(*args, **kwargs):
                yield key, str(entry["pk"])

    def _index_projection(self, ix, value, fields):
        """If every requested field is derivable from the index entry + its
        composite key alone, return a fn ``(key, entry) -> dict``; else None.

        A ``Many`` composite key is ``group_enc \\x00 [sort_enc \\x00] pk``, so
        for a group query we can serve, from the key alone and with **no index
        entry, no Dict lookup and no record read**:

          • the primary key   — the key's last part (coerced to its dtype);
          • the group field    — it equals the (coerced) query ``value`` for
                                 every hit, when the group is order-preserving
                                 (encode_value is injective; a *hashed* group is
                                 lossy, so we don't serve it);
          • the sort field     — decoded from the key for int/float/str/datetime
                                 dtypes (see _sort_value_decoder).

        Any other requested field forces the normal record-read path (None)."""
        spec = ix["spec"]
        if spec.kind != "many":
            return None
        fset = set(fields)
        if not fset:
            return None

        key_field = self._key_field
        group_field = ix["field"]
        servable = set()

        pk_coerce = None
        if key_field in fset:
            pk_coerce = self._pk_field_coercer()
            if pk_coerce is None:
                return None
            servable.add(key_field)

        coerced_group = None
        if group_field in fset:
            if ix.get("hashed"):
                return None            # hashed group value isn't recoverable
            coerced_group = self._coerce_field_value(group_field, value)
            servable.add(group_field)

        sort_field = spec.sort
        sort_dec = None
        if sort_field is not None and sort_field in fset:
            sort_dec = self._sort_value_decoder(sort_field, spec.desc)
            if sort_dec is None:
                return None
            servable.add(sort_field)

        if fset - servable:
            return None

        want_pk = key_field in fset
        want_group = group_field in fset
        want_sort = sort_field in fset if sort_field is not None else False
        ordered = list(fields)

        def project(key, pk):
            rec = {}
            parts = None
            for f in ordered:
                if want_pk and f == key_field:
                    rec[f] = pk_coerce(pk)
                elif want_group and f == group_field:
                    rec[f] = coerced_group
                else:                       # sort field
                    if parts is None:
                        parts = key.split(_SEP)
                    rec[f] = sort_dec(parts[1])
            return rec

        return project

    def _can_fused(self, ix):
        """True when the Cython fused pipeline can replace the whole scan."""
        if _accel_fused is None:
            return False
        bt = ix["struct"]
        if bt._node_layout() is None:
            return False
        if bt._dirty_nodes is not None:
            return False
        if bt._int_keys:
            return False
        d = self._primary
        if getattr(d, "_hash_keys", False) or d._hash_key_fn is not None:
            return False
        return True

    def _fused_collect(self, ix, low_key, high_key, fields, limit):
        """Run the Cython fused pipeline and materialise into Records."""
        bt = ix["struct"]
        layout = bt._node_layout()
        _, _, leaf_off, nk_off, key_off, key_w, child_off = layout
        ds_node = bt._node_dataset
        d = self._primary
        p_init = getattr(d, "_p_init", d.P_INIT)
        n_t = d.p_last - p_init + 1
        ta = np.array(d.table_addrs[:n_t], dtype=np.int64)
        caps = np.array([d._get_capacity(p_init + t)
                         for t in range(n_t)], dtype=np.int64)
        prs = np.array([d._get_probe_range(p_init + t)
                        for t in range(n_t)], dtype=np.int32)
        ht = d._hash_table
        vo = ht.schema.fields["valid"][1]
        ds_data = d._values_dataset

        pk_list, block = _accel_fused(
            ds_node.db.mapped_file, bt.root_addr, ds_node.record_size,
            leaf_off, nk_off, key_off, key_w, child_off,
            low_key.encode("utf-8"), high_key.encode("utf-8"), True, False,
            ht.db.mapped_file, n_t, ta, caps, prs, ht.record_size, vo,
            ds_data.db.mapped_file, ds_data.record_size,
        )
        if limit is not None:
            pk_list = pk_list[:limit]
            block = block[:limit]
        if not pk_list:
            return []
        recs = ds_data.read_fields_many(None, fields, _block=block)
        return [self._wrap(pk.decode("utf-8"), rec)
                for pk, rec in zip(pk_list, recs)]

    def _fused_collect_columns(self, ix, low_key, high_key, fields, limit):
        """Fused pipeline → dict of columns (no Record wrapping)."""
        bt = ix["struct"]
        layout = bt._node_layout()
        _, _, leaf_off, nk_off, key_off, key_w, child_off = layout
        ds_node = bt._node_dataset
        d = self._primary
        p_init = getattr(d, "_p_init", d.P_INIT)
        n_t = d.p_last - p_init + 1
        ta = np.array(d.table_addrs[:n_t], dtype=np.int64)
        caps = np.array([d._get_capacity(p_init + t)
                         for t in range(n_t)], dtype=np.int64)
        prs = np.array([d._get_probe_range(p_init + t)
                        for t in range(n_t)], dtype=np.int32)
        ht = d._hash_table
        vo = ht.schema.fields["valid"][1]
        ds_data = d._values_dataset

        pk_list, block = _accel_fused(
            ds_node.db.mapped_file, bt.root_addr, ds_node.record_size,
            leaf_off, nk_off, key_off, key_w, child_off,
            low_key.encode("utf-8"), high_key.encode("utf-8"), True, False,
            ht.db.mapped_file, n_t, ta, caps, prs, ht.record_size, vo,
            ds_data.db.mapped_file, ds_data.record_size,
        )
        if limit is not None:
            pk_list = pk_list[:limit]
            block = block[:limit]
        if not pk_list:
            return {f: [] for f in fields}
        arr = block.view(ds_data.schema).reshape(-1)
        cols = {}
        for f in fields:
            col = arr[f]
            if f in ds_data._text_fields:
                codec = ds_data._blob_codec_of[f]
                cols[f] = [
                    "" if (int(v["offset"]) == 0 and int(v["n_slots"]) == 0)
                    else ds_data.blob_store.read(
                        int(v["offset"]), compression=codec
                    ).decode("utf-8")
                    for v in col
                ]
            elif f in ds_data._json_fields:
                import json as _json
                codec = ds_data._blob_codec_of[f]
                cols[f] = [
                    None if (int(v["offset"]) == 0 and int(v["n_slots"]) == 0)
                    else _json.loads(ds_data.blob_store.read(
                        int(v["offset"]), compression=codec
                    ).decode("utf-8"))
                    for v in col
                ]
            elif f in ds_data._blob_fields:
                codec = ds_data._blob_codec_of[f]
                cols[f] = [
                    None if (int(v["offset"]) == 0 and int(v["n_slots"]) == 0)
                    else ds_data.blob_store.read(int(v["offset"]), compression=codec)
                    for v in col
                ]
            elif f in ds_data._utf8_fields:
                cols[f] = [v.decode("utf-8") for v in col.tolist()]
            elif f in ds_data._datetime_fields:
                cols[f] = [_micros_to_dt(v) for v in col.tolist()]
            elif f in ds_data._array_fields:
                cols[f] = [np.array(v) for v in col]
            else:
                cols[f] = col.tolist()
        cols[self._key_field] = [pk.decode("utf-8") for pk in pk_list]
        return cols

    def find(self, index_name, value, start=None, end=None, limit=None,
             fields=None, as_columns=False):
        """One-to-many lookup → records for group ``value`` (ordered by the
        index's sort field).

        ``start``/``end`` bound that sort field — a compound *equality AND
        range* query.  For a ``Many(sort="created_at")`` index:

            find("category_alias", "politics", start=date(2026, 1, 1))

        runs as a single seek + bounded scan (O(log n + matches)), so it stays
        fast no matter how much history the group holds.  Bounds are inclusive
        and may be int / float / datetime / str (matching the sort field).

        ``fields=["name", ...]`` projects each hit onto just those fields:
        one row read per record, and unrequested text/json/blob fields never
        touch the blob store — much cheaper than materializing full records
        when the schema carries heavy text.

        ``as_columns=True`` returns a ``{field: list}`` dict of columns rather
        than a list of Record objects — faster when the caller consumes fields
        independently (aggregation, counting, filtering by column)."""
        ix, low_key, high_key = self._many_bounds(index_name, "find",
                                                  value, start, end)
        # ── fused Cython pipeline (single nogil block) ──────────────────
        if (fields is not None and self._can_fused(ix)
                and self._index_projection(ix, value, fields) is None):
            if as_columns:
                return self._fused_collect_columns(
                    ix, low_key, high_key, fields, limit)
            return self._fused_collect(ix, low_key, high_key, fields, limit)
        # ── standard path ───────────────────────────────────────────────
        it = self._walk(ix, low_key, high_key, inclusive=(True, False))
        project = (self._index_projection(ix, value, fields)
                   if fields is not None else None)
        return self._collect(it, limit, fields, project)

    def _collect(self, pairs, limit, fields, project=None):
        """Materialize (full or projected) records for a ``(key, pk)`` scan.

        ``project`` (from :meth:`_index_projection`) serves each hit from the
        key alone — no primary-store lookup, no record read — when every
        requested field lives in the index.  Otherwise the hits are read in
        batches (:meth:`Dict.get_fields_many`): one gather out of the mmap
        per batch instead of one row read per hit.

        A hit whose record is missing from the primary store is a stale index
        entry — skipped, and not counted against ``limit``, which is why a
        bounded scan tops its batch up rather than slicing the walk once."""
        batch_size = 8192 if limit is None else min(limit, 8192)
        out, it = [], iter(pairs)
        while True:
            want = batch_size if limit is None else min(batch_size, limit - len(out))
            if want <= 0:
                break
            batch = list(islice(it, want))
            if not batch:
                break
            pks = [pk for _key, pk in batch]
            if project is not None:
                out.extend(self._wrap(pk, project(key, pk)) for key, pk in batch)
            elif fields is not None:
                for pk, rec in zip(pks, self._primary.get_fields_many(pks, fields)):
                    if rec is not None:
                        out.append(self._wrap(pk, rec))
            else:
                for pk in pks:
                    rec = self._primary.get(pk)
                    if rec is not None:
                        out.append(self._wrap(pk, rec))
        return out

    def range(self, index_name, low=None, high=None, limit=None, desc=False,
              fields=None):
        """Range scan on a 'range' index → records with low <= value <= high
        (either bound may be None for an open end).

        desc=True returns highest-value-first — e.g. the most recent items of a
        timestamp index, or a relevance feed, with a cheap `limit` and no need
        for a grouping field:  inbox.range("created_at", limit=50, desc=True).

        ``fields=[...]`` projects hits onto just those fields (see find())."""
        ix = self._indexes[index_name]
        if ix["spec"].kind != "range":
            raise ValueError(f"range() needs a 'range' index for {index_name!r}")
        low = self._coerce_field_value(ix["field"], low)
        high = self._coerce_field_value(ix["field"], high)
        start = None if low is None else encode_value(low)
        end = None if high is None else encode_value(high) + "\x01"
        it = self._walk(ix, start, end, inclusive=(True, False), reverse=desc)
        return self._collect(it, limit, fields)

    def latest(self, index_name, fields=None):
        """The record with the **highest** value of a 'range' index — e.g.
        the most recent report of a ``created_at`` index — or None if the
        collection is empty.  Sugar over ``range(..., limit=1, desc=True)``
        that returns the record directly instead of a one-element list."""
        hits = self.range(index_name, limit=1, desc=True, fields=fields)
        return hits[0] if hits else None

    def first(self, index_name, fields=None):
        """The record with the **lowest** value of a 'range' index (the
        oldest, the cheapest, ...) — or None if the collection is empty."""
        hits = self.range(index_name, limit=1, desc=False, fields=fields)
        return hits[0] if hits else None

    def search(self, index_name, query, where=None, mode=None, limit=None,
               with_scores=False):
        """Full-text search on a 'search' index → matching records.

        Boolean (AND/OR/AND NOT, parens, `*`) by default; ranked (bm25/tfidf)
        if the index was declared with scoring="bm25".  with_scores → list of
        (record, score).

        ``where`` filters the (relevance-ordered) hits by record fields — a
        full-text query AND structured constraints in one call:

            posts.search("body", "lait infantile",
                         where={"category_alias": "health",
                                "created_at": (date(2026, 1, 1), None)})

        It is a dict ``{field: value}`` (equality) / ``{field: (lo, hi)}``
        (inclusive range, None = open) — or any ``callable(record) -> bool``.
        The filter is applied AFTER ranking, so a selective query term keeps it
        cheap; ``limit`` is applied after filtering."""
        si = self._search.get(index_name)
        if si is None:
            raise KeyError(f"no full-text index {index_name!r}")
        pred = self._make_predicate(where)
        # When filtering, don't let the engine pre-truncate to `limit` — we need
        # the full ranked candidate list to filter, then cap.
        eng_limit = None if pred is not None else limit
        res = si["index"].search(
            query, return_ids=True, mode=mode, limit=eng_limit, with_scores=with_scores
        )
        d2p = si["docid2pk"]
        out = []
        for item in res:
            doc_id, score = item if with_scores else (item, None)
            rec = self.get_primary(d2p[int(doc_id)]["pk"])
            if rec is None or (pred is not None and not pred(rec)):
                continue
            out.append((rec, score) if with_scores else rec)
            if limit is not None and len(out) >= limit:
                break
        return out

    # ── pk-set extraction (for query() intersection) ──────────────────────

    def _pk_set(self, index_name, value, start=None, end=None):
        """Return a sorted numpy int64 array of pk *hashes* for one filter.

        For a Many index this walks the keys (or entries) and collects pks.
        For a range index it scans the bounded key range.
        Pks are returned as their murmur128 hi half (int64) for fast
        sorted intersection — the final materialisation resolves them back.
        """
        ix = self._indexes[index_name]
        spec = ix["spec"]
        if spec.kind == "many":
            _, lo, hi = self._many_bounds(index_name, "query", value, start, end)
            pks = [pk for _key, pk in self._walk(ix, lo, hi, inclusive=(True, False))]
        elif spec.kind == "range":
            low = self._coerce_field_value(ix["field"], value[0] if isinstance(value, tuple) else value)
            high = self._coerce_field_value(ix["field"], value[1] if isinstance(value, tuple) else None)
            lo = None if low is None else encode_value(low)
            hi = None if high is None else encode_value(high) + "\x01"
            pks = [pk for _key, pk in self._walk(ix, lo, hi, inclusive=(True, False))]
        else:
            raise ValueError(
                f"query() needs a 'many' or 'range' index for {index_name!r}, "
                f"got {spec.kind!r}"
            )
        return pks

    def _search_pk_set(self, index_name, query_text, mode=None):
        """Return the list of pks matching a full-text query."""
        si = self._search.get(index_name)
        if si is None:
            raise KeyError(f"no full-text index {index_name!r}")
        doc_ids = si["index"].search(
            query_text, return_ids=True, mode=mode or "boolean"
        )
        d2p = si["docid2pk"]
        return [str(d2p[int(did)]["pk"]) for did in doc_ids]

    def _count_estimate(self, ix_name, value):
        """Cheap group-size estimate: counted index → O(1), else None."""
        ix = self._indexes.get(ix_name)
        if ix is None:
            return None
        spec = ix["spec"]
        if spec.kind == "many" and getattr(spec, "counted", False):
            try:
                return self.count(ix_name, value)
            except Exception:
                return None
        return None

    def query(self, fields=None, as_columns=False, limit=None, search=None,
              **filters):
        """Multi-index intersection query — filter by N indexes at once.

        Each keyword argument is ``index_name=value``: the cheapest index is
        walked to collect pks, and the remaining filters are verified on the
        materialised records.  This avoids scanning N full pk sets when only
        one index is selective.

        Strategy: use counted-index cardinalities (O(1)) to pick the
        **smallest** group, extract its pks, materialise their records (with
        the filter fields included in the projection), and post-filter the
        rest in Python on the already-read records.  When the lead set is
        1/20th of the total, this reads 1/20th of the records instead of all
        of them — and the post-filter costs nothing because the fields are
        already in hand.

        ``search`` takes a ``(index_name, query_text)`` tuple (or a
        ``(index_name, query_text, mode)`` triple) to add a full-text filter
        to the intersection — structured + text in one call::

            col.query(country="FR", model="gpt-4",
                      search=("body", "carbon neutral"),
                      fields=["id", "title"])

        For a range index, pass a tuple ``(low, high)`` as the value (either
        may be None for an open bound)::

            col.query(category="tech", engagement=(8000, None),
                      fields=["id", "engagement"])

        ``fields``, ``as_columns``, ``limit`` behave like ``find()``.
        """
        if not filters and search is None:
            raise ValueError("query() needs at least one filter or search term")

        # ── resolve index names (accept field names too) ────────────────
        field_to_ix = {}
        for ix_name, ix in self._indexes.items():
            field_to_ix.setdefault(ix["field"], ix_name)

        resolved = []  # [(ix_name, field, value, estimated_count)]
        for name, value in filters.items():
            ix_name = name if name in self._indexes else field_to_ix.get(name)
            if ix_name is None:
                raise KeyError(
                    f"no index named {name!r} and no index on field {name!r}; "
                    f"available indexes: {list(self._indexes)}"
                )
            field = self._indexes[ix_name]["field"]
            est = self._count_estimate(ix_name, value)
            resolved.append((ix_name, field, value, est if est is not None else float("inf")))

        # ── pick the lead index (smallest estimated group) ──────────────
        resolved.sort(key=lambda r: r[3])
        lead_ix, lead_field, lead_value, _ = resolved[0]
        post_filters = resolved[1:]  # these will be checked on the records

        # ── full-text: if present, it might be the most selective ───────
        search_pks = None
        if search is not None:
            if isinstance(search, (list, tuple)):
                s_name, s_query = search[0], search[1]
                s_mode = search[2] if len(search) > 2 else None
            else:
                raise TypeError("search must be (index_name, query) or "
                                "(index_name, query, mode)")
            search_pks = set(self._search_pk_set(s_name, s_query, s_mode))

        # ── extract pks from the lead index ─────────────────────────────
        lead_pks = self._pk_set(lead_ix, lead_value)
        if search_pks is not None:
            lead_pks = [pk for pk in lead_pks if pk in search_pks]

        if not lead_pks:
            if as_columns:
                return {f: [] for f in (fields or [])}
            return []

        # ── materialise with filter fields in the projection ────────────
        filter_fields = {f for _, f, _, _ in post_filters}
        if fields is not None:
            read_fields = list(dict.fromkeys(list(fields) + list(filter_fields)))
        else:
            read_fields = None  # full records

        if read_fields is not None:
            recs = self._primary.get_fields_many(lead_pks, read_fields)
        else:
            recs = [self._primary.get(pk) for pk in lead_pks]

        # ── post-filter on the remaining criteria ───────────────────────
        result_pks, result_recs = [], []
        for pk, rec in zip(lead_pks, recs):
            if rec is None:
                continue
            ok = True
            for _, fld, val, _ in post_filters:
                rv = rec.get(fld)
                if isinstance(val, tuple) and len(val) == 2:
                    lo, hi = val
                    if lo is not None and rv < lo:
                        ok = False; break
                    if hi is not None and rv > hi:
                        ok = False; break
                elif rv != val:
                    ok = False; break
            if ok:
                result_pks.append(pk)
                result_recs.append(rec)

        if limit is not None:
            result_pks = result_pks[:limit]
            result_recs = result_recs[:limit]

        if not result_pks:
            if as_columns:
                return {f: [] for f in (fields or [])}
            return []

        # ── format output ───────────────────────────────────────────────
        if as_columns and fields is not None:
            cols = {f: [] for f in fields}
            cols[self._key_field] = []
            for pk, rec in zip(result_pks, result_recs):
                cols[self._key_field].append(pk)
                for f in fields:
                    cols[f].append(rec.get(f))
            return cols

        if fields is not None:
            # strip extra filter fields from the output records
            if filter_fields - set(fields):
                return [self._wrap(pk, {f: rec[f] for f in fields if f in rec})
                        for pk, rec in zip(result_pks, result_recs)]
            return [self._wrap(pk, rec)
                    for pk, rec in zip(result_pks, result_recs)]

        return [self._wrap(pk, rec)
                for pk, rec in zip(result_pks, result_recs)]

    @staticmethod
    def _make_predicate(where):
        """Build a record→bool filter from a dict spec or a callable (or None)."""
        if where is None:
            return None
        if callable(where):
            return where
        checks = []
        for field, cond in where.items():
            if isinstance(cond, tuple) and len(cond) == 2:
                checks.append((field, "range", cond[0], cond[1]))
            else:
                checks.append((field, "eq", cond, None))
        def pred(rec):
            for field, kind, a, b in checks:
                v = rec.get(field)
                if kind == "eq":
                    if v != a:
                        return False
                else:
                    if a is not None and v < a:
                        return False
                    if b is not None and v > b:
                        return False
            return True
        return pred

    def __contains__(self, pk):
        return str(pk) in self._primary

    def __len__(self):
        return len(self._primary)

    def __iter__(self):
        """Iterate over records (a Collection is a record store, so this yields
        records — use keys()/items() for primary keys or (pk, record) pairs)."""
        return self.values()

    def keys(self):
        return self._primary.keys()

    def values(self):
        for pk, rec in self._primary.items():
            yield self._wrap(pk, rec)

    def items(self):
        for pk, rec in self._primary.items():
            yield str(pk), self._wrap(pk, rec)

    def sample(self, n=10, random=True, seed=None):
        """Return up to ``n`` records — a quick peek at what the collection holds.

        Handy for understanding the data (e.g. handing an LLM a few example
        records to infer the shape, field meanings and value ranges).

        Args:
            n: Maximum number of records to return.
            random: If True (default), a uniform random sample (reservoir
                sampling — one full pass, no count needed). If False, the first
                ``n`` records in store iteration order (fast, stops early, no
                full scan).
            seed: Optional int for a reproducible random sample.

        Returns:
            A list of records (fewer than ``n`` if the collection is smaller).
        """
        from loom.sampling import reservoir_sample

        return reservoir_sample(self.values(), n, random=random, seed=seed)

    def describe(self, n=3, seed=None):
        """Return a compact, prompt-ready text summary of this collection.

        Includes the record count, primary key, typed schema, declared indexes
        (and full-text indexes) and a few sample records — enough for an
        agent/LLM to grasp what the collection holds in one read.
        """
        from loom.datastructures.base import DataStructure

        schema = DataStructure._extract_schema(self.dataset)
        total = len(self)
        lines = [f"Collection {self.name!r} — {total} record(s), key={self._key_field!r}"]
        if schema:
            lines.append("schema:")
            for field, dtype in schema.items():
                lines.append(f"  {field}: {dtype}")
        if self._indexes:
            lines.append("indexes:")
            for iname, ix in self._indexes.items():
                spec = ix["spec"]
                extra = ""
                if spec.kind == "many" and getattr(spec, "sort", None):
                    extra = f" (sort={spec.sort}{', desc' if spec.desc else ''})"
                on = "" if iname == ix["field"] else f" on {ix['field']!r}"
                lines.append(f"  {iname}: {spec.kind}{extra}{on}")
        if self._search:
            lines.append("full-text indexes:")
            for sname, si in self._search.items():
                lines.append(f"  {sname}: search(fields={si['fields']})")
        recs = self.sample(n, seed=seed)
        if recs:
            lines.append(f"sample ({len(recs)} of {total}):")
            for r in recs:
                lines.append(f"  {dict(r)!r}")
        return "\n".join(lines)

    @property
    def index_names(self):
        return list(self._indexes.keys())

    def __repr__(self):
        kinds = {n: ix["spec"].kind for n, ix in self._indexes.items()}
        return (f"Collection('{self.name}', key='{self._key_field}', "
                f"indexes={kinds}, n={len(self)})")


class _DroppedCollection:
    """Poison class for Collection handles whose collection was dropped.

    ``drop_collection`` swaps the handle's ``__class__`` to this: every
    method lookup (and the container dunders, which bypass __getattr__)
    raises CollectionDroppedError instead of the undefined behaviour of
    stale structures.  The instance __dict__ keeps ``name`` for the message.
    """

    def _raise(self):
        raise CollectionDroppedError(self.__dict__.get("name", "?"))

    def __getattr__(self, attr):
        self._raise()

    def __getitem__(self, key):
        self._raise()

    def __setitem__(self, key, value):
        self._raise()

    def __delitem__(self, key):
        self._raise()

    def __contains__(self, key):
        self._raise()

    def __len__(self):
        self._raise()

    def __iter__(self):
        self._raise()

    def __repr__(self):
        return f"<dropped Collection {self.__dict__.get('name', '?')!r}>"


class Record(dict):
    """A record returned by a Collection: a dict whose field assignment
    (``rec['likes'] = 6``) writes through the Collection and re-indexes."""

    def __init__(self, collection, pk, data):
        super().__init__(data)
        self._collection = collection
        self._pk = pk

    @property
    def pk(self):
        return self._pk

    def __setitem__(self, field, value):
        self._collection.update(self._pk, **{field: value})
        super().__setitem__(field, value)
