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
from datetime import date, datetime
from functools import lru_cache

import mmh3
import numpy as np

from loom.dataset import as_record


def _hash_value(value):
    """Fixed-width 128-bit murmur hash (32 hex chars) of a value — used as the
    group key for indexes on unbounded (text/blob/json) fields, so possibly-long
    values (e.g. an argument or article title) can be indexed without truncation.
    No order is preserved (equality grouping only)."""
    return f"{mmh3.hash128(str(value), signed=False):032x}"

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

    def __init__(self, sort=None, desc=False, field=None):
        self.sort = sort
        self.desc = desc
        # field defaults to the index name — set it to index the SAME field
        # under several indexes (e.g. category by engagement AND by date).
        self.field = field


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


@lru_cache(maxsize=4096)
def _encode_value_cached(value, desc):
    return encode_value(value, desc)


def _encode_sort(value, desc):
    """encode_value with a small LRU — several Many indexes typically sort on
    the same field (e.g. five indexes × created_at), so within one record's
    index pass the same (value, desc) encodes once instead of N times."""
    try:
        return _encode_value_cached(value, desc)
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

    def _remove_from_indexes(self, record, pk):
        for ix in self._indexes.values():
            key = self._index_key(ix, record, pk)
            if key is not None and key in ix["struct"]:
                del ix["struct"][key]

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

    def insert(self, record):
        """Insert a record, or upsert it if its primary key already exists
        (the stored record is replaced and every index is re-pointed)."""
        record = as_record(record)
        pk = self._pk_of(record)
        with self._db.write_lock():
            with self._db.batch(defer_save=True):
                old = self._primary.get(pk)
                if old is not None:        # upsert: drop the old index/search entries
                    self._remove_from_indexes(old, pk)
                    self._remove_from_search(pk)
                self._primary[pk] = record
                self._add_to_indexes(record, pk)
                self._add_to_search(record, pk)
        return pk

    def insert_many(self, records):
        """Bulk insert, upserting any record whose primary key already exists
        (its old index/search entries are dropped first).  Within one batch a
        repeated primary key keeps the last occurrence."""
        # Dedup within the batch (last write wins), preserving first-seen order.
        dedup = {}
        for r in records:
            r = as_record(r)
            dedup[self._pk_of(r)] = r
        pks = list(dedup.keys())
        records = list(dedup.values())
        with self._db.write_lock():
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
                            for key, val in entries:
                                struct[key] = val
        return pks

    def update(self, pk, **changes):
        pk = str(pk)
        old = self._primary.get(pk)
        if old is None:
            raise KeyError(pk)
        if self._key_field in changes and str(changes[self._key_field]) != pk:
            raise ValueError("cannot change the primary key via update()")
        new = {**old, **changes}
        with self._db.write_lock():
            with self._db.batch(defer_save=True):
                for ix in self._indexes.values():
                    if not any(f in changes for f in ix["sources"]):
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
                self._primary[pk] = new
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

    def reindex(self):
        """Rebuild every secondary index from the primary store (O(n))."""
        with self._db.write_lock():
            with self._db.batch():
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
        """Number of records in group ``value`` of a 'many' index — a
        key-only scan that never reads index values or primary records.

        ``start``/``end`` bound the index's sort field exactly like find(),
        so a Many(sort="created_at") index also counts time windows:

            col.count("narrative", "ukraine", start=date(2026, 6, 1))

        O(log n + matches) with only the key walk paid per match."""
        ix, low_key, high_key = self._many_bounds(index_name, "count",
                                                  value, start, end)
        it = ix["struct"].range_keys(low_key, high_key, inclusive=(True, False))
        return sum(1 for _ in it)

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

    def find(self, index_name, value, start=None, end=None, limit=None,
             fields=None):
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
        when the schema carries heavy text."""
        ix, low_key, high_key = self._many_bounds(index_name, "find",
                                                  value, start, end)
        it = ix["struct"].range(low_key, high_key, inclusive=(True, False))
        return self._collect(it, limit, fields)

    def _collect(self, it, limit, fields):
        """Materialize (full or projected) records for an index-entry scan."""
        out = []
        if fields is not None:
            for _key, entry in it:
                rec = self._primary.get_fields(str(entry["pk"]), fields)
                if rec is not None:
                    out.append(self._wrap(entry["pk"], rec))
                    if limit is not None and len(out) >= limit:
                        break
            return out
        for _key, entry in it:
            rec = self._primary.get(str(entry["pk"]))
            if rec is not None:
                out.append(self._wrap(entry["pk"], rec))
                if limit is not None and len(out) >= limit:
                    break
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
        it = ix["struct"].range(start, end, inclusive=(True, False), reverse=desc)
        return self._collect(it, limit, fields)

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
