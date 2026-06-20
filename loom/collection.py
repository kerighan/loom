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

Range/many keys are order-preserving encoded (signed ints zero-padded, strings
as-is) so the BTree's lexicographic order matches the natural order.

Sync: insert/update/delete/increment run under db.write_lock() + db.batch().
Not a crash-atomic cross-index transaction — reindex() rebuilds if needed.
Full-text ("search") indexes are planned (SearchIndex integration).
"""

from __future__ import annotations

import numbers

_SEP = "\x00"          # composite-key separator (numpy U preserves embedded NULs)
_INT_OFFSET = 1 << 63  # map signed int64 → unsigned for zero-padded ordering
_UINT_MAX = (1 << 64) - 1


# ── index-kind specs ──────────────────────────────────────────────────────────


class Primary:
    kind = "primary"


class Unique:
    kind = "unique"


class Range:
    kind = "range"


class Many:
    kind = "many"

    def __init__(self, sort=None, desc=False):
        self.sort = sort
        self.desc = desc


_STRING_KINDS = {"primary": Primary, "unique": Unique, "range": Range, "many": Many}


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
    bools → 0/1; everything else → str().  desc=True reverses.
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
    s = str(value)
    if desc:
        # reverse lexicographic order for strings: complement each codepoint
        return "".join(chr(0x10FFFF - ord(c)) for c in s)
    return s


class Collection:
    def __init__(self, db, name, dataset, primary_field, primary, indexes, key_size):
        """
        indexes: {idx_name: {"name", "spec", "field", "struct", "sources"}}
        """
        self._db = db
        self.name = name
        self.dataset = dataset
        self._key_field = primary_field
        self._primary = primary
        self._indexes = indexes
        self._key_size = key_size
        self._indexed_fields = set()
        for ix in indexes.values():
            self._indexed_fields.update(ix["sources"])

    # ── key construction ─────────────────────────────────────────────────

    def _pk_of(self, record):
        if self._key_field not in record:
            raise KeyError(f"record missing primary-key field {self._key_field!r}")
        return str(record[self._key_field])

    def _index_key(self, ix, record, pk):
        spec = ix["spec"]
        val = record.get(ix["field"])
        if val is None:
            return None
        if spec.kind == "unique":
            return encode_value(val)
        if spec.kind == "range":
            return encode_value(val) + _SEP + pk
        if spec.kind == "many":
            parts = [encode_value(val)]
            if spec.sort is not None:
                parts.append(encode_value(record.get(spec.sort), desc=spec.desc))
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

    def insert(self, record):
        pk = self._pk_of(record)
        with self._db.write_lock():
            with self._db.batch():
                self._primary[pk] = record
                self._add_to_indexes(record, pk)
        return pk

    def insert_many(self, records):
        pks = []
        with self._db.write_lock():
            with self._db.batch():
                for record in records:
                    pk = self._pk_of(record)
                    self._primary[pk] = record
                    self._add_to_indexes(record, pk)
                    pks.append(pk)
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
            with self._db.batch():
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
                self._primary[pk] = new
        return new

    def delete(self, pk):
        pk = str(pk)
        record = self._primary.get(pk)
        if record is None:
            raise KeyError(pk)
        with self._db.write_lock():
            with self._db.batch():
                self._remove_from_indexes(record, pk)
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

    def _wrap(self, pk, record):
        return Record(self, str(pk), record) if record is not None else None

    def __getitem__(self, pk):
        return self._wrap(pk, self._primary[str(pk)])

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
        entry = ix["struct"].get(encode_value(value))
        if entry is None:
            return default
        return self.get_primary(entry["pk"], default)

    def find(self, index_name, value, limit=None):
        """One-to-many lookup → list of records (ordered by the index's sort)."""
        ix = self._indexes[index_name]
        if ix["spec"].kind != "many":
            raise ValueError(f"find() needs a 'many' index for {index_name!r}")
        out = []
        for _key, entry in ix["struct"].prefix(encode_value(value) + _SEP):
            rec = self._primary.get(str(entry["pk"]))
            if rec is not None:
                out.append(self._wrap(entry["pk"], rec))
                if limit is not None and len(out) >= limit:
                    break
        return out

    def range(self, index_name, low=None, high=None, limit=None):
        """Range scan on a 'range' index → records with low <= value <= high
        (either bound may be None for an open end)."""
        ix = self._indexes[index_name]
        if ix["spec"].kind != "range":
            raise ValueError(f"range() needs a 'range' index for {index_name!r}")
        start = None if low is None else encode_value(low)
        end = None if high is None else encode_value(high) + "\x01"
        out = []
        for _key, entry in ix["struct"].range(start, end, inclusive=(True, False)):
            rec = self._primary.get(str(entry["pk"]))
            if rec is not None:
                out.append(self._wrap(entry["pk"], rec))
                if limit is not None and len(out) >= limit:
                    break
        return out

    def __contains__(self, pk):
        return str(pk) in self._primary

    def __len__(self):
        return len(self._primary)

    def keys(self):
        return self._primary.keys()

    def values(self):
        for pk, rec in self._primary.items():
            yield self._wrap(pk, rec)

    def items(self):
        for pk, rec in self._primary.items():
            yield str(pk), self._wrap(pk, rec)

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
