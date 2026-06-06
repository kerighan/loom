"""Collection — a dataset with one primary index and N secondary indexes
kept in sync automatically.

A ``Collection`` coordinates an existing primary ``Dict`` (which stores the
records, keyed by a primary-key field) and any number of *secondary* indexes.
Secondary indexes are **reference** indexes: each maps its own key →  the
primary key (not a full copy of the record), so there is no data duplication —
a secondary lookup costs one extra hop (secondary → primary key → record).

    users = db.collection(
        "users", User,
        key="username",
        indexes={"email": "email",                       # field name (persisted)
                 "email_lc": lambda r: r["email"].lower()},  # lambda (re-declare)
    )

    users.insert({"username": "alice", "email": "A@x.com"})
    users["alice"]                 # primary lookup
    users.get("email", "A@x.com")  # secondary lookup → record
    users.get("email_lc", "a@x.com")
    users.delete("alice")          # removed from every index

Sync & atomicity:
    insert / update / delete touch the primary index and every secondary
    index under a single ``db.write_lock()`` (serialises concurrent writers)
    wrapped in ``db.batch()`` (groups header flushes).  This is *not* a full
    crash-atomic WAL transaction across indexes — a crash mid-update could
    leave a secondary index out of sync with the primary; rebuild with
    ``reindex()`` if that ever happens.

Persistence:
    Field-name indexes are persisted, so ``db.collection(name)`` (no model)
    rebuilds them automatically on reopen.  Lambda indexes cannot be
    serialised — re-pass them in ``indexes`` to restore their sync.
"""

from __future__ import annotations


def _make_extractor(field_or_callable):
    """Return (extractor, field_name_or_None) for a field name or callable."""
    if callable(field_or_callable):
        return field_or_callable, None
    field = field_or_callable
    return (lambda record: record.get(field)), field


class Collection:
    def __init__(self, db, name, dataset, primary, key_field, indexes):
        """
        Args:
            db:        DB instance
            name:      collection name
            dataset:   the record Dataset
            primary:   Dict storing records, keyed by the primary key
            key_field: name of the primary-key field in each record
            indexes:   {index_name: (index_dict, extractor, field_name_or_None)}
        """
        self._db = db
        self.name = name
        self.dataset = dataset
        self._primary = primary
        self._key_field = key_field
        self._indexes = indexes

    # ── helpers ──────────────────────────────────────────────────────────

    def _pk_of(self, record):
        if self._key_field not in record:
            raise KeyError(
                f"record is missing primary-key field {self._key_field!r}"
            )
        return str(record[self._key_field])

    # ── writes ───────────────────────────────────────────────────────────

    def insert(self, record):
        """Insert a record and index it in the primary + every secondary."""
        pk = self._pk_of(record)
        with self._db.write_lock():
            with self._db.batch():
                self._primary[pk] = record
                for idx_dict, extractor, _field in self._indexes.values():
                    key = extractor(record)
                    if key is not None and key != "":
                        idx_dict[str(key)] = {"pk": pk}
        return pk

    def insert_many(self, records):
        """Bulk insert under a single lock + batch."""
        pks = []
        with self._db.write_lock():
            with self._db.batch():
                for record in records:
                    pk = self._pk_of(record)
                    self._primary[pk] = record
                    for idx_dict, extractor, _field in self._indexes.values():
                        key = extractor(record)
                        if key is not None and key != "":
                            idx_dict[str(key)] = {"pk": pk}
                    pks.append(pk)
        return pks

    def update(self, pk, **changes):
        """Apply field changes to a record, re-indexing any changed keys.

        The primary key itself cannot be changed via update (delete + insert
        instead).
        """
        pk = str(pk)
        old = self._primary.get(pk)
        if old is None:
            raise KeyError(pk)
        if self._key_field in changes and str(changes[self._key_field]) != pk:
            raise ValueError(
                "cannot change the primary key via update(); "
                "delete() then insert() instead"
            )
        new = {**old, **changes}
        with self._db.write_lock():
            with self._db.batch():
                for idx_dict, extractor, _field in self._indexes.values():
                    old_key = extractor(old)
                    new_key = extractor(new)
                    if old_key == new_key:
                        continue
                    if old_key is not None and old_key != "" and str(old_key) in idx_dict:
                        del idx_dict[str(old_key)]
                    if new_key is not None and new_key != "":
                        idx_dict[str(new_key)] = {"pk": pk}
                self._primary[pk] = new
        return new

    def delete(self, pk):
        """Delete a record from the primary and every secondary index."""
        pk = str(pk)
        record = self._primary.get(pk)
        if record is None:
            raise KeyError(pk)
        with self._db.write_lock():
            with self._db.batch():
                for idx_dict, extractor, _field in self._indexes.values():
                    key = extractor(record)
                    if key is not None and key != "" and str(key) in idx_dict:
                        del idx_dict[str(key)]
                del self._primary[pk]

    def reindex(self):
        """Rebuild every secondary index from the primary store.

        Use after a crash or after attaching a new index to an existing
        primary.  O(n) over the primary records.
        """
        with self._db.write_lock():
            with self._db.batch():
                for idx_dict, extractor, _field in self._indexes.values():
                    for pk, record in self._primary.items():
                        key = extractor(record)
                        if key is not None and key != "":
                            idx_dict[str(key)] = {"pk": str(pk)}

    # ── reads ────────────────────────────────────────────────────────────

    def __getitem__(self, pk):
        return self._primary[str(pk)]

    def get(self, index_name, key, default=None):
        """Look up a record through a secondary index.

        With no ``index_name`` semantics ambiguity: pass the index name and
        its key.  For a primary-key lookup use ``collection[pk]`` or
        ``.get_primary(pk)``.
        """
        if index_name not in self._indexes:
            raise KeyError(f"unknown index {index_name!r}")
        idx_dict = self._indexes[index_name][0]
        entry = idx_dict.get(str(key))
        if entry is None:
            return default
        return self._primary.get(str(entry["pk"]), default)

    def get_primary(self, pk, default=None):
        return self._primary.get(str(pk), default)

    def get_pk(self, index_name, key, default=None):
        """Return the primary key a secondary key points to (no record read)."""
        if index_name not in self._indexes:
            raise KeyError(f"unknown index {index_name!r}")
        entry = self._indexes[index_name][0].get(str(key))
        return str(entry["pk"]) if entry is not None else default

    def __contains__(self, pk):
        return str(pk) in self._primary

    def __len__(self):
        return len(self._primary)

    def keys(self):
        return self._primary.keys()

    def values(self):
        return self._primary.values()

    def items(self):
        return self._primary.items()

    @property
    def index_names(self):
        return list(self._indexes.keys())

    def __repr__(self):
        return (
            f"Collection('{self.name}', key='{self._key_field}', "
            f"indexes={self.index_names}, n={len(self)})"
        )
