"""Persistent full-text search index (inverted index) with boolean + BM25.

The index is built over a user **Dataset** (the document store), like the
other structures — `db.create_search_index(name, dataset, text_fields=[...])`.
`add(record)` inserts the record into that dataset and indexes its text
fields; only the record's **address** is kept (in a dense List by doc-id), so
documents are never duplicated.

    SearchIndex
    ├── dataset   : user Dataset (the documents — structured records)
    ├── _postings : Dict[term -> {df, last, off, nslots}]   compact postings
    ├── _docmeta  : List[{addr, dl}]   dense by doc_id: record address + length
    ├── _deleted  : Set[doc_id]                             tombstones
    └── _meta     : Dict (next_id, total_len)

Postings are a **delta + varint** byte blob per term (in the shared BlobStore):
doc-ids are assigned by a monotonic counter and indexed in id order, so each
list is sorted ascending and the gaps are tiny (1-2 bytes each).

Writes are **buffered in memory** and materialised on `flush()` (also called
automatically before any read / on close): a bulk build inserts the records in
one contiguous block and writes each term's blob once.

Boolean queries (AND / OR / AND NOT, parens, `*` wildcards) are parsed and
evaluated by the vendored copy of **eldar**'s index-level query machinery
(`loom.datastructures._boolquery`, from https://github.com/kerighan/eldar —
same author); BM25/TF-IDF ranking scores the boolean candidate set by the
query's positive terms.  No runtime dependency: search behaves identically
on every deployment (the test suite cross-checks the vendored copy against
the eldar package whenever it is installed).
"""

import re
from collections import Counter

import numpy as np

from .base import DataStructure
from .dict import Dict
from .list import List


# ── accent folding ────────────────────────────────────────────────────────────


class _UnidecodeTable(dict):
    """str.translate table that memoises unidecode per codepoint.

    unidecode transliterates strictly character by character (a table lookup
    per codepoint, no context), so caching each codepoint's replacement and
    applying it with str.translate produces byte-identical output while doing
    the per-character work in C instead of a Python loop."""

    def __missing__(self, codepoint):
        from unidecode import unidecode

        repl = unidecode(chr(codepoint))
        self[codepoint] = repl
        return repl


_UNIDECODE_TABLE = _UnidecodeTable()


def _fold_accents(text):
    if text.isascii():
        return text
    return text.translate(_UNIDECODE_TABLE)


# ── vendored tokenizer constants (from eldar 0.0.8) ──────────────────────────
# These two constants define loom's ON-DISK postings format: which tokens a
# document yields decides which terms exist in every already-built index.
# They are deliberately vendored, NOT imported from eldar — the installed
# eldar version (or a p4a/pip resolution surprise) must never be able to
# silently change how loom tokenizes.  The query parser + tree are vendored
# too (loom.datastructures._boolquery) — search has no runtime dependency.
WORD_REGEX = r'([\w]+|[,?;.:\/!()\[\]\'"’\\><+-=])'
PUNCTUATION = """'!#$%&\'()+,-./:;<=>?@[\\]^_`{|}~'"""

# Tokenizer kit: precompiled WORD_REGEX + PUNCTUATION set/table + the
# fast-path invariant check ('_' is the only word char in PUNCTUATION — see
# SearchIndex._tokens).  Built once from the vendored constants.
_TOKEN_KIT = None


def _token_kit():
    global _TOKEN_KIT
    if _TOKEN_KIT is None:
        word_chars = [c for c in PUNCTUATION if re.match(r"\w", c, re.UNICODE)]
        _TOKEN_KIT = (
            re.compile(WORD_REGEX, re.UNICODE),
            frozenset(PUNCTUATION),
            str.maketrans("", "", PUNCTUATION),
            word_chars == ["_"],
        )
    return _TOKEN_KIT


class _Item:
    """Posting leaf handed to eldar's query tree: IndexEntry.search reads
    .id (and .position for multiword queries).  Vendored — eldar.entry.Item
    is a plain dataclass and importing it would tie loom to eldar's module
    layout for no benefit (loom never stores positions anyway)."""

    __slots__ = ("id", "position")

    def __init__(self, id, position=0):
        self.id = id
        self.position = position

    def __hash__(self):
        return hash((self.id, self.position))

    def __eq__(self, other):
        return (self.id, self.position) == (other.id, other.position)




# ── varint (LEB128, unsigned) ─────────────────────────────────────────────────


def _put_varint(out: bytearray, n: int):
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            break


def _iter_varint(buf: bytes):
    i, shift, val = 0, 0, 0
    n = len(buf)
    while i < n:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            yield val
            shift = val = 0


def _decode_varints(buf: bytes):
    """Decode every varint in ``buf`` at once (vectorised numpy).

    LEB128 decode without a per-byte Python loop: locate terminator bytes
    (high bit clear), shift each byte's 7 payload bits by its position within
    its varint, and sum per group with add.reduceat.  ~10x faster than
    _iter_varint on long posting lists.  Values must fit in 64 bits (they do:
    doc-id gaps and capped term frequencies).
    """
    a = np.frombuffer(buf, dtype=np.uint8)
    ends = np.flatnonzero(a < 0x80)          # terminator byte of each varint
    if len(ends) == 0:
        return np.empty(0, dtype=np.uint64)
    a = a[: ends[-1] + 1]  # drop a trailing incomplete varint (legacy: unyielded)
    starts = np.empty(len(ends), dtype=np.intp)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    # position of each byte inside its own varint (0-based)
    pos = np.arange(len(a), dtype=np.uint64) - np.repeat(
        starts.astype(np.uint64), ends - starts + 1
    )
    payload = (a & 0x7F).astype(np.uint64) << (pos * 7)
    return np.add.reduceat(payload, starts)


class SearchIndex(DataStructure):
    """Inverted index over a user Dataset, with boolean + BM25 search.

    Example::

        docs = db.create_dataset("docs", title="utf8[120]", body="text")
        idx = db.create_search_index("idx", docs, text_fields=["title", "body"],
                                     scoring="bm25")
        i = idx.add({"title": "Fast search", "body": "inverted index ..."})
        idx.search("search AND (fast OR quick) AND NOT slow")   # ranked records
        idx.get_document(i)                                     # the record
    """

    def __init__(
        self,
        name,
        db,
        dataset=None,
        text_fields=None,
        ignore_case=True,
        ignore_accent=True,
        ignore_punctuation=True,
        doc_id_dtype="uint32",
        scoring="boolean",
        bm25_k1=1.5,
        bm25_b=0.75,
        store_documents=True,
        _parent=None,
    ):
        # store_documents=False: the index keeps only postings + doc lengths
        # (no dataset, no doc→address map) and search() returns doc-ids; the
        # caller (e.g. a Collection) owns the doc-id → record mapping.  Avoids
        # duplicating the document store.
        self._store_docs = store_documents
        self._dataset = dataset
        self._dataset_name = dataset.name if dataset is not None else None
        self._text_fields = list(text_fields) if text_fields else None
        self.ignore_case = ignore_case
        self.ignore_accent = ignore_accent
        self.ignore_punctuation = ignore_punctuation
        self._doc_id_dtype = doc_id_dtype
        if scoring not in ("boolean", "bm25"):
            raise ValueError("scoring must be 'boolean' or 'bm25'")
        self._scoring = scoring
        self._scored = scoring != "boolean"
        self._k1 = bm25_k1
        self._b = bm25_b
        self.item_schema = None
        # in-memory write buffer (flushed lazily — see flush())
        self._buf_recs = []   # [(doc_id, record, dl)]
        self._buf_post = {}    # term -> [doc_id] or [(doc_id, tf)] when scored
        self._next_id = 0
        self._dirty = False
        self._postings = self._docmeta = self._deleted = self._meta = None
        self._doclens = None  # np.int64 cache of _docmeta[...]["dl"] (bm25)

        super().__init__(name, db, _parent=_parent)

        if self._load_metadata():
            self._load()
        else:
            self._initialize()

    # ── construction / persistence ───────────────────────────────────────────

    def _default_text_fields(self):
        """All string-valued fields of the dataset (text / utf8 / U)."""
        ds = self._dataset
        fields = []
        for fname in ds.user_schema.names:
            if fname in getattr(ds, "_text_fields", ()) or fname in getattr(ds, "_utf8_fields", {}):
                fields.append(fname)
            elif ds.user_schema.fields[fname][0].kind == "U":
                fields.append(fname)
        return fields

    def _initialize(self):
        if self._store_docs:
            if self._dataset is None:
                raise ValueError("create_search_index requires a documents dataset")
            if self._text_fields is None:
                self._text_fields = self._default_text_fields()
        n = self.name
        post_ds = self._db.create_dataset(
            f"_search_{n}_postid",
            df="uint32", last=self._doc_id_dtype, off="uint64", nslots="uint32",
        )
        self._postings = self._db.create_dict(
            f"_search_{n}_postings", post_ds, max_key_len=64
        )
        # doc-meta: (addr, dl) when we store documents, else just dl (lengths
        # for BM25), dense by doc-id.
        if self._store_docs:
            meta_ds = self._db.create_dataset(f"_search_{n}_dmeta", addr="uint64", dl="uint32")
        else:
            meta_ds = self._db.create_dataset(f"_search_{n}_dmeta", dl="uint32")
        self._docmeta = self._db.create_list(f"_search_{n}_docmeta", meta_ds)
        self._deleted = self._db.create_set(f"_search_{n}_deleted", key_size=24)
        ctr_ds = self._db.create_dataset(f"_search_{n}_metads", v="int64")
        self._meta = self._db.create_dict(f"_search_{n}_meta", ctr_ds)
        self._meta["next_id"] = {"v": 0}
        self._meta["total_len"] = {"v": 0}
        self._next_id = 0
        self.save()

    def _load(self):
        m = self._load_metadata()
        self._dataset_name = m.get("dataset_name")
        self._text_fields = m.get("text_fields")
        self.ignore_case = m.get("ignore_case", True)
        self.ignore_accent = m.get("ignore_accent", True)
        self.ignore_punctuation = m.get("ignore_punctuation", True)
        self._doc_id_dtype = m.get("doc_id_dtype", "uint32")
        self._scoring = m.get("scoring", "boolean")
        self._scored = self._scoring != "boolean"
        self._k1 = m.get("bm25_k1", 1.5)
        self._b = m.get("bm25_b", 0.75)
        self._store_docs = m.get("store_documents", True)
        self._dataset = None
        self._postings = self._docmeta = self._deleted = self._meta = None
        self._doclens = None  # np.int64 cache of _docmeta[...]["dl"] (bm25)
        self._loaded_next_id = False

    def _ensure_loaded(self):
        if self._postings is not None:
            return
        n = self.name
        self._dataset = self._db.get_dataset(self._dataset_name) if self._dataset_name else None
        self._postings = self._db._datastructures[f"_search_{n}_postings"]
        self._docmeta = self._db._datastructures[f"_search_{n}_docmeta"]
        self._deleted = self._db._datastructures[f"_search_{n}_deleted"]
        self._meta = self._db._datastructures[f"_search_{n}_meta"]
        if not getattr(self, "_loaded_next_id", True):
            self._next_id = int(self._meta["next_id"]["v"])
            self._loaded_next_id = True

    def save(self, force=False):
        self.flush()
        self._save_metadata(self._config())

    def _config(self):
        return {
            "dataset_name": self._dataset_name,
            "text_fields": self._text_fields,
            "ignore_case": self.ignore_case,
            "ignore_accent": self.ignore_accent,
            "ignore_punctuation": self.ignore_punctuation,
            "doc_id_dtype": self._doc_id_dtype,
            "scoring": self._scoring,
            "bm25_k1": self._k1,
            "bm25_b": self._b,
            "store_documents": self._store_docs,
        }

    def _get_registry_params(self):
        return self._config()

    @classmethod
    def _from_registry_params(cls, name, db, params):
        # Copy: the registry dict is shared with the DB header — mutating it
        # (pop) would persist a params dict missing "dataset_name" on the next
        # header save, breaking the following reopen.  pop(..., None) also
        # recovers files already corrupted by that earlier bug.
        params = dict(params)
        dsname = params.pop("dataset_name", None)
        ds = db.get_dataset(dsname) if dsname else None
        return cls(name, db, ds, **params)

    # ── tokenisation (kept consistent with eldar's normalisation) ─────────────

    def _normalise(self, text):
        if self.ignore_case:
            text = text.lower()
        if self.ignore_accent:
            text = _fold_accents(text)
        return text

    def _tokens(self, text):
        """Tokenize exactly like eldar: WORD_REGEX findall, then (with
        ignore_punctuation) strip PUNCTUATION chars from every token and drop
        the empties.  The hot path below avoids the per-token translate()
        call: single punctuation tokens are dropped by a set lookup, and '_'
        is the only PUNCTUATION char that can appear INSIDE a \\w+ token, so
        translate() is only needed for tokens that contain it — verified
        against eldar's constant at first use (_punct_fast_ok), with the
        literal legacy loop as fallback if that invariant ever breaks."""
        word_re, punct_set, punct_table, fast_ok = _token_kit()
        toks = word_re.findall(self._normalise(text))
        if not self.ignore_punctuation:
            return [t for t in toks if t]
        if not fast_ok:   # eldar changed PUNCTUATION — exact legacy behaviour
            toks = [t.translate(punct_table) for t in toks]
            return [t for t in toks if t]
        out = []
        for t in toks:
            if t in punct_set:          # single punctuation token → dropped
                continue
            if "_" in t:                # only word char PUNCTUATION can strip
                t = t.translate(punct_table)
                if not t:
                    continue
            out.append(t)
        return out

    def _doc_text(self, record):
        return " ".join(str(record.get(f, "")) for f in self._text_fields)

    # ── write API ─────────────────────────────────────────────────────────────

    def add(self, record, text=None):
        """Buffer a record (dict matching the dataset schema) for indexing;
        returns its assigned doc-id.  Materialised on the next flush().

        text: if given, index THIS text instead of the record's own fields.
        record may be None when store_documents=False (the caller owns the
        doc-id → record mapping)."""
        if self._store_docs and not isinstance(record, dict):
            raise TypeError("record must be a dict matching the dataset schema")
        self._ensure_loaded()
        doc_id = self._next_id
        self._next_id += 1
        tokens = self._tokens(text if text is not None else self._doc_text(record))

        self._buf_recs.append((doc_id, record, len(tokens)))
        if self._scored:
            for term, count in Counter(tokens).items():
                self._buf_post.setdefault(term, []).append((doc_id, min(count, 65535)))
        else:
            for term in set(tokens):
                self._buf_post.setdefault(term, []).append(doc_id)

        self._dirty = True
        return doc_id

    def add_many(self, records):
        ids = [self.add(r) for r in records]
        self.flush()
        return ids

    def _encode(self, items, last):
        out = bytearray()
        if self._scored:
            for doc_id, tf in items:
                _put_varint(out, doc_id - last)
                _put_varint(out, tf)
                last = doc_id
        else:
            for doc_id in items:
                _put_varint(out, doc_id - last)
                last = doc_id
        return bytes(out), last

    def flush(self):
        """Materialise buffered records + postings to disk."""
        if not self._dirty:
            return
        blobs = self._db.blob_store

        if self._store_docs:
            # documents: one contiguous allocation, serialize all records into
            # one buffer and write it in a single db.write, then record the
            # (addr, length) per doc.
            ds = self._dataset
            rs = ds.record_size
            base = ds.allocate_block(len(self._buf_recs))
            buf = bytearray()
            for _doc_id, record, _dl in self._buf_recs:
                buf += ds._serialize(**record)
            ds.db.write(base, bytes(buf))
            self._docmeta.append_many(
                {"addr": base + i * rs, "dl": dl}
                for i, (_d, _r, dl) in enumerate(self._buf_recs)
            )
        else:
            # no doc-store: only per-doc lengths (for BM25)
            self._docmeta.append_many({"dl": dl} for _d, _r, dl in self._buf_recs)
        total_add = sum(dl for _d, _r, dl in self._buf_recs)

        # postings: append-encode each term's blob (rewrite once), bulk-insert
        records = []
        for term, items in self._buf_post.items():
            if term in self._postings:
                rec = self._postings[term]
                off0, ns0 = int(rec["off"]), int(rec["nslots"])
                old = blobs.read(off0) if ns0 else b""
                add_bytes, last = self._encode(items, int(rec["last"]))
                off, ns = blobs.write(old + add_bytes)
                if ns0:
                    blobs.delete(off0, ns0)
                df = int(rec["df"]) + len(items)
            else:
                add_bytes, last = self._encode(items, 0)
                off, ns = blobs.write(add_bytes)
                df = len(items)
            records.append((term, {"df": df, "last": last, "off": off, "nslots": ns}))
        self._postings.set_batch(records)

        if self._scored:
            self._meta["total_len"] = {"v": int(self._meta["total_len"]["v"]) + total_add}
        self._meta["next_id"] = {"v": self._next_id}
        self._buf_recs = []
        self._buf_post = {}
        self._dirty = False

    def delete(self, doc_id):
        """Logically delete a document (tombstone; filtered out of results)."""
        self._ensure_loaded()
        self._deleted.add(str(doc_id))

    # ── postings access ───────────────────────────────────────────────────────

    def _decode_postings(self, rec):
        ns = int(rec["nslots"])
        if ns == 0:
            return []
        blob = self._db.blob_store.read(int(rec["off"]))
        vals = _decode_varints(blob)
        if self._scored:
            if len(vals) % 2:  # degenerate blob — legacy decode & its error
                it = _iter_varint(blob)
                out, last = [], 0
                for gap in it:
                    last += gap
                    out.append((last, next(it)))
                return out
            ids = np.cumsum(vals[0::2])
            return list(zip(ids.tolist(), vals[1::2].tolist()))
        return np.cumsum(vals).tolist()

    def _term_ids(self, term):
        if term not in self._postings:
            return []
        post = self._decode_postings(self._postings[term])
        return [i for i, _ in post] if self._scored else post

    def _term_tf(self, term):
        if not self._scored or term not in self._postings:
            return {}
        return {i: tf for i, tf in self._decode_postings(self._postings[term])}

    def get(self, query_term):
        """Postings for a term as a set of Items — the leaf callback for
        eldar's query tree.  Handles `*` wildcards by scanning the dictionary."""
        if self.ignore_punctuation:
            query_term = self._strip_punct(query_term)
        result = set()
        if "*" in query_term:
            rgx = re.compile(query_term.replace("*", ".*"))  # eldar: re.match, no end-anchor
            for term in self._postings.keys():
                if rgx.match(term):
                    for i in self._term_ids(term):
                        result.add(_Item(i, 0))
            return result
        for i in self._term_ids(query_term):
            result.add(_Item(i, 0))
        return result

    # ── read API ────────────────────────────────────────────────────────────

    def search(self, query, return_ids=False, limit=None, mode=None, with_scores=False):
        """Run a query; return matching documents (or doc-ids).

        AND / OR / AND NOT, parentheses, `*` wildcards (via eldar).  mode:
        "boolean" (unranked, doc-id order) or "bm25"/"tfidf" (ranked, best
        first; needs scoring="bm25").  Defaults to the index's scoring.
        """
        from loom.datastructures._boolquery import parse_query

        self._ensure_loaded()
        self.flush()
        q = query.strip()
        if not q:
            return []
        mode = mode or self._scoring
        if mode not in ("boolean", "bm25", "tfidf"):
            raise ValueError("mode must be 'boolean', 'bm25' or 'tfidf'")
        if mode != "boolean" and not self._scored:
            raise ValueError(
                f"mode={mode!r} needs a scored index — create it with "
                f"create_search_index(..., scoring='bm25')"
            )

        tree = parse_query(
            q, ignore_case=self.ignore_case, ignore_accent=self.ignore_accent
        )
        ids = tree.search(self)
        if self._deleted is not None and len(self._deleted):
            ids = {i for i in ids if str(i) not in self._deleted}

        if mode == "boolean":
            out = sorted(ids)
            if limit is not None:
                out = out[:limit]
            return out if return_ids else [self.get_document(i) for i in out]

        ranked = self._rank(ids, tree, mode)
        if limit is not None:
            ranked = ranked[:limit]
        if return_ids:
            return ranked if with_scores else [i for i, _ in ranked]
        if with_scores:
            return [(self.get_document(i), s) for i, s in ranked]
        return [self.get_document(i) for i, _ in ranked]

    def get_document(self, doc_id):
        """Return the stored record for a doc-id (or None).  Only meaningful
        when store_documents=True; otherwise the caller owns doc-id→record."""
        if not self._store_docs:
            return None
        self._ensure_loaded()
        self.flush()
        if doc_id < 0 or doc_id >= len(self._docmeta):
            return None
        return self._dataset[int(self._docmeta[doc_id]["addr"])]

    def document_frequency(self, term):
        self._ensure_loaded()
        self.flush()
        term = self._strip_punct(self._normalise(term))
        if term not in self._postings:
            return 0
        return int(self._postings[term]["df"])

    def __len__(self):
        self._ensure_loaded()
        self.flush()
        return len(self._docmeta) - len(self._deleted)

    def __contains__(self, doc_id):
        self._ensure_loaded()
        self.flush()
        return 0 <= doc_id < len(self._docmeta) and str(doc_id) not in self._deleted

    # ── ranking (BM25 / TF-IDF) ───────────────────────────────────────────────

    def _strip_punct(self, term):
        if self.ignore_punctuation:
            return term.translate(_token_kit()[2])
        return term

    def _positive_terms(self, node, negated=False, out=None):
        """Collect the non-negated leaf terms of an eldar query tree.

        Walked STRUCTURALLY (a leaf has .query_term, a binary node has
        .left/.right, negation flips on nodes whose class name contains
        NOT) rather than by isinstance on eldar's classes: an eldar
        relayout would make isinstance checks fail *silently* — every BM25
        score would drop to 0 with no error.  Duck-typing keeps ranking
        working against any eldar whose trees have this shape."""
        if out is None:
            out = set()
        qt = getattr(node, "query_term", None)
        if qt is not None:                      # leaf (IndexEntry-shaped)
            if not negated and isinstance(qt, str) and "*" not in qt:
                out.add(self._strip_punct(qt))
            return out
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        if left is None or right is None:       # unknown node shape → skip
            return out
        if "NOT" in type(node).__name__:        # ANDNOT: right side negated
            self._positive_terms(left, negated, out)
            self._positive_terms(right, not negated, out)
        else:                                   # AND / OR
            self._positive_terms(left, negated, out)
            self._positive_terms(right, negated, out)
        return out

    def _doc_len(self, doc_id):
        arr = self._doclens
        if arr is None or doc_id >= len(arr):
            arr = self._build_doclens()
        return int(arr[doc_id])

    def _build_doclens(self):
        """(Re)build the in-memory doc-length array from _docmeta.

        BM25 reads one length per candidate document; fetching each from the
        persistent List costs ~10 µs while an array lookup is ~0.1 µs — on a
        broad query this is most of the latency.  Doc lengths are immutable
        (deletes are tombstones), so the cache only ever *extends*: a full
        block-wise build the first time, then per-item reads for the tail
        when new documents were flushed since (doc_id >= len(cache)).
        """
        n = len(self._docmeta)
        old = self._doclens
        if old is None or len(old) == 0:
            arr = np.fromiter(
                (rec["dl"] for rec in self._docmeta), dtype=np.int64, count=n
            )
        else:
            tail = np.empty(n - len(old), dtype=np.int64)
            for i in range(len(old), n):
                tail[i - len(old)] = self._docmeta[i]["dl"]
            arr = np.concatenate([old, tail])
        self._doclens = arr
        return arr

    def _rank(self, candidate_ids, tree, mode):
        import math

        candidates = set(candidate_ids)
        N = int(self._meta["next_id"]["v"])
        if not candidates or N == 0:
            return []
        terms = self._positive_terms(tree)
        avgdl = int(self._meta["total_len"]["v"]) / N
        k1, b = self._k1, self._b

        doclens = self._doclens
        if mode == "bm25" and (doclens is None or len(doclens) < len(self._docmeta)):
            doclens = self._build_doclens()

        scores = {i: 0.0 for i in candidates}
        for term in sorted(terms):
            tf_map = self._term_tf(term)
            df = len(tf_map)
            if df == 0:
                continue
            if mode == "bm25":
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            else:
                idf = math.log(N / df)
            for doc_id, tf in tf_map.items():
                if doc_id not in candidates:
                    continue
                if mode == "bm25":
                    dl = int(doclens[doc_id])
                    denom = tf + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 1.0))
                    scores[doc_id] += idf * (tf * (k1 + 1.0)) / denom
                else:
                    scores[doc_id] += (1.0 + math.log(tf)) * idf
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
