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
evaluated by **eldar** (https://github.com/kerighan/eldar); BM25/TF-IDF ranking
scores the boolean candidate set by the query's positive terms.  eldar is an
optional dependency (lazy import; `pip install eldar`).
"""

import re
from collections import Counter

from .base import DataStructure
from .dict import Dict
from .list import List


def _require_eldar():
    try:
        import eldar  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SearchIndex needs the 'eldar' package for boolean query parsing. "
            "Install it with: pip install eldar"
        ) from e


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


class SearchIndex(DataStructure):
    """Inverted index over a user Dataset, with boolean + BM25 search.

    Example:
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
        _parent=None,
    ):
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
        meta_ds = self._db.create_dataset(
            f"_search_{n}_dmeta", addr="uint64", dl="uint32"
        )
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
        self._dataset = None
        self._postings = self._docmeta = self._deleted = self._meta = None
        self._loaded_next_id = False

    def _ensure_loaded(self):
        if self._postings is not None:
            return
        n = self.name
        self._dataset = self._db.get_dataset(self._dataset_name)
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
        }

    def _get_registry_params(self):
        return self._config()

    @classmethod
    def _from_registry_params(cls, name, db, params):
        ds = db.get_dataset(params.pop("dataset_name"))
        return cls(name, db, ds, **params)

    # ── tokenisation (kept consistent with eldar's normalisation) ─────────────

    def _normalise(self, text):
        from unidecode import unidecode

        if self.ignore_case:
            text = text.lower()
        if self.ignore_accent:
            text = unidecode(text)
        return text

    def _tokens(self, text):
        from eldar.regex import WORD_REGEX
        from eldar.index import PUNCTUATION

        toks = re.findall(WORD_REGEX, self._normalise(text), re.UNICODE)
        if self.ignore_punctuation:
            table = str.maketrans("", "", PUNCTUATION)
            toks = [t.translate(table) for t in toks]
        return [t for t in toks if t]

    def _doc_text(self, record):
        return " ".join(str(record.get(f, "")) for f in self._text_fields)

    # ── write API ─────────────────────────────────────────────────────────────

    def add(self, record, text=None):
        """Buffer a record (dict matching the dataset schema) for indexing;
        returns its assigned doc-id.  Materialised on the next flush().

        text: if given, index THIS text instead of the record's own fields —
        lets a Collection store just a {pk} stub here while indexing the real
        record's text (no document duplication)."""
        if not isinstance(record, dict):
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
        ds = self._dataset

        # documents: one contiguous allocation, write records, record addresses
        n = len(self._buf_recs)
        rs = ds.record_size
        base = ds.allocate_block(n)
        total_add = 0
        for i, (doc_id, record, dl) in enumerate(self._buf_recs):
            addr = base + i * rs
            ds[addr] = record
            self._docmeta.append({"addr": addr, "dl": dl})   # index == doc_id
            total_add += dl

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
        out = []
        if self._scored:
            it = _iter_varint(blob)
            last = 0
            for gap in it:
                last += gap
                out.append((last, next(it)))
        else:
            last = 0
            for gap in _iter_varint(blob):
                last += gap
                out.append(last)
        return out

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
        """Postings for a term as a set of eldar Items — the leaf callback for
        eldar's query tree.  Handles `*` wildcards by scanning the dictionary."""
        from eldar.entry import Item

        if self.ignore_punctuation:
            query_term = self._strip_punct(query_term)
        result = set()
        if "*" in query_term:
            rgx = re.compile(query_term.replace("*", ".*"))  # eldar: re.match, no end-anchor
            for term in self._postings.keys():
                if rgx.match(term):
                    for i in self._term_ids(term):
                        result.add(Item(i, 0))
            return result
        for i in self._term_ids(query_term):
            result.add(Item(i, 0))
        return result

    # ── read API ────────────────────────────────────────────────────────────

    def search(self, query, return_ids=False, limit=None, mode=None, with_scores=False):
        """Run a query; return matching documents (or doc-ids).

        AND / OR / AND NOT, parentheses, `*` wildcards (via eldar).  mode:
        "boolean" (unranked, doc-id order) or "bm25"/"tfidf" (ranked, best
        first; needs scoring="bm25").  Defaults to the index's scoring.
        """
        _require_eldar()
        from eldar.index import parse_query

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
        """Return the stored record for a doc-id (or None if out of range)."""
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
            from eldar.index import PUNCTUATION

            return term.translate(str.maketrans("", "", PUNCTUATION))
        return term

    def _positive_terms(self, node, negated=False, out=None):
        from eldar.indexops import AND, OR, ANDNOT
        from eldar.entry import IndexEntry

        if out is None:
            out = set()
        if isinstance(node, ANDNOT):
            self._positive_terms(node.left, negated, out)
            self._positive_terms(node.right, not negated, out)
        elif isinstance(node, (AND, OR)):
            self._positive_terms(node.left, negated, out)
            self._positive_terms(node.right, negated, out)
        elif isinstance(node, IndexEntry):
            qt = node.query_term
            if not negated and isinstance(qt, str) and "*" not in qt:
                out.add(self._strip_punct(qt))
        return out

    def _doc_len(self, doc_id):
        return int(self._docmeta[doc_id]["dl"])

    def _rank(self, candidate_ids, tree, mode):
        import math

        candidates = set(candidate_ids)
        N = int(self._meta["next_id"]["v"])
        if not candidates or N == 0:
            return []
        terms = self._positive_terms(tree)
        avgdl = int(self._meta["total_len"]["v"]) / N
        k1, b = self._k1, self._b

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
                    dl = self._doc_len(doc_id)
                    denom = tf + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 1.0))
                    scores[doc_id] += idf * (tf * (k1 + 1.0)) / denom
                else:
                    scores[doc_id] += (1.0 + math.log(tf)) * idf
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
