"""Persistent full-text search index (inverted index) with boolean queries.

A `SearchIndex` is a composite structure built on loom primitives:

    SearchIndex
    ├── _postings : Dict[term -> List[{id}]]   mutable postings, doc-ids ASCENDING
    ├── _docs     : Dict[doc_id -> {doc: json}] document store (retrieval)
    ├── _deleted  : Set[doc_id]                 tombstones (logical delete)
    └── _meta     : Dict (next_id counter)

Doc-ids are assigned by a monotonic counter, and documents are indexed in id
order, so every term's postings list stays **sorted ascending without any
effort** — which is what makes boolean evaluation (set merges) efficient and
what a later delta/varint compression pass will exploit.

Boolean queries (AND / OR / AND NOT, parentheses, wildcards) are parsed and
evaluated by **eldar** (https://github.com/kerighan/eldar): its index-mode
parser builds a tree of set operations whose leaves call back into
``SearchIndex.get(term)`` — so eldar provides the query engine and loom
provides the persistent postings.  eldar is an optional dependency, imported
lazily; install it with ``pip install eldar``.

v1 scope: boolean retrieval only (postings store doc-ids, no term
frequencies/positions), post-filtering done by the caller on `get()` results.
Ranking (BM25), phrase/proximity search and indexed attribute filters are
deliberately left for later — see the module-level notes in the repo.
"""

import json
import re

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


class SearchIndex(DataStructure):
    """Inverted index with boolean search, backed by loom and parsed by eldar.

    Example:
        idx = db.create_search_index("articles", text_fields=["title", "body"])
        doc_id = idx.add({"title": "Fast search", "body": "inverted index ..."})
        idx.search("search AND (fast OR quick) AND NOT slow")   # → [doc_id, ...]
        idx.get(doc_id)                                          # → the document
        idx.delete(doc_id)
    """

    def __init__(
        self,
        name,
        db,
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
        """
        Args:
            text_fields: list of document field names to index.  None → index
                every string-valued field of each document.
            ignore_case / ignore_accent / ignore_punctuation: normalisation
                applied identically at index time and query time (kept in sync
                with eldar's parser so terms always match).
            doc_id_dtype: width of the stored doc-id ("uint32" = up to 4 G docs).
            scoring: "boolean" (default) stores only doc-ids in the postings —
                boolean retrieval only.  "bm25" also stores the per-(term,doc)
                term frequency and per-doc length, enabling ranked search; such
                an index can still answer mode="boolean" queries.
            bm25_k1, bm25_b: BM25 tuning parameters (term-frequency saturation
                and length normalisation).  Standard defaults.
        """
        self._text_fields = list(text_fields) if text_fields else None
        self.ignore_case = ignore_case
        self.ignore_accent = ignore_accent
        self.ignore_punctuation = ignore_punctuation
        self._doc_id_dtype = doc_id_dtype
        if scoring not in ("boolean", "bm25"):
            raise ValueError("scoring must be 'boolean' or 'bm25'")
        self._scoring = scoring
        self._scored = scoring != "boolean"   # postings carry tf + doc lengths
        self._k1 = bm25_k1
        self._b = bm25_b
        self.item_schema = None
        # sub-structures resolved lazily (see _ensure_loaded)
        self._postings = None
        self._docs = None
        self._deleted = None
        self._meta = None

        super().__init__(name, db, _parent=_parent)

        if self._load_metadata():
            self._load()
        else:
            self._initialize()

    # ── construction / persistence ───────────────────────────────────────────

    def _initialize(self):
        n = self.name
        # postings: {id} for boolean, {id, tf} when scored (BM25 needs tf).
        if self._scored:
            post_ds = self._db.create_dataset(
                f"_search_{n}_postid", id=self._doc_id_dtype, tf="uint16"
            )
        else:
            post_ds = self._db.create_dataset(f"_search_{n}_postid", id=self._doc_id_dtype)
        self._postings = self._db.create_dict(
            f"_search_{n}_postings", List.template(post_ds)
        )
        doc_ds = self._db.create_dataset(f"_search_{n}_docds", doc="text")
        self._docs = self._db.create_dict(f"_search_{n}_docs", doc_ds)
        self._deleted = self._db.create_set(f"_search_{n}_deleted", key_size=24)
        meta_ds = self._db.create_dataset(f"_search_{n}_metads", v="int64")
        self._meta = self._db.create_dict(f"_search_{n}_meta", meta_ds)
        self._meta["next_id"] = {"v": 0}
        # Scored index: per-doc length in a List dense-indexed by doc_id (read
        # without touching the document JSON blob), plus Σ lengths for avgdl.
        self._lengths = None
        if self._scored:
            len_ds = self._db.create_dataset(f"_search_{n}_lends", dl="uint32")
            self._lengths = self._db.create_list(f"_search_{n}_lengths", len_ds)
            self._meta["total_len"] = {"v": 0}
        self.save()

    def _load(self):
        m = self._load_metadata()
        self._text_fields = m.get("text_fields")
        self.ignore_case = m.get("ignore_case", True)
        self.ignore_accent = m.get("ignore_accent", True)
        self.ignore_punctuation = m.get("ignore_punctuation", True)
        self._doc_id_dtype = m.get("doc_id_dtype", "uint32")
        self._scoring = m.get("scoring", "boolean")
        self._scored = self._scoring != "boolean"
        self._k1 = m.get("bm25_k1", 1.5)
        self._b = m.get("bm25_b", 0.75)
        # inner structures resolved on first access (registry may still be loading)
        self._postings = self._docs = self._deleted = self._meta = None
        self._lengths = None

    def _ensure_loaded(self):
        if self._postings is not None:
            return
        n = self.name
        self._postings = self._db._datastructures[f"_search_{n}_postings"]
        self._docs = self._db._datastructures[f"_search_{n}_docs"]
        self._deleted = self._db._datastructures[f"_search_{n}_deleted"]
        self._meta = self._db._datastructures[f"_search_{n}_meta"]
        if self._scored:
            self._lengths = self._db._datastructures[f"_search_{n}_lengths"]

    def save(self, force=False):
        self._save_metadata(self._config())

    def _config(self):
        return {
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
        return cls(name, db, **params)

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

    def _doc_text(self, document):
        """Concatenate the indexable text of a document."""
        if isinstance(document, str):
            return document
        if self._text_fields is not None:
            fields = self._text_fields
        else:
            fields = [k for k, v in document.items() if isinstance(v, str)]
        return " ".join(str(document.get(f, "")) for f in fields)

    # ── write API ─────────────────────────────────────────────────────────────

    def add(self, document):
        """Index a document (str or dict) and return its assigned doc-id."""
        self._ensure_loaded()
        doc_id = int(self._meta["next_id"]["v"])
        tokens = self._tokens(self._doc_text(document))

        if self._scored:
            # Count term frequencies; store doc length (dense List indexed by
            # doc_id) + accumulate total length (for BM25's avgdl).  One posting
            # per (term, doc), carrying tf.
            from collections import Counter

            tf = Counter(tokens)
            self._docs[str(doc_id)] = {"doc": json.dumps(document)}
            self._lengths.append({"dl": len(tokens)})   # index == doc_id
            for term, count in tf.items():
                # doc_id monotonically increasing → append keeps postings sorted.
                self._postings[term].append({"id": doc_id, "tf": min(count, 65535)})
            self._meta["total_len"] = {"v": int(self._meta["total_len"]["v"]) + len(tokens)}
        else:
            self._docs[str(doc_id)] = {"doc": json.dumps(document)}
            for term in set(tokens):   # boolean: one posting per (term, doc)
                self._postings[term].append({"id": doc_id})

        self._meta["next_id"] = {"v": doc_id + 1}
        return doc_id

    def add_many(self, documents):
        """Index an iterable of documents; returns the list of doc-ids."""
        return [self.add(d) for d in documents]

    def delete(self, doc_id):
        """Logically delete a document (tombstone; filtered out of results)."""
        self._ensure_loaded()
        self._deleted.add(str(doc_id))

    # ── read API ────────────────────────────────────────────────────────────

    def get(self, query_term):
        """Postings for a single term as a set of eldar Items (id, position).

        This is the callback eldar's query tree uses for leaf terms; the
        returned objects only need a ``.id``.  Handles trailing/embedded ``*``
        wildcards by scanning the term dictionary (O(vocabulary)).
        """
        from eldar.entry import Item
        from eldar.index import PUNCTUATION

        self._ensure_loaded()
        if self.ignore_punctuation:
            query_term = query_term.translate(str.maketrans("", "", PUNCTUATION))

        result = set()
        if "*" in query_term:
            # Mirror eldar.Index.get: re.match (start-anchored, not end).
            rgx = re.compile(query_term.replace("*", ".*"))
            for term in self._postings.keys():
                if rgx.match(term):
                    for rec in self._postings[term]:
                        result.add(Item(int(rec["id"]), 0))
            return result

        if query_term not in self._postings:
            return result
        for rec in self._postings[query_term]:
            result.add(Item(int(rec["id"]), 0))
        return result

    def search(self, query, return_ids=False, limit=None, mode=None, with_scores=False):
        """Run a query and return matching documents (or doc-ids).

        Supports AND / OR / AND NOT, parentheses and ``*`` wildcards via eldar.
        Tombstoned docs are always excluded.

        mode:
            - "boolean": unranked, results sorted by doc-id ascending.
            - "bm25" / "tfidf": ranked by relevance, best first (requires a
              scored index — ``scoring="bm25"`` at creation).
            Defaults to the index's own scoring ("boolean" or "bm25").
        with_scores: if True (ranked modes), return (item, score) pairs.
        """
        _require_eldar()
        from eldar.index import parse_query

        self._ensure_loaded()
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
        ids = tree.search(self)  # candidate set of int doc-ids (boolean match)
        if self._deleted is not None and len(self._deleted):
            ids = {i for i in ids if str(i) not in self._deleted}

        if mode == "boolean":
            out = sorted(ids)
            if limit is not None:
                out = out[:limit]
            return out if return_ids else [self.get_document(i) for i in out]

        # Ranked: score the boolean candidates by the query's positive terms.
        ranked = self._rank(ids, tree, mode)        # [(doc_id, score)] desc
        if limit is not None:
            ranked = ranked[:limit]
        if return_ids:
            return ranked if with_scores else [i for i, _ in ranked]
        if with_scores:
            return [(self.get_document(i), s) for i, s in ranked]
        return [self.get_document(i) for i, _ in ranked]

    # ── ranking (BM25 / TF-IDF) ───────────────────────────────────────────────

    def _strip_punct(self, term):
        if self.ignore_punctuation:
            from eldar.index import PUNCTUATION

            return term.translate(str.maketrans("", "", PUNCTUATION))
        return term

    def _positive_terms(self, node, negated=False, out=None):
        """Collect the query's positive (non-negated) single-word terms."""
        from eldar.indexops import AND, OR, ANDNOT
        from eldar.entry import IndexEntry

        if out is None:
            out = set()
        if isinstance(node, ANDNOT):
            self._positive_terms(node.left, negated, out)
            self._positive_terms(node.right, not negated, out)  # right is subtracted
        elif isinstance(node, (AND, OR)):
            self._positive_terms(node.left, negated, out)
            self._positive_terms(node.right, negated, out)
        elif isinstance(node, IndexEntry):
            qt = node.query_term
            if not negated and isinstance(qt, str) and "*" not in qt:
                out.add(self._strip_punct(qt))
        return out

    def _term_tf(self, term):
        """{doc_id: tf} for a term (scored index only)."""
        if term not in self._postings:
            return {}
        return {int(r["id"]): int(r["tf"]) for r in self._postings[term]}

    def _doc_len(self, doc_id):
        return int(self._lengths[doc_id]["dl"])

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
        for term in sorted(terms):   # deterministic summation order
            tf_map = self._term_tf(term)
            df = len(tf_map)
            if df == 0:
                continue
            if mode == "bm25":
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            else:  # tfidf
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
        # best score first; ties broken by ascending doc-id for determinism
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    def get_document(self, doc_id):
        """Return the stored document for a doc-id (or None if absent)."""
        self._ensure_loaded()
        key = str(doc_id)
        if key not in self._docs:
            return None
        return json.loads(self._docs[key]["doc"])

    def document_frequency(self, term):
        """Number of documents a term occurs in (length of its postings)."""
        self._ensure_loaded()
        term = self._normalise(term)
        if term not in self._postings:
            return 0
        return len(self._postings[term])

    def __len__(self):
        """Number of live (non-deleted) documents."""
        self._ensure_loaded()
        return len(self._docs) - len(self._deleted)

    def __contains__(self, doc_id):
        self._ensure_loaded()
        key = str(doc_id)
        return key in self._docs and key not in self._deleted
