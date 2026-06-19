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
        """
        self._text_fields = list(text_fields) if text_fields else None
        self.ignore_case = ignore_case
        self.ignore_accent = ignore_accent
        self.ignore_punctuation = ignore_punctuation
        self._doc_id_dtype = doc_id_dtype
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
        self.save()

    def _load(self):
        m = self._load_metadata()
        self._text_fields = m.get("text_fields")
        self.ignore_case = m.get("ignore_case", True)
        self.ignore_accent = m.get("ignore_accent", True)
        self.ignore_punctuation = m.get("ignore_punctuation", True)
        self._doc_id_dtype = m.get("doc_id_dtype", "uint32")
        # inner structures resolved on first access (registry may still be loading)
        self._postings = self._docs = self._deleted = self._meta = None

    def _ensure_loaded(self):
        if self._postings is not None:
            return
        n = self.name
        self._postings = self._db._datastructures[f"_search_{n}_postings"]
        self._docs = self._db._datastructures[f"_search_{n}_docs"]
        self._deleted = self._db._datastructures[f"_search_{n}_deleted"]
        self._meta = self._db._datastructures[f"_search_{n}_meta"]

    def save(self, force=False):
        self._save_metadata(
            {
                "text_fields": self._text_fields,
                "ignore_case": self.ignore_case,
                "ignore_accent": self.ignore_accent,
                "ignore_punctuation": self.ignore_punctuation,
                "doc_id_dtype": self._doc_id_dtype,
            }
        )

    def _get_registry_params(self):
        return {
            "text_fields": self._text_fields,
            "ignore_case": self.ignore_case,
            "ignore_accent": self.ignore_accent,
            "ignore_punctuation": self.ignore_punctuation,
            "doc_id_dtype": self._doc_id_dtype,
        }

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

        self._docs[str(doc_id)] = {"doc": json.dumps(document)}
        # Unique terms only — boolean index, one posting per (term, doc).
        for term in set(self._tokens(self._doc_text(document))):
            # doc_id is monotonically increasing → append keeps postings sorted.
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

    def search(self, query, return_ids=False, limit=None):
        """Run a boolean query; return matching documents (or doc-ids).

        Supports AND / OR / AND NOT, parentheses and ``*`` wildcards via eldar.
        Results are doc-id sorted (ascending) and exclude tombstoned docs.
        """
        _require_eldar()
        from eldar.index import parse_query

        self._ensure_loaded()
        q = query.strip()
        if not q:
            return []
        tree = parse_query(
            q, ignore_case=self.ignore_case, ignore_accent=self.ignore_accent
        )
        ids = tree.search(self)  # set of int doc-ids

        if len(self._deleted):
            ids = {i for i in ids if str(i) not in self._deleted}
        ids = sorted(ids)
        if limit is not None:
            ids = ids[:limit]
        if return_ids:
            return ids
        return [self.get_document(i) for i in ids]

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
