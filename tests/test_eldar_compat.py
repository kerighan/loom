"""loom must not depend on eldar's module layout (nor on its presence,
except at query time).

The tokenizer constants are vendored — they define the on-disk postings
format and must never shift with the installed eldar version (a p4a/pip
resolution surprise shipped a different eldar than the desktop one).
parse_query is the single symbol resolved from eldar, through a list of
known layouts with a probe; the query-tree walk (_positive_terms) and the
posting Items are layout-independent (duck-typing / vendored).
"""

import importlib
import os
import sys
import tempfile
import types

import pytest

from loom import DB, Search


@pytest.fixture
def docs():
    d = tempfile.mkdtemp()
    db = DB(os.path.join(d, "t.db"))
    db.open()
    col = db.collection("docs", {"id": "utf8[8]", "text": "text"},
                        indexes={"id": "primary",
                                 "ft": Search(fields=["text"], scoring="bm25")})
    col.insert_many([{"id": f"d{i}",
                      "text": f"hello_world mercosur {'fnsea' if i % 2 else 'agri'}"}
                     for i in range(20)])
    yield col
    db.close()


def _reset_resolver():
    import loom.datastructures.search as smod
    smod._PARSE_QUERY = None


def test_vendored_constants_match_eldar_008():
    """The vendored constants must equal eldar 0.0.8's — if this fails,
    someone changed the local eldar's constants; loom deliberately keeps
    its own (existing indexes depend on them), but the divergence should
    be a conscious decision, not a surprise."""
    eldar_regex = importlib.import_module("eldar.regex")
    eldar_index = importlib.import_module("eldar.index")
    from loom.datastructures.search import WORD_REGEX, PUNCTUATION
    assert WORD_REGEX == eldar_regex.WORD_REGEX
    assert PUNCTUATION == eldar_index.PUNCTUATION


def test_indexing_works_without_eldar(monkeypatch):
    """add()/flush() only use the vendored constants — a device build can
    index documents with eldar entirely absent (queries need it)."""
    for m in [m for m in sys.modules if m == "eldar" or m.startswith("eldar.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setitem(sys.modules, "eldar", None)  # import → ImportError
    _reset_resolver()

    d = tempfile.mkdtemp()
    db = DB(os.path.join(d, "no_eldar.db"))
    db.open()
    col = db.collection("docs", {"id": "utf8[8]", "text": "text"},
                        indexes={"id": "primary",
                                 "ft": Search(fields=["text"], scoring="bm25")})
    col.insert_many([{"id": f"d{i}", "text": "hello mercosur"} for i in range(5)])
    with pytest.raises(ImportError, match="eldar"):
        col.search("ft", "mercosur")
    db.close()
    _reset_resolver()


def test_query_survives_restructured_eldar(monkeypatch, docs):
    """Simulated relayout: parse_query only re-exported at the top level,
    eldar.index / eldar.query gone.  Boolean, negation (the duck-typed
    _positive_terms walk must keep BM25 scores > 0) and wildcards work."""
    real_pq = importlib.import_module("eldar.index").parse_query
    real_pq("warmup AND warmup2")            # its internals stay cached
    fake = types.ModuleType("eldar")
    fake.parse_query = real_pq
    monkeypatch.setitem(sys.modules, "eldar", fake)
    monkeypatch.setitem(sys.modules, "eldar.index", None)
    monkeypatch.setitem(sys.modules, "eldar.query", None)
    _reset_resolver()

    odd = {f"d{i}" for i in range(20) if i % 2}
    assert {h["id"] for h in docs.search("ft", "mercosur AND fnsea")} == odd
    ranked = docs.search("ft", "mercosur AND NOT agri", with_scores=True)
    assert ranked and all(s > 0 for _h, s in ranked)
    assert {h["id"] for h, _s in ranked} == odd
    assert len(docs.search("ft", "hello*")) == 20
    _reset_resolver()


def test_incompatible_eldar_raises_clearly(monkeypatch, docs):
    """No usable parse_query anywhere → a clear ImportError naming the
    attempted layouts and the pin, not an AttributeError deep inside."""
    fake = types.ModuleType("eldar")         # top-level, no parse_query
    monkeypatch.setitem(sys.modules, "eldar", fake)
    monkeypatch.setitem(sys.modules, "eldar.index", None)
    monkeypatch.setitem(sys.modules, "eldar.query", None)
    _reset_resolver()
    with pytest.raises(ImportError, match="eldar==0.0.8"):
        docs.search("ft", "mercosur")
    _reset_resolver()
