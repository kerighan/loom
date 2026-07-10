"""Full-text search is self-contained: the query machinery is vendored
(loom/datastructures/_boolquery.py, from eldar 0.0.8 — same author).

eldar is now a TEST-ONLY dependency: whenever it is installed, the vendored
parser and the tokenizer constants are cross-checked against it, so the
internal copy can never silently drift from the reference implementation.
"""

import os
import sys
import tempfile

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
                      "text": f"hello_world mercosur {'fnsea' if i % 2 else 'agri'} "
                              f"{'déjà l’été' if i % 3 == 0 else 'plaine'}"}
                     for i in range(24)])
    yield col
    db.close()


def test_search_works_without_eldar(monkeypatch, docs):
    """Indexing AND querying — boolean, negation with BM25 scores, wildcards,
    quoted phrases — with eldar entirely absent."""
    for m in [m for m in sys.modules if m == "eldar" or m.startswith("eldar.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setitem(sys.modules, "eldar", None)  # import → ImportError

    odd = {f"d{i}" for i in range(24) if i % 2}
    assert {h["id"] for h in docs.search("ft", "mercosur AND fnsea")} == odd
    ranked = docs.search("ft", "mercosur AND NOT agri", with_scores=True)
    assert ranked and all(s > 0 for _h, s in ranked)
    assert {h["id"] for h, _s in ranked} == odd
    assert len(docs.search("ft", "hello*")) == 24
    assert {h["id"] for h in docs.search("ft", "déjà AND (fnsea OR agri)")} \
        == {f"d{i}" for i in range(24) if i % 3 == 0}


@pytest.mark.parametrize("query", [
    "mercosur", "mercosur AND fnsea", "mercosur AND NOT agri",
    "(fnsea OR agri) AND mercosur", "fnsea OR (agri AND NOT mercosur)",
    "hello*", "merc* AND NOT fn*", "DÉJÀ", "l'été",
    '"hello_world mercosur"', "plaine AND NOT (fnsea OR agri)",
    "a AND NOT b AND NOT c", "((mercosur))",
])
def test_vendored_parser_matches_eldar(query, docs):
    """The vendored tree and eldar's must select the same documents on the
    same index — eldar as reference implementation."""
    eldar_index = pytest.importorskip("eldar.index")
    si = docs._search["ft"]["index"]
    si.flush()

    from loom.datastructures._boolquery import parse_query
    got = parse_query(query, ignore_case=True, ignore_accent=True).search(si)
    expected = eldar_index.parse_query(
        query, ignore_case=True, ignore_accent=True).search(si)
    assert got == expected, f"{query!r}: vendored {got} != eldar {expected}"


def test_vendored_constants_match_eldar():
    """The vendored tokenizer constants must equal the reference eldar's —
    a divergence must be a conscious decision (existing indexes depend on
    them), never a surprise."""
    eldar_regex = pytest.importorskip("eldar.regex")
    eldar_index = pytest.importorskip("eldar.index")
    from loom.datastructures.search import WORD_REGEX, PUNCTUATION
    assert WORD_REGEX == eldar_regex.WORD_REGEX
    assert PUNCTUATION == eldar_index.PUNCTUATION
