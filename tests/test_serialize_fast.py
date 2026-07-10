"""The struct-pack fast serialize plan must be byte-identical to the generic
numpy path — including blob layout — and the tokenizer fast path must match
eldar's legacy semantics exactly.  Both are hot-path optimizations whose only
acceptable observable effect is speed.
"""

import hashlib
import os
import random
import re
import tempfile
from datetime import datetime

import numpy as np
import pytest

from loom import DB, Many, Search


def _build(path, disable_fast):
    """One mixed workload (text/json/vector/datetime, upserts, deletes)."""
    import loom.dataset as dsmod
    random.seed(31)
    rng = np.random.default_rng(31)
    orig = dsmod.Dataset._build_fast_plan
    if disable_fast:
        dsmod.Dataset._build_fast_plan = lambda self: None
    try:
        db = DB(path, header_size=131072)
        db.open()
        col = db.collection("posts", {"id": "utf8[16]", "grp": "utf8[16]",
                                      "score": "float64", "ok": "bool",
                                      "ts": "datetime", "body": "text",
                                      "meta": "json", "emb": "float32[8]"},
                            indexes={"id": "primary",
                                     "grp": Many(sort="score", counted=True),
                                     "ft": Search(fields=["body"], scoring="bm25")})
        words = "alpha under_score l'ami déjà bravo".split()
        def rec(i):
            return {"id": f"p{i}", "grp": f"g{i%5}", "score": random.random(),
                    "ok": bool(i % 2),
                    "ts": datetime(2026, 1 + i % 6, 1 + i % 27),
                    "body": " ".join(random.choices(words, k=8)) if i % 5 else "",
                    "meta": None if i % 7 == 0 else {"k": i % 3},
                    "emb": rng.normal(size=8).astype(np.float32)}
        for i in range(150):
            col.insert(rec(i))
        col.insert_many([rec(i) for i in range(150, 350)])
        for i in range(0, 350, 7):
            col.update(f"p{i}", body="updated " + words[i % 5])
        for i in range(0, 350, 11):
            col.delete(f"p{i}")
        db.close()
    finally:
        dsmod.Dataset._build_fast_plan = orig
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_fast_plan_file_is_byte_identical_to_generic():
    d = tempfile.mkdtemp()
    h_fast = _build(os.path.join(d, "fast.db"), disable_fast=False)
    h_generic = _build(os.path.join(d, "gen.db"), disable_fast=True)
    assert h_fast == h_generic


def test_tokens_match_legacy_semantics():
    """The set-lookup tokenizer path == the literal legacy translate loop,
    across every ignore_* combination, on punctuation/underscore/accent
    edge cases ('_' strips inside tokens; '\"' and '’' are NOT punctuation
    for eldar and survive as tokens)."""
    from loom.datastructures.search import (SearchIndex, _fold_accents,
                                            WORD_REGEX, PUNCTUATION)

    table = str.maketrans("", "", PUNCTUATION)

    def legacy(si, text):
        if si.ignore_case:
            text = text.lower()
        if si.ignore_accent:
            text = _fold_accents(text)
        toks = re.findall(WORD_REGEX, text, re.UNICODE)
        if si.ignore_punctuation:
            toks = [t.translate(table) for t in toks]
        return [t for t in toks if t]

    si = SearchIndex.__new__(SearchIndex)
    corpus = [
        "hello_world foo__bar _ __ ___x trail_ _lead",
        'il a dit "bonjour" — l\'été était là, déjà !',
        "C++ n'est pas C# (vraiment?) a+b=c x<y>z",
        "émile müller—strasse œuf 北京 русский",
        "a'b c’d e\"f g\\h i/j [k] (l) m,n;o:p!q?r",
        "", "___", '"', "’", "\\",
    ]
    for case in (True, False):
        for accent in (True, False):
            for punct in (True, False):
                si.ignore_case, si.ignore_accent = case, accent
                si.ignore_punctuation = punct
                for text in corpus:
                    assert si._tokens(text) == legacy(si, text), (
                        case, accent, punct, text)
