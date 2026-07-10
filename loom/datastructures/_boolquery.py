"""Boolean query parser + tree — vendored from eldar 0.0.8.

(https://github.com/kerighan/eldar, same author as loom.)  This is the
index-level flavour: ``parse_query(q)`` builds a tree of AND / OR / ANDNOT
nodes with IndexEntry leaves, and ``tree.search(index)`` evaluates it
against any object exposing ``get(term) -> set of items with .id and
.position`` (loom's SearchIndex).

Vendored so that loom's full-text search has NO runtime dependency and
behaves identically everywhere (desktop, p4a/Android, vendored copies) —
the grammar (AND / OR / AND NOT, parentheses, "quoted phrases", ``*``
wildcards) is frozen, and a pip resolution surprise once shipped a
different eldar than the desktop one.  Kept semantically identical to the
original, with ONE deliberate deviation: accent folding uses loom's
``_fold_accents`` (the same memoized table that normalises documents at
indexing time) instead of calling ``unidecode()`` — so query terms are
normalised *exactly* like the indexed tokens they must match.

The original eldar package remains the reference implementation: the test
suite cross-checks this copy against it whenever eldar is installed.
"""

import re
from collections import defaultdict


# ── query tree nodes (eldar.indexops) ────────────────────────────────────────


class Binary:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class AND(Binary):
    def search(self, index):
        left_match = self.left.search(index)
        right_match = self.right.search(index)
        return left_match.intersection(right_match)

    def __repr__(self):
        return f"({self.left}) AND ({self.right})"


class ANDNOT(Binary):
    def search(self, index):
        left_match = self.left.search(index)
        right_match = self.right.search(index)
        return left_match.difference(right_match)

    def __repr__(self):
        return f"({self.left}) AND NOT ({self.right})"


class OR(Binary):
    def search(self, index):
        left_match = self.left.search(index)
        right_match = self.right.search(index)
        return left_match.union(right_match)

    def __repr__(self):
        return f"({self.left}) OR ({self.right})"


# ── leaves (eldar.entry) ──────────────────────────────────────────────────────


def strip_quotes(query):
    if query[0] == '"' and query[-1] == '"':
        return query[1:-1]
    return query


class IndexEntry:
    def __init__(self, query_term):
        self.not_ = False

        if query_term == "*":
            raise ValueError(
                "Single character wildcards * are not implemented")

        query_term = strip_quotes(query_term)
        if " " in query_term:  # multiword query
            self.query_term = query_term.split()
            self.search = self.search_multiword
        else:
            self.query_term = query_term
            self.search = self.search_simple

    def search_simple(self, index):
        res = index.get(self.query_term)
        return {match.id for match in res}

    def search_multiword(self, index):
        docs = defaultdict(list)
        for token in self.query_term:
            items = index.get(token)
            for item in items:
                docs[item.id].append((item.position, token))

        # utils variable
        first_token = self.query_term[0]
        query_len = len(self.query_term)
        query_rest = self.query_term[1:]
        iter_rest = range(1, query_len)

        results = set()
        for doc_id, tokens in docs.items():
            tokens = sorted(tokens)
            if len(tokens) < query_len:
                continue
            for i in range(len(tokens) - query_len + 1):
                pos, tok = tokens[i]
                if tok != first_token:
                    continue
                is_a_match = True
                for j, correct_token in zip(iter_rest, query_rest):
                    next_pos, next_tok = tokens[i + j]
                    if correct_token != next_tok or next_pos != pos + j:
                        is_a_match = False
                        break
                if is_a_match:
                    results.add(doc_id)
                    break
        return results

    def __repr__(self):
        if self.not_:
            return f'NOT "{self.query_term}"'
        return f'"{self.query_term}"'


# ── parser (eldar.query helpers + eldar.index.parse_query) ───────────────────


def strip_brackets(query):
    count_left = 0
    for i in range(len(query) - 1):
        letter = query[i]
        if letter == "(":
            count_left += 1
        elif letter == ")":
            count_left -= 1
        if i > 0 and count_left == 0:
            return query

    if query[0] == "(" and query[-1] == ")":
        return query[1:-1]
    return query


def is_balanced(query):
    # are brackets balanced
    brackets_b = query.count("(") == query.count(")")
    quotes_b = query.count('"') % 2 == 0
    return brackets_b and quotes_b


def _fold(text):
    # loom's accent folding — the exact normalisation documents get at
    # indexing time (eldar calls unidecode() here; see module docstring).
    from loom.datastructures.search import _fold_accents

    return _fold_accents(text)


def parse_query(query, ignore_case=True, ignore_accent=True):
    # remove brackets around query
    if query[0] == '(' and query[-1] == ')':
        query = strip_brackets(query)
    # if there are quotes around query, make an entry
    if query[0] == '"' and query[-1] == '"' and query.count('"') == 1:
        if ignore_case:
            query = query.lower()
        if ignore_accent:
            query = _fold(query)
        return IndexEntry(query)

    # find all operators
    match = []
    match_iter = re.finditer(r" (AND NOT|AND|OR) ", query, re.IGNORECASE)
    for m in match_iter:
        start = m.start(0)
        end = m.end(0)
        operator = query[start + 1:end - 1].lower()
        match_item = (start, end)
        match.append((operator, match_item))
    match_len = len(match)

    if match_len != 0:
        # stop at first balanced operation
        for i, (operator, (start, end)) in enumerate(match):
            left_part = query[:start]
            if not is_balanced(left_part):
                continue

            right_part = query[end:]
            if not is_balanced(right_part):
                raise ValueError("Query malformed")
            break

        if operator == "or":
            return OR(
                parse_query(left_part, ignore_case, ignore_accent),
                parse_query(right_part, ignore_case, ignore_accent)
            )
        elif operator == "and":
            return AND(
                parse_query(left_part, ignore_case, ignore_accent),
                parse_query(right_part, ignore_case, ignore_accent)
            )
        elif operator == "and not":
            return ANDNOT(
                parse_query(left_part, ignore_case, ignore_accent),
                parse_query(right_part, ignore_case, ignore_accent)
            )
    else:
        if ignore_case:
            query = query.lower()
        if ignore_accent:
            query = _fold(query)
        return IndexEntry(query)
