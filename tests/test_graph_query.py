"""Tests for the Cypher-like query engine (loom/query.py).

Covers the documented grammar: single-hop and variable-length patterns,
edge directions (->, <-, --), all WHERE comparison operators, id() seeding
and IN membership, inline node-property filters, AND/OR, RETURN projection
variants (field, id(), full attr dict, implicit), LIMIT, and query_iter
laziness — plus node labels ((a:Category) syntax, label() sugar, the
label→nodes index) and multi-hop chains ((a)->(b)->(c)) with per-hop
constraints, the knowledge-graph features.

Two behaviours are pinned deliberately because they are easy to regress:

  * Variable-length quantifiers ([*N], [*lo..hi], [+]) use **shortest-path
    BFS** semantics — a node is reported at its shortest hop distance only.
    So (alice)-[*2]->(x) yields only nodes whose *shortest* path is exactly
    2, not every node reachable in some 2-hop walk.
  * The query scan iterates declared nodes (graph._nodes); a node created
    implicitly by add_edge alone (never add_node'd) is not a valid match
    source.
"""

import os
import tempfile

import pytest

from loom.database import DB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    database = DB(path)
    yield database
    database.close()
    os.unlink(path)


def _social(db, directed=True):
    """alice→bob, alice→carol, bob→carol, carol→dave."""
    g = db.create_graph(
        "g",
        {"name": "U20", "age": "int64"},
        {"weight": "float32", "since": "int64"},
        directed=directed,
    )
    g.add_nodes([
        ("alice", {"name": "Alice", "age": 30}),
        ("bob",   {"name": "Bob",   "age": 25}),
        ("carol", {"name": "Carol", "age": 40}),
        ("dave",  {"name": "Dave",  "age": 20}),
    ])
    g.add_edges([
        ("alice", "bob",   {"weight": 0.9, "since": 2020}),
        ("alice", "carol", {"weight": 0.5, "since": 2021}),
        ("bob",   "carol", {"weight": 0.7, "since": 2019}),
        ("carol", "dave",  {"weight": 0.3, "since": 2022}),
    ])
    return g


def _edge_set(rows):
    return {(r["id(a)"], r["id(b)"]) for r in rows}


# ── Pattern / direction ────────────────────────────────────────────────────


class TestPatternDirection:
    def test_one_hop_all_edges(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)-[r]->(b) RETURN id(a), id(b)")
        assert len(rows) == 4
        assert _edge_set(rows) == {
            ("alice", "bob"), ("alice", "carol"),
            ("bob", "carol"), ("carol", "dave"),
        }

    def test_shorthand_edge_no_var(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE id(a)=='alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"bob", "carol"}

    def test_reverse_edge(self, db):
        g = _social(db)
        # who points *into* carol → alice, bob
        rows = g.query("MATCH (a)<-[r]-(b) WHERE id(a)=='carol' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"alice", "bob"}

    def test_undirected_both_directions(self, db):
        g = _social(db, directed=False)
        out = {r["id(b)"] for r in
               g.query("MATCH (a)->(b) WHERE id(a)=='carol' RETURN id(b)")}
        # undirected: carol connects to alice, bob, dave
        assert out == {"alice", "bob", "dave"}

    def test_undirected_dash(self, db):
        g = _social(db, directed=False)
        rows = g.query("MATCH (a)--(b) WHERE id(a)=='alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"bob", "carol"}


# ── WHERE operators ────────────────────────────────────────────────────────


class TestWhere:
    def test_attr_gt(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE a.age > 28 RETURN id(a), id(b)")
        assert _edge_set(rows) == {
            ("alice", "bob"), ("alice", "carol"), ("carol", "dave"),
        }

    def test_attr_lt(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE b.age < 25 RETURN id(a), id(b)")
        assert _edge_set(rows) == {("carol", "dave")}

    def test_attr_eq(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE a.name == 'Bob' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"carol"}

    def test_attr_neq(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE a.name != 'Alice' RETURN id(a), id(b)")
        assert _edge_set(rows) == {("bob", "carol"), ("carol", "dave")}

    def test_attr_gte_lte(self, db):
        g = _social(db)
        gte = g.query("MATCH (a)->(b) WHERE a.age >= 30 RETURN id(a), id(b)")
        assert _edge_set(gte) == {
            ("alice", "bob"), ("alice", "carol"), ("carol", "dave"),
        }
        lte = g.query("MATCH (a)->(b) WHERE a.age <= 25 RETURN id(a), id(b)")
        assert _edge_set(lte) == {("bob", "carol")}

    def test_edge_attr_filter(self, db):
        g = _social(db)
        rows = g.query(
            "MATCH (a)-[r]->(b) WHERE r.weight > 0.6 RETURN id(a), id(b)"
        )
        assert _edge_set(rows) == {("alice", "bob"), ("bob", "carol")}

    def test_id_eq_seed(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE id(a) == 'alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"bob", "carol"}

    def test_id_in(self, db):
        g = _social(db)
        rows = g.query(
            "MATCH (a)->(b) WHERE id(a) IN ['alice', 'bob'] RETURN id(a), id(b)"
        )
        assert _edge_set(rows) == {
            ("alice", "bob"), ("alice", "carol"), ("bob", "carol"),
        }

    def test_and(self, db):
        g = _social(db)
        rows = g.query(
            "MATCH (a)->(b) WHERE a.age > 28 AND b.age < 25 RETURN id(a), id(b)"
        )
        assert _edge_set(rows) == {("carol", "dave")}

    def test_or(self, db):
        g = _social(db)
        rows = g.query(
            "MATCH (a)->(b) WHERE id(a)=='alice' OR id(a)=='carol' "
            "RETURN id(a), id(b)"
        )
        assert _edge_set(rows) == {
            ("alice", "bob"), ("alice", "carol"), ("carol", "dave"),
        }

    def test_empty_result(self, db):
        g = _social(db)
        assert g.query("MATCH (a)->(b) WHERE a.age > 100 RETURN id(a)") == []


# ── Inline node properties ────────────────────────────────────────────────


class TestInlineProps:
    def test_inline_src_props(self, db):
        g = _social(db)
        rows = g.query("MATCH (a {name:'Alice'})->(b) RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"bob", "carol"}

    def test_inline_dst_props(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b {name:'Carol'}) RETURN id(a)")
        assert {r["id(a)"] for r in rows} == {"alice", "bob"}


# ── RETURN variants ────────────────────────────────────────────────────────


class TestReturn:
    def test_return_node_fields(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE id(a)=='alice' RETURN a.name, b.name")
        assert {r["b.name"] for r in rows} == {"Bob", "Carol"}
        assert all(r["a.name"] == "Alice" for r in rows)

    def test_return_edge_field(self, db):
        g = _social(db)
        rows = g.query(
            "MATCH (a)-[r]->(b) WHERE id(a)=='alice' AND id(b)=='bob' "
            "RETURN r.weight"
        )
        assert len(rows) == 1
        assert rows[0]["r.weight"] == pytest.approx(0.9, abs=1e-5)

    def test_return_full_node_dict(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE id(a)=='bob' RETURN a")
        assert len(rows) == 1
        assert rows[0]["a"]["name"] == "Bob"
        assert rows[0]["a"]["age"] == 25

    def test_implicit_return(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)->(b) WHERE id(a)=='bob'")
        assert len(rows) == 1
        row = rows[0]
        assert row["id(a)"] == "bob"
        assert row["id(b)"] == "carol"
        assert row["a"]["name"] == "Bob"


# ── LIMIT ───────────────────────────────────────────────────────────────────


class TestLimit:
    def test_limit_truncates(self, db):
        g = _social(db)
        assert len(g.query("MATCH (a)->(b) RETURN id(a) LIMIT 2")) == 2

    def test_limit_above_total(self, db):
        g = _social(db)
        assert len(g.query("MATCH (a)->(b) RETURN id(a) LIMIT 999")) == 4


# ── Variable-length paths (shortest-path BFS semantics) ────────────────────


class TestQuantifiers:
    def test_plus_reaches_all_descendants(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)-[+]->(b) WHERE id(a)=='alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"bob", "carol", "dave"}

    def test_star_same_as_plus(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)-[*]->(b) WHERE id(a)=='alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"bob", "carol", "dave"}

    def test_exact_n_shortest_path(self, db):
        g = _social(db)
        # carol's shortest path from alice is 1 hop, so [*2] excludes it;
        # only dave has a shortest path of exactly 2 (alice→carol→dave).
        rows = g.query("MATCH (a)-[*2]->(b) WHERE id(a)=='alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"dave"}

    def test_range_min_max(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)-[*2..4]->(b) WHERE id(a)=='alice' RETURN id(b)")
        # shortest-path distances ≥ 2: only dave (dist 2)
        assert {r["id(b)"] for r in rows} == {"dave"}

    def test_optional_includes_self_and_one_hop(self, db):
        g = _social(db)
        rows = g.query("MATCH (a)-[?]->(b) WHERE id(a)=='alice' RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"alice", "bob", "carol"}


# ── Lazy iteration ──────────────────────────────────────────────────────────


class TestQueryIter:
    def test_query_iter_is_lazy(self, db):
        g = _social(db)
        it = g.query_iter("MATCH (a)->(b) RETURN id(a), id(b)")
        assert not isinstance(it, list)
        assert hasattr(it, "__next__")
        rows = list(it)
        assert len(rows) == 4

    def test_query_iter_matches_query(self, db):
        g = _social(db)
        q = "MATCH (a)->(b) WHERE id(a)=='alice' RETURN id(b)"
        assert list(g.query_iter(q)) == g.query(q)


# ── Edge cases / pinned limitations ────────────────────────────────────────


class TestEdgeCases:
    def test_edge_only_node_not_seedable(self, db):
        """A node created by add_edge alone (no add_node) is not iterated by
        the query scan, so it cannot be a match source — documented limit."""
        g = db.create_graph("g", {"name": "U20"}, {"w": "float32"}, directed=True)
        g.add_edge("p", "q", w=1.0)  # no add_node
        assert g.has_edge("p", "q") is True
        assert g.query("MATCH (a)->(b) WHERE id(a)=='p' RETURN id(b)") == []

    def test_query_after_reopen(self, db):
        g = _social(db)
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            g2 = db2._datastructures["g"]
            rows = g2.query("MATCH (a)->(b) WHERE id(a)=='alice' RETURN id(b)")
            assert {r["id(b)"] for r in rows} == {"bob", "carol"}
        finally:
            db2.close()


# ── Node labels + multi-hop chains (knowledge-graph features) ───────────────


def _kg(db):
    """category → company → employee layered knowledge graph."""
    g = db.create_graph(
        "kg",
        {"type": "U20", "name": "U40", "sector": "U20", "title": "U30"},
        {"rel": "U20"},
        directed=True,
        label_field="type",
    )
    g.add_nodes([
        ("tech",   {"type": "category", "name": "Tech"}),
        ("acme",   {"type": "company",  "name": "Acme",   "sector": "SaaS"}),
        ("globex", {"type": "company",  "name": "Globex", "sector": "Hardware"}),
        ("alice",  {"type": "employee", "name": "Alice",  "title": "CTO"}),
        ("bob",    {"type": "employee", "name": "Bob",    "title": "Eng"}),
        ("carol",  {"type": "employee", "name": "Carol",  "title": "Eng"}),
    ])
    g.add_edges([
        ("tech", "acme",     {"rel": "has"}),
        ("tech", "globex",   {"rel": "has"}),
        ("acme", "alice",    {"rel": "emp"}),
        ("acme", "bob",      {"rel": "emp"}),
        ("globex", "carol",  {"rel": "emp"}),
    ])
    return g


class TestLabelIndex:
    def test_nodes_with_label(self, db):
        g = _kg(db)
        assert sorted(g.nodes_with_label("company")) == ["acme", "globex"]
        assert sorted(g.nodes_with_label("employee")) == ["alice", "bob", "carol"]
        assert g.nodes_with_label("missing") == []

    def test_index_rebuilds_after_add_node(self, db):
        g = _kg(db)
        assert sorted(g.nodes_with_label("company")) == ["acme", "globex"]
        g.add_node("initech", type="company", name="Initech")
        assert sorted(g.nodes_with_label("company")) == ["acme", "globex", "initech"]

    def test_index_rebuilds_after_remove_node(self, db):
        g = _kg(db)
        g.nodes_with_label("employee")  # build it
        g.remove_node("carol")
        assert sorted(g.nodes_with_label("employee")) == ["alice", "bob"]

    def test_label_requires_label_field(self, db):
        g = db.create_graph("plain", {"name": "U20"}, {"w": "float32"})
        with pytest.raises(ValueError):
            g.nodes_with_label("anything")


class TestLabelQueries:
    def test_label_filter_one_hop(self, db):
        g = _kg(db)
        rows = g.query("MATCH (a:company)->(b:employee) RETURN id(a), id(b)")
        assert {(r["id(a)"], r["id(b)"]) for r in rows} == {
            ("acme", "alice"), ("acme", "bob"), ("globex", "carol"),
        }

    def test_label_filter_src_only(self, db):
        g = _kg(db)
        rows = g.query("MATCH (a:category)->(b) RETURN id(b)")
        assert {r["id(b)"] for r in rows} == {"acme", "globex"}

    def test_label_in_where_sugar(self, db):
        g = _kg(db)
        rows = g.query(
            "MATCH (a)->(b) WHERE label(a)=='company' AND label(b)=='employee' "
            "RETURN id(b)"
        )
        assert {r["id(b)"] for r in rows} == {"alice", "bob", "carol"}

    def test_label_unknown_field_raises(self, db):
        g = db.create_graph("plain", {"name": "U20"}, {"w": "float32"})
        g.add_node("x", name="X")
        with pytest.raises(ValueError):
            g.query("MATCH (a:Foo)->(b) RETURN id(b)")


class TestMultiHopChains:
    def test_three_node_chain(self, db):
        g = _kg(db)
        rows = g.query("MATCH (a:category)->(b:company)->(c:employee) RETURN id(c)")
        assert sorted(r["id(c)"] for r in rows) == ["alice", "bob", "carol"]

    def test_intermediate_node_constraint(self, db):
        """The key KG feature: constrain the middle hop (company.sector)."""
        g = _kg(db)
        rows = g.query(
            "MATCH (a:category)->(b:company)->(c:employee) "
            "WHERE b.sector=='SaaS' RETURN id(c)"
        )
        assert sorted(r["id(c)"] for r in rows) == ["alice", "bob"]

    def test_chain_returns_all_levels(self, db):
        g = _kg(db)
        rows = g.query(
            "MATCH (a:category)->(b:company)->(c:employee) "
            "WHERE id(a)=='tech' AND c.title=='CTO' "
            "RETURN id(b), c.name, label(c)"
        )
        assert rows == [{"id(b)": "acme", "c.name": "Alice", "c.type": "employee"}]

    def test_chain_binds_edge_var(self, db):
        g = _kg(db)
        rows = g.query(
            "MATCH (a:company)-[r]->(c:employee) WHERE id(a)=='acme' "
            "RETURN id(c), r.rel"
        )
        assert all(r["r.rel"] == "emp" for r in rows)
        assert {r["id(c)"] for r in rows} == {"alice", "bob"}

    def test_four_node_chain(self, db):
        g = db.create_graph(
            "line", {"type": "U10", "name": "U10"}, {"w": "float32"},
            directed=True, label_field="type",
        )
        g.add_nodes([(c, {"type": c, "name": c.upper()}) for c in "abcd"])
        g.add_edges([("a", "b", {"w": 1}), ("b", "c", {"w": 1}), ("c", "d", {"w": 1})])
        rows = g.query("MATCH (a:a)->(b:b)->(c:c)->(d:d) RETURN id(d)")
        assert [r["id(d)"] for r in rows] == ["d"]

    def test_chain_empty_when_no_match(self, db):
        g = _kg(db)
        rows = g.query(
            "MATCH (a:category)->(b:company)->(c:employee) "
            "WHERE b.sector=='Nonexistent' RETURN id(c)"
        )
        assert rows == []

    def test_variable_length_in_chain_raises(self, db):
        g = _kg(db)
        with pytest.raises(ValueError):
            g.query("MATCH (a:category)-[*2]->(b)->(c) RETURN id(c)")

    def test_limit_on_chain(self, db):
        g = _kg(db)
        rows = g.query("MATCH (a:category)->(b:company)->(c:employee) RETURN id(c) LIMIT 2")
        assert len(rows) == 2


class TestLabelPersistence:
    def test_label_field_survives_reopen(self, db):
        g = _kg(db)
        path = db.filename
        db.close()

        db2 = DB(path)
        try:
            g2 = db2._datastructures["kg"]
            assert g2._label_field == "type"
            rows = g2.query(
                "MATCH (a:category)->(b:company)->(c:employee) "
                "WHERE b.sector=='SaaS' RETURN id(c)"
            )
            assert sorted(r["id(c)"] for r in rows) == ["alice", "bob"]
        finally:
            db2.close()
