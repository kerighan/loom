"""Tests for the Graph datastructure.

Focuses on correctness under load — earlier versions silently dropped
edges from hub nodes because nested-Dict children's reconstructed
``values_capacity`` defaulted to 10 000 while the actually allocated
block was only 8 records.  Inserts past slot 8 then trampled adjacent
allocations.
"""

import os
import random
import tempfile

import pytest

from loom.database import DB


def _ba_edges(n, m=2, seed=42):
    """Generate Barabási-Albert edge list (preferential attachment)."""
    rng = random.Random(seed)
    edges = []
    repeated = list(range(m))
    for i in range(m, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(repeated))
        for j in chosen:
            edges.append((i, j))
        repeated.extend(chosen)
        repeated.append(i)
        repeated.extend([i] * (m - 1))
    return edges


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".loom")
    os.close(fd)
    db = DB(path)
    yield db
    db.close()
    os.unlink(path)


def test_basic_directed(db):
    g = db.create_graph(
        "g",
        node_schema={"name": "U30"},
        edge_schema={"weight": "float32"},
        directed=True,
    )
    g.add_node("a", name="Alice")
    g.add_node("b", name="Bob")
    g.add_edge("a", "b", weight=0.5)

    assert g.has_edge("a", "b")
    assert not g.has_edge("b", "a")
    assert g.get_edge("a", "b")["weight"] == pytest.approx(0.5)
    assert list(g.neighbors("a")) == ["b"]
    assert list(g.neighbors("b")) == []


def test_basic_undirected(db):
    g = db.create_graph(
        "g",
        node_schema={"name": "U30"},
        edge_schema={"weight": "float32"},
        directed=False,
    )
    g.add_edge("a", "b", weight=1.0)
    assert g.has_edge("a", "b")
    assert g.has_edge("b", "a")
    assert set(g.neighbors("a")) == {"b"}
    assert set(g.neighbors("b")) == {"a"}


@pytest.mark.parametrize("n", [50, 200, 1000])
def test_ba_graph_no_edges_lost(db, n):
    """Regression: BA graphs used to lose edges from hub nodes because
    reconstructed nested children defaulted ``values_capacity`` to 10 000
    while the block was only 8 records, letting later inserts overwrite
    neighbouring allocations.
    """
    edges = _ba_edges(n, m=2)
    g = db.create_graph(
        "g",
        node_schema={"name": "U30"},
        edge_schema={"weight": "float32"},
        directed=True,
    )
    for i in range(n):
        g.add_node(str(i), name=f"U{i}")
    for u, v in edges:
        g.add_edge(str(u), str(v), weight=1.0)

    missing = [(u, v) for u, v in edges if not g.has_edge(str(u), str(v))]
    assert missing == [], f"{len(missing)} edges lost (n={n})"

    # Spot-check get_edge on hub nodes (preferential attachment → low IDs
    # collect lots of incoming edges).
    hub_in = [(u, v) for u, v in edges if v in (0, 1)]
    for u, v in hub_in[:50]:
        assert g.get_edge(str(u), str(v))["weight"] == pytest.approx(1.0)


def test_ba_graph_persists_across_reopen(db):
    """Edges must survive a close / reopen cycle (saved ref state must
    include the actual values_capacity, not a hard-coded default).
    """
    edges = _ba_edges(100, m=2)
    g = db.create_graph(
        "g",
        node_schema={"name": "U30"},
        edge_schema={"weight": "float32"},
        directed=True,
    )
    for i in range(100):
        g.add_node(str(i), name=f"U{i}")
    for u, v in edges:
        g.add_edge(str(u), str(v), weight=1.0)

    path = db.filename
    db.close()

    db2 = DB(path)
    try:
        g2 = db2._datastructures["g"]
        for u, v in edges:
            assert g2.has_edge(str(u), str(v)), f"edge ({u},{v}) lost after reopen"
    finally:
        db2.close()
