"""
Persistent Graph using double-indexed nested Dict[Dict].

Stores edges once in a shared dataset, indexed in both directions
via nested Dict[str, Dict[str, edge_attrs]]:
  - outgoing[src][dst] = edge_attrs
  - incoming[dst][src] = edge_attrs

Deletion is double: remove from both outgoing and incoming.
"""

from .base import DataStructure
from .dict import Dict


class Graph(DataStructure):
    """Persistent directed or undirected graph.

    Nodes are stored in a Dict (node_id -> attrs).
    Edges are double-indexed via nested Dict[Dict] for O(1) lookup
    in both directions.

    Usage:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        class Knows(BaseModel):
            weight: float

        with DB("social.db") as db:
            g = db.create_graph("social", Person, Knows)

            g.add_node("alice", name="Alice", age=30)
            g.add_node("bob", name="Bob", age=25)
            g.add_edge("alice", "bob", weight=0.9)

            print(g["alice"])           # node attrs
            print(g.has_edge("alice", "bob"))  # True
            print(list(g.neighbors("alice")))  # ["bob"]

    Performance:
        - add_node:     O(1)
        - add_edge:     O(1)
        - get_node:     O(1)
        - has_edge:     O(1)
        - neighbors:    O(degree)
        - remove_edge:  O(1)
        - remove_node:  O(degree)
    """

    def __init__(self, name, db, node_schema, edge_schema, directed=True, _parent=None):
        self._directed = directed
        self._node_schema_input = node_schema
        self._edge_schema_input = edge_schema
        self.item_schema = None  # set after init

        super().__init__(name, db, _parent=_parent)

        metadata = self._load_metadata()
        if metadata:
            self._load()
        else:
            self._initialize_graph(node_schema, edge_schema)

    def _resolve_schema(self, schema_input):
        """Convert a Pydantic model or dict to a loom schema dict."""
        if isinstance(schema_input, dict):
            return schema_input
        # Pydantic model class
        from loom.schema import schema_from_model
        return schema_from_model(schema_input)

    def _initialize_graph(self, node_schema, edge_schema):
        node_dict = self._resolve_schema(node_schema)
        edge_dict = self._resolve_schema(edge_schema)

        # Node store
        node_ds = self._db.create_dataset(f"_graph_{self.name}_nodes", **node_dict)
        self._nodes = Dict(
            f"_graph_{self.name}_node_dict", self._db, node_ds,
            cache_size=1000, use_bloom=True,
        )

        # Edge datasets (shared by nested dicts)
        edge_ds = self._db.create_dataset(f"_graph_{self.name}_edges", **edge_dict)

        # Outgoing adjacency: src -> {dst -> edge_attrs}
        out_template = Dict.template(edge_ds, cache_size=0)
        self._out = Dict(
            f"_graph_{self.name}_out", self._db, out_template,
            cache_size=1000, use_bloom=False,
        )

        # Incoming adjacency: dst -> {src -> edge_attrs}
        in_template = Dict.template(edge_ds, cache_size=0)
        self._in = Dict(
            f"_graph_{self.name}_in", self._db, in_template,
            cache_size=1000, use_bloom=False,
        )

        self._node_schema = node_dict
        self._edge_schema = edge_dict

        self.save()

    def _initialize(self):
        pass  # handled by _initialize_graph

    def _load(self):
        metadata = self._load_metadata()
        self._node_schema = metadata["node_schema"]
        self._edge_schema = metadata["edge_schema"]
        self._directed = metadata["directed"]
        # Inner structures are loaded lazily — they may not be in
        # _datastructures yet if the registry loop hasn't reached them.
        self._nodes = None
        self._out = None
        self._in = None

    def _ensure_loaded(self):
        """Resolve inner structures by name (lazy, called on first access)."""
        if self._nodes is not None:
            return
        self._nodes = self._db._datastructures[f"_graph_{self.name}_node_dict"]
        self._out = self._db._datastructures[f"_graph_{self.name}_out"]
        self._in = self._db._datastructures[f"_graph_{self.name}_in"]

    def save(self, force=False):
        self._save_metadata({
            "node_schema": self._node_schema,
            "edge_schema": self._edge_schema,
            "directed": self._directed,
        })

    # ---- Registry protocol ----

    def _get_registry_params(self):
        return {
            "node_schema": self._node_schema,
            "edge_schema": self._edge_schema,
            "directed": self._directed,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db,
            params["node_schema"],
            params["edge_schema"],
            directed=params.get("directed", True),
        )

    # ---- Node API ----

    def add_node(self, node_id, **attrs):
        """Add a node with attributes."""
        self._ensure_loaded()
        self._nodes[str(node_id)] = attrs

    def get_node(self, node_id):
        """Get node attributes."""
        self._ensure_loaded()
        return self._nodes[str(node_id)]

    def has_node(self, node_id):
        """Check if a node exists."""
        self._ensure_loaded()
        return str(node_id) in self._nodes

    def remove_node(self, node_id):
        """Remove a node and all its edges."""
        self._ensure_loaded()
        nid = str(node_id)

        # Remove all outgoing edges
        if nid in self._out:
            out_dict = self._out[nid]
            for dst in list(out_dict.keys()):
                # Remove from incoming side
                if nid in self._in and str(dst) in self._in[str(dst)]:
                    try:
                        del self._in[str(dst)][nid]
                    except KeyError:
                        pass
            del self._out[nid]

        # Remove all incoming edges
        if nid in self._in:
            in_dict = self._in[nid]
            for src in list(in_dict.keys()):
                if str(src) in self._out:
                    try:
                        del self._out[str(src)][nid]
                    except KeyError:
                        pass
            del self._in[nid]

        # Remove node itself
        if nid in self._nodes:
            del self._nodes[nid]

    def __getitem__(self, node_id):
        """Get node attributes via g[node_id]."""
        return self.get_node(node_id)

    def __contains__(self, node_id):
        return self.has_node(node_id)

    # ---- Edge API ----

    def add_edge(self, src, dst, **attrs):
        """Add an edge from src to dst with attributes.

        For undirected graphs, the reverse edge is added automatically.
        """
        self._ensure_loaded()
        s, d = str(src), str(dst)

        # Auto-create nodes if they don't exist yet
        # (access to nested dict auto-creates the outer key)

        # Outgoing: src -> dst
        self._out[s][d] = attrs
        # Incoming: dst -> src
        self._in[d][s] = attrs

        if not self._directed:
            # Undirected: also add reverse
            self._out[d][s] = attrs
            self._in[s][d] = attrs

    def get_edge(self, src, dst):
        """Get edge attributes."""
        self._ensure_loaded()
        return self._out[str(src)][str(dst)]

    def has_edge(self, src, dst):
        """Check if an edge exists. O(1)."""
        self._ensure_loaded()
        s, d = str(src), str(dst)
        if s not in self._out:
            return False
        return d in self._out[s]

    def remove_edge(self, src, dst):
        """Remove an edge. Double deletion from both indexes."""
        self._ensure_loaded()
        s, d = str(src), str(dst)

        del self._out[s][d]
        del self._in[d][s]

        if not self._directed:
            try:
                del self._out[d][s]
            except KeyError:
                pass
            try:
                del self._in[s][d]
            except KeyError:
                pass

    # ---- Traversal ----

    def neighbors(self, node_id):
        """Yield neighbor node IDs (outgoing for directed, all for undirected)."""
        self._ensure_loaded()
        nid = str(node_id)
        if nid in self._out:
            yield from self._out[nid].keys()

    def out_edges(self, node_id):
        """Yield (dst, edge_attrs) for outgoing edges."""
        self._ensure_loaded()
        nid = str(node_id)
        if nid in self._out:
            yield from self._out[nid].items()

    def in_edges(self, node_id):
        """Yield (src, edge_attrs) for incoming edges."""
        self._ensure_loaded()
        nid = str(node_id)
        if nid in self._in:
            yield from self._in[nid].items()

    def successors(self, node_id):
        """Alias for neighbors (directed graph terminology)."""
        return self.neighbors(node_id)

    def predecessors(self, node_id):
        """Yield predecessor node IDs (incoming edges)."""
        self._ensure_loaded()
        nid = str(node_id)
        if nid in self._in:
            yield from self._in[nid].keys()

    # ---- Degree ----

    def degree(self, node_id):
        """Total degree (out + in for directed, out for undirected)."""
        if self._directed:
            return self.out_degree(node_id) + self.in_degree(node_id)
        return self.out_degree(node_id)

    def out_degree(self, node_id):
        self._ensure_loaded()
        nid = str(node_id)
        if nid not in self._out:
            return 0
        return len(self._out[nid])

    def in_degree(self, node_id):
        self._ensure_loaded()
        nid = str(node_id)
        if nid not in self._in:
            return 0
        return len(self._in[nid])

    # ---- Iteration ----

    def nodes(self):
        """Iterate over all node IDs."""
        self._ensure_loaded()
        return self._nodes.keys()

    def edges(self):
        """Iterate over all (src, dst, edge_attrs) triples."""
        self._ensure_loaded()
        for src in self._nodes.keys():
            if str(src) in self._out:
                for dst, attrs in self._out[str(src)].items():
                    yield src, dst, attrs

    @property
    def num_nodes(self):
        self._ensure_loaded()
        return len(self._nodes)

    @property
    def num_edges(self):
        return sum(1 for _ in self.edges())

    # ---- Query ----

    def query(self, cypher_str):
        """Execute a Cypher-like query on this graph.

        Args:
            cypher_str: Simplified Cypher query

        Returns:
            List of result dicts

        Examples:
            g.query("MATCH (a)-[r]->(b) WHERE a.age > 25 RETURN a.name, b.name")
            g.query("MATCH (a)->(b) WHERE a.name == 'Alice' RETURN b")
        """
        from loom.query import execute_query
        return list(execute_query(self, cypher_str))

    def query_iter(self, cypher_str):
        """Like query() but returns a lazy iterator (low memory)."""
        from loom.query import execute_query
        return execute_query(self, cypher_str)

    def __repr__(self):
        return f"Graph('{self.name}', nodes={self.num_nodes}, directed={self._directed})"

    def __len__(self):
        return self.num_nodes
