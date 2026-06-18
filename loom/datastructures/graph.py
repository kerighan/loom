"""
Persistent Graph using double-indexed nested Dict[Dict].

Stores edges once in a shared dataset, indexed in both directions
via nested Dict[str, Dict[str, edge_attrs]]:
  - outgoing[src][dst] = edge_attrs
  - incoming[dst][src] = edge_attrs

Deletion is double: remove from both outgoing and incoming.
"""

from .base import DataStructure, write_op
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

    def __init__(
        self,
        name,
        db,
        node_schema,
        edge_schema,
        directed=True,
        node_id_max_len=50,
        label_field=None,
        _parent=None,
    ):
        """
        Args:
            node_id_max_len: Max character length of node IDs (default 50).
                Use a smaller value to save disk space when IDs are short
                (e.g., 10 for numeric IDs "0"–"999999").
            label_field: Name of the node-schema field that holds each node's
                label / type (e.g. "type").  When set, query patterns may use
                the ``(a:Category)`` syntax and ``label(a)`` in WHERE, and
                label-anchored queries are seeded through an in-memory
                label→node-ids index instead of scanning every node.
        """
        self._directed = directed
        self._node_id_max_len = node_id_max_len
        self._label_field = label_field
        self._node_schema_input = node_schema
        self._edge_schema_input = edge_schema
        self.item_schema = None  # set after init
        # Lazy in-memory label index {label_value: [node_id, ...]}, built on
        # first label-anchored query and invalidated on node mutation.  Not
        # persisted: it is a pure cache, fully rebuildable from the node store
        # in one scan, so we avoid paying an index write on every add_node.
        self._label_index = None

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
            f"_graph_{self.name}_node_dict",
            self._db,
            node_ds,
            cache_size=1000,
            use_bloom=True,
        )

        # Edge datasets (shared by nested dicts)
        edge_ds = self._db.create_dataset(f"_graph_{self.name}_edges", **edge_dict)

        key_size = self._node_id_max_len

        # Outgoing adjacency: src -> {dst -> edge_attrs}
        # max_key_len drives the stored `_key` (destination id) width in the
        # shared values dataset — size it to node ids, not the U100 default,
        # or every edge record carries a 400-byte key.
        out_template = Dict.template(edge_ds, cache_size=0)
        self._out = Dict(
            f"_graph_{self.name}_out",
            self._db,
            out_template,
            cache_size=1000,
            use_bloom=False,
            key_size=key_size,
            max_key_len=key_size,
        )

        # Incoming adjacency: dst -> {src -> edge_attrs}
        in_template = Dict.template(edge_ds, cache_size=0)
        self._in = Dict(
            f"_graph_{self.name}_in",
            self._db,
            in_template,
            cache_size=1000,
            use_bloom=False,
            key_size=key_size,
            max_key_len=key_size,
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
        self._node_id_max_len = metadata.get("node_id_max_len", 50)
        self._label_field = metadata.get("label_field", None)
        self._label_index = None
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
        self._save_metadata(
            {
                "node_schema": self._node_schema,
                "edge_schema": self._edge_schema,
                "directed": self._directed,
                "node_id_max_len": self._node_id_max_len,
                "label_field": self._label_field,
            }
        )

    # ---- Registry protocol ----

    def _get_registry_params(self):
        return {
            "node_schema": self._node_schema,
            "edge_schema": self._edge_schema,
            "directed": self._directed,
            "node_id_max_len": self._node_id_max_len,
            "label_field": self._label_field,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name,
            db,
            params["node_schema"],
            params["edge_schema"],
            directed=params.get("directed", True),
            node_id_max_len=params.get("node_id_max_len", 50),
            label_field=params.get("label_field", None),
        )

    # ---- Label index (in-memory, lazy) ----

    def _ensure_label_index(self):
        """Build the {label: [node_id, ...]} index from the node store.

        Lazy and cached for the session; rebuilt after any node mutation.
        Requires ``label_field`` to be set at graph creation.
        """
        if self._label_field is None:
            raise ValueError(
                "This graph has no label_field; pass label_field=... to "
                "create_graph() to use label-anchored queries."
            )
        if self._label_index is not None:
            return self._label_index
        self._ensure_loaded()
        index = {}
        lf = self._label_field
        for nid, attrs in self._nodes.to_dict().items():
            label = attrs.get(lf)
            if label is None:
                continue
            index.setdefault(str(label), []).append(nid)
        self._label_index = index
        return index

    def nodes_with_label(self, label):
        """Return the list of node IDs whose label_field equals ``label``."""
        return list(self._ensure_label_index().get(str(label), []))

    def _invalidate_label_index(self):
        self._label_index = None

    # ---- Node API ----

    @write_op
    def add_node(self, node_id, **attrs):
        """Add a node with attributes."""
        self._ensure_loaded()
        self._nodes[str(node_id)] = attrs
        self._invalidate_label_index()

    def get_node(self, node_id):
        """Get node attributes."""
        self._ensure_loaded()
        return self._nodes[str(node_id)]

    def has_node(self, node_id):
        """Check if a node exists."""
        self._ensure_loaded()
        return str(node_id) in self._nodes

    @write_op
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
        self._invalidate_label_index()

    def __getitem__(self, node_id):
        """Get node attributes via g[node_id]."""
        return self.get_node(node_id)

    def __contains__(self, node_id):
        return self.has_node(node_id)

    # ---- Edge API ----

    @write_op
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

    @write_op
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

    def _bulk_insert_empty_adjacency(self, child, items):
        if len(child) != 0 or child.next_data_offset != 0:
            return False

        dedup = {}
        for key, value in items:
            dedup[str(key)] = value
        n_items = len(dedup)
        if n_items == 0:
            return True

        target_capacity = max(8, n_items * 2)
        target_p = (target_capacity - 1).bit_length()
        capacity = child._get_capacity(target_p)
        probe_range = min(child._get_probe_range(target_p), capacity)
        used_positions = set()
        planned = []

        for key, value in dedup.items():
            internal_key = child._to_internal_key(key)
            key_hash = child._hash(internal_key)
            bucket = key_hash % capacity
            position = None
            for offset in range(probe_range):
                candidate = (bucket + offset) % capacity
                if candidate not in used_positions:
                    position = candidate
                    used_positions.add(candidate)
                    break
            if position is None:
                return False
            planned.append((key, value, internal_key, key_hash, position))

        if target_p != child._p_init:
            table_addr = child._hash_table.allocate_block(capacity)
            child._p_init = target_p
            child.p_last = target_p
            child.table_addrs = [int(table_addr)]
        else:
            table_addr = child.table_addrs[0]

        if n_items > child.values_capacity:
            child.values_block_addr = child._values_dataset.allocate_block(n_items)
            child.values_capacity = n_items

        value_record_size = child._values_dataset.record_size
        hash_record_size = child._hash_table.record_size
        value_addr = child.values_block_addr
        has_stored_key = (
            child._store_key and "_key" in child._values_dataset.user_schema.names
        )

        for index, (key, value, internal_key, key_hash, position) in enumerate(planned):
            current_value_addr = value_addr + index * value_record_size
            record = {"_key": key, **value} if has_stored_key else value
            child._values_dataset.db.write(
                current_value_addr,
                child._values_dataset._serialize(**record),
            )

            if child._hash_keys:
                hash_record = {
                    "hash": key_hash,
                    "key": internal_key,
                    "value_addr": current_value_addr,
                    "valid": True,
                }
            else:
                hi, lo = internal_key
                hash_record = {
                    "hash_hi": hi,
                    "hash_lo": lo,
                    "value_addr": current_value_addr,
                    "valid": True,
                }
            child._hash_table.db.write(
                table_addr + position * hash_record_size,
                child._hash_table._serialize(**hash_record),
            )

        child.next_data_offset = n_items
        child.size = n_items
        child._value_freelist = []
        if (
            child._parent is not None
            and child._parent != "__nested__"
            and child._parent_key is not None
        ):
            child._parent.update_nested_ref(child._parent_key, child)
        child._auto_save_check()
        return True

    def _set_adjacency_batch(self, outer, grouped):
        for node, items in grouped.items():
            child = outer[node]
            if not self._bulk_insert_empty_adjacency(child, items):
                child.set_batch(items)

    # ---- Batch inserts ----

    @write_op
    def add_nodes(self, nodes):
        """Bulk-add nodes, deferring all header flushes to a single flush at the end.

        Dramatically faster than repeated add_node() calls because each
        add_node() triggers a header flush via allocate(). This method
        batches all flushes into one.

        Args:
            nodes: Iterable of (node_id, attrs_dict) tuples, or a dict
                   mapping node_id → attrs_dict.

        Example:
            g.add_nodes([
                ("alice", {"name": "Alice", "age": 30}),
                ("bob",   {"name": "Bob",   "age": 25}),
            ])
            # or
            g.add_nodes({"alice": {"name": "Alice"}, "bob": {"name": "Bob"}})
        """
        self._ensure_loaded()
        if isinstance(nodes, dict):
            nodes = nodes.items()
        with self._db.batch():
            for node_id, attrs in nodes:
                self._nodes[str(node_id)] = attrs
        self._invalidate_label_index()

    @write_op
    def add_edges(self, edges):
        """Bulk-add edges grouped by source/dest to minimise parent ref updates.

        Groups edges by source node for _out (and by dest node for _in) so
        that update_nested_ref is called once per distinct node instead of
        once per edge.  Combined with db.batch(), this gives 3–5× speedup
        over repeated add_edge() calls on dense graphs.

        Args:
            edges: Iterable of (src, dst, attrs_dict) tuples.

        Example:
            g.add_edges([
                ("alice", "bob",   {"weight": 0.9}),
                ("alice", "carol", {"weight": 0.5}),
                ("bob",   "carol", {"weight": 0.7}),
            ])
        """
        self._ensure_loaded()
        from collections import defaultdict

        by_src = defaultdict(list)  # src  -> [(dst, attrs), ...]
        by_dst = defaultdict(list)  # dst  -> [(src, attrs), ...]

        for src, dst, attrs in edges:
            s, d = str(src), str(dst)
            by_src[s].append((d, attrs))
            by_dst[d].append((s, attrs))
            if not self._directed:
                by_src[d].append((s, attrs))
                by_dst[s].append((d, attrs))

        with self._db.batch():
            self._set_adjacency_batch(self._out, by_src)
            self._set_adjacency_batch(self._in, by_dst)

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
        return (
            f"Graph('{self.name}', nodes={self.num_nodes}, directed={self._directed})"
        )

    def __len__(self):
        return self.num_nodes
