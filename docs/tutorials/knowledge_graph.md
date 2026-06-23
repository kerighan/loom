# Knowledge graph — FB15k

A persistent directed graph with typed nodes, interned relations, and
Cypher-like queries — on **FB15k** (a Freebase subset: 14,951 entities,
1,345 relation types, 483,142 edges).

## Get the data

```bash
base=https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15K
mkdir -p benchmarks/data
for s in train valid test; do curl -sf $base/$s.txt -o benchmarks/data/fb15k_$s.txt; done
```

Each line is a triple `head \t relation \t tail`.

## Build the graph

```python
from loom import DB

def load_triples(path):
    with open(path) as f:
        for line in f:
            h, r, t = line.rstrip("\\n").split("\\t")
            yield h, r, t

triples = list(load_triples("benchmarks/data/fb15k_train.txt"))

entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
rels     = sorted({r for _, r, _ in triples})
rel2id   = {r: i for i, r in enumerate(rels)}          # intern relations → uint16

db = DB("fb15k.db", header_size=65536)
g = db.create_graph("kg",
                    node_schema={"type": "utf8[24]"},   # a node label
                    edge_schema={"rel": "uint16"},       # interned relation id
                    directed=True, node_id_max_len=32,
                    label_field="type")
```

## Bulk insert (recommended)

`add_edges` groups edges by source (and by target) and bulk-inserts each
node's whole adjacency in one shot — ~16× faster than per-call `add_edge`.

```python
# a cheap node "type" = the domain of its first relation, e.g. "/film/film" → "film"
def node_type(node):
    return "entity"

g.add_nodes((e, {"type": node_type(e)}) for e in entities)
g.add_edges((h, t, {"rel": rel2id[r]}) for h, r, t in triples)
db.flush()
```

## Point reads

```python
g[entities[0]]                 # node attributes
g.has_edge(h, t)               # edge exists?
g.get_edge(h, t)              # edge attributes ({"rel": ...})
g.out_degree(node)
list(g.neighbors(node))        # outgoing neighbours
```

## Cypher-like queries

`query()` takes a single query string (values are inlined). Use `id(a)` to
match a node by id and `RETURN id(b)` to get neighbour ids.

```python
n = entities[0]

# 1-hop, seeded by a node id
g.query(f"MATCH (a)->(b) WHERE id(a)=='{n}' RETURN id(b)")

# 2-hop chain
g.query(f"MATCH (a)->(b)->(c) WHERE id(a)=='{n}' RETURN id(c) LIMIT 100")

# variable-length paths
g.query(f"MATCH (a)-[*2]->(b) WHERE id(a)=='{n}' RETURN id(b) LIMIT 100")

# filter on edge / node attributes
g.query(f"MATCH (a)-[r]->(b) WHERE id(a)=='{n}' AND r.rel==5 RETURN id(b)")

# label-seeded (uses the in-memory label → nodes index)
g.query("MATCH (x:entity)->(b) RETURN id(b) LIMIT 50")

# lazy iterator for large result sets
for row in g.query_iter(f"MATCH (a)->(b) WHERE id(a)=='{n}' RETURN id(b)"):
    ...
```

```{tip}
n-hop expansion follows shortest paths; label filters `(x:Label)` are seeded
from a label index, so they don't scan the whole graph.
```

## Why interning relations

FB15k has only 1,345 distinct relations but 483 k edges. Storing each edge's
relation as a `uint16` id (2 bytes) instead of a string keeps edges tiny — the
reference benchmark reports ~277 B/edge on disk including both adjacency
directions. Keep a `rel2id` / `id2rel` mapping (e.g. in a loom `Dict`) to
decode.
