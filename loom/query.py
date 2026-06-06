"""
Cypher-like query engine for loom Graph.

Grammar
-------
    MATCH <pattern> [WHERE <conditions>] [RETURN <projections>] [LIMIT <n>]

Pattern
    (a)                     node variable
    (a {name:"Alice"})      inline property filter → implicitly added to WHERE
    (a)-[r]->(b)            directed edge with variable
    (a)->(b)                directed edge without variable
    (a)<-[r]-(b)            reverse edge
    (a)-[r]-(b)             undirected edge
    (a)-[*]->(b)            1..∞ hops  (also written [+])
    (a)-[*0..]->(b)         0..∞ hops
    (a)-[?]->(b)            0..1 hop
    (a)-[*3]->(b)           exactly 3 hops
    (a)-[*2..5]->(b)        2..5 hops

    Variable-length hops use shortest-path BFS: a destination node is
    reported once, at its MINIMUM hop distance from the source.  So
    "[*N]" means "nodes whose shortest path is exactly N", not "every
    endpoint of an N-step walk" — a node reachable in 1 hop never shows up
    under [*2], and [*2] can be empty even when 2-hop paths exist.  For the
    whole neighbourhood within N hops, use "[*1..N]".  This is intentional
    (no path explosion) and differs from Neo4j path-enumeration semantics.

WHERE
    a.field == value         attribute comparison
    a.field != value
    a.field > / >= / < / <= value
    id(a) == "alice"         filter by node key
    id(a) IN ["a","b"]       key membership
    cond AND cond
    cond OR cond

RETURN
    a.name, b.age, r.weight  projected fields
    a, b                     full attribute dicts
    id(a), id(b)             node keys

LIMIT n                      truncate results

Examples
--------
    MATCH (a)-[r]->(b) WHERE a.age > 25 RETURN a.name, b.name, r.weight
    MATCH (a {name:"Alice"})->(b) RETURN id(b), b.age
    MATCH (a)-[*2..4]->(b) WHERE id(a)=="0" RETURN id(b)
    MATCH (a)-[+]->(b) WHERE a.age < 30 AND b.age > 20 RETURN a.name LIMIT 5
"""

from __future__ import annotations

import math
import operator
import re
from collections import deque
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Operator map
# ---------------------------------------------------------------------------

_OPS: dict[str, Any] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
}

# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def _parse_value(s: str) -> Any:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s in ("True", "true"):   return True
    if s in ("False", "false"): return False
    if s in ("None", "null"):   return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_list_literal(s: str) -> list:
    """Parse ["a","b"] or [1,2,3]."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected list literal, got: {s}")
    items = [_parse_value(x) for x in s[1:-1].split(",") if x.strip()]
    return items


# ---------------------------------------------------------------------------
# Pattern parser
# ---------------------------------------------------------------------------

# Edge pattern: captures quantifier group
_EDGE_RE = re.compile(
    r"(?:"
    r"<-\[(\w*)\s*(\*[0-9.]+(?:\.\.[0-9]+)?|\*|[+?])?\s*\]-"   # <-[var quant]-
    r"|<-"                                                        # <- shorthand
    r"|-\[(\w*)\s*(\*[0-9.]+(?:\.\.[0-9]+)?|\*|[+?])?\s*\]->"  # -[var quant]->
    r"|->"                                                        # -> shorthand
    r"|-\[(\w*)\s*(\*[0-9.]+(?:\.\.[0-9]+)?|\*|[+?])?\s*\]-"   # -[var quant]-
    r"|--"                                                        # undirected shorthand
    r")"
)

_NODE_BARE_RE  = re.compile(r"\((\w+)\)")
_NODE_PROPS_RE = re.compile(r"\((\w+)\s*\{([^}]*)\}\s*\)")
# Full node: (var), (var:Label), (var {props}), (var:Label {props})
_NODE_RE = re.compile(r"\(\s*(\w+)?\s*(?::\s*(\w+))?\s*(\{[^}]*\})?\s*\)")
_LABEL_CALL_RE = re.compile(r"label\(\s*(\w+)\s*\)")


def _parse_props(props_str: str) -> dict[str, Any]:
    """Parse '{name:"Alice", age:30}' → {'name': 'Alice', 'age': 30}."""
    result = {}
    for pair in re.finditer(r'(\w+)\s*:\s*("(?:[^"]*)"|\'(?:[^\']*)\'|[^,}]+)', props_str):
        key, val = pair.group(1), pair.group(2).strip()
        result[key] = _parse_value(val)
    return result


def _parse_quantifier(q: str | None) -> tuple[int, int]:
    """Return (min_hops, max_hops). max_hops=∞ means no limit."""
    INF = math.inf
    if q is None:           return (1, 1)       # plain edge
    if q == "*":            return (1, INF)
    if q == "+":            return (1, INF)
    if q == "?":            return (0, 1)
    # *N  or  *N..M
    m = re.match(r"\*(\d+)(?:\.\.(\d+))?$", q)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        return (lo, hi)
    # *0.. style
    m2 = re.match(r"\*(\d+)\.\.$", q)
    if m2:
        return (int(m2.group(1)), INF)
    return (1, 1)


def _parse_edge_token(edge_txt: str) -> dict:
    """Parse one edge token into {var, direction, min_hops, max_hops}."""
    if edge_txt.startswith("<-"):
        direction = "in"
        m = re.match(r"<-\[(\w*)\s*([*+?]?[0-9.]*(?:\.\.[0-9]+)?)?\s*\]-", edge_txt)
    elif "->" in edge_txt:
        direction = "out"
        m = re.match(r"-\[(\w*)\s*([*+?][0-9.]*(?:\.\.[0-9]+)?)?\s*\]->", edge_txt)
    else:
        direction = "both"
        m = re.match(r"-\[(\w*)\s*([*+?][0-9.]*(?:\.\.[0-9]+)?)?\s*\]-", edge_txt)

    edge_var = m.group(1) if m else ""
    quant_str = m.group(2) if (m and m.group(2)) else None
    min_hops, max_hops = _parse_quantifier(quant_str)
    return {
        "var": edge_var or None,
        "direction": direction,
        "min_hops": min_hops,
        "max_hops": max_hops,
    }


def parse_pattern(s: str) -> dict:
    """Parse a node/edge chain.

    Returns {"nodes": [{var, label, props}, ...],
             "edges": [{var, direction, min_hops, max_hops}, ...]}
    with len(edges) == len(nodes) - 1.  Supports N-node chains
    ``(a)-[r]->(b)->(c)`` and per-node labels ``(a:Category)``.
    """
    s = s.strip()
    nodes: list[dict] = []
    edges: list[dict] = []

    pos = 0
    while pos < len(s):
        while pos < len(s) and s[pos] == " ":
            pos += 1
        if pos >= len(s):
            break

        if s[pos] == "(":
            m = _NODE_RE.match(s, pos)
            if not m:
                raise ValueError(f"Bad node at pos {pos}: {s[pos:pos+20]}")
            var, label, props = m.group(1), m.group(2), m.group(3)
            nodes.append({
                "var": var or f"_n{len(nodes)}",
                "label": label,
                "props": _parse_props(props[1:-1]) if props else {},
            })
            pos = m.end()
        else:
            me = _EDGE_RE.match(s, pos)
            if not me:
                raise ValueError(f"Bad edge at pos {pos}: {s[pos:pos+20]}")
            edges.append(_parse_edge_token(me.group(0)))
            pos = me.end()

    if len(nodes) < 2 or len(edges) != len(nodes) - 1:
        raise ValueError(
            f"Pattern must alternate N nodes and N-1 edges: {s!r}"
        )

    return {"nodes": nodes, "edges": edges}


def _subst_label_calls(s: str | None, label_field: str) -> str | None:
    """Rewrite label(x) sugar to x.<label_field> for WHERE/RETURN parsing."""
    if not s:
        return s
    return _LABEL_CALL_RE.sub(rf"\1.{label_field}", s)


def _node_matches(node: dict, attrs: dict, label_field) -> bool:
    """True if a node's attrs satisfy the pattern node's label + inline props."""
    if node["label"]:
        v = attrs.get(label_field)
        if hasattr(v, "item"):
            v = v.item()
        if str(v) != node["label"]:
            return False
    for k, val in node["props"].items():
        av = attrs.get(k)
        if hasattr(av, "item"):
            av = av.item()
        if av != val:
            return False
    return True


def _first_node_seeds(graph, node0: dict, where_str, label_field) -> list | None:
    """Seed IDs for the chain's first node, or None for a full scan.

    Priority: an explicit ``id(var)==``/``IN`` in WHERE (most selective),
    then the node label via the in-memory label index.
    """
    ids = _extract_src_seeds(where_str, node0["var"])
    if ids is not None:
        return ids
    if node0["label"] and label_field:
        return graph.nodes_with_label(node0["label"])
    return None


# ---------------------------------------------------------------------------
# WHERE parser
# ---------------------------------------------------------------------------

def parse_where(where_str: str | None):
    """Return a function (bindings, node_ids) → bool.

    bindings: {var: attr_dict}
    node_ids: {var: str_id}
    """
    if not where_str or not where_str.strip():
        return lambda b, ids: True

    or_groups = re.split(r"\bOR\b", where_str, flags=re.IGNORECASE)
    or_funcs = []

    for group in or_groups:
        and_parts = re.split(r"\bAND\b", group, flags=re.IGNORECASE)
        and_funcs = []
        for part in and_parts:
            part = part.strip()
            if not part:
                continue

            # id(var) IN [list]
            m_id_in = re.match(r"id\((\w+)\)\s+IN\s+(\[.*\])\s*$", part, re.IGNORECASE)
            if m_id_in:
                var, lst_str = m_id_in.groups()
                lst = _parse_list_literal(lst_str)
                and_funcs.append(("id_in", var, lst))
                continue

            # id(var) op value
            m_id = re.match(r"id\((\w+)\)\s*(==|!=|>=|<=|>|<)\s*(.+)$", part)
            if m_id:
                var, op_str, val_str = m_id.groups()
                and_funcs.append(("id_op", var, _OPS[op_str], _parse_value(val_str)))
                continue

            # var.field op value
            m_attr = re.match(r"(\w+)\.(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$", part)
            if m_attr:
                var, field, op_str, val_str = m_attr.groups()
                and_funcs.append(("attr_op", var, field, _OPS[op_str], _parse_value(val_str)))
                continue

            raise ValueError(f"Cannot parse WHERE condition: {part!r}")

        or_funcs.append(and_funcs)

    def evaluate(bindings: dict, node_ids: dict) -> bool:
        for and_group in or_funcs:
            all_pass = True
            for cond in and_group:
                kind = cond[0]
                if kind == "id_op":
                    _, var, op_func, val = cond
                    node_id = node_ids.get(var, "")
                    if not op_func(node_id, val):
                        all_pass = False; break
                elif kind == "id_in":
                    _, var, lst = cond
                    if node_ids.get(var) not in lst:
                        all_pass = False; break
                elif kind == "attr_op":
                    _, var, field, op_func, val = cond
                    obj = bindings.get(var, {})
                    fv = obj.get(field) if isinstance(obj, dict) else None
                    if hasattr(fv, "item"):
                        fv = fv.item()
                    if fv is None or not op_func(fv, val):
                        all_pass = False; break
            if all_pass:
                return True
        return False

    return evaluate


# ---------------------------------------------------------------------------
# RETURN / LIMIT parsers
# ---------------------------------------------------------------------------

def parse_return(return_str: str | None, bindings: dict, node_ids: dict) -> dict:
    if not return_str or not return_str.strip():
        result = {}
        for k, v in bindings.items():
            if not k.startswith("_"):
                result[k] = v
        for k, v in node_ids.items():
            if not k.startswith("_"):
                result[f"id({k})"] = v
        return result

    fields = [f.strip() for f in return_str.split(",")]
    result = {}
    for f in fields:
        # id(var)
        m_id = re.match(r"id\((\w+)\)$", f)
        if m_id:
            result[f] = node_ids.get(m_id.group(1))
            continue
        # var.field
        if "." in f:
            var, attr = f.split(".", 1)
            obj = bindings.get(var, {})
            val = obj.get(attr) if isinstance(obj, dict) else None
            if hasattr(val, "item"):
                val = val.item()
            result[f] = val
        else:
            result[f] = bindings.get(f)

    return result


def parse_limit(query_str: str) -> tuple[str, int | None]:
    """Strip LIMIT from query and return (remainder, limit_n)."""
    m = re.search(r"\bLIMIT\s+(\d+)\s*$", query_str, re.IGNORECASE)
    if m:
        return query_str[:m.start()].strip(), int(m.group(1))
    return query_str, None


# ---------------------------------------------------------------------------
# Seed extraction — skip full-scan when WHERE pins the starting node
# ---------------------------------------------------------------------------

def _extract_src_seeds(where_str: str | None, src_var: str) -> list[str] | None:
    """Return explicit source node IDs from WHERE, or None for full scan.

    Detects:
        WHERE id(src_var) == "X"        → ["X"]
        WHERE id(src_var) IN ["X","Y"]  → ["X","Y"]
    Other conditions require a full scan and return None.
    """
    if not where_str:
        return None

    # An OR means the seeds of one branch don't constrain the others;
    # grabbing the first id() match would silently drop the other branches.
    # Fall back to a full scan (the WHERE function still filters correctly).
    if re.search(r"\bOR\b", where_str, re.IGNORECASE):
        return None

    # id(var) == "value" — single node
    m = re.search(
        rf'id\({re.escape(src_var)}\)\s*==\s*["\']([^"\']+)["\']',
        where_str, re.IGNORECASE
    )
    if m:
        return [m.group(1)]

    # id(var) IN ["a","b",...]
    m2 = re.search(
        rf'id\({re.escape(src_var)}\)\s+IN\s+(\[[^\]]+\])',
        where_str, re.IGNORECASE
    )
    if m2:
        return [_parse_value(x) for x in
                re.findall(r'["\']([^"\']+)["\']', m2.group(1))]

    return None   # no seed → full scan


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _get_edge(graph, src_id: str, dst_id: str, direction: str) -> dict | None:
    """Get edge attrs between two nodes (or None)."""
    try:
        if direction in ("out", "both"):
            if src_id in graph._out:
                od = graph._out[src_id]
                if dst_id in od:
                    return od[dst_id]
        if direction == "in":
            if dst_id in graph._in:
                id_ = graph._in[dst_id]
                if src_id in id_:
                    return id_[src_id]
    except Exception:
        pass
    return None


def _iter_neighbors(graph, node_id: str, direction: str):
    """Yield (neighbor_id, edge_attrs) for the given direction."""
    if direction == "out":
        if node_id in graph._out:
            yield from graph._out[node_id].items()
    elif direction == "in":
        if node_id in graph._in:
            yield from graph._in[node_id].items()
    else:  # both
        seen = set()
        if node_id in graph._out:
            for nid, attrs in graph._out[node_id].items():
                seen.add(nid)
                yield nid, attrs
        if node_id in graph._in:
            for nid, attrs in graph._in[node_id].items():
                if nid not in seen:
                    yield nid, attrs


# ---------------------------------------------------------------------------
# Query executor
# ---------------------------------------------------------------------------

def _node_iter(graph, seeds: list[str] | None):
    """Yield (node_id, node_attrs) — seeded or full bulk scan."""
    nodes = graph._nodes
    if seeds is not None:
        # Fast path: direct lookups for specific IDs
        for nid in seeds:
            try:
                yield nid, nodes[nid]
            except KeyError:
                pass
    else:
        # Full scan: bulk-load all node attrs in one mmap read
        yield from nodes.to_dict().items()


def _execute_1hop(graph, pat: dict, where_func, return_str: str | None,
                  inline_where, src_seeds: list[str] | None = None) -> Iterator[dict]:
    """Fast path for single-hop (min=max=1) patterns.

    If src_seeds is provided (from seed extraction), only those nodes
    are iterated instead of the full node set.
    """
    src_var = pat["src_var"]
    dst_var = pat["dst_var"]
    edge_var = pat["edge_var"]
    direction = pat["direction"]
    inline_src = (inline_where or {}).get(src_var, {})
    inline_dst = (inline_where or {}).get(dst_var, {})

    nodes = graph._nodes

    # If no seeds and no inline src props: bulk-load all node attrs once
    # to avoid N separate Dict lookups during the scan.
    if src_seeds is None and not inline_src:
        all_nodes = nodes.to_dict()
    else:
        all_nodes = None

    for src_id, src_data in _node_iter(graph, src_seeds):
        # Apply inline src props
        if inline_src and not all(src_data.get(k) == v for k, v in inline_src.items()):
            continue

        if direction == "out" and src_id not in graph._out:
            continue
        if direction == "in" and src_id not in graph._in:
            continue

        for dst_id, edge_data in _iter_neighbors(graph, src_id, direction):
            dst_data = (all_nodes.get(dst_id) if all_nodes is not None
                        else nodes.get(dst_id, {})) or {}

            # Apply inline dst props
            if inline_dst and not all(dst_data.get(k) == v for k, v in inline_dst.items()):
                continue

            bindings = {src_var: src_data, dst_var: dst_data}
            node_ids = {src_var: src_id, dst_var: dst_id}
            if edge_var:
                bindings[edge_var] = edge_data

            if where_func(bindings, node_ids):
                yield parse_return(return_str, bindings, node_ids)


def _execute_nhop(graph, pat: dict, where_func, return_str: str | None,
                  inline_where, src_seeds: list[str] | None = None) -> Iterator[dict]:
    """BFS for variable-length paths (min_hops..max_hops)."""
    src_var = pat["src_var"]
    dst_var = pat["dst_var"]
    direction = pat["direction"]
    min_hops = pat["min_hops"]
    max_hops = pat["max_hops"]

    # Safety cap: if max_hops is ∞, limit to something reasonable
    effective_max = int(min(max_hops, 20)) if math.isinf(max_hops) else int(max_hops)

    nodes = graph._nodes
    inline_src = (inline_where or {}).get(src_var, {})
    inline_dst = (inline_where or {}).get(dst_var, {})

    for src_id, src_data in _node_iter(graph, src_seeds):
        if inline_src and not all(src_data.get(k) == v for k, v in inline_src.items()):
            continue

        # BFS state: (current_node_id, depth)
        # visited per starting node to avoid cycles
        queue = deque([(src_id, 0)])
        visited = {src_id}
        reported = set()

        while queue:
            curr_id, depth = queue.popleft()

            if depth >= min_hops:
                curr_data = nodes.get(curr_id, {})
                dst_id = curr_id

                if inline_dst and not all(curr_data.get(k) == v for k, v in inline_dst.items()):
                    pass  # skip reporting but continue BFS
                elif dst_id not in reported:
                    bindings = {src_var: src_data, dst_var: curr_data}
                    node_ids = {src_var: src_id, dst_var: dst_id}
                    if where_func(bindings, node_ids):
                        reported.add(dst_id)
                        yield parse_return(return_str, bindings, node_ids)

            if depth < effective_max:
                for next_id, _ in _iter_neighbors(graph, curr_id, direction):
                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, depth + 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(graph, query_str: str) -> Iterator[dict]:
    """Execute a Cypher-like query on a loom Graph.

    Args:
        graph: Graph instance
        query_str: Cypher-like query string

    Yields:
        Result dicts (one per matching pattern)
    """
    graph._ensure_loaded()

    # Strip LIMIT
    query_str, limit = parse_limit(query_str.strip())

    # Parse MATCH / WHERE / RETURN
    m = re.match(
        r"MATCH\s+(.+?)(?:\s+WHERE\s+(.+?))?(?:\s+RETURN\s+(.+))?$",
        query_str, re.IGNORECASE
    )
    if not m:
        raise ValueError(f"Cannot parse query: {query_str!r}")

    pattern_str, where_str, return_str = m.groups()

    label_field = getattr(graph, "_label_field", None)
    # label(x) sugar → x.<label_field>
    if label_field:
        where_str = _subst_label_calls(where_str, label_field)
        return_str = _subst_label_calls(return_str, label_field)

    chain = parse_pattern(pattern_str)
    nodes, edges = chain["nodes"], chain["edges"]

    for nd in nodes:
        if nd["label"] and not label_field:
            raise ValueError(
                f"Pattern uses label '({nd['var']}:{nd['label']})' but the graph "
                f"was created without label_field=... (pass it to create_graph)."
            )

    where_func = parse_where(where_str)

    if len(nodes) == 2:
        # Two-node patterns keep the dedicated 1-hop / variable-length paths
        # (those support the [*], [+], [?] quantifiers).
        n0, n1 = nodes
        e = edges[0]
        src_props = dict(n0["props"])
        dst_props = dict(n1["props"])
        if n0["label"]:
            src_props[label_field] = n0["label"]
        if n1["label"]:
            dst_props[label_field] = n1["label"]
        pat = {
            "src_var": n0["var"], "src_props": src_props,
            "edge_var": e["var"], "direction": e["direction"],
            "min_hops": e["min_hops"], "max_hops": e["max_hops"],
            "dst_var": n1["var"], "dst_props": dst_props,
        }
        inline_where = {}
        if src_props:
            inline_where[n0["var"]] = src_props
        if dst_props:
            inline_where[n1["var"]] = dst_props
        src_seeds = _first_node_seeds(graph, n0, where_str, label_field)

        is_1hop = (e["min_hops"] == 1 and e["max_hops"] == 1)
        is_optional = (e["min_hops"] == 0 and e["max_hops"] == 1)
        if is_1hop:
            gen = _execute_1hop(graph, pat, where_func, return_str, inline_where,
                                src_seeds=src_seeds)
        elif is_optional:
            pat0 = {**pat, "min_hops": 0, "max_hops": 0}
            pat1 = {**pat, "min_hops": 1, "max_hops": 1}
            gen = (
                r for g in (
                    _execute_nhop(graph, pat0, where_func, return_str, inline_where,
                                  src_seeds=src_seeds),
                    _execute_1hop(graph, pat1, where_func, return_str, inline_where,
                                  src_seeds=src_seeds),
                ) for r in g
            )
        else:
            gen = _execute_nhop(graph, pat, where_func, return_str, inline_where,
                                src_seeds=src_seeds)
    else:
        # Multi-node chain: only fixed single-hop edges are supported.
        for e in edges:
            if not (e["min_hops"] == 1 and e["max_hops"] == 1):
                raise ValueError(
                    "Variable-length edges ([*], [+], [*a..b], [?]) are only "
                    "supported in two-node patterns, not multi-hop chains."
                )
        seeds = _first_node_seeds(graph, nodes[0], where_str, label_field)
        gen = _execute_chain(graph, nodes, edges, where_func, return_str,
                             seeds, label_field)

    count = 0
    for result in gen:
        yield result
        count += 1
        if limit is not None and count >= limit:
            return


def _execute_chain(graph, nodes, edges, where_func, return_str, seeds,
                   label_field) -> Iterator[dict]:
    """Execute a fixed single-hop chain (a)->(b)->(c)->...  of any length.

    Expands a frontier hop by hop, binding each node/edge variable and
    filtering on per-node label + inline props.  WHERE is applied once a
    full binding exists.  When ``seeds`` is given (id- or label-anchored),
    node attrs are resolved with per-node lookups; otherwise the whole node
    set is bulk-loaded once for the opening full scan.
    """
    node_store = graph._nodes
    all_nodes = node_store.to_dict() if seeds is None else None

    def attrs_of(nid):
        if all_nodes is not None:
            return all_nodes.get(nid, {}) or {}
        return node_store.get(nid, {}) or {}

    n0 = nodes[0]
    if seeds is not None:
        seed_iter = ((nid, attrs_of(nid)) for nid in seeds)
    else:
        seed_iter = all_nodes.items()

    # frontier entries: (current_node_id, bindings, node_ids)
    frontier = []
    for nid, attrs in seed_iter:
        if _node_matches(n0, attrs, label_field):
            frontier.append((nid, {n0["var"]: attrs}, {n0["var"]: nid}))

    for ei, edge in enumerate(edges):
        nxt = nodes[ei + 1]
        new_frontier = []
        for cur_id, bindings, node_ids in frontier:
            for nb_id, edge_attrs in _iter_neighbors(graph, cur_id, edge["direction"]):
                nb_attrs = attrs_of(nb_id)
                if not _node_matches(nxt, nb_attrs, label_field):
                    continue
                b2 = dict(bindings)
                i2 = dict(node_ids)
                b2[nxt["var"]] = nb_attrs
                i2[nxt["var"]] = nb_id
                if edge["var"]:
                    b2[edge["var"]] = edge_attrs
                new_frontier.append((nb_id, b2, i2))
        frontier = new_frontier
        if not frontier:
            return

    for _cur_id, bindings, node_ids in frontier:
        if where_func(bindings, node_ids):
            yield parse_return(return_str, bindings, node_ids)
