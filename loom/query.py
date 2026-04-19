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


def parse_pattern(s: str) -> dict:
    """
    Returns {
        src_var, src_props,
        edge_var, direction, min_hops, max_hops,
        dst_var, dst_props
    }
    """
    s = s.strip()

    # Attempt node-edge-node decomposition
    # Find all node parts (with or without inline props)
    nodes = []
    edges_raw = []

    pos = 0
    while pos < len(s):
        # Skip spaces
        while pos < len(s) and s[pos] == " ":
            pos += 1
        if pos >= len(s):
            break

        if s[pos] == "(":
            # Try props first, then bare
            mp = _NODE_PROPS_RE.match(s, pos)
            mb = _NODE_BARE_RE.match(s, pos)
            m = mp if mp else mb
            if not m:
                raise ValueError(f"Bad node at pos {pos}: {s[pos:pos+20]}")
            props = _parse_props(m.group(2)) if mp else {}
            nodes.append((m.group(1), props))
            pos = m.end()
        else:
            # Edge
            me = _EDGE_RE.match(s, pos)
            if not me:
                raise ValueError(f"Bad edge at pos {pos}: {s[pos:pos+20]}")
            edges_raw.append(me.group(0))
            pos = me.end()

    if len(nodes) < 2 or len(edges_raw) < 1:
        raise ValueError(f"Pattern must have at least 2 nodes and 1 edge: {s}")

    # Only 1-hop patterns supported for now (2 nodes, 1 edge)
    (src_var, src_props), (dst_var, dst_props) = nodes[0], nodes[1]
    edge_txt = edges_raw[0]

    # Determine direction and quantifier
    if edge_txt.startswith("<-"):
        direction = "in"
        m = re.match(r"<-\[(\w*)\s*([*+?]?[0-9.]*(?:\.\.[0-9]+)?)?\s*\]-", edge_txt)
        edge_var = m.group(1) if m else ""
        quant_str = m.group(2) if (m and m.group(2)) else None
    elif "->" in edge_txt:
        direction = "out"
        m = re.match(r"-\[(\w*)\s*([*+?][0-9.]*(?:\.\.[0-9]+)?)?\s*\]->", edge_txt)
        edge_var = m.group(1) if m else ""
        quant_str = m.group(2) if (m and m.group(2)) else None
    else:
        direction = "both"
        m = re.match(r"-\[(\w*)\s*([*+?][0-9.]*(?:\.\.[0-9]+)?)?\s*\]-", edge_txt)
        edge_var = m.group(1) if m else ""
        quant_str = m.group(2) if (m and m.group(2)) else None

    if not edge_var and not quant_str:
        # shorthand: -> or <- or --
        quant_str = None

    min_hops, max_hops = _parse_quantifier(quant_str)

    return {
        "src_var": src_var or "_src",
        "src_props": src_props,
        "edge_var": edge_var or None,
        "direction": direction,
        "min_hops": min_hops,
        "max_hops": max_hops,
        "dst_var": dst_var or "_dst",
        "dst_props": dst_props,
    }


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

def _execute_1hop(graph, pat: dict, where_func, return_str: str | None,
                  inline_where) -> Iterator[dict]:
    """Fast path for single-hop (min=max=1) patterns."""
    src_var = pat["src_var"]
    dst_var = pat["dst_var"]
    edge_var = pat["edge_var"]
    direction = pat["direction"]

    nodes = graph._nodes

    for src_id in list(nodes.keys()):
        src_data = nodes[src_id]

        # Apply inline src props
        if inline_where and not all(src_data.get(k) == v for k, v in inline_where.get(src_var, {}).items()):
            continue

        if src_id not in graph._out and direction in ("out", "both"):
            if direction == "out":
                continue
        if src_id not in graph._in and direction == "in":
            continue

        for dst_id, edge_data in _iter_neighbors(graph, src_id, direction):
            dst_data = nodes.get(dst_id, {})

            # Apply inline dst props
            if inline_where and not all(dst_data.get(k) == v for k, v in inline_where.get(dst_var, {}).items()):
                continue

            bindings = {src_var: src_data, dst_var: dst_data}
            node_ids = {src_var: src_id, dst_var: dst_id}
            if edge_var:
                bindings[edge_var] = edge_data

            if where_func(bindings, node_ids):
                yield parse_return(return_str, bindings, node_ids)


def _execute_nhop(graph, pat: dict, where_func, return_str: str | None,
                  inline_where) -> Iterator[dict]:
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

    for src_id in list(nodes.keys()):
        src_data = nodes[src_id]

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

    pat = parse_pattern(pattern_str)

    # Inline props from pattern become implicit WHERE conditions
    # We pass them separately so they're evaluated early (no attr dict lookup needed)
    inline_where = {}
    if pat["src_props"]:
        inline_where[pat["src_var"]] = pat["src_props"]
    if pat["dst_props"]:
        inline_where[pat["dst_var"]] = pat["dst_props"]

    where_func = parse_where(where_str)

    is_1hop = (pat["min_hops"] == 1 and pat["max_hops"] == 1)
    is_optional = (pat["min_hops"] == 0 and pat["max_hops"] == 1)

    if is_1hop:
        gen = _execute_1hop(graph, pat, where_func, return_str, inline_where)
    elif is_optional:
        # 0..1 hop: include both self-match and 1-hop matches
        pat0 = {**pat, "min_hops": 0, "max_hops": 0}
        pat1 = {**pat, "min_hops": 1, "max_hops": 1}
        gen = (
            r for g in (
                _execute_nhop(graph, pat0, where_func, return_str, inline_where),
                _execute_1hop(graph, pat1, where_func, return_str, inline_where),
            ) for r in g
        )
    else:
        gen = _execute_nhop(graph, pat, where_func, return_str, inline_where)

    count = 0
    for result in gen:
        yield result
        count += 1
        if limit is not None and count >= limit:
            return
