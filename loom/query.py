"""
Minimal Cypher-like query engine for loom Graph.

Supports a small but useful subset of Cypher:

    MATCH (a)-[r]->(b) WHERE a.age > 25 AND r.weight > 0.5 RETURN a.name, b.name
    MATCH (a)->(b) WHERE a.name == "Alice" RETURN b
    MATCH (a)<-[r]-(b) RETURN a, b, r

Pattern syntax:
    (var)             node variable
    -[var]->          directed edge (left to right)
    <-[var]-          directed edge (right to left)
    -[var]-           undirected edge
    ->                shorthand for -[]->
    <-                shorthand for <-[]-

WHERE clause:
    var.field == value
    var.field != value
    var.field > value
    var.field >= value
    var.field < value
    var.field <= value
    AND / OR for combining (AND binds tighter)
    "string" or 'string' for string literals, numbers parsed automatically

RETURN clause:
    var.field, var.field, ...
    var                 returns the full dict
    Omit RETURN to get all variables as dicts
"""

import re
import operator

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_PATTERN_RE = re.compile(
    r"\((\w+)\)"           # (var)
    r"|<-\[(\w*)\]-"       # <-[var]-
    r"|-\[(\w*)\]->"       # -[var]->
    r"|-\[(\w*)\]-"        # -[var]-
    r"|(->>?)"             # -> shorthand
    r"|(<-)"               # <- shorthand
)

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def _parse_value(s):
    """Parse a literal value from a WHERE clause."""
    s = s.strip()
    # String literal
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Bool
    if s == "True" or s == "true":
        return True
    if s == "False" or s == "false":
        return False
    # None
    if s == "None" or s == "null":
        return None
    # Number
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Pattern parser
# ---------------------------------------------------------------------------

def _parse_pattern(pattern_str):
    """Parse a pattern like (a)-[r]->(b) into structured form.

    Returns list of:
        ("node", var_name)
        ("edge", var_name, direction)   direction: "out", "in", "both"
    """
    tokens = []
    pos = 0
    s = pattern_str.strip()

    while pos < len(s):
        # Skip whitespace
        if s[pos] == " ":
            pos += 1
            continue

        m = _PATTERN_RE.match(s, pos)
        if not m:
            raise ValueError(f"Cannot parse pattern at position {pos}: ...{s[pos:pos+20]}")

        if m.group(1) is not None:
            tokens.append(("node", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("edge", m.group(2) or None, "in"))
        elif m.group(3) is not None:
            tokens.append(("edge", m.group(3) or None, "out"))
        elif m.group(4) is not None:
            tokens.append(("edge", m.group(4) or None, "both"))
        elif m.group(5) is not None:
            tokens.append(("edge", None, "out"))
        elif m.group(6) is not None:
            tokens.append(("edge", None, "in"))

        pos = m.end()

    return tokens


# ---------------------------------------------------------------------------
# WHERE parser
# ---------------------------------------------------------------------------

def _parse_where(where_str):
    """Parse WHERE clause into a list of condition functions.

    Returns a function (bindings) -> bool.
    """
    if not where_str or not where_str.strip():
        return lambda bindings: True

    # Split by OR first, then AND within each OR group
    or_groups = re.split(r"\bOR\b", where_str, flags=re.IGNORECASE)

    or_funcs = []
    for group in or_groups:
        and_parts = re.split(r"\bAND\b", group, flags=re.IGNORECASE)
        and_funcs = []
        for part in and_parts:
            part = part.strip()
            if not part:
                continue
            # Parse: var.field OP value
            m = re.match(r"(\w+)\.(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$", part)
            if not m:
                raise ValueError(f"Cannot parse WHERE condition: {part}")
            var, field, op_str, val_str = m.groups()
            op_func = _OPS[op_str]
            val = _parse_value(val_str)
            and_funcs.append((var, field, op_func, val))

        or_funcs.append(and_funcs)

    def evaluate(bindings):
        for and_group in or_funcs:
            all_pass = True
            for var, field, op_func, val in and_group:
                obj = bindings.get(var)
                if obj is None:
                    all_pass = False
                    break
                field_val = obj.get(field) if isinstance(obj, dict) else None
                # numpy scalar → python scalar for comparison
                if hasattr(field_val, "item"):
                    field_val = field_val.item()
                if not op_func(field_val, val):
                    all_pass = False
                    break
            if all_pass:
                return True
        return False

    return evaluate


# ---------------------------------------------------------------------------
# RETURN parser
# ---------------------------------------------------------------------------

def _parse_return(return_str, bindings):
    """Project bindings according to RETURN clause."""
    if not return_str or not return_str.strip():
        return bindings

    fields = [f.strip() for f in return_str.split(",")]
    result = {}
    for f in fields:
        if "." in f:
            var, attr = f.split(".", 1)
            obj = bindings.get(var)
            val = obj.get(attr) if isinstance(obj, dict) else None
            if hasattr(val, "item"):
                val = val.item()
            result[f] = val
        else:
            result[f] = bindings.get(f)
    return result


# ---------------------------------------------------------------------------
# Query executor
# ---------------------------------------------------------------------------

def execute_query(graph, query_str):
    """Execute a Cypher-like query on a loom Graph.

    Args:
        graph: Graph instance
        query_str: Cypher-like query string

    Yields:
        Result dicts (one per matching pattern)

    Examples:
        # All edges from nodes older than 25
        execute_query(g, "MATCH (a)-[r]->(b) WHERE a.age > 25 RETURN a.name, b.name")

        # All neighbors of Alice
        execute_query(g, "MATCH (a)->(b) WHERE a.name == 'Alice' RETURN b.name")
    """
    graph._ensure_loaded()

    # Parse query parts
    q = query_str.strip()

    # Extract MATCH, WHERE, RETURN
    m = re.match(
        r"MATCH\s+(.+?)(?:\s+WHERE\s+(.+?))?(?:\s+RETURN\s+(.+))?$",
        q, re.IGNORECASE
    )
    if not m:
        raise ValueError(f"Cannot parse query: {q}")

    pattern_str, where_str, return_str = m.groups()

    # Parse pattern
    tokens = _parse_pattern(pattern_str)
    where_func = _parse_where(where_str)

    # We support 1-hop patterns: (a)-[r]->(b)
    if len(tokens) != 3 or tokens[0][0] != "node" or tokens[1][0] != "edge" or tokens[2][0] != "node":
        raise ValueError("Only 1-hop patterns supported: (a)-[r]->(b)")

    src_var = tokens[0][1]
    edge_var = tokens[1][1]
    direction = tokens[1][2]
    dst_var = tokens[2][1]

    # Iterate all edges matching the pattern
    nodes_dict = graph._nodes
    out_index = graph._out
    in_index = graph._in

    # Choose iteration direction
    if direction == "out":
        # (a)->(b): iterate outgoing
        for src_id in list(nodes_dict.keys()):
            src_data = nodes_dict[src_id]
            if src_id not in out_index:
                continue
            for dst_id, edge_data in out_index[src_id].items():
                dst_data = nodes_dict.get(dst_id)
                bindings = {src_var: src_data, dst_var: dst_data or {}}
                if edge_var:
                    bindings[edge_var] = edge_data
                if where_func(bindings):
                    yield _parse_return(return_str, bindings)

    elif direction == "in":
        # (a)<-(b): iterate incoming
        for dst_id in list(nodes_dict.keys()):
            dst_data = nodes_dict[dst_id]
            if dst_id not in in_index:
                continue
            for src_id, edge_data in in_index[dst_id].items():
                src_data = nodes_dict.get(src_id)
                bindings = {src_var: dst_data, dst_var: src_data or {}}
                if edge_var:
                    bindings[edge_var] = edge_data
                if where_func(bindings):
                    yield _parse_return(return_str, bindings)

    else:  # "both" — undirected
        seen = set()
        for src_id in list(nodes_dict.keys()):
            src_data = nodes_dict[src_id]
            if src_id not in out_index:
                continue
            for dst_id, edge_data in out_index[src_id].items():
                edge_key = tuple(sorted([src_id, dst_id]))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                dst_data = nodes_dict.get(dst_id)
                bindings = {src_var: src_data, dst_var: dst_data or {}}
                if edge_var:
                    bindings[edge_var] = edge_data
                if where_func(bindings):
                    yield _parse_return(return_str, bindings)
