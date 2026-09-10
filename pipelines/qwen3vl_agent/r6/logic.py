"""Pure three-valued operators. No generated programs and no closed-world default."""

from .types import ProtocolError

STATES = ("supported", "contradicted", "unknown")


def negate(value):
    return {"supported": "contradicted", "contradicted": "supported", "unknown": "unknown"}[value]


def conjunction(values):
    if "contradicted" in values:
        return "contradicted"
    return "supported" if values and all(v == "supported" for v in values) else "unknown"


def expression_atoms(node, *, depth=0):
    if depth > 12 or not isinstance(node, dict):
        raise ProtocolError("invalid/deep proposition expression")
    if node.get("op") == "atom" and set(node) == {"op", "id"}:
        return {node["id"]}
    if node.get("op") not in {"and", "or", "not", "none_of"} or set(node) != {"op", "args"}:
        raise ProtocolError("unsupported proposition operator")
    if not isinstance(node["args"], list) or not node["args"]:
        raise ProtocolError("empty logical expression")
    if node["op"] == "not" and len(node["args"]) != 1:
        raise ProtocolError("not requires one argument")
    return set().union(*(expression_atoms(n, depth=depth + 1) for n in node["args"]))


def evaluate_expression(node, statuses):
    if node["op"] == "atom":
        return statuses.get(node["id"], "unknown")
    values = [evaluate_expression(n, statuses) for n in node["args"]]
    if node["op"] == "not":
        return negate(values[0])
    if node["op"] == "and":
        return conjunction(values)
    disjunction = negate(conjunction([negate(v) for v in values]))
    return negate(disjunction) if node["op"] == "none_of" else disjunction


def exact_set_status(selected, states, *, universe_complete):
    """Compatibility T <= S <= T|U and S <= N only after actual N is enumerated."""
    if not universe_complete:
        return "unknown"
    selected, universe = set(selected), set(states)
    true = {i for i, state in states.items() if state == "supported"}
    unknown = {i for i, state in states.items() if state == "unknown"}
    if not true <= selected <= true | unknown or not selected <= universe:
        return "contradicted"
    return "unknown" if unknown else "supported"
