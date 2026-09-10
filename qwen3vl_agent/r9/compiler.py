"""Question-only validation and label-free candidate domains."""

import re

from .types import ProtocolError


def option_domain(choices):
    # Every entry is a candidate/hypothesis, never an observed spatial fact.
    return {"candidate_texts": [c.text for c in choices], "hypotheses_only": True}


def validate_spec(spec, request, contract, config):
    ids = [e["id"] for e in spec["entities"]]
    roles = [e["role"] for e in spec["entities"]]
    if len(set(ids)) != len(ids) or len(set(roles)) != len(roles):
        raise ProtocolError("duplicate entity or role in QuestionSpec")
    for source in spec["source_spans"]:
        if request.question[source["start"] : source["end"]] != source["text"]:
            raise ProtocolError("QuestionSpec source span does not match the question")
    for role in ("origin_role", "forward_role"):
        value = spec["query_frame"][role]
        if value is not None and value not in roles:
            raise ProtocolError("query frame refers to an unknown role")
    for anchor in spec["time"].values():
        if (
            anchor
            and anchor["kind"] == "time"
            and (anchor["time_s"] is None or not contract.permits(anchor["time_s"]))
        ):
            raise ProtocolError("question time anchor outside permitted video")
        if anchor and anchor["kind"] == "event" and not anchor["description"]:
            raise ProtocolError("event anchor requires a description")
    threshold = spec["direction_rule"]["back_threshold_degrees"]
    if (
        threshold is not None
        and str(int(threshold) if float(threshold).is_integer() else threshold)
        not in request.question
    ):
        raise ProtocolError("direction threshold must occur in the original question")
    # Preserve important literals even if a model omits their qualifiers.
    if re.search(r"135\s*(?:degrees|°)", request.question, re.IGNORECASE) and threshold != 135:
        raise ProtocolError("omitted 135-degree back threshold")
    if spec["direction_rule"]["kind"] == "left_right_back" and threshold is None:
        raise ProtocolError("back sector requires a question-defined threshold")
    route = spec["route"]
    if route:
        used = [route[k] for k in ("start_entity", "forward_entity") if route[k]]
        used += route["waypoints"] + [e for p in route["candidate_paths"] for e in p]
        if set(used) - set(ids):
            raise ProtocolError("route references unknown entities")
    nodes, seen = spec["nodes"], set()
    if len(nodes) > config.max_query_nodes:
        raise ProtocolError("query DAG exceeds node budget")
    for node in nodes:
        if node["id"] in seen or set(node["input_nodes"]) - seen:
            raise ProtocolError("query nodes must be unique and in dependency order")
        if set(node["entity_ids"]) - set(ids):
            raise ProtocolError("query node references unknown entity")
        op, params, inputs = node["operation"], node["parameters"], node["input_nodes"]
        if op == "event_select" and not params.get("event_description"):
            raise ProtocolError("event_select requires an event description")
        if op == "unit_convert" and (len(inputs) != 1 or not params.get("output_unit")):
            raise ProtocolError("unit_convert requires one input and an output unit")
        if op in {"subtract", "compare"} and len(inputs) != 2:
            raise ProtocolError("binary numeric query requires two input nodes")
        if op in {"add", "collect"} and not inputs:
            raise ProtocolError("aggregate query requires at least one input node")
        seen.add(node["id"])
    if nodes and spec["output_node"] not in seen:
        raise ProtocolError("query output node missing")
    if not nodes and spec["output_node"] is not None:
        raise ProtocolError("output node without a query DAG")
    if request.output_unit and spec["measurement"]["unit"] != request.output_unit:
        raise ProtocolError("compiled output unit differs from the public protocol")
    return spec


def observation_task(spec):
    """No options, source row IDs, original filename, predictions or answer letters."""
    return {
        k: spec[k]
        for k in ("entities", "time", "query_frame", "measurement", "required_capabilities")
    }
