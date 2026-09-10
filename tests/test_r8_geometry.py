from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest
from r8_fakes import variable

from qwen3vl_agent.r8.config import R8Config
from qwen3vl_agent.r8.evidence import EvidenceStore
from qwen3vl_agent.r8.geometry import op, ref
from qwen3vl_agent.r8.solver import solve_constraints
from qwen3vl_agent.r8.types import InputContract


def geometry_store():
    contract = InputContract(((0.0, 10.0),), ("video", "screen_text"), "full", "caller", None)
    s = EvidenceStore("synthetic teaching rectangle", contract)
    packet = {
        "F01": {
            "id": "diagram",
            "modality": "video",
            "timestamp_seconds": 1.0,
            "scope_hash": contract.fingerprint,
        }
    }
    s.present(packet)
    objects = {k: k + "_region" for k in ("tl", "tr", "bl", "br", "total")}
    for entity in objects.values():
        s.state["entities"][entity] = {
            "id": entity,
            "description": entity,
            "scope": "board",
            "snapshot": "s1",
        }
    for key, val in (("tl", "12"), ("tr", "18"), ("bl", "8")):
        row = variable(key, val, "m^2")
        row.update(
            entity_id=objects[key],
            attribute="area",
            role="measurement",
            scope="board",
            snapshot="s1",
        )
        s.append(row, packet=packet)
    s.state["relations"]["grid"] = {
        "id": "grid",
        "kind": "rectangle_partition",
        "objects": objects,
        "raw_text": "Complete 2x2 rectangular partition sharing row heights and column widths",
        "source_kind": "structure",
        "evidence_refs": ["diagram"],
        "question_span": "",
        "scope": "board",
        "snapshot": "s1",
    }
    s.state["relations"]["positive"] = {
        "id": "positive",
        "kind": "nondegenerate",
        "objects": objects,
        "raw_text": "nondegenerate rectangles",
        "source_kind": "structure",
        "evidence_refs": ["diagram"],
        "question_span": "",
        "scope": "board",
        "snapshot": "s1",
    }
    symbols = [
        {
            "id": k,
            "entity_id": entity,
            "attribute": "area",
            "unit": "m^2",
            "domain": "real",
            "sources": ["grid"],
            "snapshot": "s1",
        }
        for k, entity in objects.items()
    ]
    p = {
        "symbols": symbols,
        "constraints": [],
        "rules": [
            {
                "id": "partition",
                "rule": "rectangle_partition",
                "premises": ["grid", "positive"],
                "mapping": {k: k for k in objects},
                "snapshot": "s1",
            }
        ],
        "target": ref("br"),
        "output_unit": "m^2",
    }
    return s, p


def test_teaching_rectangle_z_and_total():
    s, p = geometry_store()
    result = solve_constraints(p, s, R8Config())
    assert result["status"] == "solved_target", result
    assert Fraction(result["value"]["exact"]) == 12
    p["target"] = ref("total")
    result = solve_constraints(p, s, R8Config())
    assert Fraction(result["value"]["exact"]) == 50


def test_edges_may_remain_underdetermined():
    s, p = geometry_store()
    for k in ("width", "height"):
        s.state["entities"][k] = {"id": k, "description": k, "scope": "board", "snapshot": "s1"}
    relation = deepcopy(s.state["relations"]["grid"])
    relation.update(
        id="rect",
        kind="rectangle",
        source_kind="explicit_marker",
        objects={"width": "width", "height": "height", "area": "tl_region"},
    )
    s.state["relations"]["rect"] = relation
    for k in ("width", "height"):
        p["symbols"].append(
            {
                "id": k,
                "entity_id": k,
                "attribute": "length",
                "unit": "m",
                "domain": "real",
                "sources": ["rect"],
                "snapshot": "s1",
            }
        )
    p["rules"].append(
        {
            "id": "rect",
            "rule": "rectangle",
            "premises": ["rect", "positive"],
            "mapping": {"width": "width", "height": "height", "area": "tl"},
            "snapshot": "s1",
        }
    )
    assert solve_constraints(p, s, R8Config())["status"] == "solved_target"
    p.update(target=ref("width"), output_unit="m")
    assert solve_constraints(p, s, R8Config())["status"] == "ambiguous_target"


def test_conflict_tracks_sources():
    s, p = geometry_store()
    p["constraints"].append(
        {
            "id": "wrong",
            "relation": "equal",
            "args": [ref("br"), ref("tr")],
            "sources": ["grid"],
            "snapshot": "s1",
        }
    )
    result = solve_constraints(p, s, R8Config())
    assert result["status"] == "inconsistent_constraints"
    assert "wrong" in result["conflict_core"]
    assert len(p["constraints"]) == 1


@pytest.mark.parametrize(
    "mutation",
    ["snapshot", "missing_premise", "mapping", "unit", "string_ast", "domain", "unknown_constant"],
)
def test_reject_invalid_geometry(mutation):
    s, p = geometry_store()
    if mutation == "snapshot":
        p["symbols"][0]["snapshot"] = "s2"
    elif mutation == "missing_premise":
        p["rules"][0]["premises"] = ["grid"]
    elif mutation == "mapping":
        p["rules"][0]["mapping"]["tl"] = "br"
    elif mutation == "unit":
        p["symbols"][0]["unit"] = "g"
    elif mutation == "string_ast":
        p["target"] = "__import__('os')"
    elif mutation == "domain":
        p["symbols"][-1]["domain"] = "integer"
    else:
        p["target"] = {"constant": "answer50"}
    assert solve_constraints(p, s, R8Config())["status"] == "unsupported_theory"


def test_hard_timeout_is_unknown():
    s, p = geometry_store()
    result = solve_constraints(p, s, replace(R8Config(), solver_seconds=0.001))
    assert result["status"] == "timeout_or_unknown"
    assert "deadline" in result["reason"]


def test_pythagoras_requires_real_right_angle():
    s, p = geometry_store()
    objects = {k: k for k in ("a", "b", "c")}
    for k in objects:
        s.state["entities"][k] = {"id": k, "scope": "board", "snapshot": "s1"}
    r = deepcopy(s.state["relations"]["grid"])
    r.update(id="tri", kind="triangle", objects=objects, source_kind="structure")
    s.state["relations"]["tri"] = r
    r2 = deepcopy(s.state["relations"]["positive"])
    r2["objects"] = objects
    s.state["relations"]["positive"] = r2
    p = {
        "symbols": [
            {
                "id": k,
                "entity_id": k,
                "attribute": "length",
                "unit": "m",
                "domain": "real",
                "sources": ["tri"],
                "snapshot": "s1",
            }
            for k in objects
        ],
        "constraints": [],
        "rules": [
            {
                "id": "p",
                "rule": "pythagoras",
                "premises": ["tri", "positive"],
                "mapping": objects,
                "snapshot": "s1",
            }
        ],
        "target": ref("c"),
        "output_unit": "m",
    }
    assert solve_constraints(p, s, R8Config())["status"] == "unsupported_theory"
    s.state["relations"]["tri"]["kind"] = "right_triangle"
    assert solve_constraints(p, s, R8Config())["status"] == "unsupported_theory"


def test_exact_algebraic_target_with_error_bound():
    s, p = geometry_store()
    p["symbols"] = [
        {
            "id": "br",
            "entity_id": "br_region",
            "attribute": "area",
            "unit": "1",
            "domain": "positive",
            "sources": ["grid", "positive"],
            "snapshot": "s1",
        }
    ]
    p["rules"] = []
    p["constraints"] = [
        {
            "id": "sqrt2",
            "relation": "equal",
            "args": [op("square", ref("br")), {"constant": "two"}],
            "sources": ["grid"],
            "snapshot": "s1",
        }
    ]
    p["output_unit"] = "1"
    result = solve_constraints(p, s, R8Config())
    assert result["status"] == "solved_target", result
    assert result["value"]["kind"] == "algebraic"
    lower, upper = Fraction(result["value"]["lower"]), Fraction(result["value"]["upper"])
    assert lower * lower < 2 < upper * upper
    assert upper - lower <= Fraction(1, 10**30)


@pytest.mark.parametrize(
    "rule,kind,known,units,target,want",
    [
        (
            "rectangle",
            "rectangle",
            {"width": "3", "height": "4"},
            {"width": "m", "height": "m", "area": "m^2"},
            "area",
            12,
        ),
        ("square", "square", {"side": "3"}, {"side": "m", "area": "m^2"}, "area", 9),
        (
            "segment_addition",
            "on_segment",
            {"left": "2", "right": "3"},
            {"left": "m", "right": "m", "whole": "m"},
            "whole",
            5,
        ),
        (
            "triangle_angles",
            "triangle",
            {"a": "30", "b": "60"},
            {"a": "degree", "b": "degree", "c": "degree"},
            "c",
            90,
        ),
        (
            "shared_base",
            "shared_base",
            {"area1": "12", "height1": "3", "height2": "2"},
            {"area1": "m^2", "area2": "m^2", "height1": "m", "height2": "m"},
            "area2",
            8,
        ),
        (
            "shared_height",
            "shared_height",
            {"area1": "12", "base1": "3", "base2": "2"},
            {"area1": "m^2", "area2": "m^2", "base1": "m", "base2": "m"},
            "area2",
            8,
        ),
        (
            "pythagoras",
            "right_triangle",
            {"a": "3", "b": "4"},
            {"a": "m", "b": "m", "c": "m"},
            "c",
            5,
        ),
        (
            "similar_triangles",
            "similar_triangles",
            {"a1": "2", "b1": "3", "c1": "4", "a2": "4", "c2": "8"},
            {k: "m" for k in ("a1", "b1", "c1", "a2", "b2", "c2")},
            "b2",
            6,
        ),
    ],
)
def test_all_rule_families(rule, kind, known, units, target, want):
    s, _ = geometry_store()
    packet = {"F01": s.state["evidence"]["diagram"]}
    objects = {key: key + "_object" for key in units}
    symbols = []
    for key, unit in units.items():
        entity = objects[key]
        s.state["entities"][entity] = {
            "id": entity,
            "description": entity,
            "scope": "board",
            "snapshot": "s1",
        }
        attr = "area" if unit == "m^2" else "angle" if unit == "degree" else "length"
        if key in known:
            row = variable(key, known[key], unit)
            row.update(
                entity_id=entity, attribute=attr, role="measurement", scope="board", snapshot="s1"
            )
            s.append(row, packet=packet)
        symbols.append(
            {
                "id": key,
                "entity_id": entity,
                "attribute": attr,
                "unit": unit,
                "domain": "real",
                "sources": ["shape"],
                "snapshot": "s1",
            }
        )
    source = {
        "id": "shape",
        "kind": kind,
        "objects": objects,
        "raw_text": "synthetic marked geometry relation",
        "source_kind": "explicit_marker",
        "evidence_refs": ["diagram"],
        "question_span": "",
        "scope": "board",
        "snapshot": "s1",
    }
    s.state["relations"]["shape"] = source
    s.state["relations"]["nondegenerate"] = {
        **source,
        "id": "nondegenerate",
        "kind": "nondegenerate",
    }
    program = {
        "symbols": symbols,
        "constraints": [],
        "rules": [
            {
                "id": "apply",
                "rule": rule,
                "premises": ["shape", "nondegenerate"],
                "mapping": {k: k for k in units},
                "snapshot": "s1",
            }
        ],
        "target": ref(target),
        "output_unit": units[target],
    }
    result = solve_constraints(program, s, R8Config())
    assert result["status"] == "solved_target", result
    assert Fraction(result["value"]["exact"]) == want
