"""Strict role schemas; all executable values have a predicate-specific schema."""

import json
from copy import deepcopy

import jsonschema

from .types import HELPERS, OPERATIONS, ProtocolError


def obj(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def arr(items, maximum=64, minimum=0):
    return {"type": "array", "items": items, "maxItems": maximum, "minItems": minimum}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


S = {"type": "string", "minLength": 1, "maxLength": 2000}
ID = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.:-]{0,95}$"}
TEXT = {"type": "string", "maxLength": 4000}
N = {"type": "number"}
B = {"type": "boolean"}
SPAN = arr(N, 2, 2)
POINT = arr(N, 3, 2)
BOX = arr({"type": "number", "minimum": 0, "maximum": 1}, 4, 4)
SOURCE_SPAN = obj(
    {
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 1},
        "text": S,
    }
)
ANCHOR = obj(
    {
        "kind": {"enum": ["start", "end", "time", "event"]},
        "time_s": nullable(N),
        "description": TEXT,
        "ordinal": nullable({"type": "integer", "minimum": 1}),
    }
)
ENTITY = obj({"id": ID, "role": S, "description": S})
FRAME = obj(
    {
        "id": ID,
        "kind": {"enum": ["image", "camera", "body", "query", "scene", "inner_camera", "mirror"]},
        "component_id": ID,
        "plane": {"enum": ["horizontal", "vertical", "3d", "image"]},
        "parent_frame_id": nullable(ID),
        "source_observation_ids": arr(ID),
    }
)
QFRAME = obj(
    {
        "kind": {"enum": ["query", "camera", "body", "scene", "inner_camera", "mirror"]},
        "origin_role": nullable(S),
        "forward_role": nullable(S),
        "plane": {"enum": ["horizontal", "vertical", "3d", "image"]},
        "orientation_source": S,
        "heading_kind": {"enum": ["body", "camera", "motion"]},
    }
)
MEASUREMENT = obj(
    {
        "geometry_semantics": {
            "enum": ["closest_boundary", "center", "car_front", "max_extent", "floor_area", "none"]
        },
        "unit": nullable(S),
        "metric_required": B,
    }
)
ROUTE = obj(
    {
        "start_entity": nullable(ID),
        "forward_entity": nullable(ID),
        "waypoints": arr(ID, 32),
        "candidate_paths": arr(arr(ID, 32), 16),
        "stop_condition": TEXT,
        "turn_indices": arr({"type": "integer", "minimum": 0}, 32),
    },
    ["start_entity", "forward_entity", "waypoints", "candidate_paths", "stop_condition"],
)
PARAMS = obj(
    {
        "event_description": S,
        "ordinal": {"type": "integer", "minimum": 1},
        "output_unit": S,
        "comparison": {"enum": ["less", "greater", "equal"]},
    },
    [],
)
NODE = obj(
    {
        "id": ID,
        "operation": {"enum": list(OPERATIONS + HELPERS)},
        "entity_ids": arr(ID, 16),
        "input_nodes": arr(ID, 8),
        "parameters": PARAMS,
    }
)
QUESTION_SPEC = obj(
    {
        "operation": {"enum": list(OPERATIONS)},
        "entities": arr(ENTITY, 24, 1),
        "time": obj({"reference_anchor": nullable(ANCHOR), "query_anchor": nullable(ANCHOR)}),
        "query_frame": QFRAME,
        "direction_rule": obj(
            {
                "kind": {"enum": ["left_right", "left_right_back", "four_quadrants"]},
                "back_threshold_degrees": nullable(
                    {"type": "number", "exclusiveMinimum": 0, "maximum": 180}
                ),
            }
        ),
        "measurement": MEASUREMENT,
        "route": nullable(ROUTE),
        "required_capabilities": arr(S, 16),
        "source_spans": arr(SOURCE_SPAN, 32, 1),
        "nodes": arr(NODE, 16),
        "output_node": nullable(ID),
    }
)
OBSERVATION = obj(
    {
        "id": ID,
        "frame_id": ID,
        "shot_id": ID,
        "entity_candidate": ID,
        "category": nullable(S),
        "visible_attributes": arr(S, 12),
        "facing_cues": arr(S, 8),
        "bbox_xyxy": nullable(BOX),
        "coordinate_space": {"enum": ["presented_image_normalized", "original_image_normalized"]},
        "occlusion": {"enum": ["none", "partial", "full", "unknown"]},
        "visible_extent": {"enum": ["complete", "partial", "unknown"]},
        "observation_statement": S,
    }
)
LINK = obj(
    {
        "entity_id": ID,
        "candidate_ids": arr(ID, 4, 1),
        "status": {"enum": ["confirmed", "tentative", "distinct", "unresolved"]},
        "basis": {
            "enum": [
                "continuous",
                "overlap",
                "distinctive_attributes",
                "stable_neighbors",
                "appearance_only",
                "none",
            ]
        },
        "source_observation_ids": arr(ID, 12, 1),
        "valid_time": SPAN,
    }
)
GAP = obj(
    {
        "kind": {
            "enum": [
                "task",
                "identity",
                "time",
                "reference_frame",
                "relation",
                "boundary",
                "scale",
                "coverage",
                "route",
                "viewpoint",
                "protocol",
            ]
        },
        "detail": S,
        "entities": arr(ID, 16),
        "record_ids": arr(ID, 16),
        "time_interval": nullable(SPAN),
        "frame_id": nullable(ID),
        "bbox": nullable(BOX),
    }
)

VALUE_SCHEMAS = {
    "point": obj({"coordinates": POINT, "space": {"enum": ["scene", "image"]}}),
    "boundary": obj(
        {"points": arr(POINT, 128, 1), "space": {"enum": ["scene", "image"]}, "complete": B}
    ),
    "bearing": obj(
        {
            "horizontal": {"enum": ["left", "right", "on_axis", "unknown"]},
            "depth": {"enum": ["front", "back", "on_axis", "unknown"]},
            "angle_interval": nullable(SPAN),
        }
    ),
    "heading": obj(
        {
            "degrees": SPAN,
            "kind": {"enum": ["body", "camera", "motion"]},
            "anchor": S,
            "position_entity": nullable(ID),
        }
    ),
    "distance": obj(
        {"interval": SPAN, "semantics": {"enum": ["closest_boundary", "center", "car_front"]}}
    ),
    "extent": obj({"dimensions": arr(SPAN, 3, 3), "complete": B}),
    "area": obj(
        {
            "regions": arr(obj({"id": ID, "polygon": arr(arr(N, 2, 2), 128, 3)}), 16, 1),
            "coverage_complete": B,
        }
    ),
    "scale": obj(
        {
            "kind": {
                "enum": [
                    "readable_ruler",
                    "question_bound_dimension",
                    "frozen_prediction",
                    "category_prior",
                    "unknown",
                ]
            },
            "meters_per_scene_unit": SPAN,
            "reference_entity": ID,
            "reference_length_meters": nullable(SPAN),
            "reference_length_scene": nullable(SPAN),
        }
    ),
    "edge": obj(
        {
            "from": ID,
            "to": ID,
            "departure_heading": nullable(N),
            "arrival_heading": nullable(N),
            "length": nullable(SPAN),
            "cost_seconds": nullable(SPAN),
        }
    ),
    "progress": obj(
        {
            "current": ID,
            "previous": nullable(ID),
            "heading": nullable(N),
            "completed_waypoints": arr(ID, 32),
            "instruction_index": {"type": "integer", "minimum": 0},
            "stop_reached": B,
        }
    ),
    "event": obj({"description": S, "start_s": N, "end_s": N, "replay_of": nullable(ID)}),
    "viewpoint": obj(
        {
            "relation": S,
            "camera_frame": ID,
            "observer_frame": ID,
            "landmark_ids": arr(ID, 16, 1),
            "mirror_plane_id": nullable(ID),
            "axis_defined": B,
        }
    ),
    "alignment": obj(
        {
            "from_frame": ID,
            "to_frame": ID,
            "rotation_degrees": N,
            "translation": POINT,
            "landmark_ids": arr(ID, 16, 2),
        }
    ),
}
METHODS = (
    "direct_observation",
    "question_constraint",
    "multiview_visual_estimate",
    "predicted_geometry",
    "deterministic_transform",
    "category_prior",
)


def record_schema(predicate):
    return obj(
        {
            "record_id": ID,
            "predicate": {"const": predicate},
            "arguments": arr(ID, 24, 1),
            "reference_frame_id": ID,
            "valid_time": SPAN,
            "value": VALUE_SCHEMAS[predicate],
            "unit": nullable(S),
            "method": {"enum": list(METHODS)},
            "source_observation_ids": arr(ID, 24, 1),
            "parent_record_ids": arr(ID, 24),
            "status": {"enum": ["evidence_supported", "estimated", "unresolved"]},
            "uncertainty_description": TEXT,
        }
    )


RECORD = {"oneOf": [record_schema(p) for p in VALUE_SCHEMAS]}
SCHEMAS = {
    "compile": QUESTION_SPEC,
    "observe": obj(
        {"observations": arr(OBSERVATION, 24), "links": arr(LINK, 24), "gaps": arr(GAP, 8)}
    ),
    "relations": obj({"frames": arr(FRAME, 12), "records": arr(RECORD, 16), "gaps": arr(GAP, 8)}),
    "audit": obj(
        {
            "checks": arr(
                obj(
                    {
                        "record_id": ID,
                        "verdict": {"enum": ["supported", "contradicted", "insufficient"]},
                        "frame_ids": arr(ID, 16, 1),
                        "reason": S,
                    }
                ),
                16,
                1,
            ),
            "gaps": arr(GAP, 8),
        }
    ),
    "answer": obj(
        {
            "semantic_answer": {"type": ["string", "number", "array"]},
            "unit": nullable(S),
            "source_ids": arr(ID, 32),
            "reason": S,
        }
    ),
}


def validate(value, schema):
    try:
        # Reject NaN and infinities, which jsonschema otherwise accepts as numbers.
        json.dumps(value, allow_nan=False)
        jsonschema.Draft202012Validator(schema).validate(value)
    except (ValueError, TypeError, jsonschema.ValidationError) as exc:
        raise ProtocolError(str(exc).split("\n")[0][:500]) from exc
    return value


def parse(text, role):
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ProtocolError("role must return a single JSON object") from exc
    return validate(value, SCHEMAS[role])


def role_schema(role, payload):
    schema = deepcopy(SCHEMAS[role])
    if role != "relations":
        return schema
    relevant = {
        "bearing": {"bearing", "point", "alignment", "event"},
        "heading_delta": {"heading", "point", "alignment", "event"},
        "nearest_object": {"distance", "point", "boundary", "scale", "event"},
        "absolute_distance": {"distance", "point", "boundary", "scale", "event"},
        "max_extent": {"extent", "scale", "event"},
        "floor_area": {"area", "scale", "event"},
        "fill_turns": {"edge", "progress", "point", "alignment", "event"},
        "next_action": {"edge", "progress", "point", "alignment", "event"},
        "compare_paths": {"edge", "progress", "event"},
        "viewpoint_relation": {"viewpoint", "alignment", "point", "event"},
        "event_select": {"event"},
    }
    predicates = set().union(
        *(relevant.get(op, set()) for op in payload.get("operations", [payload["operation"]]))
    )
    schema["properties"]["records"]["items"] = {
        "oneOf": [record_schema(p) for p in sorted(predicates)]
    }
    return schema
