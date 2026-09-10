"""Semantic answers are mapped back to immutable original option order."""

import re

from .types import ProtocolError, finite


def normalized(value):
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    value = str(value).strip().casefold()
    value = re.sub(r"^(?:the\s+)?answer\s*(?:is|:)\s*", "", value)
    value = value.replace("backward", "back").replace("forward", "front").replace("°", " degrees")
    value = re.sub(r"\bturn\s+(left|right|back)\b", r"\1", value)
    return " ".join(re.findall(r"[\w]+(?:\.\d+)?", value))


def match_choice(semantic, choices):
    key = normalized(semantic)
    matches = [c for c in choices if normalized(c.text) == key]
    return matches[0].label if len(matches) == 1 else None


def validate_answer(value, request, allowed_source_ids, output_unit):
    if set(value["source_ids"]) - set(allowed_source_ids):
        raise ProtocolError("answer cites unavailable evidence")
    semantic = value["semantic_answer"]
    if request.protocol == "multiple_choice" and match_choice(semantic, request.choices) is None:
        raise ProtocolError("semantic answer must identify exactly one original option text")
    if request.protocol == "numeric":
        if not finite(semantic):
            raise ProtocolError("numeric output requires a finite number")
        if output_unit in {"m", "cm", "mm", "m2", "cm2", "m²"} and semantic < 0:
            raise ProtocolError("spatial length/area estimates must be nonnegative")
        if value["unit"] != output_unit:
            raise ProtocolError("numeric output must use the required unit")
    if request.protocol == "free_text" and (not isinstance(semantic, str) or not semantic.strip()):
        raise ProtocolError("free text answer must be nonempty")
    return value


def format_prediction(semantic, request):
    if semantic is None:
        return None, ""
    if request.protocol == "multiple_choice":
        label = match_choice(semantic, request.choices)
        if label is None:
            raise ProtocolError("semantic answer does not match one original option")
        return label, label
    if request.protocol == "numeric":
        if not finite(semantic):
            raise ProtocolError("invalid numeric prediction")
        return semantic, format(semantic, ".12g")
    return semantic, str(semantic)
