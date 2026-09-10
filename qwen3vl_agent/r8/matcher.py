"""Option-only parsing and exact/precision-aware matching; options never become evidence."""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from fractions import Fraction

from .types import ModelingError
from .units import IntervalQuantity, Quantity, Unit, encode, number, numeric_tokens


@dataclass
class Match:
    prediction: str | None
    status: str
    labels: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


def parse_option(text, unit, answer_type):
    clean = text.strip().rstrip(".。")
    if re.fullmatch(
        r"(?:none of (?:the )?others|none of the above|其他选项均不正确)[.!。]?",
        clean,
        re.IGNORECASE,
    ):
        return {"kind": "none"}
    if re.search(r"cannot be determined|not enough information|无法确定", clean, re.IGNORECASE):
        return {"kind": "unknown"}
    if answer_type == "date":
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                return {"kind": "value", "value": datetime.strptime(clean, fmt).date()}  # noqa: DTZ007 -- calendar date, not an instant
            except ValueError:
                pass
        return {"kind": "invalid", "reason": "invalid_or_unparsed_date"}
    if answer_type == "clock":
        from .operators import calculate

        try:
            return {"kind": "value", "value": calculate("parse_clock", [clean])}
        except ModelingError:
            return {"kind": "invalid", "reason": "invalid_clock"}
    tokens = numeric_tokens(clean, group_thousands=answer_type != "tuple")
    if not tokens:
        return {"kind": "unparsed", "reason": "no_literal_numeric_value"}
    detected = unit
    if re.search(r"percentage points?|百分点", clean, re.IGNORECASE):
        detected = "pp"
    elif "%" in clean or re.search(r"\bpercent\b", clean, re.IGNORECASE):
        detected = "%"
    else:
        unit_text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", clean).replace("^2", "²").replace("^3", "³")
        for pattern, name in (
            (r"RMB|CNY|元|¥", "CNY"),
            (r"USD|\$", "USD"),
            (r"EUR|€", "EUR"),
            (r"GBP|£", "GBP"),
            (r"PHP|₱", "PHP"),
            (r"\bkg\b", "kg"),
            (r"\bmg\b", "mg"),
            (r"\bg\b", "g"),
            (r"\bmm[²2]\b", "mm^2"),
            (r"\bcm[²2]\b", "cm^2"),
            (r"\bm[²2]\b", "m^2"),
            (r"\bmm\b", "mm"),
            (r"\bcm\b", "cm"),
            (r"\bm\b", "m"),
            (r"\bml\b", "ml"),
            (r"\bL\b", "L"),
            (r"\bminutes?\b", "min"),
        ):
            if re.search(pattern, unit_text):
                detected = name
                # A price per queried item can omit the denominator in the printed option.
                if "/" in unit and unit.split("/", 1)[0] == detected:
                    detected = unit
                break
    value = [Quantity.make(v, detected) for v in tokens]
    if re.search(r"\blower\b|\bless\b|低|少", clean, re.IGNORECASE) and len(value) == 1:
        value = [Quantity(-abs(value[0].value), value[0].unit)]
    if len(value) != 1 and answer_type != "tuple":
        return {"kind": "unparsed", "reason": "multiple_numbers_require_tuple_target"}
    return {"kind": "value", "value": value if answer_type == "tuple" else value[0]}


def equal(a, b):
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        a.compatible(b)
        return a.base == b.base
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def bounds(value):
    if isinstance(value, IntervalQuantity):
        return value.lower * value.unit.factor, value.upper * value.unit.factor, value.unit
    if isinstance(value, Quantity):
        return value.base, value.base, value.unit
    if isinstance(value, dict) and value.get("status") == "solved_target":
        v, unit = value["value"], Unit.parse(value["unit"])
        return number(v["lower"]) * unit.factor, number(v["upper"]) * unit.factor, unit
    if isinstance(value, dict) and set(value) == {"interval", "unit"}:
        unit = Unit.parse(value["unit"])
        return (
            number(value["interval"][0]) * unit.factor,
            number(value["interval"][1]) * unit.factor,
            unit,
        )
    return None


def match(value, choices, task):
    if isinstance(value, IntervalQuantity) and value.lower == value.upper:
        value = Quantity(value.lower, value.unit)
    if isinstance(value, dict) and "candidate_values" in value:
        if not value["complete"]:
            return Match(
                None, "unresolved", details={"reason": "joint candidate enumeration truncated"}
            )
        matches = [match(v, choices, task) for v in value["candidate_values"]]
        predictions = {m.prediction for m in matches}
        if len(predictions) == 1 and None not in predictions:
            identical = len({str(encode(v)) for v in value["candidate_values"]}) == 1
            return Match(
                matches[0].prediction,
                matches[0].status if identical else "option_precision",
                matches[0].labels,
                [a for m in matches for a in m.anomalies],
                {"joint_candidates": len(matches), "complete": True, "same_answer_for_all": True},
            )
        return Match(
            None, "unresolved", details={"candidate_predictions": [m.prediction for m in matches]}
        )
    if not choices:
        if isinstance(value, Quantity):
            if value.unit.label == "clock" and value.value.denominator == 1:
                seconds = int(value.value)
                days, remaining = divmod(seconds, 86400)
                h, rem = divmod(remaining, 3600)
                minute, second = divmod(rem, 60)
                clock = f"{h:02d}:{minute:02d}" + (f":{second:02d}" if second else "")
                return Match((f"day{days:+d} " if days else "") + clock, "exact")
            return Match(str(value.value), "exact", details={"unit": value.unit.label})
        if isinstance(value, date):
            return Match(value.isoformat(), "exact")
        if isinstance(value, dict) and value.get("status") == "solved_target":
            return Match(
                value["value"]["exact"],
                "exact",
                details={"unit": value["unit"], "algebraic": value["value"]},
            )
        return Match(
            None, "unresolved", details={"reason": "open numeric output is not uniquely scalar"}
        )
    parsed = [(c, parse_option(c.text, task["output_unit"], task["answer_type"])) for c in choices]
    anomalies = [
        {"label": c.label, "reason": p["reason"]} for c, p in parsed if p["kind"] == "invalid"
    ]
    if task["answer_type"] == "count" and all(
        p["kind"] != "value" or isinstance(p["value"], Quantity) and p["value"].unit.label == "%"
        for _, p in parsed
    ):
        return Match(
            None,
            "annotation_anomaly",
            anomalies=anomalies + [{"reason": "count_question_percentage_options"}],
        )
    interval = bounds(value)
    hits, evaluated, unknown = [], [], []
    precision = task["precision"]["kind"]
    for choice, item in parsed:
        if item["kind"] == "invalid":
            evaluated.append(choice.label)
            continue
        if item["kind"] in {"none", "unknown"}:
            continue
        if item["kind"] != "value":
            unknown.append(choice.label)
            continue
        candidate = item["value"]
        try:
            if interval:
                low, high, unit = interval
                if (
                    not isinstance(candidate, Quantity)
                    or candidate.unit.dimensions != unit.dimensions
                ):
                    anomalies.append(
                        {"label": choice.label, "reason": "answer_option_type_or_unit_mismatch"}
                    )
                    unknown.append(choice.label)
                    continue
                if precision == "exact":
                    if low == high == candidate.base:
                        hits.append(choice.label)
                    elif low <= candidate.base <= high:
                        unknown.append(choice.label)
                elif precision == "rounded":
                    places = task["precision"]["places"]
                    if places is None:
                        raise ModelingError("rounded precision needs places")
                    radius = Fraction(1, 2 * 10**places) * candidate.unit.factor
                    # Strict interiors avoid silently choosing a tie-rounding convention.
                    if candidate.base - radius < low <= high < candidate.base + radius:
                        hits.append(choice.label)
                    elif not (high < candidate.base - radius or low > candidate.base + radius):
                        unknown.append(choice.label)
                evaluated.append(choice.label)
            elif equal(value, candidate):
                hits.append(choice.label)
                evaluated.append(choice.label)
            else:
                evaluated.append(choice.label)
        except ModelingError:
            unknown.append(choice.label)
    if precision == "closest" and interval:
        low, high, unit = interval
        candidates = [
            (c.label, p["value"].base)
            for c, p in parsed
            if p["kind"] == "value"
            and isinstance(p["value"], Quantity)
            and p["value"].unit.dimensions == unit.dimensions
        ]
        # A candidate must remain nearest throughout the complete input/error interval.
        if candidates:

            def nearest(x):
                distance = min(abs(x - y) for _, y in candidates)
                return {label for label, y in candidates if abs(x - y) == distance}

            hits = [c.label for c in choices if c.label in nearest(low) & nearest(high)]
    if hits and (not unknown or precision == "exact"):
        if unknown:
            anomalies.append({"reason": "other_options_not_fully_parsed", "labels": unknown})
        if len(hits) > 1:
            anomalies.append(
                {
                    "reason": "equivalent_or_tied_option_labels",
                    "labels": hits,
                    "policy": "first_in_original_order",
                }
            )
        return Match(
            hits[0],
            "exact" if precision == "exact" else "option_precision",
            hits,
            anomalies,
            {"value": encode(value), "evaluated": evaluated},
        )
    if not hits and not unknown:
        none = [c.label for c, p in parsed if p["kind"] == "none"]
        if none and len(evaluated) == sum(p["kind"] not in {"none", "unknown"} for _, p in parsed):
            return Match(
                none[0],
                "exact",
                none,
                anomalies,
                {"reason": "known_result_excludes_all_numeric_options"},
            )
    return Match(
        None,
        "unresolved",
        hits,
        anomalies,
        {"unparsed_or_uncertain": unknown, "evaluated": evaluated},
    )
