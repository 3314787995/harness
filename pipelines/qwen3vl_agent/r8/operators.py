"""Deterministic operators. No model-generated code or expression strings are executed."""

import math
import re
from datetime import date, datetime, timedelta
from fractions import Fraction

from .types import ModelingError
from .units import IntervalQuantity, Unit
from .units import Quantity as Q

OPS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "sum",
    "mean",
    "abs",
    "maximum_affordable_count",
    "minimum_required_packages",
    "ratio",
    "percentage",
    "percentage_point_difference",
    "digit_sum",
    "parse_clock",
    "duration_to_minutes",
    "add_minutes",
    "parse_date",
    "add_days",
    "filter",
    "deduplicate",
    "select_nth",
    "argmin",
    "argmax",
    "select_attribute",
    "equal",
    "less_equal",
    "greater_equal",
    "domain",
    "solve_target",
    "nonzero_denominator",
    "nonnegative",
    "recompute",
    "minimality",
    "target_uniqueness",
)
PARAMS = {
    "minimum_required_packages": {"count_unit"},
    "maximum_affordable_count": {"count_unit"},
    "filter": {"attribute", "relation"},
    "deduplicate": {"key"},
    "select_nth": {"index_base"},
    "argmin": {"attribute"},
    "argmax": {"attribute"},
    "select_attribute": {"attribute"},
    "domain": {"kind"},
    "minimality": {"kind"},
    "add_minutes": {"wrap_24h"},
    "parse_date": {"format"},
    "solve_target": {"target"},
    "parse_clock": {"kind"},
}


def scalar(value):
    if not isinstance(value, Q) or not isinstance(value.value, Fraction):
        raise ModelingError("numeric quantity required")
    return value


def items(value):
    if not isinstance(value, list):
        raise ModelingError("collection required")
    return value


def attr(record, name):
    if not isinstance(record, dict) or name not in record:
        raise ModelingError("missing object attribute")
    return record[name]


def integer(value, *, dimensionless=False):
    value = scalar(value)
    if value.base.denominator != 1 or (dimensionless and value.unit.dimensions):
        raise ModelingError("integer of the required dimension expected")
    return int(value.base)


def comparable(a, b):
    if isinstance(a, Q) or isinstance(b, Q):
        scalar(a).compatible(scalar(b))
        return a.base, b.base
    if type(a) is not type(b) or not isinstance(a, (str, date, bool)):
        raise ModelingError("incomparable values")
    return a, b


def compare(a, b, relation):
    a, b = comparable(a, b)
    functions = {
        "eq": lambda: a == b,
        "ne": lambda: a != b,
        "lt": lambda: a < b,
        "le": lambda: a <= b,
        "gt": lambda: a > b,
        "ge": lambda: a >= b,
    }
    if relation not in functions:
        raise ModelingError("unsupported comparison")
    return functions[relation]()


def calculate(op, args, params=None):
    params = params or {}
    if op not in OPS or set(params) - PARAMS.get(op, set()):
        raise ModelingError("unknown operator/parameter")
    binary = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "ratio",
        "percentage_point_difference",
        "maximum_affordable_count",
        "minimum_required_packages",
        "add_minutes",
        "add_days",
        "equal",
        "less_equal",
        "greater_equal",
        "recompute",
        "select_nth",
        "filter",
    }
    expected = 3 if op == "minimality" else 2 if op in binary else 1
    if len(args) != expected:
        raise ModelingError(f"{op} expects {expected} arguments")
    a = args[0]
    if any(isinstance(v, IntervalQuantity) for v in args):
        if op == "target_uniqueness":
            return a.lower == a.upper
        if op not in {
            "add",
            "subtract",
            "multiply",
            "divide",
            "ratio",
            "percentage",
            "percentage_point_difference",
            "abs",
        }:
            raise ModelingError("operator requires resolved scalar inputs")
        endpoints = [
            [Q(v.lower, v.unit), Q(v.upper, v.unit)] if isinstance(v, IntervalQuantity) else [v]
            for v in args
        ]
        if op in {"divide", "ratio"} and endpoints[1][0].base <= 0 <= endpoints[1][-1].base:
            raise ModelingError("interval denominator may be zero")
        import itertools

        results = [calculate(op, list(values), params) for values in itertools.product(*endpoints)]
        unit = results[0].unit
        values = [v.convert(unit).value for v in results]
        if op == "abs" and a.lower <= 0 <= a.upper:
            values.append(Fraction(0))
        return IntervalQuantity(min(values), max(values), unit)
    if op in {"add", "subtract"}:
        b = scalar(args[1])
        scalar(a).compatible(b)
        return Q((a.base + (b.base if op == "add" else -b.base)) / a.unit.factor, a.unit, a.basis)
    if op in {"multiply", "divide", "ratio"}:
        a, b = scalar(a), scalar(args[1])
        if op != "multiply" and b.base == 0:
            raise ModelingError("zero denominator")
        if op == "ratio":
            a.compatible(b)
        return Q(
            a.base * b.base if op == "multiply" else a.base / b.base,
            a.unit.combine(b.unit, 1 if op == "multiply" else -1),
        )
    if op in {"sum", "mean"}:
        values = items(a)
        if not values:
            raise ModelingError("empty aggregate")
        result = scalar(values[0])
        for value in values[1:]:
            result = calculate("add", [result, value])
        return Q(result.value / (len(values) if op == "mean" else 1), result.unit, result.basis)
    if op == "abs":
        return Q(abs(scalar(a).value), a.unit, a.basis)
    if op in {"maximum_affordable_count", "minimum_required_packages"}:
        a, b = scalar(a), scalar(args[1])
        if a.base < 0 or b.base <= 0:
            raise ModelingError("nonnegative demand/budget and positive package/price required")
        quotient = calculate("divide", [a, b])
        count = (
            math.floor(quotient.base)
            if op == "maximum_affordable_count"
            else math.ceil(quotient.base)
        )
        result = Q(Fraction(count), quotient.unit)
        if not calculate(
            "minimality",
            [a, b, result],
            {"kind": "maximum" if op.startswith("maximum") else "minimum"},
        ):
            raise ModelingError("integer boundary check failed")
        if params.get("count_unit"):
            unit = Unit.parse(params["count_unit"])
            if (
                len(unit.dimensions) != 1
                or not unit.dimensions[0][0].startswith("count:")
                or unit.dimensions[0][1] != 1
            ):
                raise ModelingError("integer package result requires a count unit")
            if quotient.unit.dimensions and quotient.unit.dimensions != unit.dimensions:
                raise ModelingError("integer package count unit contradicts price basis")
            result = Q(Fraction(count), unit)
        return result
    if op == "minimality":
        a, b, n = map(scalar, args)
        quotient = calculate("divide", [a, b])
        quotient.compatible(n)
        count = integer(n)
        if count < 0 or a.base < 0 or b.base <= 0:
            return False
        if params.get("kind") == "minimum":
            return count >= quotient.base and (count == 0 or count - 1 < quotient.base)
        if params.get("kind") == "maximum":
            return count <= quotient.base < count + 1
        raise ModelingError("minimality kind must be minimum or maximum")
    if op == "percentage":
        if scalar(a).unit.dimensions:
            raise ModelingError("percentage requires a dimensionless ratio")
        return Q(a.base * 100, Unit.parse("%"))
    if op == "percentage_point_difference":
        a, b = scalar(a), scalar(args[1])
        if a.unit.dimensions or b.unit.dimensions:
            raise ModelingError("percentage-point operands must be rates")
        return Q((a.base - b.base) * 100, Unit.parse("pp"))
    if op == "digit_sum":
        text = str(integer(a)) if isinstance(a, Q) else a
        if not isinstance(text, str) or not re.fullmatch(r"\d+", text):
            raise ModelingError("digit_sum requires an unsigned digit string")
        return Q(Fraction(sum(int(c) for c in text)))
    if op == "parse_clock":
        if not isinstance(a, str):
            raise ModelingError("clock text required")
        m = re.fullmatch(r"(\d{1,6}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?", a.strip(), re.IGNORECASE)
        if not m:
            raise ModelingError("invalid clock")
        h, minute, second = int(m[1]), int(m[2]), int(m[3] or 0)
        if m[4]:
            if not 1 <= h <= 12:
                raise ModelingError("invalid 12-hour clock")
            h = h % 12 + (12 if m[4].upper() == "PM" else 0)
        kind = params.get("kind", "clock")
        if kind not in {"clock", "duration"} or (kind == "duration" and m[4]):
            raise ModelingError("invalid clock/duration convention")
        if not ((h < 24 or kind == "duration") and minute < 60 and second < 60):
            raise ModelingError("invalid clock")
        return Q(
            Fraction(h * 3600 + minute * 60 + second),
            Unit.parse("clock" if kind == "clock" else "s"),
        )
    if op == "duration_to_minutes":
        return scalar(a).convert("min")
    if op == "add_minutes":
        if scalar(a).unit.dimensions != Unit.parse("clock").dimensions:
            raise ModelingError("parsed clock required")
        delta = scalar(args[1]).convert("min").value * 60
        result = a.value + delta
        if params.get("wrap_24h", False):
            result %= 86400
        return Q(result, Unit.parse("clock"))
    if op == "parse_date":
        if not isinstance(a, str):
            raise ModelingError("date text required")
        formats = {
            "iso": "%Y-%m-%d",
            "month_name": "%B %d, %Y",
            "dmy": "%d/%m/%Y",
            "mdy": "%m/%d/%Y",
        }
        chosen = params.get("format", "iso")
        if chosen not in formats:
            raise ModelingError("unsupported date format")
        try:
            return datetime.strptime(a, formats[chosen]).date()  # noqa: DTZ007 -- calendar date only
        except ValueError as exc:
            raise ModelingError("invalid date") from exc
    if op == "add_days":
        if not isinstance(a, date):
            raise ModelingError("parsed calendar date required")
        days = scalar(args[1]).convert("day").value
        if days.denominator != 1 or abs(days) > 366000:
            raise ModelingError("integer calendar day offset required")
        try:
            return a + timedelta(days=int(days))
        except (ValueError, OverflowError) as exc:
            raise ModelingError("date outside supported range") from exc
    if op == "filter":
        return [
            r
            for r in items(a)
            if compare(attr(r, params.get("attribute")), args[1], params.get("relation"))
        ]
    if op == "deduplicate":
        key, seen, result = params.get("key", "id"), set(), []
        for record in items(a):
            identity = attr(record, key)
            if not isinstance(identity, str) or not identity:
                raise ModelingError("deduplication needs explicit stable string identities")
            if identity not in seen:
                result.append(record)
                seen.add(identity)
        return result
    if op == "select_nth":
        base = params.get("index_base", 1)
        if base not in {0, 1}:
            raise ModelingError("index_base must be 0 or 1")
        index = integer(args[1], dimensionless=True) - base
        if not 0 <= index < len(items(a)):
            raise ModelingError("selection out of bounds")
        return a[index]
    if op in {"argmin", "argmax"}:
        rows = items(a)
        if not rows:
            raise ModelingError("empty extremum set")
        key = params.get("attribute")
        best, ties = attr(rows[0], key), []
        for record in rows:
            value = attr(record, key)
            if compare(value, best, "lt" if op == "argmin" else "gt"):
                best, ties = value, [record]
            elif compare(value, best, "eq"):
                ties.append(record)
        return ties
    if op == "select_attribute":
        key = params.get("attribute")
        return [attr(row, key) for row in a] if isinstance(a, list) else attr(a, key)
    if op in {"equal", "less_equal", "greater_equal", "recompute"}:
        return compare(
            a,
            args[1],
            {"equal": "eq", "less_equal": "le", "greater_equal": "ge", "recompute": "eq"}[op],
        )
    if op == "domain":
        value = scalar(a).base
        kind = params.get("kind")
        if kind not in {"real", "integer", "positive", "nonnegative"}:
            raise ModelingError("unsupported domain")
        return {
            "real": True,
            "integer": value.denominator == 1,
            "positive": value > 0,
            "nonnegative": value >= 0,
        }[kind]
    if op == "nonzero_denominator":
        return scalar(a).base != 0
    if op == "nonnegative":
        return scalar(a).base >= 0
    if op == "target_uniqueness":
        if isinstance(a, dict) and "status" in a:
            return a["status"] == "solved_target"
        return not isinstance(a, list) or (
            len(a) > 0 and all(compare(a[0], v, "eq") for v in a[1:])
        )
    if op == "solve_target":
        # Solver documents are produced by the checked constraint backend, never arbitrary strings.
        if not isinstance(a, dict) or a.get("status") != "solved_target" or "quantity" not in a:
            raise ModelingError("solve_target requires a successful checked solver result")
        return a["quantity"]
    raise ModelingError("unsupported operation")
