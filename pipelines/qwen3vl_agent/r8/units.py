"""Exact scalars and a deliberately small, non-executable unit grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from fractions import Fraction

from .types import ModelingError


def number(value, digits=100):
    if isinstance(value, bool) or not isinstance(value, (str, int, Fraction, Decimal)):
        raise ModelingError("exact numeric input must be a string/integer/rational")
    text = str(value).strip().replace("−", "-")
    if len(text) > digits * 2 + 8 or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+)?", text):
        raise ModelingError("invalid or oversized exact number")
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ModelingError("invalid rational") from exc
    if max(len(str(abs(result.numerator))), len(str(result.denominator))) > digits:
        raise ModelingError("numeric digit limit")
    return result


def numeric_tokens(text, *, group_thousands=True):
    text = text.replace("−", "-").replace("²", "^2").replace("³", "^3")
    if group_thousands:
        text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"(?<=\d)(?=(?:kg|mg|g|cm|mm|ml|m|L|min|s|h|days?|cups?|jars?)\b)", " ", text)
    text = re.sub(r"\b(cm|mm|m)\^[23]\b", r"\1", text)
    return re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:/\d+)?(?!\w|\.\d)", text)


_ATOMS = {
    "1": ({}, Fraction(1)),
    "": ({}, Fraction(1)),
    "%": ({}, Fraction(1, 100)),
    "percent": ({}, Fraction(1, 100)),
    "pp": ({"percentage_point": 1}, Fraction(1)),
    "m": ({"length": 1}, Fraction(1)),
    "cm": ({"length": 1}, Fraction(1, 100)),
    "mm": ({"length": 1}, Fraction(1, 1000)),
    "km": ({"length": 1}, Fraction(1000)),
    "g": ({"mass": 1}, Fraction(1)),
    "kg": ({"mass": 1}, Fraction(1000)),
    "mg": ({"mass": 1}, Fraction(1, 1000)),
    "s": ({"time": 1}, Fraction(1)),
    "min": ({"time": 1}, Fraction(60)),
    "h": ({"time": 1}, Fraction(3600)),
    "day": ({"time": 1}, Fraction(86400)),
    "L": ({"volume": 1}, Fraction(1)),
    "ml": ({"volume": 1}, Fraction(1, 1000)),
    "degree": ({"angle": 1}, Fraction(1)),
    "date": ({"date": 1}, Fraction(1)),
    "clock": ({"clock": 1}, Fraction(1)),
}
for _currency in ("CNY", "USD", "EUR", "GBP", "JPY", "KRW", "CAD", "AUD", "PHP"):
    _ATOMS[_currency] = ({"currency:" + _currency: 1}, Fraction(1))
for _count in ("count", "cup", "jar", "item", "attempt", "arrow", "person", "ball", "package"):
    _ATOMS[_count] = ({"count:" + _count: 1}, Fraction(1))
_ALIASES = {
    "RMB": "CNY",
    "元": "CNY",
    "¥": "CNY",
    "$": "USD",
    "€": "EUR",
    "minutes": "min",
    "minute": "min",
    "seconds": "s",
    "second": "s",
    "hours": "h",
    "days": "day",
    "cups": "cup",
    "jars": "jar",
    "grams": "g",
    "percentages": "%",
    "percentage_points": "pp",
    "₱": "PHP",
    "£": "GBP",
}


@dataclass(frozen=True)
class Unit:
    dimensions: tuple[tuple[str, int], ...] = ()
    factor: Fraction = Fraction(1)
    label: str = "1"

    @classmethod
    def parse(cls, text="1"):
        if isinstance(text, cls):
            return text
        if not isinstance(text, str) or len(text) > 100:
            raise ModelingError("invalid unit")
        text = text.strip().replace("²", "^2").replace("³", "^3") or "1"
        text = _ALIASES.get(text, text)
        parts = re.split(r"([*/])", text)
        dimensions, factor, sign = {}, Fraction(1), 1
        for part in parts:
            if part in {"*", "/"}:
                sign = 1 if part == "*" else -1
                continue
            m = re.fullmatch(r"([^\^\s]+)(?:\^(-?[1-3]))?", part.strip())
            if not m:
                raise ModelingError("unsupported unit syntax")
            atom = _ALIASES.get(m[1], m[1])
            if re.fullmatch(r"count:[A-Za-z][A-Za-z0-9_]{0,40}", atom):
                dims, scale = {atom: 1}, Fraction(1)
            elif atom not in _ATOMS:
                raise ModelingError(f"unknown unit: {atom}")
            else:
                dims, scale = _ATOMS[atom]
            exponent = sign * int(m[2] or 1)
            factor *= scale**exponent
            for key, value in dims.items():
                dimensions[key] = dimensions.get(key, 0) + value * exponent
        return cls(tuple(sorted((k, v) for k, v in dimensions.items() if v)), factor, text)

    def combine(self, other, sign=1):
        dims = dict(self.dimensions)
        for key, value in other.dimensions:
            dims[key] = dims.get(key, 0) + sign * value
        # Result is in canonical base units; the label is explanatory, not reparsed.
        return Unit(
            tuple(sorted((k, v) for k, v in dims.items() if v)),
            Fraction(1),
            f"({self.label}){'*' if sign == 1 else '/'}({other.label})",
        )


@dataclass(frozen=True)
class Quantity:
    value: object
    unit: Unit = Unit()
    basis: str = ""

    @classmethod
    def make(cls, value, unit="1", basis="", digits=100):
        return cls(number(value, digits), Unit.parse(unit), basis)

    @property
    def base(self):
        if not isinstance(self.value, Fraction):
            raise ModelingError("scalar number required")
        return self.value * self.unit.factor

    def convert(self, unit):
        unit = Unit.parse(unit)
        if self.unit.dimensions != unit.dimensions:
            raise ModelingError("incompatible units/currencies/count basis")
        return Quantity(self.base / unit.factor, unit, self.basis)

    def compatible(self, other):
        if not isinstance(other, Quantity) or self.unit.dimensions != other.unit.dimensions:
            raise ModelingError("incompatible units")
        if self.basis and other.basis and self.basis != other.basis:
            raise ModelingError("incompatible pricing or measurement basis")


def encode(value):
    if isinstance(value, IntervalQuantity):
        return {
            "interval": [str(value.lower), str(value.upper)],
            "unit": value.unit.label,
            "dimensions": dict(value.unit.dimensions),
            "factor": str(value.unit.factor),
        }
    if isinstance(value, Quantity):
        return {
            "value": encode(value.value),
            "unit": value.unit.label,
            "dimensions": dict(value.unit.dimensions),
            "factor": str(value.unit.factor),
            "basis": value.basis,
        }
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class IntervalQuantity:
    """A conservative, exhaustive range; never a probability or a point estimate."""

    lower: Fraction
    upper: Fraction
    unit: Unit = Unit()

    def __post_init__(self):
        if self.lower > self.upper:
            raise ModelingError("reversed interval")

    def convert(self, unit):
        unit = Unit.parse(unit)
        if unit.dimensions != self.unit.dimensions:
            raise ModelingError("incompatible interval unit")
        factor = self.unit.factor / unit.factor
        return IntervalQuantity(self.lower * factor, self.upper * factor, unit)


def decimal_text(value, places=8):
    if not 0 <= places <= 50:
        raise ModelingError("display precision out of bounds")
    value = number(value)
    with localcontext() as context:
        context.prec = max(110, places + 10)
        return format(Decimal(value.numerator) / Decimal(value.denominator), f".{places}f")
