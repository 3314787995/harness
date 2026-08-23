from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from qwen3vl_agent.active_tree.types import CanonicalOption, EvidenceLedger, TaskContract


@dataclass(frozen=True)
class TemporalComposition:
    option_id: str | None
    slot_order: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    timestamps: tuple[tuple[str, float], ...]
    option_pattern: str
    confident: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "slot_order": list(self.slot_order),
            "evidence_ids": list(self.evidence_ids),
            "timestamps": [
                {"slot_id": slot_id, "timestamp_seconds": round(timestamp, 3)}
                for slot_id, timestamp in self.timestamps
            ],
            "option_pattern": self.option_pattern,
            "confident": self.confident,
            "reason": self.reason,
        }


def compose_temporal_option(
    options: list[CanonicalOption],
    contract: TaskContract,
    ledger: EvidenceLedger,
    *,
    min_gap_seconds: float,
) -> TemporalComposition:
    if contract.primary_topology != "sequence":
        return _empty("contract is not sequential")

    required_slots = [slot for slot in contract.slots if slot.required]
    if len(required_slots) < 2 or len(required_slots) > 26:
        return _empty("sequence requires between 2 and 26 slots")

    chosen = []
    for slot in required_slots:
        candidates = [
            item
            for item in ledger.active_items
            if slot.slot_id in item.slot_ids
        ]
        if not candidates:
            return _empty(f"missing active evidence for {slot.slot_id}")
        item = min(
            candidates,
            key=lambda value: (
                (value.start_seconds + value.end_seconds) / 2,
                value.evidence_id,
            ),
        )
        timestamp = (item.start_seconds + item.end_seconds) / 2
        chosen.append((slot.slot_id, timestamp, item.evidence_id))

    chronological = sorted(chosen, key=lambda value: (value[1], value[0]))
    gaps = [
        right[1] - left[1]
        for left, right in pairwise(chronological)
    ]
    if any(gap < min_gap_seconds for gap in gaps):
        return TemporalComposition(
            None,
            tuple(item[0] for item in chronological),
            tuple(item[2] for item in chronological),
            tuple((item[0], item[1]) for item in chronological),
            "",
            False,
            "event timestamps are not distinctly ordered",
        )

    symbols = {
        slot.slot_id: chr(ord("a") + index)
        for index, slot in enumerate(required_slots)
    }
    pattern_tuple = tuple(symbols[item[0]] for item in chronological)
    pattern = "".join(f"({symbol})" for symbol in pattern_tuple)
    matched = [
        option
        for option in options
        if tuple(re.findall(r"\(([a-z])\)", option.text.casefold())) == pattern_tuple
    ]
    if len(matched) != 1:
        return TemporalComposition(
            None,
            tuple(item[0] for item in chronological),
            tuple(item[2] for item in chronological),
            tuple((item[0], item[1]) for item in chronological),
            pattern,
            False,
            "chronological pattern does not map to exactly one option",
        )
    return TemporalComposition(
        matched[0].option_id,
        tuple(item[0] for item in chronological),
        tuple(item[2] for item in chronological),
        tuple((item[0], item[1]) for item in chronological),
        pattern,
        True,
        "all required slots have distinctly ordered active evidence",
    )


def _empty(reason: str) -> TemporalComposition:
    return TemporalComposition(None, (), (), (), "", False, reason)
