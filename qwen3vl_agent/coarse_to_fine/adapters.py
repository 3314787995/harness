from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerRecord:
    answer: str
    confidence: int
    round_index: int


class AnswerAdapter:
    """Normalize task-specific answers and resolve a max-round result."""

    def options_text(self) -> str:
        return ""

    def answer_instruction(self) -> str:
        return "Return a concise answer in the JSON field 'answer'."

    def normalize(self, text: str) -> str | None:
        value = text.strip()
        return value or None

    def vote(self, records: Sequence[AnswerRecord]) -> str:
        if not records:
            raise ValueError("cannot vote without answers")
        return max(records, key=lambda item: (item.confidence, item.round_index)).answer


class MultipleChoiceAdapter(AnswerAdapter):
    def __init__(self, choices: Sequence[str]) -> None:
        if not 2 <= len(choices) <= 26:
            raise ValueError("multiple-choice tasks require between 2 and 26 choices")
        self.choices = tuple(str(choice) for choice in choices)
        self.letters = tuple(chr(ord("A") + index) for index in range(len(self.choices)))

    def options_text(self) -> str:
        lines: list[str] = []
        for letter, choice in zip(self.letters, self.choices):
            cleaned = re.sub(r"^[A-Z][.):]\s*", "", choice.strip(), flags=re.IGNORECASE)
            lines.append(f"{letter}. {cleaned}")
        return "\n".join(lines)

    def answer_instruction(self) -> str:
        return (
            f"The JSON field 'answer' must contain exactly one of: {', '.join(self.letters)}. "
            "For NOT/EXCEPT questions, verify every option and choose the unsupported one."
        )

    def normalize(self, text: str) -> str | None:
        candidate = text.strip().upper()
        if candidate in self.letters:
            return candidate
        pattern = rf"(?<![A-Z])({'|'.join(self.letters)})(?![A-Z])"
        match = re.search(pattern, candidate)
        return match.group(1) if match else None

    def vote(self, records: Sequence[AnswerRecord]) -> str:
        valid = [record for record in records if record.answer in self.letters]
        if not valid:
            raise ValueError("cannot vote without valid multiple-choice answers")
        counts = {letter: sum(record.answer == letter for record in valid) for letter in self.letters}
        candidates = [letter for letter, count in counts.items() if count == max(counts.values())]

        def tie_break(letter: str) -> tuple[int, int]:
            matching = [record for record in valid if record.answer == letter]
            return (
                max(record.confidence for record in matching),
                max(record.round_index for record in matching),
            )

        return max(candidates, key=tie_break)


def build_answer_adapter(choices: Sequence[str] | None) -> AnswerAdapter:
    if choices:
        return MultipleChoiceAdapter(choices)
    return AnswerAdapter()


def answer_record_to_dict(record: AnswerRecord) -> dict[str, Any]:
    return {
        "answer": record.answer,
        "confidence": record.confidence,
        "round_index": record.round_index,
    }
