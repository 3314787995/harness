from __future__ import annotations

import re
from dataclasses import dataclass

from qwen3vl_agent.active_tree.types import AtomicEvidence, CanonicalOption

_STOPWORDS = {
    "about",
    "after",
    "again",
    "before",
    "being",
    "could",
    "does",
    "from",
    "have",
    "into",
    "many",
    "person",
    "people",
    "should",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "video",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "will",
    "with",
    "would",
}


@dataclass(frozen=True)
class EvidenceAlignment:
    option_id: str | None
    evidence_id: str | None
    score: float
    runner_up_score: float
    margin: float
    matched_phrase: str
    phrase_tokens: int
    confident: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "evidence_id": self.evidence_id,
            "score": round(self.score, 3),
            "runner_up_score": round(self.runner_up_score, 3),
            "margin": round(self.margin, 3),
            "matched_phrase": self.matched_phrase,
            "phrase_tokens": self.phrase_tokens,
            "confident": self.confident,
        }


def align_evidence_to_options(
    options: list[CanonicalOption],
    evidence: tuple[AtomicEvidence, ...],
    *,
    min_phrase_tokens: int,
    min_margin: float,
) -> EvidenceAlignment:
    """Find only high-precision, option-unique phrase matches in grounded evidence."""

    if not options or not evidence:
        return EvidenceAlignment(None, None, 0.0, 0.0, 0.0, "", 0, False)

    option_tokens = {option.option_id: _tokens(option.text) for option in options}
    option_ngrams = {
        option_id: _ngrams(tokens, minimum=2, maximum=4)
        for option_id, tokens in option_tokens.items()
    }
    token_frequency: dict[str, int] = {}
    for tokens in option_tokens.values():
        for token in set(tokens):
            token_frequency[token] = token_frequency.get(token, 0) + 1

    results: list[tuple[float, int, str, str, str]] = []
    for option in options:
        other_ngrams = set().union(
            *(
                grams
                for option_id, grams in option_ngrams.items()
                if option_id != option.option_id
            )
        )
        unique_ngrams = option_ngrams[option.option_id] - other_ngrams
        unique_tokens = {
            token
            for token in option_tokens[option.option_id]
            if token_frequency.get(token, 0) == 1
        }
        best: tuple[float, int, str, str, str] = (
            0.0,
            0,
            "",
            option.option_id,
            "",
        )
        for item in evidence:
            evidence_tokens = _tokens(item.fact)
            evidence_set = set(evidence_tokens)
            matched = [
                gram for gram in unique_ngrams if _contains(evidence_tokens, gram)
            ]
            phrase = max(matched, key=lambda value: (len(value), value), default=())
            overlap = len(unique_tokens & evidence_set)
            score = float(len(phrase) * 3 + overlap)
            candidate = (
                score,
                len(phrase),
                " ".join(phrase),
                option.option_id,
                item.evidence_id,
            )
            best = max(best, candidate)
        results.append(best)

    results.sort(reverse=True)
    score, phrase_tokens, phrase, option_id, evidence_id = results[0]
    runner_up = results[1][0] if len(results) > 1 else 0.0
    margin = score - runner_up
    confident = phrase_tokens >= min_phrase_tokens and margin >= min_margin
    return EvidenceAlignment(
        option_id,
        evidence_id,
        score,
        runner_up,
        margin,
        phrase,
        phrase_tokens,
        confident,
    )


def _tokens(text: str) -> tuple[str, ...]:
    text = re.sub(r"\[?\d+(?:\.\d+)?s?-\d+(?:\.\d+)?s?\]?", " ", text)
    values = re.findall(r"[a-z0-9]+", text.casefold())
    return tuple(value for value in values if len(value) >= 3 and value not in _STOPWORDS)


def _ngrams(
    tokens: tuple[str, ...],
    *,
    minimum: int,
    maximum: int,
) -> set[tuple[str, ...]]:
    return {
        tokens[index : index + size]
        for size in range(minimum, min(maximum, len(tokens)) + 1)
        for index in range(len(tokens) - size + 1)
    }


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )
