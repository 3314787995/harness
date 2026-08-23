from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qwen3vl_agent.coarse_to_fine.adapters import AnswerAdapter
from qwen3vl_agent.coarse_to_fine.types import DecisionError, FrameRef, TimeWindow


@dataclass(frozen=True)
class GlanceDecision:
    scope: str
    reason: str


@dataclass(frozen=True)
class SelectionDecision:
    window_ids: tuple[str, ...]
    rationale: str
    evidence_needed: str


@dataclass(frozen=True)
class ReasoningDecision:
    answer: str
    reason: str
    confidence: int
    evidence_frame_ids: tuple[str, ...]
    missing_evidence: str


def question_block(question: str, adapter: AnswerAdapter) -> str:
    options = adapter.options_text()
    if not options:
        return f"Question:\n{question.strip()}"
    return f"Question:\n{question.strip()}\n\nOptions:\n{options}"


def build_glance_prompt(question: str, adapter: AnswerAdapter) -> str:
    return f"""You route a video question before detailed temporal search.

The supplied four frames are a uniform glance over the whole video. Classify the question:
- global: the answer depends on the overall theme, repeated/global statistics, the whole story, or
  proving that something is absent. NOT/EXCEPT/never-shown questions that require checking all
  options across the video are global.
- local: the answer depends on a particular moment, object, action, text, dialogue, or short interval.

{question_block(question, adapter)}

Return exactly one compact JSON object with only the key "scope". Do not add markdown,
an explanation, or any other key. Keep the entire response under 10 tokens. Example:
{{"scope":"global"}}
"""


def build_selection_prompt(
    question: str,
    adapter: AnswerAdapter,
    windows: Sequence[TimeWindow],
    *,
    top_k: int,
    history: Sequence[dict[str, Any]],
    candidate_subtitles: dict[str, str] | None = None,
) -> str:
    window_lines: list[str] = []
    for window in windows:
        line = f"- {window.id}: {window.start_seconds:.3f}s to {window.end_seconds:.3f}s"
        subtitle = (candidate_subtitles or {}).get(window.id, "").replace("\n", " | ")
        if subtitle:
            line += f"; aligned subtitles: {subtitle}"
        window_lines.append(line)
    return f"""You select temporal windows for closer inspection.

The contact sheet tiles are labeled with the candidate window IDs and time ranges below.
Select up to {top_k} windows most likely to contain evidence for the question. Use prior
reasoning to seek missing or contradictory evidence, not merely the most visually salient tile.

{question_block(question, adapter)}

Candidate windows:
{chr(10).join(window_lines)}

Previous reasoning history (JSON):
{json.dumps(list(history), ensure_ascii=False)}

Return one compact JSON object with ONLY the key "window_ids". Do not explain the choice,
repeat the question, or add markdown. The entire response must be under 40 tokens:
{{"window_ids":["window_id"]}}
"""


def build_reasoning_prompt(
    question: str,
    adapter: AnswerAdapter,
    windows: Sequence[TimeWindow],
    frames: Sequence[FrameRef],
    *,
    subtitles: str,
    history: Sequence[dict[str, Any]],
    round_index: int,
) -> str:
    frame_lines = "\n".join(
        f"- Frame {index}: id={frame.id}, time={frame.timestamp_seconds:.3f}s"
        for index, frame in enumerate(frames, start=1)
    )
    window_lines = "\n".join(
        f"- {window.id}: {window.start_seconds:.3f}s to {window.end_seconds:.3f}s"
        for window in windows
    )
    subtitle_text = subtitles or "<no aligned subtitles supplied>"
    return f"""You are at coarse-to-fine video reasoning round {round_index}.

The supplied frames are ordered chronologically. Their exact source timestamps are listed below.
Use only visible evidence, aligned subtitles, and the recorded prior history. Report confidence as
an integer from 0 to 3: 3 means the answer is directly supported by sufficient evidence; 0-2 means
more temporal inspection is needed. Do not use confidence 3 for a guess. Keep "reason" under 30
words and "missing_evidence" under 15 words. The JSON object must always be completed.

Before choosing a multiple-choice answer, silently compare every option with the concrete evidence.
Choose the most specifically supported option, not a broader topic that merely sounds plausible.
For questions about speech, argument, asking, naming, or dialogue, explicit aligned subtitle wording
is primary evidence and should outweigh an impression inferred only from the pictures.
After choosing the option text, copy the letter attached to that exact option. Recheck that the
answer letter's option agrees with the reason; never return a label whose option contradicts it.

{question_block(question, adapter)}

Selected windows:
{window_lines}

Frame map:
{frame_lines}

Aligned subtitles:
{subtitle_text}

Previous reasoning history (JSON):
{json.dumps(list(history), ensure_ascii=False)}

{adapter.answer_instruction()}
Return exactly one JSON object, without markdown:
{{"answer":"...","reason":"evidence-grounded reason","confidence":0,\
"missing_evidence":"what to inspect next, or empty"}}
"""


def build_direct_prompt(question: str, adapter: AnswerAdapter, *, subtitles: str = "") -> str:
    subtitle_block = f"\n\nSubtitles:\n{subtitles}" if subtitles else ""
    return f"""Answer using the supplied video evidence.

{question_block(question, adapter)}{subtitle_block}

{adapter.answer_instruction()}
Return only the answer, without explanation.
"""


def build_protocol_repair_prompt(
    protocol: str,
    raw_response: str,
    *,
    schema_hint: str,
) -> str:
    return f"""You repair a truncated or malformed controller response.

Do not solve the video question, add evidence, or change a decision already present. Reformat only
the SOURCE RESPONSE into one complete compact JSON object matching the schema. If a reasoning
field is missing, use confidence 0, an empty reason, and missing_evidence "format repair".

PROTOCOL: {protocol}
SCHEMA/ALLOWED VALUES: {schema_hint}

SOURCE RESPONSE:
{raw_response}

Return JSON only, under 60 tokens.
"""


def parse_glance(text: str) -> GlanceDecision:
    payload = parse_json_object(text)
    scope = str(payload.get("scope", "")).strip().lower()
    if scope not in {"global", "local"}:
        raise DecisionError(f"Invalid glance scope: {scope!r}")
    return GlanceDecision(scope=scope, reason=str(payload.get("reason", "")).strip())


def parse_selection(text: str, valid_window_ids: set[str]) -> SelectionDecision:
    payload = parse_json_object(text)
    raw_ids = payload.get("window_ids")
    if not isinstance(raw_ids, list):
        raise DecisionError("Selection field 'window_ids' must be a list")
    window_ids: list[str] = []
    for value in raw_ids:
        window_id = str(value).strip()
        if window_id in valid_window_ids and window_id not in window_ids:
            window_ids.append(window_id)
    if not window_ids:
        raise DecisionError("Selection contains no valid window IDs")
    return SelectionDecision(
        window_ids=tuple(window_ids),
        rationale=str(payload.get("rationale", "")).strip(),
        evidence_needed=str(payload.get("evidence_needed", "")).strip(),
    )


def parse_reasoning(text: str) -> ReasoningDecision:
    payload = parse_json_object(text)
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise DecisionError("Reasoning response has no answer")
    try:
        confidence = int(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise DecisionError("Reasoning confidence must be an integer") from exc
    if not 0 <= confidence <= 3:
        raise DecisionError("Reasoning confidence must be between 0 and 3")
    raw_evidence = payload.get("evidence_frame_ids", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    return ReasoningDecision(
        answer=answer,
        reason=str(payload.get("reason", "")).strip(),
        confidence=confidence,
        evidence_frame_ids=tuple(str(item) for item in raw_evidence),
        missing_evidence=str(payload.get("missing_evidence", "")).strip(),
    )


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```")
        candidate = candidate.removesuffix("```").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise DecisionError(f"Model did not return valid JSON: {text!r}")
    if not isinstance(value, dict):
        raise DecisionError("Model response must be a JSON object")
    return value
