from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from qwen3vl_agent.coarse_to_fine.adapters import (
    AnswerAdapter,
    AnswerRecord,
    answer_record_to_dict,
    build_answer_adapter,
)
from qwen3vl_agent.coarse_to_fine.cache import (
    CachedVideo,
    SubtitleTrack,
    VideoEvidenceCache,
    build_contact_sheet,
)
from qwen3vl_agent.coarse_to_fine.config import CoarseToFineConfig
from qwen3vl_agent.coarse_to_fine.prompts import (
    build_direct_prompt,
    build_glance_prompt,
    build_protocol_repair_prompt,
    build_reasoning_prompt,
    build_selection_prompt,
    parse_glance,
    parse_reasoning,
    parse_selection,
)
from qwen3vl_agent.coarse_to_fine.types import (
    BudgetExhausted,
    DecisionError,
    EvidenceMemory,
    FrameBudget,
    TimeWindow,
    partition_window,
)
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput

logger = logging.getLogger(__name__)

_DecisionT = TypeVar("_DecisionT")

_EXPLICIT_GLOBAL_PATTERNS = (
    re.compile(
        r"\b(?:main|primary|overall)\s+(?:topic|theme|focus|purpose)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:mainly|primarily)\s+about\b", re.IGNORECASE),
    re.compile(
        r"\b(?:not|never)\b.{0,60}\b(?:shown|seen|appear(?:s|ed)?|reported)\b"
        r"|\b(?:shown|seen|reported)\b.{0,60}\bnot\b",
        re.IGNORECASE,
    ),
)


def _explicit_global_reason(question: str) -> str | None:
    """Return a narrow, auditable override for questions that require whole-video coverage."""
    for pattern in _EXPLICIT_GLOBAL_PATTERNS:
        if pattern.search(question):
            return "explicit whole-video question pattern"
    return None


class CoarseToFineVideoAgent:
    """A deterministic, traceable A1-inspired temporal search controller."""

    def __init__(
        self,
        model: BaseVideoModel,
        *,
        config: CoarseToFineConfig | Mapping[str, Any] | None = None,
        cache: VideoEvidenceCache | None = None,
    ) -> None:
        self.model = model
        self.config = (
            config
            if isinstance(config, CoarseToFineConfig)
            else CoarseToFineConfig.from_mapping(config)
        )
        self.config.validate()
        self.cache = cache or VideoEvidenceCache(
            self.config.cache_dir,
            sample_fps=self.config.sample_fps,
            max_side=self.config.cache_max_side,
            jpeg_quality=self.config.cache_jpeg_quality,
            lru_size=self.config.cache_lru_size,
        )
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.model.load()
        self._loaded = True

    def unload(self) -> None:
        self.model.unload()
        self._loaded = False

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[str] | None = None,
        images: list[str] | None = None,
        choices: Sequence[str] | None = None,
        subtitle_path: str | None = None,
        **_: Any,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Agent is not loaded. Call load() first.")
        if images:
            raise ValueError("coarse_to_fine strategy currently accepts video input only")
        if not videos or len(videos) != 1:
            raise ValueError("coarse_to_fine strategy requires exactly one video")

        question = self._latest_user_text(messages)
        if not question:
            raise ValueError("messages must contain a non-empty user question")
        video_path = videos[0]
        adapter = build_answer_adapter(choices)
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "strategy": "coarse_to_fine",
            "question": question,
            "choices": list(choices or ()),
            "video_path": video_path,
            "subtitle_path": subtitle_path,
            "degraded": False,
            "protocol_retries": [],
            "rounds": [],
        }
        cached: CachedVideo | None = None
        subtitles: SubtitleTrack | None = None
        try:
            cached = self.cache.prepare(video_path)
            trace["cache"] = cached.to_dict()
            if subtitle_path:
                subtitles = SubtitleTrack.from_srt(subtitle_path)
                trace["subtitle_cues"] = len(subtitles.cues)
            result = self._run(question, adapter, cached, subtitles, trace)
        except Exception as exc:  # noqa: BLE001 - configured behavior is explicit degradation
            logger.warning("Coarse-to-fine failed; degrading to direct inference: %s", exc)
            trace["degraded"] = True
            trace["degraded_reason"] = f"{type(exc).__name__}: {exc}"
            result = self._fallback(question, adapter, video_path, cached, subtitles, trace)
        result.metadata["wall_seconds"] = time.perf_counter() - started
        return result

    def _run(
        self,
        question: str,
        adapter: AnswerAdapter,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        trace: dict[str, Any],
    ) -> ModelOutput:
        config = self.config
        budget = FrameBudget(config.unique_frame_budget, config.cumulative_frame_budget)
        memory = EvidenceMemory(config.unique_frame_budget)
        whole = TimeWindow("ROOT", 0.0, cached.duration_seconds, depth=0)

        glance_frames = budget.consume(
            cached.uniform_frames(config.glance_frames),
            purpose="video_glance",
            min_count=min(2, config.glance_frames),
        )
        memory.add(glance_frames)
        glance_output = self.model.generate(
            [{"role": "user", "content": build_glance_prompt(question, adapter)}],
            videos=[[frame.path for frame in glance_frames]],
            max_new_tokens=config.decision_max_new_tokens,
            temperature=0.0,
        )
        glance = self._parse_with_protocol_repair(
            glance_output,
            parser=parse_glance,
            protocol="glance",
            schema_hint='{"scope":"local"}; scope must be local or global',
            trace=trace,
        )
        controller_override = _explicit_global_reason(question)
        scope = "global" if controller_override else glance.scope
        trace["glance"] = {
            "scope": scope,
            "model_scope": glance.scope,
            "controller_override": controller_override,
            "reason": glance.reason,
            "frames": [frame.to_dict() for frame in glance_frames],
            "raw_response": glance_output.text,
            "model_metadata": glance_output.metadata,
        }
        trace["scope"] = scope

        if scope == "global":
            output = self._global_answer(
                question,
                adapter,
                cached,
                subtitles,
                whole,
                budget,
                trace,
            )
            trace["rounds"].append(output[1])
            trace["stop_reason"] = "global_direct"
            trace["budget"] = budget.to_dict()
            trace["evidence_memory"] = memory.to_list()
            return ModelOutput(output[0], {"coarse_to_fine": trace})

        candidates = partition_window(
            whole,
            parts=config.initial_windows,
            round_index=1,
        )
        history: list[dict[str, Any]] = []
        answers: list[AnswerRecord] = []
        final_answer: str | None = None
        stop_reason = "max_rounds_vote"

        for round_index in range(1, config.max_rounds + 1):
            if not candidates:
                stop_reason = "no_splittable_windows"
                break
            try:
                round_trace, selected, record = self._search_round(
                    question,
                    adapter,
                    cached,
                    subtitles,
                    candidates,
                    history,
                    memory,
                    budget,
                    round_index,
                    trace,
                )
            except BudgetExhausted:
                stop_reason = "frame_budget_exhausted"
                break
            trace["rounds"].append(round_trace)
            answers.append(record)
            history.append(
                {
                    "round": round_index,
                    "answer": record.answer,
                    "confidence": record.confidence,
                    "reason": round_trace["reasoning"]["reason"],
                    "missing_evidence": round_trace["reasoning"]["missing_evidence"],
                    "selected_windows": [window.to_dict() for window in selected],
                }
            )
            if record.confidence >= config.confidence_threshold:
                final_answer = record.answer
                stop_reason = "confidence_threshold"
                break
            if round_index == config.max_rounds:
                break
            if any(window.duration_seconds / config.split_parts < config.min_window_seconds for window in selected):
                stop_reason = "minimum_window_duration"
                break
            candidates = self._split_selected(selected, round_index + 1)

        if final_answer is None:
            if not answers:
                raise BudgetExhausted("No reasoning round fit inside the frame budget")
            final_answer = adapter.vote(answers)
        trace["answers"] = [answer_record_to_dict(record) for record in answers]
        trace["final_answer"] = final_answer
        trace["stop_reason"] = stop_reason
        trace["budget"] = budget.to_dict()
        trace["evidence_memory"] = memory.to_list()
        return ModelOutput(final_answer, {"coarse_to_fine": trace})

    def _global_answer(
        self,
        question: str,
        adapter: AnswerAdapter,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        whole: TimeWindow,
        budget: FrameBudget,
        trace: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        frames = budget.consume(
            cached.uniform_frames(self.config.global_frames),
            purpose="global_answer",
            min_count=2,
        )
        subtitle_text = self._subtitle_text(subtitles, [whole])
        prompt = build_reasoning_prompt(
            question,
            adapter,
            [whole],
            frames,
            subtitles=subtitle_text,
            history=[],
            round_index=0,
        )
        output = self.model.generate(
            [{"role": "user", "content": prompt}],
            videos=[[frame.path for frame in frames]],
            max_new_tokens=self.config.reasoning_max_new_tokens,
            temperature=0.0,
        )
        decision = self._parse_with_protocol_repair(
            output,
            parser=parse_reasoning,
            protocol="global_reasoning",
            schema_hint=(
                '{"answer":"A","reason":"brief","confidence":0,'
                '"missing_evidence":"format repair"}'
            ),
            trace=trace,
        )
        answer = adapter.normalize(decision.answer)
        if answer is None:
            raise ValueError(f"Invalid task answer: {decision.answer!r}")
        return answer, {
            "round": 0,
            "mode": "global",
            "selected_windows": [whole.to_dict()],
            "frames": [frame.to_dict() for frame in frames],
            "subtitles": subtitle_text,
            "reasoning": {
                "answer": answer,
                "raw_answer": decision.answer,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "evidence_frame_ids": list(decision.evidence_frame_ids),
                "missing_evidence": decision.missing_evidence,
                "raw_response": output.text,
                "model_metadata": output.metadata,
            },
        }

    def _search_round(
        self,
        question: str,
        adapter: AnswerAdapter,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        candidates: list[TimeWindow],
        history: list[dict[str, Any]],
        memory: EvidenceMemory,
        budget: FrameBudget,
        round_index: int,
        trace: dict[str, Any],
    ) -> tuple[dict[str, Any], list[TimeWindow], AnswerRecord]:
        representatives = [cached.nearest_frame(window.midpoint_seconds) for window in candidates]
        representatives = budget.consume(
            representatives,
            purpose=f"round_{round_index}_window_selection",
            min_count=len(candidates),
        )
        labels = [
            f"{window.id}  {window.start_seconds:.1f}-{window.end_seconds:.1f}s"
            for window in candidates
        ]
        contact_sheet = build_contact_sheet(
            representatives,
            labels,
            output_dir=f"{cached.cache_dir}/contacts",
            columns=self.config.contact_sheet_columns,
        )
        candidate_subtitles = self._candidate_subtitles(subtitles, candidates)
        selection_output = self.model.generate(
            [
                {
                    "role": "user",
                    "content": build_selection_prompt(
                        question,
                        adapter,
                        candidates,
                        top_k=self.config.select_top_k,
                        history=history,
                        candidate_subtitles=candidate_subtitles,
                    ),
                }
            ],
            images=[contact_sheet],
            max_new_tokens=self.config.decision_max_new_tokens,
            temperature=0.0,
        )
        valid_window_ids = {window.id for window in candidates}
        selection = self._parse_with_protocol_repair(
            selection_output,
            parser=lambda text: parse_selection(text, valid_window_ids),
            protocol=f"round_{round_index}_selection",
            schema_hint=(
                '{"window_ids":["window_id"]}; allowed IDs: '
                + ", ".join(sorted(valid_window_ids))
            ),
            trace=trace,
        )
        chosen_ids = set(selection.window_ids[: self.config.select_top_k])
        selected = sorted(
            (window for window in candidates if window.id in chosen_ids),
            key=lambda window: window.start_seconds,
        )
        if not selected:
            raise ValueError("Window selector returned no usable candidates")

        representative_by_id = dict(zip((window.id for window in candidates), representatives))
        memory.add(representative_by_id[window.id] for window in selected)
        preferred = cached.frames_for_windows(
            selected,
            total=self.config.working_set_frames,
        )
        working_set = sorted(
            memory.working_set(preferred, limit=self.config.working_set_frames),
            key=lambda frame: frame.timestamp_seconds,
        )
        working_set = budget.consume(
            working_set,
            purpose=f"round_{round_index}_reasoning",
            min_count=2,
        )
        memory.add(working_set)
        subtitle_text = self._subtitle_text(subtitles, selected)
        reasoning_output = self.model.generate(
            [
                {
                    "role": "user",
                    "content": build_reasoning_prompt(
                        question,
                        adapter,
                        selected,
                        working_set,
                        subtitles=subtitle_text,
                        history=history,
                        round_index=round_index,
                    ),
                }
            ],
            videos=[[frame.path for frame in working_set]],
            max_new_tokens=self.config.reasoning_max_new_tokens,
            temperature=0.0,
        )
        reasoning = self._parse_with_protocol_repair(
            reasoning_output,
            parser=parse_reasoning,
            protocol=f"round_{round_index}_reasoning",
            schema_hint=(
                '{"answer":"A","reason":"brief","confidence":0,'
                '"missing_evidence":"format repair"}'
            ),
            trace=trace,
        )
        answer = adapter.normalize(reasoning.answer)
        if answer is None:
            raise ValueError(f"Invalid task answer: {reasoning.answer!r}")
        record = AnswerRecord(answer, reasoning.confidence, round_index)
        round_trace = {
            "round": round_index,
            "mode": "local",
            "candidates": [window.to_dict() for window in candidates],
            "contact_sheet": contact_sheet,
            "candidate_subtitles": candidate_subtitles,
            "selection_frames": [frame.to_dict() for frame in representatives],
            "selection": {
                "window_ids": list(selection.window_ids),
                "rationale": selection.rationale,
                "evidence_needed": selection.evidence_needed,
                "raw_response": selection_output.text,
                "model_metadata": selection_output.metadata,
            },
            "selected_windows": [window.to_dict() for window in selected],
            "frames": [frame.to_dict() for frame in working_set],
            "subtitles": subtitle_text,
            "reasoning": {
                "answer": answer,
                "raw_answer": reasoning.answer,
                "reason": reasoning.reason,
                "confidence": reasoning.confidence,
                "evidence_frame_ids": list(reasoning.evidence_frame_ids),
                "missing_evidence": reasoning.missing_evidence,
                "raw_response": reasoning_output.text,
                "model_metadata": reasoning_output.metadata,
            },
        }
        return round_trace, selected, record

    def _split_selected(
        self,
        selected: list[TimeWindow],
        next_round: int,
    ) -> list[TimeWindow]:
        children: list[TimeWindow] = []
        for window in selected:
            children.extend(
                partition_window(
                    window,
                    parts=self.config.split_parts,
                    round_index=next_round,
                    id_offset=len(children),
                )
            )
        return children

    def _subtitle_text(
        self,
        subtitles: SubtitleTrack | None,
        windows: list[TimeWindow],
    ) -> str:
        if subtitles is None:
            return ""
        return subtitles.text_for_windows(
            windows,
            max_chars=self.config.subtitle_max_chars,
        )

    def _candidate_subtitles(
        self,
        subtitles: SubtitleTrack | None,
        windows: list[TimeWindow],
    ) -> dict[str, str]:
        if subtitles is None or not windows:
            return {}
        per_window = max(120, self.config.subtitle_max_chars // len(windows))
        return {
            window.id: subtitles.text_for_windows([window], max_chars=per_window)
            for window in windows
        }

    def _parse_with_protocol_repair(
        self,
        initial_output: ModelOutput,
        *,
        parser: Callable[[str], _DecisionT],
        protocol: str,
        schema_hint: str,
        trace: dict[str, Any],
    ) -> _DecisionT:
        try:
            return parser(initial_output.text)
        except DecisionError as initial_error:
            last_error = initial_error

        candidate = initial_output.text
        for attempt in range(1, self.config.protocol_repair_attempts + 1):
            repair_output = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": build_protocol_repair_prompt(
                            protocol,
                            candidate,
                            schema_hint=schema_hint,
                        ),
                    }
                ],
                max_new_tokens=self.config.protocol_repair_max_new_tokens,
                temperature=0.0,
            )
            event = {
                "protocol": protocol,
                "attempt": attempt,
                "source_response": candidate,
                "source_error": f"{type(last_error).__name__}: {last_error}",
                "raw_response": repair_output.text,
                "model_metadata": repair_output.metadata,
            }
            try:
                decision = parser(repair_output.text)
            except DecisionError as repair_error:
                event["status"] = "failed"
                event["repair_error"] = f"{type(repair_error).__name__}: {repair_error}"
                trace["protocol_retries"].append(event)
                candidate = repair_output.text
                last_error = repair_error
                continue
            event["status"] = "success"
            trace["protocol_retries"].append(event)
            return decision

        raise DecisionError(
            f"{protocol} response remained invalid after "
            f"{self.config.protocol_repair_attempts} repair attempt(s): {last_error}"
        ) from last_error

    def _fallback(
        self,
        question: str,
        adapter: AnswerAdapter,
        video_path: str,
        cached: CachedVideo | None,
        subtitles: SubtitleTrack | None,
        trace: dict[str, Any],
    ) -> ModelOutput:
        if cached is not None:
            whole = TimeWindow("ROOT", 0.0, cached.duration_seconds, depth=0)
            subtitle_text = self._subtitle_text(subtitles, [whole])
            frames = cached.uniform_frames(self.config.global_frames)
            output = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": build_direct_prompt(
                            question,
                            adapter,
                            subtitles=subtitle_text,
                        ),
                    }
                ],
                videos=[[frame.path for frame in frames]],
            )
            trace["fallback_frames"] = [frame.to_dict() for frame in frames]
            trace["fallback_subtitles"] = subtitle_text
        else:
            output = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": build_direct_prompt(question, adapter),
                    }
                ],
                videos=[video_path],
            )
        normalized = adapter.normalize(output.text)
        answer = normalized if normalized is not None else output.text
        trace["stop_reason"] = "degraded_direct_fallback"
        trace["fallback_model_metadata"] = output.metadata
        return ModelOutput(answer, {"coarse_to_fine": trace})

    @staticmethod
    def _latest_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping) and part.get("type") == "text"
                ).strip()
        return ""
