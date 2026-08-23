from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qwen3vl_agent.coarse_to_fine.types import FrameRef, TimeWindow

TOPOLOGIES = ("local", "sequence", "multi_set", "global", "exclusion")
ACTION_KINDS = ("expand", "zoom_out", "shift", "observe", "compare", "verify", "answer")
OBSERVATION_MODES = (
    "overview",
    "inspect",
    "motion",
    "event_verify",
    "detail_ocr",
    "subtitle",
)
MODALITIES = ("visual", "subtitle", "ocr")


class ActiveTreeError(RuntimeError):
    """Base error for active-tree protocol and controller failures."""


class ProtocolError(ActiveTreeError):
    """Raised when a model role returns an invalid structured response."""


class ModelCallLimit(ActiveTreeError):
    """Raised when the controller reaches its hard safety call limit."""


@dataclass
class SceneNode:
    id: str
    start_seconds: float
    end_seconds: float
    level: int
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    peak_timestamp_seconds: float | None = None
    boundary_score: float = 0.0

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("scene start must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("scene end must exceed scene start")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def midpoint_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids

    def as_window(self) -> TimeWindow:
        return TimeWindow(
            self.id,
            self.start_seconds,
            self.end_seconds,
            depth=self.level,
            parent_id=self.parent_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "level": self.level,
            "parent_id": self.parent_id,
            "child_ids": list(self.child_ids),
            "peak_timestamp_seconds": (
                round(self.peak_timestamp_seconds, 3)
                if self.peak_timestamp_seconds is not None
                else None
            ),
            "boundary_score": round(self.boundary_score, 6),
            "is_leaf": self.is_leaf,
        }


@dataclass
class SceneTree:
    root_id: str
    nodes: dict[str, SceneNode]
    index_path: str | None = None
    cache_hit: bool = False

    @property
    def root(self) -> SceneNode:
        return self.nodes[self.root_id]

    def node(self, node_id: str) -> SceneNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scene node: {node_id}") from exc

    def children(self, node_id: str) -> list[SceneNode]:
        return [self.nodes[item] for item in self.node(node_id).child_ids]

    def siblings(self, node_id: str) -> list[SceneNode]:
        node = self.node(node_id)
        if node.parent_id is None:
            return []
        return [item for item in self.children(node.parent_id) if item.id != node_id]

    def ancestors(self, node_id: str) -> list[SceneNode]:
        result: list[SceneNode] = []
        current = self.node(node_id)
        while current.parent_id is not None:
            current = self.node(current.parent_id)
            result.append(current)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "node_count": len(self.nodes),
            "leaf_count": sum(node.is_leaf for node in self.nodes.values()),
            "index_path": self.index_path,
            "cache_hit": self.cache_hit,
            "nodes": [
                node.to_dict()
                for node in sorted(
                    self.nodes.values(),
                    key=lambda item: (item.start_seconds, item.level, item.id),
                )
            ],
        }


@dataclass(frozen=True)
class CanonicalOption:
    option_id: str
    benchmark_label: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "option_id": self.option_id,
            "benchmark_label": self.benchmark_label,
            "text": self.text,
        }


@dataclass(frozen=True)
class EvidenceSlot:
    slot_id: str
    description: str
    required: bool = True
    constraint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "description": self.description,
            "required": self.required,
            "constraint": self.constraint,
        }


@dataclass(frozen=True)
class OptionTest:
    option_id: str
    support_test: str
    refute_test: str

    def to_dict(self) -> dict[str, str]:
        return {
            "option_id": self.option_id,
            "support_test": self.support_test,
            "refute_test": self.refute_test,
        }


@dataclass
class TaskContract:
    primary_topology: str
    slots: list[EvidenceSlot]
    required_modalities: list[str] = field(default_factory=lambda: ["visual"])
    option_tests: list[OptionTest] = field(default_factory=list)
    answer_criterion: str = "direct_support"

    def required_slot_ids(self) -> set[str]:
        return {slot.slot_id for slot in self.slots if slot.required}

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_topology": self.primary_topology,
            "required_modalities": list(self.required_modalities),
            "answer_criterion": self.answer_criterion,
            "slots": [slot.to_dict() for slot in self.slots],
            "option_tests": [test.to_dict() for test in self.option_tests],
        }


@dataclass(frozen=True)
class PlannedAction:
    kind: str
    node_id: str | None = None
    mode: str | None = None
    slot_id: str | None = None
    compare_node_ids: tuple[str, ...] = ()
    expected_new_evidence: str = ""

    def signature(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.node_id,
            self.mode,
            self.slot_id,
            self.compare_node_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "node_id": self.node_id,
            "mode": self.mode,
            "slot_id": self.slot_id,
            "compare_node_ids": list(self.compare_node_ids),
            "expected_new_evidence": self.expected_new_evidence,
        }


@dataclass(frozen=True)
class AtomicEvidence:
    evidence_id: str
    node_id: str
    slot_ids: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    modality: str
    fact: str
    supports_option_ids: tuple[str, ...]
    refutes_option_ids: tuple[str, ...]
    source_frame_ids: tuple[str, ...]
    subtitle_refs: tuple[str, ...]
    observation_mode: str

    def key(self) -> tuple[Any, ...]:
        return (
            self.node_id,
            tuple(sorted(self.slot_ids)),
            round(self.start_seconds, 1),
            round(self.end_seconds, 1),
            self.modality,
            self.fact.casefold().strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "node_id": self.node_id,
            "slot_ids": list(self.slot_ids),
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "modality": self.modality,
            "fact": self.fact,
            "supports_option_ids": list(self.supports_option_ids),
            "refutes_option_ids": list(self.refutes_option_ids),
            "source_frame_ids": list(self.source_frame_ids),
            "subtitle_refs": list(self.subtitle_refs),
            "observation_mode": self.observation_mode,
        }


class EvidenceLedger:
    """Structured, de-duplicated evidence memory with explicit slot coverage."""

    def __init__(self) -> None:
        self._items: list[AtomicEvidence] = []
        self._keys: set[tuple[Any, ...]] = set()

    @property
    def items(self) -> tuple[AtomicEvidence, ...]:
        return tuple(self._items)

    @property
    def active_items(self) -> tuple[AtomicEvidence, ...]:
        """Evidence collected after routing-only breadth inspection."""

        return tuple(
            item
            for item in self._items
            if item.observation_mode != "breadth"
            and not item.observation_mode.endswith("_routing")
        )

    @property
    def routing_items(self) -> tuple[AtomicEvidence, ...]:
        """Non-decisive clues retained only to choose a finer observation."""

        return tuple(
            item
            for item in self._items
            if item.observation_mode.endswith("_routing")
            and item.observation_mode != "breadth_routing"
        )

    def add(self, items: list[AtomicEvidence]) -> int:
        added = 0
        for item in items:
            key = item.key()
            if key in self._keys:
                continue
            self._keys.add(key)
            self._items.append(item)
            added += 1
        return added

    def covered_slot_ids(self) -> set[str]:
        return {slot_id for item in self.active_items for slot_id in item.slot_ids}

    def covered_modalities(self) -> set[str]:
        return {item.modality for item in self.active_items}

    def missing_slot_ids(self, contract: TaskContract) -> list[str]:
        return sorted(contract.required_slot_ids() - self.covered_slot_ids())

    def facts_for_nodes(self, node_ids: set[str]) -> list[AtomicEvidence]:
        return [item for item in self._items if item.node_id in node_ids]

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def compact_text(
        self,
        *,
        max_items: int = 24,
        include_breadth: bool = True,
    ) -> str:
        items = self._items if include_breadth else list(self.active_items)
        if not items:
            return "(no grounded evidence yet)"
        lines: list[str] = []
        for item in items[-max_items:]:
            support = ",".join(item.supports_option_ids) or "-"
            refute = ",".join(item.refutes_option_ids) or "-"
            slots = ",".join(item.slot_ids) or "-"
            lines.append(
                f"{item.evidence_id} [{item.start_seconds:.1f}-{item.end_seconds:.1f}s] "
                f"node={item.node_id} modality={item.modality} slots={slots} "
                f"support={support} refute={refute}: {item.fact}"
            )
        return "\n".join(lines)


@dataclass
class ResourceLedger:
    max_model_calls: int
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    unique_frame_ids: set[str] = field(default_factory=set)
    cumulative_frame_views: int = 0

    @property
    def remaining_model_calls(self) -> int:
        return max(0, self.max_model_calls - len(self.model_calls))

    def ensure_call_available(self) -> None:
        if len(self.model_calls) >= self.max_model_calls:
            raise ModelCallLimit(f"model call safety limit reached: {self.max_model_calls}")

    def record_call(
        self,
        *,
        role: str,
        prompt: str,
        raw_response: str,
        model_metadata: dict[str, Any],
        frames: list[FrameRef] | None = None,
        images: list[str] | None = None,
    ) -> int:
        shown = frames or []
        new_ids = {frame.id for frame in shown} - self.unique_frame_ids
        self.unique_frame_ids.update(frame.id for frame in shown)
        self.cumulative_frame_views += len(shown)
        record = {
            "call_index": len(self.model_calls) + 1,
            "role": role,
            "prompt": prompt,
            "raw_response": raw_response,
            "model_metadata": model_metadata,
            "frame_ids": [frame.id for frame in shown],
            "image_paths": list(images or ()),
            "new_unique_frames": len(new_ids),
        }
        self.model_calls.append(record)
        return int(record["call_index"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "model_call_count": len(self.model_calls),
            "unique_frames": len(self.unique_frame_ids),
            "cumulative_frame_views": self.cumulative_frame_views,
            "input_tokens": sum(
                int(item["model_metadata"].get("input_tokens", 0))
                for item in self.model_calls
            ),
            "output_tokens": sum(
                int(item["model_metadata"].get("output_tokens", 0))
                for item in self.model_calls
            ),
            "calls": list(self.model_calls),
        }
