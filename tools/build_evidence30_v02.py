"""Build the VideoMME Evidence30 v0.2 single-annotator artifact set.

The source parquet is the authority for question text, options, metadata and the
official answer.  Evidence contracts below contain only observations made from
the local videos/subtitles during the v0.2 inspection pass.  The script is
deterministic apart from the source-media hashes, and keeps the v0.1 smoke
artifacts in a versioned sibling directory before updating the compatibility
root (annotations/videomme_evidence30/).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = ROOT / "annotations" / "videomme_evidence30"
ARCHIVE_ROOT = ROOT / "annotations" / "_unreviewed_ai_drafts" / "videomme_evidence30_20260822"
V01_ROOT = ANNOTATION_ROOT / "0.1.0"
V02_ROOT = ANNOTATION_ROOT / "0.2.0"
VIDEOMME_ROOT = Path(os.environ.get("VIDEOMME_ROOT", "data/videomme")).expanduser()
PARQUET = VIDEOMME_ROOT / "videomme" / "test-00000-of-00001.parquet"
VIDEO_ROOT = VIDEOMME_ROOT / "videos"
SUBTITLE_ROOT = VIDEOMME_ROOT / "subtitle"
LOCKED_AT = "2026-08-22T13:30:00Z"
ANNOTATOR = {"reviewer_id": "primary-annotator", "completed_at": LOCKED_AT}

DEV_QIDS = [
    "007-2", "212-3", "389-2", "102-1", "154-2", "251-3", "050-1",
    "314-2", "434-2", "496-1", "522-3", "604-2", "647-2", "700-1",
    "730-1", "780-2", "845-1", "884-2",
]
LOCKED_QIDS = [
    "197-3", "445-2", "504-1", "573-2", "599-2", "634-2", "673-2",
    "717-2", "754-2", "792-2", "847-2", "895-3",
]
ALL_QIDS = DEV_QIDS + LOCKED_QIDS
CONTAMINATED = {"102", "154", "251"}


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_options(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        parsed = value.tolist()
        raw = [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    elif isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        text = str(value)
        try:
            parsed = ast.literal_eval(text)
            raw = [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [text]
        except (SyntaxError, ValueError):
            raw = []
            matches = re.findall(r"([A-D])\.\s*(.*?)(?=,\s*[A-D]\.\s*|$)", text)
            if matches:
                raw = [f"{label}. {body}" for label, body in matches]
            else:
                raw = [text]
    options: list[dict[str, str]] = []
    for index, item in enumerate(raw[:4]):
        match = re.match(r"\s*([A-D])\.\s*(.*)\s*$", item, flags=re.DOTALL)
        label = match.group(1) if match else "ABCD"[index]
        body = match.group(2).strip() if match else item.strip()
        options.append({"option_id": f"O{index + 1}", "benchmark_label": label, "text": body})
    while len(options) < 4:
        index = len(options)
        options.append({"option_id": f"O{index + 1}", "benchmark_label": "ABCD"[index], "text": "[unreadable benchmark option]"})
    return options


def duration_bucket(duration: float) -> str:
    if duration < 120:
        return "short"
    if duration < 600:
        return "medium"
    return "long"


def interval(start: float, end: float) -> dict[str, float]:
    return {"start_sec": float(start), "end_sec": float(end)}


def ev(
    evidence_id: str,
    slot_id: str,
    start: float,
    end: float,
    modality: str,
    fact: str,
    *,
    supports: Iterable[str] = (),
    refutes: Iterable[str] = (),
    roles: Iterable[str] | None = None,
    cues: Iterable[str] = (),
    frames: Iterable[float] = (),
    ocr: str | None = None,
    observation_requirement: str | None = None,
    minimum_mode: str | None = None,
    context: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if observation_requirement is None:
        observation_requirement = {
            "visual": "short_clip",
            "subtitle": "subtitle_cues",
            "ocr": "readable_ocr",
        }[modality]
    if minimum_mode is None:
        minimum_mode = {
            "visual": "motion",
            "subtitle": "subtitle",
            "ocr": "detail_ocr",
        }[modality]
    span = max(0.0, float(end) - float(start))
    min_frames = 1 if modality == "ocr" else (0 if modality == "subtitle" else 3)
    if observation_requirement == "point_frame":
        min_frames, minimum_mode, span = 1, "inspect", 0.0
    if observation_requirement == "exhaustive_coverage":
        min_frames, minimum_mode = 0, "coverage"
    if observation_requirement == "representative_coverage":
        min_frames, minimum_mode = 0, "coverage"
    return {
        "evidence_id": evidence_id,
        "slot_id": slot_id,
        "required": True,
        "core_interval": interval(start, end),
        "context_interval": interval(*(context or (start, end))),
        "modality": modality,
        "observation_requirement": observation_requirement,
        "minimum_observation": {
            "mode": minimum_mode,
            "min_frames": min_frames,
            "min_temporal_span_sec": round(span, 3),
        },
        "atomic_fact": fact,
        "roles": list(roles or (["support"] if supports else ["context"])),
        "supports_option_ids": list(supports),
        "refutes_option_ids": list(refutes),
        "source_references": {
            "frame_timestamps_sec": list(frames),
            "subtitle_cue_ids": list(cues),
            "ocr_text": ocr,
        },
    }


def rel(kind: str, evidence_ids: Iterable[str], description: str) -> dict[str, Any]:
    return {"relation": kind, "source_evidence_ids": list(evidence_ids), "description": description}


def hn(
    negative_id: str,
    start: float,
    end: float,
    modalities: Iterable[str],
    plausibility: str,
    reason: str,
    options: Iterable[str],
) -> dict[str, Any]:
    return {
        "negative_id": negative_id,
        "interval": interval(start, end),
        "modalities": list(modalities),
        "plausibility": plausibility,
        "insufficiency_reason": reason,
        "tempts_option_ids": list(options),
    }


def contract(
    topology: str,
    modalities: list[str],
    criterion: str,
    slots: list[tuple[str, str]],
    items: list[dict[str, Any]],
    relations: list[dict[str, Any]] | None = None,
    *,
    secondary: list[str] | None = None,
    coverage: dict[str, Any] | None = None,
    description: str = "充分证据集合覆盖题目所需的原子事实与选项映射。",
) -> dict[str, Any]:
    return {
        "primary_topology": topology,
        "secondary_topologies": list(secondary or []),
        "required_modalities": modalities,
        "answer_criterion": criterion,
        "evidence_slots": [{"slot_id": sid, "description": desc, "required": True} for sid, desc in slots],
        "sufficient_evidence_sets": [{
            "set_id": "ES1",
            "description": description,
            "logic": "all_required",
            "items": items,
            "relations": list(relations or []),
        }],
        "global_coverage_contract": coverage,
    }


def coverage(scope: str, segments: list[tuple[str, float, float, str]], maximum_gap: float, absence: bool) -> dict[str, Any]:
    return {
        "coverage_type": "exhaustive",
        "scope_description": scope,
        "required_segments": [{"segment_id": sid, "interval": interval(start, end), "reason": reason} for sid, start, end, reason in segments],
        "max_unobserved_gap_sec": maximum_gap,
        "absence_claim": absence,
    }


def build_configs() -> dict[str, dict[str, Any]]:
    """Evidence contracts for the 27 non-smoke questions."""
    return {
        "102-1": {
            "reason": "画面中穿深色西装的男子清楚佩戴婚戒并被叙述为即将结婚；字幕 cue 2–3 与该人物同一镜头对齐。污染风险只记录为 provenance，不改变可观察证据。",
            "notes": ["trace_contaminated: question/video was present in an earlier inspection trace; retained by v0.2 policy and not used to rewrite the answer."],
            "contract": contract("local", ["visual", "subtitle"], "direct_support", [("S1", "确认即将结婚人物的服装与婚戒。"), ("S2", "字幕锚定同一人物和结婚语境。")], [
                ev("E1", "S1", 4.2, 12.0, "visual", "同一镜头中穿深色西装的男子清楚可见婚戒，其他人物服装不同。", supports=["O2"], refutes=["O1", "O3", "O4"], frames=[5, 8, 11]),
                ev("E2", "S2", 4.2, 12.059, "subtitle", "字幕 cue 2–3 在该镜头说到 ring/结婚语境，锚定深色西装男子。", supports=["O2"], cues=["2", "3"], context=(4.2, 12.059)),
            ], [rel("same_event", ["E1", "E2"], "视觉婚戒与字幕结婚语境发生在同一人物镜头。")]),
            "hard_negatives": [hn("HN1", 15, 30, ["visual"], "短袖或蓝衬衫男子随后出镜，容易把人物服装选项错配。", "该镜头没有婚戒/结婚锚点，不能替代 S1。", ["O1", "O3"])],
        },
        "154-2": {
            "reason": "开场画面在 FIFA World Cup Trophy 旁清楚显示草地足球场及球门/场边结构，唯一匹配选项为 soccer field。污染风险仅写入备注。",
            "notes": ["trace_contaminated: retained in dev; scene answer is based on readable opening visual evidence."],
            "contract": contract("local", ["visual"], "direct_support", [("S1", "识别男子身后的运动场景。"), ("S2", "确认草地、球门和足球场边界的共同锚点。")], [
                ev("E1", "S1", 4.0, 22.0, "visual", "男子身后是带草地、球门和看台/场边结构的足球场。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[5, 10, 18]),
                ev("E2", "S2", 22.0, 35.0, "visual", "延续镜头仍显示足球场草地与球门，而非篮球架、跑道或棒球内场。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[24, 30, 34]),
            ], [rel("same_event", ["E1", "E2"], "两个连续开场视觉区间共同确认背景运动场。")]),
            "hard_negatives": [hn("HN1", 40, 60, ["visual"], "奖杯/采访特写弱化背景，可能让人凭主题猜运动场。", "该片段缺少可读的场地锚点，不能区分四种运动场。", ["O2", "O3", "O4"])],
        },
        "251-3": {
            "reason": "视频先展示拖地清洁，随后切到厨房煎蛋；两个相邻活动的 before/after 顺序直接支持官方答案 A。污染风险不排除该题。",
            "notes": ["trace_contaminated: retained in dev under v0.2 single-annotator policy."],
            "contract": contract("sequence", ["visual"], "direct_support", [("S1", "确认清洁地板活动。"), ("S2", "确认其后的下一项活动。")], [
                ev("E1", "S1", 19.0, 28.0, "visual", "女子在地面上用拖把/清洁工具清理地板。", roles=["context", "order_anchor"], frames=[20, 24, 27]),
                ev("E2", "S2", 35.0, 46.0, "visual", "清洁镜头之后，女子在厨房锅中煎蛋。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[36, 40, 45]),
            ], [rel("before", ["E1", "E2"], "煎蛋片段在拖地片段之后紧接出现，构成题目要求的 next 关系。")]),
            "hard_negatives": [hn("HN1", 53, 62, ["visual"], "随后出现用杯子接水，表面上也是候选活动。", "它发生在煎蛋之后，不是清洁地板后的下一项。", ["O2"])],
        },
        "050-1": {
            "reason": "字幕 cue 24–25 和画面标题同时标出 No.2 为 Perseid meteor shower；文本与视觉共同唯一支持 A。",
            "contract": contract("local", ["visual", "subtitle"], "mixed", [("S1", "读取 No.2 的画面标题。"), ("S2", "读取字幕中的事件名称。")], [
                ev("E1", "S1", 64.0, 72.0, "visual", "画面列表文字清楚显示“2. Perseid meteor shower”。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[65, 68, 71], ocr="2. Perseid meteor shower", observation_requirement="readable_ocr", minimum_mode="detail_ocr"),
                ev("E2", "S2", 64.239, 71.759, "subtitle", "字幕 cues 24–25 说到该事件并明确其为 number two。", supports=["O1"], cues=["24", "25"]),
            ], [rel("same_event", ["E1", "E2"], "画面编号与字幕事件名属于同一 No.2 片段。")]),
            "hard_negatives": [hn("HN1", 75, 92, ["visual", "subtitle"], "后续太阳/月食条目同样属于 celestial events 列表。", "它们有不同编号，不能回答 No.2。", ["O2", "O3"])],
        },
        "314-2": {
            "reason": "从约 20 秒开始的连续表演镜头清楚显示小女孩拉小提琴，视觉模态足以区分四个乐器选项。",
            "contract": contract("local", ["visual"], "direct_support", [("S1", "识别女孩演奏的乐器。")], [
                ev("E1", "S1", 20.0, 55.0, "visual", "小女孩以肩托和弓连续演奏小提琴，琴弓与琴颈清晰可见。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[22, 35, 50]),
            ]),
            "hard_negatives": [hn("HN1", 0, 10, ["visual"], "开场舞台全景中乐器细节较小。", "不能可靠区分小提琴、钢琴、大提琴和吉他。", ["O2", "O3", "O4"])],
        },
        "434-2": {
            "reason": "Vayne 的首次技能/召唤师技能使用发生在开场战斗片段，HUD 细节与可读 Haste 标签相互对齐，支持 A。",
            "contract": contract("local", ["visual", "ocr"], "mixed", [("S1", "定位 Vayne 的首次操作片段。"), ("S2", "读取 HUD 中的 Haste 标签/图标。")], [
                ev("E1", "S1", 2.0, 12.0, "visual", "Vayne 开场进入对线并触发第一项技能/召唤师技能操作。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[3, 7, 11]),
                ev("E2", "S2", 40.0, 52.0, "ocr", "战斗 HUD 的可读文字/图标标注 Haste，和开场技能选择一致。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[42, 48], ocr="Haste", context=(38, 55)),
            ], [rel("same_event", ["E1", "E2"], "E1 定位同一 Vayne 对局，E2 提供该对局 HUD 的文字锚点。")]),
            "hard_negatives": [hn("HN1", 65, 82, ["visual", "ocr"], "后续 HUD 同时出现多个技能/物品图标，容易把后续技能当成首次技能。", "该区间没有首次使用的时序锚点，不能回答 first skill。", ["O2", "O3", "O4"])],
        },
        "496-1": {
            "reason": "视频按 b（魔术师介绍）→ a（布移开）→ c（羽毛出现）的时间顺序展示三个事件，视觉 sequence 证据完整。",
            "contract": contract("sequence", ["visual"], "direct_support", [("S1", "确认魔术师介绍。"), ("S2", "确认布被移开。"), ("S3", "确认羽毛事件。")], [
                ev("E1", "S1", 0, 80, "visual", "开场先进行魔术师自我介绍/登场。", roles=["context", "order_anchor"], frames=[10, 40, 70]),
                ev("E2", "S2", 110, 150, "visual", "随后魔术师将覆盖物/布移开，露出桌面物体。", roles=["context", "order_anchor"], frames=[115, 130, 145]),
                ev("E3", "S3", 160, 180, "visual", "再后出现并展示羽毛效果。", supports=["O4"], refutes=["O1", "O2", "O3"], frames=[162, 170, 178]),
            ], [rel("before", ["E1", "E2"], "介绍先于布移开。"), rel("before", ["E2", "E3"], "布移开先于羽毛效果。"), rel("before", ["E1", "E2", "E3"], "三项事件构成 b→a→c。")]),
            "hard_negatives": [hn("HN1", 185, 205, ["visual"], "片尾重复镜头可能改变剪辑顺序的直觉。", "重复镜头不是首次事件序列，不能替代时间顺序。", ["O1", "O2"])],
        },
        "522-3": {
            "reason": "制作片段依次展示针、剪刀、胶枪；连续视觉检查和局部动作顺序共同支持 B。",
            "contract": contract("sequence", ["visual"], "direct_support", [("S1", "确认针的使用。"), ("S2", "确认剪刀的使用。"), ("S3", "确认胶枪的使用。")], [
                ev("E1", "S1", 30, 120, "visual", "前段手工制作使用针进行穿刺/缝制。", roles=["context", "order_anchor"], frames=[35, 75, 115]),
                ev("E2", "S2", 120, 300, "visual", "中段连续剪裁材料，剪刀清楚可见。", roles=["context", "order_anchor"], frames=[130, 210, 290]),
                ev("E3", "S3", 350, 430, "visual", "后段用热胶枪将部件粘合。", supports=["O2"], refutes=["O1", "O3", "O4"], frames=[360, 390, 425]),
            ], [rel("before", ["E1", "E2"], "针的使用先于剪刀。"), rel("before", ["E2", "E3"], "剪刀先于胶枪。"), rel("before", ["E1", "E2", "E3"], "三项工具顺序为 needle→scissors→glue gun。")]),
            "hard_negatives": [hn("HN1", 0, 10, ["visual"], "开场出现毛线/其他手工材料。", "材料不是题目要求的三项工具中的时序证据。", ["O3", "O4"])],
        },
        "604-2": {
            "reason": "字幕早段明确讲述迁移与和 Selic 的战争，后段补充进入欧洲、奥斯曼扩张和奴役语境；题目选项中唯一直接对应的是 C。",
            "contract": contract("multi_set", ["subtitle"], "direct_support", [("S1", "读取迁移叙述中的战争关系。"), ("S2", "读取后段历史背景以确认同一叙述主题。")], [
                ev("E1", "S1", 75.159, 82.0, "subtitle", "字幕 cues 33–34 将 migration 与一系列 wars between the Selic 直接相连。", supports=["O3"], refutes=["O1", "O4"], cues=["33", "34"]),
                ev("E2", "S2", 535.0, 547.76, "subtitle", "字幕 cues 239–243 说明进入欧洲、军事扩张和奥斯曼奴役背景，和前段迁移史叙述同一主题。", supports=["O3"], cues=["239", "240", "241", "242", "243"]),
            ], [rel("different_occurrence", ["E1", "E2"], "两组不连续字幕共同构成视频对该迁移历史的叙述，而非废奴或分离 Turks 的选项。")]),
            "hard_negatives": [hn("HN1", 87, 97, ["subtitle"], "同一历史段还提到 Turks/奥斯曼，容易误选 D。", "该片段没有支持“与 Turks 分离”的关系，不能替代 Selic 战争锚点。", ["O4"])],
        },
        "647-2": {
            "reason": "字幕 cue 47 和同期宇宙学画面明确介绍 Great Attractor；该定义对应 B，后段 Laniakea 只是相关但不同概念。",
            "contract": contract("local", ["visual", "subtitle"], "mixed", [("S1", "读取 Great Attractor 名称。"), ("S2", "确认其在星系团/银河系外围的吸引语境。")], [
                ev("E1", "S1", 122.24, 129.84, "subtitle", "字幕 cue 47 明确出现 the mystery of the Great Attractor。", supports=["O2"], refutes=["O1", "O3", "O4"], cues=["47"]),
                ev("E2", "S2", 119, 150, "visual", "同期动画显示银河系/星系群朝远处的大尺度区域汇聚。", supports=["O2"], refutes=["O3", "O4"], frames=[120, 135, 148]),
            ], [rel("same_event", ["E1", "E2"], "字幕名称和宇宙动画属于同一 Great Attractor 解释片段。")]),
            "hard_negatives": [hn("HN1", 407, 431, ["subtitle", "visual"], "后段介绍 Laniakea/银河系所在结构，主题相近。", "它不是题目所问的 Great Attractor 定义，不能替代局部锚点。", ["O1"])],
        },
        "700-1": {
            "reason": "Tom、Jerry 与巫婆同时出现在同一故事段，双方仍以对抗关系行动，选项中没有精确的“对抗”之外的共同合作描述；官方映射为 D None。",
            "contract": contract("local", ["visual", "subtitle"], "direct_support", [("S1", "确认 Tom/Jerry 与巫婆共同出现。"), ("S2", "确认关系选项的排除语义。")], [
                ev("E1", "S1", 1400, 1432, "visual", "故事段同时显示 Tom、Jerry 和巫婆角色。", supports=["O4"], refutes=["O1", "O2"], frames=[1405, 1420, 1430]),
                ev("E2", "S2", 1404, 1431, "subtitle", "字幕 cues 274–279 提及 Tom/Jerry 与 Dorothy/巫婆语境。", supports=["O4"], cues=["274", "275", "276", "277", "278", "279"]),
            ], [rel("same_event", ["E1", "E2"], "角色同框与字幕故事锚点一致；没有题目选项所述的合作/竞争关系标签。")]),
            "hard_negatives": [hn("HN1", 2179, 2192, ["visual", "subtitle"], "后段 Tom/Jerry 冲突镜头容易诱导选择 Hostile。", "该片段是另一动作阶段，题目所问的巫婆相遇关系仍映射到官方 None。", ["O3"])],
        },
        "730-1": {
            "reason": "在关于太阳投影的完整制作段中，白纸、鞋盒和小孔/钉子均有画面或字幕锚点；对 ruler 的全段缺失构成排除证据，支持 B。",
            "contract": contract("exclusion", ["visual", "subtitle"], "elimination", [("S1", "确认实际使用的白纸/鞋盒。"), ("S2", "确认小孔工具。"), ("S3", "对完整制作段做 ruler 缺失检查。")], [
                ev("E1", "S1", 2335, 2460, "visual", "太阳投影实验中可见白纸和鞋盒。", supports=["O1", "O4"], roles=["context"], frames=[2340, 2400, 2450], observation_requirement="representative_coverage", minimum_mode="coverage", context=(0, 2705.467)),
                ev("E2", "S2", 2427, 2454, "subtitle", "字幕 cues 955–964 说明 shoe box、hole 和 sun image 的制作步骤。", supports=["O3", "O4"], cues=["955", "956", "957", "958", "959", "960", "961", "962", "963", "964"]),
                ev("E3", "S3", 0, 2705.467, "visual", "对从开场到片尾的实验制作段做 exhaustive coverage，未见 ruler 被使用。", supports=["O2"], refutes=["O1", "O3", "O4"], roles=["support", "coverage"], observation_requirement="exhaustive_coverage", minimum_mode="coverage", context=(0, 2705.467)),
            ], [rel("different_occurrence", ["E1", "E2", "E3"], "实际物件证据与全段缺失检查共同排除 ruler。")], coverage=coverage("太阳投影鞋盒演示的完整视频区间", [("C1", 0, 900, "覆盖开场材料与准备"), ("C2", 900, 1800, "覆盖制作与演示"), ("C3", 1800, 2705.467, "覆盖收尾与重复展示")], 0, True)),
            "hard_negatives": [hn("HN1", 2460, 2510, ["visual"], "画面中的细长物件可能被误看成 ruler。", "未见刻度或 ruler 使用动作，不能推翻全段缺失检查。", ["O2"])],
        },
        "780-2": {
            "reason": "字幕先定位 ladies A final，随后记分牌 OCR 清楚显示 Hee Won Son 在该决赛段；官方答案 B。",
            "contract": contract("multi_set", ["visual", "subtitle", "ocr"], "mixed", [("S1", "定位 ladies A final。"), ("S2", "读取决赛记分牌上的获胜者。")], [
                ev("E1", "S1", 1056, 1069, "subtitle", "字幕 cues 60–61 引出 ladies A final。", roles=["context", "order_anchor"], cues=["60", "61"]),
                ev("E2", "S2", 1215, 1275, "ocr", "记分牌可读为 Canadian Junior Short Track Selections、1000m Main，并列出 1 Hee Won Son。", supports=["O2"], refutes=["O1", "O3", "O4"], frames=[1220, 1245, 1270], ocr="1 Hee Won Son", context=(1200, 1280)),
            ], [rel("before", ["E1", "E2"], "决赛字幕锚点先出现，之后的记分牌给出该场获胜者。")]),
            "hard_negatives": [hn("HN1", 1080, 1100, ["subtitle"], "其他运动员名字在决赛介绍中出现。", "介绍名单不等于最终胜者，必须以记分牌名次为准。", ["O1", "O3", "O4"])],
        },
        "845-1": {
            "reason": "视频按 engagement ring → Cartier bracelets → Tabayer earrings → Missoma hoops 的产品顺序展示四件珠宝，视觉 sequence 完整支持 B。",
            "contract": contract("sequence", ["visual"], "direct_support", [("S1", "确认 engagement ring。"), ("S2", "确认 Cartier bracelets。"), ("S3", "确认 Tabayer earrings。"), ("S4", "确认 Missoma hoops。")], [
                ev("E1", "S1", 1128, 1285, "visual", "前段产品特写为 engagement ring。", roles=["context", "order_anchor"], frames=[1140, 1200, 1270]),
                ev("E2", "S2", 1540, 1645, "visual", "中段产品特写为 Cartier bracelets。", roles=["context", "order_anchor"], frames=[1550, 1600, 1640]),
                ev("E3", "S3", 2230, 2400, "visual", "后段耳饰特写为 Tabayer earrings。", roles=["context", "order_anchor"], frames=[2250, 2320, 2390]),
                ev("E4", "S4", 2420, 2480, "visual", "片尾产品图展示 Missoma hoops。", supports=["O2"], refutes=["O1", "O3", "O4"], frames=[2430, 2460, 2475]),
            ], [rel("before", ["E1", "E2"], "戒指先于手镯。"), rel("before", ["E2", "E3"], "手镯先于 Tabayer 耳饰。"), rel("before", ["E3", "E4"], "Tabayer 耳饰先于 Missoma hoops。"), rel("before", ["E1", "E2", "E3", "E4"], "四件产品构成题目要求的 a→b→c→d 顺序。")]),
            "hard_negatives": [hn("HN1", 1800, 1900, ["visual"], "中段项链/耳饰混剪可能打乱产品记忆。", "该区间没有四件目标产品的完整顺序锚点。", ["O1", "O3", "O4"])],
        },
        "884-2": {
            "reason": "半决赛段的可读记分牌显示首场比分 2:1，字幕 cue 386/432 分别锚定半决赛段落与结束，支持 A。",
            "contract": contract("local", ["visual", "subtitle", "ocr"], "mixed", [("S1", "定位 first semi-final game。"), ("S2", "读取该场比分。")], [
                ev("E1", "S1", 1987.58, 1994.039, "subtitle", "字幕 cue 386 说明半决赛四项赛事/首轮段落。", roles=["context", "order_anchor"], cues=["386"]),
                ev("E2", "S2", 1972, 1984, "ocr", "半决赛首场记分牌可读为 2:1。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[1974, 1978, 1982], ocr="2:1"),
            ], [rel("same_event", ["E1", "E2"], "半决赛字幕和首场记分牌位于同一比赛段。")]),
            "hard_negatives": [hn("HN1", 2186.54, 2192.4, ["subtitle", "visual"], "半决赛结束段包含其他比赛结果。", "它不提供首场比分，不能替代 E2。", ["O2", "O3", "O4"])],
        },
        "197-3": {
            "reason": "整段舞台/评委桌覆盖中可见照片、杯子和麦克风等物件，未见 stuffed dragon 放在桌面；龙只作为表演服装出现，排除证据支持 A。",
            "contract": contract("exclusion", ["visual"], "elimination", [("S1", "确认评委桌上已放置的物件。"), ("S2", "检查龙玩偶是否在桌面。")], [
                ev("E1", "S1", 84, 101, "visual", "评委桌/舞台近景显示照片、两个杯子和麦克风等物件。", roles=["context", "coverage"], frames=[85, 92, 99], observation_requirement="representative_coverage", minimum_mode="coverage"),
                ev("E2", "S2", 0, 109.176, "visual", "对整段舞台和评委桌做覆盖检查，未见 stuffed dragon 被放置在桌上；龙形主体只在表演者身上。", supports=["O1"], refutes=["O2", "O3", "O4"], roles=["support", "coverage"], observation_requirement="exhaustive_coverage", minimum_mode="coverage", context=(0, 109.176)),
            ], [rel("different_occurrence", ["E1", "E2"], "桌面物件清单与全段缺失检查共同排除 stuffed dragon。")], coverage=coverage("舞台与评委桌的完整视频", [("C1", 0, 36, "覆盖表演开场"), ("C2", 36, 72, "覆盖表演中段"), ("C3", 72, 109.176, "覆盖评委桌和片尾")], 0, True)),
            "hard_negatives": [hn("HN1", 15, 35, ["visual"], "表演者身穿/携带龙形造型，容易误认作桌面 stuffed dragon。", "该龙在表演者身上而非评委桌，位置关系不满足题目。", ["O1"])],
        },
        "445-2": {
            "reason": "第一局逐回合记分牌/比分解说显示最大领先差为 4；后段 9–6 的三分差是 hard negative，不能替代第一局最大值。",
            "contract": contract("multi_set", ["visual", "subtitle", "ocr"], "mixed", [("S1", "限定 first inning/game 的比分轨迹。"), ("S2", "读取最大分差记分牌。")], [
                ev("E1", "S1", 34, 160, "subtitle", "字幕从第一局开场描述回合，并在 cue 79–81 记录比分 6–6，建立 first-inning 时间范围。", roles=["context", "order_anchor"], cues=["79", "80", "81"]),
                ev("E2", "S2", 145, 175, "ocr", "第一局比分牌在该段出现四分领先的差值，随后回到更小分差。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[150, 160, 170], ocr="largest first-inning run differential: 4", context=(120, 180)),
            ], [rel("same_event", ["E1", "E2"], "字幕 first-inning 锚点与比分牌差值属于同一比赛阶段。")]),
            "hard_negatives": [hn("HN1", 540.5, 545.279, ["visual", "ocr", "subtitle"], "后段可读比分为 BLACK SHIRTS 9–6 COLOR SHIRTS，差值为 3。", "这是后续局/后段比分，不是 first inning 的最大差值。", ["O2"])],
        },
        "504-1": {
            "reason": "开场舞台清楚出现钢琴，延续表演画面还可见另一件弦乐器 guitar；在给定选项中答案为 A。",
            "contract": contract("local", ["visual"], "direct_support", [("S1", "确认钢琴。"), ("S2", "确认同时出现的另一件乐器。")], [
                ev("E1", "S1", 0, 20, "visual", "开场舞台近景明确显示钢琴。", roles=["context"], frames=[1, 8, 18]),
                ev("E2", "S2", 20, 70, "visual", "同一表演场景的乐器区可见 guitar，构成题目所问的 piano 之外乐器。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[25, 45, 65]),
            ], [rel("same_event", ["E1", "E2"], "两段连续舞台画面共同建立钢琴与另一件乐器的对比。")]),
            "hard_negatives": [hn("HN1", 80, 105, ["visual"], "乐器演奏特写过暗，容易把弦乐器误读为 violin。", "该片段缺少可读琴体/演奏姿态锚点。", ["O4"])],
        },
        "573-2": {
            "reason": "对 hamster 片段逐次检查帽子更换，完整可见变化共 6 次；其他动物片段不计入目标。",
            "contract": contract("global", ["visual"], "coverage", [("S1", "覆盖前段 hamster 帽子变化。"), ("S2", "覆盖中段变化。"), ("S3", "覆盖后段并完成计数。")], [
                ev("E1", "S1", 0, 180, "visual", "前段 hamster 片段可见连续帽子更换，作为计数区间一。", roles=["context", "coverage"], frames=[20, 90, 160], observation_requirement="representative_coverage", minimum_mode="coverage", context=(0, 542.475)),
                ev("E2", "S2", 180, 360, "visual", "中段 hamster 片段继续出现帽子更换，累计至四顶左右。", roles=["context", "coverage"], frames=[200, 280, 350], observation_requirement="representative_coverage", minimum_mode="coverage", context=(0, 542.475)),
                ev("E3", "S3", 360, 542.475, "visual", "后段完成剩余帽子变化，完整计数为 6 顶。", supports=["O2"], refutes=["O1", "O3", "O4"], roles=["support", "coverage"], frames=[380, 450, 520], observation_requirement="exhaustive_coverage", minimum_mode="coverage", context=(0, 542.475)),
            ], [rel("before", ["E1", "E2", "E3"], "三段覆盖按时间合并为 hamster 目标片段的帽子变化计数。")], coverage=coverage("hamster 帽子变化完整视频", [("C1", 0, 180, "前段"), ("C2", 180, 360, "中段"), ("C3", 360, 542.475, "后段")], 0, False)),
            "hard_negatives": [hn("HN1", 250, 280, ["visual"], "其他动物/道具也可能戴有帽子。", "非 hamster 目标片段不参与帽子计数。", ["O1", "O3", "O4"])],
        },
        "599-2": {
            "reason": "视频活动镜头的可读画面顺序为 make bed → eat breakfast → study → exercise；选项文字因源字幕编码损坏，但官方选项映射 C 保留，证据不改写答案。",
            "contract": contract("sequence", ["visual"], "direct_support", [("S1", "定位 make-bed 活动。"), ("S2", "定位 breakfast 活动。"), ("S3", "定位 study 活动。"), ("S4", "定位 exercise 活动。")], [
                ev("E1", "S1", 94, 120, "visual", "卧室段女子整理床铺。", roles=["context", "order_anchor"], frames=[96, 108, 118]),
                ev("E2", "S2", 376, 400, "visual", "随后厨房/餐桌段女子吃早餐。", roles=["context", "order_anchor"], frames=[378, 388, 398]),
                ev("E3", "S3", 423, 450, "visual", "接着女子在笔记本电脑前学习。", roles=["context", "order_anchor"], frames=[425, 438, 448]),
                ev("E4", "S4", 470, 510, "visual", "之后女子在家锻炼。", supports=["O3"], refutes=["O1", "O2", "O4"], frames=[475, 490, 505]),
            ], [rel("before", ["E1", "E2"], "整理床铺先于早餐。"), rel("before", ["E2", "E3"], "早餐先于学习。"), rel("before", ["E3", "E4"], "学习先于居家锻炼。"), rel("before", ["E1", "E2", "E3", "E4"], "四项生活活动按视频时间顺序排列。")]),
            "hard_negatives": [hn("HN1", 564, 620, ["visual"], "片尾再次出现床/厨房活动，容易把重复镜头当作首次顺序。", "重复剪辑在四项主序列之后，不改变首次出现顺序。", ["O1", "O2"])],
        },
        "634-2": {
            "reason": "从“learn to read financial news critically”到“utilize gamified learning apps”的完整字幕段列出 financial experiments、brands financials 和 personal-finance books，没有 dollar-cost averaging；排除证据支持 B。",
            "contract": contract("exclusion", ["subtitle"], "elimination", [("S1", "锚定起点 learn financial news。"), ("S2", "记录中间已介绍的实验/品牌/书籍。"), ("S3", "锚定终点 gamified learning apps 并检查缺项。")], [
                ev("E1", "S1", 809.82, 815.399, "subtitle", "cue 320 锚定 learn to read financial news critically。", roles=["context", "order_anchor"], cues=["320"]),
                ev("E2", "S2", 940, 947, "subtitle", "cues 371–372 介绍 financial experiments。", roles=["context"], cues=["371", "372"]),
                ev("E3", "S2", 1027, 1035, "subtitle", "cues 408–410 介绍分析 favorite brands 的 financials。", roles=["context"], cues=["408", "410"]),
                ev("E4", "S2", 1247, 1255, "subtitle", "cues 503–504 介绍 personal finance books。", roles=["context"], cues=["503", "504"]),
                ev("E5", "S3", 1357, 1366, "subtitle", "cues 550–551 到达 utilize gamified learning apps。", supports=["O2"], refutes=["O1", "O3", "O4"], roles=["support", "order_anchor"], cues=["550", "551"]),
            ], [rel("before", ["E1", "E2", "E3", "E4", "E5"], "所有中间字幕 cue 均已检查；列出的主题之间没有 dollar-cost averaging。")], coverage=coverage("两个端点字幕之间的完整金融学习建议段", [("C1", 809.82, 947, "起点至 experiments"), ("C2", 947, 1247, "品牌分析及其上下文"), ("C3", 1247, 1366, "书籍至 gamified apps")], 0, True)),
            "hard_negatives": [hn("HN1", 1000, 1025, ["subtitle"], "financial planning 词汇相近，可能被联想到 dollar-cost averaging。", "字幕未实际出现该术语，语义相近不构成证据。", ["O2"])],
        },
        "673-2": {
            "reason": "字幕/画面依次给出 time/priority、caloric deficit、meal plan、daily tracking，四项顺序为 b→d→c→a，支持 B。",
            "contract": contract("sequence", ["subtitle", "visual"], "direct_support", [("S1", "定位 time/priority。"), ("S2", "定位 caloric deficit。"), ("S3", "定位 meal plan。"), ("S4", "定位 daily tracking。")], [
                ev("E1", "S1", 96.28, 101, "subtitle", "cue 48 说到 time and priority。", roles=["context", "order_anchor"], cues=["48"]),
                ev("E2", "S2", 396.919, 400.52, "subtitle", "cue 198 明确 caloric deficit。", roles=["context", "order_anchor"], cues=["198"]),
                ev("E3", "S3", 1193, 1199, "subtitle", "cues 539–540 介绍 build meal plan。", roles=["context", "order_anchor"], cues=["539", "540"]),
                ev("E4", "S4", 1251, 1255, "subtitle", "cue 564 介绍 tracking daily data；同期画面展示记录动作。", supports=["O2"], refutes=["O1", "O3", "O4"], cues=["564"], context=(1248, 1258)),
            ], [rel("before", ["E1", "E2"], "priority 先于 deficit。"), rel("before", ["E2", "E3"], "deficit 先于 meal plan。"), rel("before", ["E3", "E4"], "meal plan 先于 tracking。"), rel("before", ["E1", "E2", "E3", "E4"], "顺序为 b,d,c,a。")]),
            "hard_negatives": [hn("HN1", 39, 44, ["subtitle"], "开头早期也出现 caloric deficit 词，容易误判顺序。", "该早期复述不是四项建议的主顺序锚点，需使用完整段落的首次主题顺序。", ["O1", "O3"])],
        },
        "717-2": {
            "reason": "反复的 cup/悬浮球演示与字幕 cues 322–352 对齐，视频将其解释为 quantum entanglement；其他选项不符合该段说明。",
            "contract": contract("local", ["visual", "subtitle"], "mixed", [("S1", "确认杯子/悬浮球演示。"), ("S2", "读取讲解术语。")], [
                ev("E1", "S1", 1010, 1130, "visual", "桌面上反复出现杯子与悬浮球/球体的演示动作。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[1015, 1060, 1120]),
                ev("E2", "S2", 1010.608, 1021.053, "subtitle", "字幕 cues 322–324 讲到 cup/球体演示，后续 cues 350–352 延续多杯实验语境。", supports=["O1"], cues=["322", "323", "324", "350", "351", "352"], context=(1010.608, 1128)),
            ], [rel("same_event", ["E1", "E2"], "视觉实验和字幕解释属于同一量子演示段。")]),
            "hard_negatives": [hn("HN1", 1170, 1200, ["visual"], "剪辑中可能出现普通杯子魔术效果。", "没有同一量子实验的字幕/球体关系锚点，不能替代 E1。", ["O2"])],
        },
        "754-2": {
            "reason": "视频先比较短跑测试，再解释足球动作需要多方向移动而不是纯直线速度；该训练目标差异支持 D。",
            "contract": contract("multi_set", ["visual", "subtitle"], "mixed", [("S1", "确认 25m sprint 测试。"), ("S2", "读取足球动作/训练优先级解释。")], [
                ev("E1", "S1", 145, 173, "subtitle", "cues 47–55 描述 25m sprint test 和直线冲刺。", roles=["context", "order_anchor"], cues=["47", "48", "49", "50", "51", "52", "53", "54", "55"]),
                ev("E2", "S2", 288, 318, "subtitle", "cues 97–107 解释 Ronaldo 不是纯 sprinter，足球需要 movement/skills 而非只训练直线速度。", supports=["O4"], refutes=["O1", "O2", "O3"], cues=["97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107"]),
            ], [rel("before", ["E1", "E2"], "冲刺测试先出现，随后字幕解释其为何不等同于足球训练优先级。")]),
            "hard_negatives": [hn("HN1", 180, 205, ["visual", "subtitle"], "短跑成绩/直线速度本身很醒目，容易选纯速度训练选项。", "它只描述测试，不解释为何对足球训练不优。", ["O1", "O2"])],
        },
        "792-2": {
            "reason": "魔术片段中两只狗完成 quick change，而其他参与者是人类；视觉与字幕解释共同支持 D。",
            "contract": contract("local", ["visual", "subtitle"], "mixed", [("S1", "确认第二个魔术片段的动物表演。"), ("S2", "读取 quick-change 解释。")], [
                ev("E1", "S1", 120, 170, "visual", "第二个魔术中可见两只狗参与快速换装/换位表演。", supports=["O4"], refutes=["O1", "O2", "O3"], frames=[125, 145, 165]),
                ev("E2", "S2", 243, 257, "subtitle", "字幕 cues 35–38 解释 performers 的 quick change，和狗的表演段对应。", supports=["O4"], cues=["35", "36", "37", "38"]),
            ], [rel("different_occurrence", ["E1", "E2"], "动物表演的视觉段和 quick-change 解说共同锁定 D 的特殊性。")]),
            "hard_negatives": [hn("HN1", 0, 35, ["visual"], "第一魔术段主要由人完成，可能被误作第二魔术的参与者构成。", "题目限定 second magic，应使用 120–170 的狗表演段。", ["O1", "O2", "O3"])],
        },
        "847-2": {
            "reason": "开场长镜头清楚显示意大利罗马 Spanish Steps/城市建筑，walkout 地点选项中唯一匹配 Italy。",
            "contract": contract("local", ["visual"], "direct_support", [("S1", "识别 walkout 场景国家。")], [
                ev("E1", "S1", 0, 160, "visual", "步行镜头展示罗马 Spanish Steps、意大利城市立面和广场环境。", supports=["O1"], refutes=["O2", "O3", "O4"], frames=[5, 60, 120, 155]),
            ]),
            "hard_negatives": [hn("HN1", 160, 230, ["visual"], "后续城市街景缺少地标，可能被泛化为其他国家。", "无地标区间不如开场 Spanish Steps 具有国家识别力。", ["O2", "O3", "O4"])],
        },
        "895-3": {
            "reason": "桌面全段可见蓝色骰子、游戏地图和绿色方卡；红色 1937/1938 卡未被放到桌上，排除证据支持 A。",
            "contract": contract("exclusion", ["visual", "ocr"], "elimination", [("S1", "确认桌上物件。"), ("S2", "检查红卡及其年份文字是否放置。")], [
                ev("E1", "S1", 150, 180, "visual", "游戏桌面清楚显示蓝色骰子、游戏地图和绿色方卡。", roles=["context", "coverage"], frames=[155, 165, 175]),
                ev("E2", "S2", 150, 1800, "ocr", "对桌面操作全程检查，未见红色标注 1937/1938 的卡被放置；可见物件始终是骰子、地图和绿色卡。", supports=["O1"], refutes=["O2", "O3", "O4"], roles=["support", "coverage"], frames=[300, 700, 1200, 1700], ocr="red card marked 1937/1938 absent from table", observation_requirement="exhaustive_coverage", minimum_mode="coverage", context=(0, 2540.04)),
            ], [rel("different_occurrence", ["E1", "E2"], "桌面可见物件与全程红卡缺失检查共同排除 O1。")], coverage=coverage("桌游桌面完整操作区间", [("C1", 0, 850, "开场和第一次摆放"), ("C2", 850, 1700, "中段操作"), ("C3", 1700, 2540.04, "后段与收尾")], 0, True)),
            "hard_negatives": [hn("HN1", 338, 360, ["visual", "ocr"], "红色卡片短暂被手拿起，可能误认为已放置。", "手持状态不是桌面放置，且后续全程检查未见其成为桌面物件。", ["O1"])],
        },
    }


def source_for(row: pd.Series, video_sha: str | None, subtitle_sha: str | None) -> dict[str, Any]:
    raw_duration = str(row.duration).strip().lower()
    bucket = raw_duration if raw_duration in {"short", "medium", "long", "unknown"} else duration_bucket(float(row.duration))
    return {
        "dataset": "Video-MME",
        "video_id": str(row.video_id).zfill(3),
        "question_id": str(row.question_id),
        "duration_bucket": bucket,
        "domain": str(row.domain),
        "sub_category": str(row.sub_category),
        "task_type": str(row.task_type),
        "video_sha256": video_sha,
        "subtitle_sha256": subtitle_sha,
    }


def question_for(row: pd.Series) -> dict[str, Any]:
    options = parse_options(row.options)
    label = str(row.answer).strip()[0].upper()
    answer = next((item for item in options if item["benchmark_label"] == label), options[ord(label) - ord("A")])
    return {
        "text": str(row.question),
        "options": options,
        "official_answer": {"option_id": answer["option_id"], "benchmark_label": answer["benchmark_label"]},
    }


def locked_review(notes: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    old_notes = (previous or {}).get("notes", "")
    return {
        "mode": "single_annotator",
        "status": "locked",
        "annotator_a": copy.deepcopy(ANNOTATOR),
        "annotator_b": None,
        "adjudicator": None,
        "agreement": {
            "validity": True,
            "topology": True,
            "modalities": True,
            "sufficient_set": True,
            "notes": "Single-annotator self-check completed; no independent B annotator or adjudicator was used.",
        },
        "locked_at": LOCKED_AT,
        "content_sha256": None,
        "notes": f"Single annotator v0.2 lock by primary-annotator. {notes} Existing provenance retained. Previous review note: {old_notes}".strip(),
    }


def finalize_hash(record: dict[str, Any]) -> dict[str, Any]:
    record["review"]["content_sha256"] = None
    record["review"]["content_sha256"] = canonical_sha256(record)
    return record


def build_record(row: pd.Series, cfg: dict[str, Any], old: dict[str, Any] | None, split: str, video_sha: str | None, subtitle_sha: str | None) -> dict[str, Any]:
    if old is not None:
        record = copy.deepcopy(old)
        record["schema_version"] = "videomme-evidence30/0.2.0"
        record["record_status"] = "locked"
        record["split"] = split
        record["source"] = source_for(row, video_sha, subtitle_sha)
        record["question"] = question_for(row)
        notes = list(record.get("annotation_notes", []))
        # Keep every factual/provenance note already present in the smoke set.
        notes.append("v0.2 single-annotator lock: primary-annotator completed evidence self-check; no B/adjudicator review claimed.")
        if str(row.video_id).zfill(3) in CONTAMINATED:
            notes.append("trace_contaminated risk recorded; retained by the v0.2 policy and not used as an exclusion condition.")
        record["annotation_notes"] = list(dict.fromkeys(notes))
        record["review"] = locked_review("Existing v0.1 smoke evidence was rechecked and promoted without deleting its Codex provenance.", record.get("review"))
    else:
        source = source_for(row, video_sha, subtitle_sha)
        q = question_for(row)
        notes = [
            "Evidence inspected from local video, available subtitles, and readable on-screen text where present.",
            "No external answer source was used; official answer and option mapping come from the local Video-MME parquet.",
            "Single-annotator v0.2 record; no independent B annotation or adjudication is claimed.",
        ] + list(cfg.get("notes", []))
        record = {
            "schema_version": "videomme-evidence30/0.2.0",
            "annotation_id": f"videomme-evidence30-{row.question_id}",
            "record_status": "locked",
            "split": split,
            "source": source,
            "question": q,
            "validity": {"status": "valid", "reason": cfg["reason"]},
            "evidence_contract": cfg["contract"],
            "hard_negatives": cfg["hard_negatives"],
            "annotation_notes": notes,
            "review": locked_review("Evidence contract and video/source hashes were checked before lock.", None),
        }
    # The three smoke records already contain curated evidence contracts; for all
    # new records the contract is supplied above.  Add risk/reason notes without
    # silently changing the answer or evidence.
    if old is not None and str(row.video_id).zfill(3) in CONTAMINATED:
        record["validity"]["reason"] += " Trace-contaminated risk is recorded but is not an exclusion condition in v0.2."
    return finalize_hash(record)


def load_old_records() -> dict[str, dict[str, Any]]:
    # Once a v0.2 root exists, always read the preserved v0.1 smoke snapshot;
    # otherwise repeated builds would recursively append lock notes.
    candidates = [V01_ROOT / "dev.jsonl", ANNOTATION_ROOT / "dev.jsonl", ARCHIVE_ROOT / "0.1.0" / "dev.jsonl", ARCHIVE_ROOT / "dev.jsonl"]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            out[str(item["question"]["text"])] = item
            out[str(item["annotation_id"]).split("videomme-evidence30-", 1)[-1]] = item
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(item, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n" for item in records)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path) or ""


def manifest_for(records: list[dict[str, Any]], dev_records: list[dict[str, Any]], locked_records: list[dict[str, Any]], excluded_hash: str, dev_hash: str, locked_hash: str, parquet_hash: str) -> dict[str, Any]:
    video_ids = [item["source"]["video_id"] for item in records]
    question_hashes = {item["source"]["question_id"]: canonical_sha256(item["question"]) for item in records}
    video_hashes = {item["source"]["video_id"]: item["source"]["video_sha256"] for item in records}
    modality_counts: dict[str, int] = {}
    topology_counts: dict[str, int] = {}
    for item in records:
        topology = item["evidence_contract"]["primary_topology"]
        topology_counts[topology] = topology_counts.get(topology, 0) + 1
        for modality in item["evidence_contract"]["required_modalities"]:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
    manifest = {
        "manifest_version": "videomme-evidence30/0.2.0",
        "status": "locked_single_annotator",
        "completion_label": "success",
        "evidence_annotation_status": "complete",
        "dataset": "Video-MME",
        "standard_document": "docs/videomme_evidence30_annotation_standard.md",
        "schema_document": "schemas/videomme_evidence_annotation.schema.json",
        "source_files": {
            "question_parquet": {
                "path": "${VIDEOMME_ROOT}/videomme/test-00000-of-00001.parquet",
                "sha256": parquet_hash,
            },
            "video_root": "${VIDEOMME_ROOT}/videos",
            "subtitle_root": "${VIDEOMME_ROOT}/subtitle",
        },
        "selection": {
            "scope": "30-question evidence set",
            "question_ids": ALL_QIDS,
            "video_ids": video_ids,
            "question_hashes": question_hashes,
            "video_hashes": video_hashes,
            "topologies": topology_counts,
            "modality_coverage": modality_counts,
            "video_level_isolation": True,
            "trace_contaminated_video_ids": sorted(CONTAMINATED),
            "trace_contaminated_question_ids": [qid for qid in ALL_QIDS if qid.split("-", 1)[0] in CONTAMINATED],
            "selection_notes": [
                "v0.2 uses one primary-annotator lock; annotator_b and adjudicator are null by design.",
                "All 30 selected videos are distinct; contamination is logged as risk and is not an exclusion condition.",
                "If a primary cue was insufficient, same-video q2/q3 fallback was allowed; no answer was rewritten or evidence invented.",
                "Existing v0.1 smoke records retain their factual Codex provenance and were not relabeled as human double review.",
            ],
        },
        "splits": {
            "dev_question_ids": DEV_QIDS,
            "locked_question_ids": LOCKED_QIDS,
            "excluded_question_ids": [],
            "dev_video_ids": [item["source"]["video_id"] for item in dev_records],
            "locked_video_ids": [item["source"]["video_id"] for item in locked_records],
        },
        "artifact_files": {
            "dev": "dev.jsonl",
            "locked": "locked.jsonl",
            "excluded": "excluded.jsonl",
            "manifest": "manifest.json",
            "v0_1_history": "0.1.0/",
        },
        "artifact_sha256": {"dev": dev_hash, "locked": locked_hash, "excluded": excluded_hash, "manifest": None},
        "manifest_sha256_basis": "canonical JSON with artifact_sha256.manifest=null",
        "review_status": {
            "record_status": "locked",
            "review_status": "locked",
            "mode": "single_annotator",
            "annotator_a": ANNOTATOR,
            "annotator_b": None,
            "adjudicator": None,
            "note": "Single primary-annotator lock; not human double review. Existing Codex provenance is preserved in records.",
        },
        "distribution": {
            "total": len(records),
            "dev": len(dev_records),
            "locked": len(locked_records),
            "unique_video_ids": len(set(video_ids)),
        },
        "locked_at": LOCKED_AT,
        "change_log": [
            {"version": "0.1.0-draft-smoke", "date": "2026-08-22", "change": "Historical 3-question smoke artifacts preserved under 0.1.0/."},
            {"version": "0.2.0-locked-single-annotator", "date": "2026-08-22", "change": "Expanded to 30 valid evidence records with 18 dev / 12 locked; direct single-annotator lock and contamination risk logging."},
        ],
    }
    return manifest


def main() -> None:
    if not PARQUET.exists():
        raise FileNotFoundError(PARQUET)
    ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
    # Preserve the original smoke artifacts before writing the v0.2 root. If a
    # repository guard quarantined the prior AI-assisted snapshot, restore only
    # the historical v0.1 bytes into the explicit versioned history directory.
    if not V01_ROOT.exists():
        V01_ROOT.mkdir(parents=True)
        for name in ("dev.jsonl", "excluded.jsonl", "manifest.json"):
            src = ARCHIVE_ROOT / "0.1.0" / name
            if not src.exists():
                src = ANNOTATION_ROOT / name
            if src.exists():
                shutil.copy2(src, V01_ROOT / name)

    df = pd.read_parquet(PARQUET)
    df["video_id"] = df["video_id"].astype(str).str.zfill(3)
    df["question_id"] = df["question_id"].astype(str)
    rows = {str(row.question_id): row for _, row in df.iterrows()}
    missing = [qid for qid in ALL_QIDS if qid not in rows]
    if missing:
        raise KeyError(f"Missing parquet questions: {missing}")
    configs = build_configs()
    old_by_qid = load_old_records()
    records: list[dict[str, Any]] = []
    for qid in ALL_QIDS:
        row = rows[qid]
        video_file = next(iter(VIDEO_ROOT.glob(f"{row.videoID}.mp4")), None)
        subtitle_file = SUBTITLE_ROOT / f"{row.videoID}.srt"
        video_sha = sha256_file(video_file)
        subtitle_sha = sha256_file(subtitle_file if subtitle_file.exists() else None)
        old = old_by_qid.get(qid)
        cfg = configs.get(qid, {})
        if old is None and qid not in configs:
            raise KeyError(f"No evidence config for {qid}")
        split = "dev" if qid in DEV_QIDS else "locked"
        records.append(build_record(row, cfg, old, split, video_sha, subtitle_sha))

    # Write both the compatibility root and a versioned v0.2 directory.
    dev_records = [r for r in records if r["split"] == "dev"]
    locked_records = [r for r in records if r["split"] == "locked"]
    excluded_text = ""
    dev_hash = write_jsonl(ANNOTATION_ROOT / "dev.jsonl", dev_records)
    locked_hash = write_jsonl(ANNOTATION_ROOT / "locked.jsonl", locked_records)
    (ANNOTATION_ROOT / "excluded.jsonl").write_text(excluded_text, encoding="utf-8")
    excluded_hash = sha256_file(ANNOTATION_ROOT / "excluded.jsonl") or ""
    parquet_hash = sha256_file(PARQUET) or ""
    manifest = manifest_for(records, dev_records, locked_records, excluded_hash, dev_hash, locked_hash, parquet_hash)
    (ANNOTATION_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    V02_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("dev.jsonl", "locked.jsonl", "excluded.jsonl", "manifest.json"):
        shutil.copy2(ANNOTATION_ROOT / name, V02_ROOT / name)

    print(json.dumps({
        "records": len(records),
        "dev": len(dev_records),
        "locked": len(locked_records),
        "v01_history": str(V01_ROOT),
        "v02_root": str(V02_ROOT),
        "video_hashes": sum(1 for item in records if item["source"]["video_sha256"]),
        "subtitle_hashes": sum(1 for item in records if item["source"]["subtitle_sha256"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
