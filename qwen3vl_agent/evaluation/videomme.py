from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMMEQuestion:
    video_id: str
    question_id: str
    duration: str
    domain: str
    sub_category: str
    task_type: str
    question: str
    options: tuple[str, ...]
    answer: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> VideoMMEQuestion:
        return cls(
            video_id=str(row["videoID"]),
            question_id=str(row["question_id"]),
            duration=str(row["duration"]),
            domain=str(row["domain"]),
            sub_category=str(row["sub_category"]),
            task_type=str(row["task_type"]),
            question=str(row["question"]),
            options=tuple(str(option) for option in row["options"]),
            answer=str(row["answer"]).strip().upper(),
        )

    def video_path(self, video_dir: str | Path) -> Path:
        return Path(video_dir).expanduser().resolve() / f"{self.video_id}.mp4"

    def subtitle_path(self, subtitle_dir: str | Path) -> Path:
        return Path(subtitle_dir).expanduser().resolve() / f"{self.video_id}.srt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "question_id": self.question_id,
            "duration": self.duration,
            "domain": self.domain,
            "sub_category": self.sub_category,
            "task_type": self.task_type,
            "question": self.question,
            "options": list(self.options),
            "answer": self.answer,
        }


def load_videomme_questions(
    parquet_path: str | Path,
    *,
    question_ids: Iterable[str] | None = None,
) -> list[VideoMMEQuestion]:
    import pyarrow.parquet as pq

    requested_order = list(question_ids or ())
    requested = set(requested_order)
    table = pq.read_table(Path(parquet_path).expanduser().resolve())
    questions = [
        VideoMMEQuestion.from_row(row)
        for row in table.to_pylist()
        if not requested or str(row["question_id"]) in requested
    ]
    if requested:
        found = {question.question_id for question in questions}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown Video-MME question IDs: {', '.join(missing)}")
        order = {question_id: index for index, question_id in enumerate(requested_order)}
        questions.sort(key=lambda question: order[question.question_id])
    return questions
