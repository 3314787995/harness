# VideoMME-Evidence30 v0.2

Status: `ai_assisted_debug_complete`

The canonical immutable artifact set is `0.2.0/`: 30 question-level
AI-assisted internal reference records, split into 18 `dev` and 12 `locked`
questions from 30 distinct videos. These records are not independently
human-authored or double-reviewed gold annotations and must not be described as
such.

The 18 dev records may be inspected repeatedly while debugging. After the
pipeline is frozen, the 12 locked records are used once; their item-level
results and traces remain sealed during the first aggregate review. The dev
split intentionally contains no `global` example, so the one locked global
question remains completely unseen.

Every active record is `record_status: "locked"` and
`validity.status: "valid"`. The strict evidence structure is retained as a
reference, while engineering validation uses the relaxed temporal-exposure
policy documented in `docs/evidence30_debug_protocol.md`.

The records retain factual Codex provenance in their notes. Trace-contaminated
videos are retained as a logged risk under the v0.2 policy and are all assigned
to dev. The v0.1 smoke artifacts remain under `0.1.0/`; compatibility copies at
this directory's root are not the canonical 0.2.0 inputs.
