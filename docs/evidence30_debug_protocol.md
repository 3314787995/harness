# Evidence30 relaxed debug protocol

- Policy ID: `evidence30-relaxed-debug/1.0`
- Canonical references: `annotations/videomme_evidence30/0.2.0/`
- Split: 18 dev + 12 locked, 30 distinct videos
- Intended use: internal engineering/debug validation
- Reference quality: AI-assisted internal reference, not independent human gold
- Runtime: local Qwen3-VL-2B, deterministic generation, subtitles enabled
- Direct input: 16 timestamps uniformly selected from the full 1 fps cache, with
  a per-frame ceiling of 131072 pixels for the local 8 GiB GPU
- Old coarse-to-fine global branch: 16 uniformly sampled frames in one call;
  local rounds retain their 12-frame working set
- Direct subtitles: at most 8000 characters, balanced over 16 full-video time
  segments rather than truncating the beginning of long videos

## 1. Split protocol

The 18 dev questions are visible and may be rerun while code, prompts, budgets,
and scoring are changed. The 12 locked questions are reserved for one
aggregate-only run after freezing. All known trace-contaminated questions are
in dev. Dev intentionally has no `global` question; no replacement is added,
so the locked global case remains completely unseen.

Every comparison runs the same three strategies in order:

1. `direct`
2. `coarse_to_fine`
3. `active_tree`

This produces 54 dev executions and 36 locked executions.

## 2. Relaxed exposure scoring

The scorer evaluates only media that the model actually received. The unified
exposure trace records stage, modality, frame timestamps, subtitle intervals,
and source strategy. Annotation answers, context intervals, atomic facts,
hard negatives, and required-slot definitions are never passed to the model.

A reference evidence item is hit when:

- `visual` or `ocr`: at least one exposed frame timestamp is inside its
  inclusive `context_interval`;
- `subtitle`: at least one exposed subtitle interval overlaps its inclusive
  `context_interval`.

Any hit evidence item may cover its corresponding required slot. A question is
relaxed-grounded when every required slot in at least one sufficient set is
covered. Core-only hits, exact facts, relations, hard-negative exposure, and
dual verification are retained as diagnostic metrics but are not engineering
pass conditions.

## 3. Commands

Run the no-model integrity check first:

```powershell
python -m qwen3vl_agent.evidence30_cli preflight
```

Run all three strategies on the visible dev split:

```powershell
python -m qwen3vl_agent.evidence30_cli run --split dev --config configs\evidence30_2b.yaml --output-dir runs\evidence30\dev_v1
```

After dev passes the engineering gate, freeze exact source, prompts, explicit
configuration, scorer, annotations, model files, and environment:

```powershell
python -m qwen3vl_agent.evidence30_cli freeze --config configs\evidence30_2b.yaml --dev-summary runs\evidence30\dev_v1\summary.json --output runs\evidence30\freeze_v1.json
```

Run the locked split once with the same `freeze_id`:

```powershell
python -m qwen3vl_agent.evidence30_cli run --split locked --config configs\evidence30_2b.yaml --freeze-manifest runs\evidence30\freeze_v1.json --output-dir runs\evidence30\locked_v1
```

## 4. Engineering gate

The dev freeze gate and final locked pipeline pass both require:

- every expected strategy/question execution completes;
- every output is a legal option;
- normalized exposure trace, stop reason, and resource ledger are complete;
- no fatal error, degraded fallback, or budget violation.

Successful protocol repair retries are allowed. Accuracy, verified rate, and
relaxed grounding are reported without hard thresholds: a wrong or unverified
answer is a model/mechanism outcome, not an engineering failure.

The old coarse-to-fine path may make one text-only protocol repair call after a
malformed/truncated controller response. The repair sees no new media, may only
reformat the existing response, and is recorded in the trace and token ledger.

## 5. Locked handling

Locked progress and the first summary never print question IDs, item results,
or per-question traces. Detailed records are written to `sealed/items.jsonl`;
the visible summary reports only its SHA-256. Groups with fewer than three
questions are omitted, preventing the single global question from being
inferred.

An interrupted run may resume only with the same `freeze_id` and output
directory. Any code, prompt, explicit config, scoring, model, environment, or
annotation hash mismatch is rejected. Unsealing item details marks this
locked version consumed. Adjusting the pipeline after aggregate inspection also
requires a new held-out set for any later effect claim.
