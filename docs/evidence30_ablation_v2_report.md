# Evidence30 Oracle context + unified answer head ablation

## Scope

This is a diagnostic experiment on the 18 Evidence30 dev questions only. It does not read,
unseal, or rerun the consumed 12-question locked split. References remain AI-assisted internal
debug data rather than independent human gold.

Policy: `evidence30-oracle-unified-head/1.1`

Run signature:
`9aab79d4f207170f0d9c9aa929def01c01e6ad7021329677e2cdae338ff679bc`

## Controlled inputs

All three packet sources use the same deterministic local Qwen3-VL-2B model and the same fresh,
one-pass multiple-choice prompt. No prior answer, planner output, evidence-ledger text, atomic fact,
hard negative, official answer, or reference interval number is included in the prompt.

- `direct_replay`: the 16 frames and exposed subtitle cues from the completed dev direct trace.
- `active_tree_replay`: frames actually shown in the completed active-tree trace. Evidence-ledger
  and verification citations are retained first, then the remaining exposed frames are sampled
  uniformly, with a matched maximum of 16 frames.
- `oracle_context`: choose the valid sufficient evidence set with the smallest union of context
  intervals. Visual/OCR and subtitle intervals are sampled independently so a broad visual interval
  cannot consume the subtitle budget. The interval labels and reference semantics are not shown to
  the model.

The model receives at most 16 chronological cached frames, at most 8,000 subtitle characters, and
returns a single option label. Preflight requires every Oracle packet to cover every required slot
under the relaxed temporal-exposure scorer.

## Superseded v1 run

The first diagnostic run (`runs/evidence30/ablations_v1`) exposed a packet-construction defect on a
mixed-modality question: a full-video visual context was merged with a narrow late subtitle context,
so the 8,000-character limit was exhausted by early subtitles. That v1 run is retained as a debug
artifact but must not be used for conclusions.

Policy 1.1 separates windows by modality and adds an Oracle-grounding assertion to preflight. The
official v2 run achieves 18/18 Oracle context grounding and 18/18 Oracle core temporal hits.

## Engineering result

- 54/54 model calls completed with legal labels.
- No fatal error, OOM, fallback, or missing packet.
- v2 direct and active replay predictions and input-token counts reproduce v1 exactly (18/18 each).
- 69 pytest tests, Ruff, and compileall pass.

## Aggregate result

| Packet source | Accuracy | Relaxed grounded | Slot coverage | Mean selected frames | Mean input tokens |
|---|---:|---:|---:|---:|---:|
| direct replay | 6/18 (33.3%) | 61.1% | 70.4% | 16.0 | 3,276.7 |
| active-tree replay | 6/18 (33.3%) | 38.9% | 60.6% | 13.7 | 2,717.8 |
| Oracle context | 9/18 (50.0%) | 100.0% | 100.0% | 14.3 | 1,357.2 |

Unified-head paired accuracy:

- active replay vs direct replay: 2 wins / 2 losses / 14 ties.
- Oracle vs direct replay: 4 wins / 1 loss / 13 ties.
- Oracle vs active replay: 5 wins / 2 losses / 11 ties.

Answer-process comparison against the original dev run:

- Direct replay and original direct both score 6/18; three predictions change but gains and losses
  cancel exactly.
- Active replay scores 6/18 while the original full active-tree scores 9/18. Replay gains two and
  loses five relative to the full chain. This comparison includes both removal of the reasoning
  history and compression of cumulative observations to one matched 16-frame packet, so it is not
  a pure prompt-only effect.

## Diagnostic interpretation

### Search/selection is a real bottleneck

Under the same answer head and frame cap, active replay does not beat uniform direct replay and has
lower temporal slot coverage. Oracle context produces a net three-question gain over either replay
source. The current active selector therefore does not provide a better matched-budget evidence
packet on these 18 dev questions.

### The full active chain is not purely harmful

The original full chain beats its compressed replay by a net three questions. Multi-round context,
additional cumulative observations, or intermediate reasoning helps some questions even though the
chain rarely reaches its intended verified terminal state.

### Temporal Oracle coverage is not sufficient for 2B

All 18 Oracle packets hit every annotated context and core interval, but only 9 are answered
correctly. The failure is concentrated in less localized evidence:

- short: 5/6 correct;
- medium: 3/5 correct;
- long: 1/7 correct.

Correct Oracle packets have a mean of 62 candidate cached frames and a mean core-interval width of
26 seconds. Wrong Oracle packets have 386 candidate frames and a mean core-interval width of 134
seconds. A temporal hit only means at least one supplied timestamp overlaps the interval; it does
not prove the decisive instant was visible or understood. The result therefore supports a combined
diagnosis: active search/localization is weak, while 2B plus a 16-frame one-pass packet also struggles
when the annotated evidence remains broad or compositionally difficult.

## Limits

- This is a tuned 18-question dev diagnostic, not a held-out effect estimate.
- One question equals 5.6 percentage points, so paired differences are descriptive only.
- `oracle_context` is an interval-guided diagnostic, not a semantic Oracle or a guaranteed accuracy
  upper bound.
- The matched active packet cannot reproduce the full chain's sequential exposure to more than 16
  cumulative frames.
