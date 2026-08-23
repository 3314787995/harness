from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_trace_html(trace: dict[str, Any], output_path: str | Path) -> Path:
    """Render one self-contained, human-readable active-tree timeline."""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    event_cards = "\n".join(_event_card(index, event) for index, event in enumerate(trace.get("events", []), 1))
    evidence_rows_parts: list[str] = []
    for item in trace.get("evidence_ledger", []):
        interval = f"{item.get('start_seconds')}-{item.get('end_seconds')}s"
        evidence_rows_parts.append(
            "<tr>"
            f"<td>{_escape(item.get('evidence_id'))}</td>"
            f"<td>{_escape(item.get('node_id'))}</td>"
            f"<td>{_escape(interval)}</td>"
            f"<td>{_escape(item.get('modality'))}</td>"
            f"<td>{_escape(item.get('fact'))}</td>"
            "</tr>"
        )
    evidence_rows = "\n".join(evidence_rows_parts)
    verification = _escape(json.dumps(trace.get("verification_attempts", []), ensure_ascii=False, indent=2))
    raw = _escape(json.dumps(trace, ensure_ascii=False, indent=2, default=str))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ActiveTree replay</title>
<style>
body{{font:14px/1.5 system-ui;margin:24px;background:#f6f7f9;color:#1f2937}}
.hero,.card{{background:white;border:1px solid #d8dee8;border-radius:10px;padding:16px;margin:12px 0}}
.ok{{color:#08783e}} .bad{{color:#b42318}} .frames{{display:flex;gap:8px;flex-wrap:wrap}}
.frames img{{width:180px;height:110px;object-fit:cover;border-radius:6px;border:1px solid #ddd}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101828;color:#e5e7eb;padding:12px;border-radius:8px}}
</style></head><body>
<section class="hero"><h1>Active Evidence Tree replay</h1>
<p><strong>Question:</strong> {_escape(trace.get('question'))}</p>
<p><strong>Answer:</strong> {_escape(trace.get('final_benchmark_label'))} / {_escape(trace.get('final_option_id'))}
<strong class="{'ok' if trace.get('verified') else 'bad'}">verified={str(bool(trace.get('verified'))).lower()}</strong>
stop={_escape(trace.get('stop_reason'))}</p></section>
<h2>Timeline</h2>{event_cards or '<p>No events.</p>'}
<h2>Evidence ledger</h2><table><thead><tr><th>ID</th><th>Node</th><th>Time</th><th>Mode</th><th>Fact</th></tr></thead>
<tbody>{evidence_rows}</tbody></table>
<h2>Verification</h2><pre>{verification}</pre>
<details><summary>Full trace JSON</summary><pre>{raw}</pre></details>
</body></html>"""
    target.write_text(document, encoding="utf-8")
    return target


def _event_card(index: int, event: dict[str, Any]) -> str:
    frames = []
    for frame in event.get("frames", []):
        path = frame.get("path")
        if not path:
            continue
        try:
            uri = Path(str(path)).resolve().as_uri()
        except ValueError:
            continue
        frames.append(
            f'<figure><img src="{html.escape(uri, quote=True)}" alt="frame">'
            f'<figcaption>{_escape(frame.get("id"))} {_escape(frame.get("timestamp_seconds"))}s</figcaption></figure>'
        )
    summary = {key: value for key, value in event.items() if key not in {"frames", "subtitles"}}
    subtitles = event.get("subtitles") or ""
    subtitle_html = ""
    if subtitles:
        subtitle_html = (
            "<details><summary>Subtitles</summary>"
            f"<pre>{_escape(subtitles)}</pre></details>"
        )
    return (
        f'<section class="card"><h3>{index}. {_escape(event.get("type"))}</h3>'
        f'<pre>{_escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))}</pre>'
        f'<div class="frames">{"".join(frames)}</div>'
        f"{subtitle_html}"
        "</section>"
    )


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))
