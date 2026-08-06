from __future__ import annotations

import html
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import json_safe
from .storage import JobPaths


def display(value: float, unit: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    if unit == "ratio":
        return f"{value:.0%}"
    if unit == "seconds":
        return f"{value:.2f}s"
    if unit == "cents":
        return f"{value:.0f}"
    return f"{value:.1f}"


def make_comparison_chart(data: dict, output) -> None:
    metrics = [item for item in data["metrics"] if item["reliable"]]
    if not metrics:
        metrics = data["metrics"][:3]
    labels = [item["label"] for item in metrics]
    first = np.array([item["take_1"] for item in metrics], dtype=float)
    second = np.array([item["take_2"] for item in metrics], dtype=float)
    normalized_first = []
    normalized_second = []
    for before, after in zip(first, second, strict=False):
        if not np.isfinite(before) or not np.isfinite(after):
            normalized_first.append(0.0)
            normalized_second.append(0.0)
            continue
        scale = max(abs(before), abs(after), 1e-6)
        normalized_first.append(before / scale)
        normalized_second.append(after / scale)
    y = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11, max(4.0, len(labels) * 0.85)))
    fig.patch.set_facecolor("#f3efe5")
    axis.set_facecolor("#f3efe5")
    axis.barh(y + 0.18, normalized_first, height=0.32, color="#405651", label="Take 1")
    axis.barh(y - 0.18, normalized_second, height=0.32, color="#ff765f", label="Take 2")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xticks([])
    axis.spines[:].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Speaker-relative change map", loc="left", fontsize=17, fontweight="bold", color="#102a27")
    axis.text(0, -0.11, "Bars are normalized within each metric; use the table for exact values.", transform=axis.transAxes, fontsize=9, alpha=0.62)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_markdown(data: dict) -> str:
    lines = [
        f"# {data['title']}",
        "",
        "AGAPE Comparative Audit — Take 1 versus Take 2",
        "",
        "## Decision boundary",
        "",
        data["comparison_boundary"],
        "",
        f"Overall comparison confidence: **{data['quality']['confidence']}**",
        "",
        "## What changed",
        "",
    ]
    for index, finding in enumerate(data["findings"], start=1):
        lines.extend([
            f"### {index}. {finding['title']}",
            "",
            f"**Observed:** {finding['observation']}",
            "",
            f"**Interpretation:** {finding['interpretation']}",
            "",
            f"**Next experiment:** {finding['experiment']}",
            "",
            f"**Confidence:** {finding['confidence']}",
            "",
        ])
    lines.extend(["## Exact comparison", "", "| Signal | Take 1 | Take 2 | Reliable? |", "|---|---:|---:|:---:|"])
    for item in data["metrics"]:
        lines.append(f"| {item['label']} | {display(item['take_1'], item['unit'])} | {display(item['take_2'], item['unit'])} | {'yes' if item['reliable'] else 'no'} |")
    lines.extend([
        "",
        "## Human check",
        "",
        "Do not accept a candidate gain automatically. Replay the cited moments, decide whether the change serves the communication goal, and record that judgment as calibration data.",
        "",
        "No transcript was created or used.",
    ])
    return "\n".join(lines) + "\n"


def render_html(data: dict) -> str:
    findings = []
    for index, finding in enumerate(data["findings"], start=1):
        findings.append(f"""
        <article class="finding {html.escape(finding['kind'])}"><div class="number">{index:02d}</div><div>
        <p class="meta">{html.escape(finding['confidence'])} confidence</p><h3>{html.escape(finding['title'])}</h3>
        <p>{html.escape(finding['observation'])}</p><p><b>Interpretation:</b> {html.escape(finding['interpretation'])}</p>
        <p class="experiment"><b>Next experiment:</b> {html.escape(finding['experiment'])}</p></div></article>""")
    rows = []
    for item in data["metrics"]:
        rows.append(f"<tr><td>{html.escape(item['label'])}</td><td>{display(item['take_1'], item['unit'])}</td><td>{display(item['take_2'], item['unit'])}</td><td>{'yes' if item['reliable'] else 'no'}</td></tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(data['title'])} — Second Take Lab</title><style>
:root{{--ink:#102a27;--paper:#f3efe5;--coral:#ff765f;--lime:#d9ff57;--line:rgba(16,42,39,.18)}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.shell{{width:min(1040px,calc(100% - 40px));margin:auto}}
header{{padding:35px 0;border-bottom:1px solid var(--line)}}.brand{{font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.meta{{font:10px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--coral)}}
h1{{max-width:900px;margin:45px 0 18px;font-size:clamp(48px,7vw,76px);line-height:.95;letter-spacing:-.055em}}.lede{{font-size:18px;max-width:760px;opacity:.68}}
.quality{{display:grid;grid-template-columns:repeat(3,1fr);margin:38px 0;border:1px solid var(--line)}}.quality div{{padding:20px;border-right:1px solid var(--line)}}.quality div:last-child{{border:0}}.quality b{{display:block;font-size:25px}}
.chart{{width:100%;border:1px solid var(--line);margin:10px 0 50px}}h2{{font-size:38px;letter-spacing:-.04em;margin:55px 0 20px}}.finding{{display:grid;grid-template-columns:54px 1fr;gap:20px;padding:28px 0;border-top:1px solid var(--line)}}
.number{{width:43px;height:43px;display:grid;place-items:center;border-radius:50%;background:var(--ink);color:var(--paper);font:12px ui-monospace,monospace}}.candidate_gain .number{{background:#7f9f19}}.experiment{{background:rgba(255,118,95,.11);padding:13px 16px}}h3{{font-size:25px;margin:6px 0}}
table{{width:100%;border-collapse:collapse;margin-bottom:55px}}th,td{{text-align:left;padding:13px;border-bottom:1px solid var(--line)}}.boundary{{background:var(--ink);color:var(--paper);padding:28px;margin:60px 0}}footer{{padding:35px 0;border-top:1px solid var(--line);opacity:.65}}
@media(max-width:680px){{.quality{{grid-template-columns:1fr}}.quality div{{border-right:0;border-bottom:1px solid var(--line)}}}}
</style></head><body><header><div class="shell"><div class="brand">Second Take Lab</div><p class="meta">Powered by AGAPE · Comparative Audit · no transcript</p></div></header>
<main class="shell"><h1>{html.escape(data['title'])}</h1><p class="lede">A speaker-relative comparison of two takes. AGAPE reports measurable changes and candidate gains; the human decides whether each change serves the goal.</p>
<section class="quality"><div><b>{html.escape(data['quality']['confidence'])}</b><span>comparison confidence</span></div><div><b>{data['quality']['joint_visual_coverage']:.0%}</b><span>minimum visual coverage</span></div><div><b>{data['quality']['duration_similarity']:.0%}</b><span>duration similarity</span></div></section>
<img class="chart" src="timeline.png" alt="Take comparison chart"><h2>What changed</h2>{''.join(findings)}<h2>Exact comparison</h2><table><thead><tr><th>Signal</th><th>Take 1</th><th>Take 2</th><th>Reliable?</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<section class="boundary"><p class="meta">Decision boundary</p><p>{html.escape(data['comparison_boundary'])}</p><p>No transcript was created or used.</p></section></main><footer><div class="shell">Second Take Lab · Powered by AGAPE (Audiovisual Gesture And Prosody Engine)</div></footer></body></html>"""


def write_comparison_reports(data: dict, paths: JobPaths) -> None:
    make_comparison_chart(data, paths.chart)
    paths.report_md.write_text(render_markdown(data), encoding="utf-8")
    paths.report_html.write_text(render_html(data), encoding="utf-8")
    paths.features.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
