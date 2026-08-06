from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .storage import JobPaths


def safe_number(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "not available"
    return f"{value:.{digits}f}"


def json_safe(value):
    """Return standards-compliant JSON data, replacing NaN/Inf with null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def make_chart(data: dict, output: Path) -> None:
    timeline = data["fusion"]["timeline"]
    times = np.array([row["time"] for row in timeline]) / 60.0
    energy = np.array([row["audio_energy_z"] for row in timeline])
    motion = np.array([row["body_motion_z"] for row in timeline])
    facing = np.array([row["camera_facing"] for row in timeline])
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.8), sharex=True, gridspec_kw={"hspace": 0.12})
    fig.patch.set_facecolor("#f3efe5")
    for axis in axes:
        axis.set_facecolor("#f3efe5")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.14)
        axis.tick_params(labelsize=8, colors="#405651")
    axes[0].plot(times, energy, color="#ff765f", linewidth=1.8)
    axes[0].axhline(0, color="#102a27", linewidth=0.6, alpha=0.3)
    axes[0].set_ylabel("vocal\nenergy", rotation=0, ha="right", va="center", fontsize=9)
    axes[1].plot(times, motion, color="#102a27", linewidth=1.8)
    axes[1].axhline(0, color="#102a27", linewidth=0.6, alpha=0.3)
    axes[1].set_ylabel("body\nmotion", rotation=0, ha="right", va="center", fontsize=9)
    axes[2].plot(times, facing, color="#7f9f19", linewidth=1.8)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("camera-facing\nproxy", rotation=0, ha="right", va="center", fontsize=9)
    axes[2].set_xlabel("shared media clock (minutes)", fontsize=9)
    for pause in data["audio"]["pauses"]:
        for axis in axes:
            axis.axvspan(pause["start"] / 60, pause["end"] / 60, color="#102a27", alpha=0.055)
    fig.suptitle("Synchronized delivery timeline", x=0.075, ha="left", color="#102a27", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_markdown(data: dict) -> str:
    lines = [
        f"# {data['title']}",
        "",
        "Second Take Lab — synchronized audiovisual audit powered by AGAPE",
        "",
        "## Interpretation boundary",
        "",
        "This report uses no transcript. Camera-facing is a head-direction proxy, not verified eye contact. Observations describe visible or acoustic patterns and do not infer emotion, personality, honesty, diagnosis, or intent.",
        "",
        "## Priority findings",
        "",
    ]
    for index, finding in enumerate(data["fusion"]["findings"], start=1):
        stamps = ", ".join(item["label"] for item in finding["evidence"]) or "whole-sample pattern"
        lines.extend([
            f"### {index}. {finding['title']}",
            "",
            f"**Observed:** {finding['observation']}",
            "",
            f"**Next-take experiment:** {finding['experiment']}",
            "",
            f"**Evidence:** {stamps} · **confidence:** {finding['confidence']}",
            "",
        ])
    summary = data["fusion"]["summary"]
    lines.extend([
        "## Synchronized metrics",
        "",
        f"- Analyzed duration: {summary['duration_seconds']:.1f} seconds",
        f"- Face tracker coverage: {data['vision']['summary']['face_coverage']:.0%}",
        f"- Pose tracker coverage: {data['vision']['summary']['pose_coverage']:.0%}",
        f"- Hand visibility coverage: {data['vision']['summary']['hand_coverage']:.0%}",
        f"- Vocal emphasis events: {summary['vocal_emphasis_events']}",
        f"- Emphasis/movement alignment: {summary['emphasis_motion_alignment_ratio']:.0%}",
        f"- Energy/movement correlation: {safe_number(summary['crossmodal_energy_motion_correlation'])}",
        "",
        "## Storage receipt",
        "",
        "Large media derivatives are deleted after a successful run unless retention was explicitly enabled. See `manifest.json` for the exact receipt.",
    ])
    return "\n".join(lines) + "\n"


def render_html(data: dict, markdown: str) -> str:
    findings = []
    for index, finding in enumerate(data["fusion"]["findings"], start=1):
        stamps = " · ".join(item["label"] for item in finding["evidence"]) or "whole-sample pattern"
        findings.append(f"""
        <article class="finding {html.escape(finding['kind'])}">
          <div class="finding-number">{index:02d}</div>
          <div><p class="meta">{html.escape(finding['confidence'])} confidence · {html.escape(stamps)}</p>
          <h3>{html.escape(finding['title'])}</h3>
          <p>{html.escape(finding['observation'])}</p>
          <p class="experiment"><b>Try on the next take:</b> {html.escape(finding['experiment'])}</p></div>
        </article>""")
    summary = data["fusion"]["summary"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(data['title'])} — Second Take Lab</title>
<style>
:root{{--ink:#102a27;--paper:#f3efe5;--coral:#ff765f;--lime:#d9ff57;--line:rgba(16,42,39,.18)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.shell{{width:min(1040px,calc(100% - 40px));margin:auto}} header{{padding:42px 0 34px;border-bottom:1px solid var(--line)}}
.brand{{font-weight:750;letter-spacing:.08em;text-transform:uppercase}} .eyebrow,.meta{{font:10px/1.4 ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--coral)}}
h1{{max-width:820px;margin:40px 0 16px;font-size:clamp(48px,7vw,78px);line-height:.94;letter-spacing:-.055em}} .lede{{max-width:720px;font-size:18px;opacity:.68}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);margin:45px 0;border:1px solid var(--line)}} .metric{{padding:22px;border-right:1px solid var(--line)}} .metric:last-child{{border:0}}
.metric b{{display:block;font-size:28px;letter-spacing:-.04em}} .metric span{{font-size:11px;opacity:.55}} .chart{{width:100%;margin:20px 0 55px;border:1px solid var(--line)}}
h2{{margin:65px 0 25px;font-size:40px;letter-spacing:-.04em}} .finding{{display:grid;grid-template-columns:56px 1fr;gap:22px;padding:28px 0;border-top:1px solid var(--line)}}
.finding-number{{width:44px;height:44px;display:grid;place-items:center;border-radius:50%;background:var(--ink);color:var(--paper);font:12px ui-monospace,monospace}} .finding.strength .finding-number{{background:#7f9f19}}
.finding h3{{margin:7px 0 9px;font-size:25px;letter-spacing:-.03em}} .finding p{{margin:0 0 12px;max-width:800px}} .experiment{{padding:13px 16px;background:rgba(255,118,95,.11)}}
.boundary{{margin:70px 0;padding:28px;background:var(--ink);color:var(--paper)}} footer{{padding:35px 0 55px;border-top:1px solid var(--line);font-size:12px;opacity:.62}}
@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}.metric:nth-child(2){{border-right:0}}.metric:nth-child(-n+2){{border-bottom:1px solid var(--line)}}}}
</style></head><body>
<header><div class="shell"><div class="brand">Second Take Lab</div><p class="eyebrow">Powered by AGAPE · synchronized audiovisual audit · no transcript</p></div></header>
<main class="shell"><h1>{html.escape(data['title'])}</h1><p class="lede">A time-coded review of vocal delivery and visible movement on one shared media clock.</p>
<section class="metrics">
<div class="metric"><b>{summary['duration_seconds'] / 60:.1f}m</b><span>analyzed</span></div>
<div class="metric"><b>{data['vision']['summary']['face_coverage']:.0%}</b><span>face coverage</span></div>
<div class="metric"><b>{data['vision']['summary']['pose_coverage']:.0%}</b><span>pose coverage</span></div>
<div class="metric"><b>{summary['emphasis_motion_alignment_ratio']:.0%}</b><span>emphasis alignment</span></div>
</section>
<img class="chart" src="timeline.png" alt="Synchronized timeline chart">
<h2>Priority findings</h2>{''.join(findings)}
<section class="boundary"><p class="eyebrow">Interpretation boundary</p><p>This report uses no transcript. Camera-facing is a head-direction proxy, not verified eye contact. Observations describe measured patterns and do not infer emotion, personality, honesty, diagnosis, or intent.</p></section>
</main><footer><div class="shell">Second Take Lab · Powered by AGAPE (Audiovisual Gesture And Prosody Engine)</div></footer>
</body></html>"""


def write_reports(data: dict, paths: JobPaths) -> None:
    make_chart(data, paths.chart)
    markdown = render_markdown(data)
    paths.report_md.write_text(markdown, encoding="utf-8")
    paths.report_html.write_text(render_html(data, markdown), encoding="utf-8")
    paths.features.write_text(
        json.dumps(json_safe(data), indent=2, allow_nan=False), encoding="utf-8"
    )
