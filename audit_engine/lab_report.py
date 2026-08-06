from __future__ import annotations

import html
import json
import math

from .report import json_safe
from .storage import JobPaths


def num(value, *, percent: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{value:.0%}" if percent else f"{value:.2f}"


def render_markdown(data: dict) -> str:
    result = data["result"]
    lines = [
        f"# {data['title']}",
        "",
        "Second Take Lab — AGAPE self-supervised calibration",
        "",
        f"**Overall:** {'PASS' if result['overall_passed'] else 'NEEDS WORK'}",
        "",
        f"- Repeatability control: {'pass' if result['repeatability_passed'] else 'fail'}",
        f"- Known delays detected: {result['delay_tests_passed']}/{result['delay_tests_total']}",
        f"- Directional sensitivity: {result['directional_sensitivity']:.0%}",
        "",
        "## Experiments",
        "",
    ]
    for test in [data["control"], *data["delay_tests"]]:
        metrics = test["metrics"]
        lines.extend([
            f"### {'PASS' if test['passed'] else 'FAIL'} — {test['kind']} ({test['expected_delay_seconds']:.2f}s)",
            "",
            f"- Alignment: {num(metrics['alignment_before'], percent=True)} → {num(metrics['alignment_after'], percent=True)}",
            f"- Matched-event lag: {num(metrics['matched_lag_before'])}s → {num(metrics['matched_lag_after'])}s",
            f"- Energy/motion correlation: {num(metrics['correlation_before'])} → {num(metrics['correlation_after'])}",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "This calibration tests whether AGAPE detects known synchronization changes. It does not establish that any delivery style is universally better. No transcript was created or used.",
        "",
        "Temporary source and perturbation videos were deleted after successful calibration unless retention was explicitly enabled.",
    ])
    return "\n".join(lines) + "\n"


def render_html(data: dict) -> str:
    result = data["result"]
    cards = []
    for test in [data["control"], *data["delay_tests"]]:
        metrics = test["metrics"]
        cards.append(f"""<article class="test {'pass' if test['passed'] else 'fail'}">
        <p class="meta">{'pass' if test['passed'] else 'fail'} · known delay {test['expected_delay_seconds']:.2f}s</p>
        <h3>{html.escape(test['kind'].replace('_', ' ').title())}</h3>
        <div class="measure"><span>Alignment</span><b>{num(metrics['alignment_before'], percent=True)} → {num(metrics['alignment_after'], percent=True)}</b></div>
        <div class="measure"><span>Matched lag</span><b>{num(metrics['matched_lag_before'])}s → {num(metrics['matched_lag_after'])}s</b></div>
        <div class="measure"><span>Energy/motion correlation</span><b>{num(metrics['correlation_before'])} → {num(metrics['correlation_after'])}</b></div>
        </article>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(data['title'])}</title><style>
    :root{{--ink:#102a27;--paper:#f3efe5;--coral:#ff765f;--green:#7f9f19;--line:rgba(16,42,39,.18)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.shell{{width:min(1040px,calc(100% - 40px));margin:auto}}header{{padding:35px 0;border-bottom:1px solid var(--line)}}.brand{{font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.meta{{font:10px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--coral)}}h1{{font-size:clamp(48px,7vw,76px);line-height:.95;letter-spacing:-.055em;margin:50px 0 18px}}.verdict{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);margin:40px 0 60px}}.verdict div{{padding:22px;border-right:1px solid var(--line)}}.verdict div:last-child{{border:0}}.verdict b{{display:block;font-size:28px}}h2{{font-size:38px;letter-spacing:-.04em}}.tests{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}.test{{border:1px solid var(--line);padding:24px}}.test.pass{{border-top:5px solid var(--green)}}.test.fail{{border-top:5px solid var(--coral)}}.test h3{{font-size:24px;margin:8px 0 20px}}.measure{{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:11px 0;gap:20px}}.boundary{{background:var(--ink);color:var(--paper);padding:28px;margin:60px 0}}footer{{padding:35px 0;border-top:1px solid var(--line)}}@media(max-width:700px){{.tests,.verdict{{grid-template-columns:1fr}}.verdict div{{border-right:0;border-bottom:1px solid var(--line)}}}}
    </style></head><body><header><div class="shell"><div class="brand">Second Take Lab</div><p class="meta">Powered by AGAPE · Self-supervised calibration</p></div></header><main class="shell"><h1>{html.escape(data['title'])}</h1><p>Controlled audiovisual perturbations with known ground truth.</p><section class="verdict"><div><b>{'PASS' if result['overall_passed'] else 'NEEDS WORK'}</b><span>overall</span></div><div><b>{result['directional_sensitivity']:.0%}</b><span>directional sensitivity</span></div><div><b>{result['delay_tests_passed']}/{result['delay_tests_total']}</b><span>known delays detected</span></div></section><h2>Experiments</h2><section class="tests">{''.join(cards)}</section><section class="boundary"><p class="meta">Boundary</p><p>This tests detection of known synchronization changes—not whether a delivery style is universally better. No transcript was created or used.</p></section></main><footer><div class="shell">AGAPE — Audiovisual Gesture And Prosody Engine</div></footer></body></html>"""


def write_lab_reports(data: dict, paths: JobPaths) -> None:
    paths.report_md.write_text(render_markdown(data), encoding="utf-8")
    paths.report_html.write_text(render_html(data), encoding="utf-8")
    paths.features.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
