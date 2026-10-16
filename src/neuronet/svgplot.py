"""A line chart written out as SVG by hand, so plotting needs no dependency.

Draws each series as a 2px median line over a faint band spanning its range,
with a legend, direct end labels and hairline gridlines. Two themes, because
GitHub shows a README in light or dark and a chart should be drawn for each
rather than flipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

# Series colours were checked with a CVD/contrast validator against these
# surfaces; the amber and teal are the site's accents stepped into the band
# where lines stay legible.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "text": "#0b0f14",
        "muted": "#525c69",
        "grid": "#e5e8ec",
        "axis": "#b9c0c8",
        "series": ("#b8730f", "#0e9483"),
    },
    "dark": {
        "surface": "#131a22",
        "text": "#e7ecf2",
        "muted": "#8695a7",
        "grid": "#222c38",
        "axis": "#3a4656",
        "series": ("#c98500", "#22a392"),
    },
}

FONT = "'IBM Plex Sans', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Series:
    label: str
    line: Sequence[float]
    low: Sequence[float]
    high: Sequence[float]


def _nice_step(span: float, target_ticks: int) -> float:
    raw = span / max(1, target_ticks)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * magnitude >= raw:
            return m * magnitude
    return 10 * magnitude


def _fmt(v: float) -> str:
    return f"{v:.0f}" if v == int(v) else f"{v:.1f}"


def line_chart(
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    theme: str = "light",
    width: int = 880,
    height: int = 460,
) -> str:
    t = THEMES[theme]
    n = max(len(s.line) for s in series)
    y_max = max(max(s.high) for s in series)
    y_step = _nice_step(y_max, 5)
    y_top = math.ceil(y_max / y_step) * y_step
    x_step = _nice_step(n - 1, 6)

    left, right, top, bottom = 56, 150, 104, 52
    pw, ph = width - left - right, height - top - bottom

    def sx(i: float) -> float:
        return left + pw * i / max(1, n - 1)

    def sy(v: float) -> float:
        return top + ph * (1 - v / y_top)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{escape(FONT)}" role="img" '
        f'aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{t["surface"]}"/>',
        f'<text x="{left - 40}" y="34" font-size="18" font-weight="600" fill="{t["text"]}">'
        f"{escape(title)}</text>",
        f'<text x="{left - 40}" y="56" font-size="13" fill="{t["muted"]}">'
        f"{escape(subtitle)}</text>",
    ]

    # Legend: a short line key per series, label in text colour.
    lx = float(left - 40)
    for k, s in enumerate(series):
        colour = t["series"][k]
        out.append(
            f'<line x1="{lx}" y1="78" x2="{lx + 18}" y2="78" stroke="{colour}" '
            f'stroke-width="2" stroke-linecap="round"/>'
        )
        out.append(
            f'<text x="{lx + 24}" y="82" font-size="13" fill="{t["text"]}">{escape(s.label)}</text>'
        )
        lx += 24 + 7.2 * len(s.label) + 28

    # Gridlines and y ticks.
    v = 0.0
    while v <= y_top + 1e-9:
        y = sy(v)
        stroke = t["axis"] if v == 0 else t["grid"]
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{stroke}" '
            f'stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end" '
            f'fill="{t["muted"]}" style="font-variant-numeric: tabular-nums">{_fmt(v)}</text>'
        )
        v += y_step

    # X ticks.
    i = 0.0
    while i <= n - 1 + 1e-9:
        x = sx(i)
        out.append(
            f'<text x="{x:.1f}" y="{top + ph + 20}" font-size="12" text-anchor="middle" '
            f'fill="{t["muted"]}" style="font-variant-numeric: tabular-nums">{_fmt(i)}</text>'
        )
        i += x_step
    out.append(
        f'<text x="{left + pw / 2:.1f}" y="{height - 12}" font-size="12" text-anchor="middle" '
        f'fill="{t["muted"]}">{escape(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" font-size="12" '
        f'text-anchor="middle" fill="{t["muted"]}">{escape(y_label)}</text>'
    )

    # Bands first, then lines, so every line sits on top of every band.
    for k, s in enumerate(series):
        upper = " ".join(f"{sx(j):.1f},{sy(h):.1f}" for j, h in enumerate(s.high))
        lower = " ".join(f"{sx(j):.1f},{sy(lo):.1f}" for j, lo in reversed(list(enumerate(s.low))))
        out.append(
            f'<polygon points="{upper} {lower}" fill="{t["series"][k]}" fill-opacity="0.12"/>'
        )
    for k, s in enumerate(series):
        pts = " ".join(f"{sx(j):.1f},{sy(v):.1f}" for j, v in enumerate(s.line))
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{t["series"][k]}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # End dots with a surface ring, and direct labels at the right edge.
    ends = sorted(((sy(s.line[-1]), k) for k, s in enumerate(series)), reverse=True)
    placed: list[float] = []
    for y_end, k in ends:
        s = series[k]
        x_end = sx(len(s.line) - 1)
        out.append(
            f'<circle cx="{x_end:.1f}" cy="{y_end:.1f}" r="4" fill="{t["series"][k]}" '
            f'stroke="{t["surface"]}" stroke-width="2"/>'
        )
        label_y = y_end
        for p in placed:  # keep labels from colliding, with a leader if moved
            if abs(label_y - p) < 30:
                label_y = p - 30
        placed.append(label_y)
        if label_y != y_end:
            out.append(
                f'<line x1="{x_end + 6:.1f}" y1="{y_end:.1f}" x2="{x_end + 12:.1f}" '
                f'y2="{label_y:.1f}" stroke="{t["axis"]}" stroke-width="1"/>'
            )
        out.append(
            f'<text x="{x_end + 14:.1f}" y="{label_y - 2:.1f}" font-size="15" font-weight="600" '
            f'fill="{t["text"]}">{_fmt(round(s.line[-1], 1))}</text>'
        )
        out.append(
            f'<text x="{x_end + 14:.1f}" y="{label_y + 13:.1f}" font-size="11" '
            f'fill="{t["muted"]}">{escape(s.label)}</text>'
        )

    out.append("</svg>")
    return "\n".join(out) + "\n"
