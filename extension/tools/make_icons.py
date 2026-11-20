"""
Render the Captchosa mark (icons/src/*.svg) to the PNG icons the manifest and the
toolbar use. Needs cairosvg (system python has it: `python3 extension/tools/make_icons.py`).

The mark: an open, rounded "C" with a check mark whose long stroke leaves through
the C's opening, on a dark rounded tile. Variants recolour the check (and the C for
"inactive") so the toolbar shows the state at a glance.
"""

from pathlib import Path

import cairosvg

EXT = Path(__file__).resolve().parent.parent
SRC = EXT / "icons" / "src"
OUT = EXT / "icons"
SIZES = (16, 32, 48, 128)

VARIANTS = {
    # name: (tile top, tile bottom, C colour, check start, check end)
    "active": ("#1f2a44", "#0b1222", "#f1f5f9", "#fcd34d", "#f59e0b"),
    "inactive": ("#4b5563", "#374151", "#cbd5e1", "#9ca3af", "#9ca3af"),
    "solving": ("#1f2a44", "#0b1222", "#f1f5f9", "#93c5fd", "#3b82f6"),
    "error": ("#1f2a44", "#0b1222", "#f1f5f9", "#fca5a5", "#ef4444"),
}


def svg(top, bottom, c_colour, check_a, check_b, tile=True):
    bg = (f'<rect x="4" y="4" width="120" height="120" rx="30" fill="url(#bg)"/>'
          f'<rect x="4.5" y="4.5" width="119" height="119" rx="29.5" fill="none" '
          f'stroke="#ffffff" stroke-opacity=".08"/>') if tile else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bottom}"/>
    </linearGradient>
    <linearGradient id="ck" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0" stop-color="{check_a}"/><stop offset="1" stop-color="{check_b}"/>
    </linearGradient>
  </defs>
  {bg}
  <path d="M83.1 91.6 A36 36 0 1 1 83.1 36.4" fill="none" stroke="{c_colour}" stroke-width="14"
        stroke-linecap="round"/>
  <path d="M43 63 L58 78 L104 30" fill="none" stroke="url(#ck)" stroke-width="14"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    for name, colours in VARIANTS.items():
        text = svg(*colours)
        (SRC / f"logo-{name}.svg").write_text(text)
        for size in SIZES:
            target = OUT / (f"icon{size}.png" if name == "active" else f"{name}-{size}.png")
            cairosvg.svg2png(bytestring=text.encode(), write_to=str(target),
                             output_width=size, output_height=size)
    # the bare mark (no tile) for the popup header and the on-page badge
    (SRC / "mark.svg").write_text(svg(*VARIANTS["active"], tile=False))
    (EXT / "icons" / "logo.svg").write_text(svg(*VARIANTS["active"]))
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
