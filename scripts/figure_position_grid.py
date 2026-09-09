"""F9 — the (layer × position) selection grid (§18.24/§18.25/§18.26).

The direct analogue of Arditi et al.'s Figure 11, computed for
Llama-3.1-8B-Instruct rather than Llama-3-8B-Instruct.

Design notes, so the choices are inspectable rather than taste:

- The job is MAGNITUDE ("where does ablation bypass refusal"), so the encoding is
  a SEQUENTIAL single-hue ramp, light->dark. Not a rainbow, and not diverging:
  although the score is signed, the negative tail bottoms out at -1.18 against a
  positive range to +19.32, so a diverging scale centred at zero would spend half
  its range on 6% of the data. The floor is clamped at 0 and the caption says so.
- Cells that carry an argument are directly labelled with their value, so the
  reading never depends on colour alone.
- Style matches the existing figure set (figures_stageb.py): dpi 130, font 8,
  tight_layout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": False})

TOKENS = {-1: "'\\n\\n'", -2: "<|end_header_id|>", -3: "'assistant'",
          -4: "<|start_header_id|>", -5: "<|eot_id|>"}


def main():
    g = json.loads((config.RESULTS / "stageb_18_24_grid.json").read_text())
    grid = np.array(g["grid"])                       # [layer, position]
    positions = g["positions"]
    G = grid.T                                       # [position, layer], wide aspect

    fig, ax = plt.subplots(figsize=(12.2, 3.5))
    im = ax.imshow(np.clip(G, 0, None), aspect="auto", cmap="Blues",
                   vmin=0, vmax=float(G.max()), interpolation="nearest")

    ax.set_xticks(range(0, 32, 2))
    ax.set_xticklabels(range(0, 32, 2))
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels([f"{p}   {TOKENS[p]}" for p in positions], fontsize=7)
    ax.set_xlabel("layer the direction is read from")
    ax.set_ylabel("read position")
    ax.set_xticks(np.arange(-.5, 32, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(positions), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.026)
    cb.set_label("bypass score  (baseline − ablated refusal-opener log-odds)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # Cells are outlined, never over-plotted with floating text: a label placed on
    # the grid either collides with a neighbouring row or is clipped at the axis.
    # Identity moves to the legend, which also carries each cell's value, so the
    # reading never depends on outline colour alone.
    handles = []
    for layer, pos, color, label in [
        (12, -5, "C3", "Arditi Table 5 selection for Llama-3-8B — and our argmax"),
        (11, -1, "C2", "second-highest cell overall"),
        (10, -1, "C0", "inherited from the capture default — rank 5 of 160"),
    ]:
        yi = positions.index(pos)
        ax.add_patch(plt.Rectangle((layer - .5, yi - .5), 1, 1, fill=False,
                                   edgecolor=color, linewidth=2.2, zorder=5))
        handles.append(plt.Line2D([0], [0], marker="s", color="none", markersize=8,
                                  markerfacecolor="none", markeredgecolor=color,
                                  markeredgewidth=2.2,
                                  label=f"({layer}, {pos})  {G[yi, layer]:.1f}   {label}".replace("-", "\u2212")))
    # Below the axes, not inside them: an in-plot legend here covers live cells on
    # the -1 row, and obscuring data to label it is not a trade worth making.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.30),
              ncol=3, fontsize=6.3, framealpha=1.0, borderpad=0.5,
              handletextpad=0.7, columnspacing=1.4, frameon=False)

    ax.annotate("−4 is a pure template token: dead at every layer (max 2.6)",
                (23.5, positions.index(-4)), ha="center", va="center", fontsize=6.4,
                color="0.25", bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                        ec="0.6", lw=0.7), zorder=6)

    ax.set_title("F9  refusal-direction selection grid, Llama-3.1-8B-Instruct — "
                 "the analogue of Arditi et al. Fig. 11 for 3.1\n"
                 "our argmax (12, −5) is the cell Arditi publish for Llama-3-8B; "
                 "the cell we inherited by library default ranks 5 of 160",
                 fontsize=8)
    fig.tight_layout()
    out = config.FIGURES / "F9_position_grid.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  score range {G.min():+.2f} to {G.max():+.2f} "
          f"(colour floor clamped at 0; {int((G < 0).sum())} of {G.size} cells negative)")


if __name__ == "__main__":
    main()
