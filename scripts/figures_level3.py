"""SPEC §7 figure F3 (causal test) and the §13b efficacy step function."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    r3 = pd.read_parquet(config.RESULTS / "level3.parquet")
    eff = pd.read_csv(config.RESULTS / "steering_efficacy.csv")

    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.5))

    # --- panel A: the efficacy step function (§13b) ---
    a = ax[0]
    for s, g in eff.groupby("series"):
        a.plot(g.layer, g.tgt_mass, "o-", ms=2.8, label=f"{s}: target injected")
        a.plot(g.layer, g.src_mass, "--", lw=1, alpha=.5, label=f"{s}: source injected")
    a.axvline(22, color="C3", ls="-.", lw=1)
    a.axvspan(0, 21.5, color="0.5", alpha=.12)
    a.annotate("causally INERT\nLevel 3 not well posed", (10, .30), ha="center",
               fontsize=6.8, color="0.3")
    a.annotate("live", (27, .05), fontsize=7, color="C3")
    a.set_xlabel("layer"); a.set_ylabel("prob. mass on the injected name")
    a.set_title("F3a  causal accessibility is a step function at L22\n"
                "(Level 1 is flat over L5-30 — they dissociate)", fontsize=8)
    a.legend(fontsize=6, loc="upper left")

    # --- panel B: E_BC by strategy at live layers ---
    b = ax[1]
    live = r3[(r3.efficacious) & (r3.coord_layer == r3.layer)]
    labels, w = [], 0.26
    xs = np.arange(len(live))
    b.bar(xs - w, live.E_BC_linear, w, label="linear", color="C7")
    b.bar(xs, live.E_BC_manifold_unsup, w, label="manifold, unsupervised", color="C3")
    b.bar(xs + w, live.E_BC_manifold_sup, w, label="manifold, supervised", color="C0")
    for xi, (_, r) in enumerate(live.iterrows()):
        if r.degenerate:
            b.axvspan(xi - .45, xi + .45, color="0.5", alpha=.16, zorder=0)
            b.annotate("degen.", (xi, 0.06), ha="center", fontsize=5.8, color="0.25")
    b.set_xticks(xs)
    b.set_xticklabels([f"{r.series[:3]}\nL{r.layer}" for _, r in live.iterrows()], fontsize=6.5)
    b.set_ylabel("$E_{BC}$ (lower = better)")
    b.set_title("F3b  supervised beats linear; unsupervised is\nWORSE than linear "
                "at every live layer", fontsize=8)
    b.legend(fontsize=6.5)

    # --- panel C: r with CI against the pre-registered threshold ---
    c = ax[2]
    show = r3[r3.efficacious & ~r3.degenerate].copy()
    show["lbl"] = [f"{r.series[:3]} L{r.layer}" +
                   (f"←L{r.coord_layer}" if r.coord_layer != r.layer else "")
                   for _, r in show.iterrows()]
    y = np.arange(len(show))
    c.errorbar(show.r_median, y,
               xerr=[show.r_median - show.r_lo, show.r_hi - show.r_median],
               fmt="o", ms=4, capsize=2.5, color="C3")
    c.axvline(0.5, color="C0", lw=1.4)
    c.annotate("success: r < 0.5\n(§3f / §12f)", (0.5, len(show) - 0.4), fontsize=6.8,
               color="C0", ha="left")
    c.axvline(0, color="0.6", lw=.8, ls=":")
    c.set_yticks(y); c.set_yticklabels(show.lbl, fontsize=6.5)
    c.set_xlabel("$r = (E_{unsup}-E_{sup})/(E_{lin}-E_{sup})$")
    c.set_xscale("symlog", linthresh=1)
    c.set_title("F3c  the pre-registered criterion is not met\nanywhere", fontsize=8)

    fig.tight_layout()
    fig.savefig(config.FIGURES / "F3_causal_test.png", bbox_inches="tight")
    print("wrote F3")


if __name__ == "__main__":
    main()
