"""Figures for §18 — replacing the panels whose claims were withdrawn.

F4b and F5c plot the geodesic/chord straightness result (§18.10: the statistic
does not measure curvature), the tangent-`v_ref` cosines (§15d), and the
transductive AUC (N10). Those panels cannot be reused. These are their
replacements plus the new causal results.

Colour is assigned to the ENTITY and held fixed across every figure, so a reader
who learns "red is the supervised direction" in F12 reads F11 the same way:

    v_ref (supervised)      C3   red
    PC1  (trivial baseline) C2   green
    d    (Isomap)           C0   blue
    nulls / random          grey

Sequential single-hue is used only where the job is magnitude (F9). Nothing here
uses two y-scales on one axis; where two measures must be compared across layers
they are stacked panels sharing an x-axis, or a scatter of one against the other.
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

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})

C_VREF, C_PC1, C_D, C_NULL = "C3", "C2", "C0", "0.55"
R = config.RESULTS


def load(name):
    return json.loads((R / name).read_text())


# ---------------------------------------------------------------- F10 structure
def f10():
    idr = load("stageb_18_4f.json")
    powr = load("stageb_18_4e.json")
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.5))

    # (a) ID against the noise floor, not against the bank alone
    a = ax[0]
    lo, hi = idr["synthetic_1d_id_ci"]
    a.axhspan(lo, hi, color=C_NULL, alpha=0.25, zorder=0,
              label=f"a TRUE 1-D structure\nwith this data's noise: {idr['synthetic_1d_id_mean']:.2f}")
    a.axhline(idr["synthetic_1d_id_mean"], color=C_NULL, lw=1.5, ls="--", zorder=1)
    for x, key, col, lab in [(0, "matched_refusal", C_D, "refusal set"),
                             (1, "matched_bank", C_NULL, "instruction bank")]:
        m = idr[f"{key}_mean"]
        c0, c1 = idr[f"{key}_ci"]
        a.errorbar([x], [m], yerr=[[m - c0], [c1 - m]], fmt="o", ms=7, color=col,
                   capsize=4, lw=2, zorder=3)
        a.annotate(f"{m:.2f}", (x, m), xytext=(9, 0), textcoords="offset points",
                   va="center", fontsize=7.5, color=col, fontweight="bold")
    a.set_xticks([0, 1]); a.set_xticklabels(["refusal", "bank"])
    a.set_xlim(-0.5, 1.6); a.set_ylabel("intrinsic dimension (Levina–Bickel MLE)")
    a.legend(fontsize=6.3, loc="lower right")
    a.set_title("F10a  ID ≈ 5 IS the noise floor (§18.14)\n"
                "n-matched at m=480; a 1-D structure reads 5.43", fontsize=8)

    # (b) the falsification: does the statistic respond to known curvature?
    b = ax[1]
    from scripts.n1_curvature_falsification import run_one, half_angle_for_ratio
    rng = np.random.default_rng(config.SEED)
    truth, logged, ambient = [], [], []
    # the same requested ratios §18.10 swept, so `truth` reproduces its table
    for r in [1.000, 1.010, 1.050, 1.250, 1.500]:
        res = run_one(600, half_angle_for_ratio(r), 64, 0.0, 1.0, 12, rng)
        truth.append(res["truth"]); logged.append(res["logged"]); ambient.append(res["ambient"])
    b.plot(truth, truth, ls=":", color="0.6", lw=1.5, label="perfect recovery")
    b.plot(truth, ambient, "o-", color=C_D, lw=2, ms=6,
           label="geodesic / ambient chord")
    b.plot(truth, logged, "s-", color=C_VREF, lw=2, ms=6,
           label="geodesic / embedded chord\n(the statistic §14a used)")
    b.annotate(f"returns {logged[-1]:.3f} on a manifold\n"
               f"whose true ratio is {truth[-1]:.3f}", (truth[-1], logged[-1]),
               xytext=(-6, 20), textcoords="offset points", ha="right", fontsize=6.6,
               color=C_VREF, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_VREF, lw=0.7))
    b.set_xlabel("true geodesic/chord ratio (known by construction)")
    b.set_ylabel("measured ratio")
    b.legend(fontsize=6.3, loc="upper left")
    b.set_title("F10b  the straightness statistic is FLAT in curvature\n"
                "σ=0, so nothing is hidden by noise (§18.10)", fontsize=8)

    # (c) power: observed against its own null, per k
    c = ax[2]
    ks = sorted(int(k) for k in powr["per_k"])
    obs = [powr["per_k"][str(k)]["observed_ambient"] for k in ks]
    p95 = [powr["per_k"][str(k)]["ambient_null_p95"] for k in ks]
    x = np.arange(len(ks))
    c.bar(x - 0.19, p95, 0.36, color=C_NULL, label="null 95th pct (straight + matched noise)")
    c.bar(x + 0.19, obs, 0.36, color=C_D, label="observed")
    for xi, (o, p) in enumerate(zip(obs, p95)):
        c.annotate(f"{o:.2f}", (xi + 0.19, o), xytext=(0, 2), textcoords="offset points",
                   ha="center", fontsize=6.4, color=C_D)
    c.set_xticks(x); c.set_xticklabels([f"k={k}" for k in ks])
    c.set_ylabel("geodesic / ambient chord")
    c.legend(fontsize=6.3, loc="upper right")
    c.set_title("F10c  observed clears the null at every k —\n"
                "but MDR > 1.50, so no curvature MAGNITUDE is claimable", fontsize=8)

    fig.tight_layout(); fig.savefig(config.FIGURES / "F10_structure_corrected.png",
                                    bbox_inches="tight")
    print("wrote F10_structure_corrected.png")


# ----------------------------------------------------------- F11 discrimination
def f11():
    ac = load("stageb_18_4ac.json")
    bg = load("stageb_18_4bg.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))

    a = ax[0]
    layers = sorted(int(l) for l in ac)
    held = [ac[str(l)]["auroc_holdout"] for l in layers]
    trans = [ac[str(l)]["auroc_transductive_reproduction"] for l in layers]
    nullp95 = bg["random_null"]["auroc"]["p95"]
    a.axhspan(0.5, nullp95, color=C_NULL, alpha=0.22, zorder=0,
              label=f"random directions, ≤p95 ({nullp95:.3f})")
    a.plot(layers, held, "o-", color=C_D, lw=2, ms=7, zorder=3,
           label="held out (fit on 400, no XSTest)")
    a.plot(layers, trans, "s--", color=C_VREF, lw=1.6, ms=6, alpha=0.85, zorder=2,
           label="the logged construction (fit ON the eval set)")
    for l, h in zip(layers, held):
        a.annotate(f"{h:.3f}", (l, h), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=6.4, color=C_D)
    a.set_xlabel("layer"); a.set_ylabel("XSTest AUROC (safe vs contrast)")
    a.set_ylim(0.45, 1.04); a.set_xticks(layers)
    a.legend(fontsize=6.3, loc="lower right")
    a.set_title("F11a  the headline survives decontamination\n"
                "0.9935 held out vs 0.9951 transductive at L28 (§18.12)", fontsize=8)

    b = ax[1]
    rows = bg["panel"]
    names = ["Isomap coord → d", "PC1 of the fit bank", "k-means (k=2)"]
    cols = [C_D, C_PC1, "C1"]
    xs = np.arange(3)
    cos = [rows[i]["cosine"] for i in range(3)]
    auc = [rows[i]["auroc"] for i in range(3)]
    b.bar(xs - 0.19, cos, 0.36, color=cols, alpha=0.95, label="cos(·, v_ref)")
    b.bar(xs + 0.19, auc, 0.36, color=cols, alpha=0.5, hatch="///",
          edgecolor="white", label="held-out AUROC")
    for xi, (cv, av) in enumerate(zip(cos, auc)):
        b.annotate(f"{cv:.3f}", (xi - 0.19, cv), xytext=(0, 2), textcoords="offset points",
                   ha="center", fontsize=6.4)
        b.annotate(f"{av:.3f}", (xi + 0.19, av), xytext=(0, 2), textcoords="offset points",
                   ha="center", fontsize=6.4)
    b.axhline(bg["random_null"]["cosine"]["p95"], color=C_NULL, ls=":", lw=1.4)
    b.annotate("random-direction cosine p95 = 0.030", (2.45, 0.055), ha="right",
               fontsize=6.3, color="0.35")
    b.set_xticks(xs); b.set_xticklabels(names, fontsize=7)
    b.set_ylim(0, 1.12)
    b.set_ylabel("cosine  /  AUROC   (both unitless, 0–1)")
    b.legend(fontsize=6.3, loc="upper left", ncol=2)
    b.set_title("F11b  PC1 matches on AUROC and BEATS on alignment\n"
                "the manifold machinery is not load-bearing (§18.13)", fontsize=8)

    fig.tight_layout(); fig.savefig(config.FIGURES / "F11_discrimination.png",
                                    bbox_inches="tight")
    print("wrote F11_discrimination.png")


# ------------------------------------------------------------------- F12 causal
def f12():
    sw = load("stageb_18_5_sweep.json")
    rr = load("stageb_18_5_rerun.json")
    l10 = load("stageb_18_21_layer10.json")
    ac = load("stageb_18_4ac.json")
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.5))

    a = ax[0]
    ls = sorted(int(k) for k in sw["layers"])
    drops = [sw["layers"][str(l)]["drop"] for l in ls]
    a.plot(ls, drops, "-", color=C_VREF, lw=2)
    a.fill_between(ls, 0, drops, color=C_VREF, alpha=0.16)
    for l, lab, col, off in [(10, "L10  0.947", C_VREF, (14, -2)),
                             (28, "L28  0.066", C_D, (0, 14))]:
        a.plot([l], [sw["layers"][str(l)]["drop"]], "o", ms=8, color=col, zorder=4)
        a.annotate(lab, (l, sw["layers"][str(l)]["drop"]), xytext=off,
                   textcoords="offset points", ha="left" if off[0] else "center",
                   va="center", fontsize=7, color=col, fontweight="bold")
    a.set_ylim(-0.06, 1.12)
    a.set_xlabel("layer the direction is read from")
    a.set_ylabel("refusal-rate drop under ablation")
    a.set_title("F12a  the causal window is layers 7–15\n"
                "L28 — where all the geometry was measured — is inert (§18.20)", fontsize=8)

    b = ax[1]
    arms = [("v_ref", "1_v_ref", C_VREF), ("PC1", "6_pc1", C_PC1),
            ("d", "2_d", C_D), ("d⊥", "3_d_perp", C_NULL)]
    vals = [rr["drops"][k] for _, k, _ in arms]
    cols = [c for _, _, c in arms]
    b.bar(range(4), vals, 0.6, color=cols)
    b.axhline(rr["constrained_mean_drop"], color="0.25", ls="--", lw=1.5,
              label=f"random at cos={rr['target_cos']:.2f} to v_ref: "
                    f"{rr['constrained_mean_drop']:.3f}")
    b.axhline(rr["random_mean_drop"], color=C_NULL, ls=":", lw=1.4,
              label=f"unconstrained random: {rr['random_mean_drop']:.3f}")
    b.legend(fontsize=6.2, loc="upper right", framealpha=0.95)
    for i, v in enumerate(vals):
        b.annotate(f"{v:.3f}", (i, v), xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=7, fontweight="bold", color=cols[i])
    b.set_xticks(range(4)); b.set_xticklabels([n for n, _, _ in arms])
    b.set_ylabel("refusal-rate drop"); b.set_ylim(0, 1.08)
    b.set_title("F12b  ablation at L10: PC1 ≡ v_ref, d sits in its\n"
                "own cos-matched null, d⊥ is nothing (§18.21)", fontsize=8)

    c = ax[2]
    pts = {8: ac["8"]["auroc_holdout"], 10: l10["10"]["auroc_d"], 11: l10["11"]["auroc_d"],
           22: ac["22"]["auroc_holdout"], 28: ac["28"]["auroc_holdout"],
           31: ac["31"]["auroc_holdout"]}
    xs = [sw["layers"][str(l)]["drop"] for l in pts]
    ys = list(pts.values())
    c.scatter(xs, ys, s=70, color=[C_VREF if l in (10, 11) else C_D for l in pts], zorder=3)
    offs = {8: (9, -2), 10: (-10, -12), 11: (-10, 8), 22: (7, 6), 28: (7, -10), 31: (7, 3)}
    for l, x, y in zip(pts, xs, ys):
        c.annotate(f"L{l}", (x, y), xytext=offs[l], textcoords="offset points",
                   fontsize=7.5, ha="right" if offs[l][0] < 0 else "left")
    c.set_xlim(-0.08, 1.06)
    c.margins(y=0.16)
    c.set_xlabel("causal: refusal-rate drop under ablation")
    c.set_ylabel("representational: held-out XSTest AUROC")
    c.set_title("F12c  the dissociation — decodable late,\n"
                "manipulable early (§18.21)", fontsize=8)

    fig.tight_layout(); fig.savefig(config.FIGURES / "F12_causal.png", bbox_inches="tight")
    print("wrote F12_causal.png")


# ------------------------------------------------------------ F13 dose-response
def f13():
    d = load("stageb_18_29_dose.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, col, lab in [("v_ref", C_VREF, "v_ref (supervised)"),
                           ("pc1", C_PC1, "PC1 (unsupervised, trivial)"),
                           ("d", C_D, "d (Isomap coordinate)")]:
        cur = d["curves"][name]
        ok = [c for c in cur if c["passes_gate"]]
        bad = [c for c in cur if not c["passes_gate"]]
        ax.plot([c["multiplier"] for c in ok], [c["delta"] for c in ok], "o-",
                color=col, lw=2, ms=6, label=lab)
        if bad:
            ax.plot([c["multiplier"] for c in bad], [c["delta"] for c in bad], "o--",
                    color=col, lw=1.2, ms=6, mfc="white", alpha=0.55)
    ax.annotate("hollow markers + dashed: that dose fails the\n"
                "coherence gate for THAT direction — the model is\n"
                "degrading, and log-odds falls because the text is\n"
                "degenerate, not because steering stopped working.\n"
                "d still passes at 2×, where v_ref and PC1 do not.",
                (0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=6.2, color="0.3",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.7))
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlabel("dose, matched on each direction's own harmful−harmless projection gap")
    ax.set_ylabel("Δ refusal-opener log-odds")
    ax.legend(fontsize=6.6, loc="upper left")
    ax.set_title("F13  steering (sufficiency) at L10 on a non-saturating readout\n"
                 "PC1 tracks v_ref within 1.8 log-odds; d lags by up to 13 (§18.30)",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "F13_dose_response.png",
                                    bbox_inches="tight")
    print("wrote F13_dose_response.png")


if __name__ == "__main__":
    f10(); f11(); f12(); f13()
