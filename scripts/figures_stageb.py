"""SPEC §7 figures F4 (refusal fork) and F5 (behavioural check)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

import config
from src import geometry as G

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False})
SB = config.DATA / "stageb"


def main():
    st = pd.read_parquet(config.RESULTS / "stageb_structure.parquet")
    va = pd.read_parquet(config.RESULTS / "stageb_validation.parquet")
    df = pd.read_csv(SB / "prompts_with_behavior.csv")
    A = torch.load(config.ACTS / "stageb.pt", weights_only=False)["acts"].numpy()

    # ---------------- F4: the fork ----------------
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.4))
    a = ax[0]
    x = np.arange(len(st))
    a.bar(x - .2, st.id_mle, .4, label="refusal set (MLE)", color="C3")
    a.bar(x + .2, st.id_bank_mle, .4, label="general-instruction bank", color="0.65")
    a.axhline(1.0, color="C0", lw=1.2, ls="--")
    a.annotate("ID = 1 (the fork's\nstraight-line branch)", (0.5, 1.15), fontsize=6.5, color="C0")
    a.set_xticks(x); a.set_xticklabels([f"L{l}" for l in st.layer])
    a.set_ylabel("intrinsic dimension")
    a.set_title("F4a  ID is ~5 — but NOT elevated over\na generic prompt bank (§14a)", fontsize=8)
    a.legend(fontsize=6.5)

    b = ax[1]
    b.errorbar(st.layer, st.geo_chord_mean, yerr=st.geo_chord_iqr / 2,
               fmt="o-", ms=4, capsize=3, color="C3")
    b.axhline(1.0, color="C0", lw=1.2, ls="--")
    b.annotate("1.0 = perfectly straight", (8, 1.002), fontsize=6.8, color="C0")
    b.set_ylim(0.98, 1.10)
    b.set_xlabel("layer"); b.set_ylabel("geodesic / chord")
    b.set_title("F4b  the structure is STRAIGHT\n(1.010-1.020 at every layer)", fontsize=8)

    c = ax[2]
    for _, r in st.iterrows():
        cosv = json.loads(r.tangent_cos)
        c.plot(np.linspace(0, 1, len(cosv)), cosv, "o-", ms=2.6, alpha=.8,
               label=f"L{int(r.layer)} (coord~v_ref {r.coord_vs_vref_cos:.2f})")
    c.set_ylim(0, 1); c.set_xlabel("position along the principal curve")
    c.set_ylabel("|cos(tangent, v_ref)|")
    c.set_title("F4c  per-tangent cosines are noisy in a\n5-D cloud; the scalar coord "
                "alignment is primary", fontsize=8)
    c.legend(fontsize=5.8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "F4_refusal_fork.png", bbox_inches="tight")

    # ---------------- F5: behaviour + the natural experiment ----------------
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.4))
    prim = df.in_primary.to_numpy()
    d = df[prim].reset_index(drop=True)
    X = np.ascontiguousarray(A[prim][:, config.WURGAFT_LAYER, :])
    Xp, _ = G.pca_project(X, config.PCA_DIM)
    ids = G.intrinsic_dimension(Xp)
    Y, _, _ = G.isomap_embed(Xp, max(2, G.d_hat_from(ids, "id_mle")), 12)
    coord = Y[:, 0]
    if spearmanr(coord, d.refusal_prob).statistic < 0:
        coord = -coord
    cols = {"harmful": "C3", "borderline": "C1", "harmless": "C0"}
    for s, col in cols.items():
        m = (d.source == s).to_numpy()
        ax[0].scatter(coord[m], d.refusal_prob[m], s=9, alpha=.65, color=col, label=s)
    rho = spearmanr(coord, d.refusal_prob).statistic
    ax[0].set_xlabel("unsupervised first coordinate (no labels used)")
    ax[0].set_ylabel("refusal probability")
    ax[0].set_title(f"F5  recovered coordinate vs behaviour\nSpearman {rho:+.3f} "
                    f"(layer {config.WURGAFT_LAYER})", fontsize=8)
    ax[0].legend(fontsize=6.5)

    b = ax[1]
    b.plot(va.layer, va.rho_geo_behaviour, "o-", ms=4, color="C0", label="behaviour")
    b.plot(va.layer, va.partial_geo_beh_given_edit, "o--", ms=3.4, color="C0", alpha=.6,
           label="behaviour | surface")
    b.plot(va.layer, va.rho_geo_edit, "s-", ms=4, color="C3", label="surface form")
    b.plot(va.layer, va.partial_geo_edit_given_beh, "s--", ms=3.4, color="C3", alpha=.6,
           label="surface | behaviour")
    b.plot(va.layer, va.rho_geo_len, ":", color="0.5", label="prompt length")
    b.set_xlabel("layer"); b.set_ylabel("Spearman vs recovered geodesics")
    b.set_title("F5b  §12h control: behaviour dominates surface\n"
                "7:1 at L28 — the opposite of Stage A (§12g)", fontsize=8)
    b.legend(fontsize=6); b.set_ylim(0, .8)

    c = ax[2]
    m = df.source.isin(["borderline", "xstest_contrast"]).to_numpy()
    sub = df[m].reset_index(drop=True)
    Xc = np.ascontiguousarray(A[m][:, config.WURGAFT_LAYER, :])
    Xcp, _ = G.pca_project(Xc, config.PCA_DIM)
    idc = G.intrinsic_dimension(Xcp)
    Yc, _, _ = G.isomap_embed(Xcp, max(2, G.d_hat_from(idc, "id_mle")), 12)
    lab = (sub.source == "xstest_contrast").to_numpy()
    cc = Yc[:, 0]
    if spearmanr(cc, lab.astype(float)).statistic < 0:
        cc = -cc
    auc = mannwhitneyu(cc[lab], cc[~lab]).statistic / (lab.sum() * (~lab).sum())
    c.hist(cc[~lab], bins=28, alpha=.7, color="C1", label="XSTest safe (8% refused)")
    c.hist(cc[lab], bins=28, alpha=.7, color="C3", label="XSTest contrast (92% refused)")
    c.set_xlabel("unsupervised first coordinate")
    c.set_ylabel("prompts")
    c.set_title(f"F5c  matched wording, opposite harm\nAUC = {auc:.3f} "
                f"(surface distance 0.685 vs 0.682)", fontsize=8)
    c.legend(fontsize=6.3)
    fig.tight_layout(); fig.savefig(config.FIGURES / "F5_refusal_behaviour.png",
                                    bbox_inches="tight")
    print("wrote F4, F5")


if __name__ == "__main__":
    main()
