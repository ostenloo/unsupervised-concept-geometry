"""SPEC §7 figures F1, F2 and F7 for Stage A."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from src import chemistry as C, geometry as G, evaluation as E

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    d = pd.read_parquet(config.RESULTS / "stage_a.parquet")
    m = d[d.fingerprint_type == "morgan_count"]
    prof = m.groupby("layer").agg(
        rsa=("rsa_uncensored", "mean"), rsa_sd=("rsa_uncensored", "std"),
        base=("rsa_pca_baseline_uncensored", "mean"),
        within=("rsa_within_final_token", "mean"), null=("shuffle_null_rsa", "mean"),
        id_mle=("id_mle", "mean"), id_twonn=("id_twonn", "mean"),
        alk=("series_spearman_alkane", "mean"), alk_p=("series_spearman_alkane_pca", "mean"),
        alc=("series_spearman_alcohol", "mean"), alc_p=("series_spearman_alcohol_pca", "mean"),
    ).reset_index()
    L_peak = int(prof.loc[prof.rsa.idxmax(), "layer"])
    ksd = float(prof.rsa_sd.mean())

    # ---------------- F2: layer profile ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    a = ax[0]
    a.fill_between(prof.layer, prof.rsa - prof.rsa_sd, prof.rsa + prof.rsa_sd,
                   alpha=.18, color="C0", lw=0)
    a.plot(prof.layer, prof.rsa, "o-", ms=2.6, color="C0", label="Isomap (uncensored RSA)")
    a.plot(prof.layer, prof.base, "s--", ms=2.4, color="C1", label="PCA baseline")
    a.plot(prof.layer, prof.within, "^-", ms=2.4, color="C2",
           label="within shared final token")
    a.plot(prof.layer, prof.null, ":", color="0.5", label="shuffle null")
    band = prof.rsa.max() - ksd
    inside = prof.layer[prof.rsa >= band]
    a.axhspan(band, prof.rsa.max(), color="C0", alpha=.07)
    a.axvspan(inside.min(), inside.max(), color="C0", alpha=.05)
    a.axvline(L_peak, color="C0", lw=.9, ls="-.")
    a.axvline(28, color="C3", lw=.9, ls="-.")
    a.annotate(f"L_peak={L_peak}", (L_peak, .04), fontsize=7, color="C0", rotation=90, va="bottom")
    a.annotate("Wurgaft 28", (28, .04), fontsize=7, color="C3", rotation=90, va="bottom")
    a.set_xlabel("layer"); a.set_ylabel("Spearman vs Tanimoto (uncensored)")
    a.set_title(f"F2  Level 1 by depth — plateau, not peak\n"
                f"{int((prof.rsa>=band).sum())}/32 layers within one k-sweep sd of the max",
                fontsize=8)
    a.legend(fontsize=6.3, loc="lower right"); a.set_ylim(-.03, None)

    b = ax[1]
    b.plot(prof.layer, prof.id_mle, "o-", ms=2.6, label="Levina–Bickel MLE")
    b.plot(prof.layer, prof.id_twonn, "s--", ms=2.4, label="TwoNN")
    b.axvline(L_peak, color="C0", lw=.9, ls="-."); b.axvline(28, color="C3", lw=.9, ls="-.")
    b.set_xlabel("layer"); b.set_ylabel("intrinsic dimension")
    b.set_title("F2b  ID by depth (never a single number)", fontsize=8)
    b.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(config.FIGURES / "F2_layer_profile.png", bbox_inches="tight")

    # ---------------- F1: instrument validation ----------------
    blob = torch.load(config.ACTS / "family1.pt", weights_only=False)
    acts = blob["acts"].numpy(); names = blob["names"]
    df = pd.read_csv(config.DATA / "compounds.csv")
    Dt = C.tanimoto_distance_matrix(df.smiles.tolist())
    X = np.ascontiguousarray(acts[:, L_peak, :])
    Xp, _ = G.pca_project(X, config.PCA_DIM)
    Y, D_geo, _ = G.isomap_embed(Xp, n_components=2, k=12)

    from sklearn.manifold import MDS
    from scipy.spatial import procrustes
    Z = MDS(n_components=2, dissimilarity="precomputed", random_state=config.SEED,
            normalized_stress=False).fit_transform(Dt.astype(np.float64))
    Z2, Y2, disp = procrustes(Z, Y.astype(np.float64))

    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.5))
    classes = df.cls.fillna("other").to_numpy()
    top = pd.Series(classes).value_counts().index[:8]
    for i, cl in enumerate(top):
        s = classes == cl
        ax[0].scatter(Z2[s, 0], Z2[s, 1], s=11, alpha=.85, label=cl, color=f"C{i}")
        ax[1].scatter(Y2[s, 0], Y2[s, 1], s=11, alpha=.85, color=f"C{i}")
    ax[0].set_title("MDS of Tanimoto (ground truth)", fontsize=8)
    ax[1].set_title(f"Isomap of activations, layer {L_peak}\nProcrustes-aligned "
                    f"(disparity {disp:.2f})", fontsize=8)
    ax[0].legend(fontsize=5.6, ncol=2, loc="best")

    c = ax[2]
    w = .35; xs = np.arange(2)
    c.bar(xs - w/2, [prof.rsa.max(), prof.loc[prof.layer == 28, "rsa"].iloc[0]], w,
          label="Isomap", color="C0")
    c.bar(xs + w/2, [prof.loc[prof.layer == L_peak, "base"].iloc[0],
                     prof.loc[prof.layer == 28, "base"].iloc[0]], w,
          label="PCA baseline", color="C1")
    c.set_xticks(xs); c.set_xticklabels([f"L_peak={L_peak}", "layer 28"])
    c.set_ylabel("uncensored RSA"); c.legend(fontsize=6.5)
    c.set_title("F1 inset  pipeline vs naive baseline", fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "F1_instrument.png", bbox_inches="tight")

    # ---------------- Level 2 + F7 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.2))
    a = ax[0]
    a.plot(prof.layer, prof.alk, "o-", ms=2.5, color="C0", label="alkane, Isomap")
    a.plot(prof.layer, prof.alk_p, "s--", ms=2.3, color="C0", alpha=.5, label="alkane, PCA")
    a.plot(prof.layer, prof.alc, "o-", ms=2.5, color="C2", label="alcohol, Isomap")
    a.plot(prof.layer, prof.alc_p, "s--", ms=2.3, color="C2", alpha=.5, label="alcohol, PCA")
    a.set_xlabel("layer"); a.set_ylabel("|Spearman| vs carbon count")
    a.set_title("Level 2 — series recovery.\nPCA baseline matches it.", fontsize=8)
    a.legend(fontsize=6); a.set_ylim(0, 1.02)

    # Both spaces: the ID of the raw residual stream and of the PCA-64 subspace
    # the pipeline actually embeds differ by a factor of two, and reporting only
    # one invites the reader to attach the wrong number to the wrong claim.
    Xr = np.ascontiguousarray(acts[:, L_peak, :]).astype(np.float32)
    b = ax[1]
    for space, Xc, ls in (("raw 4096-d", Xr, "-"), ("PCA-64", Xp.astype(np.float32), "--")):
        ic = E.id_curve(Xc, k_values=(10,))
        g2 = ic.groupby("n").id_mle.agg(["mean", "std"]).reset_index()
        b.errorbar(g2.n, g2["mean"], yerr=g2["std"].fillna(0), marker="o", ms=3,
                   capsize=2, ls=ls, label=f"MLE k=10, {space}")
    b.set_xlabel("subsample size (N)"); b.set_ylabel("intrinsic dimension")
    b.set_title(f"F7a  ID vs sampling density, layer {L_peak}\n"
                f"still rising at N=218: treat as a lower bound", fontsize=8)
    b.legend(fontsize=6.5)

    c = ax[2]
    for true_id, col in ((1, "C0"), (2, "C1")):
        nv = E.id_vs_noise(intrinsic=true_id, n=200)
        c.plot(nv.noise_ratio, nv.id_mle_k10, "o-", ms=3, color=col,
               label=f"synthetic, true ID={true_id}")
        c.axhline(true_id, color=col, ls=":", lw=.8)
    c.set_xscale("symlog", linthresh=.02)
    c.set_xlabel("noise / spacing ratio"); c.set_ylabel("estimated ID")
    c.set_title("F7b  calibration: where the estimator\nstops seeing the manifold", fontsize=8)
    c.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(config.FIGURES / "F7_level2_and_id_curve.png",
                                    bbox_inches="tight")
    print(f"L_peak={L_peak}, k-sweep sd={ksd:.4f}, procrustes disparity={disp:.3f}")
    print("wrote F1, F2, F7")


if __name__ == "__main__":
    main()
