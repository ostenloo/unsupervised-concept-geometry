"""§18.4a' and §18.4c' — build `d`, decontaminate `v_ref`, recompute the AUROC.

Pre-registration d6aa671, as amended by §18.11 (N10). Runs on the transferred
`.npy` activations; CPU only, no torch.

§18.4a': the pipeline returns an Isomap coordinate, not an ambient direction.
Regress the coordinate on the 64-d PCA scores (well-posed, unlike n=600/p=4096
OLS), lift to R^4096, report in-sample and cross-validated R².

§18.4c' as revised by N10: the logged 0.995 was fit on the evaluation set with a
label-chosen sign. The honest number fits on the 400 harmful+harmless, which
contain no XSTest of either kind, and scores XSTest by projection onto `d_400`
with the sign fixed from the training set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import pearsonr, spearmanr

import config
from src import geometry as G

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"
K = 12                    # the logged Stage B value
SOLVER = "dense"          # §18.7 / N5: pinned, not "auto"
LAYERS = (8, 22, config.WURGAFT_LAYER, 31)


def difference_in_means(H, B):
    """§4c / §11f. Not 'Arditi's direction' -- see §11f for why."""
    v = H.mean(0) - B.mean(0)
    return v / np.linalg.norm(v)


def fit_coordinate(X, k=K):
    """PCA(64) -> Isomap, exactly the call sequence of stageb_structure.py:105-112."""
    Xp, pca = G.pca_project(X, config.PCA_DIM)
    ids = G.intrinsic_dimension(Xp)
    d_hat = G.d_hat_from(ids, "id_mle")
    Y, D_geo, info = G.isomap_embed(Xp, max(2, d_hat), k, eigen_solver=SOLVER)
    return dict(Xp=Xp, pca=pca, coord=Y[:, 0], d_hat=d_hat,
                connected=info["graph_connected"], id_mle=ids["id_mle"])


def build_direction(fit, X):
    """§18.4a': regress coord on the 64-d scores, lift to ambient, audit linearity."""
    scores, pca, coord = fit["Xp"], fit["pca"], fit["coord"]

    ols = LinearRegression().fit(scores, coord)
    r2_in = float(ols.score(scores, coord))
    cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    r2_cv = float(np.mean(cross_val_score(ols, scores, coord, cv=cv, scoring="r2")))

    d = pca.components_.T @ ols.coef_          # 64 -> 4096
    d = d / np.linalg.norm(d)

    # C2: in-sample R² at p=4096 is vacuous, so ridge CV only. The gap against
    # the 64-d fit is what PCA(64) discarded.
    Xc = X - X.mean(0, keepdims=True)
    ridge = RidgeCV(alphas=np.logspace(0, 6, 13))
    r2_ridge_cv = float(np.mean(cross_val_score(ridge, Xc, coord, cv=cv, scoring="r2")))

    return d, dict(r2_in_sample_64=r2_in, r2_cv_64=r2_cv, r2_cv_ridge_4096=r2_ridge_cv)


def main():
    df = pd.read_csv(SB / "prompts.csv")
    A = np.load(NPY / "stageb.npy")
    assert A.dtype == np.float32 and A.shape[0] == len(df), (A.shape, len(df))

    src = df.source.to_numpy()
    prim = df.in_primary.to_numpy()
    m_harmful = src == "harmful"
    m_harmless = src == "harmless"
    m_border = src == "borderline"
    m_contrast = src == "xstest_contrast"
    m_400 = m_harmful | m_harmless                      # clean: no XSTest at all
    m_xstest = m_border | m_contrast                    # the evaluation set
    print(f"600 in-primary={prim.sum()} | clean-400={m_400.sum()} | "
          f"XSTest eval={m_xstest.sum()} (border {m_border.sum()}, contrast {m_contrast.sum()})")
    print(f"solver={SOLVER} k={K} seed={config.SEED}\n")

    out = {}
    for layer in LAYERS:
        X = np.ascontiguousarray(A[:, layer, :])

        # --- the two fits ------------------------------------------------
        f600 = fit_coordinate(X[prim])
        f400 = fit_coordinate(X[m_400])
        d600, lin600 = build_direction(f600, X[prim])
        d400, lin400 = build_direction(f400, X[m_400])

        # --- §18.4c' step 3: sign from the TRAINING set, never test labels --
        for d, mask in ((d600, prim), (d400, m_400)):
            p = X[mask] @ d
            hi = p[m_harmful[mask]].mean()
            lo = p[(m_harmless | m_border)[mask]].mean()
            if hi < lo:
                d *= -1

        # --- v_ref, contaminated and clean (E4) ---------------------------
        v400 = difference_in_means(X[m_harmful], X[prim & ~m_harmful])   # as logged
        v200 = difference_in_means(X[m_harmful], X[m_harmless])          # clean

        # --- §18.4a' step 3: is the d fork empty? -------------------------
        cos_dd = float(abs(d400 @ d600))
        cos_vv = float(v200 @ v400)

        # --- Pearson r, the logged metric (C1) ----------------------------
        def pear(fit, mask, v):
            proj = fit["Xp"] @ (fit["pca"].components_ @ v)
            return float(abs(pearsonr(proj, fit["coord"]).statistic))

        r_logged = pear(f600, prim, v400)      # the 0.946 pairing
        r_clean = pear(f400, m_400, v200)      # the clean pairing

        # --- §18.4c' step 2: genuine held-out AUROC -----------------------
        lab = m_contrast[m_xstest].astype(int)
        auc_holdout = float(roc_auc_score(lab, X[m_xstest] @ d400))

        # --- reproduce the logged construction (figures_stageb.py:99-110) --
        fx = fit_coordinate(X[m_xstest])
        cc = fx["coord"]
        if spearmanr(cc, lab.astype(float)).statistic < 0:   # label-chosen sign
            cc = -cc
        auc_transductive = float(roc_auc_score(lab, cc))

        out[layer] = dict(
            d_hat_600=f600["d_hat"], d_hat_400=f400["d_hat"],
            connected_600=f600["connected"], connected_400=f400["connected"],
            lin600=lin600, lin400=lin400,
            cos_d400_d600=cos_dd, cos_vref200_vref400=cos_vv,
            cos_d400_vref200=float(d400 @ v200), cos_d600_vref400=float(d600 @ v400),
            pearson_r_logged_pairing=r_logged, pearson_r_clean_pairing=r_clean,
            auroc_holdout=auc_holdout, auroc_transductive_reproduction=auc_transductive,
        )

        star = " <-- headline layer" if layer == config.WURGAFT_LAYER else ""
        print(f"=== L{layer}{star}")
        print(f"  §18.4a' linearity   R2 in-sample(64d) {lin400['r2_in_sample_64']:.3f} | "
              f"CV(64d) {lin400['r2_cv_64']:.3f} | CV ridge(4096d) {lin400['r2_cv_ridge_4096']:.3f}")
        print(f"  §18.4a' d fork      |cos(d400,d600)| = {cos_dd:.4f}"
              f"   {'-> fork empty' if cos_dd >= 0.99 else '-> BOTH must be carried'}")
        print(f"  §18.4c' v_ref       cos(v200,v400)  = {cos_vv:.4f}"
              f"   {'-> immaterial' if cos_vv >= 0.99 else '-> v200 is primary'}")
        print(f"  cos(d400,v_ref200)  = {out[layer]['cos_d400_vref200']:+.4f} "
              f"(arm 4b constraint)")
        print(f"  Pearson r           logged pairing {r_logged:.3f} | clean pairing {r_clean:.3f}")
        print(f"  AUROC               held-out (proj on d400) {auc_holdout:.4f} | "
              f"logged construction {auc_transductive:.4f}")
        print(f"  d_hat               600-fit {f600['d_hat']} | 400-fit {f400['d_hat']} | "
              f"connected {f600['connected']}/{f400['connected']}")

    (config.RESULTS / "stageb_18_4ac.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {config.RESULTS / 'stageb_18_4ac.json'}")


if __name__ == "__main__":
    main()
