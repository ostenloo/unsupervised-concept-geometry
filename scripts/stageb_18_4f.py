"""§18.4f" — ID uncertainty: correct estimator, n-matched, with the N9 noise floor.

Pre-registration d6aa671 as amended by §18.11. CPU, on the transferred `.npy`.

Three defects are fixed together:

E2  the 5.21/5.89 contrast is Levina-Bickel MLE, not TwoNN, and `id_mle` is the
    composite mean over k in {10, 20} (geometry.py:130-136), not a single k.
N2  the logged contrast compares 600 points against 2000 with a density-dependent
    estimator. Both banks are subsampled to the SAME n here, so the comparison is
    n-matched by construction.
N3  bootstrap with replacement silently breaks the estimator -- duplicated points
    give T_1 = 0 and are dropped at geometry.py:92. Subsampling WITHOUT
    replacement throughout.

N9's control then asks what `id_mle` returns for a structure whose true intrinsic
dimension is 1, at this data's own noise-to-spacing ratio, using the §18.4e'
generator with true_ratio = 1.0 (a straight line carrying the real residuals).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config
from src import geometry as G
from scripts.stageb_18_4ac import fit_coordinate, build_direction
from scripts.stageb_18_4e import Generator

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"

N_DRAW = 200        # subsampling replicates
M = 480             # 80% of the 600-point refusal set; both banks matched to it


def id_mle_of(X):
    """The logged composite: PCA(64) fit on this set, then mean LB-MLE over k grid."""
    Xp, _ = G.pca_project(np.ascontiguousarray(X), config.PCA_DIM)
    vals = [G.levina_bickel_mle(Xp.astype(np.float32), k=k) for k in config.ID_MLE_K]
    return float(np.nanmean(vals))


def subsample_ids(X, m, n_draw, rng):
    """m-out-of-n WITHOUT replacement (N3). PCA refit per draw, as the logged
    pipeline fits PCA per set."""
    out = []
    for _ in range(n_draw):
        idx = rng.choice(X.shape[0], size=m, replace=False)
        out.append(id_mle_of(X[idx]))
    return np.array(out)


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    df = pd.read_csv(SB / "prompts.csv")
    A = np.load(NPY / "stageb.npy")
    bank = np.load(NPY / "bank.npy")
    prim = df.in_primary.to_numpy()
    layer = config.WURGAFT_LAYER
    rng = np.random.default_rng(config.SEED)

    X = np.ascontiguousarray(A[prim][:, layer, :])
    B = np.ascontiguousarray(bank[:, layer, :])
    print(f"L{layer}: refusal {X.shape}, bank {B.shape}")

    # --- the logged numbers, reproduced at their original n ------------------
    logged_ref, logged_bank = id_mle_of(X), id_mle_of(B)
    print(f"\nlogged construction (n=600 vs n=2000): "
          f"refusal {logged_ref:.2f} | bank {logged_bank:.2f} | "
          f"diff {logged_bank - logged_ref:+.2f}   [§14a: 5.21 / 5.89]")

    # --- N2: the same estimator at matched n --------------------------------
    print(f"\n--- n-matched contrast, m={M} without replacement, {N_DRAW} draws ---")
    ref_d = subsample_ids(X, M, N_DRAW, rng)
    bank_d = subsample_ids(B, M, N_DRAW, rng)
    diff = bank_d - ref_d
    for name, a in (("refusal", ref_d), ("bank", bank_d)):
        lo, hi = ci(a)
        print(f"  {name:>8}: {a.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    dlo, dhi = ci(diff)
    overlap = dlo <= 0.0 <= dhi
    print(f"  bank - refusal: {diff.mean():+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]")
    print(f"  -> difference CI {'INCLUDES' if overlap else 'excludes'} zero")

    # --- how much of the logged gap was n alone? ----------------------------
    bank_at_2000 = logged_bank
    print(f"  bank at n=2000 {bank_at_2000:.2f} vs bank at n={M} {bank_d.mean():.2f} "
          f"-> {bank_at_2000 - bank_d.mean():+.2f} attributable to sample size alone")

    # --- N9: what does id_mle read for a TRULY 1-dimensional structure? -----
    print("\n--- N9 noise-floor control ---")
    fit = fit_coordinate(X)
    d, _ = build_direction(fit, X)
    gen = Generator(X, d)
    Xp, _ = G.pca_project(X, config.PCA_DIM)
    nn = NearestNeighbors(n_neighbors=2).fit(Xp)
    spacing = float(np.median(nn.kneighbors(Xp)[0][:, 1]))
    print(f"  residual sigma/dim {gen.sigma:.4f} | median NN spacing (PCA64) {spacing:.3f} "
          f"| ratio {gen.sigma / spacing:.3f}")

    synth = np.array([id_mle_of(gen.draw(1.000)) for _ in range(50)])
    slo, shi = ci(synth)
    print(f"  id_mle on a TRUE 1-D structure carrying this data's residuals: "
          f"{synth.mean():.3f}  95% CI [{slo:.3f}, {shi:.3f}]  (n=50)")

    reaches = shi >= ci(ref_d)[0]
    print(f"  observed refusal ID {ref_d.mean():.3f} vs 1-D control {synth.mean():.3f}")
    print(f"  -> the 1-D control {'REACHES' if reaches else 'does not reach'} "
          f"the observed ID")
    if reaches:
        print("  -> §18.11 rule fires: no claim about ~5 dimensions; report an upper bound.")
    else:
        print("  -> ID claim survives, with a quantified noise floor.")

    out = dict(
        layer=int(layer), m=M, n_draw=N_DRAW,
        logged_refusal_n600=logged_ref, logged_bank_n2000=logged_bank,
        matched_refusal_mean=float(ref_d.mean()), matched_refusal_ci=list(ci(ref_d)),
        matched_bank_mean=float(bank_d.mean()), matched_bank_ci=list(ci(bank_d)),
        diff_mean=float(diff.mean()), diff_ci=[dlo, dhi], diff_ci_includes_zero=bool(overlap),
        n_bias_estimate=float(bank_at_2000 - bank_d.mean()),
        sigma_per_dim=gen.sigma, nn_spacing_pca64=spacing,
        sigma_over_spacing=float(gen.sigma / spacing),
        synthetic_1d_id_mean=float(synth.mean()), synthetic_1d_id_ci=[slo, shi],
        one_d_control_reaches_observed=bool(reaches),
    )
    (config.RESULTS / "stageb_18_4f.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {config.RESULTS / 'stageb_18_4f.json'}")


if __name__ == "__main__":
    main()
