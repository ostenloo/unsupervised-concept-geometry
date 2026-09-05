"""Evaluation levels 1 and 2 (SPEC §3f), plus the nulls of §3j.

Level 3 (steering) needs the model and lives in the stage-A scripts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from scipy.spatial import procrustes

from src.geometry import assert_geometry_dtype
import config


def _upper(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


# --- Level 1 ----------------------------------------------------------------

def rsa(D_recovered: np.ndarray, D_truth: np.ndarray) -> float:
    """Spearman between upper triangles. SPEC §3f Level 1."""
    assert_geometry_dtype(D_recovered, "rsa/recovered")
    assert_geometry_dtype(D_truth, "rsa/truth")
    if D_recovered.shape != D_truth.shape:
        raise ValueError(f"shape mismatch {D_recovered.shape} vs {D_truth.shape}")
    a, b = _upper(D_recovered), _upper(D_truth)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    return float(spearmanr(a[finite], b[finite]).statistic)


def procrustes_disparity(Y: np.ndarray, D_truth: np.ndarray, n_components: int,
                         seed: int = None) -> float:
    """MDS-embed D_truth to n_components, align Y to it, return disparity in [0, 1].

    Lower is better. procrustes() is invariant to translation, uniform scale and
    rotation, which is what acceptance test 10 requires of any recovery score.
    """
    assert_geometry_dtype(Y, "procrustes/Y")
    seed = config.SEED if seed is None else seed
    n_components = max(1, min(n_components, D_truth.shape[0] - 1))
    mds = MDS(n_components=n_components, dissimilarity="precomputed",
              random_state=seed, normalized_stress=False)
    Z = mds.fit_transform(np.asarray(D_truth, dtype=np.float64))
    # procrustes needs equal shapes; pad the narrower side with zero columns,
    # which adds no variance and leaves the alignment well defined.
    d = max(Y.shape[1], Z.shape[1])
    Yp = np.pad(np.asarray(Y, dtype=np.float64), ((0, 0), (0, d - Y.shape[1])))
    Zp = np.pad(Z, ((0, 0), (0, d - Z.shape[1])))
    _, _, disparity = procrustes(Zp, Yp)
    return float(disparity)


# --- Level 2 ----------------------------------------------------------------

def series_recovery(coord: np.ndarray, n_carbons: np.ndarray) -> float:
    """|Spearman| between a recovered 1-D coordinate and carbon count.

    Absolute value because the sign of an embedding axis is arbitrary --
    acceptance test 10 (sign/offset invariance) demands the score not move
    when the coordinate is reversed.
    """
    coord = np.asarray(coord, dtype=np.float64).ravel()
    n_carbons = np.asarray(n_carbons, dtype=np.float64).ravel()
    if coord.size < 3:
        return float("nan")
    return float(abs(spearmanr(coord, n_carbons).statistic))


def series_coordinate(Y_global: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The recovered 1-D coordinate along one homologous series.

    DECISION (spec ambiguity, §3f Level 2 vs §2's k sweep): a series has 10
    members, so a per-series Isomap refit with k in {8,12,16,24} is degenerate or
    fully connected and the k sweep carries no information. We take the global
    embedding restricted to the series and project onto its own leading axis.
    A per-series refit is available in scripts/stage_a_levels.py as a secondary
    number; both go in the writeup.
    """
    sub = np.asarray(Y_global[mask], dtype=np.float64)
    sub = sub - sub.mean(0, keepdims=True)
    if sub.shape[0] < 2:
        return np.zeros(sub.shape[0])
    u, s, vt = np.linalg.svd(sub, full_matrices=False)
    return (sub @ vt[0]).astype(np.float64)


# --- §3j nulls --------------------------------------------------------------

def shuffle_null_rsa(D_recovered: np.ndarray, D_truth: np.ndarray,
                     n_perm: int = 20, seed: int = None) -> dict:
    """Permute the compound<->activation assignment; RSA must collapse to ~0."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = D_truth.shape[0]
    vals = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        vals.append(rsa(D_recovered, np.ascontiguousarray(D_truth[np.ix_(p, p)])))
    vals = np.asarray(vals, dtype=np.float64)
    return {"shuffle_null_rsa": float(np.nanmean(vals)),
            "shuffle_null_rsa_max_abs": float(np.nanmax(np.abs(vals))),
            "n_perm": n_perm}


def synthetic_circle(n: int = 12, ambient: int = 64, noise: float = 0.05,
                     jitter: float = 0.35, seed: int = None):
    """n points on a circle, rotated into `ambient` dims. True ID = 1.

    `jitter` perturbs the angular sampling. It is not cosmetic: TwoNN estimates
    ln(2)/mean(ln(r2/r1)), and on *exactly* equispaced samples both nearest
    neighbours are equidistant, so r2/r1 -> 1 and the estimate diverges (a
    12-point equispaced circle returns ~22 instead of ~1). That is a degeneracy
    of regular lattices, not of the data; real compound and prompt sets are never
    equispaced. Jittered sampling is the honest test.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    step = 2 * np.pi / n
    t = np.sort(np.arange(n) * step + rng.uniform(-jitter, jitter, n) * step)
    base = np.stack([np.cos(t), np.sin(t)], axis=1)
    X = _embed(base, ambient, noise, rng, spacing=step)
    return X.astype(np.float32), t


def synthetic_grid(side: int = 5, ambient: int = 64, noise: float = 0.05,
                   curvature: float = 0.3, jitter: float = 0.35, seed: int = None):
    """side x side grid with mild curvature (a saddle), in `ambient` dims. True ID = 2.

    Jittered for the same reason as synthetic_circle: a perfect square lattice
    gives every point four equidistant neighbours and sends TwoNN to ~35.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    g = np.linspace(-1, 1, side)
    uu, vv = np.meshgrid(g, g, indexing="ij")
    u, v = uu.ravel(), vv.ravel()
    step = g[1] - g[0]
    u = u + rng.uniform(-jitter, jitter, u.shape) * step
    v = v + rng.uniform(-jitter, jitter, v.shape) * step
    base = np.stack([u, v, curvature * (u ** 2 - v ** 2)], axis=1)
    X = _embed(base, ambient, noise, rng, spacing=step)
    return X.astype(np.float32), np.stack([u, v], axis=1)


def synthetic_isotropic(n: int = 300, ambient: int = 64, seed: int = None):
    """Isotropic Gaussian: ID estimators must return a large value, not 1-3."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    return rng.standard_normal((n, ambient)).astype(np.float32)


def _embed(base: np.ndarray, ambient: int, noise: float, rng,
           spacing: float = None) -> np.ndarray:
    """Rotate `base` into `ambient` dims and add isotropic noise.

    `noise` is a fraction of the on-manifold sampling `spacing`, not an absolute
    scale, and the perturbation is normalised so its expected norm is
    `noise * spacing` regardless of `ambient`.

    This matters. Isotropic noise in D dims is a D-dimensional object: hold it
    fixed while N grows and the inter-point spacing eventually falls below the
    noise ball, at which point every ID estimator correctly reports the noise
    dimension instead of the manifold's. With absolute noise=0.01, a "1-D"
    circle in 64-D reads ID 2.7 at N=12 and 15.6 at N=300 -- rising with sample
    size, which looks like estimator failure and is not. Scaling with spacing
    keeps the manifold resolvable at every N, so the tests measure the estimator
    rather than the noise floor.

    The same ratio governs the real data: any ID reported on activations is a
    statement about structure *above the local noise scale at that sampling
    density*, which is why §3j's isotropic null is not optional.
    """
    d = base.shape[1]
    if ambient < d:
        raise ValueError("ambient dimension smaller than intrinsic")
    Q, _ = np.linalg.qr(rng.standard_normal((ambient, ambient)))
    X = base @ Q[:, :d].T
    if noise <= 0:
        return X
    scale = noise * (1.0 if spacing is None else spacing) / np.sqrt(ambient)
    return X + scale * rng.standard_normal(X.shape)


def cyclic_score(Y: np.ndarray) -> float:
    """Crude topology flag for acceptance test 1.

    On a closed loop the 2-D embedding's polar angle visits every point once and
    the sorted angular gaps are near-uniform; on an arc there is one large gap.
    Returns max_gap / (2*pi/n): ~1 for a circle, >>1 for a line.
    """
    Z = np.asarray(Y[:, :2], dtype=np.float64)
    Z = Z - Z.mean(0, keepdims=True)
    ang = np.sort(np.arctan2(Z[:, 1], Z[:, 0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    return float(gaps.max() / (2 * np.pi / len(ang)))


# --- censored pairs (SPEC §11b) --------------------------------------------

def rsa_censored(D_recovered: np.ndarray, D_truth: np.ndarray,
                 censor_at: float = 1.0) -> dict:
    """Level 1 split into an uncensored rank correlation and a censoring test.

    ~36% of compound pairs sit at Tanimoto distance exactly 1.0: they share no
    substructure at all, so the value says only "unrelated" and carries no
    information about *how* unrelated. Those are censored observations, not tied
    ones, and handing them to a Spearman as ties is a modeling error rather than
    an inconvenience. So:

      1. `rsa_uncensored` -- Spearman on pairs with D_truth < censor_at, which is
         the only subset where the ground truth actually orders anything;
      2. a two-sample test asking whether the recovered distance is larger for
         censored than uncensored pairs, which is the whole of what the censored
         pairs can tell us.

    `rsa_all` is reported alongside for continuity with the pre-registered §3f,
    not because it is the better number.

    Isomap is largely insulated from the censoring -- geodesics are built from
    short edges and long distances are inferred -- so this bites Level 1 RSA far
    harder than it bites the embedding. Run it for the PCA baseline too, or the
    comparison is not like-for-like.
    """
    from scipy.stats import mannwhitneyu

    assert_geometry_dtype = __import__("src.geometry", fromlist=["x"]).assert_geometry_dtype
    assert_geometry_dtype(D_recovered, "rsa_censored/recovered")
    assert_geometry_dtype(D_truth, "rsa_censored/truth")

    a, b = _upper(D_recovered), _upper(D_truth)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    censored = b >= censor_at - 1e-6
    n_unc = int((~censored).sum())

    out = {
        "rsa_all": rsa(D_recovered, D_truth),
        "n_pairs": int(a.size),
        "n_censored": int(censored.sum()),
        "frac_censored": float(censored.mean()) if a.size else float("nan"),
        "n_uncensored": n_unc,
    }
    out["rsa_uncensored"] = (
        float(spearmanr(a[~censored], b[~censored]).statistic) if n_unc >= 3 else float("nan")
    )
    if censored.any() and n_unc >= 3:
        u = mannwhitneyu(a[censored], a[~censored], alternative="greater")
        out["censored_farther_u"] = float(u.statistic)
        out["censored_farther_p"] = float(u.pvalue)
        # Rank-biserial: 0 = no separation, 1 = every censored pair is farther.
        out["censored_farther_effect"] = float(
            2 * u.statistic / (censored.sum() * n_unc) - 1
        )
        out["median_recovered_censored"] = float(np.median(a[censored]))
        out["median_recovered_uncensored"] = float(np.median(a[~censored]))
    return out


# --- ID as a curve (SPEC §11d, figure F7) ----------------------------------

def id_curve(X: np.ndarray, k_values=(10, 20), subsample_fracs=(0.25, 0.5, 0.75, 1.0),
             n_repeats: int = 5, seed: int = None) -> "object":
    """ID against neighbourhood size k and against sample size.

    ID is a statement about structure above the local noise scale at a given
    sampling density, so the scale dependence is the result -- a single scalar
    hides exactly the artifact a reader will ask about. Returns a tidy DataFrame
    (rows: frac x repeat x k) for figure F7.
    """
    import pandas as pd
    from src.geometry import levina_bickel_mle, participation_ratio

    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = X.shape[0]
    rows = []
    for frac in subsample_fracs:
        m = max(8, int(round(frac * n)))
        reps = 1 if m >= n else n_repeats
        for rep in range(reps):
            idx = np.arange(n) if m >= n else rng.choice(n, size=m, replace=False)
            Xs = np.ascontiguousarray(X[idx])
            for k in k_values:
                rows.append(dict(frac=frac, n=m, repeat=rep, k=k,
                                 id_mle=levina_bickel_mle(Xs, k=k),
                                 id_pr=participation_ratio(Xs)))
    return pd.DataFrame(rows)


def id_vs_noise(intrinsic: int = 2, ambient: int = 64, n: int = 300,
                noise_ratios=(0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
                seed: int = None) -> "object":
    """ID against noise-to-spacing ratio on a synthetic manifold of known dimension.

    The calibration panel of F7: it shows where the estimator stops reporting the
    manifold and starts reporting the noise ball, which is what sets how much to
    trust any ID measured on real activations.
    """
    import pandas as pd
    from src.geometry import levina_bickel_mle

    rows = []
    for r in noise_ratios:
        if intrinsic == 1:
            X, _ = synthetic_circle(n=n, ambient=ambient, noise=r, seed=seed)
        else:
            side = int(round(np.sqrt(n)))
            X, _ = synthetic_grid(side=side, ambient=ambient, noise=r, seed=seed)
        rows.append(dict(noise_ratio=r, n=X.shape[0], true_id=intrinsic,
                         id_mle_k10=levina_bickel_mle(X, k=10),
                         id_mle_k20=levina_bickel_mle(X, k=20)))
    return pd.DataFrame(rows)


# --- final-token confound control ------------------------------------------

def within_group_rsa(D_recovered: np.ndarray, D_truth: np.ndarray,
                     group: np.ndarray, min_size: int = 6) -> dict:
    """RSA computed only within groups of points sharing a read position.

    Family 1 reads the LAST token of the compound name, and the gated set has
    only ~70 distinct final tokens over ~220 compounds -- `Ġacid` alone covers
    15%. So a reader can reasonably ask whether recovered structure is final-token
    identity rather than chemistry. Restricting to pairs that share a final token
    holds that variable fixed: structure surviving inside the `Ġacid` block cannot
    be explained by the block label.

    Returns the pooled within-group Spearman and the per-group breakdown.
    """
    group = np.asarray(group)
    per, all_a, all_b = {}, [], []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        if idx.size < min_size:
            continue
        sub_r = np.ascontiguousarray(D_recovered[np.ix_(idx, idx)])
        sub_t = np.ascontiguousarray(D_truth[np.ix_(idx, idx)])
        per[str(g)] = {"n": int(idx.size), "rsa": rsa(sub_r, sub_t)}
        all_a.append(_upper(sub_r))
        all_b.append(_upper(sub_t))
    if not all_a:
        return {"rsa_within_pooled": float("nan"), "groups": {}}
    a, b = np.concatenate(all_a), np.concatenate(all_b)
    ok = np.isfinite(a) & np.isfinite(b)
    return {"rsa_within_pooled": float(spearmanr(a[ok], b[ok]).statistic),
            "n_pairs": int(ok.sum()), "n_groups": len(per), "groups": per}
