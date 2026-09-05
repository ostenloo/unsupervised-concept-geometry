"""Level 3 machinery (SPEC §3f): manifolds, behaviour space, and E_BC.

Follows Wurgaft App. A.3 (spline in the PCA subspace), A.4 (behaviour manifold in
Hellinger coordinates, fit in the tangent plane and decoded with the exponential
map), A.6 (replace the top-64 PCA components, keep the orthogonal residual) and
A.7 / Eq. 3 (cumulative Bhattacharyya energy).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from src.geometry import assert_geometry_dtype


# --- activation-space manifold (App. A.3) ----------------------------------

def fit_spline(points: np.ndarray, t: np.ndarray):
    """Natural cubic spline through `points`, parameterised by `t`.

    `t` is the intrinsic parameter: the unsupervised Level-2 coordinate for the
    unsupervised manifold, ground-truth carbon count for the supervised
    reference. Normalised to [0, 1] so the two are parameterised comparably --
    otherwise the arc-length spacing of waypoints differs between them for a
    reason that has nothing to do with the geometry.
    """
    assert_geometry_dtype(points, "fit_spline")
    t = np.asarray(t, dtype=np.float64)
    order = np.argsort(t)
    ts, ps = t[order], np.asarray(points, dtype=np.float64)[order]
    # Ties would make the spline ill-posed; nudge them apart deterministically.
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1e-9
    ts = (ts - ts[0]) / (ts[-1] - ts[0])
    return CubicSpline(ts, ps, bc_type="natural"), ts, order


def waypoints_manifold(spline, t_a: float, t_b: float, K: int) -> np.ndarray:
    """K points along the fitted curve between the two endpoints' parameters."""
    return np.asarray(spline(np.linspace(t_a, t_b, K)), dtype=np.float32)


def waypoints_linear(p_a: np.ndarray, p_b: np.ndarray, K: int) -> np.ndarray:
    """Straight line in activation space -- the comparison strategy."""
    lam = np.linspace(0.0, 1.0, K)[:, None]
    return ((1 - lam) * p_a[None, :] + lam * p_b[None, :]).astype(np.float32)


# --- behaviour manifold in Hellinger coordinates (App. A.4) ----------------

def to_hellinger(P: np.ndarray) -> np.ndarray:
    """Probability simplex -> positive orthant of the unit sphere."""
    P = np.clip(np.asarray(P, dtype=np.float64), 0, None)
    P = P / np.clip(P.sum(axis=-1, keepdims=True), 1e-12, None)
    return np.sqrt(P)


def _log_map(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sphere log map at base point m."""
    c = np.clip(x @ m, -1.0, 1.0)
    theta = np.arccos(c)
    v = x - c[..., None] * m
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.where(nrm > 1e-12, theta[..., None] / np.clip(nrm, 1e-12, None), 0.0)
    return v * scale


def _exp_map(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sphere exponential map at base point m."""
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.cos(nrm) * m[None, :] + np.where(nrm > 1e-12, np.sin(nrm) * v / np.clip(nrm, 1e-12, None), 0.0)
    return out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12, None)


def fit_behavior_manifold(dists: np.ndarray, t: np.ndarray, n_dense: int = 600):
    """Fit M_y through behaviour distributions and return densely sampled points.

    Fitted in the tangent plane at the Hellinger mean and decoded with the
    exponential map, so the curve stays on the sphere (App. A.4).
    """
    S = to_hellinger(dists)
    m = S.mean(axis=0)
    m = m / np.linalg.norm(m)
    V = _log_map(m, S)
    spline, ts, order = fit_spline(V.astype(np.float32), t)
    dense_t = np.linspace(ts[0], ts[-1], n_dense)
    return _exp_map(m, np.asarray(spline(dense_t))), dense_t


# --- E_BC (Eq. 3 / App. A.7) -----------------------------------------------

def bhattacharyya_distance(P: np.ndarray, Q_sqrt: np.ndarray) -> np.ndarray:
    """d_BC(p, q) = -ln <sqrt(p), sqrt(q)>, for P a batch of distributions."""
    Ps = to_hellinger(P)
    bc = np.clip(Ps @ Q_sqrt.T, 1e-12, 1.0)
    return -np.log(bc)


def e_bc(traj_dists: np.ndarray, manifold_sqrt: np.ndarray) -> float:
    """Cumulative Bhattacharyya energy: sum over waypoints of the distance from
    the trajectory's behaviour to the nearest point on M_y."""
    D = bhattacharyya_distance(traj_dists, manifold_sqrt)
    return float(D.min(axis=1).sum())


# --- the §12f statistic ----------------------------------------------------

def r_statistic(e_unsup: float, e_sup: float, e_lin: float) -> float:
    """r = (E_unsup - E_sup) / (E_linear - E_sup). Success is r < 0.5 (SPEC §12f)."""
    denom = e_lin - e_sup
    return float((e_unsup - e_sup) / denom) if denom != 0 else float("nan")


def cluster_bootstrap_r(per_pair: dict, members: list, n_draws: int = 1000,
                        seed: int = 0) -> dict:
    """Bootstrap `r` over COMPOUNDS, not pairs (SPEC §12f).

    Centroid pairs share compounds, so resampling pairs treats dependent
    observations as independent and understates SE -- which would make the
    2*SE_pooled disagreement threshold one we cross by construction. Here each
    draw resamples the series members with replacement and keeps only the pairs
    whose BOTH endpoints survive, weighted by endpoint multiplicity. That is a
    cluster bootstrap at the compound level and needs no new forward passes.

    `per_pair` maps (i, j) -> {"lin":…, "unsup":…, "sup":…}.
    """
    rng = np.random.default_rng(seed)
    members = list(members)
    rs, degen = [], 0
    for _ in range(n_draws):
        draw = rng.choice(len(members), size=len(members), replace=True)
        mult = np.bincount(draw, minlength=len(members))
        num_l = num_u = num_s = 0.0
        for (i, j), v in per_pair.items():
            w = mult[i] * mult[j]
            if w == 0:
                continue
            num_l += w * v["lin"]; num_u += w * v["unsup"]; num_s += w * v["sup"]
        if num_l == 0:
            continue
        denom = num_l - num_s
        if denom <= 0:
            degen += 1
            continue
        rs.append((num_u - num_s) / denom)
    rs = np.asarray(rs, dtype=float)
    frac_degen = degen / max(1, n_draws)
    return {
        "r_median": float(np.median(rs)) if rs.size else float("nan"),
        "r_lo": float(np.percentile(rs, 2.5)) if rs.size else float("nan"),
        "r_hi": float(np.percentile(rs, 97.5)) if rs.size else float("nan"),
        "frac_denominator_nonpositive": float(frac_degen),
        "degenerate": bool(frac_degen > 0.05),      # alpha = 0.05, SPEC §12f
        "n_draws_used": int(rs.size),
    }


# --- intervention efficacy precondition (SPEC §12i) ------------------------

def intervention_is_efficacious(tgt_mass: float, src_mass: float,
                                min_ratio: float = 0.25) -> bool:
    """Does replacement at this layer actually move behaviour?

    Level 3 compares how far three steering trajectories stray from the behaviour
    manifold. If the intervention has no behavioural effect, all three return the
    *source* behaviour, which is ON the manifold, so every E_BC is uniformly small
    and the strategies are indistinguishable -- `r` then approaches 0 and looks
    like success while measuring nothing.

    That is not hypothetical: at layer 8 injecting the target centroid puts
    **0.000** probability mass on the target name (source injection gives 0.478),
    and Level 3 duly returned r = -0.04, a spurious pass. At layer 28 the same
    injection gives 0.448 against a source mass of 0.509 and the test is live.

    So efficacy is a PRECONDITION, checked and reported before `r` is formed:
    injecting the target centroid must put at least `min_ratio` of the source
    injection's mass onto the *target* name. The §12f degenerate-case clause does
    not catch this -- it tests whether the denominator is positive, not whether
    the intervention does anything, and at layer 8 the denominator was positive
    but tiny (0.065).
    """
    if src_mass <= 0:
        return False
    return bool(tgt_mass / src_mass >= min_ratio)
