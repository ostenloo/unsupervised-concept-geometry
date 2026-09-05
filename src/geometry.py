"""The unsupervised pipeline (SPEC §3e).

Input is activations only. No labels, no D_truth, no class information reaches
anything in this module -- that separation is the whole point of the experiment,
so keep it that way.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

import config


# --- dtype guard (acceptance test 6) ---------------------------------------

def assert_geometry_dtype(X, where: str = ""):
    """No bf16 (or any non-float32/64) tensor may reach a distance computation.

    bf16 carries ~3 decimal digits of mantissa; distance ties break every
    neighbour method downstream. SPEC §2 and acceptance test 6.

    Torch-aware on purpose: capture code holds torch tensors, and numpy cannot
    even represent bfloat16 (`.numpy()` on a bf16 tensor raises), so a numpy-only
    guard would never see the dtype this test exists to catch.
    """
    dtype = getattr(X, "dtype", None)
    try:
        import torch
        if isinstance(X, torch.Tensor):
            if dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"geometry input must be float32/float64, got torch {dtype} "
                    f"at {where or '<unknown>'}"
                )
            return X
    except ImportError:
        pass
    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"geometry input must be float32/float64, got {dtype} at {where or '<unknown>'}"
        )
    return X


# --- step 0: projection -----------------------------------------------------

def pca_project(X: np.ndarray, n_components: int = None, seed: int = None):
    """Project to the PCA subspace used by every downstream method. Returns (Y, pca)."""
    assert_geometry_dtype(X, "pca_project")
    n_components = config.PCA_DIM if n_components is None else n_components
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=n_components, random_state=config.SEED if seed is None else seed)
    return p.fit_transform(X).astype(np.float32), p


# --- step 1: intrinsic dimension -------------------------------------------

def participation_ratio(X: np.ndarray) -> float:
    """(sum λ)^2 / sum λ^2 on the covariance spectrum -- the lPCA-style global estimate."""
    assert_geometry_dtype(X, "participation_ratio")
    Xc = X - X.mean(0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def levina_bickel_mle(X: np.ndarray, k: int, mackay_ghahramani: bool = True) -> float:
    """Levina-Bickel maximum-likelihood intrinsic dimension at neighbourhood size k.

    Implemented here rather than taken from skdim: skdim's global `MLE.fit()`
    ignores its `K` argument (K=5 and K=50 return the identical value), and the
    spec pins this estimator at k in {10, 20}. A number that does not depend on
    the parameter it is reported under is worse than no number.

        m_k(x_i)^-1 = 1/(k-1) * sum_{j=1..k-1} log( T_k(x_i) / T_j(x_i) )

    with T_j the distance from x_i to its j-th nearest neighbour. With
    `mackay_ghahramani`, average the *inverses* across points before inverting
    (the standard bias correction); otherwise average m_k directly as in the
    original paper.
    """
    from sklearn.neighbors import NearestNeighbors

    assert_geometry_dtype(X, "levina_bickel_mle")
    n = X.shape[0]
    if k >= n:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    T = dist[:, 1:]                       # drop self-distance
    # Duplicate or near-duplicate points give T_j = 0 and a divergent log.
    ok = T[:, 0] > 0
    if ok.sum() < 2:
        return float("nan")
    T = T[ok]
    logs = np.log(T[:, [k - 1]]) - np.log(T[:, : k - 1])
    inv_m = logs.mean(axis=1)
    inv_m = inv_m[inv_m > 0]
    if inv_m.size < 2:
        return float("nan")
    if mackay_ghahramani:
        return float(1.0 / inv_m.mean())
    return float(np.mean(1.0 / inv_m))


def intrinsic_dimension(X: np.ndarray) -> dict:
    """Three estimators, always reported together (SPEC §3e step 1).

    Never collapse these to one number: at N ~ 300 in 64-D they are expected to
    disagree (SPEC §9 risk 3). `spread` is max-min; > 1.5 means the dimension is
    not determined by the data and the writeup must say so.
    """
    import skdim

    assert_geometry_dtype(X, "intrinsic_dimension")
    X64 = np.asarray(X, dtype=np.float64)  # skdim is happier in f64
    out = {}

    try:
        out["id_twonn"] = float(skdim.id.TwoNN().fit(X64).dimension_)
    except Exception as e:                      # noqa: BLE001 - report, never crash a sweep
        out["id_twonn"] = float("nan")
        out["id_twonn_error"] = str(e)

    for k in config.ID_MLE_K:
        key = f"id_mle_k{k}"
        try:
            out[key] = levina_bickel_mle(X64, k=k)
        except Exception as e:                  # noqa: BLE001
            out[key] = float("nan")
            out[f"{key}_error"] = str(e)
    # Canonical "id_mle" for the parquet schema: mean over the k grid.
    mles = [out[f"id_mle_k{k}"] for k in config.ID_MLE_K]
    out["id_mle"] = float(np.nanmean(mles)) if not all(np.isnan(mles)) else float("nan")

    try:
        out["id_lpca"] = float(skdim.id.lPCA().fit(X64).dimension_)
    except Exception as e:                      # noqa: BLE001
        out["id_lpca"] = float("nan")
        out["id_lpca_error"] = str(e)

    out["id_participation_ratio"] = participation_ratio(X)

    triple = [out["id_twonn"], out["id_mle"], out["id_lpca"]]
    finite = [v for v in triple if np.isfinite(v)]
    out["id_spread"] = float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")
    out["id_determined"] = bool(np.isfinite(out["id_spread"]) and out["id_spread"] <= 1.5)
    return out


def d_hat_from(id_result: dict, estimator: str = "id_twonn") -> int:
    """Round to nearest int, floored at 1. SPEC §3e also runs d_hat +/- 1."""
    v = id_result.get(estimator, float("nan"))
    if not np.isfinite(v):
        v = id_result.get("id_mle", 2.0)
    return max(1, int(round(float(v))))


# --- step 2: embedding ------------------------------------------------------

def isomap_embed(X: np.ndarray, n_components: int, k: int):
    """Isomap. Returns (Y, D_geo, info).

    D_geo is the geodesic distance matrix from the neighbour graph -- that, not
    the embedding, is what Level 1 RSA consumes.
    """
    assert_geometry_dtype(X, "isomap_embed")
    n = X.shape[0]
    k_eff = min(k, n - 1)
    iso = Isomap(n_neighbors=k_eff, n_components=min(n_components, n - 1))
    Y = iso.fit_transform(X).astype(np.float32)
    D_geo = np.asarray(iso.dist_matrix_, dtype=np.float32)
    info = {
        "k_requested": k,
        "k_effective": k_eff,
        # A disconnected neighbour graph gives infinite geodesics; sklearn fills
        # them, but the embedding is meaningless and the run must be flagged.
        "graph_connected": bool(np.isfinite(D_geo).all() and (D_geo.max() < np.inf)),
        "reconstruction_error": float(iso.reconstruction_error()),
    }
    return Y, D_geo, info


def pca_baseline(X: np.ndarray, n_components: int):
    """Step 3: the naive baseline that runs alongside every result (SPEC §3e).

    Top-d_hat PCA coordinates and their Euclidean distances.
    """
    assert_geometry_dtype(X, "pca_baseline")
    Y, _ = pca_project(X, n_components=n_components)
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(Y)).astype(np.float32)
    return Y, D


# --- whitening (SPEC §3h) ---------------------------------------------------

def whitening_transform(bank: np.ndarray, eps: float = 1e-6):
    """W = Cov(bank)^(-1/2) and the bank mean, per Park et al.'s causal inner product.

    Returns a function h -> W (h - mean). Fit on an independent activation bank,
    never on the compounds themselves.
    """
    assert_geometry_dtype(bank, "whitening_transform")
    mu = bank.mean(0, keepdims=True)
    Xc = np.asarray(bank - mu, dtype=np.float64)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T

    def apply(H: np.ndarray) -> np.ndarray:
        assert_geometry_dtype(H, "whiten.apply")
        return ((np.asarray(H, dtype=np.float64) - mu) @ W.T).astype(np.float32)

    return apply
