"""SPEC §6 acceptance tests -- the CPU-only subset.

Tests 4 (weekday loop), 8 (compound gates incl. the knowledge gate) and 9
(alpha=0 reproduces base generation) need the model and live in
tests/test_acceptance_gpu.py. Everything here must pass before any real run and
before each session.
"""
import numpy as np
import pytest

import config
from src import geometry as G
from src import evaluation as E
from src import chemistry as C


# --- test 1: synthetic circle ----------------------------------------------

def test_circle_ordering_id_and_topology():
    X, t = E.synthetic_circle(n=12, ambient=64)
    ids = G.intrinsic_dimension(X)
    # Bounds are set from the measured behaviour in test_id_bias_characterisation,
    # not from the textbook value. A 12-point circle is thin data: MLE lands near
    # 1.3-1.8 and TwoNN carries a ~+1 upward bias on low-dimensional manifolds.
    assert 0.5 <= ids["id_mle"] <= 2.5, ids
    assert ids["id_twonn"] <= 3.0, ids
    # lPCA and the participation ratio measure *linear* dimension, so a circle
    # correctly reads 2, not 1. Asserted so a real result is never misread as
    # "the manifold is 2-D" when lPCA is the only estimator saying so.
    assert 1.5 <= ids["id_lpca"] <= 2.5, ids

    Y, D_geo, info = G.isomap_embed(X, n_components=2, k=3)
    assert info["graph_connected"]

    # Ordering: Isomap geodesics along a closed loop must track arc-length
    # distance, which on a circle is the wrapped angular difference.
    ang = np.abs(t[:, None] - t[None, :])
    arc = np.minimum(ang, 2 * np.pi - ang).astype(np.float32)
    assert E.rsa(D_geo, arc) > 0.95

    # Topology flagged cyclic: no large angular gap in the 2-D embedding.
    assert E.cyclic_score(Y) < 2.0


# --- test 2: synthetic 5x5 grid --------------------------------------------

def test_grid_id_two_and_procrustes_recovery():
    X, uv = E.synthetic_grid(side=5, ambient=64)
    ids = G.intrinsic_dimension(X)
    assert 1.5 <= ids["id_mle"] <= 3.0, ids
    assert ids["id_twonn"] <= 4.5, ids
    # Again linear dimension: the saddle's curvature term occupies a third
    # ambient direction, so lPCA reads 3 for a 2-D manifold.
    assert 2.5 <= ids["id_lpca"] <= 3.5, ids

    Y, D_geo, info = G.isomap_embed(X, n_components=2, k=8)
    assert info["graph_connected"]

    from scipy.spatial.distance import squareform, pdist
    D_true = squareform(pdist(uv)).astype(np.float32)
    assert E.rsa(D_geo, D_true) > 0.9
    assert E.procrustes_disparity(Y, D_true, n_components=2) < 0.1


# --- test 3: isotropic Gaussian --------------------------------------------

def test_isotropic_gaussian_id_is_large():
    X = E.synthetic_isotropic(n=300, ambient=64)
    ids = G.intrinsic_dimension(X)
    # The failure mode this guards against is an estimator that returns 1-3 on
    # structureless data, which would make every "low ID" result meaningless.
    assert ids["id_twonn"] > 8.0, ids
    assert ids["id_participation_ratio"] > 20.0, ids


# --- test 5: shuffle null ---------------------------------------------------

def test_shuffle_null_collapses_rsa():
    X, uv = E.synthetic_grid(side=6, ambient=64)
    from scipy.spatial.distance import squareform, pdist
    D_true = squareform(pdist(uv)).astype(np.float32)
    _, D_geo, _ = G.isomap_embed(X, n_components=2, k=8)

    assert E.rsa(D_geo, D_true) > 0.9          # signal present
    null = E.shuffle_null_rsa(D_geo, D_true, n_perm=20)
    assert abs(null["shuffle_null_rsa"]) < 0.1, null


# --- test 6: no bf16 reaches a distance computation ------------------------

def test_dtype_guard_rejects_low_precision():
    import torch

    # numpy has no bfloat16, so the guard has to catch this on the torch side --
    # which is exactly where capture code sits, before the float32 cast.
    bf16 = torch.randn(20, 8, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        G.assert_geometry_dtype(bf16, "test")
    with pytest.raises(TypeError):
        G.assert_geometry_dtype(torch.randn(20, 8, dtype=torch.float16), "test")
    G.assert_geometry_dtype(torch.randn(20, 8, dtype=torch.float32), "test")

    # ...and on the numpy side, for everything downstream of the cast.
    f16 = np.zeros((20, 8), dtype=np.float16)
    with pytest.raises(TypeError):
        G.assert_geometry_dtype(f16, "test")
    with pytest.raises(TypeError):
        G.pca_project(f16)
    with pytest.raises(TypeError):
        G.isomap_embed(f16, n_components=2, k=4)
    G.assert_geometry_dtype(np.zeros((5, 5), dtype=np.float32), "test")

    # The cast that capture.py must perform is lossless into float32.
    assert bf16.to(torch.float32).numpy().dtype == np.float32
    G.assert_geometry_dtype(bf16.to(torch.float32).numpy(), "test")


# --- test 7: Tanimoto is a metric ------------------------------------------

def test_tanimoto_metric_axioms():
    from data.compound_source import candidate_pool
    smiles = [r[1] for r in candidate_pool()]
    D = C.tanimoto_distance_matrix(smiles, kind="morgan")
    ok, msg = C.check_metric_axioms(D, n_triples=1000)
    assert ok, msg
    assert D.dtype == np.float32


# --- test 10: sign / offset invariance -------------------------------------

def test_recovery_scores_invariant_to_sign_and_rotation():
    X, uv = E.synthetic_grid(side=5, ambient=64)
    Y, _, _ = G.isomap_embed(X, n_components=2, k=8)
    from scipy.spatial.distance import squareform, pdist
    D_true = squareform(pdist(uv)).astype(np.float32)

    base = E.procrustes_disparity(Y, D_true, n_components=2)
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for Yv in (-Y, Y + 3.0, (Y @ R.T).astype(np.float32), (2.5 * Y).astype(np.float32)):
        assert abs(E.procrustes_disparity(Yv, D_true, n_components=2) - base) < 1e-6

    coord = E.series_coordinate(Y, np.ones(len(Y), dtype=bool))
    n_c = uv[:, 0]
    s = E.series_recovery(coord, n_c)
    assert abs(E.series_recovery(-coord, n_c) - s) < 1e-9
    assert abs(E.series_recovery(coord + 10.0, n_c) - s) < 1e-9


# --- pipeline hygiene -------------------------------------------------------

def test_pipeline_never_sees_labels():
    """Guard on the §4a design constraint, enforced here for Stage A too.

    intrinsic_dimension and isomap_embed take activations and nothing else; if a
    future edit adds a label argument this test is the thing that should break.
    """
    import inspect
    for fn in (G.intrinsic_dimension, G.isomap_embed, G.pca_baseline, G.pca_project):
        params = set(inspect.signature(fn).parameters) - {"X", "self"}
        assert not (params & {"labels", "y", "D_truth", "classes", "n_carbons"}), fn


# --- estimator characterisation (not in SPEC §6, but the numbers it pins down
# are what makes the real ID triple interpretable) --------------------------

def test_levina_bickel_tracks_known_dimension():
    """Our own MLE, validated against Gaussians of known dimension.

    skdim's global MLE.fit() ignores its K argument (K=5 and K=50 return the
    same value), so this estimator is implemented in src/geometry.py and has to
    be checked here. The downward bias above d~10 is real and expected: d=20
    reads ~14. That bound is the reason a large ID on real data should be read
    as "high", not as a specific number.
    """
    rng = np.random.default_rng(0)
    for d, lo, hi in [(1, 0.8, 1.3), (2, 1.7, 2.4), (3, 2.6, 3.5),
                      (5, 4.3, 5.6), (10, 8.0, 11.0)]:
        base = rng.standard_normal((500, d))
        Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
        X = (base @ Q[:, :d].T).astype(np.float32)
        est = G.levina_bickel_mle(X, k=10)
        assert lo <= est <= hi, f"true d={d}, mle_k10={est:.2f}"

    # K must actually change the answer -- the skdim bug this replaces.
    X = rng.standard_normal((300, 8)).astype(np.float32)
    assert G.levina_bickel_mle(X, k=5) != G.levina_bickel_mle(X, k=50)


def test_id_bias_characterisation():
    """ID must not diverge with sample size on a fixed manifold.

    Guards the noise-scaling fix in _embed: with noise held at an absolute
    scale, a 1-D circle in 64-D reads ID 2.7 at N=12 and 15.6 at N=300, because
    the 64-dimensional noise ball eventually swallows the inter-point spacing.
    Rising ID with N is the signature of that bug, here and on real data.
    """
    ests = {}
    for n in (25, 100, 300):
        X, _ = E.synthetic_circle(n=n)
        ests[n] = G.intrinsic_dimension(X)
    for n, r in ests.items():
        assert r["id_mle"] < 2.5, (n, r)
        assert r["id_twonn"] < 4.0, (n, r)
    # Monotone divergence is the failure mode; MLE must not grow with N.
    assert ests[300]["id_mle"] <= ests[25]["id_mle"] + 0.5, ests
