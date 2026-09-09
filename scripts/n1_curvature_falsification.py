"""§18.4d" step 3 — does the logged geodesic/chord statistic measure curvature?

N1 claims it does not: `stageb_structure.py:112` calls
`geodesic_chord(D_geo, Y)`, whose chord term is `pdist(Y)` — a distance in the
Isomap embedding, which Isomap fits to those same geodesics. If N1 is right,
then on an arc of KNOWN curvature the logged statistic returns ~1.0 at every
true ratio, while the ambient-chord statistic recovers the true ratio.

Pre-registered outcome (§18.4d" step 3): logged ~1.0 everywhere; ambient
tracks truth. If that is NOT what happens, N1 is wrong and §18.4d" is withdrawn
before anything downstream is touched.

Synthetic only. No activations, no GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import brentq

import config
from src import geometry as G


def _load_geodesic_chord():
    """Lift `geodesic_chord` out of stageb_structure.py without importing it.

    That module imports torch at module scope and torch is not installed on this
    laptop (§18.2 gate A1 deliberately avoids needing it). Editing the module to
    make it importable would edit a script that produced logged results, so the
    function's own AST node is compiled instead -- the code under test is
    byte-identical to the code that produced §14a.
    """
    import ast

    src = (Path(__file__).resolve().parent / "stageb_structure.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "geodesic_chord":
            ns = {"np": np}
            exec(compile(ast.Module([node], []), "stageb_structure.py", "exec"), ns)
            return ns["geodesic_chord"]
    raise RuntimeError("geodesic_chord not found in stageb_structure.py")


geodesic_chord = _load_geodesic_chord()


def half_angle_for_ratio(r: float) -> float:
    """phi such that phi / sin(phi) == r. r = 1 gives phi = 0 (a straight line)."""
    if r <= 1.0:
        return 0.0
    return brentq(lambda p: p / np.sin(p) - r, 1e-9, np.pi - 1e-9)


def make_arc(n, phi, ambient, sigma, radius, rng):
    """n points on a circular arc of half-angle phi in a 2-plane of R^ambient.

    Returns (X, t) with t the generating angle, so the TRUE arc length and TRUE
    chord between any pair are known in closed form and need not be estimated.
    """
    t = rng.uniform(-phi, phi, size=n) if phi > 0 else rng.uniform(-1.0, 1.0, size=n)
    if phi > 0:
        P = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    else:
        # phi = 0 is the degenerate straight line; use the arc-length parameter
        # directly so the spread along the curve matches the curved cases.
        P = np.stack([np.zeros(n), radius * t], axis=1)
    basis = np.linalg.qr(rng.normal(size=(ambient, 2)))[0]
    X = P @ basis.T
    X += rng.normal(scale=sigma, size=X.shape)
    return X.astype(np.float32), t


def true_ratio_on_selected(t, phi, radius, min_sep_quantile=0.75):
    """True arc/chord over the same 'well-separated' rule geodesic_chord uses.

    Selection is on the true chord, so this is the ground truth for the pair set
    the statistic is computed over -- not the whole-arc approximation phi/sin phi.
    """
    d = np.abs(t[:, None] - t[None, :])
    iu = np.triu_indices_from(d, 1)
    dt = d[iu]
    if phi > 0:
        arc = radius * dt
        chord = 2.0 * radius * np.sin(dt / 2.0)
    else:
        arc = chord = radius * dt
    keep = chord >= np.quantile(chord, min_sep_quantile)
    return float(np.mean(arc[keep] / np.clip(chord[keep], 1e-12, None)))


def run_one(n, phi, ambient_dim, sigma, radius, k, rng):
    X, t = make_arc(n, phi, ambient_dim, sigma, radius, rng)
    truth = true_ratio_on_selected(t, phi, radius)

    # The actual pipeline, same call sequence as stageb_structure.py:105-112.
    Xp, _ = G.pca_project(X, config.PCA_DIM)
    ids = G.intrinsic_dimension(Xp)
    d_hat = G.d_hat_from(ids, "id_mle")
    n_comp = max(2, d_hat)
    Y, D_geo, info = G.isomap_embed(Xp, n_components=n_comp, k=k)

    return dict(
        truth=truth,
        logged=float(np.mean(geodesic_chord(D_geo, Y))),    # chord = pdist(Y)
        ambient=float(np.mean(geodesic_chord(D_geo, Xp))),  # chord = pdist(Xp)
        d_hat=d_hat, n_comp=n_comp, connected=info["graph_connected"],
    )


def main():
    N = 600            # matches the Stage B in-primary set
    AMBIENT = 64       # PCA(64) is the space the real pipeline embeds in
    RADIUS = 1.0
    K = 12             # the logged Stage B value
    TRUE_RATIOS = [1.000, 1.010, 1.050, 1.250, 1.500]

    # Nearest-neighbour spacing along the curve is ~arc/N ~ 0.003 at these
    # settings. Noise above that swamps the manifold locally, so the kNN graph
    # wanders and geodesics inflate. Sweeping sigma across that threshold shows
    # which regime each statistic's behaviour belongs to.
    SIGMAS = [0.0, 0.0003, 0.001, 0.003, 0.01]

    print(f"n={N} ambient={AMBIENT} k={K} radius={RADIUS} seed={config.SEED}")
    print(f"nn-spacing along curve ~ {2 * RADIUS / N:.4f}\n")

    verdicts = []
    for sigma in SIGMAS:
        rng = np.random.default_rng(config.SEED)  # same draw across sigma
        print(f"--- sigma = {sigma:.4f} "
              f"({'noise < spacing' if sigma < 2 * RADIUS / N else 'noise >= spacing'}) ---")
        print(f"{'true':>7} {'logged':>8} {'ambient':>9} {'amb/null':>9} "
              f"{'d_hat':>6} {'conn':>6}")
        rows = []
        for r in TRUE_RATIOS:
            res = run_one(N, half_angle_for_ratio(r), AMBIENT, sigma, RADIUS, K, rng)
            rows.append(res)
        null = rows[0]["ambient"]   # the straight-line case: pure estimator bias
        for res in rows:
            print(f"{res['truth']:7.3f} {res['logged']:8.3f} {res['ambient']:9.3f} "
                  f"{res['ambient'] / null:9.3f} {res['d_hat']:6d} "
                  f"{str(res['connected']):>6}")

        truth = np.array([x["truth"] for x in rows])
        logged = np.array([x["logged"] for x in rows])
        calib = np.array([x["ambient"] for x in rows]) / null
        logged_span = float(logged.max() - logged.min())
        calib_err = float(np.max(np.abs(calib - truth) / truth))
        print(f"  logged span {logged_span:.3f} over true 1.000-{truth.max():.3f} | "
              f"null-calibrated ambient max rel. err {calib_err:.1%}\n")
        verdicts.append((sigma, logged_span, calib_err))

    print("=" * 66)
    logged_flat = all(span < 0.05 for _, span, _ in verdicts)
    ambient_ok = any(err < 0.10 for _, _, err in verdicts)
    if logged_flat and ambient_ok:
        print("N1 CONFIRMED. The logged statistic is flat in true curvature at every")
        print("noise level tested. The ambient statistic recovers true curvature once")
        print("calibrated against its own straight-line null -- which is what §18.4e'")
        print("exists to supply. Uncalibrated it carries a large multiplicative bias.")
    elif not logged_flat:
        print("N1 NOT CONFIRMED: the logged statistic responds to true curvature.")
        print("§18.4d\" must be withdrawn before anything downstream is touched.")
    else:
        print("INCONCLUSIVE: logged flat, but the ambient replacement does not")
        print("recover truth at any noise level tested.")


if __name__ == "__main__":
    main()
