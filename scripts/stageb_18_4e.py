"""§18.4e' — power check on synthetic curvature, option (B).

Pre-registration d6aa671. CPU, on the transferred `.npy`.

E7/N4 fixed the design: generate in the full 4096-d ambient space using the real
data's covariance -- NOT isotropic noise, and not in the 64-d space, which would
make PCA(64) a no-op. Noise is produced by resampling the real residual vectors
about the 1-D line along `d`, with random signs. That reproduces the true
residual covariance exactly without forming or factorising a 4096x4096 matrix.

The pipeline then runs whole, with PCA(64) fit on the synthetic bank.

MDR is computed for BOTH statistics: the ambient-chord replacement (§18.4d") and
the logged one, which §18.10 predicts has no power at any ratio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import config
from src import geometry as G
from scripts.stageb_18_4ac import SOLVER, fit_coordinate, build_direction

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"

TRUE_RATIOS = [1.000, 1.005, 1.01, 1.02, 1.05, 1.10, 1.25, 1.50]
N_REP_PRIMARY = 50      # at k=12, the logged value
N_REP_OTHER = 20        # at the rest of K_SWEEP
N_COMP = 5              # the logged d_hat; the ambient statistic does not use it


def _geodesic_chord():
    src = (Path(__file__).resolve().parent / "stageb_structure.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "geodesic_chord":
            ns = {"np": np}
            exec(compile(ast.Module([node], []), "stageb_structure.py", "exec"), ns)
            return ns["geodesic_chord"]
    raise RuntimeError("not found")


geodesic_chord = _geodesic_chord()


def half_angle_for_ratio(r):
    return 0.0 if r <= 1.0 else brentq(lambda p: p / np.sin(p) - r, 1e-9, np.pi - 1e-9)


class Generator:
    """Synthetic arcs carrying the real data's residual covariance.

    `spread` is matched to the observed extent along `d`, so the signal-to-noise
    ratio of the synthetic bank matches the real one by construction -- which is
    the whole point of the exercise.
    """

    def __init__(self, X, d, seed=config.SEED, density="uniform"):
        """`density` controls how arc-length positions are drawn.

        "uniform" spreads points evenly, which is what the §18.15 run used.
        "empirical" resamples the observed projections onto `d`, reproducing the
        real data's clustering along the curve. Gaps between clusters lengthen
        graph geodesics, so a uniform null under-states the inflation a straight
        structure would produce and can make clustering look like curvature.
        """
        self.n, self.p = X.shape
        Xc = X - X.mean(0, keepdims=True)
        proj = Xc @ d
        self.density = density
        self.proj_centered = (proj - proj.mean()).astype(np.float32)
        self.spread = float(proj.max() - proj.min())
        self.resid = (Xc - np.outer(proj, d)).astype(np.float32)   # about the 1-D line
        self.e1 = d.astype(np.float32)
        self.rng = np.random.default_rng(seed)
        # sigma per dimension, for the record and for §18.4f"'s N9 control
        self.sigma = float(np.sqrt((self.resid ** 2).sum() / self.resid.size))

    def _orthonormal_partner(self):
        v = self.rng.normal(size=self.p).astype(np.float32)
        v -= (v @ self.e1) * self.e1
        return v / np.linalg.norm(v)

    def draw(self, true_ratio, n=None):
        n = n or self.n
        phi = half_angle_for_ratio(true_ratio)
        if self.density == "empirical":
            s = self.rng.choice(self.proj_centered, size=n, replace=True)
        else:
            s = self.rng.uniform(-self.spread / 2, self.spread / 2, size=n).astype(np.float32)
        e2 = self._orthonormal_partner()
        if phi > 0:
            radius = self.spread / (2 * phi)
            t = s / radius
            P = radius * (np.cos(t)[:, None] * self.e1 + np.sin(t)[:, None] * e2)
        else:
            P = s[:, None] * self.e1
        idx = self.rng.integers(0, self.resid.shape[0], size=n)
        sign = self.rng.choice([-1.0, 1.0], size=n).astype(np.float32)[:, None]
        return (P + sign * self.resid[idx]).astype(np.float32)


def statistics(X, k):
    """The pipeline, whole: PCA(64) fit on THIS bank, then Isomap."""
    Xp, _ = G.pca_project(X, config.PCA_DIM)
    Y, D_geo, info = G.isomap_embed(Xp, N_COMP, k, eigen_solver=SOLVER)
    return (float(np.mean(geodesic_chord(D_geo, Xp))),    # ambient  (§18.4d")
            float(np.mean(geodesic_chord(D_geo, Y))),     # logged
            bool(info["graph_connected"]))


def mdr(null, alt_by_ratio, ratios):
    """Threshold = 95th pct of the null; MDR = smallest ratio detected >=80%."""
    thr = float(np.percentile(null, 95))
    power = {r: float(np.mean(np.asarray(alt_by_ratio[r]) > thr)) for r in ratios}
    detected = [r for r in ratios if r > 1.0 and power[r] >= 0.80]
    return thr, power, (min(detected) if detected else None)


def main():
    df = pd.read_csv(SB / "prompts.csv")
    A = np.load(NPY / "stageb.npy")
    prim = df.in_primary.to_numpy()
    layer = config.WURGAFT_LAYER
    X = np.ascontiguousarray(A[prim][:, layer, :])

    # N6: the structural claims stay on the 600, so `d` here is d_600.
    fit = fit_coordinate(X)
    d, _ = build_direction(fit, X)
    gen = Generator(X, d)

    Xp_real, _ = G.pca_project(X, config.PCA_DIM)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(Xp_real)
    spacing = float(np.median(nn.kneighbors(Xp_real)[0][:, 1]))
    print(f"L{layer}: n={X.shape[0]} spread along d = {gen.spread:.2f} | "
          f"residual sigma/dim = {gen.sigma:.4f} | median NN spacing (PCA64) = {spacing:.3f}")
    print(f"observed ambient statistic at k=12 (for reference): "
          f"{statistics(X, 12)[0]:.4f}\n")

    out = {"layer": int(layer), "spread": gen.spread, "sigma_per_dim": gen.sigma,
           "nn_spacing_pca64": spacing, "per_k": {}}

    for k in config.K_SWEEP:
        nrep = N_REP_PRIMARY if k == 12 else N_REP_OTHER
        amb, log = {}, {}
        for r in TRUE_RATIOS:
            a, l = [], []
            for _ in range(nrep):
                sa, sl, _ = statistics(gen.draw(r), k)
                a.append(sa); l.append(sl)
            amb[r], log[r] = a, l
        obs_amb, obs_log, _ = statistics(X, k)

        thr_a, pow_a, mdr_a = mdr(amb[1.000], amb, TRUE_RATIOS)
        thr_l, pow_l, mdr_l = mdr(log[1.000], log, TRUE_RATIOS)

        print(f"=== k={k} ({nrep} replicates) ===")
        print(f"{'true':>7} {'ambient mean':>13} {'power':>7}   {'logged mean':>12} {'power':>7}")
        for r in TRUE_RATIOS:
            print(f"{r:7.3f} {np.mean(amb[r]):13.3f} {pow_a[r]:7.2f}   "
                  f"{np.mean(log[r]):12.4f} {pow_l[r]:7.2f}")
        print(f"  ambient: null p95 = {thr_a:.3f} | observed = {obs_amb:.3f} | "
              f"MDR = {mdr_a if mdr_a else '>1.50 (no detection)'}")
        print(f"  logged : null p95 = {thr_l:.4f} | observed = {obs_log:.4f} | "
              f"MDR = {mdr_l if mdr_l else '>1.50 (no detection)'}")
        print(f"  observed vs null: ambient {'ABOVE' if obs_amb > thr_a else 'inside'} "
              f"the null's 95th percentile\n")

        out["per_k"][int(k)] = dict(
            n_replicates=nrep, observed_ambient=obs_amb, observed_logged=obs_log,
            ambient_null_p95=thr_a, logged_null_p95=thr_l,
            ambient_mdr=mdr_a, logged_mdr=mdr_l,
            ambient_power={str(r): pow_a[r] for r in TRUE_RATIOS},
            logged_power={str(r): pow_l[r] for r in TRUE_RATIOS},
            ambient_mean={str(r): float(np.mean(amb[r])) for r in TRUE_RATIOS},
            logged_mean={str(r): float(np.mean(log[r])) for r in TRUE_RATIOS},
        )

    (config.RESULTS / "stageb_18_4e.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {config.RESULTS / 'stageb_18_4e.json'}")


if __name__ == "__main__":
    main()
