"""§18.4b (baseline panel) and §18.4g (ratio-only k slice).

Pre-registration d6aa671. CPU, on the transferred `.npy`.

§18.4b asks whether the manifold machinery is load-bearing. Every baseline is
computed on the same bank as the `d` it is compared against (the clean 400), and
both metrics are reported for every row: Pearson r across points for
coordinate-valued methods, cosine for direction-valued ones, per C1.

§18.4g reports BOTH curvature statistics across the pre-registered K_SWEEP --
the logged one, which §18.10 showed is flat in curvature, and the ambient-chord
replacement of §18.4d". Gates §18.4e'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

import config
from src import geometry as G
from scripts.stageb_18_4ac import (SOLVER, K, difference_in_means,
                                   fit_coordinate, build_direction)

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"


def _geodesic_chord():
    """The logged statistic, compiled from its own AST -- see §18.10."""
    src = (Path(__file__).resolve().parent / "stageb_structure.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "geodesic_chord":
            ns = {"np": np}
            exec(compile(ast.Module([node], []), "stageb_structure.py", "exec"), ns)
            return ns["geodesic_chord"]
    raise RuntimeError("not found")


geodesic_chord = _geodesic_chord()


def main():
    df = pd.read_csv(SB / "prompts.csv")
    A = np.load(NPY / "stageb.npy")
    src = df.source.to_numpy()
    m_harmful, m_harmless = src == "harmful", src == "harmless"
    m_400 = m_harmful | m_harmless
    m_xstest = (src == "borderline") | (src == "xstest_contrast")
    lab = (src[m_xstest] == "xstest_contrast").astype(int)
    layer = config.WURGAFT_LAYER

    X = np.ascontiguousarray(A[:, layer, :])
    Xfit, Xeval = X[m_400], X[m_xstest]
    harmful_fit = m_harmful[m_400]

    v200 = difference_in_means(X[m_harmful], X[m_harmless])
    fit = fit_coordinate(Xfit)
    d, lin = build_direction(fit, Xfit)
    if (Xfit @ d)[harmful_fit].mean() < (Xfit @ d)[~harmful_fit].mean():
        d = -d

    proj_on_vref = fit["Xp"] @ (fit["pca"].components_ @ v200)

    def row(name, direction=None, coord=None):
        """Both metrics for every row (C1). Direction-valued methods also yield a
        coordinate (their projection); coordinate-valued ones need not yield a
        direction."""
        if direction is not None:
            direction = direction / np.linalg.norm(direction)
            if (Xfit @ direction)[harmful_fit].mean() < (Xfit @ direction)[~harmful_fit].mean():
                direction = -direction
            coord = Xfit @ direction
            cos = float(direction @ v200)
            auc = float(roc_auc_score(lab, Xeval @ direction))
        else:
            cos, auc = None, None
        r = float(abs(pearsonr(proj_on_vref, coord).statistic))
        return dict(method=name, pearson_r=r, cosine=cos, auroc=auc)

    rows = [row("Isomap coord 1 -> d (ours)", direction=d)]
    rows[0]["pearson_r"] = float(abs(pearsonr(proj_on_vref, fit["coord"]).statistic))

    # PC1 of the SAME fit bank -- the PCA the pipeline already computes (S3).
    pc1 = fit["pca"].components_[0]
    rows.append(row("PC1 of the fit bank", direction=pc1))

    km = KMeans(n_clusters=2, n_init=10, random_state=config.SEED).fit(fit["Xp"])
    c = km.cluster_centers_
    kdir = fit["pca"].components_.T @ (c[1] - c[0])
    rows.append(row("k-means (k=2) centroid difference", direction=kdir))

    rng = np.random.default_rng(config.SEED)
    rnd = [row("random", direction=rng.normal(size=X.shape[1])) for _ in range(100)]
    for key in ("pearson_r", "cosine", "auroc"):
        vals = np.array([x[key] for x in rnd])
        print(f"  random n=100 {key:>10}: mean {vals.mean():.3f} "
              f"p95 {np.percentile(vals, 95):.3f} max {vals.max():.3f}")

    print(f"\n=== §18.4b baseline panel, L{layer}, fit on the clean 400 ===")
    print(f"{'method':<36} {'Pearson r':>10} {'cos(v_ref)':>11} {'AUROC':>8}")
    for r in rows:
        cos = f"{r['cosine']:.3f}" if r["cosine"] is not None else "   --"
        auc = f"{r['auroc']:.4f}" if r["auroc"] is not None else "   --"
        print(f"{r['method']:<36} {r['pearson_r']:10.3f} {cos:>11} {auc:>8}")

    ours, pc = rows[0], rows[1]
    load_bearing = not (pc["pearson_r"] >= 0.90 and abs(pc["auroc"] - ours["auroc"]) <= 0.01)
    print(f"\n  branch rule: PC1 r={pc['pearson_r']:.3f} (>=0.90?), "
          f"|dAUROC|={abs(pc['auroc'] - ours['auroc']):.4f} (<=0.01?)")
    print(f"  -> manifold machinery IS{'' if load_bearing else ' NOT'} load-bearing")

    # --- §18.4g: both curvature statistics across the pre-registered sweep ---
    print(f"\n=== §18.4g k slice, L{layer}, fit set = 600 in-primary (§18.4c' N6) ===")
    prim = df.in_primary.to_numpy()
    Xp600, _ = G.pca_project(np.ascontiguousarray(X[prim]), config.PCA_DIM)
    ids = G.intrinsic_dimension(Xp600)
    d_hat = G.d_hat_from(ids, "id_mle")
    print(f"{'k':>4} {'logged (geo/embed)':>19} {'ambient (geo/chord)':>20} {'conn':>6}")
    ks = {}
    for k in config.K_SWEEP:
        Y, D_geo, info = G.isomap_embed(Xp600, max(2, d_hat), k, eigen_solver=SOLVER)
        lg = float(np.mean(geodesic_chord(D_geo, Y)))
        am = float(np.mean(geodesic_chord(D_geo, Xp600)))
        ks[k] = dict(logged=lg, ambient=am, connected=info["graph_connected"])
        print(f"{k:>4} {lg:19.4f} {am:20.4f} {str(info['graph_connected']):>6}")

    out = dict(layer=int(layer), linearity=lin, panel=rows,
               manifold_load_bearing=bool(load_bearing),
               random_null={k: dict(mean=float(np.mean([x[k] for x in rnd])),
                                    p95=float(np.percentile([x[k] for x in rnd], 95)))
                            for k in ("pearson_r", "cosine", "auroc")},
               k_slice=ks)
    (config.RESULTS / "stageb_18_4bg.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {config.RESULTS / 'stageb_18_4bg.json'}")


if __name__ == "__main__":
    main()
