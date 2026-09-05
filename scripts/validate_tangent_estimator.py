"""SPEC §15b — validate the post-hoc binned-centroid tangent estimator.

Two checks, both specified in §15b before running:
  1. random-direction null: does |cos(tangent, v_ref)| clear chance?
  2. power check on known curvature: can the estimator detect rotation where
     rotation is known to exist (Stage A homologous-series centroid paths)?

The failure mode of a smoother is returning smoothness, so without these,
"tangents are coherent" and "the estimator cannot produce anything else" are
indistinguishable.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import config
from src import geometry as G
from scripts.stageb_structure import principal_curve_tangents, difference_in_means

R_NULL = 200
SB = config.DATA / "stageb"


def main():
    rng = np.random.default_rng(config.SEED)
    df = pd.read_csv(SB / "prompts.csv")
    prim = df.in_primary.to_numpy()
    A = torch.load(config.ACTS / "stageb.pt", weights_only=False)["acts"].numpy()[prim]
    harmful = (df[prim].source == "harmful").to_numpy()

    print("=== CHECK 1: random-direction null (R=%d) ===" % R_NULL)
    print(f"{'layer':>5} {'obs mean|cos|':>13} {'null mean':>10} {'null sd':>8} "
          f"{'z':>7} {'null p95':>9} {'clears?':>8}")
    rows = []
    for layer in (8, 22, config.WURGAFT_LAYER, 31):
        X = np.ascontiguousarray(A[:, layer, :])
        Xp, pca = G.pca_project(X, config.PCA_DIM)
        ids = G.intrinsic_dimension(Xp)
        Y, _, _ = G.isomap_embed(Xp, max(2, G.d_hat_from(ids, "id_mle")), 12)
        T = principal_curve_tangents(Xp, Y[:, 0], 10)

        v = difference_in_means(A, harmful, layer)
        vs = pca.components_ @ v
        vs /= np.linalg.norm(vs)
        obs = float(np.abs(T @ vs).mean())

        null = []
        for _ in range(R_NULL):
            r = rng.standard_normal(Xp.shape[1])
            r /= np.linalg.norm(r)
            null.append(float(np.abs(T @ r).mean()))
        null = np.asarray(null)
        z = (obs - null.mean()) / null.std()
        clears = obs > np.percentile(null, 95)
        rows.append(dict(layer=layer, obs=obs, null_mean=float(null.mean()),
                         null_sd=float(null.std()), z=float(z),
                         null_p95=float(np.percentile(null, 95)), clears=bool(clears)))
        print(f"{layer:5d} {obs:13.3f} {null.mean():10.3f} {null.std():8.3f} "
              f"{z:7.2f} {np.percentile(null,95):9.3f} {'YES' if clears else 'NO':>8}")

    print("\n=== CHECK 2: power on KNOWN curvature (Stage A series paths) ===")
    print("An estimator that cannot see rotation where rotation exists cannot")
    print("support 'straight' as a conclusion anywhere.")
    f2 = {}
    for series in ("alkane", "alcohol"):
        blob = torch.load(config.ACTS / f"family2_{series}.pt", weights_only=False)
        f2[series] = blob["acts"].numpy()          # [10, 32, hidden]
    print(f"\n{'series':9} {'layer':>5} {'mean |cos| between':>19} {'first-vs-last':>14} "
          f"{'verdict':>10}")
    print(f"{'':9} {'':>5} {'consecutive tangents':>19} {'tangent cos':>14}")
    pw = []
    for series, acts in f2.items():
        for layer in (8, config.WURGAFT_LAYER):
            P = np.ascontiguousarray(acts[:, layer, :])
            Pp, _ = G.pca_project(P, min(9, P.shape[0] - 1))
            # the series' own ordinal coordinate is its ground-truth parameter
            T = principal_curve_tangents(Pp, np.arange(len(Pp), dtype=float),
                                         n_points=5, min_bin=1)
            cons = float(np.mean([abs(np.dot(T[i], T[i + 1])) for i in range(len(T) - 1)]))
            ends = float(abs(np.dot(T[0], T[-1])))
            rot = ends < 0.9
            pw.append(dict(series=series, layer=layer, consecutive=cons,
                           first_last=ends, detects_rotation=rot))
            print(f"{series:9} {layer:5d} {cons:19.3f} {ends:14.3f} "
                  f"{'ROTATES' if rot else 'straight':>10}")

    out = {"random_direction_null": rows, "power_check": pw}
    (config.RESULTS / "tangent_validation.json").write_text(json.dumps(out, indent=2))

    all_clear = all(r["clears"] for r in rows)
    any_rot = any(p["detects_rotation"] for p in pw)
    print("\n=== §15b verdict ===")
    print(f"  cosines clear the random-direction null at every layer : {all_clear}")
    print(f"  estimator detects rotation where curvature is known    : {any_rot}")
    if not all_clear:
        print("  -> §15b requires WITHDRAWING the tangent-alignment row of §14a;")
        print("     the coordinate-level alignment (0.946) stands alone.")
    if not any_rot:
        print("  -> the estimator cannot demonstrate rotation, so 'straight' rests on")
        print("     geodesic/chord alone (which has its own null and is stronger).")


if __name__ == "__main__":
    main()
