"""SPEC §12g: chemistry vs orthography across depth, scored the same way §12b was.

Computed on the UNCENSORED pair set (the one the headline Level 1 lives on) and
across the k grid, so the partial profile can be judged by the same noise-unit
prominence statistic that rejected a peak in the raw profile. "By inspection" is
not a criterion.

    python scripts/orthography_profile.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from src import chemistry as C, geometry as G, evaluation as E


def main():
    blob = torch.load(config.ACTS / "family1.pt", weights_only=False)
    acts = blob["acts"].numpy()
    df = pd.read_csv(config.DATA / "compounds.csv")
    names = list(df["name"])
    Dt = C.tanimoto_distance_matrix(df.smiles.tolist())

    rows = []
    for layer in range(config.N_LAYERS):
        X = np.ascontiguousarray(acts[:, layer, :])
        Xp, _ = G.pca_project(X, config.PCA_DIM)
        ids = G.intrinsic_dimension(Xp)
        d_hat = G.d_hat_from(ids, "id_mle")
        for k in config.K_SWEEP:
            _, D_geo, _ = G.isomap_embed(Xp, n_components=d_hat, k=k)
            for unc in (True, False):
                r = E.orthography_vs_chemistry(D_geo, Dt, names, uncensored=unc)
                r.update(layer=layer, k=k)
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_parquet(config.RESULTS / "orthography_profile.parquet", index=False)

    for unc in (True, False):
        sub = out[out.uncensored == unc]
        tag = "UNCENSORED (headline pair set)" if unc else "all pairs"
        prof = sub.groupby("layer")[["partial_geo_truth_given_name",
                                     "partial_geo_name_given_truth",
                                     "rho_geo_truth", "rho_geo_name"]].mean()
        ksd = float(sub.groupby("layer").partial_geo_truth_given_name.std().mean())
        tail = prof.partial_geo_truth_given_name.loc[4:]
        z = (tail.max() - tail.median()) / ksd
        n_orth = int((prof.partial_geo_name_given_truth
                      > prof.partial_geo_truth_given_name).sum())
        print(f"\n=== {tag} ===")
        print(f"  chemistry|orthography: max {tail.max():.3f} at layer {int(tail.idxmax())}, "
              f"median {tail.median():.3f}, layer 28 {prof.partial_geo_truth_given_name.loc[28]:.3f}")
        print(f"  k-sweep sd = {ksd:.4f}")
        print(f"  PROMINENCE z = {z:.2f}   (§12b rejected a peak at 0.50-1.00)")
        print(f"  layers where orthography exceeds chemistry: {n_orth}/32")

    piv = (out[out.uncensored].groupby("layer")[["rho_geo_truth", "rho_geo_name",
           "partial_geo_truth_given_name", "partial_geo_name_given_truth"]].mean())
    piv.to_csv(config.RESULTS / "orthography_profile.csv")
    print("\nuncensored profile (mean over k):")
    print(piv.loc[[0, 4, 5, 8, 10, 14, 15, 20, 24, 28, 31]].round(3).to_string())


if __name__ == "__main__":
    main()
