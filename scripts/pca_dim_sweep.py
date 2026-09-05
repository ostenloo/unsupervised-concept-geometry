"""Is the §3g plateau real, or an artifact of a fixed 64-d projection?

The layer profile embeds every layer through PCA-64. Raw ID at layer 8 is ~13 and
still climbing at N=218, and raw ID varies with depth (16.0 at layer 4, 8.1 at
layer 28). If raw dimensionality varies across layers, a FIXED 64-d truncation
removes proportionally more from the layers that carry more structure, which
would flatten a profile that is not actually flat.

So: rerun Level 1 across depth at several projection dimensions, and on raw
activations. If flatness survives, §3g is solid. If the profile sharpens as d
grows, the plateau is partly the instrument.

    python scripts/pca_dim_sweep.py
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

K_SWEEP = (8, 12, 16, 24)   # full sweep: each d needs its OWN noise estimate


def main():
    blob = torch.load(config.ACTS / "family1.pt", weights_only=False)
    acts = blob["acts"].numpy()
    df = pd.read_csv(config.DATA / "compounds.csv")
    Dt = C.tanimoto_distance_matrix(df.smiles.tolist())

    rows = []
    for dim in (16, 32, 64, 128, 217, None):     # None = raw 4096-d
        for layer in range(config.N_LAYERS):
            X = np.ascontiguousarray(acts[:, layer, :])
            if dim is None:
                Xp = X
            else:
                Xp, _ = G.pca_project(X, n_components=dim)
            ids = G.intrinsic_dimension(Xp)
            d_hat = G.d_hat_from(ids, "id_mle")
            for k in K_SWEEP:
                _, D_geo, info = G.isomap_embed(Xp, n_components=d_hat, k=k)
                r = E.rsa_censored(D_geo, Dt)
                rows.append(dict(pca_dim=(dim if dim else 4096), raw=(dim is None),
                                 layer=layer, k=k, d_hat=d_hat, id_mle=ids["id_mle"],
                                 rsa_uncensored=r["rsa_uncensored"],
                                 rsa_all=r["rsa_all"],
                                 connected=info["graph_connected"]))
        print(f"  dim={dim if dim else 'raw':>4}: done")

    out = pd.DataFrame(rows)
    out.to_parquet(config.RESULTS / "pca_dim_sweep.parquet", index=False)

    print("\nuncensored RSA by layer and projection dimension (mean over k)")
    piv = out.pivot_table(index="layer", columns="pca_dim", values="rsa_uncensored")
    print(piv.loc[[0, 2, 4, 6, 8, 10, 13, 15, 20, 24, 28, 31]].round(3).to_string())

    # Each d gets its OWN noise yardstick: the across-k spread at that d, which is
    # a genuine measurement-noise estimate. An earlier version used the spread of
    # the profile ACROSS LAYERS -- that is the structure under test, not noise --
    # and counted "flat" layers against a fixed yardstick borrowed from d=64.
    # The two were unconnected and the table was mislabelled.
    print("\nprofile shape, each d against its OWN k-sweep noise:")
    print(f"{'dim':>6} {'k-sd':>7} {'mean rsa':>9} {'max':>7} {'L28':>7} "
          f"{'flat/32':>8} {'flat>=L4/28':>12} {'argmax':>7}")
    for dim, g in out.groupby("pca_dim"):
        prof = g.groupby("layer").rsa_uncensored.mean()
        ksd = float(g.groupby("layer").rsa_uncensored.std().mean())
        flat = int((prof >= prof.max() - ksd).sum())
        tail = prof.loc[4:]
        flat4 = int((tail >= tail.max() - ksd).sum())
        print(f"{dim:>6} {ksd:>7.4f} {prof.mean():>9.3f} {prof.max():>7.3f} "
              f"{prof.loc[28]:>7.3f} {flat:>8} {flat4:>12} {int(prof.idxmax()):>7}")

    print("\nID by projection dimension (mean over layers 4-31):")
    print(out[out.layer >= 4].groupby("pca_dim")[["id_mle", "d_hat"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
