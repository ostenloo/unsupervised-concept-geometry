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

K = 12


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
            _, D_geo, info = G.isomap_embed(Xp, n_components=d_hat, k=K)
            r = E.rsa_censored(D_geo, Dt)
            rows.append(dict(pca_dim=(dim if dim else 4096), raw=(dim is None),
                             layer=layer, d_hat=d_hat, id_mle=ids["id_mle"],
                             rsa_uncensored=r["rsa_uncensored"],
                             rsa_all=r["rsa_all"],
                             connected=info["graph_connected"]))
        print(f"  dim={dim if dim else 'raw':>4}: done")

    out = pd.DataFrame(rows)
    out.to_parquet(config.RESULTS / "pca_dim_sweep.parquet", index=False)

    print("\nuncensored RSA by layer and projection dimension")
    piv = out.pivot_table(index="layer", columns="pca_dim", values="rsa_uncensored")
    print(piv.loc[[0, 2, 4, 6, 8, 10, 13, 15, 20, 24, 28, 31]].round(3).to_string())

    print("\nprofile shape per projection dimension:")
    print(f"{'dim':>6} {'argmax':>7} {'max':>7} {'L28':>7} {'min(4-31)':>10} "
          f"{'peak-to-plateau':>16} {'flat layers':>12}")
    for dim, g in out.groupby("pca_dim"):
        g = g.sort_values("layer")
        mx = g.rsa_uncensored.max(); am = int(g.loc[g.rsa_uncensored.idxmax(), "layer"])
        l28 = float(g.loc[g.layer == 28, "rsa_uncensored"].iloc[0])
        tail = g[g.layer >= 4].rsa_uncensored
        # "flat layers" uses the same yardstick as the main profile: the k-sweep sd.
        flat = int((tail >= mx - 0.0336).sum())
        print(f"{dim:>6} {am:>7} {mx:>7.3f} {l28:>7.3f} {tail.min():>10.3f} "
              f"{mx - tail.min():>16.3f} {flat:>12}")

    print("\nID by projection dimension (mean over layers 4-31):")
    print(out[out.layer >= 4].groupby("pca_dim")[["id_mle", "d_hat"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
