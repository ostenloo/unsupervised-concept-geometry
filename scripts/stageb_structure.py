"""SPEC §4c-§4e and §12h — Stage B structure, direction, and controls.

§4a is the binding constraint: the pipeline runs on ACTIVATIONS ALONE. Behaviour
is loaded only in §4e, after every structural number is already fixed.

    python scripts/stageb_structure.py
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
from src import evaluation as E

SB = config.DATA / "stageb"


# --- §4c: a difference-in-means refusal direction --------------------------

def difference_in_means(acts, labels_harmful, layer):
    """v_ref at one layer. SPEC §11f: this is NOT 'Arditi's direction'.

    The non-trivial part of Arditi et al. is the selection sweep over layer and
    position scored on a validation set, which we do not reproduce. We compute
    the difference of means at a layer we select ourselves and name it
    accordingly.
    """
    H = acts[labels_harmful, layer, :]
    B = acts[~labels_harmful, layer, :]
    v = H.mean(0) - B.mean(0)
    return v / np.linalg.norm(v)


# --- §4d: curvature --------------------------------------------------------

def geodesic_chord(D_geo, Y, min_sep_quantile=0.75):
    """geodesic / chord over well-separated pairs. Near 1.0 = straight."""
    from scipy.spatial.distance import squareform, pdist
    chord = squareform(pdist(Y))
    iu = np.triu_indices_from(chord, 1)
    c, g = chord[iu], D_geo[iu]
    keep = c >= np.quantile(c, min_sep_quantile)
    ratio = g[keep] / np.clip(c[keep], 1e-9, None)
    return ratio


def principal_curve_tangents(Y, coord, n_points=10, min_bin=8):
    """Tangents at evenly spaced points along a 1-D principal curve.

    Implemented by binning along the intrinsic coordinate and taking finite
    differences between consecutive bin centroids. The obvious alternative --
    a smoothing spline per dimension -- was tried and is unusable here: with
    variance-scaled smoothing over 600 points it flattens the fit until the
    derivative is numerical noise (tangent cosines collapsed to 0.00-0.05 at
    every layer), and ties in the coordinate make UnivariateSpline return NaN.
    Binning has neither failure mode and needs no smoothing parameter.

    No label is used: the coordinate is the embedding's own first component.
    """
    o = np.argsort(coord)
    Ys = Y[o]
    n_bins = n_points + 1
    edges = np.linspace(0, len(Ys), n_bins + 1).astype(int)
    cent = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= min_bin:
            cent.append(Ys[a:b].mean(axis=0))
    cent = np.stack(cent)
    T = np.diff(cent, axis=0)
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    ok = (nrm > 1e-9).ravel()
    return T[ok] / nrm[ok]


def main():
    df = pd.read_csv(SB / "prompts.csv")
    blob = torch.load(config.ACTS / "stageb.pt", weights_only=False)
    acts = blob["acts"].numpy()
    bank = torch.load(config.ACTS / "bank.pt", weights_only=False)["acts"].numpy()

    prim = df.in_primary.to_numpy()
    src = df.source.to_numpy()
    A = acts[prim]
    print(f"primary set {A.shape}, bank {bank.shape}")

    # --- select the layer for the direction --------------------------------
    # Arditi sweep layer AND position on a validation set; we do neither, so we
    # fix the layer from our own §3g profile and say so (§11f).
    harmful = (src[prim] == "harmful")

    rows = []
    LAYERS = (8, 22, config.WURGAFT_LAYER, 31)
    for layer in LAYERS:
        X = np.ascontiguousarray(A[:, layer, :])
        Xb = np.ascontiguousarray(bank[:, layer, :])
        Xp, _ = G.pca_project(X, config.PCA_DIM)

        ids = G.intrinsic_dimension(Xp)
        ids_bank = G.intrinsic_dimension(G.pca_project(Xb, config.PCA_DIM)[0])
        d_hat = G.d_hat_from(ids, "id_mle")

        Y, D_geo, info = G.isomap_embed(Xp, n_components=max(2, d_hat), k=12)
        ratio = geodesic_chord(D_geo, Y)
        coord = Y[:, 0]
        v = difference_in_means(A, harmful, layer)
        # Tangents live in the PCA subspace; project v the same way to compare.
        pca_full = G.pca_project(X, config.PCA_DIM)[1]
        v_sub = pca_full.components_ @ v
        v_sub = v_sub / np.linalg.norm(v_sub)
        # Isomap axes are not PCA axes; compare in the shared PCA subspace by
        # regressing the embedding back onto it.
        T = principal_curve_tangents(Xp, coord, 10)
        # Also record how well the intrinsic coordinate itself aligns with the
        # difference-in-means direction -- a scalar check on the curve.
        coord_v_cos = abs(float(np.corrcoef((Xp @ v_sub), coord)[0, 1]))
        cos = np.abs(T @ v_sub)

        rows.append(dict(
            layer=layer, n_prompts=int(prim.sum()), d_hat=d_hat,
            id_twonn=ids["id_twonn"], id_mle=ids["id_mle"], id_lpca=ids["id_lpca"],
            id_spread=ids["id_spread"], id_determined=ids["id_determined"],
            id_bank_twonn=ids_bank["id_twonn"], id_bank_mle=ids_bank["id_mle"],
            geo_chord_mean=float(np.mean(ratio)),
            geo_chord_iqr=float(np.subtract(*np.percentile(ratio, [75, 25]))),
            geo_chord_median=float(np.median(ratio)),
            tangent_cos_mean=float(cos.mean()), tangent_cos_min=float(cos.min()),
            tangent_cos_max=float(cos.max()),
            tangent_cos=json.dumps([round(float(c), 4) for c in cos]),
            n_tangents=int(len(cos)), coord_vs_vref_cos=coord_v_cos,
            graph_connected=info["graph_connected"],
        ))
        print(f"  L{layer:2d}: ID mle={ids['id_mle']:.2f} twonn={ids['id_twonn']:.2f} "
              f"(bank {ids_bank['id_mle']:.2f}) | geo/chord {np.mean(ratio):.3f} "
              f"| cos(tangent,v_ref) {cos.min():.2f}-{cos.max():.2f} "
              f"(n={len(cos)}) | corr(coord,v_ref)={coord_v_cos:.3f}")

    res = pd.DataFrame(rows)
    res.to_parquet(config.RESULTS / "stageb_structure.parquet", index=False)
    print("\n" + res[["layer", "id_mle", "id_twonn", "id_bank_mle", "geo_chord_mean",
                      "geo_chord_iqr", "tangent_cos_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
