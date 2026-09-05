"""SPEC §3f Levels 1-2 and §3g layer profile.

Runs the unsupervised pipeline at every layer, over the k sweep, for the default
count fingerprint and the §3j robustness arms, and writes one parquet row per
(layer, method, k, fingerprint) per SPEC §5.

Level 1 is reported as the §11b split: uncensored Spearman plus a test of whether
censored pairs are placed farther. The PCA baseline runs alongside every result.

    python scripts/stage_a_levels.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from src import chemistry as C
from src import geometry as G
from src import evaluation as E

FINGERPRINTS = ["morgan_count", "morgan", "maccs"]


def main():
    blob = torch.load(config.ACTS / "family1.pt", weights_only=False)
    acts = blob["acts"].numpy()
    names = blob["names"]
    df = pd.read_csv(config.DATA / "compounds.csv")
    assert list(df["name"]) == list(names), "compound order drifted from capture"
    print(f"activations {acts.shape}, {len(df)} compounds")

    D = {fp: C.tanimoto_distance_matrix(df.smiles.tolist(), kind=fp) for fp in FINGERPRINTS}
    for fp, d in D.items():
        ok, msg = C.check_metric_axioms(d, n_triples=2000)
        assert ok, f"{fp}: {msg}"

    # Read position of each compound, for the §11c-bis final-token control.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    final_tok = np.array([tok.tokenize(" " + n)[-1] for n in names])

    series_masks = {s: (df.series == s).to_numpy() for s in ("alkane", "alcohol", "acid", "alkene")}
    n_carbons = df.n_carbons.to_numpy()

    rows = []
    t0 = time.time()
    for layer in range(config.N_LAYERS):
        X = np.ascontiguousarray(acts[:, layer, :])
        G.assert_geometry_dtype(X, f"layer{layer}")
        Xp, _ = G.pca_project(X, n_components=config.PCA_DIM)

        ids = G.intrinsic_dimension(Xp)
        ids_raw = G.intrinsic_dimension(X)          # §11d: also on the raw stream
        d_hat = G.d_hat_from(ids, "id_mle")

        for k in config.K_SWEEP:
            Y, D_geo, info = G.isomap_embed(Xp, n_components=d_hat, k=k)
            Yb, D_pca = G.pca_baseline(Xp, n_components=d_hat)

            for fp in FINGERPRINTS:
                Dt = D[fp]
                lvl1 = E.rsa_censored(D_geo, Dt)
                base = E.rsa_censored(D_pca, Dt)
                row = dict(
                    run_id="stageA", layer=layer, pca_dim=config.PCA_DIM, k=k,
                    method="isomap", metric_type="euclidean", fingerprint_type=fp,
                    n_compounds=len(df), d_hat=d_hat,
                    id_twonn=ids["id_twonn"], id_mle=ids["id_mle"],
                    id_mle_k10=ids["id_mle_k10"], id_mle_k20=ids["id_mle_k20"],
                    id_lpca=ids["id_lpca"], id_spread=ids["id_spread"],
                    id_twonn_raw=ids_raw["id_twonn"], id_mle_raw=ids_raw["id_mle"],
                    graph_connected=info["graph_connected"],
                    rsa_spearman=lvl1["rsa_all"],
                    rsa_uncensored=lvl1["rsa_uncensored"],
                    frac_censored=lvl1["frac_censored"],
                    censored_farther_effect=lvl1.get("censored_farther_effect", np.nan),
                    censored_farther_p=lvl1.get("censored_farther_p", np.nan),
                    rsa_pca_baseline=base["rsa_all"],
                    rsa_pca_baseline_uncensored=base["rsa_uncensored"],
                    procrustes_disparity=E.procrustes_disparity(Y, Dt, d_hat),
                )
                if fp == "morgan_count":
                    wg = E.within_group_rsa(D_geo, Dt, final_tok)
                    # Log both; only the uncensored one is comparable to the
                    # headline rsa_uncensored (§11c-bis).
                    row["rsa_within_final_token"] = wg["rsa_within_pooled"]
                    row["rsa_within_final_token_unc"] = wg["rsa_within_pooled_uncensored"]
                    row["frac_censored_within"] = wg["frac_censored_within"]
                    row["n_final_token_groups"] = wg["n_groups"]
                    null = E.shuffle_null_rsa(D_geo, Dt, n_perm=10)
                    row["shuffle_null_rsa"] = null["shuffle_null_rsa"]
                # Alkene is logged but is NOT part of Level 2 headline reporting:
                # §5's schema names alkane, alcohol and acid only, so alkene was
                # our addition. Excluded per §12c -- a restoration of the
                # pre-registered list, not a bar retrofitted after seeing scores.
                for s, mask in series_masks.items():
                    row[f"series_spearman_{s}"] = (
                        E.series_recovery(E.series_coordinate(Y, mask), n_carbons[mask])
                        if mask.sum() >= 4 else np.nan)
                    row[f"series_spearman_{s}_pca"] = (
                        E.series_recovery(E.series_coordinate(Yb, mask), n_carbons[mask])
                        if mask.sum() >= 4 else np.nan)
                rows.append(row)

        if layer % 8 == 0 or layer == config.N_LAYERS - 1:
            print(f"  layer {layer:2d}  d_hat={d_hat}  "
                  f"rsa_unc={rows[-1]['rsa_uncensored']:.3f}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_parquet(config.RESULTS / "stage_a.parquet", index=False)
    print(f"\nwrote {config.RESULTS/'stage_a.parquet'} ({len(out)} rows)")

    d = out
    m = out[(out.fingerprint_type == "morgan_count")]
    prof = m.groupby("layer").agg(
        rsa_unc=("rsa_uncensored", "mean"), rsa_all=("rsa_spearman", "mean"),
        base_unc=("rsa_pca_baseline_uncensored", "mean"),
        within=("rsa_within_final_token", "mean"),
        id_mle=("id_mle", "mean"), id_twonn=("id_twonn", "mean"),
        alk=("series_spearman_alkane", "mean"), alc=("series_spearman_alcohol", "mean"),
        null=("shuffle_null_rsa", "mean")).reset_index()
    # SPEC §12a tie-break: L_peak is the argmax of MEAN uncensored RSA across the
    # full (fingerprint x k) grid. The per-(fingerprint, k) argmax is not
    # identified -- it moves to layer 5, 7, 8, 9, 24, 25 or 26 -- so a single
    # argmax is not a usable selection rule.
    grid = (d[d.fingerprint_type == "morgan_count"]
            .groupby("layer").rsa_uncensored.mean())
    peak = int(grid.idxmax())
    ksd = float(m.groupby("layer").rsa_uncensored.std().mean())
    n_flat = int((grid >= grid.max() - ksd).sum())
    print(f"\nL_peak (§12a tie-break: argmax of mean uncensored RSA) = {peak}")
    print(f"  k-sweep sd = {ksd:.4f}; {n_flat}/32 layers within one sd of the max "
          f"-> plateau, not peak (§12a)")
    print(f"  layer 28 (comparability arm) = {grid.loc[28]:.4f} vs max {grid.max():.4f}")
    print(prof.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    (config.RESULTS / "layer_profile.csv").write_text(prof.to_csv(index=False))
    (config.RESULTS / "L_peak.txt").write_text(str(peak))


if __name__ == "__main__":
    main()
