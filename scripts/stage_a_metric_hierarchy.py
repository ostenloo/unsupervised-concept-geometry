"""SPEC §3h (whitened metric) and §3i (hierarchy check).

§3h: Park et al. show hierarchy appears under a whitened inner product and
vanishes under Euclidean. Repeat Level 1 under both. If whitening moves RSA by
more than 0.1, the metric is load-bearing and Stage B must be run under both.

§3i: class labels used ONLY for evaluation. Is the continuous (series) structure
orthogonal to the discrete (class) structure, as a direct-sum-of-polytopes
picture predicts?
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from src import chemistry as C
from src import geometry as G
from src import evaluation as E

LAYERS = (8, config.WURGAFT_LAYER)


def main():
    df = pd.read_csv(config.DATA / "compounds.csv")
    A = torch.load(config.ACTS / "family1.pt", weights_only=False)["acts"].numpy()
    bank = torch.load(config.ACTS / "bank_bare.pt", weights_only=False)["acts"].numpy()
    Dt = C.tanimoto_distance_matrix(df.smiles.tolist())

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    ft = np.array([tok.tokenize(" " + n)[-1] for n in df["name"]])

    print("=== §3h: Level 1 under Euclidean vs whitened ===")
    rows = []
    for layer in LAYERS:
        X = np.ascontiguousarray(A[:, layer, :])
        B = np.ascontiguousarray(bank[:, layer, :])
        whiten = G.whitening_transform(B)
        Xw = whiten(X)

        out = {}
        for tag, M_ in (("euclidean", X), ("whitened", Xw)):
            Xp, _ = G.pca_project(M_, config.PCA_DIM)
            ids = G.intrinsic_dimension(Xp)
            d_hat = G.d_hat_from(ids, "id_mle")
            Y, D_geo, _ = G.isomap_embed(Xp, d_hat, 12)
            lvl1 = E.rsa_censored(D_geo, Dt)
            _, D_pca = G.pca_baseline(Xp, d_hat)
            base = E.rsa_censored(D_pca, Dt)
            orth = E.orthography_vs_chemistry(D_geo, Dt, list(df["name"]), ft)
            out[tag] = dict(rsa_unc=lvl1["rsa_uncensored"], rsa_all=lvl1["rsa_all"],
                            base_unc=base["rsa_uncensored"], id_mle=ids["id_mle"],
                            d_hat=d_hat,
                            chem_given_orth=orth["partial_geo_truth_given_name"],
                            orth_given_chem=orth["partial_geo_name_given_truth"])
        delta = out["whitened"]["rsa_unc"] - out["euclidean"]["rsa_unc"]
        rows.append(dict(layer=layer, delta_rsa=delta,
                         **{f"euc_{k}": v for k, v in out["euclidean"].items()},
                         **{f"whi_{k}": v for k, v in out["whitened"].items()}))
        print(f"  L{layer:2d}: RSA euclidean {out['euclidean']['rsa_unc']:.3f} -> "
              f"whitened {out['whitened']['rsa_unc']:.3f}  (delta {delta:+.3f})  "
              f"{'LOAD-BEARING' if abs(delta) > 0.1 else 'not load-bearing'}")
        print(f"        ID {out['euclidean']['id_mle']:.2f} -> {out['whitened']['id_mle']:.2f}; "
              f"orthography|chem {out['euclidean']['orth_given_chem']:.3f} -> "
              f"{out['whitened']['orth_given_chem']:.3f}")
    pd.DataFrame(rows).to_parquet(config.RESULTS / "metric_comparison.parquet", index=False)

    print("\n=== §3i: hierarchy (class labels used ONLY for evaluation) ===")
    cls = df.cls.fillna("other").to_numpy()
    organic = ~np.isin(cls, ["inorganic"])
    hrows = []
    for layer in LAYERS:
        X = np.ascontiguousarray(A[:, layer, :])
        B = np.ascontiguousarray(bank[:, layer, :])
        whiten = G.whitening_transform(B)
        for tag, M_ in (("euclidean", X), ("whitened", whiten(X))):
            mu_org = M_[organic].mean(0)
            cosines = {}
            for c in sorted(set(cls[organic])):
                m = (cls == c) & organic
                if m.sum() < 5:
                    continue
                d = M_[m].mean(0) - mu_org
                cosines[c] = float(np.dot(mu_org, d) /
                                   (np.linalg.norm(mu_org) * np.linalg.norm(d) + 1e-12))
            # within-series tangent (alkane), unsupervised direction from §3f L2
            am = (df.series == "alkane").to_numpy()
            sub = M_[am]
            sub = sub - sub.mean(0)
            tang = np.linalg.svd(sub, full_matrices=False)[2][0]
            cls_cos = []
            for c in sorted(set(cls[organic])):
                m = (cls == c) & organic
                if m.sum() < 5:
                    continue
                d = M_[m].mean(0) - mu_org
                cls_cos.append(abs(float(np.dot(tang, d) / (np.linalg.norm(d) + 1e-12))))
            hrows.append(dict(layer=layer, metric=tag, n_classes=len(cosines),
                              mean_abs_cos_parent_child=float(np.mean(np.abs(list(cosines.values())))),
                              mean_abs_cos_series_vs_class=float(np.mean(cls_cos)),
                              cosines=json.dumps({k: round(v, 3) for k, v in cosines.items()})))
            print(f"  L{layer:2d} {tag:9s}: |cos(mu_organic, mu_c - mu_organic)| mean "
                  f"{hrows[-1]['mean_abs_cos_parent_child']:.3f} over {len(cosines)} classes "
                  f"| |cos(series tangent, class offsets)| mean "
                  f"{hrows[-1]['mean_abs_cos_series_vs_class']:.3f}")
    pd.DataFrame(hrows).to_parquet(config.RESULTS / "hierarchy.parquet", index=False)
    print("\n  Park et al. predict ORTHOGONALITY (cos ~ 0) between a parent and its")
    print("  child offsets under the whitened inner product, and structure vanishing")
    print("  under Euclidean. Values near 0 support it; near 1 do not.")


if __name__ == "__main__":
    main()
