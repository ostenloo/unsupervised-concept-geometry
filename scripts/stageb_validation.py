"""SPEC §4e behavioural validation and §12h nuisance-variable controls.

§12h was pre-registered before Stage B activations were examined, because §12g
found the same class of confound in Stage A after the fact. The refusal analog of
name-string similarity is prompt SURFACE similarity; the co-varying nuisances are
prompt LENGTH and SOURCE identity, which differ by construction across the three
corpora.

    python scripts/stageb_validation.py
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


def surface_distance(prompts, kind="edit"):
    """§12h: prompt surface similarity, two ways."""
    import difflib
    n = len(prompts)
    D = np.zeros((n, n), dtype=np.float32)
    if kind == "edit":
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = 1.0 - difflib.SequenceMatcher(
                    None, prompts[i], prompts[j]).ratio()
    else:  # token-level Jaccard over word 3-grams
        def grams(s):
            w = s.lower().split()
            return set(zip(w, w[1:], w[2:])) or set(w)
        gs = [grams(p) for p in prompts]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = gs[i], gs[j]
                u = len(a | b)
                D[i, j] = D[j, i] = 1.0 - (len(a & b) / u if u else 0.0)
    return D


def partial(a, b, c):
    return E.partial_spearman(a, b, c)


def main():
    df = pd.read_csv(SB / "prompts_with_behavior.csv")
    blob = torch.load(config.ACTS / "stageb.pt", weights_only=False)
    acts = blob["acts"].numpy()
    prim = df.in_primary.to_numpy()
    d = df[prim].reset_index(drop=True)
    A = acts[prim]

    prompts = list(d.prompt)
    D_edit = surface_distance(prompts, "edit")
    D_gram = surface_distance(prompts, "gram")
    length = d.n_words.to_numpy(dtype=float)
    D_len = np.abs(length[:, None] - length[None, :]).astype(np.float32)
    harmful = (d.source == "harmful").to_numpy()
    D_src = (d.source.to_numpy()[:, None] != d.source.to_numpy()[None, :]).astype(np.float32)

    iu = np.triu_indices(len(d), 1)
    rows = []
    for layer in (8, 22, config.WURGAFT_LAYER, 31):
        X = np.ascontiguousarray(A[:, layer, :])
        Xp, _ = G.pca_project(X, config.PCA_DIM)
        ids = G.intrinsic_dimension(Xp)
        Y, D_geo, _ = G.isomap_embed(Xp, max(2, G.d_hat_from(ids, "id_mle")), 12)
        coord = Y[:, 0]
        if spearmanr(coord, d.refusal_prob).statistic < 0:
            coord = -coord

        # --- §4e: behaviour, used for the first time here --------------------
        rp = d.refusal_prob.to_numpy()
        rg = d.refusal_generated.to_numpy(dtype=float)
        raw_prob = float(spearmanr(coord, rp).statistic)
        raw_gen = float(spearmanr(coord, rg).statistic)
        # partial on length and on source, per §12h step 4
        par_prob_len = partial(coord, rp, length)
        par_gen_len = partial(coord, rg, length)

        # --- §12h: pairwise surface controls ---------------------------------
        g = D_geo[iu]
        row = dict(
            layer=layer, id_mle=ids["id_mle"],
            coord_vs_refusal_prob=raw_prob, coord_vs_refusal_generated=raw_gen,
            coord_vs_prob_given_length=par_prob_len,
            coord_vs_gen_given_length=par_gen_len,
            coord_vs_length=float(spearmanr(coord, length).statistic),
            rho_geo_edit=float(spearmanr(g, D_edit[iu]).statistic),
            rho_geo_gram=float(spearmanr(g, D_gram[iu]).statistic),
            rho_geo_len=float(spearmanr(g, D_len[iu]).statistic),
            rho_geo_source=float(spearmanr(g, D_src[iu]).statistic),
        )
        # behaviour distance: |refusal_prob difference| as the pairwise target
        D_beh = np.abs(rp[:, None] - rp[None, :]).astype(np.float32)
        b = D_beh[iu]
        row["rho_geo_behaviour"] = float(spearmanr(g, b).statistic)
        row["partial_geo_beh_given_edit"] = partial(g, b, D_edit[iu])
        row["partial_geo_edit_given_beh"] = partial(g, D_edit[iu], b)
        row["partial_geo_beh_given_len"] = partial(g, b, D_len[iu])
        row["partial_geo_len_given_beh"] = partial(g, D_len[iu], b)
        rows.append(row)
        print(f"  L{layer:2d}: coord~refusal_prob {raw_prob:+.3f} "
              f"(|length {par_prob_len:+.3f}) | geo~behaviour {row['rho_geo_behaviour']:+.3f} "
              f"(|edit {row['partial_geo_beh_given_edit']:+.3f}) "
              f"| geo~edit {row['rho_geo_edit']:+.3f} "
              f"(|beh {row['partial_geo_edit_given_beh']:+.3f}) "
              f"| geo~len {row['rho_geo_len']:+.3f}")

    res = pd.DataFrame(rows)
    res.to_parquet(config.RESULTS / "stageb_validation.parquet", index=False)

    # --- XSTest contrast pairs: minimal wording change, maximal harm change ---
    print("\n=== XSTest contrast control (§12h) ===")
    ct = df[df.source == "xstest_contrast"].reset_index(drop=True)
    sf = df[df.source == "borderline"].reset_index(drop=True)
    print(f"  {len(sf)} safe vs {len(ct)} contrast prompts")
    print(f"  mean surface distance within safe set : "
          f"{surface_distance(list(sf.prompt))[np.triu_indices(len(sf),1)].mean():.3f}")
    both = pd.concat([sf, ct], ignore_index=True)
    Dm = surface_distance(list(both.prompt))
    cross = Dm[:len(sf), len(sf):]
    print(f"  mean surface distance safe<->contrast : {cross.mean():.3f}")
    print(f"  refusal rate: safe {sf.refusal_generated.mean():.3f} "
          f"vs contrast {ct.refusal_generated.mean():.3f}")


if __name__ == "__main__":
    main()
