"""SPEC §3f Level 3 — behavioural equivalence via steering (the causal test).

Three strategies between Family-2 centroid pairs: linear, manifold-unsupervised
(intrinsic parameter = the Level-2 coordinate, no labels), manifold-supervised
(intrinsic parameter = ground-truth carbon count). Scored by cumulative
Bhattacharyya energy to the behaviour manifold.

Run at layer 8 (primary, §12a tie-break) and layer 28 (Wurgaft comparability),
with §12f's r statistic, compound-level cluster bootstrap, and the pre-declared
disagreement rule.

    python scripts/level3_steering.py
"""
from __future__ import annotations

import itertools, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from src import model as M
from src import geometry as G
from src import evaluation as E
from src import steering as S

# SPEC §11e: the 8 base prompts are PARAPHRASES of the Family-2 frame, so the
# behaviour manifold fitted from Family-2 outputs is the one the trajectories are
# scored against. Drawing them from Family 1 would average over a different
# behaviour manifold than the one fitted.
PARAPHRASES = {
    "alkane": [
        "The straight-chain alkane with {n} carbon atoms is called",
        "The name of the straight-chain alkane containing {n} carbon atoms is",
        "A straight-chain alkane that has {n} carbon atoms is known as",
        "The unbranched alkane with {n} carbon atoms is called",
        "In organic chemistry, the straight-chain alkane with {n} carbon atoms is named",
        "The linear alkane possessing {n} carbon atoms is called",
        "Chemists call the straight-chain alkane with {n} carbon atoms",
        "The straight-chain alkane having {n} carbon atoms goes by the name",
    ],
    "alcohol": [
        "The straight-chain primary alcohol with {n} carbon atoms is called",
        "The name of the straight-chain primary alcohol containing {n} carbon atoms is",
        "A straight-chain primary alcohol that has {n} carbon atoms is known as",
        "The unbranched primary alcohol with {n} carbon atoms is called",
        "In organic chemistry, the straight-chain primary alcohol with {n} carbon atoms is named",
        "The linear primary alcohol possessing {n} carbon atoms is called",
        "Chemists call the straight-chain primary alcohol with {n} carbon atoms",
        "The straight-chain primary alcohol having {n} carbon atoms goes by the name",
    ],
}
SERIES_NAMES = {
    "alkane": ["methane", "ethane", "propane", "butane", "pentane",
               "hexane", "heptane", "octane", "nonane", "decane"],
    "alcohol": ["methanol", "ethanol", "propanol", "butanol", "pentanol",
                "hexanol", "heptanol", "octanol", "nonanol", "decanol"],
}
K_WAY = config.N_WAYPOINTS
N_PAIRS = config.MAX_STEER_PAIRS


def name_token_ids(tok, names):
    out = {}
    for nm in names:
        ids = set()
        for var in (nm, " " + nm, nm.capitalize(), " " + nm.capitalize()):
            enc = tok.encode(var, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[nm] = sorted(ids)
    return out


def bin_distribution(probs, name_ids, names):
    d = np.zeros((probs.shape[0], len(names) + 1), dtype=np.float64)
    for j, nm in enumerate(names):
        d[:, j] = probs[:, name_ids[nm]].sum(axis=1)
    d[:, -1] = np.clip(1.0 - d[:, :-1].sum(axis=1), 0.0, None)
    return d


@torch.no_grad()
def capture_paraphrases(L, series):
    """[8 paraphrases, 10 n, n_layers, hidden] activations + [8,10,11] behaviour."""
    names = SERIES_NAMES[series]
    tok = L.tokenizer
    nid = name_token_ids(tok, names)
    prompts = [p.format(n=n) for p in PARAPHRASES[series] for n in range(1, 11)]
    acts = M.capture(L, prompts)                      # [80, 32, 4096] float32
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    logits = L.model(**enc).logits[:, -1, :].to(torch.float32)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    dist = bin_distribution(probs, nid, names)
    return (acts.numpy().reshape(8, 10, config.N_LAYERS, config.HIDDEN_DIM),
            dist.reshape(8, 10, len(names) + 1), nid)


@torch.no_grad()
def steer_batch(L, layer, prompts, replacements, nid, names):
    """Forward `prompts` replacing the last-token activation at `layer`, per row."""
    tok = L.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    rep = torch.as_tensor(replacements, dtype=torch.bfloat16, device="cuda")
    fired = {"n": 0}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        t[:, -1, :] = rep
        fired["n"] += 1
        return (t,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else t

    h = L.blocks[layer].register_forward_hook(hook)
    try:
        logits = L.model(**enc).logits[:, -1, :].to(torch.float32)
    finally:
        h.remove()
    assert fired["n"] == 1, f"steering hook fired {fired['n']} times"
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return bin_distribution(probs, nid, names)


def measure_efficacy(L, series, layer, pca, Zc, nid, names, pairs):
    """SPEC §12i precondition: does replacement at this layer move behaviour?"""
    src, tgt = [], []
    for (i, j) in pairs[:6]:
        base = [p.format(n=i + 1) for p in PARAPHRASES[series]]
        for bucket, idx in ((src, i), (tgt, j)):
            full = pca.inverse_transform(Zc[idx][None, :].astype(np.float64)).astype(np.float32)
            d = steer_batch(L, layer, base, np.repeat(full, len(base), axis=0), nid, names)
            bucket.append(d.mean(axis=0)[idx])
    return float(np.mean(src)), float(np.mean(tgt))


def run_series_layer(L, series, layer, acts, dists, nid, unsup_coord):
    names = SERIES_NAMES[series]
    A = acts[:, :, layer, :].reshape(80, -1).astype(np.float32)     # 8 x 10 flattened
    pca_dim = min(config.PCA_DIM, A.shape[0] - 1)
    Z, pca = G.pca_project(A, n_components=pca_dim)
    Zc = Z.reshape(8, 10, -1).mean(axis=0)                          # centroids [10, d]
    Bc = dists.mean(axis=0)                                         # behaviour [10, 11]

    n_carbons = np.arange(1, 11, dtype=float)
    sp_unsup, t_unsup, _ = S.fit_spline(Zc, unsup_coord)
    sp_sup, t_sup, _ = S.fit_spline(Zc, n_carbons)
    My, _ = S.fit_behavior_manifold(Bc, n_carbons)
    order_agree = float(np.mean(np.argsort(np.argsort(unsup_coord))
                                == np.argsort(np.argsort(n_carbons))))

    pairs = sorted(itertools.combinations(range(10), 2),
                   key=lambda p: -abs(p[0] - p[1]))[:N_PAIRS]
    src_mass, tgt_mass = measure_efficacy(L, series, layer, pca, Zc, nid, names, pairs)
    efficacious = S.intervention_is_efficacious(tgt_mass, src_mass)

    per_pair = {}
    for (i, j) in pairs:
        base = [p.format(n=i + 1) for p in PARAPHRASES[series]]
        traj = {}
        traj["lin"] = S.waypoints_linear(Zc[i], Zc[j], K_WAY)
        traj["unsup"] = S.waypoints_manifold(sp_unsup, t_unsup[i], t_unsup[j], K_WAY)
        traj["sup"] = S.waypoints_manifold(sp_sup, t_sup[i], t_sup[j], K_WAY)
        res = {}
        for strat, W in traj.items():
            full = pca.inverse_transform(W.astype(np.float64)).astype(np.float32)
            out = []
            for s in range(0, K_WAY, 8):
                chunk = full[s:s + 8]
                pr = [b for _ in chunk for b in base]
                rp = np.repeat(chunk, len(base), axis=0)
                out.append(steer_batch(L, layer, pr, rp, nid, names))
            # average behaviour over the 8 base prompts, per waypoint
            d = np.concatenate(out, axis=0).reshape(K_WAY, len(base), -1).mean(axis=1)
            res[strat] = S.e_bc(d, My)
            if strat == "lin" and (i, j) == pairs[0]:
                res["_traj_lin"] = d
            if strat == "unsup" and (i, j) == pairs[0]:
                res["_traj_unsup"] = d
        per_pair[(i, j)] = {k: v for k, v in res.items() if not k.startswith("_")}
    return per_pair, order_agree, pairs, src_mass, tgt_mass, efficacious


def main():
    L = M.load()
    df = pd.read_csv(config.DATA / "compounds.csv")
    f1 = torch.load(config.ACTS / "family1.pt", weights_only=False)
    f1_acts = f1["acts"].numpy()

    out_rows, t0 = [], time.time()
    for series in ("alkane", "alcohol"):
        acts, dists, nid = capture_paraphrases(L, series)
        print(f"\n[{series}] paraphrase capture {acts.shape}; "
              f"mean mass on concept names {1 - dists[..., -1].mean():.2f}")
        # §12i: steering is causally inert below layer 22 (measured step function),
        # so Level 3 is only well posed in the live region. Layer 8 is kept for the
        # record as the §12a primary, flagged INERT. The final entry is a
        # best-case hybrid: unsupervised coordinate taken at layer 8 (the most
        # representationally aligned layer) while steering at a live layer.
        for layer, coord_layer in ((8, 8), (22, 22), (24, 24),
                                   (config.WURGAFT_LAYER, config.WURGAFT_LAYER),
                                   (31, 31), (config.WURGAFT_LAYER, 8)):
            # UNSUPERVISED coordinate: Level 2, from Family 1, no labels.
            mask = (df.series == series).to_numpy()
            X = np.ascontiguousarray(f1_acts[:, coord_layer, :])
            Xp, _ = G.pca_project(X, config.PCA_DIM)
            ids = G.intrinsic_dimension(Xp)
            Y, _, _ = G.isomap_embed(Xp, G.d_hat_from(ids, "id_mle"), 12)
            sub = df[mask].copy()
            sub["coord"] = E.series_coordinate(Y, mask)
            sub = sub.sort_values("series_index")
            unsup = sub["coord"].to_numpy()
            if np.corrcoef(unsup, sub.n_carbons)[0, 1] < 0:
                unsup = -unsup                      # sign of an axis is arbitrary

            per_pair, agree, pairs, sm, tm, eff = run_series_layer(
                L, series, layer, acts, dists, nid, unsup)
            tot = {k: float(np.mean([v[k] for v in per_pair.values()]))
                   for k in ("lin", "unsup", "sup")}
            r = S.r_statistic(tot["unsup"], tot["sup"], tot["lin"])
            bs = S.cluster_bootstrap_r(per_pair, list(range(10)))
            row = dict(series=series, layer=layer, coord_layer=coord_layer,
                       n_pairs=len(pairs),
                       order_agreement=agree, src_mass=sm, tgt_mass=tm,
                       efficacious=eff,
                       E_BC_linear=tot["lin"], E_BC_manifold_unsup=tot["unsup"],
                       E_BC_manifold_sup=tot["sup"], r=r, **bs)
            out_rows.append(row)
            tag = f"L{layer}" + (f" (coord L{coord_layer})" if coord_layer != layer else "")
            print(f"  {tag:20s} efficacy src={sm:.3f} tgt={tm:.3f} "
                  f"{'LIVE' if eff else 'INERT -- r is meaningless (§12i)'}")
            print(f"            E_BC lin={tot['lin']:.3f} unsup={tot['unsup']:.3f} "
                  f"sup={tot['sup']:.3f}  r={r:.3f} "
                  f"[{bs['r_lo']:.2f},{bs['r_hi']:.2f}] "
                  f"degen={bs['degenerate']} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(out_rows)
    res.to_parquet(config.RESULTS / "level3.parquet", index=False)
    print("\n" + res.to_string(index=False))

    # §12f disagreement rule
    print("\n=== §12f disagreement rule ===")
    verdict = {}
    for series, g in res.groupby("series"):
        g = g[g.coord_layer == g.layer]
        a = g[g.layer == 8].iloc[0]
        b = g[g.layer == config.WURGAFT_LAYER].iloc[0]
        if not a["efficacious"] or not b["efficacious"]:
            dead = [int(x["layer"]) for x in (a, b) if not x["efficacious"]]
            v = (f"PRECONDITION FAILED at layer(s) {dead}: replacement does not move "
                 f"behaviour there, so r is not interpretable (§12i)")
        elif a["degenerate"] or b["degenerate"]:
            v = "DEGENERATE: denominator not significantly positive; report raw E_BC"
        else:
            flip = (a["r"] < 0.5) != (b["r"] < 0.5)
            se_a = (a["r_hi"] - a["r_lo"]) / 3.92
            se_b = (b["r_hi"] - b["r_lo"]) / 3.92
            pooled = float(np.sqrt(se_a ** 2 + se_b ** 2))
            big = abs(a["r"] - b["r"]) > 2 * pooled
            v = ("LAYER-SENSITIVE (verdict flip): neither layer is the result" if flip
                 else "same verdict, layer-dependent effect size" if big
                 else "layer-robust")
            v += f"  |r(8)-r(28)|={abs(a['r']-b['r']):.3f}, 2*SE_pooled={2*pooled:.3f}"
        verdict[series] = v
        print(f"  {series:8s} r(8)={a['r']:.3f} r(28)={b['r']:.3f} -> {v}")
    (config.RESULTS / "level3_verdict.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
