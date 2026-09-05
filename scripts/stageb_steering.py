"""SPEC §4f under the two-branch rule pre-registered in §15a.

Behaviour space is the §4f approximation, labelled as such: the next-token
distribution restricted to {refusal openers, compliance openers, other}, mapped
to Hellinger coordinates, reusing the §3f E_BC machinery.

Intervention type: REPLACEMENT, for comparability with Stage A. Wurgaft replace,
Arditi add a scaled vector; the mismatch is stated in the writeup per §4f.
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
from src import model as M
from src import geometry as G
from src import steering as S
from scripts.capture_stageb import REFUSAL_PREFIXES

SB = config.DATA / "stageb"
LAYER = config.WURGAFT_LAYER
K = config.N_WAYPOINTS
N_BASE = 8

COMPLIANCE_PREFIXES = ["Sure", "Here", "Certainly", "Of course", "To ", "The ",
                       "You can", "I'd be happy", "Absolutely", "Yes"]


def opener_ids(tok, prefixes):
    ids = set()
    for p in prefixes:
        for var in (p, " " + p):
            enc = tok.encode(var, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def bin3(probs, ref_ids, com_ids):
    d = np.zeros((probs.shape[0], 3), dtype=np.float64)
    d[:, 0] = probs[:, ref_ids].sum(1)
    d[:, 1] = probs[:, com_ids].sum(1)
    d[:, 2] = np.clip(1 - d[:, :2].sum(1), 0, None)
    return d


@torch.no_grad()
def steer(L, prompts, replacements, ref_ids, com_ids):
    tok = L.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    rep = torch.as_tensor(replacements, dtype=torch.bfloat16, device="cuda")
    fired = {"n": 0}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        t[:, -1, :] = rep
        fired["n"] += 1
        return (t,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else t

    h = L.blocks[LAYER].register_forward_hook(hook)
    try:
        lg = L.model(**enc).logits[:, -1, :].to(torch.float32)
    finally:
        h.remove()
    assert fired["n"] == 1
    return bin3(torch.softmax(lg, -1).cpu().numpy(), ref_ids, com_ids)


def main():
    df = pd.read_csv(SB / "prompts_with_behavior.csv")
    prim = df.in_primary.to_numpy()
    d = df[prim].reset_index(drop=True)
    A = torch.load(config.ACTS / "stageb.pt", weights_only=False)["acts"].numpy()[prim]

    L = M.load()
    tok = L.tokenizer
    ref_ids = opener_ids(tok, REFUSAL_PREFIXES)
    com_ids = sorted(set(opener_ids(tok, COMPLIANCE_PREFIXES)) - set(ref_ids))
    print(f"behaviour bins: {len(ref_ids)} refusal openers, {len(com_ids)} compliance openers, other")

    X = np.ascontiguousarray(A[:, LAYER, :])
    Xp, pca = G.pca_project(X, config.PCA_DIM)
    ids = G.intrinsic_dimension(Xp)
    Y, _, _ = G.isomap_embed(Xp, max(2, G.d_hat_from(ids, "id_mle")), 12)
    coord = Y[:, 0]
    if spearmanr(coord, d.refusal_prob).statistic < 0:
        coord = -coord

    # 10 centroids along the unsupervised coordinate -> the structure to steer along
    o = np.argsort(coord)
    edges = np.linspace(0, len(o), 11).astype(int)
    cents, tvals, members = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = o[a:b]
        cents.append(Xp[idx].mean(0)); tvals.append(coord[idx].mean()); members.append(idx)
    cents = np.stack(cents).astype(np.float32); tvals = np.asarray(tvals)
    sp, ts, _ = S.fit_spline(cents, tvals)

    # Behaviour manifold from the OBSERVED three-bin distributions of the
    # unsteered prompts in each bin.
    #
    # An earlier version built this as [refusal_prob, 1-refusal_prob, 0], i.e.
    # with a hard zero in the "other" bin. That puts M_y on an edge of the
    # simplex while the steered trajectories live in its interior, so E_BC
    # measured the "other"-coordinate mismatch rather than refusal behaviour --
    # and reported divergence while the behaviour trajectories were in fact
    # nearly identical (divergence 0.042, both reaching refusal 1.00). Measure
    # the real distributions instead.
    with torch.no_grad():
        Bc = []
        for idx in members:
            pr = [M.chat(L, d.prompt[k]) for k in idx[:16]]
            enc = tok(pr, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            lg = L.model(**enc).logits[:, -1, :].to(torch.float32)
            Bc.append(bin3(torch.softmax(lg, -1).cpu().numpy(), ref_ids, com_ids).mean(0))
    Bc = np.stack(Bc)
    Bc = Bc / Bc.sum(1, keepdims=True)
    My, _ = S.fit_behavior_manifold(Bc, tvals)
    # Sanity: the points M_y was fitted through must lie ON it.
    resid = S.e_bc(Bc, My) / len(Bc)
    print(f"  M_y fit residual (mean E_BC of its own points): {resid:.4f} "
          f"{'OK' if resid < 0.05 else 'BAD FIT -- E_BC is not interpretable'}")
    print(f"  bin behaviour (refusal, compliance, other), first and last: "
          f"{np.round(Bc[0],3).tolist()} -> {np.round(Bc[-1],3).tolist()}")

    PAIRS = [(0, 9), (1, 8), (0, 7), (2, 9), (1, 9)]
    rows = []
    for (i, j) in PAIRS:
        base = [d.prompt[k] for k in members[i][:N_BASE]]
        base = [M.chat(L, p) for p in base]
        traj = {
            "linear": S.waypoints_linear(cents[i], cents[j], K),
            "manifold": S.waypoints_manifold(sp, ts[i], ts[j], K),
        }
        out = {}
        for name, W in traj.items():
            full = pca.inverse_transform(W.astype(np.float64)).astype(np.float32)
            acc = []
            for s in range(0, K, 8):
                ch = full[s:s + 8]
                pr = [b for _ in ch for b in base]
                rp = np.repeat(ch, len(base), axis=0)
                acc.append(steer(L, pr, rp, ref_ids, com_ids))
            out[name] = np.concatenate(acc).reshape(K, len(base), 3).mean(1)
        # per-waypoint divergence BETWEEN the two strategies
        dvg = np.array([S.bhattacharyya_distance(out["linear"][[w]],
                                                 S.to_hellinger(out["manifold"][[w]]))[0, 0]
                        for w in range(K)])
        rows.append(dict(pair=f"{i}->{j}",
                         ebc_linear=S.e_bc(out["linear"], My),
                         ebc_manifold=S.e_bc(out["manifold"], My),
                         divergence_mean=float(dvg.mean()),
                         divergence_max=float(dvg.max()),
                         refusal_end_linear=float(out["linear"][-1, 0]),
                         refusal_end_manifold=float(out["manifold"][-1, 0]),
                         refusal_start=float(out["linear"][0, 0])))
        print(f"  {i}->{j}: E_BC lin={rows[-1]['ebc_linear']:.3f} man={rows[-1]['ebc_manifold']:.3f} "
              f"| per-waypoint divergence mean={dvg.mean():.4f} max={dvg.max():.4f} "
              f"| refusal {rows[-1]['refusal_start']:.2f}->{rows[-1]['refusal_end_linear']:.2f}(lin)"
              f"/{rows[-1]['refusal_end_manifold']:.2f}(man)")

    res = pd.DataFrame(rows)
    res.to_parquet(config.RESULTS / "stageb_steering.parquet", index=False)
    print("\n" + res.round(4).to_string(index=False))

    # --- §15a two-branch verdict ---
    mean_div = float(res.divergence_mean.mean())
    ebc_gap = float(abs(res.ebc_linear - res.ebc_manifold).mean())
    # Stage A reference scale: the linear/supervised-manifold E_BC gap at L28.
    a3 = pd.read_parquet(config.RESULTS / "level3.parquet")
    a3 = a3[(a3.layer == LAYER) & (a3.coord_layer == LAYER)]
    stagea_gap = float((a3.E_BC_linear - a3.E_BC_manifold_sup).mean())
    agree = mean_div < 0.05 and ebc_gap < 0.25 * stagea_gap
    print(f"\n=== §15a verdict ===")
    print(f"  mean per-waypoint divergence   : {mean_div:.4f}")
    print(f"  mean |E_BC gap| between strats : {ebc_gap:.4f}")
    print(f"  Stage A linear-vs-manifold gap : {stagea_gap:.4f} (reference scale)")
    print(f"  -> BRANCH {'A: trajectories AGREE' if agree else 'B: trajectories DIVERGE'}")
    print("     " + ("Causal complement to the representational negative. Per §15a this is "
                     "CONFIRMATION, NOT DISCOVERY -- near-forced given coord~v_ref=0.946."
                     if agree else
                     "Evidence about the CONSTRUCTION, not about refusal (§15a branch B). "
                     "Strengthens the Level 3 negative; not a Stage B finding."))
    (config.RESULTS / "stageb_steering_verdict.json").write_text(json.dumps(
        {"branch": "A" if agree else "B", "mean_divergence": mean_div,
         "ebc_gap": ebc_gap, "stagea_reference_gap": stagea_gap}, indent=2))


if __name__ == "__main__":
    main()
