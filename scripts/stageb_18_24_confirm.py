"""§18.24 confirmation: generation-based check at three cells, plus the d-nulls
re-tested at Arditi's own position choice.

The grid used the cheap log-odds metric. This confirms it with the substring rate
and the CE capability gate at: the grid argmax, the external anchor (11,-1), and
the inherited cell (10,-1). It then re-runs arms 2/3/4b at (10,-5) -- position
-5 is `<|eot_id|>`, the cell Arditi's Table 5 selects for Llama-3 8B -- so the
branch-3 qualification rests on a measurement rather than an assumption.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np, pandas as pd, torch
import config
from src import ablation as AB, model as M
from scripts.stageb_18_4ac import fit_coordinate, build_direction
from scripts.stageb_18_5 import generate, refused, ce_on, boot_ci

SB, NPY = config.DATA / "stageb", config.ACTS / "npy"
CELLS = [(12, -5), (11, -1), (10, -1)]
ARM_CELLS = [(10, -1), (10, -5)]

df = pd.read_csv(SB / "prompts.csv"); z = np.load(NPY / "stageb_positions.npz")
kept, positions, keep_layers = z["kept"], list(z["positions"]), list(z["keep_layers"])
means_h, means_l = z["means_h"], z["means_l"]
src = df.source.to_numpy(); mh, ml = src == "harmful", src == "harmless"; m4 = mh | ml
ev = json.loads((SB / "eval_18_5.json").read_text())
L = M.load()
hchats = [M.chat(L, p) for p in ev["harmful"]]; lchats = [M.chat(L, p) for p in ev["harmless"]]
base_h = [refused(t) for t in generate(L, hchats)]; base_l = generate(L, lchats)
ce_base = ce_on(L, lchats, base_l)
print(f"baseline refusal {np.mean(base_h):.3f}, CE {ce_base:.4f}\n")

def unit(v):
    t = torch.tensor(np.asarray(v), dtype=torch.float32); return t / t.norm()

def vref(layer, pos):
    pi = positions.index(pos)
    v = means_h[pi, layer] - means_l[pi, layer]; return v / np.linalg.norm(v)

out = {"baseline_refusal": float(np.mean(base_h)), "ce_base": float(ce_base), "cells": {}}
print(f"{'cell':>12} {'refusal':>8} {'drop':>7} {'CE':>8}")
for layer, pos in CELLS:
    with AB.ablate(L, unit(vref(layer, pos)), alpha=1.0):
        r = [refused(t) for t in generate(L, hchats)]; ce = ce_on(L, lchats, base_l)
    out["cells"][f"L{layer}_p{pos}"] = dict(refusal=float(np.mean(r)),
                                            drop=float(np.mean(base_h) - np.mean(r)),
                                            ce=float(ce))
    print(f"  ({layer:>2},{pos:>3}) {np.mean(r):8.3f} {np.mean(base_h)-np.mean(r):7.3f} {ce:8.4f}")

print(f"\n=== arms 2/3/4b at the two cells (branch 3) ===")
out["arms"] = {}
for layer, pos in ARM_CELLS:
    pi, li = positions.index(pos), keep_layers.index(layer)
    X = np.ascontiguousarray(kept[pi, li]); Xf, hf = X[m4], mh[m4]
    v = vref(layer, pos)
    fit = fit_coordinate(Xf); d, _ = build_direction(fit, Xf)
    if (Xf @ d)[hf].mean() < (Xf @ d)[~hf].mean(): d = -d
    dp = d - (d @ v) * v; dp /= np.linalg.norm(dp)
    tgt = float(d @ v)
    rng = np.random.default_rng(config.SEED); con = []
    for _ in range(3):
        r0 = rng.normal(size=X.shape[1]).astype(np.float32); r0 -= (r0 @ v) * v
        r0 /= np.linalg.norm(r0)
        c = tgt * v + np.sqrt(1 - tgt ** 2) * r0; con.append(c / np.linalg.norm(c))
    res = {}
    for nm, vec in [("2_d", d), ("3_d_perp", dp)] + [(f"4b_{i}", con[i]) for i in range(3)]:
        with AB.ablate(L, unit(vec), alpha=1.0):
            rr = [refused(t) for t in generate(L, hchats)]
        res[nm] = float(np.mean(base_h) - np.mean(rr))
    cm = float(np.mean([res[f"4b_{i}"] for i in range(3)]))
    out["arms"][f"L{layer}_p{pos}"] = dict(cos_d_vref=tgt, drops=res, cos_matched_mean=cm)
    print(f"  ({layer},{pos}) cos(d,vref)={tgt:.3f} | d drop {res['2_d']:.3f} | "
          f"d_perp {res['3_d_perp']:.3f} | cos-matched null {cm:.3f} | "
          f"d {'inside' if abs(res['2_d']-cm) < 0.10 else 'OUTSIDE'} its matched null")
json.dump(out, open(config.RESULTS / "stageb_18_24_confirm.json", "w"), indent=2)
print(f"\nwrote {config.RESULTS/'stageb_18_24_confirm.json'}")
