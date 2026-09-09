"""§18.29 — steering dose-response on refusal-opener log-odds.

Same intervention as §18.23 (single site at L10, matched on each direction's own
projection gap); only the readout and the dose density change. Log-odds is
continuous and needs no decode, so the gate can be evaluated at every dose.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np, pandas as pd, torch
import config
from src import ablation as AB, model as M
from scripts.capture_stageb import refusal_opener_ids
from scripts.stageb_18_4ac import fit_coordinate, build_direction
from scripts.stageb_18_5 import generate, refused, ce_on
from scripts.stageb_18_24_grid import opener_logodds

SB, NPY = config.DATA / "stageb", config.ACTS / "npy"
LAYER = json.loads((config.RESULTS / "stageb_18_5_rerun.json").read_text())["selected_layer"]
MULTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
GEN_AT = [0.5, 1.0, 2.0]           # substring cross-check subset
CE_TOL = 1.0

df = pd.read_csv(SB / "prompts.csv"); A = np.load(NPY / "stageb.npy")
src = df.source.to_numpy(); mh, ml = src == "harmful", src == "harmless"; m4 = mh | ml
X = np.ascontiguousarray(A[:, LAYER, :]); Xf, hf = X[m4], mh[m4]
ev = json.loads((SB / "eval_18_5.json").read_text())
L = M.load(); opener = sorted(set(refusal_opener_ids(L.tokenizer).values()))
lchats = [M.chat(L, p) for p in ev["harmless"]]

fit = fit_coordinate(Xf); d, _ = build_direction(fit, Xf)
if (Xf @ d)[hf].mean() < (Xf @ d)[~hf].mean(): d = -d
v = X[mh].mean(0) - X[ml].mean(0); v = v / np.linalg.norm(v)
pc = fit["pca"].components_[0].copy()
if (Xf @ pc)[hf].mean() < (Xf @ pc)[~hf].mean(): pc = -pc
dirs = {"d": d, "v_ref": v, "pc1": pc}
gaps = {k: float((X[mh] @ u).mean() - (X[ml] @ u).mean()) for k, u in dirs.items()}
print(f"steering at L{LAYER}; matching unit (projection gap): "
      + ", ".join(f"{k} {g:.3f}" for k, g in gaps.items()))

base_lo = opener_logodds(L, lchats, opener)
base_gen = generate(L, lchats); ce_base = ce_on(L, lchats, base_gen)
print(f"baseline log-odds {base_lo.mean():+.3f} | CE {ce_base:.4f} | "
      f"substring refusal {np.mean([refused(t) for t in base_gen]):.3f}\n")

out = {"layer": int(LAYER), "gaps": gaps, "baseline_logodds": float(base_lo.mean()),
       "ce_base": float(ce_base), "curves": {}}
per = {}
print(f"{'dir':<7} {'mult':>5} {'alpha':>7} {'log-odds':>10} {'delta':>8} {'CE':>8} "
      f"{'gate':>6} {'substr':>7}")
for name, u0 in dirs.items():
    u = torch.tensor(np.asarray(u0), dtype=torch.float32); u = u / u.norm()
    out["curves"][name] = []
    for m in MULTS:
        alpha = m * gaps[name]
        with AB.add_direction(L, u, alpha=float(alpha), layers=LAYER):
            lo = opener_logodds(L, lchats, opener)
            ce = ce_on(L, lchats, base_gen)
            sub = None
            if m in GEN_AT:
                sub = float(np.mean([refused(t) for t in generate(L, lchats)]))
        ok = bool(ce <= ce_base + CE_TOL)
        per[(name, m)] = lo
        out["curves"][name].append(dict(multiplier=m, alpha=float(alpha),
                                        logodds=float(lo.mean()),
                                        delta=float(lo.mean() - base_lo.mean()),
                                        ce=float(ce), passes_gate=ok, substring=sub))
        print(f"{name:<7} {m:5.2f} {alpha:7.2f} {lo.mean():10.3f} "
              f"{lo.mean()-base_lo.mean():+8.3f} {ce:8.3f} {'ok' if ok else 'FAIL':>6} "
              f"{'-' if sub is None else f'{sub:.3f}':>7}")

def boot_diff(a, b, n=2000):
    rng = np.random.default_rng(config.SEED)
    dd = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(n)]
    return float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))

print(f"\n=== consistency over doses passing the CE gate ===")
gate = {(n, m): c["passes_gate"] for n in out["curves"]
        for c, m in zip(out["curves"][n], MULTS)}
usable = [m for m in MULTS if all(gate[(n, m)] for n in dirs)]
print(f"  usable doses: {usable}")
ver = {}
for m in usable:
    lo1, hi1 = boot_diff(per[("d", m)], per[("v_ref", m)])
    lo2, hi2 = boot_diff(per[("pc1", m)], per[("v_ref", m)])
    ver[m] = dict(d_vs_vref=[lo1, hi1], overlap_d=bool(lo1 <= 0 <= hi1),
                  pc1_vs_vref=[lo2, hi2], overlap_pc1=bool(lo2 <= 0 <= hi2))
    print(f"  x{m:<5} d-v_ref [{lo1:+.2f},{hi1:+.2f}] {'overlap' if lo1<=0<=hi1 else 'DIVERGE'}"
          f" | pc1-v_ref [{lo2:+.2f},{hi2:+.2f}] {'overlap' if lo2<=0<=hi2 else 'DIVERGE'}")
out["consistency"] = {str(k): v for k, v in ver.items()}
out["verdict"] = dict(
    pc1_matches_vref=all(v["overlap_pc1"] for v in ver.values()),
    d_matches_vref=all(v["overlap_d"] for v in ver.values()),
    n_usable_doses=len(usable))
print(f"\n  PC1 tracks v_ref at every usable dose: {out['verdict']['pc1_matches_vref']}")
print(f"  d   tracks v_ref at every usable dose: {out['verdict']['d_matches_vref']}")
json.dump(out, open(config.RESULTS / "stageb_18_29_dose.json", "w"), indent=2)
print(f"\nwrote {config.RESULTS/'stageb_18_29_dose.json'}")
