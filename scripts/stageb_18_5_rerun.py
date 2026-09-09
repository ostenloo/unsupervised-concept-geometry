"""§18.19 rules 1-2: select the layer by the pre-registered rule, re-run the arms.

Rule 1: largest refusal drop, subject to CE on harmless within 0.05 of baseline;
ties to the earlier layer. The sweep measured drop only, so CE is measured here
for each candidate before the selection is final -- a layer that removes refusal
by destroying the model does not qualify.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np, pandas as pd, torch
import config
from src import ablation as AB, model as M, geometry as G
from scripts.stageb_18_4ac import fit_coordinate, build_direction, difference_in_means
from scripts.stageb_18_5 import generate, refused, ce_on, boot_ci

SB, NPY = config.DATA / "stageb", config.ACTS / "npy"
CE_TOL = 0.05
df = pd.read_csv(SB / "prompts.csv"); A = np.load(NPY / "stageb.npy")
src = df.source.to_numpy(); mh, ml = src == "harmful", src == "harmless"; m4 = mh | ml
ev = json.loads((SB / "eval_18_5.json").read_text())
sweep = json.loads((config.RESULTS / "stageb_18_5_sweep.json").read_text())

L = M.load()
hchats = [M.chat(L, p) for p in ev["harmful"]]
lchats = [M.chat(L, p) for p in ev["harmless"]]
base_h = [refused(t) for t in generate(L, hchats)]
base_l = generate(L, lchats)
ce_base = ce_on(L, lchats, base_l)
print(f"baseline: refusal {np.mean(base_h):.3f}, CE {ce_base:.4f}\n")

def vref_at(layer):
    X = np.ascontiguousarray(A[:, layer, :])
    return difference_in_means(X[mh], X[ml])

def unit(v):
    t = torch.tensor(np.asarray(v), dtype=torch.float32); return t / t.norm()

cands = sorted([int(k) for k, v in sweep["layers"].items() if v["drop"] >= 0.5],
               key=lambda k: (-sweep["layers"][str(k)]["drop"], k))
print(f"candidates with drop >= 0.5: {cands}")
selected, ce_by_layer = None, {}
for layer in cands:
    with AB.ablate(L, unit(vref_at(layer)), alpha=1.0):
        ce = ce_on(L, lchats, base_l)
    ce_by_layer[layer] = float(ce)
    ok = ce <= ce_base + CE_TOL
    print(f"  L{layer}: drop {sweep['layers'][str(layer)]['drop']:.3f}, CE {ce:.4f} "
          f"(baseline {ce_base:.4f} + {CE_TOL}) -> {'QUALIFIES' if ok else 'fails CE gate'}")
    if ok and selected is None:
        selected = layer
print(f"\nselected layer: {selected}")
if selected is None:
    (config.RESULTS / "stageb_18_5_rerun.json").write_text(json.dumps(
        {"selected_layer": None, "ce_by_layer": ce_by_layer, "ce_base": float(ce_base),
         "verdict": "no layer removes refusal without failing the CE capability gate"}, indent=2))
    raise SystemExit("No layer passes the CE gate — §18.19 rule 3 applies.")

# --- rebuild every direction AT the selected layer -----------------------
X = np.ascontiguousarray(A[:, selected, :]); Xf = X[m4]; hf = mh[m4]
fit = fit_coordinate(Xf); d, lin = build_direction(fit, Xf)
if (Xf @ d)[hf].mean() < (Xf @ d)[~hf].mean(): d = -d
v = vref_at(selected)
pc1 = fit["pca"].components_[0].copy()
if (Xf @ pc1)[hf].mean() < (Xf @ pc1)[~hf].mean(): pc1 = -pc1
dp = d - (d @ v) * v; dp /= np.linalg.norm(dp)
target = float(d @ v)
rng = np.random.default_rng(config.SEED)
randoms = [(lambda r: r / np.linalg.norm(r))(rng.normal(size=X.shape[1]).astype(np.float32))
           for _ in range(5)]
constrained = []
for _ in range(5):
    r = rng.normal(size=X.shape[1]).astype(np.float32); r -= (r @ v) * v; r /= np.linalg.norm(r)
    c = target * v + np.sqrt(1 - target ** 2) * r; constrained.append(c / np.linalg.norm(c))
print(f"at L{selected}: cos(d,v_ref) = {target:+.4f} (arm 4b constraint), "
      f"cos(pc1,v_ref) = {float(pc1 @ v):+.4f}, linearity CV R2 = {lin['r2_cv_64']:.3f}")

arms = [("1_v_ref", v), ("2_d", d), ("3_d_perp", dp), ("6_pc1", pc1)]
arms += [(f"4_random_{i}", randoms[i]) for i in range(5)]
arms += [(f"4b_cos{target:.2f}_{i}", constrained[i]) for i in range(5)]
res = {"5_baseline": dict(refusal_rate_harmful=float(np.mean(base_h)),
                          refusal_rate_harmless=float(np.mean([refused(t) for t in base_l])),
                          ce_harmless=float(ce_base))}
flags = {"5_baseline": base_h}
print(f"\n{'arm':<20} {'refusal(harmful)':>17} {'refusal(harmless)':>18} {'CE':>8}")
print(f"{'5_baseline':<20} {np.mean(base_h):17.3f} "
      f"{res['5_baseline']['refusal_rate_harmless']:18.3f} {ce_base:8.4f}")
for name, vec in arms:
    with AB.ablate(L, unit(vec), alpha=1.0):
        hg, lg = generate(L, hchats), generate(L, lchats)
        ce = ce_on(L, lchats, base_l)
    rh = [refused(t) for t in hg]; flags[name] = rh
    res[name] = dict(refusal_rate_harmful=float(np.mean(rh)),
                     refusal_rate_harmless=float(np.mean([refused(t) for t in lg])),
                     ce_harmless=float(ce))
    print(f"{name:<20} {np.mean(rh):17.3f} {res[name]['refusal_rate_harmless']:18.3f} {ce:8.4f}")
    (config.RESULTS / f"gen_18_5r_{name}.json").write_text(json.dumps({"harmful": hg, "harmless": lg}, indent=2))

b = res["5_baseline"]["refusal_rate_harmful"]
drop = {k: b - x["refusal_rate_harmful"] for k, x in res.items()}
rnd = np.mean([drop[k] for k in drop if k.startswith("4_")])
con = np.mean([drop[k] for k in drop if k.startswith("4b_")])
print(f"\n=== §18.5b branch rules at L{selected} ===")
print(f"  drops: v_ref {drop['1_v_ref']:.3f} | d {drop['2_d']:.3f} | pc1 {drop['6_pc1']:.3f} "
      f"| d_perp {drop['3_d_perp']:.3f} | random {rnd:.3f} | cos-matched {con:.3f}")
r2 = drop["2_d"] >= 0.8 * drop["1_v_ref"]
ce_ok = abs(res["2_d"]["ce_harmless"] - ce_base) <= CE_TOL
print(f"  arm2 >= 0.8*arm1: {r2} ({drop['2_d']:.3f} vs {0.8*drop['1_v_ref']:.3f}); "
      f"arm2 CE within {CE_TOL} of baseline: {ce_ok}")
print(f"  -> d {'IS' if (r2 and ce_ok) else 'is NOT'} causally equivalent to v_ref")
lo3, hi3 = boot_ci(flags["3_d_perp"], flags["4_random_0"])
print(f"  arm3 vs random null, refusal-difference CI [{lo3:+.3f}, {hi3:+.3f}]")
print(f"  arm3 drop {drop['3_d_perp']:.3f} vs nulls (random {rnd:.3f}, cos-matched {con:.3f})")
lo6, hi6 = boot_ci(flags["6_pc1"], flags["1_v_ref"])
print(f"  arm6 vs arm1 CI [{lo6:+.3f}, {hi6:+.3f}] -> PC1 "
      f"{'interchangeable with' if lo6 <= 0 <= hi6 else 'NOT interchangeable with'} v_ref (§18.17)")
(config.RESULTS / "stageb_18_5_rerun.json").write_text(json.dumps(
    dict(selected_layer=selected, ce_by_layer=ce_by_layer, ce_base=float(ce_base),
         target_cos=target, cos_pc1_vref=float(pc1 @ v), linearity_cv_r2=lin["r2_cv_64"],
         results=res, drops=drop, random_mean_drop=float(rnd), constrained_mean_drop=float(con),
         arm2_equivalent=bool(r2 and ce_ok), arm3_vs_random_ci=[lo3, hi3],
         arm6_vs_arm1_ci=[lo6, hi6]), indent=2))
print(f"\nwrote {config.RESULTS / 'stageb_18_5_rerun.json'}")
