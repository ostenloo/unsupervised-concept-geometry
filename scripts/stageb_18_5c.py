"""§18.5c — the sufficiency (addition) arm, absorbing §4f. GPU.

`x <- x + alpha*û` on harmless prompts, for û in {d400, v_ref200, pc1}.

Matched on PROJECTION MAGNITUDE, not raw alpha: each direction's unit step is
its own mean harmful-minus-harmless projection gap, so the curves are compared at
equal displacement along each direction rather than at an arbitrary shared scale.
Comparing raw alpha across directions is meaningless and §18.5c says so.

pc1 is included for the same reason arm 6 exists (§18.17): after §18.4b the
question is whether the three directions are interchangeable, not whether `d`
works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import torch

import config
from src import ablation as AB
from src import model as M
from scripts.stageb_18_5 import generate, refused, boot_ci, ce_on
from scripts.stageb_18_4ac import fit_coordinate, build_direction, difference_in_means

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"
MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]
SELECTED_LAYER = json.loads((config.RESULTS / "stageb_18_5_rerun.json").read_text())["selected_layer"]


def main():
    gate = config.RESULTS / "stageb_18_5_tests.json"
    if not gate.exists() or not json.loads(gate.read_text()).get("all_pass"):
        raise SystemExit("§18.5a' acceptance tests have not passed — §18.5c does not run.")

    df = pd.read_csv(SB / "prompts.csv")
    A = np.load(NPY / "stageb.npy")
    src = df.source.to_numpy()
    mh, ml = src == "harmful", src == "harmless"
    X = np.ascontiguousarray(A[:, SELECTED_LAYER, :])

    # Directions must be rebuilt AT the causal layer. The saved .npz holds the
    # L28 set, and steering with an L28 direction inside L10's residual stream
    # would be a different experiment wearing this one's label.
    print(f"steering at the CAUSAL layer L{SELECTED_LAYER} (§18.21), not L28 — "
          f"directions rebuilt there")
    m4 = mh | ml
    Xf = X[m4]
    hf = mh[m4]
    fit = fit_coordinate(Xf)
    d, _ = build_direction(fit, Xf)
    if (Xf @ d)[hf].mean() < (Xf @ d)[~hf].mean():
        d = -d
    v_ref = difference_in_means(X[mh], X[ml])
    pc1 = fit["pca"].components_[0].copy()
    if (Xf @ pc1)[hf].mean() < (Xf @ pc1)[~hf].mean():
        pc1 = -pc1

    ev = json.loads((SB / "eval_18_5.json").read_text())
    L = M.load()
    lchats = [M.chat(L, p) for p in ev["harmless"]]

    dirs = {"d400": d, "v_ref": v_ref, "pc1": pc1}
    gaps = {k: float((X[mh] @ v).mean() - (X[ml] @ v).mean()) for k, v in dirs.items()}
    print("mean harmful-minus-harmless projection gap (the matching unit):")
    for k, g in gaps.items():
        print(f"  {k:<7} {g:8.3f}")

    base = generate(L, lchats)
    ce_base = ce_on(L, lchats, base)
    out = {"gaps": gaps, "multipliers": MULTIPLIERS,
           "baseline_refusal_harmless": float(np.mean([refused(t) for t in base])),
           "curves": {}}
    print(f"\nbaseline refusal on harmless: {out['baseline_refusal_harmless']:.3f}")
    print(f"  baseline CE {ce_base:.4f}")
    print(f"\n{'direction':<8} {'mult':>5} {'alpha':>8} {'refusal(harmless)':>18} "
          f"{'CE':>9} {'distinct':>9}  gate")

    flags = {}
    for name, v in dirs.items():
        u = torch.tensor(np.asarray(v), dtype=torch.float32)
        u = u / u.norm()
        out["curves"][name] = []
        for m in MULTIPLIERS:
            alpha = m * gaps[name]
            with AB.add_direction(L, u, alpha=float(alpha), layers=SELECTED_LAYER):
                gen = generate(L, lchats)
                ce = ce_on(L, lchats, base)
            r = [refused(t) for t in gen]
            # §18.22 coherence gate: degenerate text scores as "not refusing".
            distinct = float(np.mean([len(set(t.split())) / max(len(t.split()), 1)
                                      for t in gen]))
            ok = (ce <= ce_base + 1.0) and (distinct >= 0.35)
            flags[(name, m)] = r
            out["curves"][name].append(dict(multiplier=m, alpha=float(alpha),
                                            refusal_rate=float(np.mean(r)),
                                            ce=float(ce), distinct_ratio=distinct,
                                            passes_gate=bool(ok)))
            print(f"{name:<8} {m:5.1f} {alpha:8.2f} {np.mean(r):18.3f} "
                  f"{ce:9.3f} {distinct:9.3f}  {'ok' if ok else 'DEGENERATE'}")
            (config.RESULTS / f"gen_18_5c_{name}_{m}.json").write_text(json.dumps(gen, indent=2))

    print("\n=== consistency: do the dose-response curves overlap at every level? ===")
    verdict = {}
    gate_ok = {(n, m): c["passes_gate"] for n in out["curves"]
               for c, m in zip(out["curves"][n], MULTIPLIERS)}
    usable = [m for m in MULTIPLIERS
              if all(gate_ok[(n, m)] for n in ("d400", "v_ref", "pc1"))]
    print(f"  doses passing the coherence gate for all three directions: {usable}")
    if len(usable) < 2:
        out["verdict"] = ("NO POWER: fewer than two doses survive the coherence "
                          "gate, so agreement between curves is not measurable")
        print(f"\n  verdict: {out['verdict']}")
        (config.RESULTS / "stageb_18_5c.json").write_text(json.dumps(out, indent=2))
        raise SystemExit(0)
    for m in usable:
        lo, hi = boot_ci(flags[("d400", m)], flags[("v_ref", m)])
        lo2, hi2 = boot_ci(flags[("pc1", m)], flags[("v_ref", m)])
        ov = (lo <= 0 <= hi)
        ov2 = (lo2 <= 0 <= hi2)
        verdict[m] = dict(d400_vs_vref=[lo, hi], overlap_d400=bool(ov),
                          pc1_vs_vref=[lo2, hi2], overlap_pc1=bool(ov2))
        print(f"  x{m:<4} d400-v_ref [{lo:+.3f},{hi:+.3f}] {'overlap' if ov else 'DIVERGE'} | "
              f"pc1-v_ref [{lo2:+.3f},{hi2:+.3f}] {'overlap' if ov2 else 'DIVERGE'}")
    out["consistency"] = {str(k): v for k, v in verdict.items()}

    all_ov = all(v["overlap_d400"] for v in verdict.values())
    high_only = (not all_ov) and verdict[usable[0]]["overlap_d400"]
    out["verdict"] = ("consistent" if all_ov else
                      "consistent in the linear regime, diverging under strong steering"
                      if high_only else "divergent")
    print(f"\n  verdict: {out['verdict']}")
    (config.RESULTS / "stageb_18_5c.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {config.RESULTS / 'stageb_18_5c.json'}")


if __name__ == "__main__":
    main()
