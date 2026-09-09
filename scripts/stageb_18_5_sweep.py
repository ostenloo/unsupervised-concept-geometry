"""§18.19 — Arditi's layer-selection sweep, which §11f declined to reproduce.

Arm 1 failed at L28. All 32 layers are cached, so the difference-in-means
direction can be built at each and ablated. Refusal drop only; CE is measured for
the selected layer in the re-run, not for all 32.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np, pandas as pd, torch
import config
from src import ablation as AB, model as M
from scripts.stageb_18_5 import generate, refused

SB, NPY = config.DATA / "stageb", config.ACTS / "npy"
df = pd.read_csv(SB / "prompts.csv"); A = np.load(NPY / "stageb.npy")
src = df.source.to_numpy(); mh, ml = src == "harmful", src == "harmless"
ev = json.loads((SB / "eval_18_5.json").read_text())
L = M.load(); hchats = [M.chat(L, p) for p in ev["harmful"]]

base = [refused(t) for t in generate(L, hchats)]
print(f"baseline refusal {np.mean(base):.3f} (n={len(base)})\n")
print(f"{'layer':>5} {'refusal':>8} {'drop':>7}")
out = {"baseline": float(np.mean(base)), "layers": {}}
for layer in range(config.N_LAYERS):
    X = np.ascontiguousarray(A[:, layer, :])
    v = X[mh].mean(0) - X[ml].mean(0); v = v / np.linalg.norm(v)
    u = torch.tensor(v, dtype=torch.float32); u = u / u.norm()
    with AB.ablate(L, u, alpha=1.0):
        r = [refused(t) for t in generate(L, hchats)]
    drop = float(np.mean(base) - np.mean(r))
    out["layers"][layer] = dict(refusal=float(np.mean(r)), drop=drop)
    print(f"{layer:>5} {np.mean(r):8.3f} {drop:7.3f}")
best = max(out["layers"], key=lambda k: out["layers"][k]["drop"])
out["best_layer"] = int(best); out["best_drop"] = out["layers"][best]["drop"]
print(f"\nbest layer {best} with drop {out['best_drop']:.3f} "
      f"-> {'anchor established' if out['best_drop'] >= 0.5 else 'FAILED REPLICATION (§18.19 rule 3)'}")
(config.RESULTS / "stageb_18_5_sweep.json").write_text(json.dumps(out, indent=2))
