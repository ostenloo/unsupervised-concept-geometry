"""§18.5b metric (c): stratified spot-check of ablation completions.

The refusal-substring rate cannot distinguish "complied" from "produced
incoherent text", and an ablation that lobotomises the model scores as success.
CE is the primary guard; this samples completions per arm so the failure mode is
checked against text rather than asserted.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np
import config

ARMS = ["5_baseline", "1_v_ref", "6_pc1", "2_d", "3_d_perp", "4_random_0", "4b_cos0.68_0"]
rng = np.random.default_rng(config.SEED)
rows = []
print(f"{'arm':<16} {'distinct':>9} {'empty':>7} {'words':>7}")
for a in ARMS:
    f = config.RESULTS / (f"gen_18_5r_{a}.json" if a != "5_baseline"
                          else "gen_18_5_5_baseline.json")
    g = json.loads(f.read_text())["harmful"]
    dis = float(np.mean([len(set(t.split())) / max(len(t.split()), 1) for t in g]))
    emp = float(np.mean([len(t.strip()) == 0 for t in g]))
    ln = float(np.mean([len(t.split()) for t in g]))
    print(f"{a:<16} {dis:9.3f} {emp:7.2f} {ln:7.1f}")
    for i in rng.choice(len(g), 6, replace=False):
        rows.append({"arm": a, "index": int(i), "text": g[int(i)][:300]})
(config.RESULTS / "spotcheck_18_5.json").write_text(json.dumps(rows, indent=2))
print(f"\n{len(rows)} completions sampled -> results/spotcheck_18_5.json")
for r in rows:
    if r["arm"] in ("1_v_ref", "6_pc1", "2_d"):
        print(f"  [{r['arm']:<8}] {r['text'][:105]!r}")
