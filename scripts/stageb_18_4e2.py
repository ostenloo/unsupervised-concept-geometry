"""§18.4e' addendum — is "observed above null" curvature, or clustering?

The §18.15 null draws arc positions uniformly. The real 600-prompt set is three
groups (harmful / harmless / borderline) with gaps along `d`, and gaps lengthen
graph geodesics. A uniform null therefore under-states the inflation a STRAIGHT
structure with this data's density would produce, which would show up as
apparent curvature. This re-runs the null with arc positions resampled from the
observed projections onto `d`, changing nothing else.

The N9 control is re-run the same way, since clustering also affects kNN
distances and hence id_mle.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np, pandas as pd
import config
from src import geometry as G
from scripts.stageb_18_4ac import fit_coordinate, build_direction
from scripts.stageb_18_4e import Generator, statistics
from scripts.stageb_18_4f import id_mle_of

N_REP = 50
df = pd.read_csv(config.DATA / "stageb" / "prompts.csv")
A = np.load(config.ACTS / "npy" / "stageb.npy")
X = np.ascontiguousarray(A[df.in_primary.to_numpy()][:, config.WURGAFT_LAYER, :])
fit = fit_coordinate(X); d, _ = build_direction(fit, X)

out = {}
print(f"{'k':>4} {'observed':>9} {'null p95 (uniform)':>19} {'null p95 (empirical)':>21} {'verdict':>22}")
for k in config.K_SWEEP:
    obs = statistics(X, k)[0]
    res = {}
    for mode in ("uniform", "empirical"):
        g = Generator(X, d, density=mode)
        vals = [statistics(g.draw(1.000), k)[0] for _ in range(N_REP)]
        res[mode] = dict(mean=float(np.mean(vals)), p95=float(np.percentile(vals, 95)))
    above = obs > res["empirical"]["p95"]
    print(f"{k:>4} {obs:9.3f} {res['uniform']['p95']:19.3f} {res['empirical']['p95']:21.3f} "
          f"{'ABOVE null' if above else 'inside null':>22}")
    out[int(k)] = dict(observed=obs, **{f"null_{m}": res[m] for m in res},
                       above_empirical_null=bool(above))

g = Generator(X, d, density="empirical")
synth = np.array([id_mle_of(g.draw(1.000)) for _ in range(N_REP)])
obs_id = id_mle_of(X)
print(f"\nN9 control, density-matched: id_mle on a TRUE 1-D structure = "
      f"{synth.mean():.3f} [{np.percentile(synth,2.5):.3f}, {np.percentile(synth,97.5):.3f}]")
print(f"observed refusal id_mle = {obs_id:.3f} -> 1-D control "
      f"{'REACHES' if np.percentile(synth,97.5) >= obs_id else 'does not reach'} it")
out["n9_density_matched"] = dict(mean=float(synth.mean()),
                                 ci=[float(np.percentile(synth,2.5)), float(np.percentile(synth,97.5))],
                                 observed=float(obs_id))
(config.RESULTS / "stageb_18_4e2.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {config.RESULTS/'stageb_18_4e2.json'}")
