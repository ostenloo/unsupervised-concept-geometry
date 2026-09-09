"""§18.7 platform acceptance test — run late, and recorded as such.

§18.7 required reproducing a logged number BEFORE trusting any locally computed
one. It was not run before §18.4; this runs it after. That ordering is weaker --
a failure here would invalidate results already recorded -- and the lateness is
the point of the record.

Compares against the full-precision value in results/stageb_structure.parquet,
not §14a's rounded 1.020. Both eigen_solver settings are reported: "auto" is what
the logged run used, "dense" is what §18.4 pinned under N5.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import ast, json
from pathlib import Path
import numpy as np, pandas as pd
import config
from src import geometry as G

src = Path("scripts/stageb_structure.py").read_text()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("geodesic_chord", "difference_in_means"):
        exec(compile(ast.Module([node], []), "s.py", "exec"), ns)
geodesic_chord = ns["geodesic_chord"]

logged = pd.read_parquet(config.RESULTS / "stageb_structure.parquet")
row = logged[logged.layer == config.WURGAFT_LAYER].iloc[0]
target = float(row.geo_chord_mean)
print(f"logged geo_chord_mean at L{config.WURGAFT_LAYER}: {target:.10f}  "
      f"(§14a prints it as {target:.3f})")
print(f"logged id_mle {float(row.id_mle):.6f} | id_twonn {float(row.id_twonn):.6f} | "
      f"d_hat {int(row.d_hat)}")

df = pd.read_csv(config.DATA / "stageb" / "prompts.csv")
A = np.load(config.ACTS / "npy" / "stageb.npy")
X = np.ascontiguousarray(A[df.in_primary.to_numpy()][:, config.WURGAFT_LAYER, :])
Xp, _ = G.pca_project(X, config.PCA_DIM)
ids = G.intrinsic_dimension(Xp)
d_hat = G.d_hat_from(ids, "id_mle")
print(f"\nlocal  id_mle {ids['id_mle']:.6f} | id_twonn {ids['id_twonn']:.6f} | d_hat {d_hat}")

out = {"target": target, "logged_id_mle": float(row.id_mle),
       "local_id_mle": float(ids["id_mle"]), "d_hat_match": int(d_hat) == int(row.d_hat)}
TOL = 0.002
ok_any = False
for solver in ("auto", "dense"):
    Y, D_geo, info = G.isomap_embed(Xp, max(2, d_hat), 12, eigen_solver=solver)
    val = float(np.mean(geodesic_chord(D_geo, Y)))
    delta = abs(val - target)
    ok = delta <= TOL
    ok_any |= ok and solver == "auto"
    print(f"  solver={solver:<6} geo_chord_mean {val:.10f}  |delta| {delta:.2e}  "
          f"{'PASS' if ok else 'FAIL'} (tol {TOL})")
    out[f"local_{solver}"] = val
    out[f"delta_{solver}"] = delta
    out[f"pass_{solver}"] = bool(ok)

# D_geo is deterministic shortest-paths -- §18.7 prefers it as the criterion.
print(f"\n  D_geo-derived check (solver-independent): mean geodesic "
      f"{float(np.mean(D_geo[np.triu_indices_from(D_geo,1)])):.6f}")
out["verdict"] = ("PASS — local platform reproduces the logged value within tolerance"
                  if ok_any else
                  "FAIL — §18.4's local numbers must be re-run on fedora")
print(f"\n{out['verdict']}")
json.dump(out, open(config.RESULTS / "platform_acceptance.json", "w"), indent=2)
