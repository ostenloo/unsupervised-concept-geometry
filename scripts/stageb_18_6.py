"""§18.6 — §3a resampling stability and the §3b k sweep remainder.

§3a: refit the pipeline on 80% resamples (WITHOUT replacement, per N3) and report
the distribution of |cos(d_resample, d_full)|. Absolute, because Isomap component
sign is arbitrary (N5). This tests stability under RESAMPLING; the pipeline is
deterministic given data once eigen_solver is pinned, so stability under
algorithmic randomness is not what is being measured and is not claimed.

§3b: ID, cos(d, v_ref), Pearson r and held-out AUROC at each k in K_SWEEP.
§18.4g already covered the curvature statistics across k.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import config
from src import geometry as G
from scripts.stageb_18_4ac import fit_coordinate, build_direction, difference_in_means

df = pd.read_csv(config.DATA / "stageb" / "prompts.csv")
A = np.load(config.ACTS / "npy" / "stageb.npy")
src = df.source.to_numpy()
m_harmful, m_harmless = src == "harmful", src == "harmless"
m_400 = m_harmful | m_harmless
m_xstest = (src == "borderline") | (src == "xstest_contrast")
lab = (src[m_xstest] == "xstest_contrast").astype(int)
layer = config.WURGAFT_LAYER
X = np.ascontiguousarray(A[:, layer, :])
Xfit, Xeval = X[m_400], X[m_xstest]
hf = m_harmful[m_400]
v200 = difference_in_means(X[m_harmful], X[m_harmless])

def oriented(fit, Xf):
    d, _ = build_direction(fit, Xf)
    return -d if (Xf @ d)[hf[:len(Xf)]].mean() < (Xf @ d)[~hf[:len(Xf)]].mean() else d

d_full = oriented(fit_coordinate(Xfit), Xfit)

print("=== §3a resampling stability (80% without replacement, 20 refits) ===")
rng = np.random.default_rng(config.SEED)
cs = []
for _ in range(20):
    idx = rng.choice(Xfit.shape[0], size=int(0.8 * Xfit.shape[0]), replace=False)
    Xs = Xfit[idx]
    f = fit_coordinate(Xs)
    ds, _ = build_direction(f, Xs)
    cs.append(abs(float(ds @ d_full)))
cs = np.array(cs)
print(f"  |cos(d_resample, d_full)|: mean {cs.mean():.4f} min {cs.min():.4f} "
      f"p5 {np.percentile(cs,5):.4f} max {cs.max():.4f}")

print("\n=== §3b k sweep at L28 (fit on the clean 400) ===")
print(f"{'k':>4} {'id_mle':>8} {'d_hat':>6} {'cos(d,v_ref)':>13} {'Pearson r':>10} {'AUROC':>8} {'conn':>6}")
out = {"stability_abs_cos": cs.tolist(), "k_sweep": {}}
for k in config.K_SWEEP:
    f = fit_coordinate(Xfit, k=k)
    d = oriented(f, Xfit)
    proj = f["Xp"] @ (f["pca"].components_ @ v200)
    r = float(abs(pearsonr(proj, f["coord"]).statistic))
    auc = float(roc_auc_score(lab, Xeval @ d))
    cos = float(d @ v200)
    print(f"{k:>4} {f['id_mle']:8.2f} {f['d_hat']:6d} {cos:13.3f} {r:10.3f} {auc:8.4f} "
          f"{str(f['connected']):>6}")
    out["k_sweep"][int(k)] = dict(id_mle=float(f["id_mle"]), d_hat=int(f["d_hat"]),
                                  cos=cos, pearson_r=r, auroc=auc,
                                  connected=bool(f["connected"]))
(config.RESULTS / "stageb_18_6.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {config.RESULTS/'stageb_18_6.json'}")
