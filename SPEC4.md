# §18 — Amendment: SPEC2 reconciled against the record (v2)

**Status:** v2, Wed 9 Sep 2026. Supersedes v1 and SPEC2. v1 contained seven errors found by repo audit; they are listed and fixed below rather than silently corrected. Commit this before any script is written.

---

## §18.0 Audit trail — v1 errors, fixed

| ID | Error in v1 | Fix |
|---|---|---|
| E1 | Misquoted SPEC.md:5 — "base deadline Fri 4 Sep." That is the **writing** date. | Base deadline **Sun 6 Sep**, extension to Fri 11 Sep. §18.3 and B2 corrected. |
| E2 | Bootstrapped TwoNN; the 5.21/5.89 contrast is **MLE-vs-MLE** (SPEC.md:1151-1156). | §18.4f bootstraps `id_mle` — the composite mean of Levina–Bickel at k ∈ {10,20}, `src/geometry.py:130-136`. |
| E3 | Transfer manifest incomplete, size ~5× low. | Full manifest in §18.2, ~1.5 GB. |
| E4 | §18.4c decontaminated the coordinate but **not `v_ref`**. | §18.4c′ — `v_ref` recomputed from 200v200. |
| E5 | §18.5 silent on which `d` it ablates. | §18.4a′ resolves by measurement before the fork can open. |
| E6 | "Every residual-stream write" ≠ block-output hooks; test 2 would pass while leaking. | §18.5a rewritten — sub-layer hooks, sub-layer assertion. |
| E7 | Synthetic bank skipped PCA(64), inflating the noise floor. | §18.4d rewritten — synthetic generated in the 64-d space with matched spectrum. |

Verified correct in v1 and unchanged: C3, C6, §18.4c provenance, §18.6 layer claim, §18.7 env pins, §18.2's split-gate structure, §18.4a, §18.4e, §18.5a's intent.

---

## §18.1 Corrections to SPEC2

Unchanged from v1: **C1** (0.946 is Pearson, not cosine), **C2** (n=600/p=4096 OLS is vacuous), **C3** (use `K_SWEEP = (8,12,16,24)`, `config.py:29`), **C4** (§1e is a broken-estimator diagnosis, not an agreement claim), **C5** (§16a stands; no whitening re-run), **C6** (float32, `config.py:14`; bf16 recapture would not be faithful), **C7** (§1 is compute-gated too; SPEC2's fallback rule void).

**C8 (new).** SPEC2 and v1 both wrote "TwoNN" for the ID contrast. The logged contrast is MLE. Every prose reference must say which estimator.

---

## §18.2 Compute gates — split (structure unchanged, manifest corrected)

**Gate A1 — one-time transfer.** Not "≈5 minutes"; re-costed at ~1.5 GB.

Manifest, in priority order:

| File | Size | Needed by |
|---|---|---|
| `acts/stageb.pt` (800 prompts, not 600) | 419 MB | §18.4a–e, §18.6 |
| `acts/bank.pt` (2000-prompt instruction bank) | 1.05 GB | §18.4f — control-bank ID |
| `data/stageb/xstest_raw.csv` | small | §18.7 hashes, §18.4c′ |
| `data/stageb/harmless_train.json` | small | §18.7 hashes, §18.5b harmless set |
| `pip freeze` from fedora | — | §18.7; not recoverable later |

Convert `.pt` → `.npy` **on fedora** (float32, no dtype change), verify by SHA-256 against source after transfer.

`acts/bank_bare.pt` (1.05 GB) is **skipped by decision, not oversight** — but the decision has a condition. If the writeup claims chat-template formatting is not driving the geometry, `bank_bare` is exactly that control and must come across. Decide before the A1 window opens; re-opening it later may not be possible.

**Gate A2 — sustained ssh + idle GPU.** §18.5 only.

| A1 | A2 | Outcome |
|---|---|---|
| ✓ | ✓ | Full spec. |
| ✓ | ✗ | §18.4 and §18.6 in full, locally. §18.5 pre-registered but unrun; abstract stays correlational, causal gap stated with arm structure specified. |
| ✗ | — | Writeup from logged record only. |

---

## §18.3 Deadline branches (E1 corrected)

SPEC.md:5: written Fri 4 Sep 2026; **application deadline Sun 6 Sep**, possible extension to Fri 11 Sep. SPEC.md:404: venue is a MATS application with a required LLM-assistance disclosure. Today is Wed 9 Sep — base deadline passed three days ago. Commit `c725bdb` does not disambiguate submitted-vs-stalled.

- **B1 — extension to Fri 11 granted.** ~2 days. SPEC2 §5 schedule, Sept 10 18:00 freeze.
- **B2 — submitted on/around Sun 6 Sep.** Submitted artifact is fixed. New results are an addendum; **do not retro-edit submitted claims** or present post-submission results as having been in the application. Freeze void; run §18.5 properly rather than fast.
- **B3 — missed.** Standalone writeup / next-cycle piece. Freeze void, P2 items return: Stage A diagnosis tests, full layer sweep, second concept family.

Under B2/B3 the freeze and the n=100 eval sizes are artifacts of a dead constraint — re-derive, don't inherit.

---

## §18.4 Revised §1 (P0; gated on A1 only)

### §18.4a′ — Close the `d` fork by measurement, then build `d`

E5 is closable with a number rather than a choice. **Do this first.**

1. Fit the pipeline on the logged 600 in-primary prompts → coordinate → `d₆₀₀`.
2. Fit on the decontaminated 400 (§18.4c′) → `d₄₀₀`.
3. Report **cos(`d₄₀₀`, `d₆₀₀`)**. If ≥0.99, the fork is empty; record that and carry `d₄₀₀` with a one-line note. If <0.99, both are carried through every table and §18.5 ablates `d₄₀₀` as primary with `d₆₀₀` as a robustness arm.

**Primary is `d₄₀₀`** under either outcome. The causal claim should be about the direction recovered under clean conditions.

Construction (C2 fix, unchanged): regress the leading Isomap coordinate on the **64-d PCA scores** (n, p=64 — well-posed); report in-sample and 5-fold CV R²; lift `d = V₆₄ @ β`, normalized. Separately report cross-validated **ridge** R² on the full 4096-d centered activations — CV only; in-sample is uninformative at this n. The gap measures what PCA(64) discarded.

Report cos(`d`, `v_ref`). **That measured value, not 0.946, is arm 4b's constraint.**

### §18.4b — Baseline panel, both metrics per row

| Method | Pearson r (coord vs. proj on `v_ref`) | cos(·, `v_ref`) | XSTest AUROC |
|---|---|---|---|
| Isomap coord 1 | ✓ | ✓ | ✓ |
| **PC1 of the 600-prompt in-primary set** (S3 — the PCA the pipeline already computes; recomputed on the 400 for the decontaminated condition, **not** the 2000-prompt bank) | ✓ | ✓ | ✓ |
| k-means (k=2) centroid difference | ✓ | ✓ | ✓ |
| Random directions, n=100 | distribution | distribution | distribution |

Every baseline is computed on the same bank as the `d` it is compared against. Name the metric in every cell; never present one as the other.

**Branch rule.** PC1 ≥0.90 Pearson r *and* AUROC within 0.01 → manifold machinery not load-bearing. Reframe to "recoverable by any unsupervised linear method including trivial ones," which is positive evidence for completeness. Write that abstract sentence now.

### §18.4c′ — Decontaminate the coordinate **and** `v_ref` (E4)

`prompts.csv` is 800 rows: 200 harmful / 200 harmless / 200 XSTest borderline (all `in_primary=True`) + 200 XSTest contrast (`in_primary=False`).

`stageb_structure.py:30-42`: `difference_in_means(H = acts[harmful], B = acts[~harmful])`. The complement is 400 = 200 harmless + 200 XSTest borderline. **`v_ref`'s negative pole is half XSTest.** Refitting Isomap on 400 leaves the reference direction fit on the evaluation set, and the r column, the cosine column, `d_⊥`, and arm 4b's constraint all inherit it.

**Resolution: recompute.**
1. `v_ref_200` = difference-in-means, 200 harmful vs 200 harmless only.
2. Report **cos(`v_ref_200`, `v_ref_400`)** before anything else. If ≥0.99 the contamination is immaterial and you say so with a number; if not, `v_ref_200` is primary throughout — including §18.5 arm 1 and the arm 4b constraint — and `v_ref_400` appears as a labelled robustness row.
3. The logged 0.946 belongs to `d₆₀₀` × `v_ref_400`. It stays in the record labelled as such. The headline r is recomputed under the clean pairing and **will differ**.

Stating the asymmetry instead of fixing it was the other option; it is rejected because it would have to be restated in every table cell indefinitely, and the recompute is a difference of means over cached activations.

AUROC: refit on the 400 harmful+harmless, evaluate on all 400 XSTest as genuine holdout. **That is the headline.** 0.995 reported alongside, labelled transductive; the gap is reportable.

### §18.4d′ — Power check, generated in the PCA(64) space (E7)

v1's synthetic bank skipped PCA(64). Isotropic ambient noise loses ~98% of its variance through that projection, so a synthetic null that bypasses it sits far above the real noise floor and MDR comes out badly pessimistic.

Two problems, not one — the second is not fixed by simply adding the PCA step:

1. **The pipeline must be run whole**, PCA(64) included, with PCA **fit on the synthetic bank** (not the real basis).
2. **Real residual-stream noise is strongly anisotropic.** Isotropic ambient noise pushed through PCA(64) is still not matched to the real post-PCA residual structure. Generate instead **directly in the 64-d space**, with per-component noise matched to the eigenvalue spectrum of the real bank's residuals about a 1-D principal curve along `d`. Report the spectrum used.

Otherwise as SPEC2 §1d: circular arcs, true ratio `φ/sin φ`, sweep {1.000, 1.005, 1.01, 1.02, 1.05, 1.10, 1.25, 1.50}, 50 replicates, n matched. Null at ratio 1.000 gives the threshold (95th pct); MDR = smallest true ratio detected at ≥80% power. Run at k=12 and report MDR per-k across `K_SWEEP` — the floor is k-conditional.

**Branch rule.** MDR > 1.02 → the claim is "curvature below our detection threshold of X," not "the manifold is straight." Committed now.

### §18.4e — Estimator failure diagnosis (unchanged, verified correct)

Report the spline estimator's failure with evidence: cosine range 0.00–0.05, NaN-on-ties, smoothing configuration. Mechanism — smoothing flattened the derivative into numerical noise — stated as **demonstrated for this configuration**; generalization to the estimator as such is an **untested hypothesis**. Same discipline as the Stage A diagnoses.

### §18.4f′ — ID uncertainty, correct estimator (E2)

SPEC.md:1151-1156 columns are `ID (MLE) | ID (TwoNN) | bank ID (MLE)`. At L28: **5.21 = MLE (refusal), 6.89 = TwoNN (refusal), 5.89 = bank MLE**. The 5.21-vs-5.89 contrast is MLE-vs-MLE.

Bootstrap `id_mle`, reproducing the composite in `src/geometry.py:130-136` — mean of Levina–Bickel at k ∈ {10,20}, not a single k. 200 resamples of both banks. Report CIs on 5.21, on 5.89, and on the difference.

**Additional, not in the audit:** 5.21 (MLE) vs 6.89 (TwoNN) is a ~30% disagreement between two estimators on identical data. That spread bounds how much weight "the refusal cloud is ~5-dimensional" can carry, and it should be reported as a stated limit on the ID claim rather than left to a reader to notice. Bootstrap TwoNN too, for that comparison only — but the §14d contrast remains MLE-vs-MLE.

**Branch rule.** If the MLE CIs overlap, the §14d control-bank-comparison framing does not survive and must not be written.

### §18.4g — Ratio-only k slice

Geodesic/chord at `K_SWEEP = (8,12,16,24)`. Gates §18.4d′.

---

## §18.5 Revised §2 (P0; gated on A2)

### §18.5a′ — Hook granularity and a test that certifies the right thing (E6)

`config.py:25` pins `HOOK_MODULE_TEMPLATE = "model.layers.{layer}"` — **block outputs**. Arditi ablates at every residual-stream write: embedding output, each attention output, each MLP output. With block-output hooks, attention can write a û-component that the MLP reads within the same block before the next ablation fires.

This is not a cosmetic deviation. **The leak biases toward a false null on arms 2 and 3** — the arms whose nulls are pre-committed as supporting the completeness conclusion. A leaky hook manufactures exactly the result being pre-registered as expected.

And v1's test 2 would have passed while leaking: asserting `max|û·x| < ε` at block boundaries checks precisely where the hooks fire.

**Resolution — extend the hooks. This is not optional.** Hook `model.embed_tokens`, `model.layers.{i}.self_attn`, and `model.layers.{i}.mlp` outputs.

If they cannot be extended in the time available: **arm 3's null becomes uninterpretable and must be reported as such.** You may not pre-commit "arm 3 null supports completeness" while running an instrument that produces nulls artifactually. One or the other.

**Acceptance test, before any eval:**
1. **Null-op:** zero-magnitude projection reproduces baseline generation token-for-token.
2. **Completeness, at sub-layer outputs:** during full generation, assert `max|û·x| < ε` at **every hooked sub-layer output** and every position including generated ones. Block-boundary assertion is insufficient and is what E6 flags.
3. **KV-cache consistency:** ablated generation with and without KV cache must match on ≥5 prompts. Divergence means keys/values were computed pre-ablation — silent partial ablation. If they diverge, disable the cache and absorb the slowdown.

Log test output in the results file. An unasserted ablation is not evidence.

**Timing (SPEC2 §4c):** the 20-prompt measurement runs on this new path, not the single-forward path, and is therefore the first exercise of new code. The position-gating assertion lives there.

### §18.5b — Arms

Directional ablation `x ← x − ûûᵀx` at every hooked write, every layer, every position, throughout generation.

| Arm | Direction | Purpose |
|---|---|---|
| 1 | `v_ref_200` (§18.4c′) | replication anchor |
| 2 | `d₄₀₀` (§18.4a′) | does the unsupervised direction cause refusal |
| 3 | `d_⊥ = d₄₀₀ − proj_{v_ref_200}(d₄₀₀)` | independent causal content |
| 4 | random unit, n=5 seeds | null |
| 4b | random, constrained to **measured** cos(`d₄₀₀`, `v_ref_200`) | "is `d` more than a vector near `v_ref`" |
| 5 | none | baseline |

Robustness arms (`d₆₀₀`, `v_ref_400`) only if §18.4a′/§18.4c′ return cosines <0.99.

**Eval sets.** Harmful: JailbreakBench Behaviors, string-deduplicated against the 400-prompt fit bank; report n after dedup. Harmless (S2): **held-out rows of `harmless_train.json`** — not a fresh Alpaca pull, which would risk overlapping both the 200 in the fit set and the 2000 in `bank.pt`. Dedup against both, report n, same reporting standard as the harmful set.

**Metrics.** (a) Arditi refusal-substring rate — primary. (b) CE loss on harmless completions — capability gate. (c) Manual stratified spot-check of ~40 completions, documented — covers the incoherence failure mode where a lobotomized model scores as "not refusing." (d) LLM judge secondary if an API key exists; never on the critical path, never a local model sharing the GPU.

**Branch rules, committed:**
- Arm 2 drop ≥ 0.8 × arm 1 drop, arm 2 CE within noise of arm 5 → `d` causally equivalent to `v_ref`.
- Arm 3 ≤ arms 4/4b + CI → `d`'s causal power fully explained by `v_ref` overlap; supports completeness. **Valid only if §18.5a′ test 2 passed at sub-layer granularity.**
- Arm 3 significantly above both nulls → single-direction account incomplete; that becomes the headline.

### §18.5c — Steering

Addition arm on harmless prompts, `û ∈ {d₄₀₀, v_ref_200}`, **matched on projection magnitude, not raw α**. Consistency = dose-response curves overlap within bootstrap CI at every level. Divergence at high magnitude only → "consistent in the linear regime, diverging under strong steering" — a finding, not a failure.

---

## §18.6 §3 reduction (verified correct, unchanged)

- **§3d (whitening):** cite §16a, do not run.
- **§3c (layers):** `src/model.py:97` confirms capture is `[n, 32, 4096]` float32, so this is CPU-only on the transferred file. §14a already gives ratio 1.010–1.020 at 8/22/28/31 — layer-stability **for the ratio**. What remains: §18.4b metrics (r, cosine, AUROC) at 8/22/31 alongside 28. Gated on A1, not A2.
- **§3a, §3b:** P1, CPU, gated on A1.

Under A1✓/A2✗, §18.4 and §18.6 both run in full. Only §18.5 is lost.

---

## §18.7 Reproducibility pins

- Model SHA: `config.py:11` — reference, do not re-pin.
- Stage B dataset commit: already recorded.
- XSTest, Alpaca/`harmless_train`: unrecorded at run time. **Pin by SHA-256 content hash** of the transferred files. Record verbatim: "snapshot unrecorded at run time; re-pinned by content hash on 9 Sep 2026." No retroactive implication of pinning.
- fedora env: `pip freeze` during the A1 window.
- **Local env (S1):** this laptop has no venv and no numpy/scipy/sklearn/pandas/skdim. Create and pin one. Running on macOS arm64 when everything logged came from Linux x86 is a platform change with different BLAS.
  **Platform acceptance test:** before trusting any new local number, reproduce one logged one — geodesic/chord at L28, k=12, expected 1.020. **Tolerance: ±0.002.** Outside tolerance, §18.4 does not run locally; it runs on fedora during an A2 window or not at all.
- **Stage A diagnoses remain untested hypotheses.** The alkane monotone-ordinal account and the IUPAC orthographic confound are stated predictions, not established mechanisms, and §18.4d′ does not test either. The writeup must say so explicitly. Under B3, testing them returns to scope.

---

## §18.8 Open — requires Austin

1. **Tailscale up / fedora powered on.** Determines which row of §18.2 fires. Record outcome and timestamp here.
2. **Deadline branch (B1/B2/B3).**
3. **Venue/length**, if B1.
4. **ssh authorization scope**, if A2.
5. **`bank_bare.pt`** — transfer or not (§18.2). Depends on whether the template control is claimed.

---

## §18.9 Claim-dependency graph

Build **before** §18.4 resolves, branches open. Graphviz/mermaid in-repo, rendered for the writeup, annotated post-hoc with which branch fired.

Nodes:
- §18.4a′ CV R² low → C2 overclaiming fires; "recovers a direction" is not supportable (S4)
- §18.4b → novelty claim
- §18.4c′ → AUROC claim **and** every `v_ref`-referenced quantity
- §18.4d′ → straightness claim
- §18.4f′ → §14d ID contrast
- §18.5a′ test 2 → **gates** arm 3's interpretability
- §18.5b arm 3 → the conclusion