# §18 — Amendment: SPEC2 reconciled against the record

**Status:** written Wed 9 Sep 2026. Supersedes SPEC2 (`stage_b_spec_v1.md`) wherever the two conflict. SPEC2 was drafted without repo access; §18.1 lists every place it was wrong. This file is the §0 pre-registration artifact. Commit it before any script is written.

---

## §18.1 Corrections to SPEC2

**C1 — 0.946 is a Pearson correlation, not a cosine.** `stageb_structure.py:124` computes `|corrcoef(Xp @ v_sub, coord)|` — a correlation across points between the projection onto `v_ref` and the Isomap coordinate. SPEC2's §1a, §1b, and §2a arm 4b all treated it as an alignment/cosine. Consequences: §1b must report both metrics per row; §2a arm 4b's constraint is the *measured* cos(`d`, `v_ref`) from §18.4a, not 0.946; the writeup must say "Pearson correlation across points" wherever it currently says correlation-with-`v_ref`.

**C2 — SPEC2's §1a regression is vacuous as specified.** With n=600 and p=4096, unregularized OLS interpolates and R²=1 carries no information. Corrected procedure in §18.4a.

**C3 — k sweep.** SPEC2's {5,10,15,20,30} was invented and omits the k=12 used in the logged run. `config.py` pins `K_SWEEP = (8, 12, 16, 24)`. Use the pre-registered sweep. Tails may be *added* as marked extensions (4, 32); the existing values must not be renumbered.

**C4 — §1e is not a robustness comparison.** §14a records the spline estimator producing cosines 0.00–0.05 with NaN-on-ties. That is a broken estimator, not a garden-fork. "The two estimators agree" is withdrawn as a framing. Corrected claim in §18.4e.

**C5 — §16a stands.** SPEC2's §3d (whitened metric) was written without knowledge of §16a, which already concluded whitening is not load-bearing and that Stage B need not be run under both. Cite §16a; do not re-run. §3c likewise reduces — see §18.6.

**C6 — dtype.** `config.py:14` pins `GEOMETRY_DTYPE = "float32"` under acceptance test 6 (bf16 ties break neighbour methods). The tensor is ~315 MB, not ~160 MB, and a bf16 recapture would not be a faithful re-derivation of the logged results. Any recapture must be float32.

**C7 — SPEC2 mis-scoped the compute risk.** SPEC2 treated §2a as the at-risk item and §1 as safe. In fact §1a/§1b/§1e/§1f all require the cached activations, so §1 is compute-gated too — and §1 is the part that carries the proposal-only fallback. SPEC2's fallback rule ("drop to §1 only") is void as written. Replaced by §18.2.

---

## §18.2 Compute gates — split, not unified

The gate is **not** "fedora reachable for the duration." It is two separate gates, and conflating them is what produced C7.

**Gate A1 — one-time reachability (≈5 minutes of uptime).**
If fedora comes up even briefly: verify `acts/` holds the Stage B tensor, convert to `.npy` on fedora (float32, no dtype change), `scp` ~315 MB to this laptop, verify by SHA-256 against the source. All of §1 is numpy/sklearn/scipy — no torch, no GPU, no MPS. Once transferred, **§1 is permanently decoupled from box availability** and runs locally regardless of what happens to fedora afterwards.

Do the transfer first, before any analysis, even if the box looks stable. It converts a standing dependency into a one-time one.

**Gate A2 — sustained ssh + idle GPU (hours).**
Required only for §2. Needs tmux/nohup for a job outliving the session.

**Branch table:**

| A1 | A2 | Outcome |
|---|---|---|
| ✓ | ✓ | §1 local, §2 on fedora. Full spec. |
| ✓ | ✗ | §1 in full locally. §2 becomes a proposal section with the design pre-registered but unrun. Abstract stays correlational; the causal gap is stated as future work with the arm structure specified. |
| ✗ | — | Nothing runs. Deliverable is a writeup from the existing logged record only, with §1 and §2 both stated as designed-not-run. |

Record which branch fired, with the timestamp, in §18.8.

---

## §18.3 Deadline branches

SPEC.md:5 records the base deadline as Fri 4 Sep with possible extension to Fri 11 Sep. SPEC.md:404 records the venue as a MATS application with a required LLM-assistance disclosure. Today is Wed 9 Sep, so the base deadline has passed. Last commit `c725bdb` does not disambiguate submitted-vs-stalled.

**B1 — extension to Sept 11 granted.** ~2 days. SPEC2's §5 schedule stands with the Sept 10 18:00 freeze. Triage as written.

**B2 — already submitted on/around Sept 5.** The submitted artifact is fixed. New results become an addendum or a separate writeup; **do not retro-edit claims that were submitted**, and do not present post-submission results as if they were in the application. The freeze is void, and the correct move inverts: run §2a properly (larger n, judge included) rather than fast.

**B3 — missed, no extension.** Deliverable is a standalone writeup or next-cycle portfolio piece. Freeze void. Triage inverts further: the P2 items SPEC2 deferred come back in — Stage A diagnosis tests (non-monotone series with a lexical control), full layer sweep, extension to a second concept family.

Under B2 and B3, SPEC2's hard-freeze rule and its n=100 eval sizes are both artifacts of a deadline that no longer applies and should be re-derived rather than inherited.

**Only Austin can resolve this.** It is the second of two blocking unknowns and it changes what §3 is *for*.

---

## §18.4 Revised §1 (P0; gated on A1 only)

### §18.4a — Construct `d` and audit its linearity

The pipeline is PCA(64) → Isomap; no ambient `d` exists in the repo.

1. Regress the leading Isomap coordinate on the **64-d PCA scores** (n=600, p=64 — well-posed). Report in-sample and 5-fold cross-validated R².
2. Lift: `d = V₆₄ @ β`, normalized. This is a legitimate ambient direction confined to the PCA(64) subspace.
3. Separately, cross-validated **ridge** on the full 4096-d centered activations. Report CV R² only; in-sample R² is uninformative at this n (C2). The gap between (1) and (3) measures what the PCA truncation discarded.
4. Report cos(`d`, `v_ref`). **This number, not 0.946, is the constraint for §18.5 arm 4b.**

If CV R² in (1) is high, that is a result — the Isomap coordinate is a linear readout — and belongs in the findings, not the appendix.

### §18.4b — Baseline panel, both metrics per row

| Method | Pearson r (coord vs. proj on `v_ref`) | cos(·, `v_ref`) | XSTest AUROC |
|---|---|---|---|
| Isomap coord 1 (logged) | 0.946 | via §18.4a | 0.995 (transductive — see §18.4c) |
| PC1 of fit bank | ✓ | ✓ | ✓ |
| k-means (k=2) centroid difference | ✓ | ✓ | ✓ |
| Random directions, n=100 | distribution | distribution | distribution |

Coordinate-valued methods give r natively; direction-valued methods give cosine natively. Report both for every row, name the metric, never present one as the other.

**Branch rule (unchanged from SPEC2).** If PC1 reaches ≥0.90 Pearson r *and* AUROC within 0.01, the manifold machinery is not load-bearing. Reframe to "the refusal axis is recoverable by any unsupervised linear method including trivial ones, which is positive evidence for completeness of the linear account." Write that abstract sentence now.

### §18.4c — XSTest refit (already diagnosed)

§14 establishes the 600-prompt fit bank includes the 200 XSTest safe-borderline prompts; the 200 contrast prompts were held out. So 0.995 is transductive on one side.

Refit on the 400 harmful+harmless only; evaluate on all 400 XSTest as genuine holdout. **That number is the headline.** Report 0.995 alongside, labelled transductive; the gap is itself reportable.

### §18.4d — Power check on synthetic curvature

As SPEC2 §1d, with two changes: run at k=12 (the logged value) and report MDR per-k across `K_SWEEP`, since the detection floor is k-conditional. Noise σ estimated from float32 residuals about a 1-D principal curve along `d`.

**Branch rule.** If MDR > 1.02, the claim is "curvature below our detection threshold of X," not "the manifold is straight." Committed now.

### §18.4e — Estimator failure diagnosis (replaces SPEC2 §1e)

Report the spline estimator's failure with evidence attached: cosine range 0.00–0.05, NaN-on-ties, and the smoothing configuration used. State the mechanism — smoothing flattened the derivative into numerical noise — as **demonstrated for this configuration**, and whether it generalizes to the estimator as such as an **untested hypothesis**. Same discipline as the Stage A diagnoses (§18.7).

This is stronger than SPEC2's "agreement" framing: a documented dead estimator with reproducible failure evidence is unambiguous.

### §18.4f — ID uncertainty

Bootstrap both banks (200 resamples), recompute TwoNN, report CIs on 5.21, on 5.89, and on the difference. If the CIs overlap, the §14d control-bank-comparison framing does not survive and must not be written.

### §18.4g — Ratio-only k slice

Geodesic/chord at `K_SWEEP = (8, 12, 16, 24)`. Gates §18.4d.

---

## §18.5 Revised §2 (P0; gated on A2)

### §18.5a — New hook, with acceptance test *before* any eval

§17a records the existing hook as not position-safe under `generate()` — which is exactly the code path §2 requires. The 20-prompt timing measurement (SPEC2 §4c) must run on the new ablation path, not the single-forward path, and is therefore also the first exercise of new code.

**Acceptance test, must pass before any eval run:**
1. **Null-op:** hook installed with zero-magnitude projection reproduces baseline generation token-for-token.
2. **Completeness:** during a full generation, assert `max |û·x| < ε` over *every* layer and *every* position in the cache, including generated positions. This is the §17a failure mode; the assertion is the gate.
3. **KV-cache consistency:** ablated generation with and without KV cache must match on ≥5 prompts. Cached keys/values computed pre-ablation would yield silent partial ablation. If they diverge, disable the cache for the eval and absorb the slowdown.

Log the test output in the results file. An unasserted ablation is not evidence.

### §18.5b — Arms

Directional ablation `x ← x − ûûᵀx` at every residual-stream write, every layer, every position, throughout generation.

| Arm | Direction | Purpose |
|---|---|---|
| 1 | `v_ref` | replication anchor |
| 2 | `d` | does the unsupervised direction cause refusal |
| 3 | `d_⊥ = d − proj_{v_ref}(d)` | sharp test of independent causal content |
| 4 | random unit, n=5 seeds | null |
| 4b | random, constrained to cos(·,`v_ref`) = **measured value from §18.4a** | "is `d` more than a vector near `v_ref`" |
| 5 | none | baseline |

**Eval sets.** Harmful: JailbreakBench Behaviors, **string-level deduplicated against the 400-prompt fit bank** (JBB shares lineage with AdvBench/HarmBench; report n after dedup). Harmless: Alpaca. n per SPEC2 under B1; re-derive under B2/B3.

**Metrics.** (a) Arditi refusal-substring rate — primary. (b) CE loss on harmless completions — capability gate, not a footnote. (c) Manual stratified spot-check of ~40 completions, documented — covers the incoherence failure mode, where a lobotomized model scores as "not refusing." (d) LLM judge as secondary if an API key exists; never on the critical path, never a local model sharing the GPU.

**Branch rules (committed now):**
- Arm 2 drop ≥ 0.8 × arm 1 drop, arm 2 CE within noise of arm 5 → `d` is causally equivalent to `v_ref`.
- Arm 3 drop ≤ arms 4/4b + CI → `d`'s causal power is fully explained by `v_ref` overlap. **This is the expected result and it supports the completeness conclusion.** Pre-committed so it cannot be spun later.
- Arm 3 significantly above both nulls → single-direction account is incomplete; that becomes the headline.

### §18.5c — Steering (absorbs §4f)

Addition arm on harmless prompts, `û ∈ {d, v_ref}`, **matched on projection magnitude, not raw α**. Consistency = dose-response curves overlap within bootstrap CI at every level. Divergence at high magnitude only → "consistent in the linear regime, diverging under strong steering," which is a finding, not a failure.

---

## §18.6 §3 reduction

- **§3d (whitening): cite §16a, do not run.**
- **§3c (layers): mostly done.** SPEC.md records capture at all 32 layers in float32, and §14a already gives ratio 1.010–1.020 at layers 8/22/28/31 — that establishes layer-stability *for the ratio*. The Wurgaft-inheritance vulnerability is about the headline claims, so what remains is running the §18.4b metrics (r, cosine, AUROC) at 8/22/31 alongside 28. **CPU only, gated on A1, not A2.**
- **§3a (resampling stability), §3b (full k sweep):** P1, CPU, gated on A1.

Note the consequence: under branch A1✓/A2✗, §1 *and* the entire useful remainder of §3 still run in full. Only §2 is lost.

---

## §18.7 Reproducibility pins

- Model SHA: already pinned, `config.py:11` — reference, do not re-pin.
- Stage B dataset commit: already recorded.
- XSTest, Alpaca: snapshots unrecorded at run time. **Hash the on-disk files (SHA-256) and pin by content hash.** Record verbatim: "snapshot unrecorded at run time; re-pinned by content hash on 9 Sep 2026." Do not imply retroactive pinning.
- Environment: fedora carries torch 2.11.0+cu130 / transformers 5.12.1 system-wide (requirements.txt:1-2, config.py:23). Capture `pip freeze` from fedora during the A1 window — it will not be recoverable later.
- **Stage A diagnoses remain untested hypotheses.** The alkane monotone-ordinal account and the IUPAC orthographic confound are stated predictions, not established mechanisms, and §18.4d does not test either. The writeup must say so explicitly; without that sentence, "we diagnosed two failure modes" reads as a finding. Under B3, testing them moves back in scope.

---

## §18.8 Open — requires Austin

1. **Tailscale up / fedora powered on.** Determines which row of §18.2 fires. Record the outcome and timestamp here.
2. **Deadline branch (B1/B2/B3).** Determines the freeze, the eval sizes, and what §3 is for.
3. **Venue/length**, if B1. Not recoverable from the repo.
4. **Scope of ssh authorization**, if A2 is wanted — see chat.

## §18.9 Claim-dependency graph

Build **before** §1 resolves, with branches open — that is the artifact's purpose. Graphviz/mermaid source in-repo, rendered for the writeup, annotated post-hoc with which branch fired.

Nodes: §18.4b → novelty claim; §18.4c → AUROC claim; §18.4d → straightness claim; §18.4f → §14d ID contrast; §18.5b arm 3 → the conclusion itself.