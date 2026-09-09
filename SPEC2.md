# Stage B — Remaining Work Spec (v1)

**Assumed deadline: Sept 11.** Today is Sept 9. Budget ≈ 1.5 days compute, 0.5 day writeup. Verify the deadline before committing to this schedule; if it's actually Sept 6+no extension, drop to §1 only and write up.

**Notation.** `v_ref` = Arditi supervised difference-in-means direction at L28. `d` = your unsupervised direction. `d_⊥` = `d − proj_{v_ref}(d)`, renormalized.

---

## §0. Pre-registration block — commit BEFORE running anything

Write the decision rules below into a file, `git commit`, record the hash. This is the artifact that makes the §14d estimator-swap disclosure credible rather than defensive. Nothing in §1–§3 gets run until this is committed.

Also pin now, in the same commit:
- model revision hash (`meta-llama/Llama-3.1-8B-Instruct` @ specific SHA, not `main`)
- dataset commits/versions for XSTest, AdvBench/JailbreakBench, Alpaca
- code commit, RNG seeds, exact estimator implementations (**both** the smoothing-spline and binned-centroid versions)
- environment (`pip freeze`, CUDA version)

---

## §1. P0 — Must run. Cheap. Each one can invalidate a headline claim.

### §1a. Linearization audit of `d` (do this first — everything downstream needs it)

Isomap returns a coordinate, not an ambient direction. Ablation and steering need `d ∈ R^4096`. Specify and report how you get it:

- OLS-regress leading Isomap coordinate on centered L28 activations → normalized coefficient vector = `d`.
- **Report R² of that linear fit.**

This is not bookkeeping. If R² is high (>0.9), that is *itself* evidence for the linear account and belongs in the results, not the appendix. If R² is low, "unsupervised recovery of a direction" is overclaiming — you recovered a coordinate that isn't well-represented as a direction, and the 0.946 correlation is comparing a nonlinear coordinate to a linear one.

**Cost:** minutes, cached activations.

### §1b. Simple-baseline panel (Nanda: "check simple baselines")

Compute, against the same refusal bank, with no labels:

| Method | corr with `v_ref` | XSTest AUROC |
|---|---|---|
| Isomap coord 1 → `d` (yours) | 0.946 | 0.995 |
| PC1 of refusal bank | ? | ? |
| k-means (k=2) centroid difference | ? | ? |
| Random direction (n=100, distribution) | ? | ? |

**Pre-registered decision rule.** If PC1 reaches ≥0.90 correlation *and* AUROC within 0.01 of yours, the manifold-learning machinery is not load-bearing. Do not bury this. Reframe: the contribution becomes *"the refusal axis is recoverable by any unsupervised linear method, including trivial ones — which is positive evidence that the linear account is complete,"* which is a defensible and honest claim, and arguably a cleaner one than "Isomap finds it." Decide now which abstract you write under each branch.

**Cost:** minutes.

### §1c. XSTest provenance audit

Determine whether XSTest prompts were in the bank the embedding was fit on. If yes, 0.995 is **transductive**, not held-out, and must be labeled as such.

If in-sample: refit on the refusal/control bank only, evaluate on XSTest as a genuine holdout, report the new number. If it drops materially, the new number is the headline.

**Cost:** minutes to audit; ~30 min if a refit is needed.

### §1d. Power check on synthetic known curvature — **this is the highest-value cheap item**

The geodesic/chord ≈ 1.01–1.02 result is a *null*. A null with unknown power is uninterpretable, and kNN geodesic estimation on noisy data inflates path length, so your null distribution almost certainly sits above 1.0 already.

Procedure:
1. Estimate ambient noise scale σ from the refusal bank: fit a 1-D principal curve along `d`, take residual RMS per dimension.
2. Generate synthetic banks: n points (matched to bank size) on a circular arc in a 2-plane in R^4096, arc length matched to observed spread along `d`, plus isotropic Gaussian noise at σ in all dims.
3. True ratio for a circular arc of half-angle φ is `φ / sin φ`. Sweep true ratio ∈ {1.000, 1.005, 1.01, 1.02, 1.05, 1.10, 1.25, 1.50}.
4. For each, run **the actual pipeline** (same kNN construction, same geodesic estimator, same k) × 50 replicates.
5. Null (true ratio = 1.000) gives the reference distribution. Threshold = 95th percentile of null.
6. **Minimum detectable ratio (MDR)** = smallest true ratio at which ≥80% of estimates exceed threshold.

**Pre-registered decision rule.** If MDR > 1.02, the observed value sits inside the noise floor and the claim is *"curvature is below our detection threshold of X"* — not *"the manifold is straight."* Commit to that wording now. If MDR ≤ 1.01, the straightness claim survives and gains a quantified power statement, which is strictly better than the current unquantified null.

**Ordering dependency:** the MDR is *k*-conditional — it holds only at the k used to build the graph. Run the ratio-only slice of §3b (sweep k, record geodesic/chord only) *before* §1d, then report MDR at the k you're using plus a note on whether the ratio is k-stable. If the ratio moves with k, the power statement has to be given per-k or the detection floor is meaningless.

**Cost:** pure numpy, <1 hour. No GPU.

### §1e. Dual-estimator report (closes the §14d pre-registration gap)

Run **both** the original pre-registered per-dimension smoothing-spline estimator and the binned-centroid estimator on the same cached data. Report side by side.

If they agree, the swap is defused — it was a robustness variant, not a garden-forking path. If they disagree, that is the finding, and disclosing it yourself is the only version of this that survives review.

**Cost:** minutes, cached data.

### §1f. Uncertainty on the ID estimates (promoted from P1)

Bootstrap the refusal and control banks (200 resamples), recompute TwoNN, report CIs on both and on the difference.

5.21 vs 5.89 is currently a difference with no error bars — it is not yet a claim, and it's one of the numbers §14d's dual-reporting decision rests on. If the CIs overlap, the control-bank comparison framing you'd planned to lead with does not survive, and you need to know that before writing it.

**Cost:** minutes, cached activations. No reason this sits behind §2.

### §1g. Ratio-only k slice (promoted from §3b)

Sweep k ∈ {5, 10, 15, 20, 30}, record geodesic/chord ratio only. Gates §1d (see above). The full §3b sweep (ID, correlation, AUROC at each k) stays at P1.

**Cost:** minutes.

---

## §2. P0 — Causal validation of `d` (the Oozeer gap). GPU. Biggest time sink.

Correlation + AUROC establish that `d` *predicts* refusal. Nothing yet shows it *causes* refusal, and nothing shows it does so independent of `v_ref`.

### §2a. Directional ablation, 6 arms

Ablation protocol, matching Arditi rather than a single-site edit: for unit direction `û`, apply `x ← x − ûûᵀx` to **every residual-stream write at every layer and every token position** throughout generation. A single-site L28 prompt-final-token edit produces an ambiguous null (the direction can be re-read downstream) and should not be used as the primary condition.

| Arm | Direction ablated | Purpose |
|---|---|---|
| 1 | `v_ref` | replication anchor (should reproduce Arditi) |
| 2 | `d` | does the unsupervised direction cause refusal |
| 3 | `d_⊥` | **the sharp test** — does `d` carry causal content beyond its `v_ref` overlap |
| 4 | random unit vectors, n=5 seeds | null |
| 4b | random vectors constrained to cos(·, `v_ref`) = 0.946 | *"is `d` more than a vector near `v_ref`"* null |
| 5 | none | baseline |

**Eval sets:** 100 harmful (JailbreakBench or AdvBench, held out from the fitting bank) + 100 harmless (Alpaca).

**Metrics:** (a) refusal-substring match rate, (b) LLM-judge safety score, (c) **CE loss on harmless completions** — the capability control. Without (c), an ablation that simply degrades the model trivially "reduces refusal." Report all three; (c) is a gate, not a footnote.

**Pre-registered decision rules:**
- Arm 2 refusal drop ≥ 0.8 × arm 1 drop, with arm 2 CE within noise of arm 5 → `d` is *causally equivalent* to `v_ref`. This upgrades "unsupervised recovery" from geometric to functional and is the strongest available version of your contribution.
- Arm 3 drop ≤ arm 4/4b drop + CI → `d`'s causal power is fully explained by `v_ref` overlap. **State now that this is the expected result and that it supports, rather than weakens, the "linear account is complete" conclusion.** Pre-committing removes the temptation to spin it later.
- Arm 3 drop significantly above both nulls → `d` carries independent causal content, the single-direction account is incomplete, and *that* becomes the headline. Write one sentence of this abstract now so the branch is real.

**Cost:** 6 arms × 200 prompts × ~64 tokens, 8B bf16 on the 5090. Batched, ~45–90 min. Start it early; this is the arm most likely to eat the schedule.

### §2b. Fold §4f (steering consistency) in as the sufficiency arm

Don't run §4f as a separate study — it's the addition half of the same experiment.

Add `x ← x + α·û` on harmless prompts, for `û ∈ {d, v_ref}`, sweeping α over {0.5, 1, 2, 4} × the mean harmful-minus-harmless projection magnitude for that direction.

**Match on projection magnitude, not raw α** — otherwise the two curves differ by an arbitrary scale factor and the comparison is meaningless.

**Pre-registered rule:** consistency = the two dose-response curves (refusal rate vs. matched projection magnitude) overlap within bootstrap CI at every α. Divergence at high α only → report as "consistent in the linear regime, diverging under strong steering," which is a real and interesting finding, not a failure.

**Cost:** +8 conditions × 100 prompts, ~30 min.

---

## §3. P1 — Run if §1–§2 finish by Sept 10 midday. All CPU on cached activations.

### §3a. Resampling stability
Bootstrap the refusal/control banks (20 resamples @ 80%), refit the full pipeline, report:
- distribution of cos(`d_resample`, `d_full`)
- **CIs on the ID estimates.** 5.21 vs 5.89 is currently a difference without error bars; it is not yet a claim.

Distinguish explicitly in the writeup: *stable under resampling* vs. *stable under algorithmic randomness*. Your kNN graph and geodesic estimation are deterministic given data, so what you can claim is the former. Say which one you tested.

### §3b. Hyperparameter sensitivity
Sweep k ∈ {5, 10, 15, 20, 30} for the kNN graph. Report geodesic/chord ratio, ID, and correlation with `v_ref` at each. If the 1.01–1.02 ratio is k-stable, it's a result; if it's k-dependent, it's a caveat that a reviewer will find before you do.

### §3c. Layer sweep
One forward pass with all layers hooked, then repeat the CPU-side analysis at layers {14, 20, 24, 28, 31}. Layer 28 is currently inherited from Wurgaft, not derived — this is a known vulnerability and the fix is one cache pass. Report whether conclusions are layer-stable. If they are, the inheritance stops mattering.

### §3d. Whitened metric (§3h)
Recompute geodesic/chord under a metric whitened by the control-bank covariance. Euclidean distance in the residual stream is a choice, not a given. Report as a single robustness row, not a new section.

---

## §4. P2 — Proposal section only. Do not run.

- Partial correlation on Stage A string similarity — cheap, but the surface-form rewrite control is the sharper test and you don't have time to do it properly. Argue in text that XSTest already controls surface form, which is exactly why 0.995 is the load-bearing number and 0.946 is not.
- §3i hierarchy check.
- **Stage A diagnosis tests.** The two failure modes — alkane series being monotone-ordinal (null straight line fits perfectly) and IUPAC string similarity introducing an orthographic confound — are *stated hypotheses with predictions*, not established mechanisms, and the P0 power check does not test either. Testing them properly means a non-monotone chemical series with known curvature plus a lexical control, which you don't have time for. **Writeup requirement, not optional:** say explicitly that the Stage A diagnoses remain untested hypotheses. The current framing is honest only if that sentence is present; without it, "we diagnosed two failure modes" reads as a finding.
- OLMo-3 pretraining checkpoint sweep.
- Extension to other concept families (Zhu et al. depression symptom vectors; persona vectors).

Frame these as "what I'd do with more time/compute" — for a MATS application that section is doing real work, and it's cheaper and more honest than half-running them.

---

## §5. Schedule

| When | What |
|---|---|
| Sept 9, now | §0 pre-registration commit. Non-negotiable, everything else waits on it. |
| Sept 9, afternoon | §1a, §1b, §1c, §1e (all fast). Start §1d in parallel. Kick off all-layer activation cache for §3c. |
| Sept 9, evening | Launch §2a. Let it run overnight if needed. |
| Sept 10, morning | §2a results, §2b. |
| Sept 10, afternoon | §3a–§3d, whichever fit. |
| **Sept 10, 18:00** | **HARD FREEZE. No new experiments.** |
| Sept 10 evening – Sept 11 | Writeup only. |

Pre-commit the freeze. The failure mode here is a late-breaking result that's too interesting not to chase and too unvalidated to report.

**Fallback rule:** if §2a doesn't complete cleanly, report it as unrun rather than partially run. A stated gap is survivable; a half-powered causal claim is not.

---

## §6. Writeup deltas

- **Abstract order unchanged:** XSTest AUROC leads (subject to §1c), unsupervised recovery second (subject to §1b), geometry third (subject to §1d).
- **Add a power sentence** to the straightness claim. A quantified null beats an unquantified one, and this is now the difference between "we found nothing" and "we can exclude curvature above X."
- **Limitations paragraph:** A/B/C established, D/E per §2/§3 outcomes. Self-report the §14d estimator swap with the §1e dual results attached — that turns a liability into evidence of process.
- **Cite Oozeer et al. for scope discipline:** you are making a faithfulness-of-*representation* claim, not a minimality/localization circuit claim. Their causal-abstraction vocabulary gives you a principled reason for the narrower claim rather than it reading as a hedge.
- **Do not claim circuit discovery anywhere.** The representational claim constrains what any circuit story must be consistent with; that's the claim, and it's enough.
- **"Why should this be trusted" paragraph.** Pre-registration, null baselines, explicit failure-mode framing, and causal-not-just-correlational validation are the Table 1 guidelines, and after §1–§2 you satisfy most of them. Point at that directly. Given Nanda's stream, "treat results as false until proven otherwise" is the compressed version and worth naming. Note you're a co-author — cite it as a standard you're held to, not as external endorsement of your own work, or it reads as circular.

---

## §7. Claim-dependency graph (30 min, do it Sept 9)

You never picked which piece of the position paper to operationalize. Of the three options, the claim-dependency graph is the one that earns its cost here, because §1b, §1c, §1d, and §1f each have a branch that invalidates a different headline:

- §1b fails → the novelty claim changes (manifold learning not load-bearing)
- §1c fails → the AUROC claim changes (transductive, not held-out)
- §1d fails → the straightness claim changes (below detection floor)
- §1f fails → the §14d ID contrast changes (CIs overlap)
- §2a arm 3 fires → the *conclusion* inverts (linear account incomplete)

Draw the graph: results → claims → abstract sentences. It takes half an hour, it makes the §0 branch structure legible instead of a list of conditionals, and it's a usable application figure. The audit checklist is redundant with §0. Skip PSL scoring — self-scoring your own work against your own paper's rubric is not a move that lands.