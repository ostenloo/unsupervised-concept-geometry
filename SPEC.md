# Implementation Spec: Unsupervised Recovery of Concept Geometry — Chemistry as Ground Truth, Refusal as Target

**Audience:** coding agent implementing end to end.
**Budget:** 16–20 hours of supervised work. One GPU with ≥24 GB (8B model in bf16).
**Calendar:** written Fri 4 Sep 2026. Application deadline Sun 6 Sep, possible extension to Fri 11 Sep. §8 gives a plan for both.
**Supersedes:** all prior specs in this project.

---

## 0. What this is

### The claim under test

Recent work (Wurgaft et al. 2026, arXiv 2605.05115) shows that steering a language model along a fitted *activation manifold* produces smooth, natural behavioral transitions, while linear steering "teleports" through unnatural intermediate states. Their manifolds are fit with ground-truth knowledge of the concept's structure (days form a cycle, ages an ordinal line). Their own limitations section names two gaps: an **unsupervised** fitting protocol, and extension to **abstract concepts such as refusal**, where they hedge that the geometry may not exist.

This project does both.

- **Stage A** builds an unsupervised pipeline — intrinsic dimension estimation, then manifold embedding — and validates it on **chemical compounds**, where an exact external ground-truth distance (Tanimoto similarity on molecular fingerprints) exists for every pair of points.
- **Stage B** applies the validated pipeline to **refusal**, where no ground truth exists, and reports which of two outcomes obtains.

### The Stage B fork, fixed in advance

- **Straight, one-dimensional.** Arditi et al.'s single-direction account of refusal is correct; manifold and linear steering coincide for refusal; the geometric framework has nothing to add there. A clean negative that maps where the framework applies.
- **Curved, or intrinsic dimension > 1.** The single-direction account is incomplete; linear steering cuts through structure; the geometric framework extends to at least one safety-relevant concept.

Both are results. The writeup must not favor either before the data is in.

### Why chemistry for Stage A

1. **Pairwise distances, not an ordering.** Tanimoto gives a real-valued distance between every pair of compounds. Weekdays give seven points and a cycle. Distance matrices are what Isomap, diffusion maps, RSA and Procrustes consume.
2. **Enough points.** Intrinsic-dimension estimators are badly biased below ~100 samples. Chemistry supplies hundreds of compounds; weekdays supply seven.
3. **A naive method fails on it.** PCA + `atan2` recovers the weekday circle, so passing that test is uninformative. Chemical similarity space is neither one-dimensional nor a circle.
4. **Two structure types in one domain.** Homologous series (methane → decane) are continuous 1-D manifolds; functional-group classes are a discrete hierarchy. The same data lets us ask whether manifold structure and hierarchical-orthogonal structure coexist.

---

## 1. Prior work and exactly what each leaves open

Read these before coding. Every item below is a claim the writeup will make; each must be accurate.

| Paper | What it established | What it did not do |
|---|---|---|
| **Wurgaft et al. 2026** (2605.05115) — manifold steering | Activation and behavior manifolds are approximately isometric; steering along `M_h` is ~2.8× more natural (lower `E_BC`) than linear steering; pullback from behavior recovers `M_h`. Llama 3.1 8B, layer 28, replace-not-add intervention. | Never estimates intrinsic dimension (assumed from ground truth). Never sweeps layers (28 chosen as "late enough"). Cyclic tasks used `atan2(PC1,PC2)` — already unsupervised; the supervision gap is in **sequential** (ordinal index) and **ICLR graph** (graph coords) tasks. Never states base vs. instruct. App. C.2: weekday centroids split by increment size reveal a second, entropy-carrying dimension (70B only). |
| **Singh & Chopra 2026** (2605.27970) — perceptual geometry | Color, pitch, taste, emotion geometry (MDS + RSA/GPA vs. human baselines) is weak early, **peaks in intermediate layers, attenuates late**, across LLaMA-3-8B, LLaMA-3.2-3B, Gemma-7B. Emotion is more persistent through depth. | Purely descriptive; fixed 2/3-D MDS; no ID estimation; no intervention. **Directly contradicts Wurgaft's "late layer" rationale.** |
| **Park, Choe, Jiang & Veitch 2025** (2406.01506) — hierarchy | Under a whitened "causal inner product" on the **unembedding** space, hierarchy is encoded as orthogonality (`ℓ_w ⊥ ℓ_z − ℓ_w` for `z ≺ w`); categorical concepts are polytopes/simplices. Fig. 12: **structure disappears under the naive Euclidean inner product.** | Never touches the residual stream (names internal layers as open). No manifold framing. Intervention is one logit-diff check, not a path. Predicts **flatness** (direct sums of polytopes). |
| **LatentChem 2026** (2602.07075) | Uses Tanimoto-vs-cosine RSA as a structure-preservation check on latent thought vectors. App. A: a Riemannian "chemical manifold" with a curvature–resolution theorem. | Chemistry is **injected** by a separate molecular encoder, not read from the LLM's text representations. Curvature `κ` is assumed, never measured. |
| **Arditi et al. 2024** (2406.11717) — refusal | Refusal is mediated by a single direction across 13 chat models; ablation removes refusal, addition induces it. Released pipeline. | Never asks whether the direction is the tangent of something curved, or whether structure exists off the line. |

Two synthesis points to carry into the writeup:

- **Metric choice.** Wurgaft use Euclidean distance in a PCA subspace. Park show Euclidean hides hierarchy that whitening reveals. Our pipeline is Euclidean by default; §3g runs a whitened comparison so a "no structure" result is not a metric artifact.
- **Depth.** Wurgaft steer at 28/32 (≈0.875 depth). Singh & Chopra say concept geometry peaks intermediate. §3f resolves this for chemistry by profiling every layer; §4 uses the result for refusal.

---

## 2. Environment and fixed frame

```
python 3.11, torch + CUDA, transformers, accelerate, safetensors
numpy, scipy, scikit-learn, pandas, matplotlib
rdkit                (fingerprints, Tanimoto, name validation)
scikit-dimension     (ID estimators: twoNN, MLE, lPCA)
```

### Model

`meta-llama/Llama-3.1-8B-Instruct`. Pin the revision hash in `config.py`.

Rationale: Stage B requires an instruct model (refusal is an instruction-tuning artifact). Wurgaft et al. never state their variant — the word "instruct" does not appear in the paper and their Llama citation is the LLaMA-1 paper. The `goodfire-ai/causalab` repo (branch `manifold_steering`) should be checked in hour one; if they used base, run the §3a smoke test on both and report whether concept manifolds survive instruction tuning.

### Fixed constants

| Parameter | Value | Note |
|---|---|---|
| Model dtype | bfloat16 | |
| Geometry dtype | **float32** | cast every captured activation on save; bf16 distance ties break every neighbor method below |
| Hook point | residual stream, output of decoder block `L` | `model.model.layers[L]` output; record it |
| Position | last token of the prompt | |
| Layers captured | **all 32** in a single pass | see §3c; layer is a profiled variable, not a fixed one |
| PCA dim for embedding | 64 | matches Wurgaft |
| Neighbor count `k` | sweep {8, 12, 16, 24} | report stability; Isomap is sensitive to this |
| Seeds | fixed per run, logged | |
| Decoding (Stage B generation) | greedy | α=0 must reproduce base exactly |

---

## 3. Stage A — chemistry (target ≈ 9 h)

### 3a. Smoke test on the upstream substrate (≈ 1 h)

Before any chemistry, reproduce the weekday loop from Wurgaft Table 1 on this model:

```
Q: What day is {k} days after {entity}?\nA:     entities Monday–Sunday, k = one…seven
```

49 prompts, centroid by answer, PCA to 3-D, look for the loop in PC1/PC2 at layer 28. This is a substrate check, not a result. If the loop is absent, the model variant or hook point is wrong — stop and fix.

### 3b. Compound set construction and gating (≈ 1.5 h)

**Target:** 250–400 compounds, each with a common name the model reliably knows and the tokenizer handles cleanly.

**Candidate pool.** Draw from three families, roughly balanced:

- **Homologous series** (the continuous-manifold arm): straight-chain alkanes C1–C10, 1-alkenes C2–C10, 1-alcohols C1–C10, carboxylic acids C1–C10. Include every member; these are the steering targets in §3e Level 3.
- **Functional-group classes** (the hierarchy arm): 20–40 common compounds each from alkanes, alcohols, ketones, aldehydes, carboxylic acids, esters, amines, aromatics, halides. These give class labels with known parent–child structure (aromatic ⊂ organic, etc.).
- **Common inorganics and everyday compounds** (breadth): water, ammonia, methane, carbon dioxide, sodium chloride, glucose, ethanol, caffeine, aspirin, etc.

**Three gates, applied in order. Log what each removes.**

1. **RDKit validity.** Every name must resolve to a parsable SMILES (use a name→SMILES source such as PubChem synonyms exported offline, or a hand-curated CSV; do not rely on the LLM to produce SMILES).
2. **Tokenizer gate.** Name tokenizes to ≤ 3 tokens under the Llama-3.1 tokenizer, with the leading-space variant. Print and inspect; IUPAC names fragment badly and the last-token activation of a fragmented name measures the fragment.
3. **Knowledge gate.** For each compound, ask the model two closed questions greedily: (i) molecular formula, (ii) functional-group class (multiple choice from the class list). Keep only compounds where both are correct. Report the pass rate. If pass rate < 70% on the initial pool, expand the pool rather than lowering the bar.

Freeze the resulting list as `data/compounds.csv` with columns `name, smiles, class, series, series_index, n_carbons`.

### 3c. Ground truth

> **Amended — see §11a.** The binary fingerprint specified below is degenerate on
> homologous series (heptane through eicosane all at distance exactly 0.000).
> Morgan *count* fingerprints are the default; binary and MACCS move to §3j.

For every pair `(i, j)`:

```
fp_i  = Morgan fingerprint, radius 2, 2048 bits   (RDKit)
T_ij  = Tanimoto(fp_i, fp_j)
d_ij  = 1 − T_ij
```

`d` is a proper metric on binary fingerprints (Jaccard distance). Store `D_truth` as an `N×N` float32 array. Also compute, for robustness, MACCS-key Tanimoto and report whether recovered structure correlates comparably under both fingerprints; disagreement means the result depends on fingerprint choice and must be flagged.

Additional per-compound ground-truth scalars: `n_carbons`, `molecular_weight` (RDKit), `series_index`.

### 3d. Activation capture (≈ 0.5 h compute)

Two prompt families. Both are needed; they serve different levels.

**Family 1 — compound as input** (for structure recovery, Levels 1–2, §3f, §3g):

```
"The chemical compound {name} is"
```

Capture the residual stream at the last token of `{name}` at **every layer**, float32. Shape per compound: `[32, hidden_dim]`. Store `acts/family1.pt` as `[N, 32, hidden_dim]`.

Use exactly one template. Do not average over templates in v1; that introduces a design variable with no time to ablate.

**Family 2 — compound as output** (for steering, Level 3):

```
"The straight-chain alkane with {n} carbon atoms is called"
"The straight-chain primary alcohol with {n} carbon atoms is called"
```

For each series, `n = 1…10`. Capture at the last prompt token. Also capture the output distribution restricted to the ten concept-name tokens plus an `other` bin (aggregate leading-space and case variants, per Wurgaft App. A.2). These are the behavior-space points.

Check accuracy on Family 2 first: the model must produce the correct name greedily for ≥ 8/10 per series, or that series is dropped from Level 3.

### 3e. Unsupervised pipeline

Input: `N` activation vectors at one layer, projected to 64-D PCA. **No labels, no `D_truth`, no class information.**

**Step 1 — intrinsic dimension.** *(Amended — see §11d: reported as a curve; TwoNN
excluded from the §3a weekday centroids.)* Three estimators, always reported together:

- two-NN (Facco et al.)
- Levina–Bickel MLE, `k ∈ {10, 20}`
- local PCA / participation ratio

Report as a triple with the spread. If the estimators disagree by > 1.5, state that the dimension is not determined by the data.

**Step 2 — embedding.** With `d_hat` from Step 1 (round to nearest integer, and also run `d_hat ± 1`):

- **Isomap**, `n_components = d_hat`, over the `k` sweep
- **Diffusion maps** (second method; keep if time allows, cut first under §8)

Output: recovered coordinates `Y ∈ R^{N × d_hat}` and the geodesic distance matrix `D_geo` from the Isomap neighbor graph.

**Step 3 — naive baseline, always run alongside.** Top-`d_hat` PCA coordinates and their Euclidean distances. Every evaluation below is reported for both the full pipeline and this baseline. The tasks where they diverge are the evidence that the pipeline is doing work; where they agree, say so.

### 3f. Evaluation levels

**Level 1 — global distance recovery.** *(Amended — see §11b: 36% of pairs are
censored at distance 1.0, not tied, and are reported separately.)*

- RSA: Spearman correlation between upper-triangular entries of `D_geo` and `D_truth`.
- Procrustes: after MDS-embedding `D_truth` to `d_hat` dimensions, align `Y` to it; report disparity.
- Do both for the pipeline and the PCA baseline.

**Level 2 — series recovery.** Restrict to each homologous series. Spearman correlation between the recovered 1-D coordinate along that series (first Isomap component on the subset) and `n_carbons`. This is the direct analog of Wurgaft's letters/ages ordinal recovery, done without the ordinal index.

**Level 3 — behavioral equivalence via steering (the causal test).** For the alkane and alcohol series only, at the layer selected in §3g:

1. Fit a manifold through the ten Family-2 centroids using the **unsupervised** coordinate from Level 2 as the intrinsic parameter (natural cubic spline in the 64-D PCA subspace, as Wurgaft App. A.3).
2. Fit the same spline using the ground-truth `n_carbons` as intrinsic parameter — the supervised reference.
3. Fit the behavior manifold `M_y` from the Family-2 output distributions in Hellinger coordinates (Wurgaft App. A.4; fit in the tangent plane, decode with the exponential map).
4. For up to 20 centroid pairs, steer with `K = 50` waypoints under three strategies: **linear** (straight line, replace full activation), **manifold-unsupervised**, **manifold-supervised**. Replace the top-64 PCA components and keep the orthogonal residual, as Wurgaft App. A.6.
5. Score each trajectory by cumulative Bhattacharyya energy `E_BC` to the nearest point on `M_y` (Wurgaft Eq. 3 / App. A.7). Average over 8 base prompts.

**Success criterion, stated before running:** `E_BC(unsupervised)` is closer to `E_BC(supervised)` than to `E_BC(linear)`. Report all three with standard errors over pairs.

Also record the qualitative signature: does linear steering "teleport" (mass jumping from methane to hexane without passing through propane/butane) while manifold steering walks the series?

### 3g. Layer profile (≈ 1 h; cheap because all layers are already captured)

Run Steps 1–2 and Level 1 at **every layer**. Plot RSA-vs-`D_truth` and ID estimates against layer index.

This resolves the Wurgaft/Singh-&-Chopra conflict for this domain:

- If alignment peaks intermediate and decays by layer 28, Singh & Chopra's rise-peak-fall extends to chemistry and Wurgaft's "late enough" rationale is wrong here.
- If alignment is flat or still rising at 28, the perceptual result does not generalize to this domain.

**Layer selection rule for Level 3 and Stage B:** the layer of peak Level-1 RSA (`L_peak`). Additionally run Level 3 at layer 28 for direct comparison with Wurgaft's choice. Both numbers go in the writeup; do not pick whichever looks better after the fact.

### 3h. Metric comparison (≈ 0.5 h)

Park et al. show hierarchical structure appears under a whitened inner product and vanishes under Euclidean. Repeat Level 1 at `L_peak` with activations whitened by the covariance of an independent activation bank:

```
W = Cov(bank_L)^(-1/2)      # bank: 2,000 general-instruction prompts, same layer, same position
h_white = W (h − mean(bank_L))
```

Report Level 1 under both metrics. If whitening materially changes RSA (Δ > 0.1), the metric is load-bearing and Stage B must be run under both.

### 3i. Hierarchy check (≈ 0.5 h; secondary, report briefly)

Using class labels **only for evaluation**, at `L_peak`:

1. Compute class-mean vectors `μ_c` for each functional-group class and a parent mean `μ_organic`.
2. Test Park's orthogonality prediction in the residual stream: cosine between `μ_organic` and `μ_c − μ_organic` for each class, under Euclidean and whitened metrics.
3. Compute the cosine between the within-series tangent (alkane series direction from Level 2) and the between-class difference vectors.

This is descriptive. The question is whether the continuous (series) and discrete (class) structures occupy orthogonal subspaces, as a direct-sum-of-polytopes picture would predict. One figure, two sentences in the writeup.

### 3j. Null controls (mandatory)

- **Shuffle null.** Permute the compound↔activation assignment; rerun Level 1. RSA must collapse to ≈ 0.
- **Isotropic null.** 300 Gaussian points in 64-D; ID estimators must return a large value, not 1–3.
- **Synthetic circle and grid.** 12 points on a circle and a 5×5 grid, embedded in 64-D with mild curvature; Isomap must recover ordering, ID must return ≈1 and ≈2.
- **Fingerprint robustness.** Level 1 under Morgan vs. MACCS; report both.

---

## 4. Stage B — refusal (target ≈ 7 h)

### 4a. The design constraint

Stage A had `D_truth`. Refusal has none. The pipeline must therefore run on **activations alone**, with behavior used only afterward as validation. Do not bin prompts by refusal rate and then look for one-dimensional structure — that constructs the answer.

### 4b. Prompt set (≈ 1 h)

~600 prompts, three sources, roughly balanced:

- **Harmful:** harmful splits from `andyrdt/refusal_direction` (AdvBench / HarmBench / JailbreakBench derivatives).
- **Harmless:** Alpaca split from the same repo.
- **Borderline — essential:** XSTest, purpose-built for prompts that resemble harmful requests but are benign. Without an intermediate region there is no interior for a manifold to curve through and the answer is trivially "two clusters."

Apply the chat template exactly once; print one fully rendered prompt and read it. Capture last-prompt-token activations at **all layers**, float32.

Separately, per prompt, record two behavioral measures held back until §4e:

- `refusal_prob`: probability mass on a set of refusal-opener tokens at the first response position (verify token IDs, including leading-space variants).
- `refusal_generated`: greedy generation + Arditi's substring refusal metric.

### 4c. The published direction (≈ 1 h)

Run Arditi's pipeline on `Llama-3.1-8B-Instruct` to obtain their selected direction `v_ref` and selected layer `L_ref`. Do not override their selection. The Together-AI safety-eval stage is unnecessary; if it blocks, extract the direction and skip it. Record the upstream commit hash.

### 4d. Structure measurement

Run at three layers: `L_ref` (Arditi's), `L_peak` (from §3g), and 28 (Wurgaft's). Report all three; do not select post hoc.

**Intrinsic dimension** on the ~600 raw activations (never on centroids), three estimators, with the general-instruction bank at the same layer as the "unstructured" reference.

**Curvature.** Fit Isomap-1D and a principal curve. Report the distribution over well-separated pairs of `geodesic_length / chord_length`. Near 1.0 = straight.

**Tangent alignment with `v_ref` — the key number.** Compute the cosine between `v_ref` and the principal-curve tangent at 10 evenly spaced points along the curve.

- Straight, tangent ≈ `v_ref` everywhere → independent recovery of Arditi's result by a method that assumed nothing about refusal. Strong convergent validation.
- Tangent rotates along the curve → the single-direction account is incomplete in a localizable way.

**Metric comparison.** If §3h found whitening load-bearing, repeat all of the above whitened.

### 4e. Post-hoc behavioral validation

Only now: Spearman correlation between the recovered first intrinsic coordinate and `refusal_prob`; and between recovered coordinate and `refusal_generated`. A strong monotone relationship means the unsupervised coordinate found the behavioral axis. A weak one means the dominant structure at that layer is something else — topic, length, register — and that is reported as-is.

### 4f. Steering consistency (≈ 1.5 h)

**If §4d finds the structure straight:** manifold and linear steering must give nearly identical trajectories. Run both between five harmless↔harmful centroid pairs and confirm. Divergence here indicates a pipeline error.

**If curved:** run both and compare. The behavior-space metric is an approximation — restrict the next-token distribution to {refusal-opener tokens, compliance-opener tokens, other}, map to Hellinger coordinates, and reuse the `E_BC` machinery. **Label this as an approximation in the writeup**; it is the weakest link and a reader will find it.

Note the intervention-type mismatch: Wurgaft replace the activation with a centroid; Arditi add a scaled vector. Use replacement here for comparability with Stage A, and say so.

---

## 5. Logging

One parquet per stage, written incrementally. Never hold results in memory until the end.

**Stage A, one row per (layer, method, k, metric, fingerprint):**
```
run_id, timestamp, model_revision, layer, pca_dim, k, method, metric_type,
n_compounds, id_twonn, id_mle, id_lpca, id_spread,
rsa_spearman, procrustes_disparity, rsa_pca_baseline,
series_spearman_alkane, series_spearman_alcohol, series_spearman_acid,
E_BC_linear, E_BC_manifold_unsup, E_BC_manifold_sup, E_BC_se,
fingerprint_type, shuffle_null_rsa, notes
```

**Stage B, one row per (layer, metric_type):**
```
run_id, layer, metric_type, n_prompts, id_twonn, id_mle, id_lpca,
id_bank_reference, geo_chord_ratio_mean, geo_chord_ratio_iqr,
tangent_cosine_vs_vref (list of 10), coord_vs_refusal_prob_spearman,
coord_vs_refusal_generated_spearman, E_BC_linear, E_BC_manifold, notes
```

Plus per-prompt Stage B table: `prompt_id, source, refusal_prob, refusal_generated, coord_1, coord_2`.

---

## 6. Acceptance tests

Run before any real run and before each session. All must pass.

1. Synthetic circle → ordering recovered, ID ≈ 1, topology flagged cyclic.
2. Synthetic 5×5 grid → ID ≈ 2, Procrustes recovery high.
3. Isotropic Gaussian → ID large.
4. Weekday loop visible at layer 28 (§3a).
5. Shuffle null collapses RSA to ≈ 0.
6. No bf16 tensor reaches any distance computation (assert dtype).
7. Tanimoto matrix symmetric, zero diagonal, triangle inequality holds on 1,000 random triples.
8. Compound list passes all three gates; gate removal counts logged.
9. Stage B: greedy generation at α=0 reproduces base output token-for-token; hook counter increments; hooks removed after every run.
10. Sign/offset invariance: coordinate-recovery scores unchanged under reversal or rotation of recovered coordinates.

---

## 7. Figures

**F1 — Instrument validation.** Recovered Isomap coordinates vs. MDS-of-Tanimoto, Procrustes-aligned, colored by functional-group class. Inset: RSA for pipeline vs. PCA baseline.

**F2 — Layer profile.** RSA vs. layer, with ID estimates on a second axis. Mark `L_peak` and 28. This is the Wurgaft/Singh-&-Chopra resolution figure.

**F3 — Causal test.** `E_BC` for linear / manifold-unsupervised / manifold-supervised on alkanes and alcohols, at `L_peak` and 28. Below: one example trajectory per strategy showing teleportation vs. walking.

**F4 — Refusal fork.** Left: geodesic/chord ratio distribution at three layers. Right: cosine between curve tangent and `v_ref` along the curve. The shape of this panel is the Stage B answer.

**F5 — Refusal behavioral check.** Recovered coordinate vs. `refusal_prob`, points colored by source (harmful / harmless / XSTest).

**F6 (secondary) — Hierarchy.** Orthogonality cosines under Euclidean vs. whitened metrics.

---

## 8. Time plan and cut order

### If submitting Sun 6 Sep (≈ 16 h available from now)

| Block | Hours | Content |
|---|---|---|
| Setup + smoke test + compound gating | 3 | §2, §3a, §3b, §3c |
| Capture + pipeline + Levels 1–2 + layer profile + nulls | 4 | §3d–§3g, §3j |
| Level 3 steering, alkanes only, `L_peak` and 28 | 2 | §3f L3 |
| Stage B capture, Arditi direction, structure measurement, behavioral check | 4 | §4b–§4e |
| Writeup + figures | 3 | §7, §10 |

Cut §3h, §3i, §4f, diffusion maps, MACCS robustness, alcohol series.

### If extended to Fri 11 Sep (≈ 20+ h)

Add back in this order: §4f steering consistency → §3h metric comparison (and its Stage B rerun if load-bearing) → alcohol series in Level 3 → §3i hierarchy check → diffusion maps → MACCS.

### Never cut

The synthetic acceptance tests, the shuffle null, the isotropic null, the PCA baseline alongside every result, the three-layer reporting in Stage B, and the §1 accuracy about which upstream tasks were actually supervised.

---

## 9. Risks, ranked

1. **Compound knowledge gate fails at scale.** An 8B model may know fewer common compounds than expected. Mitigation: start with the 150 most common names; expand only if pass rate is high. Report the gate pass rate as a result in its own right.
2. **Model variant mismatch with upstream** (§2). Resolve in hour one from the `causalab` repo.
3. **ID estimators disagree.** Expected at N ≈ 300 in 64-D. Report the triple; never a single number.
4. **Layer 28 vs. `L_peak` disagree and the steering result flips between them.** That is a finding, not a failure. Report both.
5. **Whitening changes the answer** (§3h). If so, Stage B doubles in cost. Under deadline, report Stage B Euclidean-only and flag the metric caveat prominently.
6. **Tokenizer fragmentation of compound names.** Gate 2 catches it, but check the gated set once more visually.
7. **Stage B behavior metric is an approximation** (§4f). Flag, don't defend.
8. **Isomap `k` sensitivity.** Sweep and report the stable range; if there is none, say so.
9. **Goodfire will likely run the chemistry version.** Advantage is scope: one arm done carefully with a published-ground-truth metric, connected to two other papers' gaps, before they do.
10. **Family-1 prompt template is a single arbitrary choice.** State it as such; a template ablation is future work.

---

## 10. Writeup framing

Lead with the instrument, not the domain. The contribution is an unsupervised pipeline that (a) recovers exact chemical ground truth without labels, (b) resolves a depth disagreement between two 2026 papers for this domain, and (c) when pointed at refusal returns one of two pre-registered outcomes.

State plainly which upstream tasks were already unsupervised (`atan2` for cyclic) so the reviewer does not discover it. State that Tanimoto was chosen because it is a proper metric with a citable definition and no human judgment in the loop. State every approximation (Stage B behavior space, single prompt template, replacement-vs-addition intervention) in the limitations, not the appendix.

For the MATS application specifically: the LLM-assistance disclosure should say a language model helped survey the literature and draft this spec; every measurement, decision on layer selection, and interpretation of the fork is the applicant's.
---

## 11. Amendments — measured, Fri 4 Sep 2026

Recorded **before any activation was captured for Stage A**, from RDKit output and
tokenizer output alone. The §3a weekday smoke test had been run (substrate check
only); no compound activations existed. Each item below changes a choice
pre-registered above; the original text is left intact so the change is visible.

### 11a. Ground-truth fingerprint: counts, not bits (amends §3c)

**Measured degeneracy of the specified metric.** Binary Morgan, radius 2, 2048
bits, on the gated compound set:

| | binary Morgan | MACCS | **Morgan counts** |
|---|---|---|---|
| heptane↔octane↔nonane↔decane↔undecane↔eicosane | **all exactly 0.000** | **all exactly 0.000** | strictly monotone |
| pairs at distance exactly 0.0 | 0.22% | 0.24% | **0.00%** |
| distinct distance values (N=167) | 294 | 531 | **567** |
| pairs at distance exactly 1.0 | 35.9% | 28.2% | 35.9% |

(N=167 at the time of measurement; the compound set has since grown to 218.)

Every interior carbon of a straight chain has the same radius-2 environment, so
past ~7 carbons the *set* of environments stops growing and the bit vector stops
changing. The alkane homologous series is this spec's own continuous-manifold arm
and the Level 2 / Level 3 target: a ground truth that places heptane and decane at
distance zero cannot validate an ordinal recovery over that series. This is not a
robustness wrinkle; it is the instrument being blind on the structure it exists to
measure.

**Decision.** Morgan **count** fingerprints (radius 2, 2048 bits) are the default
`D_truth`. Binary Morgan and MACCS both move to §3j as robustness arms, reported
alongside every Level 1 number.

**The metric, named exactly.** `rdkit.DataStructs.BulkTanimotoSimilarity` applied
to the `UIntSparseIntVect` from
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetCountFingerprint(mol)`.
On count vectors that function computes

```
    S(a,b) = Σ min(a_i, b_i) / Σ max(a_i, b_i)          (MinMax / Ruzicka)
    d(a,b) = 1 − S(a,b)                                  (Soergel distance)
```

which is a proper metric. Verified against a hand-computed Σmin/Σmax on four
pairs. This is stated because the *other* function commonly called "continuous
Tanimoto", `a·b / (|a|² + |b|² − a·b)`, is **not** a metric, and the 5,000-triple
triangle-inequality check in acceptance test 7 passes under both — the check
cannot distinguish them, so the function called must be named rather than
inferred.

### 11b. Censored pairs are not ties (amends §3f Level 1)

35.9% of pairs sit at distance exactly 1.0 under every fingerprint tested: they
share no substructure at all, so the value carries no information about *how*
unrelated the two compounds are. These are **censored observations**, not tied
ones, and feeding them to a Spearman as ties is a modeling error. Level 1 is
therefore reported as two numbers:

1. **Rank correlation on the uncensored subset** (`D_truth < 1.0`).
2. **A two-sample test** on whether recovered distance is larger for censored than
   for uncensored pairs.

Isomap is largely insulated from this — geodesics are built from short edges and
long distances are inferred — so the censoring bites Level 1 RSA considerably
harder than it bites the embedding. Report the uncensored-subset RSA for the PCA
baseline too, or the comparison is not like-for-like.

### 11c. Compound naming and the gate-2 read position (amends §3b)

**Bare names.** The IUPAC `1-` locant is catastrophic under this tokenizer
(`1-propanol` → 5 tokens). The flat ≤3-token gate deleted 8/10 alcohols and 7/9
alkenes — **both Level 3 steering series**. Bare names (`propanol`, `butene`)
cost ≤3 and recover 10/10 and 9/9.

**Gate 2 gates the read position, not the string.** The spec's own rationale is
"the last-token activation of a fragmented name measures the fragment". Family 1
reads the *last* token of the name, and for every two-word `X acid` name that
token is `Ġacid` — a complete word, never a fragment. A flat budget rejected
butyric/caproic/enanthic/pelargonic acid for stem fragmentation the read position
never sees, costing 4/10 of the acid series. Gate 2 is therefore:

| | budget |
|---|---|
| single-word names | ≤3 tokens (unchanged) |
| multi-word names | ≤5 tokens total **and** final word ≤2 tokens |

This admits 29 multi-word names, all with clean final tokens (`Ġacid`, `ate`,
`Ġoxide`, `Ġalcohol`), completes the acid series at **10/10**, and refills the
ester class. It rejects `carbon tetrachloride`, `acetic anhydride` and
`dimethyl phthalate`, whose final words are themselves shredded.

**The acid series is an orthographic control, not a compromise.** Its names
(formic, acetic, propionic, butyric, valeric, caproic, enanthic, caprylic,
pelargonic, capric) carry no lexical ordering whatsoever, while the fingerprints
do. The standing objection to Stage A is that recovered structure is string
similarity; a complete 10-member series where the two are decoupled by
construction is the cleanest available answer, and is presented that way rather
than apologised for as inconsistent naming.

**Isomer denotation, measured (gate 3b).** Bare names are isomerically ambiguous
and gate 3 cannot catch a misassignment — isomers share a molecular formula *and*
a functional-group class. So denotation is probed directly, with **three
phrasings**, because one misleads:

- Asking "is butanol primary, secondary or tertiary?" is worthless. Forced to one
  word the model calls decanol **tertiary**; open-ended it says "it is a primary
  alcohol". That question measures format compliance, not knowledge.
- Asking for the *name* the bare term denotes is the right probe, but still needs
  replication: "IUPAC name of butene" returns **but-2-ene** while "systematic name
  of butene" returns **but-1-ene**, from the same model.

| series | result | consequence |
|---|---|---|
| **alcohol** | 96% probe agreement, 7/8 names unanimous; "is the hydroxyl on a terminal carbon?" is **8/8 Yes** | our straight-chain primary SMILES is **validated**; the Level 3 steering series is sound |
| **alkene** | 71% agreement, 2/7 unanimous; four phrasings give but-2-ene, but-1-ene, position "2", and `C=C=C=C` | **no stable denotation.** We keep the 1-isomer (conventional and self-consistent) and flag every alkene `D_truth` entry as carrying isomer uncertainty. Level 2 is unaffected: carbon count is unambiguous under every probe. Alkenes are **not** promoted to a Level 3 series. |

Incidentally the model's free-text SMILES were chemically nonsense throughout
(`C=C=C=C` for butene, `C=CCC=C` for pentene), which retrospectively vindicates
§3b gate 1's instruction not to source SMILES from the LLM.

### 11c-bis. The final-token confound (new control)

Family 1 reads the last token of the name, and the gated set has only ~70
distinct final tokens over ~220 compounds — `Ġacid` alone covers 15%, and only 47
compounds have a final token unique to them. A reader can reasonably ask whether
recovered structure is *final-token identity* rather than chemistry.

Control: `evaluation.within_group_rsa` recomputes Level 1 restricted to pairs
sharing a final token. Structure surviving inside the 41-member `Ġacid` block
cannot be explained by the block label. The shuffle null does **not** cover this,
because permuting compound↔activation destroys token structure and chemical
structure together.

**The comparison must be like-for-like, and my first version of it was not.**
`within_group_rsa` pools *all* within-group pairs while the headline Level 1
number is restricted to *uncensored* pairs. All four numbers, layer 8:

| | rho | pairs | censored |
|---|---|---|---|
| overall, all pairs | 0.332 | 23,653 | 32.7% |
| **overall, uncensored only (headline)** | **0.342** | 15,907 | 0% |
| within final token, all pairs | 0.605 | 1,345 | 5.9% |
| **within final token, uncensored only** | **0.556** | 1,265 | 0% |

So the honest comparison is **0.556 against 0.342**, and censoring is *not* what
explains the gap: the censored pairs are already excluded from the headline, and
removing them moves the overall number by only 0.010 (0.332 → 0.342). An earlier
draft of this section asserted a censoring mechanism; it was wrong on the numbers.

Nor is it a subset-size artifact. Against **200 size-matched random groupings**
(same group sizes, random membership, uncensored): mean 0.342, sd 0.041 — exactly
the overall value. The observed 0.556 is **z = 5.3**.

**An earlier draft claimed the grouping variable is "chemistry, not orthography".
That is refuted by its own numbers and is reversed in §12g.** Grouping by
functional class gives 0.439 against final token's 0.556 — if chemistry were the
operative variable the chemical partition should do at least as well as its
orthographic proxy, and it does worse. Put on one scale by giving each partition
its own size-matched null, final token is **z = 9.5** and functional class
**z = 4.2**. The orthographic partition is the stronger one. See §12g, where the
confound is measured directly rather than by proxy.

**What the control does and does not license.** It clears what it was built for:
if recovered structure were read-position identity, a fixed final token would show
*no* structure, and it shows a great deal. But 0.556 is a *local, within-block*
quantity and Level 1's claim is *global*, so it is not a better estimate of the
headline and must never be quoted as one.

One tempting alternative explanation is measured and wrong, recorded so it is not
re-proposed: the groups are *not* dominated by homologous series (26% membership),
and within-group Tanimoto does *not* reduce to carbon-count difference: the
Spearman against absolute carbon-count difference is 0.29 in `Ġacid`, 0.20 in
`ene`, 0.22 in `ane`, and negative in `ine` and `amine` (re-derived with
p-values; `ane`, `one`, `amine` and `ol` are not significant at 0.05).

**The Level 1 ceiling therefore remains 0.345 with Procrustes disparity 0.83.**

### 11d. Intrinsic dimension is a curve, not a scalar (amends §3e step 1)

Two measured facts about the estimators:

- **TwoNN diverges on regular lattices.** It estimates
  `ln2 / mean(ln(r₂/r₁))`; on equispaced samples both nearest neighbours are
  equidistant, so `r₂/r₁ → 1` and the estimate blows up. This spec's own 12-point
  circle returns **22** equispaced and **1.5** jittered. Synthetic manifolds are
  therefore jitter-sampled.
  **Consequence for §3a: never run TwoNN on the weekday centroids.** The seven
  answer-day centroids are near-equispaced by construction — the smoke test
  measures a max angular gap ratio of 1.25 — which is exactly the divergence
  condition. Excluded deliberately; not to be rediscovered as a bug.
- **ID is a statement about structure above the local noise scale.** With noise
  held at an absolute scale, a 1-D circle in 64-D reads ID 2.7 at N=12 and 15.6 at
  N=300: the 64-dimensional noise ball swallows the inter-point spacing and the
  estimator correctly reports the noise. Rising ID with N is the signature.

So ID is published **as a curve**, one figure (**F7**): against `k ∈ {10, 20}`,
against subsample size, and — for the synthetics — against noise-to-spacing ratio.
This pre-empts the obvious question of whether the Stage B number is an artifact
of sampling density.

**What the estimator does and does not support.** Our Levina–Bickel implementation
(skdim's global `MLE.fit()` ignores its `K` argument, so it is implemented
directly) is accurate at d=1 (0.97) and honest through d=10 (9.13), with a known
downward bias above that (d=20 → 14.1). Stage B's fork is **ID≈1 versus ID>1**,
which the estimator supports well. It does **not** support any claim of the form
"the refusal manifold is exactly 12-dimensional", and no such claim will be made.

### 11e. Level 3 base prompts (resolves the §3f/§3d ambiguity)

The 8 base prompts averaged over in §3f Level 3 step 5 are **paraphrases of the
Family-2 frame**, varying surface wording only. Level 3 fits the behavior manifold
`M_y` from Family-2 output distributions, so the base prompts must elicit the same
output space or the Hellinger coordinates are not comparable across prompts.
Drawing them from Family 1 would average trajectories over a different behavior
manifold than the one fitted. §3d's "use exactly one template" remains scoped to
Family 1.

### 11f. The refusal direction (amends §4c)

Arditi et al.'s pipeline is reimplemented rather than run: difference-in-means is
fifteen lines, their repo is an afternoon of dependency work.

**Precision required in the writeup.** The non-trivial part of Arditi et al. is
not the difference of means — it is the selection sweep over layer and position
scored on a validation set. We fix layer and position from the §3g profile and
compute the direction directly. It is therefore described as **"a difference-in-means
refusal direction at the selected layer"**, never as "Arditi's direction".
If their released tensor for `Llama-3.1-8B-Instruct` is a single download, take it
as a cross-check: the cosine between the two is free validation, and a *low*
cosine is itself a reportable result.


---

## 12. Amendments after the Stage A layer profile — Sat 5 Sep 2026

Recorded after §3d capture and §3g, before Level 3.

### 12a. `L_peak` is not identified; the repair is a tie-break, not a different layer

§3g selects "the layer of peak Level-1 RSA". The sweep shows that quantity is not
identified in this domain: uncensored RSA rises over layers 0–4 (0.139 → 0.329)
and then plateaus, **19 of 32 layers sit within one k-sweep sd (0.034) of the
maximum**, and the argmax moves to layer 5, 7, 8, 9, 24, 25 or 26 depending on
fingerprint and `k`.

**Tie-break rule (one line, stated before Level 3):** `L_peak` is the argmax of
*mean* uncensored RSA across the full (fingerprint × k) grid for the default
count fingerprint. That gives **layer 8**, and it is the primary layer.

Layer 28 remains the comparability arm exactly as §3g requires. It is **not**
promoted to primary: the pre-registered rule targets a quantity, and the honest
repair is a stable estimator *of that quantity*, not a different layer chosen for
external comparability — which would be the same post-hoc move as picking
whichever number looked better.

**Disagreement rule, pre-declared now:** if Level 3 differs between layer 8 and
layer 28, the finding is **layer-sensitive and neither is *the* result**. This is
§9 risk 4 made operational.

**The plateau is a prediction, not just an annoyance.** If Level-1 alignment is
flat across 19 layers, Level 3 should be flat across them too. If Level 3 instead
turns out sharply layer-dependent, the plateau does not mean what §3g assumed it
meant, and that discrepancy is itself the finding.

### 12b. The plateau is not an artifact of the fixed 64-d projection

The profile embeds every layer through PCA-64 while raw ID varies with depth
(16.0 at layer 4, 12.9 at layer 8, 8.1 at layer 28) and is still climbing at
N=218. A fixed truncation could flatten a real profile by removing
proportionally more from the layers carrying more structure. Tested directly —
Level 1 across all 32 layers at `d ∈ {16, 32, 64, 128, 217, raw 4096}`:

| projection | argmax | max RSA | layer 28 | peak-to-plateau range | layers within one k-sd |
|---|---|---|---|---|---|
| 16 | 28 | 0.270 | 0.270 | 0.101 | 15 |
| 32 | 5 | 0.333 | 0.314 | 0.151 | 13 |
| **64** | 9 | 0.351 | 0.327 | 0.115 | 20 |
| 128 | 23 | 0.348 | 0.312 | **0.056** | 19 |
| 217 | 24 | 0.350 | 0.320 | 0.079 | 18 |
| raw 4096 | 24 | 0.350 | 0.320 | 0.079 | 18 |

**Two statistics here were wrong and are replaced.**

Peak-to-plateau range (max−min over 32 layers) is contaminated by the low early
layers: a monotone rise-then-flat profile produces a large range with no peak.
Worse, the "own sd" column in an earlier version of this table was
`std(RSA across layers)` at a single `k` — that is the structure under test, not
noise — while the "flat layers" column counted against a *fixed* yardstick
borrowed from d=64. The two columns were unconnected and the header was wrong,
which made the dismissal of d=16/32 as "noisier, not sharper" unsupported.

Recomputed with the **full k sweep at every `d`**, so each dimension gets a real
measurement-noise estimate (mean across-k sd), plus a scale-free prominence
statistic — how far the best layer stands above a median layer, in that
dimension's own noise units:

| d | k-sweep sd | max | layer 28 | flat/32 (own sd) | **prominence z** | argmax |
|---|---|---|---|---|---|---|
| 16 | 0.0323 | 0.267 | 0.266 | 18 | **0.50** | 7 |
| 32 | 0.0407 | 0.324 | 0.299 | 16 | **0.86** | 7 |
| **64** | 0.0336 | 0.345 | 0.322 | 19 | **0.57** | 8 |
| 128 | 0.0230 | 0.350 | 0.328 | 14 | **1.00** | 23 |
| 217 | 0.0268 | 0.345 | 0.321 | 19 | **0.84** | 24 |
| raw 4096 | 0.0268 | 0.345 | 0.321 | 19 | **0.84** | 24 |

**At every `d` the best layer stands 0.50–1.00 noise units above a median layer.**
A sharp peak would stand many. The low dimensions need no dismissal: 0.50 and
0.86 sit inside the same range as d=128's 1.00 and d=217's 0.84. Flat-counts
alone are not comparable across `d` because the band width is the noise estimate
(d=128 looks least flat only because its k-sd is smallest), which is why
prominence is the statistic quoted.

(`d=217` and raw 4096 agree to three decimals, as they must: with N=218 points a
217-component PCA is a lossless rotation. A free check that the projection path
is not distorting distances.)

### 12c. Level 2 series scope: pre-registered headline, alkene exploratory

The §3d gate — "the model must produce the correct name greedily for ≥ 8/10 per
series, or that series is dropped" — is a **Family-2 greedy-accuracy** gate scoped
to **Level 3**. Both Level 3 series scored **10/10**, so both pass as written.
That gate never applied to acids or alkenes, which have no Family-2 template.

The "acid 8/10" figure reported earlier is a *different quantity* — how many acid
names survived gate 3 — and conflating the two was an error of reporting, not of
gating.

For Level 2, §5's schema names `series_spearman_alkane`, `_alcohol`, `_acid`.
**Alkene was never pre-registered**; we added it. Calling its removal a
"restoration" overstates the case — we added it, saw 0.86, and removed it, and
that is a data-dependent action whichever direction it points. The accurate
framing: **the pre-registered set is the headline, and alkene was an exploratory
addition, reported with its score.**

**Invariance check, which settles the concern.** The Level 2 verdict is identical
either way (mean over layers ≥4):

| series set | Isomap | PCA baseline | PCA better on |
|---|---|---|---|
| pre-registered (alkane/alcohol/acid) | 0.871 | 0.886 | 2 of 3 |
| with alkene included | 0.869 | 0.905 | 3 of 4 |

Including alkene makes the negative result **stronger**, not weaker, so the
exclusion cannot have manufactured it.

The scores:

| series | pre-registered in §5 | Level 2 abs-rho at layer 8 | status |
|---|---|---|---|
| alkane | yes | 0.92 | reported |
| alcohol | yes | 0.97 | reported |
| acid | yes | 0.88 | reported |
| **alkene** | **no** | 0.86 | **excluded** from Level 2; class-membership use only (also §11c: no stable isomeric denotation) |

### 12d. Fractional depth, for the record

Layer 28 is a comparable index here: Wurgaft use Llama 3.1 8B and this project
uses `Llama-3.1-8B-Instruct`; both have **32 decoder blocks** (asserted at load in
`src/model.load`), so layer 28 is 0.875 fractional depth in both. Comparability by
index is therefore sound in this instance. What remains open is §2's base-vs-instruct
question, which does not affect depth indexing. Had the depths differed, the
comparison would have had to be made at matched fractional depth, and the case for
28 would have been weaker still.

### 12e. The Level 2 negative qualifies §0 (amends §0, "Why chemistry")

§0 point 3 justifies chemistry on the grounds that "a naive method fails on it":
PCA + `atan2` recovers the weekday circle, so passing that test is uninformative.
**Measured, that justification holds for Level 1 and fails for Level 2.**

- **Level 1**: Isomap 0.345 vs PCA baseline 0.278. The pipeline does work.
- **Level 2**: Isomap 0.88–0.97 and PCA **0.87–0.96**, with PCA sometimes better
  (acid at layer 2: 0.93 vs 0.74). Homologous-series ordinal structure is
  linearly accessible; no manifold method is needed to recover it.

Per §3e Step 3 this belongs in the body, not a footnote: on the ordinal task that
is the direct analog of Wurgaft's letters/ages recovery, **the naive baseline is
not beaten**. The unsupervised-pipeline claim rests on Level 1 and Level 3, and
Level 2 should be presented as a task where pipeline and baseline agree.


### 12f. "Differ" defined before Level 3 runs

§12a pre-declares that a Level 3 disagreement between layers 8 and 28 makes the
finding layer-sensitive. That is a shape without a magnitude, and deciding the
magnitude while looking at the answer is the failure §3g already cost a day to.
So, fixed now.

**Quantity.** §3f's success criterion is that `E_BC(unsupervised)` is closer to
`E_BC(supervised)` than to `E_BC(linear)`. Normalise it to one number per layer:

```
    r = (E_BC_unsup − E_BC_sup) / (E_BC_linear − E_BC_sup)
```

`r = 0` means the unsupervised manifold matches the supervised reference exactly;
`r = 1` means it is no better than linear steering. **Success is `r < 0.5`**,
which is §3f's criterion restated.

**Bootstrap unit: compounds, not pairs.** Centroid pairs share compounds, so a
pair-level resample treats dependent observations as independent and understates
SE badly — which would make `2·SE_pooled` a threshold crossed by construction.
Resample the ten series members with replacement (1,000 draws), recompute all
three `E_BC` values and `r` within each draw, per series, per layer.

**Thresholds, both pre-declared:**

1. **Verdict flip.** If `r < 0.5` at one layer and `r ≥ 0.5` at the other, the
   result is layer-sensitive and **neither layer is *the* result**.
2. **Magnitude, absent a flip.** If `|r(8) − r(28)| > 2 × SE_pooled`, where
   `SE_pooled = sqrt(SE(8)² + SE(28)²)`, the result is layer-sensitive even
   though both layers agree on the verdict — reported as "same verdict,
   layer-dependent effect size".

If neither trigger fires, layers 8 and 28 agree and the Level 3 result is
reported as layer-robust — which is what §12a's plateau predicts should happen.

**The degenerate case, and it must fire on the bootstrap.** `r` is a ratio of
differences, so its bootstrap distribution goes heavy-tailed as
`E_BC_linear − E_BC_sup` approaches zero — the same failure the clause below
anticipates, but visible in the resamples before it is visible in the point
estimate. So we report **the fraction of bootstrap draws with
`E_BC_linear − E_BC_sup ≤ 0`**, and if that fraction **exceeds α = 0.05** we
declare the comparison degenerate — the task does not discriminate the
strategies — and report the three raw `E_BC` values without forming `r` at all.
Report the bootstrap **median and 2.5/97.5 percentiles** of `r` rather than
mean ± SE, since a heavy-tailed ratio has no useful mean.


### 12g. Orthography is a first-class confound (revised: pair set corrected)

**This reverses a claim in an earlier draft of §11c-bis** — that the grouping
variable is "chemistry, not orthography" — and then **withdraws two consequences
this section itself drew**, which were computed on the wrong pair set.

**The proxy comparison was uninterpretable.** Final token gives ρ = 0.556 and
functional class 0.439; the partitions differ in granularity, group count, size
distribution and within-group Tanimoto range, and range restriction moves ρ by
itself. Each partition against its *own* size-matched null:

| partition | ρ (uncensored) | null | **z** | groups |
|---|---|---|---|---|
| final token | 0.556 | 0.343 ± 0.022 | **9.5** | 9 |
| functional class | 0.439 | 0.343 ± 0.023 | **4.2** | 12 |

The orthographic partition is more than twice as far above its null, which is the
opposite of what the withdrawn claim asserted.

#### The pair set, and a reconciliation check

A first version of this section computed the partials over **all** pairs while the
headline Level 1 number is **uncensored-only**. That is not a detail:
ρ(Tanimoto, name-string) is **0.144 over all pairs but 0.190 uncensored**, because
censored pairs are the zero-overlap ones and are also more name-dissimilar. Every
number below is on the uncensored set.

The partial formula reproduces exactly from the three zero-order correlations at
every layer (`(r_gT − r_gN·r_TN) / sqrt((1−r_gN²)(1−r_TN²))`, checked to three
decimals, 32/32 layers), so the arithmetic closes. The apparent puzzle — a
"plateau" layer dropping to 0.08 under a ρ=0.144 correction — was not a partialling
artifact: the **all-pairs** raw correlation at layers 14–15 was genuinely low
(0.186, 0.138). The §3g plateau was established on the *uncensored* series, where
those layers read 0.275 and 0.251. Two different quantities were being compared.

#### What survives: the confound itself, and it is larger than first reported

| layer | geo~Tanimoto (raw) | geo~name (raw) | chem \| orth | orth \| chem |
|---|---|---|---|---|
| 0 | 0.139 | 0.389 | 0.073 | 0.373 |
| 5 | 0.333 | 0.328 | 0.292 | 0.286 |
| **8** | 0.345 | 0.339 | **0.304** | 0.297 |
| 15 | 0.251 | 0.527 | 0.182 | 0.505 |
| 24 | 0.328 | 0.430 | 0.278 | 0.397 |
| 28 | 0.322 | 0.427 | 0.271 | 0.394 |

**Orthography exceeds chemistry at 30 of 32 layers** (uncensored; the first draft
said 28/32 on the wrong pair set). Only layers 5 and 8 are exceptions, and there
the two are effectively tied (0.292 vs 0.286; 0.304 vs 0.297). Chemistry does
survive controlling for orthography, so the strong confound — "the recovered
geometry is just name similarity" — is false. But name-string similarity is a
contributor of comparable or greater magnitude at essentially every depth, and
`within_group_rsa` could never have bounded it. Layer 0 reads raw orthography
0.389 against raw chemistry 0.139, as a token embedding must, which is a sanity
check that the measure detects what it claims.

#### Withdrawn: the depth-structure consequences

Two claims made here on the all-pairs computation do **not** survive scoring on
the correct pair set by the same criterion §12b used:

| profile | prominence z | verdict |
|---|---|---|
| raw uncensored RSA (§12b, d=64) | 0.57 | no peak |
| **partial chemistry \| orthography, uncensored** | **0.77** | **no peak** |
| partial chemistry, all pairs (wrong set) | 2.16 | would have been a peak |

- **Withdrawn: "the verdict shifts toward Singh & Chopra."** Their
  rise-peak-attenuate pattern does *not* appear once the partial is computed on
  the headline pair set. The decontaminated profile is a plateau by the same
  noise-unit statistic that rejected a peak in the raw one.
- **Withdrawn: "this independently vindicates layer 8."** The partial profile has
  no peak to vindicate it with. Layer 8 is still the argmax (0.304) but layer 28
  is 0.271, a gap of 0.033 against a k-sweep sd of 0.0367. **The §12a tie-break
  stands on its own**, which is the correct and unglamorous outcome.
- **Corrected wording.** The earlier text described "a falling chemical term plus
  a rising orthographic one", which treats partials as additive components of the
  raw curve. They are not: the raw RSA is a mixture and the partial is chemistry's
  share of it. Both the raw and the partial profiles are plateaus; the plateau is
  **not** an artifact of the confound.

#### Level 2 and Level 3 are insulated, for one reason

Not because Level 3 is causal — a causal intervention can run along an
orthographic direction as easily as a chemical one. Both levels are insulated
because **their target variable is carbon count within a homologous series, and
name-string distance carries no information about it**:

| series | rho(name distance, abs carbon-count difference) | Level 2 abs-rho |
|---|---|---|
| alkane | −0.011 | 0.867 |
| alcohol | −0.011 | 0.976 |
| acid | −0.085 | 0.905 |
| alkene | −0.057 | 0.933 |

`meth/eth/prop/but/pent` are lexically unrelated, so ordinal recovery within a
series cannot be string similarity.

**Stated precisely:** this rules out *string similarity*, not lexical knowledge in
general. The model may well hold the series as a memorised ordered list. For Stage
A that is not a problem — the pipeline only requires the ground truth to be
present in the activations, not to have been derived chemically — but Level 2 must
therefore be described as recovering an **ordinal** structure, not a "chemical"
one.

#### Limitations

difflib ratio is a crude proxy for what the tokenizer and early layers encode; a
partial correlation removes only the component linear in ranks; these are summary
numbers over 218 compounds. The direction and size are stable across layers and
across the k grid, but "orthography ≈ 0.30 at layer 8" is an order of magnitude,
not a coefficient. This belongs prominently in the writeup's limitations per §10:
it is the largest threat to the Stage A instrument claim found so far, and it
lands on the headline Level 1 number rather than on a side control.

### 12h. Stage B inherits this, pre-registered now (amends §4d)

The refusal analog of name-string similarity is **prompt surface similarity**. If
refusal geometry tracks how the prompts are worded, the curvature and
intrinsic-dimension claims of §4d inherit exactly the problem §12g found in Stage
A — and finding that out after the fact a second time is not acceptable.

Added to the §3j/§4d null battery as a **mandatory nuisance-variable control**,
fixed before Stage B activations are examined:

1. Build `D_surface` over the ~600 prompts: normalised character edit distance
   and token-level Jaccard over n-grams (report both).
2. Report the raw Spearman of recovered geodesics against `D_surface`, and the
   **partial** against it controlling for the recovered refusal coordinate, and
   vice versa — the same two-way partial used in §12g.
3. Also control the obvious co-varying nuisances that differ by source split:
   **prompt length in tokens**, and **source identity** (harmful / harmless /
   XSTest), since the three corpora differ in register and length as well as in
   harmfulness.
4. §4e's behavioural correlation is reported **both** raw and partialled on
   `D_surface` and length. If the coordinate-to-`refusal_prob` relationship does
   not survive partialling, the honest reading is that the recovered axis is
   surface form or length, and §4e already commits to reporting that outcome
   as-is.

A curved refusal manifold that is really a prompt-phrasing manifold would be the
same result as §12g, one stage later, and the pre-registration is what prevents
it being discovered by a reviewer instead of by us.


---

## 13. Level 3 results and §12i — Sat 5 Sep 2026

### 13a. The precondition Level 3 needed and did not have

Level 3 compares how far three steering trajectories stray from the behaviour
manifold. **If the intervention has no behavioural effect, all three trajectories
return the *source* behaviour, which lies ON the manifold**, so every `E_BC` is
uniformly small, the strategies are indistinguishable, and `r → 0` — which looks
exactly like success while measuring nothing.

That is not hypothetical. The first Level 3 run reported `r = −0.04` (alkane) and
`r = −0.06` (alcohol) at layer 8 — an apparent clean pass of §3f's criterion. It
was spurious:

| layer | inject source centroid → mass on source name | inject **target** centroid → mass on **target** name |
|---|---|---|
| 8 | 0.478 | **0.000** |
| 28 | 0.509 | **0.448** |

At layer 8 replacement does not move behaviour at all. The tell was visible in
the numbers before the diagnostic: the denominator `E_lin − E_sup` was 0.065 at
layer 8 against 0.541 at layer 28, and the bootstrap CI for `r` spanned the whole
decision threshold ([−0.49, 0.80]). §12f's degenerate-case clause did not catch
it, because that clause asks whether the denominator is *positive*, not whether
the intervention *does anything*.

**Added as a hard precondition (§12i), checked and reported before `r` is
formed:** injecting the target centroid must place at least 25% of the source
injection's mass onto the target name. A layer failing this is reported as
**INERT** and its `r` is not interpreted.

### 13b. Causal accessibility is a step function at layer 22

Measured across all 32 layers, 6 centroid pairs, both series — probability mass
placed on the *target* name when the target centroid is injected:

| layers | alkane | alcohol |
|---|---|---|
| 0–15 | 0.000–0.001 | 0.000 |
| 16–21 | 0.005 → 0.067 | 0.001 → 0.019 |
| **22** | **0.371** | **0.286** |
| 23–31 | 0.442–0.569 | 0.454–0.511 |

Source-injection mass is flat at ≈0.44–0.58 everywhere, so this is not a general
failure of the intervention — replacement always reproduces the source behaviour.
What changes at layer 22 is whether a *different* point on the manifold can be
written in and read out downstream.

**This is the sharpest depth result in the project, and it dissociates cleanly
from Level 1.** Level-1 representational alignment is flat across layers 5–30
(§12b, prominence 0.50–1.00 at every projection dimension). Causal accessibility
is a step function at 22. **A layer can encode chemical similarity faithfully and
still be causally inert.**

Three consequences:

1. **§12a's tie-break selected a causally inert layer.** Layer 8 was chosen as
   the argmax of mean uncensored Level-1 RSA. That criterion measures
   representation and is silent about intervention. It is not wrong for Level 1,
   but it must not be used to select a layer for Level 3.
2. **§12a's plateau prediction resolves, and against the plateau.** It was
   pre-declared that if Level 1 is flat over 19 layers then Level 3 should be flat
   too, and that sharp layer-dependence would mean the plateau does not mean what
   §3g assumed. Level 3 is sharply layer-dependent. **The plateau does not mean
   what §3g assumed** — Level-1 flatness is not evidence that any layer will do.
3. **Wurgaft's layer 28 is vindicated for the causal test**, on grounds unrelated
   to their stated "late enough" rationale and unrelated to RSA: it is inside the
   causally live region, and layers chosen by representational alignment are not.
   Note this cuts against §12g's reading, where layer 28 looked *worse* because
   orthography dominates chemistry there. Representational quality and causal
   accessibility point in opposite directions across depth, and the honest
   statement is that no single layer is best for both.

### 13c. Level 3 result: the unsupervised pipeline fails the causal test

Run at every causally live layer plus layer 8 for the record, 20 centroid pairs,
K=50 waypoints, 8 Family-2 paraphrase base prompts, `r` bootstrapped over
compounds (SPEC §12f). Success is `r < 0.5`.

| series | layer | efficacy (tgt mass) | E_BC linear | E_BC unsup | E_BC **sup** | r | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| alkane | 8 | 0.000 | 0.444 | 0.377 | 0.379 | −0.03 | [−0.49, 0.83] | **INERT — not interpretable** |
| alkane | 22 | 0.255 | 1.422 | 1.983 | 1.231 | 3.94 | [1.51, 23.96] | degenerate (24% of draws) |
| alkane | 24 | 0.381 | 1.230 | 3.841 | 0.730 | 6.21 | [3.92, 10.10] | **fail** |
| alkane | 28 | 0.394 | 1.159 | 2.084 | 0.618 | 2.71 | [1.78, 3.91] | **fail** |
| alkane | 31 | 0.495 | 1.519 | 3.849 | 0.764 | 4.09 | [2.85, 5.90] | **fail** |
| alkane | 28 ← coord L8 | 0.394 | 1.159 | **1.092** | 0.618 | **0.88** | [0.47, 1.31] | **fail**, but closest |
| alcohol | 8 | 0.000 | 0.314 | 0.281 | 0.283 | −0.05 | [−2.53, 0.32] | **INERT — not interpretable** |
| alcohol | 22 | 0.249 | 1.699 | 2.731 | 1.948 | — | [1.81, 57.03] | degenerate (63% of draws) |
| alcohol | 24 | 0.470 | 1.526 | 5.137 | 0.817 | 6.09 | [4.62, 8.51] | **fail** |
| alcohol | 28 | 0.452 | 1.264 | 4.246 | 0.633 | 5.73 | [4.44, 7.24] | **fail** |
| alcohol | 31 | 0.466 | 1.381 | 3.222 | 0.645 | 3.50 | [1.76, 8.60] | **fail** |
| alcohol | 28 ← coord L8 | 0.452 | 1.264 | 3.197 | 0.633 | 4.07 | [2.74, 5.52] | **fail** |

**The headline result, stated plainly.**

1. **The supervised manifold works, and reproduces Wurgaft's direction.** At layer
   28 it beats linear steering by 1.87× (alkane, 0.618 vs 1.159) and 2.00×
   (alcohol, 0.633 vs 1.264). Wurgaft report ≈2.8×. The framework replicates.
2. **The unsupervised manifold fails, and fails *worse than linear*.** At every
   live layer and for both series, `E_BC(unsupervised) > E_BC(linear)`: the
   unsupervised coordinate is inaccurate enough that a spline parameterised by it
   is a worse path than a straight line. `r` is 2.7–6.2 with confidence intervals
   excluding 0.5 by a wide margin.
3. **The pre-registered §3f success criterion is not met.** The best case anywhere
   is the alkane hybrid at `r = 0.88`, CI [0.47, 1.31] — still a failure, though
   its lower bound touches the threshold.

**What does not explain it.** The obvious candidate — that the unsupervised
coordinate is simply too noisy — is not supported in the form one would expect.
Alcohol has the *best* Level-2 recovery anywhere (|ρ| = 0.976 at layer 8) and its
hybrid still fails at `r = 4.07`, while alkane's weaker coordinate (|ρ| = 0.867)
gives the best result in the table. Exact-position order agreement (0.0–0.6)
likewise does not order the results. We report the failure without a confident
mechanism rather than fit one after the fact; a spline is sensitive to the *local*
ordering in a way a rank correlation does not capture, but that is a hypothesis
for future work, not a finding here.

**Layer 22 is correctly flagged degenerate** (24% and 63% of bootstrap draws have
a non-positive denominator), which is what the §12f α=0.05 clause exists for. It
sits exactly at the efficacy threshold, where supervised and linear steering are
not yet distinguishable.

### 13d. What this means for the project's claim

Stage A was to validate an unsupervised pipeline before pointing it at refusal.
The verdict across the three levels:

| level | result |
|---|---|
| **Level 1** | Isomap 0.345 vs PCA 0.278 — the pipeline beats the baseline, but Procrustes disparity is 0.83 and roughly 0.24 of the pairwise signal is name-string similarity (§12g) |
| **Level 2** | 0.87–0.98 recovery, but the **PCA baseline matches it** (§12e); ordinal, not chemical (§12g) |
| **Level 3** | **fails** — the unsupervised manifold is worse than linear steering at every causally live layer |

**The honest summary is that the unsupervised pipeline is not validated.** It
recovers global structure better than PCA and ordinal structure as well as PCA,
but it does not produce a coordinate accurate enough to support the causal test
that was the point of building it. The supervised reference passing the same test
on the same data is what rules out a broken harness: the machinery works, the
unsupervised parameterisation is what fails.

**Consequence for Stage B, and it is a real one.** §0 framed Stage B as applying
"the validated pipeline" to refusal. The pipeline is not validated at Level 3, so
Stage B's §4f steering comparison inherits a known-failing component, and any
Stage B curvature result must be reported as coming from a pipeline that did not
pass its own causal test on ground truth. The §4d structure measurements
(intrinsic dimension, curvature, tangent alignment with a difference-in-means
direction) do not depend on Level 3 and remain interpretable; §4f does. This
should be stated in the abstract, not the limitations.
