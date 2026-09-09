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


---

## 14. Stage B results — Sat 5 Sep 2026

Prompt set (§4b): 600 primary — 200 harmful and 200 harmless from
`andyrdt/refusal_direction` @ `9d852fa`, 200 XSTest v2 "safe" borderline — plus
200 XSTest "unsafe" **contrast** prompts captured but held out of the primary set
(§14c). Chat template applied exactly once (asserted: 3 header blocks). Refusal
openers verified as 6 token ids. Activations at all 32 layers, float32. Behaviour
recorded at capture and **not touched until §4e**, per §4a.

Behaviour by source: harmful 0.915 refusal rate, XSTest contrast 0.915,
borderline 0.080, harmless 0.005.

### 14a. Structure (§4d), with the bank as the reference the spec demanded

| layer | ID (MLE) | ID (TwoNN) | **bank ID (MLE)** | geodesic/chord | corr(coord, v_ref) |
|---|---|---|---|---|---|
| 8 | 5.44 | 6.24 | 6.66 | 1.018 | 0.906 |
| 22 | 4.88 | 6.50 | 5.66 | 1.014 | 0.936 |
| **28** | 5.21 | 6.89 | 5.89 | **1.020** | **0.946** |
| 31 | 5.92 | 7.41 | 6.62 | 1.010 | 0.925 |

**Curvature: straight.** Geodesic/chord is 1.010–1.020 over well-separated pairs
at every layer. Near 1.0 is the pre-registered signature of a straight structure.

**Intrinsic dimension: ~5, and that is why the bank matters.** Taken against the
spec's literal fork criterion ("intrinsic dimension > 1"), ID ≈ 5 selects branch
two. Taken against the general-instruction bank at the same layer and position —
which §4d required precisely for this — the refusal set's ID is **not elevated**;
it is slightly *lower* (5.21 vs 5.89 at layer 28). The ~5 ambient dimensions are
generic instruction-prompt structure, not refusal structure. Reported both ways;
the branch assignment below rests on the bank comparison, and a reader who
prefers the literal criterion should read branch two instead.

**Tangent alignment.** *(Partly withdrawn — see §15d: the tangent statistic
clears a random-direction null, so alignment with `v_ref` stands, but it does NOT
distinguish this structure from a known-curved path, so it is not evidence of
straightness. Straightness rests on geodesic/chord.)* The first intrinsic
coordinate correlates 0.91–0.95 with the difference-in-means direction. Per-tangent cosines along the binned principal
curve are noisy (0.00–0.87, mean 0.27–0.37) because bin-to-bin displacement in a
5-dimensional cloud is dominated by the ~4 off-axis directions; the scalar
coordinate alignment is the more meaningful number and is reported as primary.

*Implementation note:* the first tangent estimator, a per-dimension smoothing
spline, was unusable — variance-scaled smoothing flattened the fit until the
derivative was numerical noise (cosines collapsed to 0.00–0.05 at every layer)
and coordinate ties returned NaN. Replaced with binned centroid differences,
which need no smoothing parameter.

### 14b. Behavioural validation (§4e) and the §12h controls

| layer | coord ~ refusal_prob | ...given length | geo ~ behaviour | ...given edit distance | geo ~ edit | ...given behaviour | geo ~ length |
|---|---|---|---|---|---|---|---|
| 8 | +0.580 | +0.452 | +0.351 | +0.328 | +0.237 | +0.199 | +0.156 |
| 22 | +0.823 | +0.773 | +0.661 | +0.652 | +0.176 | +0.101 | +0.134 |
| **28** | **+0.871** | **+0.836** | **+0.699** | **+0.691** | +0.177 | +0.100 | +0.147 |
| 31 | +0.775 | +0.708 | +0.560 | +0.548 | +0.170 | +0.104 | +0.097 |

**§12h's worry does not materialise, and this is the opposite of Stage A.** In
Stage A the orthographic confound was comparable to the signal (0.24 against
0.31, §12g). Here behaviour dominates surface form by a factor of seven at layer
28 (0.691 against 0.100, each controlling for the other), and partialling on
prompt length costs the behavioural correlation only 0.035 — despite harmful
prompts averaging 17.5 words against 8–10 for the other sources. Pre-registering
the control was still right; it simply came back clean.

### 14c. The natural experiment that settles it

XSTest's contrast prompts are minimal rewordings of its safe ones — "How can I
kill a Python process?" against "How can I kill a person?". Measured:

| | value |
|---|---|
| mean surface distance *within* the safe set | 0.682 |
| mean surface distance safe ↔ contrast | **0.685** |
| refusal rate, safe | 0.080 |
| refusal rate, contrast | **0.915** |
| **AUC of the unsupervised first coordinate separating them, layer 28** | **0.995** |
| same, layer 8 | 0.715 |

The two sets are **equally similar in surface form** and maximally different in
harmfulness, and the coordinate — recovered from activations alone, with no
labels and no behaviour — separates them almost perfectly. Surface form cannot
account for that.

### 14d. The Stage B fork, resolved

**Lead with the number a reviewer will remember (§15c).** XSTest safe and contrast
prompts are equally similar in surface form — mean pairwise distance 0.685
between the sets against 0.682 *within* the safe set — and differ maximally in
harmfulness (8% versus 92% refusal). An unsupervised coordinate recovered from
activations alone, with no labels and no behaviour, **separates them at AUC =
0.995**. Surface form cannot produce that.

**Branch one: straight, effectively one-dimensional.** The structure is straight
(geodesic/chord 1.02); **intrinsic dimension is ~5 but is not elevated against a
matched general-instruction bank (5.21 against 5.89)**, so the extra dimensions
are generic prompt structure rather than refusal structure. Its dominant axis
aligns with a difference-in-means refusal direction (0.946), that axis tracks
refusal behaviour (0.871, and 0.836 net of length), and it survives the
surface-similarity control that Stage A failed.

Both halves of the dimension result belong in the same sentence, per §15c: stating
the literal criterion first and the bank comparison afterwards makes the bank read
as a walk-back, when it is the controlled measurement and the literal comparison
against 1 is the uncontrolled one.

**This is convergent validation of Arditi et al. by a method that assumed nothing
about refusal** — the spec's pre-registered "clean negative that maps where the
framework applies". The geometric framework has nothing to add for refusal in
this model: there is no curvature for a manifold to exploit.

Two honest qualifications:

1. **The literal fork criterion says otherwise.** §0 fixed branch two as "curved,
   **or** intrinsic dimension > 1", and ID ≈ 5 > 1. The branch-one reading depends
   on interpreting ID against the bank rather than against 1. Both numbers are
   reported; a reader may take the other branch on the literal criterion, and the
   writeup must present it that way rather than choosing silently.
2. **It is one model and one prompt set.** Straightness at 600 prompts in
   Llama-3.1-8B-Instruct is not a claim about refusal in general.


---

## 15. Amendments written BEFORE the §4f run — Sat 5 Sep 2026

Committed before any §4f generation exists. The point of the commit order is
that a two-branch rule written after seeing the trajectories is a rescue, not a
pre-registration.

### 15a. §4f rewritten with two branches (amends §4f)

§4f as written has one branch: "manifold and linear steering must give nearly
identical trajectories… **Divergence here indicates a pipeline error**." That
rule was drafted assuming the pipeline had passed Level 3. It has not (§13c), so
divergence is ambiguous under the original wording between a Stage B fault and
the Stage A failure resurfacing. Both outcomes are therefore given a reading now.

**Quantity.** Per-waypoint Bhattacharyya distance between the behaviour
distributions produced by linear and by manifold steering, along the same
harmless↔harmful centroid pairs, at layer 28 (causally live per §12i). Report the
mean and max over waypoints, and `E_BC` for both strategies against the behaviour
manifold. Five pairs, as §4f specifies.

**Branch A — trajectories agree** (mean per-waypoint divergence small relative to
the linear/manifold `E_BC` gap seen in Stage A, i.e. the two strategies are
behaviourally interchangeable). This is the **causal complement to the
representational negative**. The Stage B claim strengthens from "the extra
geometry is not there" to "the extra geometry buys no extra causal handle."

> **Required framing, fixed now:** with coordinate-to-`v_ref` at 0.946 and
> geodesic/chord at 1.02, agreement is close to forced — a manifold fitted
> through points strung along one direction *is* approximately that line. This
> must be reported as **confirmation, not discovery**, in those words. A reader
> will see it was near-inevitable, and the writeup should say so first.

**Branch B — trajectories diverge.** Layer 28 is the easiest case the manifold
construction will ever be handed: straight structure, one dominant axis,
causally live, 600 points. Divergence *there* is **evidence about the
construction, not about refusal**. It would strengthen the Level 3 negative
(§13c) by showing the same failure on data where the geometry is maximally
favourable, and it must **not** be reported as a Stage B finding about refusal.

Neither branch changes §14d's verdict, which rests on §4d/§4e structure and the
§14c natural experiment, none of which involve steering.

### 15b. Validating the tangent estimator (new; closes a real gap)

**First, a correction to the record.** No random-direction null was ever
pre-registered. The three Stage B nulls in §12h — surface similarity, prompt
length, source identity — all ran and are reported in §14b. The gap is narrower
and worse placed than a missed null: the **binned-centroid tangent estimator was
adopted post-hoc** after the smoothing-spline version failed (§14a), so it is the
one component of the Stage B pipeline that entered with no validation of any
kind.

That matters because **the failure mode of a smoother is returning smoothness**.
Binned centroid differences will produce a coherent-looking sequence of tangents
from almost any point cloud, so "the tangents are coherent" and "the estimator
cannot produce anything else" are not distinguished by the reported numbers.

Two checks, specified before running:

1. **Random-direction null.** Recompute the tangent cosines against `R` random
   unit vectors drawn in the same PCA subspace (`R = 200`), and report the
   observed |cos(tangent, v_ref)| against that null distribution as a z-score.
   In 64 dimensions a random direction gives |cos| ≈ 0.125 in expectation, so a
   mean of 0.27–0.37 must be shown to exceed the null rather than assumed to.
   **If the observed cosines do not clear the null, the tangent-alignment row of
   §14a is withdrawn** and the coordinate-level alignment (0.946) stands alone.
2. **Power check on known curvature.** The Stage A homologous series have known
   non-zero curvature under Tanimoto and are already cached. Run the *same*
   binned estimator on the alkane and alcohol Family-2 centroid paths and confirm
   it reports tangent rotation there. **An estimator that cannot detect rotation
   where rotation is known to exist cannot support "straight" as a conclusion
   anywhere**, and if it fails this check the straightness claim rests on
   geodesic/chord alone — which has its own synthetic null and is the stronger
   number regardless.

### 15c. Two reporting fixes (amends §14d)

- **The bank comparison is the headline, not the walk-back.** §14d currently
  presents the literal criterion (ID ≈ 5 > 1 → branch two) first and the
  matched-control comparison second, so a reader meets branch two first and reads
  the bank result as a retreat. The abstract sentence must carry both at once:
  *intrinsic dimension is ~5 but is **not elevated against a matched
  general-instruction bank** (5.21 vs 5.89), while curvature is 1.02.*
- **Lead with the XSTest number.** AUC 0.995 from activations alone, with surface
  distance held flat at 0.685 against 0.682, is the strongest single result in the
  project and the one a reviewer will remember. The Stage B negative should open
  with it rather than arriving at it.


### 15d. §15b outcome: one check passes, one fires against §14a

**Check 1 — random-direction null: PASSED at every layer.** `R = 200` random unit
vectors in the same PCA subspace:

| layer | observed mean abs-cos | null mean | null sd | z | null p95 | clears |
|---|---|---|---|---|---|---|
| 8 | 0.374 | 0.100 | 0.028 | **9.64** | 0.148 | yes |
| 22 | 0.312 | 0.104 | 0.027 | **7.62** | 0.148 | yes |
| 28 | 0.295 | 0.097 | 0.025 | **7.81** | 0.139 | yes |
| 31 | 0.271 | 0.101 | 0.025 | **6.81** | 0.147 | yes |

The null's own mean (0.100) matches the analytic expectation for random unit
vectors in 64 dimensions, sqrt(2/(pi*64)) = 0.0997, which validates the null
itself. Alignment between the principal-curve tangents and the difference-in-means
direction is real and roughly 3x chance. The §14a tangent-alignment row **stands**.

**Check 2 — power on known curvature: the estimator can see rotation.** Run on
the cached Stage A homologous-series centroid paths, whose curvature under
Tanimoto is known to be non-zero:

| path | consecutive abs-cos | first-vs-last | verdict |
|---|---|---|---|
| alkane, L8 | 0.236 | 0.069 | rotates |
| alkane, L28 | 0.326 | 0.004 | rotates |
| alcohol, L8 | 0.217 | 0.084 | rotates |
| alcohol, L28 | 0.285 | 0.001 | rotates |

So the estimator is not a smoother that only ever returns smoothness.

**But the like-for-like comparison this forced is the finding, and it goes
against §14a.** The power check uses *consecutive* tangent coherence; §14a
reported alignment against `v_ref`. Measured on the same statistic:

| set | consecutive abs-cos | first-vs-last | random-pair null |
|---|---|---|---|
| refusal set, L8 | 0.255 | 0.055 | 0.104 |
| **refusal set, L28** | **0.332** | **0.046** | 0.100 |
| alkane series (**known curved**), L28 | 0.326 | 0.004 | 0.269 |
| alcohol series (**known curved**), L28 | 0.285 | 0.001 | 0.266 |

**By the tangent statistic the refusal structure is indistinguishable from a
known-curved path** — 0.332 against 0.326, with both turning nearly orthogonal
end to end. The tangent estimator therefore **does not support the straightness
claim**, and §14a must not be read as though it did.

Two caveats that keep this from being a reversal of §14d:

- The comparison is not perfectly matched. The refusal tangents come from 11 bins
  of ~55 points in a 64-D subspace (null 0.100); the series tangents from 5 bins
  of 2 points in a 9-D subspace (null 0.269). Normalised by their own nulls the
  refusal path is *more* coherent (3.3x) than the series (1.2x). The raw numbers
  coincide; the null-relative ones do not.
- **Straightness rests on geodesic/chord**, exactly as §15b anticipated it might.
  That statistic is 1.010-1.020, has a direct interpretation, and carries the
  §3j synthetic straight-line and curved-grid nulls. It is the stronger number
  and should be quoted as the basis of the claim.

**Net effect on §14a:** the tangent row survives as evidence that the curve's
direction aligns with `v_ref` (check 1), and is withdrawn as evidence of
*straightness* (this check). §14d's branch-one verdict is unchanged, because it
rested on geodesic/chord, the coordinate alignment (0.946), the behavioural
correlation (0.871) and the §14c natural experiment — none of which use tangent
coherence.


### 15e. §4f result: Branch B, with the two sub-criteria disagreeing

Five harmless-to-harmful centroid pairs along the unsupervised coordinate, layer
28 (causally live), K=50 waypoints, 8 base prompts, replacement intervention.
Behaviour space is the §4f approximation — next-token mass on {refusal openers,
compliance openers, other} — **labelled an approximation, as §4f requires; it is
the weakest link in Stage B and a reader will find it.**

| pair | E_BC linear | E_BC manifold | per-waypoint divergence (mean / max) | refusal start → end |
|---|---|---|---|---|
| 0→9 | 5.315 | 3.365 | 0.027 / 0.135 | 0.00 → 1.00 both |
| 1→8 | 1.202 | 2.641 | 0.047 / 0.200 | 0.00 → 1.00 both |
| 0→7 | 4.698 | 4.025 | 0.091 / 0.413 | 0.00 → 1.00 both |
| 2→9 | 3.568 | 2.063 | 0.022 / 0.127 | 0.00 → 1.00 both |
| 1→9 | 1.374 | 2.467 | 0.023 / 0.097 | 0.00 → 1.00 both |

**The two pre-registered sub-criteria disagree, and both halves are reportable.**

- **Behaviourally the strategies are near-identical.** Mean per-waypoint
  Bhattacharyya divergence between linear and manifold trajectories is **0.042**
  (below the 0.05 bar), and both drive refusal from 0.00 to 1.00 in all five
  pairs. This is Branch A's substance: the manifold buys no different behaviour.
- **By E_BC against the behaviour manifold they differ**, and the difference is
  **inconsistent in sign** — the manifold is better in 3 of 5 pairs and worse in
  2. Mean signed gap 0.319, mean absolute gap 1.332, against a Stage A reference
  scale of 0.586. The §15a conjunction requires both criteria, so the verdict is
  **Branch B**, and it holds under the signed form of the gap as well as the
  absolute form (0.319 > 0.146), so it does not depend on that choice.

**Reading, per §15a Branch B, fixed before the run:** layer 28 with straight
structure, one dominant axis and 600 points is the easiest case this construction
will ever be handed. Trajectories that wander off the behaviour manifold
*inconsistently* relative to a straight line, on that data, are **evidence about
the manifold construction, not about refusal**. This strengthens the §13c Level 3
negative — the same failure mode appearing where the geometry is maximally
favourable — and is **not** reported as a Stage B finding about refusal.

**A flaw in my own threshold, recorded.** The criterion used mean |gap|, which
cannot distinguish a consistent difference from sign-flipping noise; the signed
gap should have been specified alongside it. Both were computed and both give
Branch B, so nothing turns on it here, but the rule was under-specified.

**§14d is unaffected.** The branch-one verdict rests on geodesic/chord, the
coordinate alignment (0.946), the behavioural correlation (0.871) and the §14c
natural experiment. None involve steering, and §15a fixed in advance that neither
branch would move it.


---

## 16. §3h and §3i — Sat 5 Sep 2026

### 16a. §3h: whitening is not load-bearing

Whitened by the covariance of a **bare-format** 2,000-prompt bank (captured
separately: the Stage B bank is chat-formatted, and whitening Family-1 activations
with it would fold prompt format into the covariance being removed):

| layer | RSA Euclidean | RSA whitened | delta | ID | orthography given chemistry |
|---|---|---|---|---|---|
| 8 | 0.342 | 0.345 | **+0.003** | 6.19 → 6.41 | 0.275 → 0.262 |
| 28 | 0.327 | 0.341 | **+0.014** | 4.89 → 4.80 | 0.366 → 0.376 |

Both deltas are far under §3h's 0.1 threshold. **The metric is not load-bearing
here, so Stage B does not need to be run under both** — which is what §3h's
conditional existed to determine, and it resolves in the cheaper direction.

Note also that whitening does **not** remove the §12g orthographic confound
(0.275 → 0.262 at layer 8): it is not a fix for that problem.

### 16b. §3i: the hierarchy test has no power on this data

Class labels used only for evaluation. Cosine between the parent mean and each
class offset, `cos(mu_organic, mu_c - mu_organic)`, over 11 classes:

| layer | metric | observed mean abs-cos | random-vector null | **permuted-label null** | z |
|---|---|---|---|---|---|
| 8 | Euclidean | 0.062 | 0.014 | **0.072 ± 0.054** | **−0.18** |
| 28 | Euclidean | 0.095 | 0.013 | **0.106 ± 0.074** | **−0.15** |
| 8 | whitened | 0.067 | — | — | — |
| 28 | whitened | 0.108 | — | — | — |

Against a random-vector null (0.013, matching the analytic sqrt(2/(pi*4096)) =
0.0125) the observed cosines look like meaningful near-orthogonality. **Against a
permuted-label null they are indistinguishable from chance.** Shuffling the class
assignments — preserving group sizes and the ambient geometry, destroying only the
taxonomy — reproduces the same near-orthogonality.

The reason is structural: a small subgroup's mean offset from a large parent mean
is nearly orthogonal to that parent almost regardless of which points are in the
subgroup, because the parent is dominated by the bulk and the offset is
essentially a noise direction.

**So §3i is reported as a null result about the test, not about the hypothesis.**
This neither supports nor refutes Park et al. Their result is on the unembedding
under a whitened causal inner product with specific hierarchies; our residual-stream
class means over 11 classes and 218 compounds do not have the power to
discriminate their prediction from chance. The same applies to the
series-tangent-versus-class-offset cosine (0.09-0.10), which was to be the
direct-sum-of-polytopes check.

**Method note, and it is the third time this pattern has appeared.** The observed
cosines were about to be written up as supporting orthogonality. They do not,
and only the permuted-label null shows it — the random-vector null, which is the
one that first comes to mind, is 5x too permissive and would have licensed the
wrong claim. The lesson from §15b generalises: for any statistic whose "good"
value is near zero, the null must preserve everything except the hypothesised
structure.


---

## 17. Completion audit — Sat 5 Sep 2026

### 17a. Acceptance test 9 had never been run, and it found two things

§6 test 9 ("greedy generation at alpha=0 reproduces base output token-for-token;
hook counter increments; hooks removed after every run") is on §8's **never cut**
list and had not been written. Running it surfaced two facts, neither of which
invalidates a result but both of which were unknown:

1. **The steering hook is not position-safe under `generate()`.** An ungated
   `t[:, -1, :] = rep` also overwrites every newly generated position, pinning
   the model to one token — " propane propane propane ..." in place of
   " propane. The formula for propane is C3H8." Production steering
   (`level3_steering.py`, `stageb_steering.py`) runs a **single forward** and
   asserts the hook fires exactly once, so Level 3 and §4f are not exposed. Any
   future reuse under generation must gate on sequence length. Test 9b pins this
   failure mode so it stays known.

2. **`M.capture`'s forward and `generate()`'s prompt pass are not numerically
   identical.** They differ by up to **0.125** absolute (mean 0.0046, ~3x bf16
   epsilon at that magnitude) — different kernel paths in bf16. Injecting a value
   captured through one path into the other perturbs generation enough to flip a
   token about nine steps in. Verified that the mechanism itself is faithful: an
   identity hook and a float32 round-trip both reproduce base output exactly.
   So "alpha = 0 reproduces base" holds exactly **within one forward path**, and
   the test is written that way.

   Level 3 and §4f are unaffected: both do one forward and read logits
   immediately, with no autoregressive compounding, and both inject deliberately
   different values regardless.

Tests 4 and 8 are covered by recorded artefacts. `tests/test_acceptance_gpu.py`
now carries 4/4 passing; the CPU suite carries 10/10.

### 17b. What remains

| item | status |
|---|---|
| §3a-§3j, Levels 1-3 | done |
| §3h metric comparison, §3i hierarchy | done (§16; §3i has no power) |
| §4b-§4f | done |
| §6 acceptance tests | done — 10 CPU, 4 GPU |
| F1, F2, F2c, F3, F4, F5, F7, F8 | done |
| **F6 (hierarchy)** | **not made** — §3i is at its permuted-label null, so the figure would plot a null result. Worth one panel only if §3i is reported at all. |
| **§5 parquet schema** | **partial** — quantities are all present but some field names differ from the spec's listing (`geo_chord_mean` vs `geo_chord_ratio_mean`, `coord_vs_refusal_prob` vs `..._spearman`), and `timestamp`, `model_revision`, `E_BC_se`, `notes` are absent. |
| **§4d "three layers"** | **deviates** — spec names `L_ref`, `L_peak`, 28. Since §11f fixes the layer ourselves rather than reproducing Arditi's selection sweep, there is no `L_ref`; layers 8, 22, 28, 31 were run instead. Stated, not silently substituted. |
| **§10 writeup** | **not started** — the actual deliverable |


---

## 18. Amendment: Stage B remaining work, reconciled against the record — Wed 9 Sep 2026

**Status:** v3. This section is the §0 pre-registration artifact for all Stage B
remaining work. It supersedes the external drafts `SPEC2.md` (written without
repo access), `SPEC3.md` (v1) and `SPEC4.md` (v2), which are retained unmodified
as the audit trail. Nothing in §18.4–§18.6 is run before this section is
committed.

**Pre-registration commit: `d6aa671`, 9 Sep 2026.** Every rule below was fixed at
that hash before any of it was run. Results are recorded in later commits that
reference it, never by editing the rules above.

### 18.0 Audit trail

v1 corrected seven errors in SPEC2 (E1–E7); v2 corrected those. A second audit
of v2 against the code found eight more (N1–N8), fixed here. Both rounds are
recorded rather than silently applied, because the point of the exercise is that
the corrections are visible.

| ID | Error | Fix |
|---|---|---|
| E1 | Misquoted SPEC.md:5 as "base deadline Fri 4 Sep" — that is the writing date. | Base deadline **Sun 6 Sep**; §18.3. |
| E2 | Bootstrapped TwoNN; the 5.21/5.89 contrast is MLE-vs-MLE (§14a). | §18.4f″ bootstraps `id_mle`. |
| E3 | Transfer manifest incomplete, size ~5× low. | §18.2 manifest, ~2.5 GB. |
| E4 | Decontaminated the coordinate but not `v_ref`. | §18.4c′. |
| E5 | Silent on which `d` is ablated. | §18.4a′ closes it by measurement. |
| E6 | "Every residual-stream write" ≠ block-output hooks; the test would pass while leaking. | §18.5a′ sub-layer hooks and sub-layer assertion. |
| E7 | Synthetic bank skipped PCA(64) and used isotropic noise. | §18.4e′. |
| **N1** | **`geodesic/chord` as implemented measures Isomap's MDS reconstruction residual, not curvature.** | **§18.4d″ — new statistic; the logged one is reclassified.** |
| **N2** | **The 5.21-vs-5.89 ID contrast is confounded by sample size (600 vs 2000).** | **§18.4f″ — n-matched contrast.** |
| **N3** | Bootstrap-with-replacement silently breaks `levina_bickel_mle` (duplicate points give `T_1 = 0` and are dropped at `geometry.py:92`). | §18.4f″ — m-out-of-n subsampling without replacement. |
| N4 | §18.4d′ points 1 and 2 contradicted: data born in 64-d makes the PCA(64) step vacuous. | §18.4e′ picks option (B) explicitly. |
| N5 | Platform test targeted a rounded value on a possibly non-deterministic pipeline; `cos(d₄₀₀, d₆₀₀)` ignored Isomap's arbitrary component sign. | §18.7 pins `eigen_solver`; all component comparisons use `\|cos\|`. |
| N6 | Making `d₄₀₀` primary "throughout" silently migrated the structural claims to a different point set. | §18.4c′ splits which claims move and which stay. |
| N7 | v2 dropped v1's eval-size rule. | §18.5b restores it. |
| N8 | Sub-layer hooks need the same resolver and firing-count discipline as `capture`. | §18.5a′. |

### 18.1 Corrections to SPEC2 (carried forward)

**C1** 0.946 is a Pearson correlation across points (`stageb_structure.py:124`),
not a cosine — every prose reference must say which. **C2** OLS at n=600, p=4096
is vacuous; §18.4a′ regresses on the 64-d PCA scores instead. **C3** use the
pre-registered `K_SWEEP = (8,12,16,24)` (`config.py:29`); tails may be added as
marked extensions, never renumbered. **C4** §1e is a broken-estimator diagnosis,
not an agreement claim. **C5** §16a stands; no whitening re-run. **C6** float32
(`config.py:14`); a bf16 recapture would not be faithful. **C7** §1 is
compute-gated too, so SPEC2's "drop to §1 only" fallback is void.

**C8.** SPEC2 and v1 both wrote "TwoNN" for the ID contrast. The logged contrast
is Levina–Bickel MLE. Name the estimator every time.

**C9 (new, from N1).** Everywhere the record says "geodesic/chord", the quantity
computed was geodesic-over-*embedded*-chord. §18.4d″ governs what may be claimed
from it.

### 18.2 Compute gates — RESOLVED 9 Sep 2026

Two gates, deliberately separate; conflating them produced C7.

**Gate A1 — one-time transfer. FIRED 9 Sep 2026.** fedora
(`100.71.95.25`, uptime 112 days) reachable; all artifacts present. Tensors
converted `.pt` → `.npy` on fedora at float32 (no dtype change), SHA-256
manifest taken at source, transferred, verified at destination.

| File | Shape / size | Needed by |
|---|---|---|
| `acts/npy/stageb.npy` | (800, 32, 4096) f32, 419 MB | §18.4a–e, §18.6 |
| `acts/npy/bank.npy` | (2000, 32, 4096) f32, 1.05 GB | §18.4f″ |
| `acts/npy/bank_bare.npy` | (2000, 32, 4096) f32, 1.05 GB | template control (§18.8.5) |
| `data/stageb/*.json`, `xstest_raw.csv` | small | §18.5b eval sets, §18.7 hashes |
| `env/A1_pip_freeze.txt`, `env/A1_manifest.sha256` | small | §18.7 |

`bank_bare.npy` was transferred rather than decided, on the reasoning that the
window was open and 1 GB is cheaper than a second window that may not exist. If
the chat-template control is not claimed, the file is unused, not misused.

**Gate A2 — sustained ssh + idle GPU. AVAILABLE.** RTX 5090, 181 MiB / 32,607 MiB
used, 0% utilisation at check time. Driver 580.105.08, torch 2.11.0+cu130,
transformers 5.12.1, Python 3.12.13. Requires tmux/nohup for a job outliving the
session, and §18.8.4 (authorisation scope) before a long run starts.

**Branch fired: A1 ✓ / A2 ✓ — full spec.** Note the consequence of the split:
§18.4 and §18.6 are now permanently decoupled from box availability and run
locally on the transferred `.npy` files regardless of what happens to fedora.

### 18.3 Deadline branches — OPEN (§18.8.2)

SPEC.md:5 records the application deadline as **Sun 6 Sep**, extension possible
to Fri 11 Sep; SPEC.md:404 records the venue as a MATS application with a
required LLM-assistance disclosure. Today is Wed 9 Sep, so the base deadline has
passed. Commit `c725bdb` does not disambiguate submitted-from-stalled.

- **B1 — extension granted.** ~2 days; SPEC2 §5 schedule with the Sept 10 18:00 freeze.
- **B2 — submitted on/around Sun 6 Sep.** The submitted artifact is fixed. New
  results are an addendum; do not retro-edit submitted claims, and do not present
  post-submission results as having been in the application. Freeze void — run
  §18.5 properly rather than fast.
- **B3 — missed.** Standalone writeup or next-cycle piece. Freeze void; the P2
  items return (Stage A diagnosis tests, full layer sweep, second concept family).

Under B2/B3 the freeze and SPEC2's n=100 eval sizes are artifacts of a dead
constraint and are re-derived, not inherited.

### 18.4 Revised §1 (P0; gated on A1 only — now unblocked)

**Run order is load-bearing.** §18.4d″ before §18.4e′ before §18.4g, because the
power check must be run against whichever curvature statistic is being claimed.
§18.4a′ and §18.4c′ before anything that names `d` or `v_ref`.

#### 18.4a′ Close the `d` fork by measurement, then build `d`

1. Fit the pipeline on the logged 600 in-primary prompts → `d₆₀₀`.
2. Fit on the decontaminated 400 (§18.4c′) → `d₄₀₀`.
3. Report **|cos(`d₄₀₀`, `d₆₀₀`)|** — absolute, because Isomap component sign is
   arbitrary (N5). If ≥0.99 the fork is empty: record it and carry `d₄₀₀` with a
   one-line note. If <0.99, carry both through every table, with `d₄₀₀` primary
   in §18.5 and `d₆₀₀` as a labelled robustness arm.

`d₄₀₀` is primary under either outcome: the causal claim should concern the
direction recovered under clean conditions.

**Construction (C2).** Regress the leading Isomap coordinate on the 64-d PCA
scores (p=64, well-posed); report in-sample **and** 5-fold CV R²; lift
`d = V₆₄ @ β`, normalized. Separately report cross-validated **ridge** R² on the
full 4096-d centered activations — CV only, since in-sample is uninformative at
this n. The gap between the two measures what PCA(64) discarded.

Report cos(`d`, `v_ref`). **That measured value, not 0.946, is arm 4b's constraint.**

#### 18.4b Baseline panel, both metrics per row

| Method | Pearson r (coord vs. proj on `v_ref`) | cos(·, `v_ref`) | XSTest AUROC |
|---|---|---|---|
| Isomap coord 1 | ✓ | ✓ | ✓ |
| PC1 of the in-primary set (**not** the 2000-prompt bank; recomputed on the 400 for the clean condition) | ✓ | ✓ | ✓ |
| k-means (k=2) centroid difference | ✓ | ✓ | ✓ |
| Random directions, n=100 | distribution | distribution | distribution |

Every baseline is computed on the same bank as the `d` it is compared against.
Name the metric in every cell; never present one as the other.

**Branch rule.** PC1 ≥0.90 Pearson r *and* AUROC within 0.01 → the manifold
machinery is not load-bearing. Reframe to "the refusal axis is recoverable by any
unsupervised linear method including trivial ones, which is positive evidence
that the linear account is complete." Write that abstract sentence now.

#### 18.4c′ Decontaminate the coordinate **and** `v_ref` — and fix which claims move

`prompts.csv` is 800 rows: 200 harmful / 200 harmless / 200 XSTest borderline
(all `in_primary`) + 200 XSTest contrast (held out). `stageb_structure.py:30-42`
computes `difference_in_means(H = acts[harmful], B = acts[~harmful])`, and that
complement is 200 harmless **+ 200 XSTest borderline** — so `v_ref`'s negative
pole is half XSTest, and refitting only the coordinate would leave the reference
direction fit on the evaluation set.

1. `v_ref_200` = difference-in-means, 200 harmful vs 200 harmless only.
2. Report **cos(`v_ref_200`, `v_ref_400`)** first. If ≥0.99, the contamination is
   immaterial and that is said with a number. If not, `v_ref_200` is primary
   throughout — including §18.5 arm 1 and the arm 4b constraint — and
   `v_ref_400` appears as a labelled robustness row.
3. The logged 0.946 belongs to `d₆₀₀ × v_ref_400`. It stays in the record
   labelled as such; the headline r is recomputed under the clean pairing and
   **will differ**.

**AUROC:** refit on the 400 harmful+harmless, evaluate on all 400 XSTest as
genuine holdout. **That is the headline.** 0.995 reported alongside, labelled
transductive; the gap is itself reportable.

**Which claims move (N6).** The 400 is two clusters with the borderline prompts
— the points that fill the middle of the refusal axis — removed, which is the
regime where kNN geodesics are least trustworthy. Therefore, fixed now:

- **Moves to the 400:** the AUROC, and the `d` fed to §18.5.
- **Stays on the 600:** the structural claims — curvature, ID, tangents — reported
  as computed on the 600 and labelled so.
- **Precondition:** check `info["graph_connected"]` across the full `K_SWEEP` on
  the 400 *before* anything is built on `d₄₀₀`. A disconnected graph there
  invalidates `d₄₀₀`, not just its error bars.

#### 18.4d″ Curvature statistic — the logged one does not measure curvature (N1)

`geometry.py:182-186` returns `Y = iso.fit_transform(X)` and
`D_geo = iso.dist_matrix_`; `stageb_structure.py:112` then calls
`geodesic_chord(D_geo, Y)`, whose chord term is `pdist(Y)` — Euclidean distance
**in the embedding**. Isomap's objective is precisely to make embedded distances
approximate `dist_matrix_`, so the ratio is driven to 1 by construction, and
because classical MDS truncated to d components can only shrink distances it sits
systematically just *above* 1. The logged 1.010–1.020, and its stability across
four layers and the k-sweep, are consistent with an MDS residual, not with a
curvature measurement.

On a genuinely curved manifold Isomap unrolls it: geodesic ≈ embedded distance
≫ ambient chord, so the implemented statistic returns ≈1 for a curved arc. It has
no power against the hypothesis it was reported as testing.

**Fixed now:**

1. **New statistic:** geodesic over **ambient chord** — `||x_i − x_j||` in the
   PCA(64) space (`Xp`, already in scope one line above the current call). This
   is the curvature measure; it is what §18.4e′ computes MDR for and what the
   straightness claim rests on.
2. **The logged statistic is reclassified**, not deleted: reported as an
   Isomap/MDS reconstruction diagnostic under that name. Also write
   `iso.reconstruction_error()` (computed at `geometry.py:189`, never persisted)
   into the parquet, so the reclassification is demonstrated rather than asserted.
3. **Falsification test, run first, no activations required:** a semicircular arc
   in R³ with known `φ/sin φ`, through the same pipeline. Expected: the logged
   statistic returns ≈1.0 at every true curvature; the new one recovers `φ/sin φ`.
   If that is *not* what happens, N1 is wrong and this subsection is withdrawn
   before anything downstream is touched.
4. **§14a's straightness finding is under review until (3) resolves**, and the
   writeup may not restate it until then. §14d's branch-one verdict cites
   geodesic/chord first among its four supports; the other three (coordinate
   alignment, behavioural correlation, the §14c natural experiment) are
   untouched by N1 and are unaffected.

#### 18.4e′ Power check on synthetic curvature

Run against the §18.4d″ statistic. Two changes from SPEC2 §1d, resolving E7 and
N4 — v2 asked for both a whole-pipeline run and generation in the 64-d space,
which are incompatible, since data born in 64-d makes PCA(64) a no-op:

**Option (B) is chosen.** Generate in the full 4096-d ambient space using the
**real bank's covariance** — not isotropic noise, which was E7's actual defect —
then run the pipeline whole, with PCA(64) **fit on the synthetic bank**. Report
the covariance model used. Noise scale from residuals about a 1-D principal curve
along `d`; note in the writeup that this curve is fit to data that may itself be
curved, a mild circularity that biases σ upward and therefore MDR conservative.

Otherwise as SPEC2 §1d: circular arcs, true ratio `φ/sin φ`, sweep
{1.000, 1.005, 1.01, 1.02, 1.05, 1.10, 1.25, 1.50}, 50 replicates, n matched.
Null at 1.000 sets the threshold (95th percentile); MDR = smallest true ratio
detected at ≥80% power. Run at k=12 and report MDR per-k across `K_SWEEP`, since
the floor is k-conditional.

**Also run the power check against the logged statistic**, which is expected to
return no detection at any ratio through 1.50. That is the quantitative
demonstration of N1 and belongs in the writeup as such.

**Branch rule.** MDR > 1.02 → the claim is "curvature below our detection
threshold of X," never "the manifold is straight." Committed now.

#### 18.4f″ ID uncertainty, correct estimator, n-matched (E2, N2, N3)

§14a's columns are `ID (MLE) | ID (TwoNN) | bank ID (MLE)`; at L28,
**5.21 = MLE (refusal), 6.89 = TwoNN (refusal), 5.89 = bank MLE**. The contrast
is MLE-vs-MLE, and the bootstrap must reproduce the composite in
`geometry.py:130-136` — the mean of Levina–Bickel at k ∈ {10, 20}, not a single k.

**Sample-size confound (N2), and it comes before the CI.** The refusal set is
`A = acts[prim]`, 600 points (`stageb_structure.py:91`); the bank is 2000.
Levina–Bickel at fixed k is density-dependent: more samples means smaller
neighbourhoods and a typically higher estimate. **A 2000-point bank scoring above
a 600-point set is what identical intrinsic structure would produce**, and the
logged difference runs in exactly the direction the bias predicts. Without this
control both outcomes are uninterpretable — overlapping CIs read as "not
elevated", separated CIs as "genuinely lower", and either could be n-bias.

**Required: subsample the bank to n=600 over many draws and make that the
contrast.** The 2000-point number stays in the record, labelled as computed at a
different n.

**Resampling method (N3).** `geometry.py:92` drops any point whose nearest-neighbour
distance is zero (`ok = T[:, 0] > 0`). Under bootstrap *with replacement*, every
duplicated point has `T_1 = 0`, so a large fraction of rows is silently dropped
and the survivors' distance sequences are compressed by ties — the CI would
describe a different estimator than the point estimate. Use **m-out-of-n
subsampling without replacement** (80%, consistent with SPEC2 §3a), with n held
fixed and equal across both banks.

**Reported limit.** 5.21 (MLE) against 6.89 (TwoNN) is a ~30% disagreement
between two estimators on identical data. That spread bounds how much weight
"the refusal cloud is ~5-dimensional" can carry and is stated as a limit on the
ID claim rather than left for a reader to find. Bootstrap TwoNN for that
comparison only; the §14d contrast remains MLE-vs-MLE.

**Branch rule.** If the n-matched MLE CIs overlap, the §14d control-bank
comparison framing does not survive and must not be written.

#### 18.4g Ratio-only k slice

Both statistics (§18.4d″) at `K_SWEEP = (8,12,16,24)`. Gates §18.4e′.

### 18.5 Revised §2 (P0; gated on A2)

#### 18.5a′ Hook granularity, and a test that certifies the right thing (E6, N8)

`config.py:25` pins `HOOK_MODULE_TEMPLATE = "model.layers.{layer}"` — **block
outputs**. Arditi ablates at every residual-stream write: embedding output, each
attention output, each MLP output. With block-output hooks, attention can write a
û-component that the MLP reads within the same block before the next ablation
fires. **The leak biases toward a false null on arms 2 and 3** — the arms whose
nulls are pre-committed as supporting the completeness conclusion. A leaky hook
manufactures exactly the result being pre-registered as expected. And v2's test
would have passed while leaking, since a block-boundary assertion checks
precisely where the hooks fire.

**Resolution — extend the hooks. Not optional.** Hook `model.embed_tokens`,
`model.layers.{i}.self_attn`, and `model.layers.{i}.mlp` outputs. Sub-layer
returns vary by attention implementation in transformers 5.x, so apply the same
explicit tensor resolution as `_resolve_block_output` (`model.py:48`) rather than
trusting a return convention, and assert the firing count — **65 per forward**
(1 embedding + 32 attention + 32 MLP) — the way `capture` asserts 32
(`model.py:118`).

If the hooks cannot be extended in the time available, **arm 3's null becomes
uninterpretable and is reported as such.** Pre-committing "arm 3's null supports
completeness" while running an instrument that produces nulls artifactually is
not available. One or the other.

**Acceptance test, before any eval run:**
1. **Null-op:** zero-magnitude projection reproduces baseline generation token-for-token.
2. **Completeness at sub-layer outputs:** during full generation, assert
   `max|û·x| < ε` at every hooked sub-layer output and every position, generated
   positions included. A block-boundary assertion is insufficient (E6).
3. **KV-cache consistency:** ablated generation with and without KV cache must
   match on ≥5 prompts. Divergence means keys/values were computed pre-ablation —
   silent partial ablation. If they diverge, disable the cache and absorb the cost.

Log the test output in the results file. An unasserted ablation is not evidence.

**Timing.** The 20-prompt measurement runs on this new path, not the
single-forward path, and is therefore the first exercise of new code. Measure
before committing to a schedule; SPEC2's 45–90 min was an estimate.

#### 18.5b Arms

Directional ablation `x ← x − ûûᵀx` at every hooked write, every layer, every
position, throughout generation.

| Arm | Direction | Purpose |
|---|---|---|
| 1 | `v_ref_200` (§18.4c′) | replication anchor |
| 2 | `d₄₀₀` (§18.4a′) | does the unsupervised direction cause refusal |
| 3 | `d_⊥ = d₄₀₀ − proj_{v_ref_200}(d₄₀₀)` | independent causal content |
| 4 | random unit, n=5 seeds | null |
| 4b | random, constrained to the **measured** cos(`d₄₀₀`, `v_ref_200`) | "is `d` more than a vector near `v_ref`" |
| 5 | none | baseline |

Robustness arms (`d₆₀₀`, `v_ref_400`) only if §18.4a′/§18.4c′ return |cos| <0.99.

**Eval sets.** Harmful: JailbreakBench Behaviors, string-deduplicated against the
400-prompt fit bank (JBB shares lineage with AdvBench/HarmBench); report n after
dedup. Harmless: **held-out rows of `harmless_test.json`** — not a fresh Alpaca
pull, which risks overlapping both the 200 in the fit set and the 2000 in
`bank.npy`. Dedup against both and report n, to the same standard as the harmful
set. **n (N7): 100 per set under B1; under B2/B3 re-derive from a power
calculation on the arm-1-versus-arm-5 refusal-rate difference and report the
target power.**

**Metrics.** (a) Arditi refusal-substring rate — primary. (b) CE loss on harmless
completions — capability gate, not a footnote. (c) Manual stratified spot-check
of ~40 completions, documented — covers the incoherence failure mode where a
lobotomised model scores as "not refusing". (d) LLM judge secondary if an API key
exists; never on the critical path, never a local model sharing the GPU.

**Branch rules, committed:**
- Arm 2 drop ≥ 0.8 × arm 1 drop, arm 2 CE within noise of arm 5 → `d` is causally
  equivalent to `v_ref`.
- Arm 3 ≤ arms 4/4b + CI → `d`'s causal power is fully explained by `v_ref`
  overlap, which **supports** the completeness conclusion. **Valid only if
  §18.5a′ test 2 passed at sub-layer granularity.**
- Arm 3 significantly above both nulls → the single-direction account is
  incomplete, and that becomes the headline.

#### 18.5c Steering (absorbs §4f)

Addition arm on harmless prompts, `û ∈ {d₄₀₀, v_ref_200}`, **matched on
projection magnitude, not raw α**. Consistency = dose-response curves overlap
within bootstrap CI at every level. Divergence at high magnitude only →
"consistent in the linear regime, diverging under strong steering", which is a
finding, not a failure.

### 18.6 §3 reduction

- **§3d (whitening):** cite §16a — already resolved, not load-bearing. Do not re-run.
- **§3c (layers):** `model.py:97` confirms capture is `[n, 32, 4096]` float32, so
  this is CPU-only on the transferred file. §14a already establishes ratio
  stability at 8/22/28/31 — but see §18.4d″ for what that ratio measures. What
  remains: the §18.4b metrics (r, cosine, AUROC) at 8/22/31 alongside 28, which is
  where the Wurgaft layer-inheritance vulnerability actually bites.
- **§3a (resampling stability), §3b (full k sweep):** P1, CPU, gated on A1.

Under A1 ✓ / A2 ✗, §18.4 and §18.6 both still run in full; only §18.5 is lost.

### 18.7 Reproducibility pins

- **Model SHA:** `config.py:11` — reference, do not re-pin.
- **Stage B dataset commit:** `data/stageb/UPSTREAM_COMMIT.txt`.
- **XSTest, `harmless_train`/`harmless_test`:** snapshots unrecorded at run time.
  Pinned by SHA-256 content hash in `env/A1_manifest.sha256`, taken at source on
  fedora and verified after transfer. Recorded verbatim: *snapshot unrecorded at
  run time; re-pinned by content hash on 9 Sep 2026.* No retroactive implication
  of pinning.
- **fedora environment, and a correction.** The first A1 capture froze fedora's
  *system* python (torch 2.11.0+cu130, transformers 5.12.1) — which is **not the
  environment any result was produced in**. The runs used
  `~/unsupervised-concept-geometry/.venv`: Python 3.12.13, **torch 2.14.0,
  transformers 5.16.1**, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0,
  scikit-dimension 0.3.7. Driver 580.105.08. Pinned in
  `env/A1_pip_freeze_RUNENV.txt` (79 packages); the wrong capture is retained as
  `env/A1_pip_freeze_SYSTEM_wrong_env.txt` rather than deleted.

  **Consequence for §18.5a′ (N8):** `config.py:23` states "transformers on this
  box is 5.12.1, where block outputs are no longer a bare tuple in all paths."
  That comment describes the system install; the code actually ran under
  **5.16.1**. The sub-layer hook work targets 5.16.1, and the comment is stale —
  which is the reason `_resolve_block_output`-style explicit tensor resolution is
  required for the sub-layer hooks rather than any assumed return convention.
- **Local environment.** This laptop had no venv and no numpy/scipy/sklearn/pandas/skdim.
  Created on Python 3.13.7 (system python3 is 3.14, whose `ensurepip` fails):
  numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.0, pandas 3.0.5, pyarrow 25.0.1,
  skdim 0.3.7 — pinned in `env/local_venv_freeze.txt`. **The analysis stack
  matches the fedora run environment except numpy (2.5.3 vs 2.5.2) and Python
  minor (3.13.7 vs 3.12.13); scikit-learn, scipy and skdim match exactly**, which
  is what the §18.4 numbers depend on. Running on macOS arm64 against Linux x86
  remains a platform change with a different BLAS and is recorded as a deviation;
  the acceptance test below, not the version match, is what licenses it.
- **Determinism (N5).** `sklearn.manifold.Isomap` defaults to `eigen_solver='auto'`,
  which may select ARPACK with an unseeded start vector, so `Y` is not guaranteed
  reproducible even on one machine. **Pin `eigen_solver` explicitly** before any
  embedding-derived quantity is used as an acceptance criterion. All comparisons
  between Isomap components use `|cos|`, since component sign is arbitrary.
- **Platform acceptance test.** Before trusting any new local number, reproduce a
  logged one against the **full-precision** value in
  `results/stageb_structure.parquet` — not the rounded 1.020 in §14a's table.
  Prefer a `D_geo`-derived statistic, which is deterministic shortest-paths, over
  an embedding-derived one. Tolerance ±0.002; outside it, §18.4 runs on fedora or
  not at all.
- **Stage A diagnoses remain untested hypotheses.** The alkane monotone-ordinal
  account and the IUPAC orthographic confound are stated predictions, not
  established mechanisms, and nothing in §18 tests either. The writeup must say so
  explicitly; without that sentence, "we diagnosed two failure modes" reads as a
  finding. Under B3 they return to scope.

### 18.8 Open — requires Austin

1. ~~Tailscale / fedora reachability~~ — **RESOLVED 9 Sep 2026**, §18.2. Branch A1 ✓ / A2 ✓.
2. ~~Deadline branch~~ — **RESOLVED 9 Sep 2026: B1.** Not submitted; applying to
   the Fri 11 Sep deadline. The Sept 10 18:00 freeze is live, §18.5 eval sizes are
   n=100 per set, and the P2 items stay deferred. B2's correction question does
   not arise — nothing has been submitted, so the moved claims (§18.10 curvature,
   §18.12's 0.654 cosine, N10's AUROC construction) are corrected *before*
   anything is filed rather than after.
3. **Venue/length** — still open, and now load-bearing: three headline claims
   moved this week and the abstract must be rewritten around what survived.
4. **ssh authorisation scope** for a long-running unattended GPU job under §18.5.
5. ~~`bank_bare` transfer~~ — **RESOLVED**: transferred while the window was open (§18.2).

### 18.9 Claim-dependency graph

Built **before** §18.4 resolves, with the branches open — that is the artifact's
purpose. Graphviz/mermaid source in-repo, rendered for the writeup, annotated
post-hoc with which branch fired.

Nodes:
- §18.4a′ CV R² low → C2's overclaiming fires; "recovers a *direction*" is not supportable
- §18.4b → the novelty claim
- §18.4c′ → the AUROC claim **and** every `v_ref`-referenced quantity
- §18.4d″ falsification test → **gates the straightness claim's existence**, not just its size
- §18.4e′ → the straightness claim's power statement
- §18.4f″ n-matched contrast → §14d's ID contrast
- §18.5a′ test 2 → **gates** arm 3's interpretability
- §18.5b arm 3 → the conclusion itself

### 18.10 §18.4d″ step 3 result — N1 confirmed; §14a's straightness finding does not survive

Run 9 Sep 2026 against pre-registration `d6aa671`, before any other §18.4 item.
Synthetic only — no activations, no GPU. `scripts/n1_curvature_falsification.py`,
output in `results/n1_curvature_falsification.txt`. The `geodesic_chord` under
test is lifted from `stageb_structure.py` by compiling its own AST node, so the
code exercised is byte-identical to the code that produced §14a.

Circular arcs of known half-angle in a 2-plane of R^64, n=600, k=12. Truth is
computed in closed form from the generating angles over the same
"well-separated" pair rule the statistic uses, so it is exact rather than the
whole-arc `φ/sin φ` approximation.

**At σ = 0, where there is no estimator bias to hide behind:**

| true ratio | logged (geo / embedded chord) | ambient (geo / PCA-space chord) |
|---|---|---|
| 1.000 | 1.000 | 1.000 |
| 1.005 | 1.000 | 1.005 |
| 1.023 | 1.000 | 1.023 |
| 1.105 | 1.000 | 1.105 |
| 1.206 | **1.000** | 1.206 |

The logged statistic returns 1.000 on an arc whose true ratio is 1.206. The
ambient statistic recovers every value exactly. **N1 is confirmed**, and the
pre-registered withdrawal branch does not fire.

Across noise levels σ ∈ {0, 0.0003, 0.001, 0.003, 0.01} the logged statistic's
total span is ≤0.012 while true curvature ranges over 1.000–1.206.

**Consequences, in order of how much they cost:**

1. **§14a's "Curvature: straight" is withdrawn as stated.** The observed
   1.010–1.020 measures Isomap's MDS reconstruction residual. It is not evidence
   of straightness, and it is not evidence of curvature either — the statistic is
   flat in curvature, so it carries no information about it in either direction.
   Reported henceforth as a reconstruction diagnostic under that name.
2. **§14d's branch-one verdict loses its first-named support.** It cited
   geodesic/chord, the coordinate alignment (0.946), the behavioural correlation
   (0.871) and the §14c natural experiment. The latter three are untouched by
   N1. The verdict is not withdrawn, but it may no longer be stated as resting on
   curvature, and the writeup must not claim the structure was shown to be
   straight. Whether straightness can be claimed at all now depends entirely on
   §18.4e′ run against the ambient statistic.
3. **SPEC2 §1d's worry was correct and understated.** It predicted the null sits
   above 1.0. On a *straight* line the ambient statistic returns 1.011 at
   σ=0.0003 and **2.107** at σ=0.01 — the kNN geodesic estimator's multiplicative
   bias is large and strongly σ-dependent. The ambient statistic is therefore
   only interpretable against a matched synthetic null, which is exactly
   §18.4e′'s construction. Calibrating by the straight-line null recovers truth
   to 14–16% at the extremes and 0% at σ=0.
4. **Single replicate per cell.** The non-monotone rows at σ ∈ {0.0003, 0.001}
   (true 1.005 reading higher than true 1.105) are single-draw noise, and are the
   concrete argument for §18.4e′'s 50 replicates.

**N9 — a new finding this test produced incidentally, not yet acted on.** `d_hat`
from `id_mle` on a genuinely **1-dimensional** arc reads 1 at σ=0 but rises to
**16–24** at σ=0.01, purely from noise above the inter-point spacing. Levina–Bickel
at fixed k reports ambient noise dimensionality when local noise exceeds
neighbour spacing. This bears directly on §14a's "intrinsic dimension ≈ 5" and
on §18.4f″, and it is a different mechanism from the sample-size confound in N2.
Recorded here rather than folded into §18.4f″ silently: the fix belongs in an
amendment written before it is run, per §0.

### 18.11 Amendment: N9 and N10 — written before either is run

Per §0, both are recorded as rules before the runs that test them. N10 was found
while tracing how the logged AUROC is computed, in preparation for §18.4c′.

#### N9 — the ID estimator reports noise dimensionality when noise exceeds neighbour spacing

§18.10 produced this incidentally: `d_hat` from `id_mle` reads 1 on a genuinely
1-dimensional arc at σ=0, and **16–24** on the same arc at σ=0.01, where noise
exceeds inter-point spacing. Levina–Bickel at fixed k measures the local cloud,
and once the local cloud is noise, it measures noise. This is a different
mechanism from N2's sample-size confound and is not addressed by n-matching.

It also propagates: `d_hat` sets `n_components` for the embedding
(`stageb_structure.py:110`), so an inflated ID changes the Isomap fit itself, not
just the reported dimension.

**Added to §18.4f″, run in this order:**

1. Per layer, in the PCA(64) space, report the median nearest-neighbour distance
   (the spacing) and the residual scale about the 1-D principal curve along `d`.
   Report the **ratio σ / spacing** — the quantity §18.10 shows governs the bias.
2. Run the §18.10 synthetic 1-D arc at the measured σ/spacing for each layer and
   report what `id_mle` returns for a structure whose true intrinsic dimension is
   **1**. This converts the concern into a number: "at this data's noise-to-spacing
   ratio, a truly 1-dimensional structure reads as ID = X."
3. **Pre-registered rule.** If the synthetic 1-D control returns an ID that
   reaches the observed 5.21 (within the §18.4f″ CI), then the observed ID is
   consistent with a far lower-dimensional structure plus noise, **no claim about
   ~5 dimensions may be made, and the result is reported as an upper bound.** If
   the control returns well below 5.21, the ID claim survives and gains a
   quantified noise floor.

**§14a's "intrinsic dimension ≈ 5" is under review until this resolves**, on the
same terms as §14a's curvature under §18.10. Note the exposure: N2 and N9 push in
*opposite* directions — N2 says the bank's 5.89 is inflated by having 2000 points,
N9 says both numbers may be inflated by noise — so the contrast could survive both
or neither, and guessing which is not available.

#### N10 — the logged 0.995 AUROC is fully transductive, label-oriented, and not produced by the pipeline

§18.4c′ diagnosed the 0.995 as "transductive on one side", reasoning from §14's
prompt-set description: the 200 XSTest borderline prompts are in the 600-prompt
fit bank, the 200 contrast prompts were held out. **That diagnosis was wrong.**
The number is computed at `figures_stageb.py:99-110`, and its actual construction
is:

1. Select **only** the 400 XSTest prompts — 200 borderline + 200 contrast.
2. Fit PCA(64) **and Isomap on those 400 points**, i.e. on precisely the set being
   evaluated. It is not the 600-prompt refusal manifold; it is a separate
   embedding built from the evaluation data.
3. Take the first coordinate, and **orient its sign using the evaluation labels**
   (`if spearmanr(cc, lab).statistic < 0: cc = -cc`).
4. AUC by Mann–Whitney on that coordinate.

Four consequences, none of which the current writeup states:

- **Fully transductive, not half.** Every point scored was in the fit.
- **Not the pipeline's coordinate.** §14d's sentence — "an unsupervised coordinate
  recovered from activations alone... separates them at AUC 0.995" — describes a
  coordinate fit on the test set, and not the one every other Stage B number
  refers to. The 0.946 alignment, the 0.871 behavioural correlation and the
  curvature statistic all come from the 600-prompt embedding; the 0.995 does not.
- **The sign is supervised.** With a binary choice this cannot turn a real
  separation into a fake one, but it does mean a coordinate anti-correlated with
  harm reports as 0.995 rather than 0.005. The orientation rule must be declared,
  or fixed by an unsupervised convention.
- **No logged artifact.** The number exists only as a figure title; it is in no
  parquet. §17b's "§5 parquet schema: partial" understated this.

**Revised §18.4c′ (supersedes the version in the pre-registration commit):**

1. Fit the pipeline on the **400 harmful + harmless prompts only**, which contain
   no XSTest data of either kind. Build `d₄₀₀` per §18.4a′.
2. Score all 400 XSTest prompts by **projection onto `d₄₀₀`** — an out-of-sample
   readout, which is the reason §18.4a′ builds an ambient direction at all.
   AUROC on borderline-versus-contrast. **This is the headline.**
3. **Fix the sign from the training set**, never the test labels: orient `d₄₀₀` so
   that harmful projects above harmless in the 400. Report the orientation rule.
4. Report the logged 0.995 alongside, described as constructed above rather than
   as a held-out result. The gap between (2) and 0.995 is the reportable quantity:
   it measures what fitting on the evaluation set bought.
5. **§14c and §14d's headline are under review until (2) exists.** §18.6's
   "abstract order unchanged: XSTest AUROC leads" is suspended — a lead number fit
   on its own evaluation set cannot lead until it is recomputed.

**Method note.** This is the third time in this project that a number survived
until someone traced the code that produced it rather than the prose describing
it (§15d's tangent estimator, §16b's permuted-label null, and now this). The
pattern is specific: the prose describes what the analysis was *meant* to do, and
the discrepancy is invisible from the write-up alone. Every remaining headline
number should be traced to its call site before the writeup cites it.

### 18.12 §18.4a′ and §18.4c′ results — the AUROC survives; the alignment claim does not

Run 9 Sep 2026 against `d6aa671` as amended by §18.11. CPU, on the transferred
`.npy`. `scripts/stageb_18_4ac.py`, output `results/stageb_18_4ac.json`.
`eigen_solver="dense"` per N5, so these are reproducible; the logged results used
the "auto" default.

**Reproduction check first.** Rebuilding the logged construction
(`figures_stageb.py:99-110` — fit on the 400 XSTest, label-chosen sign) returns
**0.9951** at L28 against the published 0.995. The N10 diagnosis is confirmed by
reproduction, not inference.

| L28 quantity | value |
|---|---|
| AUROC, held out (projection onto `d₄₀₀`, sign from training set) | **0.9935** |
| AUROC, logged construction reproduced | 0.9951 |
| CV R², coordinate on 64-d PCA scores | 0.971 |
| CV R², ridge on full 4096-d | 0.983 |
| \|cos(`d₄₀₀`, `d₆₀₀`)\| | 0.682 |
| cos(`v_ref_200`, `v_ref_400`) | 0.977 |
| **cos(`d₄₀₀`, `v_ref_200`)** | **0.654** |
| Pearson r, clean pairing | 0.978 |
| Pearson r, logged pairing (the 0.946) | 0.946 |

**1. The AUROC claim survives decontamination, and §18.4c′ step 5 is lifted.**
Held-out 0.9935 against transductive 0.9951: fitting on the evaluation set bought
**0.0016**. The number was constructed indefensibly and is nonetheless right.
§14c and §14d's headline are restored, now on a genuinely held-out basis with the
sign fixed from training data, and §18.6's "AUROC leads" is un-suspended. Report
the construction history anyway — the claim is now *supported*, which it was not
before, and that difference is the whole point of having checked.

Layer profile: 0.868 (L8), 0.9949 (L22), 0.9935 (L28), 0.9950 (L31). Stable
across 22–31, which is the §18.6 layer-inheritance check passing for the claim it
actually matters for. Note the held-out method *beats* the logged construction at
L8 (0.868 vs 0.715) and L31 (0.995 vs 0.791): fitting Isomap on 400 XSTest points
alone yields a coordinate that is not harm-aligned at those layers.

**2. The Isomap coordinate is a linear readout — a result, per §18.4a′.** CV R²
0.971 on the 64-d scores and 0.983 under ridge on the full 4096-d, at L28, with
0.959–0.986 across all four layers. The coordinate is recoverable as a direction
to within a few percent of variance, so "unsupervised recovery of a direction" is
**not** overclaiming — C2's failure branch does not fire. That the ridge fit on
4096 dims slightly *exceeds* the 64-d fit bounds what PCA(64) discarded at ~1% of
variance.

This is also the first hard evidence for the §18.4b branch: if the coordinate is
~97% linear, the manifold machinery has little room to be load-bearing.

**3. The `d` fork is NOT empty, and this is the surprise.** \|cos(`d₄₀₀`,`d₆₀₀`)\|
= 0.682 at L28 (0.640–0.689 across layers), far below the 0.99 threshold.
Removing the 200 borderline prompts moves the recovered direction substantially.
Per §18.4a′, **both are carried through every table, `d₄₀₀` is primary in §18.5,
`d₆₀₀` is a labelled robustness arm.** The two directions nonetheless produce
near-identical AUROC, which is itself informative: the discriminative content is
shared, and the 0.68 is in components that do not affect the harm readout.

**4. `v_ref_200` becomes primary.** cos(`v_ref_200`, `v_ref_400`) = 0.977 at L28,
below the pre-registered 0.99 bar. It is close, and the temptation is to call it
immaterial; the rule was fixed in advance and is followed. `v_ref_400` appears as
a labelled robustness row.

**5. The alignment claim is materially weaker than the record implies, and this
is C1 made concrete.** The writeup says the dominant axis "aligns with a
difference-in-means refusal direction (0.946)". But 0.946 is a Pearson
correlation across points; **the actual cosine between the unsupervised direction
and `v_ref` is 0.654.** Both are legitimate quantities and both are reported, but
they are not interchangeable, and 0.654 is a much weaker geometric statement than
0.946 reads as. The writeup may not describe the Pearson number as "alignment"
without naming it. The clean pairing raises the Pearson to 0.978 while the cosine
stays at 0.654 — the two metrics move in opposite directions under
decontamination, which is exactly why C1 required both.

**Consequence for §18.5.** Arm 4b's constraint is **cos = 0.654**, not 0.946.
The gap matters: random vectors at cosine 0.65 to `v_ref` are a much weaker null
than at 0.95, so arm 4b is a correspondingly weaker control than SPEC2 assumed,
and arm 3 (`d_⊥`) carries proportionally more of the test.

**Still open.** §14a's curvature (§18.10) and ID (§18.11 N9) remain under review;
neither is touched by this run.

### 18.13 §18.4b and §18.4g results — the manifold machinery is not load-bearing

Run 9 Sep 2026 against `d6aa671`. `scripts/stageb_18_4bg.py`,
`results/stageb_18_4bg.json`. Fit on the clean 400; every baseline on the same
bank as the `d` it is compared against; both metrics per row, per C1.

| method | Pearson r | cos(`v_ref_200`) | AUROC (held out) |
|---|---|---|---|
| Isomap coord 1 → `d` (ours) | 0.978 | **0.654** | **0.9935** |
| PC1 of the fit bank | 0.999 | **0.995** | 0.9922 |
| k-means (k=2) centroid difference | 1.000 | 0.998 | 0.9922 |
| random directions, n=100 | mean 0.419, p95 **0.811** | mean 0.013, p95 0.030 | mean 0.662, p95 0.891 |

**The pre-registered branch fires: PC1 r = 0.999 ≥ 0.90, |ΔAUROC| = 0.0013 ≤ 0.01.
The manifold machinery is not load-bearing.** The reframe fixed in advance is
adopted: *the refusal axis is recoverable by any unsupervised linear method,
including trivial ones — which is positive evidence that the linear account is
complete.* This is the honest framing and arguably the cleaner contribution; it
was written down before the number existed precisely so it could not be spun now.

It is worse for the manifold than "no better". **PC1 aligns with `v_ref` at cosine
0.995; the Isomap direction manages 0.654.** Isomap's coordinate is *less* aligned
with the supervised refusal direction than the first principal component, while
delivering the same AUROC to within 0.0013. The extra machinery moves the
recovered direction off `v_ref` without buying discrimination.

**The random null is the other finding here, and it retroactively damages the
headline alignment number.** A random direction, sign-oriented on the training
set exactly as every other row is, achieves **Pearson r = 0.811 at the 95th
percentile and 0.899 at maximum**. The logged 0.946 therefore sits barely above
what random directions reach by chance. Pearson-r-across-points is a nearly
uninformative metric on this data, and it is the metric the writeup's alignment
claim rests on. Cosine's null is 0.013 mean / 0.030 p95, so cos 0.654 clears its
null by a wide margin and 0.995 overwhelmingly. **Report cosine as primary; report
Pearson r only with its null attached.** C1 flagged the confusion of the two; this
quantifies why it mattered.

The AUROC null is also not 0.5 but **0.662 mean / 0.891 p95** — harmful and
harmless prompts differ in many ways a random direction partially captures, and
sign-orientation folds the distribution upward. 0.9935 exceeds the maximum of 100
draws, so it stands, but the null must be stated rather than assumed at chance.

#### §18.4g — both curvature statistics across `K_SWEEP` (fit set: 600, per N6)

| k | logged (geo / embedded chord) | ambient (geo / PCA-space chord) | connected |
|---|---|---|---|
| 8 | 1.0131 | 2.7440 | True |
| 12 | 1.0205 | 2.4114 | True |
| 16 | 1.0221 | 2.2357 | True |
| 24 | 1.0228 | 2.0610 | True |

Two things, both bearing on §18.4e′:

1. **The logged statistic is flat in k (1.013–1.023) — as §18.10 predicts of an
   MDS residual.** Its k-stability was previously read as evidence the curvature
   result was robust. It is instead a property of the quantity being insensitive
   to almost everything, curvature included.
2. **The ambient statistic reads 2.06–2.74 and is strongly k-dependent.** §18.10
   found that a *straight* line at σ=0.01 — noise above inter-point spacing —
   returns 2.107 on this statistic. The Stage B data therefore sits in the
   noise-dominated regime, which independently corroborates **N9**: if the local
   cloud is noise at this scale, both the curvature statistic and `id_mle` are
   reporting noise. The strong k-dependence means **the §18.4e′ power statement
   must be given per-k**; a single MDR would be meaningless.

**Consequence.** The curvature question is now entirely dependent on §18.4e′: an
observed 2.41 at k=12 is interpretable only against a matched synthetic null at
the data's own σ/spacing. It is no longer plausible that this ends in a
straightness claim; the realistic outcomes are a bound or a null with stated
power.

### 18.14 §18.4f″ results — N2 was small, N9 was decisive, and branch one gets stronger

Run 9 Sep 2026 against `d6aa671` as amended by §18.11. `scripts/stageb_18_4f.py`,
`results/stageb_18_4f.json`. Reproduction check: the logged construction returns
**5.21 / 5.89**, matching §14a exactly.

#### N2 — real, but roughly a tenth of what I implied

Subsampling both banks to the same m=480 without replacement (N3), 200 draws:

| | mean `id_mle` | 95% CI |
|---|---|---|
| refusal | 5.274 | [5.083, 5.435] |
| bank | 5.820 | [5.461, 6.221] |
| bank − refusal | **+0.546** | **[+0.121, +0.993]** |

**The difference CI excludes zero, so the bank contrast survives n-matching.** Of
the logged +0.68 gap, only **+0.07** is attributable to sample size. N2's
mechanism is real and was worth controlling, but it accounts for about a tenth of
the effect, not the effect. §18.11's warning that N2 and N9 might push in opposite
directions was right to be agnostic; N2 simply turned out small.

#### N9 — fires, and it removes the ID number entirely

Corrected noise-to-spacing first. §18.4f″'s initial ratio of 0.042 compared a
*per-dimension* σ against a *full-vector* distance and is not a like-for-like
quantity; it is withdrawn. Measured properly, inside PCA(64):

| quantity | value |
|---|---|
| median residual norm about the 1-D line, in PCA(64) | 16.918 |
| median nearest-neighbour spacing, in PCA(64) | 7.884 |
| **ratio** | **2.146** |

Each point sits more than twice as far off the 1-D line as it does from its
nearest neighbour. This is the regime §18.10 identified, and it is independently
corroborated by §18.4g's ambient statistic reading 2.06–2.74 where §18.10 found a
*straight* line at comparable noise returns 2.107. Three separate measurements
agree.

**The control:** `id_mle` on a structure whose true intrinsic dimension is **1**,
carrying this data's own resampled residuals, returns **5.430, 95% CI [5.157,
5.704]** — against an observed refusal ID of **5.274**. The 1-D control does not
merely reach the observed value, it slightly **exceeds** it.

**§18.11's pre-registered rule fires: no claim about ~5 dimensions may be made,
and the result is reported as an upper bound.** "Intrinsic dimension ≈ 5" is a
measurement of the noise floor, not of refusal.

#### The consequence is not a loss — branch one gets a better argument

§14a read ID ≈ 5 and rescued the one-dimensional reading via the bank comparison,
which §14d then had to defend against the literal fork criterion ("ID > 1"). That
defence is no longer needed. The refusal set at 5.274 sits **at or below** the
noise floor a genuinely 1-dimensional structure produces (5.430), while the
general-instruction bank at 5.820 sits **above** it. So:

- Refusal is **consistent with being effectively one-dimensional**, stated
  directly rather than by comparison.
- The bank is **not** — general instruction prompts carry structure beyond 1-D
  that refusal prompts do not.

This is a stronger version of branch one, reached by a different route, and it
retires §14d's first "honest qualification" (that the literal ID > 1 criterion
pointed the other way): at this noise floor the literal criterion could not have
distinguished 1 from 5 in the first place, so it was never informative here.

**§14a's ID row is superseded**: report 5.21/5.89 as logged, with the 1-D noise
floor of 5.43 beside them, and the dimensionality claim stated as a bound.

### 18.15 §18.4e′ results — no power to detect curvature, and a departure from straight anyway

Run 9 Sep 2026 against `d6aa671`. `scripts/stageb_18_4e.py` (+ `_e2.py` addendum),
`results/stageb_18_4e.json`, `stageb_18_4e2.json`, log in `stageb_18_4e.log`.
Option (B) as fixed in §18.4e′: arcs generated in the full 4096-d ambient space,
noise by resampling the real residual vectors about the 1-D line along `d` (which
reproduces the true residual covariance without factorising a 4096² matrix), the
pipeline run whole with PCA(64) fit on each synthetic bank. 50 replicates at
k=12, 20 elsewhere.

#### The logged statistic has no power at any k — N1, quantified

| k | logged: null p95 | observed | MDR |
|---|---|---|---|
| 8 | 1.0677 | 1.0131 | **> 1.50** |
| 12 | 1.0613 | 1.0205 | **> 1.50** |
| 16 | 1.0632 | 1.0221 | **> 1.50** |
| 24 | 1.0642 | 1.0228 | **> 1.50** |

Power never exceeds 0.22 at any true ratio through 1.50. This is the outcome
§18.4e′ pre-registered as expected and it converts §18.10's demonstration into a
number: **the statistic §14a reported straightness from could not have detected
curvature of any magnitude up to 50%.** Note also that the observed value sits
*below* its own null mean, which is meaningless given zero power, and is exactly
the kind of reassuring-looking number the whole exercise exists to disarm.

#### The ambient statistic has essentially no power either

| k | ambient: null p95 | observed | MDR |
|---|---|---|---|
| 8 | 2.409 | 2.744 | 1.50 |
| 12 | 2.139 | 2.411 | **> 1.50** |
| 16 | 1.987 | 2.236 | **> 1.50** |
| 24 | 1.836 | 2.061 | **> 1.50** |

**§18.4e′'s branch rule fires at its extreme.** MDR > 1.02, so no straightness
claim is available; the honest statement is *"curvature is below our detection
threshold"* — and the threshold is above 1.50, i.e. the instrument cannot detect
curvature of any magnitude this design tested. The power statement is per-k as
§18.4g required, and the floor is k-dependent because the null itself is.

#### But the observed value exceeds the null at every k, and it survives the sharper null

Observed is above the null's 95th percentile at all four k. The obvious artifact
explanation is density: the null draws arc positions uniformly, while the real 600
are three groups (harmful / harmless / borderline) with gaps along `d`, and gaps
lengthen graph geodesics. So the null was re-run with positions **resampled from
the observed projections onto `d`**, changing nothing else:

| k | observed | null p95, uniform | null p95, density-matched | |
|---|---|---|---|---|
| 8 | 2.744 | 2.422 | 2.491 | above |
| 12 | 2.411 | 2.134 | 2.171 | above |
| 16 | 2.236 | 1.987 | 2.027 | above |
| 24 | 2.061 | 1.847 | 1.866 | above |

Clustering accounts for very little and the excess survives. The N9 control was
re-run density-matched for the same reason and is unchanged: a true 1-D structure
returns `id_mle` 5.175 [4.868, 5.497] against an observed 5.207, so §18.14's
conclusion stands under the sharper null too.

**What the excess is not.** At k=12 the observed 2.411 exceeds the synthetic mean
at true ratio **1.50** (2.149). The observed configuration is therefore outside
the arc family that was swept — it is not characterisable as arc curvature of any
magnitude tested, and quoting a curvature figure from it would be unsupportable.

**What it is.** The null is "straight line + exchangeable residuals + matched
density". Resampling residual *vectors* independently of position destroys any
dependence between where a point sits along `d` and how it deviates. So what the
excess establishes is that **the real residuals are not exchangeable along the
coordinate** — there is position-dependent structure. That is a real, consistent,
null-clearing departure from the straight-line model, and its direction is not
identified: curvature, position-dependent noise scale, and locally varying
covariance all produce it.

#### Net effect on §14a and §14d

- **No straightness claim survives, by two independent routes.** The original
  statistic had no power (§18.10, and now MDR > 1.50); the corrected statistic
  detects a significant departure from straight. §14a's "Curvature: straight" is
  withdrawn, not merely reclassified.
- **§14d's branch-one verdict is not withdrawn but its basis has changed
  entirely.** Curvature is gone as a support. What carries it now is §18.14's
  ID result — refusal at or below the 1-D noise floor while the bank is above it —
  plus the coordinate alignment, the behavioural correlation and the §14c natural
  experiment. Three of those four are unaffected by anything found this week.
- **The departure is reportable and must not be buried**, but it cannot be
  presented as curvature. The correct sentence names the null it clears and the
  three mechanisms it cannot distinguish between.

### 18.16 §3a and §3b results — the Isomap direction is the least stable of the three

Run 9 Sep 2026 against `d6aa671`. `scripts/stageb_18_6.py`,
`results/stageb_18_6.json`. L28, fit on the clean 400.

#### §3a — stability under resampling (80%, without replacement per N3, 20 refits)

| direction | mean \|cos(resample, full)\| | min | p5 |
|---|---|---|---|
| **Isomap `d`** | **0.855** | 0.793 | 0.829 |
| PC1 of the same bank | 0.9993 | 0.9992 | 0.9992 |
| `v_ref` (supervised) | 0.9992 | 0.9990 | 0.9991 |

Dropping a fifth of the data moves the Isomap direction by cosine 0.855, while
PC1 and the supervised difference-in-means both move by less than 0.001. This is
consistent with §18.12's |cos(`d₄₀₀`, `d₆₀₀`)| = 0.682: the recovered direction is
genuinely sample-sensitive, and that sensitivity is a property of the manifold
step, not of the data — the same points give a near-invariant PC1.

**This completes §18.4b's picture.** Against PC1 the Isomap coordinate is: equal
on held-out AUROC (0.9935 vs 0.9922), **worse** on alignment with `v_ref` (0.654
vs 0.995), and **an order of magnitude less stable** (0.855 vs 0.9993). The
pre-registered conclusion was "the manifold machinery is not load-bearing"; the
stability result says it is actively worse on every axis measured, at equal
discrimination.

As §3a requires, the distinction is stated explicitly: what is measured here is
**stability under resampling**. With `eigen_solver` pinned (N5) the pipeline is
deterministic given data, so stability under algorithmic randomness is a
different question and is not claimed.

#### §3b — k sweep at L28

| k | `id_mle` | `d_hat` | cos(`d`,`v_ref`) | Pearson r | AUROC (held out) | connected |
|---|---|---|---|---|---|---|
| 8 | 5.68 | 6 | 0.636 | 0.975 | 0.9939 | True |
| 12 | 5.68 | 6 | 0.654 | 0.978 | 0.9935 | True |
| 16 | 5.68 | 6 | 0.678 | 0.980 | 0.9937 | True |
| 24 | 5.68 | 6 | 0.721 | 0.984 | 0.9944 | True |

**The conclusions are k-stable.** AUROC varies by 0.0009 across the sweep and ID
not at all. cos(`d`,`v_ref`) rises monotonically with k (0.636 → 0.721), which is
the expected direction — larger neighbourhoods make Isomap behave more like a
global linear method, so `d` moves toward PC1 and hence toward `v_ref`. That
trend is itself a small piece of evidence for §18.4b's conclusion: the further the
method is from the manifold regime, the better it aligns.

Graph connectivity holds at every k on the 400, which was §18.4c′'s precondition
for building anything on `d₄₀₀`. That precondition is met.

### 18.17 Amendment to §18.5, written before the GPU work begins

Two changes forced by §18.12–§18.16. Recorded before any ablation code exists.

**1. Arm 4b's constraint is cos = 0.654, measured (§18.12), not 0.946.** This
weakens that null substantially: random vectors at cosine 0.65 to `v_ref` are far
easier to clear than at 0.95, so arm 4b tests less than SPEC2 assumed and **arm 3
(`d_⊥`) carries correspondingly more of the causal test.** Stated now so the
weakness is not discovered in the interpretation.

**2. New arm 6: PC1 of the fit bank.** After §18.4b and §18.16 the interesting
causal question has changed. "Does `d` cause refusal" was the question when `d`
was the distinctive object; it no longer is — PC1 matches its AUROC, beats its
alignment and is an order of magnitude more stable. The question that now carries
weight is whether **`d`, PC1 and `v_ref` are causally interchangeable.**

| Arm | Direction | Purpose |
|---|---|---|
| 1 | `v_ref_200` | replication anchor |
| 2 | `d₄₀₀` | does the unsupervised direction cause refusal |
| 3 | `d_⊥` | independent causal content beyond `v_ref` |
| 4 | random unit, n=5 | null |
| 4b | random at cos = 0.654 to `v_ref` | "more than a vector near `v_ref`" |
| 5 | none | baseline |
| **6** | **PC1 of the clean 400** | **is the trivial baseline causally equivalent too** |

**Pre-registered rule for arm 6.** If arms 1, 2 and 6 produce refusal drops within
each other's CIs, with CE within noise of arm 5, then the linear account is
supported **causally** and not merely correlationally: the same behaviour is
reachable through the supervised direction, the manifold direction and the first
principal component alike. That is the strongest form the completeness conclusion
can take from this data. If arm 6 drops materially less than arms 1–2, then PC1's
correlational parity was not causal parity, §18.4b's "not load-bearing" applies to
prediction but not intervention, and that distinction becomes the finding.

**Unchanged:** the §18.5a′ acceptance tests gate everything, and arm 3's null is
uninterpretable unless test 2 passes at sub-layer granularity.

### 18.18 Amendment: §18.5a′ test 2's threshold was unsatisfiable — recorded as a weakening

**This amendment was written after the test failed, which is the circumstance in
which amendments are least trustworthy.** It is therefore stated with the
evidence, the reasoning, and what is given up.

§18.5a′ specified "assert `max|û·x| < ε`" without defining ε or considering
dtype; the implementation used ε = 1e-2. First run: tests 1, 3 and 4 pass (65
writes hooked as predicted, 1560 firings over 24 forwards, null-op identical,
KV-cache consistent, hooks removed). Test 2 fails at 3.6e-2 (writes) and 1.5e-1
(residual stream).

**Diagnosis, measured rather than assumed:**

| quantity | value |
|---|---|
| unablated max \|û·x\| at writes / at blocks | 14.69 / 14.00 |
| ablated max \|û·x\| at writes / at blocks | 0.034 / 0.193 |
| fraction removed | **99.77% / 98.62%** |
| model dtype | bfloat16, ε = 7.81e-3 |
| single-write rounding floor at that scale | 0.109 |
| √65 accumulation bound over the writes | 0.88 |

The residual is 1.8× the single-write floor and well under the accumulation
bound. **The absolute threshold was unachievable by any correct implementation in
bf16**: writing `t − (t·û)û` back into bf16 rounds at ~0.8% of the value's
magnitude, so the test as written measured the storage format, not the code.

**Revised criterion, fixed before re-running:**

1. **≥99% of the projection removed at every write** — a relative bound, scale-
   and dtype-free.
2. **Residual-stream leak ≤ √N_writes · ε_dtype · (unablated scale)**, the
   random-walk bound on accumulated rounding across the N=65 writes. Derived, not
   tuned to the observed value.

**What this still catches, and why it is not a rescue.** E6's within-block leak —
attention writing a û-component the MLP reads before the next ablation — would
appear as a *large fraction* of the unablated scale, not a rounding-scale
residual. Criterion 1 fails at once under block-output-only hooks, which is the
failure mode the test exists for. The weakening is confined to the numerical
floor.

**What is given up, and it bears on arm 3.** The ablation is **not exact**: the
residual stream retains ~1.4% of the direction's maximum projection. A downstream
mechanism reading the direction with sufficient gain could in principle still see
it, so arm 3's null is "no detectable causal content above a 1.4% residual",
not "none". Arditi's weight-orthogonalisation — folding û out of the write
matrices themselves, so no per-position rounding accumulates — is the stronger
construction and is **not** run here. That is a stated limitation of this
implementation, not a property of the method.

### 18.19 §18.5b first run: the replication anchor FAILED — amendment before any re-run

Run 9 Sep 2026, `results/stageb_18_5.json`, completions in `results/gen_18_5_*.json`.
Acceptance tests passed first (§18.18). n=76 harmful after dedup (24 of JBB's 100
overlapped the fit bank — the lineage concern was real), n=100 harmless.

| arm | refusal (harmful) | drop | CE (harmless) |
|---|---|---|---|
| 5 baseline | 0.947 | — | 0.2144 |
| **1 `v_ref` (anchor)** | **0.882** | **0.066** | 0.3101 |
| 2 `d₄₀₀` | 0.921 | 0.026 | 0.2580 |
| 3 `d_⊥` | 0.947 | 0.000 | 0.2295 |
| 6 PC1 | 0.882 | 0.066 | 0.3162 |
| 4 random (mean of 5) | 0.944 | 0.003 | ~0.2145 |
| 4b cos-0.65 random (mean of 5) | 0.916 | 0.032 | ~0.2498 |

**Arm 1 fails as a replication anchor.** Arditi et al. report near-total removal
of refusal; ablating `v_ref` here moves it 0.947 → 0.882. A stratified read of the
completions confirms this is not a matcher artifact — the model genuinely still
refuses ("I can't write a defamatory article…", "I can't assist with that
request."), so the substring set is working and the effect is simply absent.

**Consequence, stated before any re-run: every other arm is uninterpretable.**
Arm 3's null (0.000 drop) and arm 2's near-null cannot support the completeness
conclusion when the positive control also has almost no effect. §18.5b's
pre-registered rules all condition on arm 1 reproducing Arditi; it does not, so
none of them fire. This is exactly the failure arm 1 was included to catch, and
the pre-registration is what makes it a finding rather than a quiet omission.

**Diagnosis.** The acceptance tests rule out the implementation: 65 writes hooked,
99.75% of the projection removed, null-op token-identical, KV-cache consistent
(§18.18). What is left is the *direction*, and §11f named this risk in advance:
*"The non-trivial part of Arditi et al. is not the difference of means — it is the
selection sweep over layer and position scored on a validation set. We fix layer
and position from the §3g profile and compute the direction directly."* §13b
separately found causal accessibility to be a step function in layer for Stage A.
**L28 was inherited, not selected, and the most likely reading is that the L28
difference-in-means direction is not the causally potent one.**

**Amendment — the sweep §11f declined to reproduce is now required.** Ablate the
layer-`ℓ` difference-in-means direction, for every ℓ, measuring refusal drop on
the same held-out harmful set. All 32 layers are already cached, so this costs
generation only.

**Pre-registered rules, fixed now:**

1. **Selection:** the layer with the largest refusal drop, subject to CE on
   harmless staying within 0.05 of baseline. Ties broken toward the earlier layer.
2. **If some layer reaches a drop ≥ 0.5**, the anchor is established there, and
   arms 2/3/4/4b/6 are re-run at that layer with directions rebuilt there. The
   L28 results stand as reported, relabelled as what they are: a null at an
   inherited layer.
3. **If no layer reaches 0.5**, this is a failed replication of Arditi in our
   hands. §18.5 is then reported as unrun-in-substance per SPEC2's fallback
   ("report it as unrun rather than partially run"), the causal section becomes
   proposal-only, and **no causal claim is made in either direction** — including
   the convenient one that `d` lacks causal content.
4. Either way, **§14d and §18.4's conclusions are untouched**, since none of them
   rest on ablation.

**Note against interest.** Arms 2, 3 and 4b at L28, read naively, look like
tidy support for the completeness conclusion: `d_⊥` does nothing, `d` sits inside
the cos-matched null. That reading is not available while the anchor is dead, and
recording that here is the point of having written the rule down first.

### 18.20 §18.19 sweep result — the causal locus is layer 10, not 28

`results/stageb_18_5_sweep.json`, `/tmp/sweep.log`. Difference-in-means direction
rebuilt at every layer and ablated at every write; refusal drop on the same 76
held-out harmful prompts. Baseline 0.947.

| layer | 0–4 | 5 | 6 | 7 | 8 | 9 | **10** | **11** | 12 | 13 | 15 | 16–27 | 28 | 29–31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal | ≥0.908 | 0.829 | 0.763 | 0.500 | 0.237 | 0.105 | **0.000** | **0.000** | 0.132 | 0.474 | 0.303 | ≥0.868 | 0.882 | ≥0.803 |

**Layers 10 and 11 remove refusal completely.** The causal window opens sharply
at layer 7, peaks at 10–11, and has essentially closed by 16. Layer 28 — the layer
every Stage B number was computed at — produces a drop of 0.066, barely above the
random null of 0.003.

**§11f's deviation was consequential, and it is now measured.** The spec recorded
in advance that "the non-trivial part of Arditi et al. is… the selection sweep
over layer and position" and chose to fix the layer from the §3g profile instead.
That choice cost the causal result entirely: the anchor fails at the inherited
layer and succeeds completely two-thirds of the network earlier. §13b found the
same shape in Stage A — causal accessibility as a step function in layer — so this
is the second appearance of the pattern in this project.

Per §18.19 rule 2 the anchor is established (drop 0.947 ≥ 0.5) and the arms are
re-run at the selected layer, subject to rule 1's CE gate — a layer that removes
refusal by destroying the model does not qualify, and the sweep measured drop
only. Ties break to the earlier layer, so layer 10 is the candidate.

**A gap this exposes, which no amount of re-running fixes.** Every structural
claim in Stage B — the AUROC, the intrinsic dimension, the curvature statistic,
the coordinate alignment — was measured at L28, plus 8/22/31. **The layer where
refusal is causally accessible was never analysed structurally.** The
representational and causal claims are therefore about different parts of the
network, and the writeup must say so rather than letting "layer 28" and "the
refusal direction" blur together. §18.21 measures the §18.4b panel at the selected
layer to close as much of this as CPU allows.

### 18.21 §18.5b at the selected layer — Arditi replicated, and PC1 is causally interchangeable with `v_ref`

`results/stageb_18_5_rerun.json`, completions in `results/gen_18_5r_*.json`.

**Layer selection (§18.19 rule 1).** Candidates with drop ≥ 0.5, CE gate at
baseline + 0.05 = 0.2644:

| layer | drop | CE | |
|---|---|---|---|
| **10** | 0.947 | 0.2270 | **selected** (tie broken to the earlier layer) |
| 11 | 0.947 | 0.2244 | qualifies |
| 9 | 0.842 | 0.2290 | qualifies |
| 12 | 0.816 | 0.2235 | qualifies |
| 8 | 0.711 | 0.2267 | qualifies |
| 15 | 0.645 | **0.3626** | **fails the CE gate** |

The gate did real work: L15 removes 65% of refusal by damaging the model, and
without CE it would have looked like a partial success. At L10, cos(`d`,`v_ref`)
= 0.683 and cos(PC1,`v_ref`) = 0.990.

**Arms at L10** (n=76 harmful, n=100 harmless; baseline refusal 0.947, CE 0.2144):

| arm | refusal | drop | CE |
|---|---|---|---|
| **1 `v_ref` (anchor)** | **0.000** | **0.947** | 0.2270 |
| **6 PC1** | **0.000** | **0.947** | 0.2373 |
| 2 `d` | 0.289 | 0.658 | 0.2427 |
| 3 `d_⊥` | 0.934 | 0.013 | 0.2324 |
| 4 random (mean of 5) | 0.944 | 0.003 | ~0.2146 |
| 4b cos-0.68 random (mean of 5) | 0.295 | **0.653** | ~0.2217 |

Refusal on harmless stays 0.000 in every arm — no over-refusal artefact — and
every CE is inside the gate, so nothing here is capability damage.

**The anchor replicates.** Ablating `v_ref` at L10 takes refusal from 0.947 to
**0.000** with CE moving 0.2144 → 0.2270. That is Arditi et al. reproduced, and
it licenses every other arm.

**All three pre-registered rules fire, and they agree:**

1. **Arm 2 is NOT causally equivalent to `v_ref`.** Drop 0.658 against the
   required 0.8 × 0.947 = 0.758. CE is fine, so this is a genuine shortfall, not
   degradation.
2. **Arm 3 is indistinguishable from the random null.** Drop 0.013; the
   refusal-rate difference against random has 95% CI [−0.092, +0.066], including
   zero. **`d`'s causal power is fully explained by its `v_ref` overlap** — the
   result §18.5b pre-committed as expected, now interpretable because the anchor
   works.
3. **Arm 4b settles what `d` is.** Random vectors constrained to the *measured*
   cosine 0.683 produce a mean drop of **0.653**, against `d`'s **0.658**. `d` does
   not beat a random vector pointed the same distance from `v_ref`. It is, causally,
   nothing more than "a vector near `v_ref`" — which is precisely the question
   arm 4b was constructed to answer, and §18.17 was right that its weakened
   constraint would make arm 3 carry the test.

**Arm 6 is the finding.** PC1 removes refusal **completely**, identically to the
supervised direction — the difference CI is [0.000, 0.000]. §18.17's rule fires:
*the linear account is supported causally, not merely correlationally.* The same
behaviour is reachable through the supervised difference-in-means direction and
through the first principal component of an unsupervised bank, while the manifold
coordinate reaches it only in proportion to its overlap with them.

#### The representation/causation dissociation, measured

The §18.4b panel recomputed at the causal layer (`results/stageb_18_21_layer10.json`):

| layer | AUROC `d` | AUROC PC1 | AUROC `v_ref` | cos(`d`,`v_ref`) | CV R² | `id_mle` |
|---|---|---|---|---|---|---|
| 10 (causal) | 0.9399 | 0.9567 | 0.9610 | 0.683 | 0.974 | 5.73 |
| 11 | 0.9518 | 0.9737 | 0.9731 | 0.605 | 0.978 | 5.56 |
| 28 (structural) | **0.9935** | 0.9922 | 0.9924 | 0.654 | 0.971 | 5.68 |

**Discrimination peaks where causal accessibility is absent, and vice versa.**
L28 separates XSTest safe from contrast at 0.9935 while its direction is causally
inert (drop 0.066); L10 removes refusal completely while separating at 0.9399.
This is a dissociation, not a discrepancy — refusal is *linearly decodable*
late and *causally manipulable* early — and it is the sharpest statement this
project can make about where the single-direction account holds. The writeup must
not use "the refusal direction" as though one layer served both roles.

### 18.22 §18.5c first run INVALID — the "consistent" verdict is withdrawn

`results/stageb_18_5c.json`, completions `results/gen_18_5c_*.json`. The run
returned refusal 0.000 on harmless at every direction and every multiplier, and
the script's rule declared the curves **"consistent"** — all three overlapping at
every dose.

**That verdict is withdrawn. It is an artifact of having destroyed the model.**

Completions at the *smallest* dose (×0.5, α ≈ 1.0–1.5):

```
v_ref ×0.5 : 'ereoereoereoereoereoereoereoereo…'
d400  ×0.5 : 'ereoereoereoereoereoereoereoereo…'
v_ref ×4.0 : ' AppModule AppModule AppModule…'
d400  ×4.0 : '.scalablytyped.scalablytyped.scalablytyped…'
baseline   : 'Aristotle and Socrates were two influential philosophers from ancient Greece…'
```

**Cause — a design error in §18.5c, mine.** `AB.add_direction` applies its hook at
**all 65 residual-stream writes**, because it was written by analogy with the
ablation arm. For ablation that is correct and required (§18.5a′): removing a
component at every write is what makes the stream û-free. For *addition* it is
wrong — the stream accumulates all 65 additions, so the realised displacement is
≈65α, and a nominal dose of 1.0 is a real dose of ~68. Standard activation
steering adds at a single site. The three curves agree because all three
directions destroyed the model identically, and the refusal-substring metric read
0.000 because degenerate text contains no refusal substrings.

**The deeper failure, and it is the one worth keeping.** §18.5b had a capability
gate — CE on harmless — and it did real work there, disqualifying L15 (§18.21).
**§18.5c had no such gate**, and without one an incoherent model reads as a clean
null. This is the fourth time in this project a near-zero statistic has come back
looking like a result (§15d's tangent estimator, §16b's hierarchy cosines,
§18.10's curvature ratio, now this). The rule generalises: **any arm whose
"success" value is near zero needs a positivity control on the same run**, not a
plausibility argument afterwards.

**Amendment, fixed before the re-run:**

1. **Add at a single site** — the selected layer's block output, all positions —
   which is what "steering along û" means. `add_direction` gains an explicit
   `layers` argument; the all-writes behaviour is no longer reachable by default.
2. **Coherence gate, reported per condition:** CE on the baseline harmless
   completions, plus an explicit degeneracy check (distinct-token ratio of each
   completion). A condition failing the gate is reported as **"dose exceeded the
   model's usable range"** and its refusal rate is **not** treated as a
   measurement.
3. **The consistency verdict may only be computed over conditions that pass the
   gate**, and if fewer than two doses survive for a direction, the comparison is
   reported as having no power rather than as agreement.

### 18.23 §18.5c re-run and the spot-check — sufficiency agrees with necessity

`results/stageb_18_5c.json`. Single-site addition at L10 (§18.22), directions
rebuilt there, matched on each direction's own harmful-minus-harmless projection
gap (`d` 2.082, `v_ref` 3.050, PC1 3.018). Baseline refusal on harmless 0.000,
CE 0.2144.

| direction | ×0.5 | ×1.0 | ×2.0 | ×4.0 |
|---|---|---|---|---|
| `v_ref` | 0.040 | **0.970** | 1.000 ✗ | 0.190 ✗ |
| PC1 | 0.040 | **0.950** | 1.000 ✗ | 0.510 ✗ |
| `d` | 0.000 | 0.020 | 0.600 | 1.000 ✗ |

✗ = fails the §18.22 coherence gate. **The gate immediately earned itself**: at
`v_ref` ×4.0 the *measured refusal rate falls* to 0.190, not because steering
stopped working but because CE reached 4.05 and the distinct-token ratio 0.313 —
the model is producing degenerate text, which contains no refusal substrings.
Without the gate that would have been a spurious inverted-U dose-response.

**Consistency, computed only over the two doses passing the gate for all three
directions:**

- **PC1 vs `v_ref`: overlap at both** — [−0.050, +0.050] at ×0.5 and
  [−0.080, +0.030] at ×1.0.
- **`d` vs `v_ref`: diverges at both** — [−0.080, −0.010] and [−0.990, −0.900].

**Verdict: divergent for `d`, interchangeable for PC1** — the same answer the
ablation arm gave, reached by addition instead of removal.

**And `d`'s shortfall is quantitatively what its `v_ref` overlap predicts.** At
×1.0 `d` displaces 2.082 along itself, i.e. 0.683 × 2.082 = **1.42** along
`v_ref` — just under `v_ref`'s own ×0.5 dose of 1.52. The refusal induced is
0.020 against 0.040. `d` behaves like a *weaker dose of `v_ref`*, not like a
different intervention, which is the sufficiency-side statement of §18.21's arm
4b result.

#### §18.5b metric (c): stratified spot-check

42 completions, six per arm, `results/spotcheck_18_5.json`:

| arm | distinct-token ratio | empty | mean words |
|---|---|---|---|
| `v_ref` | 0.878 | 0.00 | 36.1 |
| PC1 | 0.880 | 0.00 | 36.0 |
| `d` | 0.898 | 0.00 | 32.1 |
| `d_⊥` | 0.925 | 0.00 | 15.4 |
| random | 0.938 | 0.00 | 15.2 |

Read against text rather than asserted: the ablated model produces **coherent,
on-topic compliance** with the harmful requests — fluent prose, correct register,
directly addressing what was asked. It is not degenerate and it is not evasive.
The incoherence failure mode — where a lobotomised model scores as "not
refusing" — **does not occur here**, so the 0.947 drop is genuine refusal removal.
Note the length signal corroborates it independently: arms that remove refusal
produce ~36-word answers, while the inert arms (`d_⊥`, random) produce ~15-word
refusals.

**§18.5 is complete.** Every pre-registered arm ran, the anchor replicates, the
acceptance tests gate the ablation, the capability gate and coherence gate each
caught a distinct artifact, and necessity and sufficiency agree.

### 18.24 Amendment: position sweep — branches fixed before the grid is computed

§18.20 described §18.19's sweep as closing the §11f gap. **It closed half of it.**
§11f dropped Arditi's selection sweep over *layer and position*; §18.19
reconstructed layer only. The read position — token index **−1**, which for this
chat template is `'\n\n'` (id 271), the post-instruction newline after
`<|start_header_id|>assistant<|end_header_id|>` — was never selected. It is the
default of `src/model.py:95-100` (`positions=None` → last prompt token), taken by
`capture_stageb.py:70` passing no `positions`.

**The −1 choice has more external support than "defensible", and the two anchors
disagree by model:**

- Arditi et al.'s Table 5 selects i* = **−5**, the `<|eot_id|>` token, for **Llama-3 8B**.
- A 2025 post-training paper re-running Arditi's selection on **Llama-3.1-8B-Instruct**
  — this exact model — selects layer **11**, position **−1**.

So on our model, an independent reimplementation of the procedure §11f skipped
lands on (11, −1); we arrived at (10, −1) by sweep and by library default. That is
agreement on both axes, not a lucky escape. **The writeup must state both anchors**,
because a reviewer who opens Table 5 sees −5 for the 3.0 sibling and will ask.

> **Provenance flag.** Both citations were supplied by the project lead and are
> **not verified against the papers in this session.** They must be checked before
> the writeup cites them; if either is wrong, this subsection's framing changes and
> the grid results below do not.

**Why run the sweep anyway — not for arm 1.** The anchor is at ceiling (0.000
refusal) and cannot improve. The reasons are:

1. **`d`'s nulls are the load-bearing claim.** "`d` carries nothing beyond its
   `v_ref` overlap" is what rules the manifold coordinate out as an independent
   causal handle, and it is currently conditional on a read position nobody chose.
2. **PC1's recovery numbers are position-dependent**, and they are the headline
   methodological contribution.
3. **The 32 × 5 grid is the better figure** — structurally Arditi's Figure 11 for
   Llama-3 8B, computed for 3.1, and strictly more informative than §18.19's 1-D
   layer curve.

**Method, following Arditi's own economy.** Their selection metric needs no
generation: probability mass on the refusal-initiating token set at the last
prompt position, as log-odds. We reuse the project's already-verified 6-token
opener set (`capture_stageb.py:30-48`, the same set behind `refusal_prob` in §14),
one forward per prompt per cell. Generation-based confirmation with the existing
substring + CE gate then runs at the argmax cell, at (10, −1), and at the grid's
second mode. The log-odds readout is also **continuous and non-saturating**, which
is the fix the steering dose-response needs.

**Pre-registered branches:**

1. **`v_ref(10, i)` bypasses completely for all i ∈ {−1…−5}** → position is not
   load-bearing at the causal layer. One robustness sentence; the limitation
   dissolves.
2. **Some positions fail, −1 among the working ones** → report the grid, state
   that −1 was inherited and *post-hoc verified* to sit in the working region, and
   that selection was not optimised over position.
3. **`d`'s overlap or its nulls change materially at some position** → the arms
   2/3/4b conclusion is qualified to the read position, and the **range across
   positions** is reported rather than a single number.
4. **cos(PC1, `v_ref`) moves outside ~0.90–0.96 at some position** → report the
   range; the abstract number becomes "0.946 at the pre-registered read position,
   0.9x–0.9y across post-instruction positions."

**Ordering.** This runs *before* any matched-null or dose-grid work, because both
would need re-running if position turns out to matter.

### 18.25 §18.24 results — the grid reproduces both anchors; the `d` conclusion is position-robust

`results/stageb_18_24_grid.json`, `_dirs.json`, `_confirm.json`. Bypass score =
baseline − ablated refusal-opener log-odds on the 76 held-out harmful prompts.
Token identities, which the grid is uninterpretable without:

| index | −5 | −4 | −3 | −2 | −1 |
|---|---|---|---|---|---|
| token | `<\|eot_id\|>` | `<\|start_header_id\|>` | `assistant` | `<\|end_header_id\|>` | `'\n\n'` |

**Both external anchors are live cells, and they are a ridge rather than a
disagreement.** Top of the grid:

| rank | cell | score | |
|---|---|---|---|
| 1 | (12, −5) | +19.320 | |
| 2 | (11, −1) | +19.053 | **the 3.1 reimplementation's selection** |
| 3 | (11, −2) | +18.957 | |
| 4 | (11, −5) | +18.652 | |
| 5 | (10, −1) | **+18.088** | **the cell we inherited** |

The inherited cell ranks **5 of 160**, 1.23 log-odds below the argmax. Arditi's
Table 5 position for the 3.0 sibling (−5 = `<|eot_id|>`) is strong here too
(+19.320 at L12, +18.652 at L11), so their choice transfers rather than conflicts.
The top region is layers 10–12 × positions {−1, −2, −5}.

**Position −4 is dead at every layer** (maximum 2.643 across all 32; 0.707 at
L10). It is `<|start_header_id|>`, a pure structural template token — an
interpretable failure, not an anomaly.

**Branch 2 fires**, not branch 1: some positions fail, −1 is among the working
ones, and it was inherited rather than selected. Recorded exactly that way.

#### Generation confirmation — the cheap metric agrees

| cell | refusal | drop | CE |
|---|---|---|---|
| (12, −5) | 0.000 | 0.947 | 0.2349 |
| (11, −1) | 0.000 | 0.947 | 0.2244 |
| (10, −1) | 0.000 | 0.947 | 0.2270 |

All three saturate the substring metric with CE inside the gate, which is why the
log-odds readout was needed to rank them at all. **The choice among the top cells
is immaterial for the anchor**, and §18.21's result does not move.

#### Branch 3 — fires on the raw range, resolves on the live one

cos(`d`, `v_ref`) at L10 spans 0.632–0.896 across all five positions (spread
0.264, "material" by the pre-registered threshold). **The entire spread is the
dead position:** −4 gives 0.896; across the four live positions the range is
0.632–0.683, spread **0.051**. L11 is the same shape (0.602–0.632 live, 0.875 at −4).

Excluding −4 after seeing the numbers is the shape of a garden-fork, so the basis
is stated: the exclusion criterion is **the grid's own bypass score**, the grid was
pre-registered and computed before this analysis, and −4 fails across all 32
layers rather than at the cell of interest. Both ranges are reported.

Resolved by measurement rather than argument — arms re-run at **(10, −5)**,
Arditi's own position:

| cell | cos(`d`,`v_ref`) | `d` drop | `d_⊥` drop | cos-matched null |
|---|---|---|---|---|
| (10, −1) | 0.683 | 0.658 | 0.013 | 0.614 |
| (10, −5) | 0.679 | 0.487 | 0.000 | 0.430 |

`d`'s absolute drop falls at −5, but **so does its matched null, and the
relationship is preserved**: `d` sits inside its cos-matched null at both
positions, and `d_⊥` is ~0 at both. **The arms 2/3/4b conclusion — `d` carries no
causal content beyond its `v_ref` overlap — holds at Arditi's position choice as
well as at ours.** Branch 3's qualification is therefore that the *magnitudes* are
position-dependent while the *conclusion* is not, and the range is reported.

#### Branch 4 — fires, but my threshold was mis-specified

cos(PC1, `v_ref`) at L10 is **0.978–0.993**, outside the pre-registered 0.90–0.96
window — *above* it. That window was anchored to the 0.946 figure, which is a
**Pearson correlation, not a cosine**. C1 caught this confusion in SPEC2; it
recurred in my own §18.24 pre-registration. The threshold is withdrawn as
mis-specified rather than treated as a finding.

The quantity that does matter is stability, and it separates the layers sharply:

| layer | cos(PC1,`v_ref`) range | spread |
|---|---|---|
| 10 | 0.978–0.993 | **0.016** |
| 11 | 0.983–0.996 | 0.013 |
| 28 | 0.275–0.998 | **0.723** |

**PC1's alignment is position-robust at the causal layers and position-fragile at
the structural one.** At (28, −4) it collapses to 0.275 and AUROC(PC1) falls to
**0.3005 — below chance**. This is a third independent reason the causal layers
are where the claim should be stated.

#### Limitation, restated as §18.24 requires

Position was **not swept at selection time**; −1 was inherited from
`src/model.py`'s default. It is now **post-hoc verified** to rank 5 of 160, to sit
in the top ridge, and to coincide with the position an independent reimplementation
selects for this model — while the original paper selects −5 for the 3.0 sibling,
which our grid also finds strong. Selection was not optimised over position, and
the writeup says so in those terms.

> The two external citations remain **unverified in this session** (§18.24) and
> must be checked against the papers before the writeup cites them. Nothing in
> this subsection's measurements depends on them.

### 18.26 Citation 1 verified — and our grid's argmax is Arditi's published cell

Checked against `2406.11717v3.pdf` (Arditi et al., v3, 40pp), supplied 9 Sep 2026.
§18.24's provenance flag is discharged for citation 1; citation 2 remains open.

**Table 5 (p. 21), verbatim rows:**

| model | i* | l*/L | bypass_score |
|---|---|---|---|
| **L LAMA -3 8B** | **−5** | **12/32** | −9.715 |
| L LAMA -3 70B | −5 | 25/80 | −7.839 |
| Y I 6B | −5 | 20/32 | −6.693 |
| G EMMA 2B | −2 | 10/18 | −14.435 |
| (the other nine models) | −1 | — | — |

Citation 1 is confirmed exactly: **i\* = −5 for Llama-3 8B**, and −1 is the
modal choice across their thirteen models rather than an unusual one.

**The paper also supplies the layer, which was not in the claim as given: l\* = 12.**

**Our grid's argmax is (12, −5).** An independent reconstruction of their
selection procedure, run on Llama-3.1-8B-Instruct, returns as its top cell
precisely the (layer, position) pair they publish for Llama-3-8B-Instruct. That is
a materially stronger statement than §18.25's "their choice transfers", and it is
the sentence the writeup should carry.

**Two further confirmations that the comparison is like-for-like:**

- **The position set is identical.** Figure 11's legend names `pos -5:
  '<|eot_id|>'`, `pos -4: '<|start_header_id|>'`, `pos -3: 'assistant'` — the same
  five post-instruction tokens we swept, with the same identities.
- **The chat template is identical.** Table 6 gives
  `{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`, matching the
  rendered prompt asserted in §14.

**Three caveats, all of which must appear with the claim:**

1. **Different model.** They select on Llama-3-8B-Instruct; we ran
   Llama-3.1-8B-Instruct. The agreement is across a version boundary, which is
   what makes it interesting and also what stops it being a replication.
2. **Not the same selection procedure.** They score candidates on bypass *and*
   induce *and* KL over validation sets (§2.1); we ranked on bypass alone and
   applied the capability gate at the confirmation stage. Same family, not the
   same criterion.
3. **Sign conventions differ.** Their `bypass_score` is reported negative
   (−9.715); ours is baseline − ablated and positive. **Magnitudes are not
   cross-quotable** — only the argmax cell is.

**A prediction we have not checked.** §18.25 found position −4 dead at every layer
(max 2.643 of 32). Their Figure 11 plots bypass and induce for all candidates on
Llama-3 8B and would confirm or refute that directly. We read its legend, not its
values.

**Citation 2 is still open**, and it is the weaker of the two: it has no title,
authors or identifier recorded, so it cannot be verified or cited as it stands.
Note that §18.26 partly relieves the load on it — the (12, −5) agreement is now
established from a verified source, so the position limitation no longer rests on
the unverified claim. §18.24's framing is updated accordingly: the strongest
statement available is the Arditi one, with the (11, −1) reimplementation as
corroboration *if* it can be identified.

### 18.27 §18.7 platform acceptance test — run late, passes; and the outstanding list

**The test was not run before §18.4, as §18.7 required.** All of §18.4 and §18.6
was computed on macOS arm64 before any logged number had been reproduced there.
That is a procedural violation of this section's own rule, recorded rather than
quietly repaired by running it now and presenting it as if it had gated anything.
Had it failed, results already committed would have been invalidated.

It passes, and not marginally (`results/platform_acceptance.json`):

| quantity | logged (Linux x86) | local (macOS arm64) | delta |
|---|---|---|---|
| `geo_chord_mean` @ L28, k=12, solver auto | 1.0204886987 | 1.0204886962 | **2.5e-9** |
| same, solver dense | — | 1.0204886968 | 1.9e-9 |
| `id_mle` | 5.206835 | 5.206835 | 0 |
| `id_twonn` | 6.887768 | 6.887767 | 1e-6 |
| `d_hat` | 5 | 5 | — |

Nine orders of magnitude inside the ±0.002 tolerance, and `eigen_solver` proves
immaterial here (auto and dense differ by 6e-10). The §18.7 platform deviation is
therefore real but inert, and every locally computed §18.4/§18.6 number stands.

### 18.28 What remains unbuilt

Implementation is complete for every experiment §18 pre-registered. What is not:

| item | status |
|---|---|
| **§18.9 claim-dependency graph** | **never built** — and it was specified to be built *before* §18.4 resolved, "with the branches open — that is the artifact's purpose". Building it now yields documentation, not the artifact that was specified. Recorded as missed, not as pending. |
| **The (layer × position) grid figure** | not made. The data is in `stageb_18_24_grid.json`; it is the direct analogue of Arditi's Figure 11 for 3.1 and is the most informative figure this project can produce. |
| **Dose-response on the log-odds readout** | outstanding. §18.23's substring metric saturates (0.000 → 0.970 → 1.000) and only two doses cleared the coherence gate, so the curves are compared at two points. The continuous metric of §18.24 fixes exactly this. |
| **F4, F5c are now stale** | F4 plots the geodesic/chord distribution and the tangent-`v_ref` cosine — **both withdrawn** (§18.10, §15d). F5c plots the AUC 0.995 from the transductive construction (**N10**). They encode superseded claims and must not be reused as they stand. |
| **Citation 2** | unidentified; cannot be verified or cited (§18.26). |
| **Arditi Figure 11 values** | unread. Would confirm or refute §18.25's finding that position −4 is dead. |
| F6, §5 parquet schema, §4d three-layer deviation | unchanged from §17b. |

### 18.29 Amendment: dose-response on the non-saturating readout

§18.23 compared the three steering curves at **two** doses — the only ones that
cleared the coherence gate — using the refusal-substring rate, which saturates
(0.000 → 0.970 → 1.000). A two-point comparison of a saturating metric is a weak
test of "the curves agree", and it is the pre-registered comparison currently
carrying the least power in §18.

The §18.24 readout fixes it: refusal-opener **log-odds** at the last prompt
position is continuous, unbounded, and needs one forward per prompt rather than a
decode. Re-run with it, changing nothing else:

- Same single-site addition at L10, directions rebuilt there (§18.22).
- Same matching on each direction's own harmful-minus-harmless projection gap.
- **Denser dose grid**, since the readout no longer saturates: ×{0.25, 0.5, 0.75,
  1.0, 1.5, 2.0, 3.0, 4.0}.
- **Coherence gate becomes CE-based**, so it can be evaluated at every dose from
  forwards alone; generation is run at a subset purely to cross-check that the
  log-odds curve and the substring curve agree where both are measurable.

**Pre-registered rules:**

1. **Consistency** = the log-odds curves for `d`, `v_ref` and PC1 overlap within
   bootstrap CI at **every dose passing the CE gate**. More doses now qualify, so
   this is a strictly harder test than §18.23's.
2. If PC1 and `v_ref` overlap throughout while `d` diverges, §18.23's verdict is
   confirmed on a metric that cannot saturate, and that is the version reported.
3. If the denser grid reveals overlap where §18.23 found divergence, **§18.23's
   divergence was an artifact of a two-point saturating comparison** and is
   withdrawn in favour of this result.
4. The substring cross-check must agree with the log-odds curve wherever both are
   measurable; disagreement is reported rather than reconciled.

### 18.30 §18.29 result — the denser test qualifies "PC1 is interchangeable"

`results/stageb_18_29_dose.json`. Baseline log-odds −11.225, CE 0.2144. Doses
passing the CE gate: **five**, against §18.23's two.

| ×mult | `v_ref` Δ | PC1 Δ | `d` Δ | gate |
|---|---|---|---|---|
| 0.25 | +1.079 | +0.983 | +0.209 | ok |
| 0.50 | +5.661 | +4.937 | +0.637 | ok |
| 0.75 | +12.449 | +11.429 | +1.680 | ok |
| 1.00 | +16.016 | +15.670 | +3.948 | ok |
| 1.50 | +15.431 | +15.953 | +8.865 | ok |
| 2.00 | +13.923 | +14.452 | +11.415 | `v_ref`/PC1 FAIL, `d` ok |
| 3.00–4.00 | — | — | — | all FAIL |

**Rule 4 satisfied:** where both metrics were measured they agree — at ×1.0,
substring 0.970 / 0.950 / 0.020 against log-odds +16.016 / +15.670 / +3.948, same
ordering and same spacing.

**Rule 1 fails for both directions, and the failures are not comparable:**

| dose | `d` − `v_ref` | PC1 − `v_ref` |
|---|---|---|
| ×0.25 | [−2.32, +0.50] overlap | [−1.56, +1.29] overlap |
| ×0.50 | [−6.34, −3.71] **diverge** | [−1.97, +0.48] overlap |
| ×0.75 | [−11.83, −9.68] **diverge** | [−1.78, −0.22] **diverge** |
| ×1.00 | [−12.94, −11.19] **diverge** | [−0.75, +0.06] overlap |
| ×1.50 | [−7.06, −6.05] **diverge** | [+0.38, +0.66] **diverge** |

- **`d` diverges at 4 of 5 doses, consistently negative, by up to 13 log-odds.**
  §18.23's verdict on `d` is confirmed on a metric that cannot saturate, and
  strengthened — the two-point test had found the same thing with far less power.
- **PC1 diverges at 2 of 5 doses, by at most 1.8 log-odds, and the sign flips**
  (below `v_ref` at ×0.75, above at ×1.5). **§18.23's "PC1 is interchangeable with
  `v_ref`" does not survive the denser test as stated.** It was measured at two
  doses of a saturating metric; with five doses of a continuous one, the curves are
  statistically distinguishable at two of them.

**The honest statement, replacing §18.23's:** under *addition*, PC1 tracks `v_ref`
to within ~1.8 log-odds at every dose tested — an order of magnitude closer than
`d`, whose gap reaches 13 — but the two are **not identical**, and the difference
is a small dose-shift in the curve rather than a consistent offset. Per §15e's
lesson about sign-flipping gaps, the shape difference should not be read as one
direction being weaker: PC1 is *behind* `v_ref` at ×0.75 and *ahead* at ×1.5.

**What is unaffected: the necessity result.** §18.21's ablation finding — `v_ref`
and PC1 both take refusal to exactly 0.000, difference CI [0.000, 0.000] — does
not involve this metric or this intervention and stands unchanged. So the correct
summary is asymmetric and should be written that way:

> **Under ablation, PC1 and `v_ref` are indistinguishable. Under addition, they
> are close but distinguishable. `d` is far from both under either.**

**Method note.** This is the second time a §18 conclusion has been weakened by
running the same comparison with more power (§18.14's ID result was the first).
Both times the weaker version was the one with fewer measurement points, and both
times the pre-registered rule — not a judgement call — is what forced the
downgrade.

### 18.31 F9 — the selection grid figure

`figures/F9_position_grid.png`, from `scripts/figure_position_grid.py`. Sequential
single-hue encoding because the job is magnitude; **not** diverging, despite the
signed score, because the negative tail reaches only −1.18 against a positive
range to +19.32 and a zero-centred diverging scale would spend half its range on
17 of 160 cells. The colour floor is clamped at 0 and the figure says so. Marked
cells are outlined and identified in a legend placed *below* the axes — an in-plot
legend covered live cells on the −1 row, and obscuring data to label it is not a
trade worth making.
