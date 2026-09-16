# Judge Validation Study Methodology

## Purpose

This study measures how closely one local LLM judge agrees with one human labeler
when both apply EvalBench's three-level `overall_quality_v1` rubric to outputs from
two target models. It validates the judge, not the relative quality of the target
models. Target-model comparison remains a separate paired analysis.

The implementation is complete, but this document does not claim a study result.
A result exists only after two real target runs, 60 human labels, a single rubric
calibration, and an untouched holdout report have been committed.

## Fixed design

The study uses 60 outputs drawn from two completed runs over identical stored case
snapshots. The source runs must use different target-model identifiers. Before any
labeling, selection uses a recorded pseudorandom seed and assigns every output to
one of two fixed splits:

- 30 calibration examples for one round of rubric clarification.
- 30 holdout examples for the final judge-validation report.

Across all 60 examples, each target model contributes 30 outputs and each of the
four suite categories contributes 15. Because 15 cannot divide evenly between two
models, each model-by-category stratum contributes either seven or eight outputs.
Each split contains 15 outputs from each target model; each category contributes
seven or eight outputs to each split. Presentation order is randomized from the
same seed.

The JSON study artifact records the seed, source run and model identifiers, case
snapshots, answers, split assignment, rubric provenance, and randomized order.
Selection is repeatable for the same database state, run IDs, and seed.

## Human labeling

The human labeler assigns exactly one ordinal score to each output:

| Score | Meaning |
| --- | --- |
| 0 | Incorrect, non-responsive, centrally instruction-violating, or unsafe. |
| 1 | Useful or partly correct, but with a substantive defect. |
| 2 | Correct, relevant, sufficiently complete, instruction-following, and safe. |

`evalbench judge label` shows the rubric, prompt, optional expected reference, and
candidate answer. It does not show source run or model identity. The study artifact
retains that provenance for auditability, so blinding applies to the labeling
interface rather than to anyone manually inspecting the JSON file.

The command writes the artifact atomically after every score and skips completed
examples when resumed. The project uses a single human labeler. Consequently, the
study cannot estimate inter-rater reliability or adjudication effects.

## Independent LLM judge

The judge must not match either target-model identifier. The documented judge is
`gemma3:4b`, although the artifact records the exact identifier used. EvalBench sends
the rubric as a system instruction and the evaluation prompt, expected reference,
and candidate answer as untrusted JSON data. Ollama runs non-streaming structured
output at temperature zero against a schema containing only `score` and `rationale`.

Judge responses are cached by all inputs that can affect the score: complete rubric
content and version, judge model, provider schema version, case ID, prompt, expected
reference, and answer hash. Study scoring writes each response and its rationale
atomically, can resume after interruption, and refuses a different judge model or
provider schema after scoring has begun.

Temperature zero improves repeatability but does not prove determinism across model,
runtime, or hardware versions. The committed artifact is the evidence used for the
published result.

## Calibration and holdout firewall

The bundled rubric begins with `status: calibration`. While that status is active,
EvalBench refuses to label, judge, or report the holdout split.

The permitted sequence is:

1. Label all 30 calibration examples.
2. Score those examples with the independent judge.
3. Generate and inspect the calibration report.
4. Clarify the rubric once, record the revision in version control, and set its
   status to `frozen`.
5. Run `evalbench judge freeze` to bind the study to that frozen rubric.
6. Label and judge the 30 holdout examples without further rubric changes.
7. Generate and publish the holdout report regardless of agreement strength.

The freeze command requires complete human and judge calibration scores and proves
that every holdout score is still empty. It preserves the calibration rubric hash
and calibration results, then records separate frozen-rubric provenance for the
holdout. Calibration reports therefore continue to identify the draft rubric,
while holdout reports identify the frozen rubric.

After holdout access is opened, the rubric must not be tuned again. If a defect is
found, report it as a limitation or start a new study with a new artifact and rubric
version; do not reinterpret the existing holdout.

## Agreement statistics

All statistics pair the human score with the judge score for the same sample.

### Raw agreement

Raw agreement is the fraction of pairs with identical scores:

```text
agreement = matching pairs / all pairs
```

It is directly interpretable but does not adjust for agreement expected from the
observed score marginals.

### Quadratic-weighted Cohen's kappa

The primary chance-adjusted statistic is quadratic-weighted Cohen's kappa over the
ordered labels 0, 1, and 2. A disagreement's weight is:

```text
w(i, j) = (i - j)^2 / (K - 1)^2, where K = 3
kappa = 1 - observed weighted disagreement / expected weighted disagreement
```

This makes a 0-versus-2 disagreement four times as costly as an adjacent 0-versus-1
or 1-versus-2 disagreement. If both raters use only one identical score, expected
disagreement is zero and kappa is reported as `null` rather than inventing a value.

### Confidence interval

EvalBench uses 10,000 paired nonparametric bootstrap resamples by default, seed 42,
and the 2.5th and 97.5th percentiles for a 95% interval. Each resample draws complete
human/judge pairs with replacement. Degenerate resamples with undefined kappa are
excluded; if every resample is undefined, both interval bounds are `null`.

### Diagnostic breakdowns

The report includes a 3-by-3 confusion matrix with human scores as rows and judge
scores as columns. It also reports sample count, raw agreement, weighted kappa, and
the confusion matrix for each suite category. Category subsets contain only seven
or eight holdout examples, so their statistics are descriptive and highly uncertain.

No agreement threshold is selected after seeing the holdout. The report must be
published even if agreement is weak.

## Reproduction procedure

Use exact installed model identifiers in place of the placeholders. Do not change
the suite, model builds, or generation settings between the paired target runs.

```bash
# 1. Run both target models over the same suite.
evalbench run --provider ollama --model TARGET_MODEL_A \
  --suite suites/core.yaml --database results/evalbench.db
evalbench run --provider ollama --model TARGET_MODEL_B \
  --suite suites/core.yaml --database results/evalbench.db
evalbench runs --database results/evalbench.db

# 2. Freeze sampling and split assignment before looking at labels.
evalbench judge select BASELINE_RUN_ID CANDIDATE_RUN_ID \
  --database results/evalbench.db \
  --output labels/judge-study.json \
  --sample-size 60 --seed 42

# 3. Complete calibration and preserve its report.
evalbench judge label labels/judge-study.json --split calibration
evalbench judge score labels/judge-study.json --split calibration \
  --judge-model gemma3:4b --database results/evalbench.db
evalbench judge report labels/judge-study.json --split calibration \
  --bootstrap-samples 10000 --seed 42 \
  --output labels/calibration-report.json

# 4. Clarify the rubric once, change its status to frozen, and commit that change.
evalbench judge freeze labels/judge-study.json

# 5. Complete the untouched holdout and generate the final report.
evalbench judge label labels/judge-study.json --split holdout
evalbench judge score labels/judge-study.json --split holdout \
  --judge-model gemma3:4b --database results/evalbench.db
evalbench judge report labels/judge-study.json --split holdout \
  --bootstrap-samples 10000 --seed 42 \
  --output labels/holdout-report.json
```

Before publication, commit the frozen rubric, study artifact, calibration report,
and holdout report together. Review model outputs for secrets or personal data before
making the artifact public. Record the Ollama version, exact installed model names,
hardware, operating system, source run IDs, and Git commit in the final result note.

## Interpretation and limitations

- The human scores are one person's judgments, not a consensus gold standard.
- Thirty holdout examples give a noisy overall estimate and very small category
  subsets; the bootstrap interval should be emphasized over the point estimate.
- Results apply to the recorded rubric, judge model, target outputs, and local runtime,
  not to LLM judges in general.
- Balanced sampling supports comparisons across the two target models and four
  categories but does not estimate real-world task prevalence.
- Expected references can make scoring more consistent, but their quality constrains
  both the human and model judge.
- Calibration is intentionally limited to one revision. This reduces holdout leakage
  but cannot eliminate researcher judgment in rubric design.
- Content-addressed caches make reruns auditable; they do not erase nondeterminism in
  the first uncached inference.

These limitations belong in the published result whether agreement is strong or weak.
