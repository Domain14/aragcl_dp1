# Controlled comparison of duplicate-message influence

Research question: with the duplicate graph, message direction and augmented
views held fixed, does changing the weight of duplicate messages improve
validation F1?

## Run in Paperspace

Extract this entire package into a new folder. In a terminal opened inside
its `aragcl_dp` folder, run:

```bash
python run_lambda_comparison.py
```

In a notebook whose working directory is that same folder, use:

```python
!python run_lambda_comparison.py
```

Use the existing Python environment that ran your last experiment. The runner
uses all files in `/storage/weibo1`, your existing 500-conversation cohort.
Keep that folder and its contents unchanged to match the earlier experiments.
No new data sampling or package installation is needed for this comparison.

Results are written to `results/lambda-comparison-2026-09-29` under the
directory from which you run the command. Existing nonempty results folders
are protected. To repeat the experiment, supply a new output folder:

```bash
python run_lambda_comparison.py --output-dir results/lambda-comparison-repeat-01
```

The dedicated runner fixes the four weights below. The earlier
`sweep_lambda.py` does not implement this matched experiment protocol.

## What the command compares

| Saved variant | Lambda | Meaning |
|---|---:|---|
| `lambda_0` | 0 | Duplicate messages contribute zero to the encoder |
| `lambda_0p1` | 0.1 | Lower duplicate-message influence |
| `lambda_0p25` | 0.25 | Intermediate duplicate-message influence |
| `lambda_0p5` | 0.5 | Previously used influence |

All four retain the same unfiltered inferred duplicate links, including
exact `转发微博` matches, and their duplicate metadata for augmentation.
All four enable the same duplication-aware architecture and direct duplicate
messages from later copies to the earliest matching reference. Only lambda
changes between these variants. Lambda zero does not remove duplicate
metadata from augmentation; it is not the static-graph baseline.

Fixed settings: seeds 0, 1, 2, 3 and 4; split seed 0; the same stratified
80/20 split; a shared training-only character TF-IDF fit; batch size 8;
15 pretraining epochs; up to 25 fine-tuning epochs; patience 6; CPU with
one thread. This gives 20 runs. Checkpoint selection remains maximum
validation F1 with the first epoch winning ties, and class predictions
still use the largest logit. No threshold tuning is added.

Each seed starts every variant from identical model weights and resets
pretraining, fine-tuning and evaluation random seeds separately. The runner
also fingerprints the actual ordered augmented views, including features,
node masks, attribute masks, both edge relations and timestamps. It rejects
a comparison if those traces differ within a seed. This checks the realized
views, in addition to matching their aggregate corruption counts.

## Saved evidence

Each seed/variant folder includes a selected checkpoint, predictions,
training history, metrics, actual augmentation usage and an ordered
augmentation trace. The metrics and checkpoint record the trace fingerprint.
The main folder includes:

- `summary.json`: mean, sample standard deviation and paired F1 differences.
- `seed_metrics.csv` and `seed_metrics.json`: individual run results.
- `manifest.json`: settings, source/data fingerprints, split, model configs
  and whether the augmentation traces matched within each seed.
- `source_snapshot.zip`: the code used for the experiment.

When finished, share the entire `results/lambda-comparison-2026-09-29`
folder as a ZIP so the scores and their supporting evidence can be reviewed.

## What the results can establish

Review all five paired F1 differences for each weight, alongside precision,
recall and accuracy. Improvement is not guaranteed. Lambda 0 and 0.5 provide
checks against the earlier no-duplicate-message and copies-to-reference
controls when cohort, inputs, environment and settings match.

Local fixture tests verify identical augmented views, execution-order
independence and exact endpoint reproduction of those controls. They are
software checks, not new Weibo research results. The full 20-run research
experiment is to be run in Paperspace.

These are exploratory results on one validation partition. The same
validation data selects checkpoints and is reused to compare lambda values.
Five training seeds are not five-fold cross-validation. Freeze the selected
configuration before an independent held-out evaluation for final claims.
Exact repeated text establishes inferred content duplication, not verified
resharing relationships.
