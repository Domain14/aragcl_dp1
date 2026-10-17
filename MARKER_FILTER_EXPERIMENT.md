# Next experiment: exclude one generic marker from duplicate links

Research question: does excluding the exact generic text `转发微博` from
duplicate-link construction improve ARAGCL-DP compared with its current
copies-to-reference version?

The rule was chosen from the previous run's training-data audit: this marker
accounted for 255 of 852 inferred training links. It is a hypothesis about
link quality, not a proven performance improvement or reshare detector.

## What changes

`duplicate_policy="exclude_weibo_marker"` suppresses duplicate links only
when the entire text equals `转发微博` after trimming surrounding whitespace
and lowercasing. It does not remove posts, text features or reply connections.
It does not exclude other short text, substrings, punctuation variants or
near matches. Marker text remains in the training-only TF-IDF vocabulary.

Duplicate sets and frequencies reflect the retained links. As a result,
duplication-based importance and augmentation can also change. This is a
test of the full filtering intervention, not encoder aggregation alone.
Default commands still use the original matching rule.

## Run in Paperspace

Extract the updated package into a new code folder and run from its
`aragcl_dp` directory. Keep your existing `/storage/weibo1` raw-data folder.

```bash
python run_matched_comparisons.py --dataset-dir /storage/weibo1 \
  --output-dir results/marker-filter-2026-09-28 \
  --models copy_to_reference filtered_copy_to_reference \
           no_dup_edges filtered_no_dup_edges static \
  --seeds 0 1 2 3 4 --split-seed 0 \
  --pretrain-epochs 15 --finetune-epochs 25 --patience 6 \
  --batch-size 8 --device cpu --num-threads 1
```

This runs five conditions across five training seeds: 25 runs. Use a new,
empty output directory. The supplied earlier manifest used all 500 files in
`/storage/weibo1` with no subsampling; the command preserves that choice.
Lambda remains 0.5, base augmentation rates remain 0.2, and all conditions
share one split, one training-only text-feature fit and matched initial
weights. Stage-specific random seeds are reset for every condition.

| Condition | Duplicate links used by encoder | Duplicate metadata used by augmentation |
|---|---|---|
| `copy_to_reference` | Current unfiltered links, copies to reference | Unfiltered |
| `filtered_copy_to_reference` | Marker-filtered links, copies to reference | Filtered |
| `no_dup_edges` | Disabled | Unfiltered |
| `filtered_no_dup_edges` | Disabled | Filtered |
| `static` | None | Removed |

The primary comparison is `filtered_copy_to_reference` versus
`copy_to_reference`. The filtered no-message control helps assess the effect
of filtered inputs on augmentation. Within each filtering policy, enabling
versus disabling duplicate aggregation uses the same sampled views. Across
policies, removing links changes both probabilities and the number of random
draws, so identical seeds do not guarantee identical masks. The filter
comparison must not be described as an isolated centrality experiment.

## Preflight on the exact earlier cohort

All 500 raw files match the previous Paperspace manifest. The split, batch
order, text vocabulary, IDF values and unfiltered graph input fingerprints
match it too. This verification built graphs without training new models.

| Partition | Posts retained | Reply edges retained | Duplicate links before → after | Affected conversations |
|---|---:|---:|---:|---:|
| Training: 400 conversations | 24,820 | 24,420 | 852 → 597 | 81 |
| Validation: 100 conversations | 5,665 | 5,565 | 204 → 131 | 14 |

The filter changes duplicate frequency, and consequently duplication-based
importance, at 81 training nodes and 14 validation nodes. Validation duplicate
conversations decrease from 42 to 36, but every conversation remains present.

## What gets saved

The usual checkpoints, predictions, histories, summaries and source snapshot
remain. Additional evidence includes:

- `duplicate_filter_audit.json`: before/after counts for each conversation,
  train/validation totals and explicit confirmation of preserved fields.
- Each run's `augmentation_usage.json`: actual node deletion, sampled
  attribute masking, masking among surviving nodes, and removal of both edge
  relations during pretraining. Rates state their denominators. Edge removal
  includes both node deletion and edge dropout; zero-edge rates are null.
- `graph_construction` in metrics and checkpoints: whether inference needs
  the marker filter and whether the representation is static. Reconstruct
  the inputs with this policy when loading a checkpoint.

Recording uses no extra random draws and stores only counts. Tests confirm
unchanged outputs, gradients, random state and control training results.
Counts represent exposures over repeated views and epochs, not unique posts.

## How we will evaluate it

Report all five paired F1 differences, mean and sample standard deviation,
precision, recall and accuracy. Review predictions and actual augmentation
rates. For subgroup analysis, keep the original 42 duplicate-containing
validation conversations fixed; the 14 filter-affected conversations can be
examined separately. Do not compare the old 42-case group with the new
36-case group as though they were the same cases.

An improvement, tie or deterioration is informative. Keep the rule fixed
for this run rather than widening it based on favorable validation outcomes.
These results remain exploratory: the same validation set selects the best
checkpoint and scores it. Five training seeds on one split are not five-fold
cross-validation, and final claims still need reserved test evaluation.

After completion, share the new results folder as a ZIP so the saved
predictions, data settings and augmentation usage can be checked together.
