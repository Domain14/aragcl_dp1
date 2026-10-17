# First experiment: duplicate-message direction

Question: does gathering information from later matching posts into their
earliest reference help rumor classification more than sending information
from that reference to each copy?

This tests one modeling choice. It does not guarantee better results.

## What is now explicit

The data builder still links existing posts using nonblank text equality after
trimming whitespace and lowercasing. It adds no synthetic nodes. Its stored
edges point from the earliest matching reference to each later matching post.
Ties in timestamps follow input order. These are inferred content links, not
verified repost paths; the available data lacks explicit reshare-source IDs.

The encoder has two message directions:

| Setting | What receives duplicate information? |
|---|---|
| `reference_to_copy` (existing default) | Each later match receives information from the earliest reference. |
| `copy_to_reference` (new experiment) | The earliest reference receives information from the later matches in its stored duplication set. |

The proposal's Section 3.7 describes reference-to-copy propagation edges,
while Equation 3.7 describes gathering information from a node's duplication
set. This experiment makes that choice explicit. Reversing the stored star
does not implement an all-pairs same-content graph and is not an unequivocal
correction to the proposal.

Only duplicate messages reverse inside the encoder. Reply messages, stored
links, timestamps, duplicate counts, node deletion, feature masking and the
augmentation rules are unchanged. Both GCN and GAT support the option. Old
configurations omit it and retain their previous behavior; new checkpoints
save the setting so they can be reconstructed accurately.

## Small correctness example

For four artificial node features `[10, 2, 3, 5]`, with stored links `1 -> 2`
and `1 -> 3`, the duplicate-message sums are:

- Reference to copies: `[0, 0, 2, 2]`.
- Copies to reference: `[0, 8, 0, 0]`.

For a GCN with no reply edges, self-loops, identity weights, zero bias and
lambda 0.5, the corresponding outputs are `[10, 2, 4, 6]` and
`[10, 6, 3, 5]`. This is a correctness check, not a performance result.

## The next Paperspace run

Run from the updated `aragcl_dp` folder:

```bash
python run_matched_comparisons.py --dataset-dir /storage/weibo1 \
  --output-dir results/duplicate-direction-2026-09-28 \
  --models full copy_to_reference no_dup_edges static \
  --seeds 0 1 2 3 4 --split-seed 0 \
  --pretrain-epochs 15 --finetune-epochs 25 --patience 6 \
  --batch-size 8 --device cpu --num-threads 1
```

Use the same raw-data folder and sampling arguments as the previous run. The
command uses all JSON files in that folder. Only use `--max-cascades 500
--sample-seed 0` if the previous run used those settings to sample a larger
folder. Use a new, empty output directory. This produces 20 runs: four
conditions across five training seeds.

`full` is the existing reference-to-copy model. `copy_to_reference` changes
only the encoder's duplicate-message direction. `no_dup_edges` keeps duplicate
metadata for augmentation but switches off duplicate aggregation. `static`
removes duplicate links and metadata. All four share the same data split,
training-only text features, initial weights and stage-specific random seeds.
The two direction variants use the same augmentation sampling operations.

Keep lambda at 0.5 and the existing duplicate-matching rule for this experiment.
The primary comparison is copies-to-reference versus full; compare both with
no-duplicate aggregation and the static graph as controls. Inspect all five
paired F1 differences, their mean and sample standard deviation, plus precision
and recall. Do not select a favorable seed. Similar results are a useful finding:
direction alone may not address the model's limitation.

Scores remain validation-only because the same validation set selects the
checkpoint and scores it. Five training seeds on a fixed split do not provide
five-fold cross-validation or independent test performance. Changes involving
text filtering, augmentation strength or lambda should be separate, recorded
experiments, with final claims checked on a reserved test set.

The runner saves `summary.json`, `seed_metrics.csv`, checkpoints, per-cascade
predictions, learning histories, file hashes, configuration and a source
snapshot. Share the results folder after the run so matching and predictions
can be checked, in addition to the console log.
