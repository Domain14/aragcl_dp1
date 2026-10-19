# PHEME event-aware classification experiment

The PHEME adapter reads the `all-rnr-annotated-threads` directory layout.
It maps `rumour` to 1 and `nonrumour` to 0, preserves the nested reply tree,
uses text from the source and reaction JSON files, and records any excluded
unclear or incomplete threads. Multiple top-level components remain separate
components in one graph; the loader does not invent reply edges between them.

The pre-specified protocol is in
`experiment_protocols/pheme_event_aware.json`. The primary comparison keeps
the ARAGCL-DP model fixed and compares reply-only graphs with the same graphs
plus exact normalized within-thread content links. The rule is fixed to
non-empty text after trimming outer whitespace and lowercasing. These are
inferred links, not verified retweet provenance.

The read-only loader audit of the supplied copy found 6,423 usable threads:
2,401 rumour and 4,022 non-rumour. One unclear thread and one incomplete
thread are excluded. The structure contains 84 multi-component forests; all
components are retained without adding edges between them. Exact normalized
duplicate matching yields 382 links across 253 threads (about 3.9% of
threads), so the intervention is sparse in this dataset. This is descriptive
dataset evidence, not a model-performance result.

## Run

First copy the project and PHEME data to the Paperspace machine. In the
project folder, run the read-only audit:

```bash
python run_pheme_event_aware.py \
  --dataset-dir /storage/all-rnr-annotated-threads \
  --audit-only
```

The audit reports label counts and the exact leave-one-event-out fold sizes.
If the audit completes, launch the matched experiment in a fresh output
folder:

```bash
python run_pheme_event_aware.py \
  --dataset-dir /storage/all-rnr-annotated-threads \
  --output-dir results/pheme-loeo-v1 \
  --device cuda
```

Use `--device cpu` if CUDA is unavailable. Defaults are four training seeds,
15 contrastive pretraining epochs, up to 25 supervised epochs, patience 6,
batch size 8, and at most 2,048 character TF-IDF features. TF-IDF is fitted
only on the inner training partition separately in each event fold. The
held-out event is used only for final prediction after validation-based
checkpoint selection.

The run performs 9 event folds x 4 seeds x 2 graph conditions. Save the
entire output directory. `summary.json` reports pooled out-of-fold metrics
and paired differences; `seed_fold_metrics.json` and the per-run folders
retain event-level scores, predictions, histories, and checkpoints. A run
that stops early must not be summarized as a complete leave-one-event-out
result.

Do not alter the matching rule, model, lambda, feature settings, or stopping
rule after inspecting held-out event scores. Any changed design is a new
exploratory experiment and should be labelled accordingly.
