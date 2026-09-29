# Graph masking and matched comparisons — 27 September 2026

Node dropout now excludes a post from the encoder and graph readout. Previously,
zeroed posts could still contribute through duplicate edges, self-loops, layer
biases, and the mean used for graph pooling.

## What changed

- Augmented views carry a node-retention mask and separate augmented reply and
  duplicate-content edge relations. Both relations lose edges incident to a
  dropped post, and both receive the configured edge-drop policy.
- The encoder temporarily compacts the retained graph before message passing.
  Deleted posts cannot contribute to either edge relation or to pooling.
- Root posts remain protected. Graph embeddings keep their original batch order,
  which preserves positive-pair alignment between contrastive views. Returned
  node embeddings retain their original row indices, with zero rows for deleted
  posts.
- Attribute masking remains distinct: a post with masked features is still a
  graph member. Node dropout removes its membership for that view.
- Contrastive training and the existing robustness callbacks consume the mask.
  Ordinary classification still uses the complete graph.

## Comparison protocol

`run_matched_comparisons.py` compares full ARAGCL-DP, the variant that ignores
duplicate edges in the encoder, and the static graph variant. The latter also
removes duplicate metadata, so it changes more than the encoder relation alone.

The experiment uses one fixed stratified training/validation split and a shared
TF-IDF feature space fitted only on training posts. Within each training seed,
the variants start from the same parameter values and use the same batch order,
training budgets, learning rate, and checkpoint-selection rule. Random state is
reset separately for each model and training stage. Full and no-duplicate-edge
models also sample the same supplied edge relations, avoiding augmentation RNG
differences caused only by disabling duplicate aggregation.

Different graph constructions can produce different importance scores, retained
nodes, and random-number consumption later in training. Matching seeds and
initial parameters does not make every augmentation identical across variants.

Report the individual seed results, mean and sample standard deviation, and
paired differences. These runs measure training-seed variability on this split;
they are not cross-validation and do not measure variability across data splits.

## Interpretation boundaries

- Reported metrics use the validation partition that selects checkpoints. An
  independent test evaluation is still required for final performance claims.
- The original 500-file Paperspace cohort was not available locally. A run using
  local training files is a new cohort, even if its sample size is also 500.
  Its results must not be used as a causal before/after comparison with the
  15 September scores.
- Identical normalized text defines inferred duplicate-content links; it does
  not verify actual resharing.
- This fix does not validate the old early-detection callback, which still uses
  feature masking and a batch-wide cutoff. It also does not correct the RQ3
  alignment/Hamming calculations or make timestamps part of the encoder.
- The dedicated runner performs the three classification comparisons and does
  not invoke those outstanding early-detection or representation diagnostics.

## Regression coverage

The masking tests compare masked encodings with explicitly constructed retained
graphs, check that changing deleted content cannot affect predictions, and check
zero gradients for deleted posts. They cover GCN and GAT, both edge relations,
root protection, singleton cascades, graph ordering, and empty retained graphs.
