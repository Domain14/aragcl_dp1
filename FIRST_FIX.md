ARAGCL-DP — first implementation repair

This working copy repairs evaluation mode and augmentation probability handling. It is a development version. The other methodological limitations identified in the research review remain open.

Changes

- Validation, final comparison scores, lambda-sweep scoring, robustness scoring, and representation extraction now disable model dropout and gradient recording. Each evaluation restores the model's previous mode afterward.
- Supervised and contrastive training explicitly enter training mode every epoch.
- The existing best-checkpoint deep copy and restoration are retained.
- Random node, edge, and attribute augmentation use their configured base probabilities directly.
- Equal importance scores use uniform base probabilities. This is an explicit fallback policy: equal scores provide no ranking. The policy needs to be stated in the methodology and produces different behavior from the old zero-probability fallback.
- Nonconstant importance scores preserve their relative probabilities even at very small numerical scales. Empty edge sets, invalid rates, and nonfinite importance scores have explicit handling.
- Root features remain exempt from node/attribute masking. Root-incident edges can be dropped by edge augmentation, consistent with the existing edge-drop rule.

Verification in Paperspace

Use this folder as a separate project copy. In its terminal, with the project's dependencies installed, run:

```bash
python -m unittest discover -s tests -v
```

The checks include a tiny integration run through the JSON loader and the full model runner. All cascades in that check are fabricated: it reads none of the real training, validation, or test files, and its scores are not research results. The test compares each final reported metric with the selected checkpoint's metric and repeats ARAGCL-DP inference while advancing the random generator between calls.

The test suite does not require extra test-framework packages beyond the project dependencies. Its CUDA-specific check runs only if CUDA is available. Local package versions and results are recorded in `VALIDATION.json`; the local CPU validation environment does not prescribe a CUDA/PyTorch downgrade for Paperspace.

Notebook API change

The evaluator now requires the actual model so it can control dropout:

```python
val_loss, metrics = evaluate(model, forward_fn, val_batches)
```

All supplied scripts have been updated. Update external notebook calls that still use the old two-argument form. For direct predictions:

```python
from train.trainer import evaluation_mode

with evaluation_mode(model):
    logits = model.classify(graph, batch=batch_idx)
```

This makes inference mode explicit; it does not suppress intentionally random graph perturbations in robustness experiments.

Before a scientific comparison

- Node dropping still zeros features and removes reply edges while retaining node rows and original duplication edges. Complete graph-wide deletion/masking and readout behavior still need a defined implementation.
- The RAGCL and GRACE factory configurations remain equivalent procedures. Baseline labels and published-method fidelity still require correction.
- Inferred same-text relationships remain the duplication mechanism. No synthetic node creation or verified repost-provenance reconstruction has been added.
- The structural-fidelity reference remains circular; the early-detection callback retains future topology; timestamp jitter is not consumed by the model; and RQ3 duplicate alignment is not wired to actual node pairs.
- The existing runners use repeated validation holdouts, not an independent final test evaluation. Their device configuration is not yet wired to move the pipeline onto a GPU.

Run the verification suite first. Address these remaining issues before interpreting a new full comparison as final thesis evidence. Earlier scores used different augmentation behavior and dropout-active validation; a corrected experiment needs new training runs after the method is settled.
