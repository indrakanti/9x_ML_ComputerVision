# Module 10 — Regularization, Augmentation, Schedulers & Checkpointing

Module 09 showed how to train a CNN. Module 10 asks the next production question:

> How do we make training harder to overfit, easier to resume, and safer to reproduce?

This chapter adds four controls that appear in real training systems:

- regularization
- data augmentation
- learning-rate scheduling
- checkpoint policy

The code stays intentionally small so each mechanism remains visible.

## Learning objectives

You should be able to:

- explain overfitting and generalization
- distinguish L2 regularization from optimizer weight decay
- explain dropout training vs inference behavior
- implement deterministic image augmentation
- use a learning-rate scheduler correctly
- save latest and best checkpoints
- resume optimizer and scheduler state
- record training provenance in a deployment manifest
- explain why augmentation/dropout do not belong in native inference
- validate deployment metadata in Python and C++17

## 1. Overfitting

A model overfits when it becomes increasingly good on its training samples while failing to generalize.

A useful mental model is:

```text
training error ↓
validation error initially ↓
validation error later ↑
```

The goal is not to minimize training loss at any cost.

The goal is to learn a representation that performs well on unseen data.

## 2. L2 regularization and weight decay

A common regularized objective is:

$$
L_{total}
=
L_{data}
+
\lambda \sum_i w_i^2
$$

This discourages unnecessarily large weights.

In many optimizers, `weight_decay` is the practical interface.

Important distinction:

- L2 penalty is a term in the loss
- decoupled weight decay can be implemented separately from the gradient of the data loss

The exact optimizer semantics matter.

## 3. Dropout

During training, dropout randomly masks activations:

$$
\tilde{h}_i
=
\frac{m_i h_i}{1-p}
$$

where $m_i$ is Bernoulli-distributed.

During inference:

```text
dropout is disabled
```

This is why `model.train()` and `model.eval()` are not cosmetic.

## 4. Data augmentation

Augmentation changes the training distribution without changing the target label.

This module uses deterministic transforms suitable for the synthetic pattern task:

- horizontal flip
- vertical flip
- additive bounded noise

The transform receives an explicit random generator so tests are reproducible.

Production rule:

> augmentation configuration is part of experiment provenance.

## 5. Learning-rate scheduling

A scheduler changes the optimizer learning rate over time.

For StepLR:

$$
\eta_t
=
\eta_0 \gamma^{\lfloor t/s \rfloor}
$$

where:

- $\eta_0$ is initial learning rate
- $s$ is step interval
- $\gamma$ is decay factor

The ordering matters:

```text
optimizer.step()
scheduler.step()
```

unless a specific scheduler documents different semantics.

## 6. Best vs latest checkpoint

These are not the same artifact.

### latest

Used to resume interrupted training.

It should include:

- epoch
- model state
- optimizer state
- scheduler state
- best metric so far
- configuration/provenance

### best

Used to preserve the model that achieved the best validation metric.

A later epoch may be worse than an earlier one.

Therefore:

```text
latest != best
```

## 7. Resume semantics

Correct resume means restoring the full optimization state.

Restoring only model weights changes the future training trajectory because the optimizer and scheduler states are missing.

The minimum contract is:

```text
model
optimizer
scheduler
epoch
best metric
training configuration
```

## 8. Deployment manifest

Training-only mechanisms should not leak into native inference.

The module writes a small deployment manifest:

```text
version=1
selected_checkpoint=best
dropout_inference=disabled
augmentation_inference=disabled
preprocessing_revision=synthetic-v1
class_count=2
```

The manifest records decisions that matter to deployment without embedding optimizer state in the runtime.

## 9. C++17 counterpart

The native counterpart validates the same manifest.

It checks:

- supported manifest version
- best/latest selection value
- dropout disabled for inference
- augmentation disabled for inference
- positive class count
- preprocessing revision present

This is a meaningful native contract because production code should reject inconsistent deployment metadata before model execution.

## 10. Python ↔ C++ parity

Both runtimes parse the same manifest and emit the normalized contract.

CI compares:

- selected checkpoint
- dropout inference policy
- augmentation inference policy
- preprocessing revision
- class count

## 11. Failure modes

- augmenting validation data accidentally
- applying random transforms during inference
- leaving dropout in training mode
- resuming model weights without optimizer state
- stepping the scheduler at the wrong time
- overwriting the best checkpoint with a worse epoch
- changing augmentation without recording it
- using a deployment checkpoint whose preprocessing contract changed
- confusing training checkpoint metadata with inference runtime metadata

## 12. Engineering rules

1. keep training and validation transforms separate
2. seed stochastic transforms in reproducible tests
3. record augmentation configuration
4. record regularization configuration
5. checkpoint optimizer and scheduler state
6. distinguish latest from best
7. restore full state on resume
8. disable training-only behavior at inference
9. validate deployment metadata in native code
10. make preprocessing revision explicit

## 13. Relation to autonomous-driving perception

These controls become essential for the later surround-camera detector:

```text
multi-camera training data
      ↓
augmentation policy
      ↓
regularized detector training
      ↓
scheduler / checkpoints
      ↓
best validated model
      ↓
deployment manifest
      ↓
C++ / GPU runtime
```

A detector that cannot be reproduced or whose preprocessing contract is unclear is not production-ready.

## 14. Book takeaway

> **A robust training system is not just a model and an optimizer. It is a controlled experiment whose stochastic behavior, schedule, checkpoint state, and deployment contract are all explicit.**

## 15. Exercises

1. Add random brightness augmentation.
2. Compare SGD weight decay with AdamW.
3. Add early stopping.
4. Add ReduceLROnPlateau.
5. Save augmentation configuration in the checkpoint.
6. Resume training and verify learning-rate continuity.
7. Corrupt the deployment manifest and verify C++ rejects it.
8. Add model-artifact checksum metadata.
9. Add dataset revision metadata.
10. Explain which training controls must disappear during inference.

## Next

Module 11 introduces transfer learning and the first full ONNX/native inference checkpoint.
