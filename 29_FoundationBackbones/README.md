# Module 29 — Universal / Foundation Backbones

A foundation backbone is valuable when one reusable visual representation can support many downstream tasks.

The systems pattern is:

```text
image / frame
    ↓
large pretrained backbone
    ↓
shared feature representation
    ├→ classification adapter
    ├→ detection adapter
    ├→ segmentation adapter
    └→ retrieval / language adapter
```

The important contract is not only the backbone weights. It is the **feature interface** exposed to downstream tasks.

## Learning objectives

You should be able to:

- explain the difference between a task-specific model and a reusable foundation backbone
- define a stable feature contract
- distinguish frozen-backbone transfer from full fine-tuning
- implement lightweight linear task adapters
- compute adapter parameter overhead
- reason about multi-task feature reuse
- reproduce adapter outputs in C++17
- understand versioning, normalization, and feature-shape risks

## 1. Shared representation

A foundation encoder maps input $x$ into a reusable feature vector or feature hierarchy:

$$
f = E(x)
$$

A task-specific adapter then computes:

$$
y_t = A_t(f)
$$

The backbone can remain frozen while adapters change.

## 2. Feature contract

A downstream task depends on more than feature length.

A robust contract may include:

- model/version identifier
- feature dimension
- feature level / layer name
- tensor layout
- spatial stride
- normalization
- dtype
- preprocessing revision
- coordinate convention

Changing any of these can silently break an adapter even when tensor dimensions still match.

## 3. Frozen backbone + adapters

This module uses a deterministic 4-D shared feature vector.

Two adapters consume the same features:

- 3-class classification adapter
- 2-D dense/geometry adapter

Each adapter is a linear projection:

$$
y = Wf+b
$$

The adapter is tiny compared with a hypothetical large backbone.

## 4. Parameter efficiency

For feature dimension $D$ and output dimension $O$, a linear adapter uses:

$$
DO + O
$$

parameters.

If a backbone has millions or billions of parameters, adding small adapters can support many tasks with little incremental storage.

## 5. Multi-task reuse

Foundation-backbone economics improve when multiple tasks reuse the same expensive representation.

But reuse is only valid when the feature contract is appropriate for those tasks.

A classification-friendly global embedding may be insufficient for dense geometry. Multi-scale/spatial features may be required.

## 6. Executable reference

The artifact contains:

- feature-contract version
- feature dimension
- frozen shared feature vector
- classification adapter weights/bias
- geometry adapter weights/bias
- notional backbone parameter count

Python computes:

- classification logits
- geometry outputs
- adapter parameter counts
- adapter overhead ratio

C++17 reproduces the adapter outputs and parameter counts.

## 7. Feature versioning

A foundation backbone should be treated like a platform API.

If an adapter was trained against:

```text
backbone=v3
feature_level=stage4
dimension=1024
normalization=layernorm-v2
```

then changing any of those values should trigger explicit compatibility review.

## 8. Foundation-model tradeoffs

Large universal backbones can improve transfer but create costs:

- larger model storage
- higher latency
- larger activation memory
- complex licensing/provenance
- expensive pretraining
- deployment-runtime support requirements
- adaptation/evaluation burden across tasks

The largest model is not automatically the best production backbone.

## 9. Deployment implications

Possible deployment strategies include:

1. **shared runtime backbone**
   - execute once
   - distribute features to multiple heads

2. **task-specific compiled models**
   - duplicate backbone execution
   - simpler task isolation

3. **offline foundation backbone**
   - use foundation model to bootstrap labels/features
   - deploy smaller distilled task model

The right architecture depends on latency, isolation, safety, memory, and update cadence.

## 10. Failure modes

- changing feature normalization without retraining adapters
- using a global embedding for a task needing spatial detail
- assuming adapter compatibility across backbone revisions
- counting only adapter parameters and ignoring backbone runtime cost
- sharing a backbone across tasks without fault-containment analysis
- ignoring feature tensor lifetime and fan-out bandwidth

## 11. Engineering rules

1. version the feature interface
2. record feature layer and shape
3. record preprocessing/normalization
4. quantify adapter parameter overhead
5. validate every adapter after backbone changes
6. measure shared-backbone fan-out cost
7. distinguish training reuse from runtime sharing

## 12. AV relevance

A reusable visual backbone could feed detection, segmentation, depth, free-space, or scene-understanding heads.

For a safety-critical AV platform, however, runtime sharing must be evaluated against fault containment, compute determinism, and independent update needs.

## 13. Book takeaway

> **A foundation backbone is a platform interface, not just a large model. Reuse succeeds only when the feature contract is stable, versioned, and appropriate for every downstream task.**

## Reproducible commands

```bash
python 29_FoundationBackbones/python/foundation_walkthrough.py
pytest 29_FoundationBackbones/tests
cmake -S 29_FoundationBackbones -B build-m29 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m29
ctest --test-dir build-m29 --output-on-failure
python 29_FoundationBackbones/python/foundation_native_parity.py \
  build-m29/cv9x_foundation_adapters \
  29_FoundationBackbones/artifacts/foundation.txt
```

## Next

Module 30 studies linear probing, adapters, LoRA, and parameter-efficient fine-tuning.
