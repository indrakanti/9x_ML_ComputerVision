# Module 21 — DETR-Style Set Prediction

DETR reframes detection as direct set prediction. Instead of dense anchors followed by NMS, a fixed number of object queries predict a set of detections and training uses bipartite matching.

## Learning objectives

You should be able to:

- explain fixed object queries
- distinguish set prediction from dense anchor assignment
- build a matching cost from class and box terms
- explain one-to-one Hungarian matching
- understand the no-object class
- explain why DETR can avoid NMS
- reproduce deterministic assignment in C++17
- connect matching quality to training stability

## 1. Set prediction

A DETR-style detector emits a fixed number of prediction slots:

```text
query 0 → class + box
query 1 → class + box
...
query N → class + box
```

Ground truth is an unordered set. Training therefore needs a permutation-invariant assignment.

## 2. Matching cost

This module uses:

$$
C_{ij}
=
\lambda_{cls} C_{cls}
+
\lambda_{L1}\|b_i-g_j\|_1
+
\lambda_{IoU}(1-IoU(b_i,g_j))
$$

Lower cost is better.

The class term here is a simple negative probability for the target class so the arithmetic remains transparent.

## 3. Bipartite assignment

Each ground-truth object is matched to at most one prediction, and each prediction to at most one ground truth.

For the tiny executable example, the assignment is solved exactly by enumerating permutations. Production DETR implementations use optimized Hungarian matching.

## 4. No-object predictions

Queries not matched to a ground-truth object are trained toward a no-object class.

This fixed-size output contract is one reason DETR can remove the traditional NMS stage.

## 5. Native contract

The artifact contains:

- predicted boxes
- predicted class probabilities
- ground-truth boxes/classes
- cost weights

Python and C++17 compute the same cost matrix and optimal assignment.

## 6. Deployment implications

Inference is conceptually simpler than anchor-based detection:

```text
model outputs fixed queries
      ↓
class filtering
      ↓
boxes
```

But transformer decoder cost, query count, and attention memory still matter.

## 7. Failure modes

- comparing boxes in inconsistent coordinate systems
- using class logits/probabilities inconsistently
- wrong cost weights
- treating matching as greedy
- forgetting the no-object class
- assuming NMS-free means post-processing-free

## 8. Engineering rules

1. version matching-cost weights
2. preserve class ordering
3. preserve box convention
4. test the full cost matrix
5. test assignment, not only final loss
6. keep query count explicit

## 9. AV relevance

Set-based detection is attractive for scenes with many interacting objects because the detector learns a global allocation of prediction slots instead of relying on local suppression heuristics.

## 10. Book takeaway

> **DETR turns detection into permutation-invariant set prediction; one-to-one matching replaces anchor assignment and duplicate suppression during training.**

## Reproducible commands

```bash
python 01_Foundations_to_Modern_Vision/21_DETR/python/detr_walkthrough.py
pytest 01_Foundations_to_Modern_Vision/21_DETR/tests
cmake -S 01_Foundations_to_Modern_Vision/21_DETR -B build-m21 -G Ninja -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m21
ctest --test-dir build-m21 --output-on-failure
python 01_Foundations_to_Modern_Vision/21_DETR/python/detr_native_parity.py \
  build-m21/cv9x_detr_match \
  01_Foundations_to_Modern_Vision/21_DETR/artifacts/matching.txt
```

## Stage 1.5 applied lab

This module's deterministic implementation remains the mechanics reference. The real-data companion is:

- [Applied Lab 02 — Oxford-IIIT Pet Detection and Segmentation](../../applied_labs/02_OxfordPetDetectionSegmentation/README.md)

The lab trains TinyYOLO, TinyDETR, and TinyUNet end to end using real public images and trimap masks while keeping dataset download optional.

## Next

Module 22 moves from object sets to dense per-pixel semantic prediction.
