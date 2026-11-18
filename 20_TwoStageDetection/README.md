# Module 20 — Two-Stage Detection and Region Proposals

Two-stage detectors separate candidate generation from final classification/refinement.

The canonical flow is:

```text
backbone feature map
      ↓
region proposal network
      ↓
proposal decode + ranking + NMS
      ↓
ROI feature extraction
      ↓
classification + box refinement
```

## Learning objectives

You should be able to:

- explain why two-stage detectors separate proposal generation from final prediction
- decode anchor-relative proposal deltas
- rank proposals by objectness
- apply proposal NMS
- understand ROI pooling / ROI Align motivation
- distinguish RPN objectness from final class confidence
- reproduce proposal generation in C++17
- compare one-stage and two-stage deployment tradeoffs

## 1. Region Proposal Network

An RPN predicts for each anchor:

- objectness
- box regression deltas

The RPN does not need to decide the final object class.

## 2. Delta decoding

For anchor center $(x_a,y_a)$ and size $(w_a,h_a)$:

$$
x=x_a+t_xw_a
$$

$$
y=y_a+t_yh_a
$$

$$
w=w_ae^{t_w}
$$

$$
h=h_ae^{t_h}
$$

Then convert center/size into xyxy coordinates.

## 3. Proposal filtering

Proposal generation usually includes:

```text
decode anchors
    ↓
clip to image
    ↓
remove invalid/small boxes
    ↓
sort by objectness
    ↓
pre-NMS top-K
    ↓
NMS
    ↓
post-NMS top-K
```

The exact ordering and thresholds are part of the deployment contract.

## 4. ROI feature extraction

The second stage needs a fixed-size feature representation for variable-size proposals.

ROI Pooling quantizes regions into bins and applies pooling.

ROI Align avoids coarse coordinate quantization by sampling continuous feature locations.

This module includes a compact ROI-pooling example for intuition; production detectors generally use optimized ROI Align operators.

## 5. One-stage vs two-stage

One-stage:

```text
dense prediction → decode → NMS
```

Two-stage:

```text
dense proposals → proposal NMS → ROI features → final head
```

Two-stage detectors can spend more computation on a smaller set of candidate regions.

## 6. C++17 native counterpart

The native reference loads:

- anchors
- deltas
- objectness scores
- top-K policy
- NMS threshold

and emits the same final proposal list as Python.

## 7. Deployment implications

Two-stage pipelines introduce additional synchronization and memory considerations:

- proposal tensor materialization
- top-K sorting
- NMS
- ROI gather/sampling
- second-stage batching
- variable proposal count

Static graph execution may still contain dynamic work because proposal count varies by frame.

## 8. Failure modes

- mixing center-size and xyxy coordinates
- applying deltas with the wrong anchor width/height
- clipping before versus after filtering inconsistently
- confusing RPN objectness with final class scores
- changing pre-NMS top-K between runtimes
- ROI coordinate scale mismatch
- variable proposal count causing hidden latency tails

## 9. Engineering rules

1. version anchor convention
2. make proposal top-K explicit
3. preserve objectness ordering
4. share NMS implementation semantics
5. distinguish proposal and final-head confidence
6. monitor worst-case proposal counts
7. measure second-stage latency as a function of proposal count

## 10. AV relevance

Two-stage detection is useful for understanding the accuracy/latency trade-space even when a production AV stack chooses a one-stage or transformer-based detector. Proposal-count variability is especially relevant to deterministic compute budgets.

## 11. Book takeaway

> **Two-stage detection spends computation selectively: first decide where to look, then classify and refine those regions. That selectivity introduces its own dynamic-runtime contract.**

## Reproducible commands

```bash
python 20_TwoStageDetection/python/two_stage_walkthrough.py
pytest 20_TwoStageDetection/tests
cmake -S 20_TwoStageDetection -B build-m20 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m20
ctest --test-dir build-m20 --output-on-failure
python 20_TwoStageDetection/python/two_stage_native_parity.py \
  build-m20/cv9x_two_stage \
  20_TwoStageDetection/artifacts/proposals.txt
```

## Next

Module 21 introduces DETR-style set prediction and transformer detection.
