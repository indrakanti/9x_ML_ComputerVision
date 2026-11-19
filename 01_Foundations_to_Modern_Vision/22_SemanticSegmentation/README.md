# Module 22 — Semantic Segmentation

Semantic segmentation predicts a class label for every pixel.

## Learning objectives

You should be able to:

- distinguish semantic segmentation from detection
- understand per-pixel logits
- apply pixel-wise cross entropy
- convert logits to labels with argmax
- build a confusion matrix
- compute per-class IoU and mIoU
- understand ignore labels
- reproduce segmentation metrics in C++17
- reason about output resolution and memory cost

## 1. Dense prediction

For an input image of height $H$ and width $W$, a $C$-class segmentation model typically emits:

$$
[C,H,W]
$$

logits.

Prediction is:

$$
\hat y_{ij}=\arg\max_c z_{cij}
$$

## 2. Per-pixel loss

Training commonly uses cross entropy over all valid pixels.

An ignore label excludes pixels that should not contribute to the loss or metric.

## 3. Confusion matrix

For class $c$:

- TP: predicted $c$, ground truth $c$
- FP: predicted $c$, ground truth not $c$
- FN: ground truth $c$, predicted not $c$

Then:

$$
IoU_c=\frac{TP_c}{TP_c+FP_c+FN_c}
$$

Mean IoU is the average over evaluated classes.

## 4. Executable reference

The artifact contains:

- a 3-class, 2×3 ground-truth label map
- deterministic per-pixel logits
- one ignore label

Python and C++17 compute:

- predicted label map
- confusion matrix
- per-class IoU
- mIoU

## 5. Deployment implications

Segmentation outputs can be large.

A 1920×1080 image with many output channels can create significant memory traffic. Production designs often reduce resolution, use efficient decoders, or keep argmax/upsampling on-device.

## 6. Failure modes

- softmax before argmax when it is unnecessary
- wrong channel axis
- resizing labels with bilinear interpolation
- including ignore pixels in metrics
- averaging IoU over absent classes inconsistently
- comparing mIoU values from different label sets

## 7. Engineering rules

1. version label IDs
2. preserve ignore-label semantics
3. specify output resolution
4. specify interpolation mode
5. compute confusion matrices deterministically
6. keep metric definitions identical across languages

## 8. AV relevance

Semantic segmentation supports free-space, drivable-area, lane-region, curb, and scene-understanding tasks. Pixel-level outputs are valuable but expensive, so resolution and memory movement matter.

## 9. Book takeaway

> **Semantic segmentation turns classification into a dense spatial contract: every pixel has a label, and every metric depends on exact label semantics.**

## Reproducible commands

```bash
python 22_SemanticSegmentation/python/semantic_walkthrough.py
pytest 22_SemanticSegmentation/tests
cmake -S 22_SemanticSegmentation -B build-m22 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m22
ctest --test-dir build-m22 --output-on-failure
python 22_SemanticSegmentation/python/semantic_native_parity.py \
  build-m22/cv9x_semantic_metrics \
  22_SemanticSegmentation/artifacts/segmentation.txt
```

## Stage 1.5 applied lab

This module's deterministic implementation remains the mechanics reference. The real-data companion is:

- [Applied Lab 02 — Oxford-IIIT Pet Detection and Segmentation](../applied_labs/02_OxfordPetDetectionSegmentation/README.md)

The lab trains TinyYOLO, TinyDETR, and TinyUNet end to end using real public images and trimap masks while keeping dataset download optional.

## Next

Module 23 extends dense labels into instance-aware and panoptic outputs.
