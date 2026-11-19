# Module 18 — Detection Fundamentals: Boxes, IoU, NMS and mAP

Object detection adds a localization contract to classification. A detector must answer both **what** is present and **where** it is.

## Learning objectives

You should be able to:

- represent boxes consistently as `[x1,y1,x2,y2]`
- compute intersection-over-union (IoU)
- understand anchors and box offsets
- apply confidence thresholds
- perform class-aware non-maximum suppression (NMS)
- build precision/recall curves from ranked detections
- compute interpolated average precision (AP)
- explain mean average precision (mAP)
- reproduce IoU/NMS decisions in C++17
- understand why metric definitions must be versioned

## 1. Bounding-box contract

This module uses continuous coordinates:

$$
b=(x_1,y_1,x_2,y_2),\quad x_2>x_1,\quad y_2>y_1
$$

Area is:

$$
A(b)=(x_2-x_1)(y_2-y_1)
$$

Coordinate convention must be explicit. Inclusive integer-pixel formulas are different.

## 2. Intersection over Union

For boxes $A$ and $B$:

$$
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

IoU is zero for non-overlapping boxes and one for identical boxes.

## 3. Anchors and offsets

Anchor-based detectors predict corrections relative to a reference box.

A common parameterization is:

$$
t_x=\frac{x-x_a}{w_a},\quad
t_y=\frac{y-y_a}{h_a}
$$

$$
t_w=\log\frac{w}{w_a},\quad
t_h=\log\frac{h}{h_a}
$$

Decoding reverses those equations.

## 4. Non-maximum suppression

NMS removes duplicate detections:

```text
sort by score
    ↓
keep highest
    ↓
suppress lower boxes with IoU > threshold
    ↓
repeat
```

NMS should normally be class-aware.

## 5. Precision and recall

For ranked predictions:

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

A prediction is a true positive only if it matches an unmatched ground-truth object above the chosen IoU threshold.

## 6. Average precision

This module uses the precision-envelope area under the precision/recall curve.

For monotonically increasing recall points:

$$
AP=\sum_i (R_i-R_{i-1})P_{interp}(R_i)
$$

Different benchmarks may use different IoU thresholds and interpolation rules. Always name the metric definition.

## 7. mAP

Mean AP averages AP across classes and, depending on the benchmark, across IoU thresholds.

Examples:

- AP@0.50
- AP@0.75
- COCO-style AP averaged over 0.50:0.95

These numbers are not interchangeable.

## 8. Python / C++17 parity

The deterministic artifact contains scored boxes.

Both runtimes compute:

- pairwise IoU evidence
- NMS keep indices

This is the native contract later detector modules depend on.

## 9. Failure modes

- mixing xywh and xyxy
- inclusive-vs-continuous area mismatch
- suppressing boxes across classes
- matching multiple predictions to the same ground truth
- reporting AP without its IoU definition
- evaluating on training data
- applying NMS differently across runtimes

## 10. Engineering rules

1. version box convention
2. validate positive width/height
3. sort deterministically by score
4. make NMS threshold explicit
5. make class behavior explicit
6. preserve metric implementation with experiment artifacts
7. compare exact keep indices across runtimes

## 11. AV relevance

For autonomous perception, metric quality must be connected to operational failure modes. A high aggregate mAP can still hide poor performance on small pedestrians, distant vehicles, or rare classes.

## 12. Book takeaway

> **Detection quality is inseparable from geometry and evaluation policy. Box convention, matching, NMS, and AP definition are part of the model contract.**

## Reproducible commands

```bash
python 01_Foundations_to_Modern_Vision/18_DetectionFundamentals/python/detection_walkthrough.py
pytest 01_Foundations_to_Modern_Vision/18_DetectionFundamentals/tests
cmake -S 01_Foundations_to_Modern_Vision/18_DetectionFundamentals -B build-m18 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m18
ctest --test-dir build-m18 --output-on-failure
python 01_Foundations_to_Modern_Vision/18_DetectionFundamentals/python/detection_native_parity.py \
  build-m18/cv9x_detection_fundamentals \
  01_Foundations_to_Modern_Vision/18_DetectionFundamentals/artifacts/detections.txt
```

## Stage 1.5 applied lab

This module's deterministic implementation remains the mechanics reference. The real-data companion is:

- [Applied Lab 02 — Oxford-IIIT Pet Detection and Segmentation](../../applied_labs/02_OxfordPetDetectionSegmentation/README.md)

The lab trains TinyYOLO, TinyDETR, and TinyUNet end to end using real public images and trimap masks while keeping dataset download optional.

## Next

Module 19 builds a YOLO-style one-stage detector decode path.
