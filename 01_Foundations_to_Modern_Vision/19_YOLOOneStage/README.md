# Module 19 — YOLO-Style One-Stage Detection

One-stage detectors predict localization and classification directly from dense feature-grid outputs.

This module focuses on the deployment-critical path:

```text
raw head tensor
    ↓
sigmoid / exp decode
    ↓
boxes + objectness + class scores
    ↓
confidence threshold
    ↓
class-aware NMS
    ↓
final detections
```

## Learning objectives

You should be able to:

- explain one-stage detection
- decode grid-relative center coordinates
- decode anchor-relative width and height
- combine objectness and class confidence
- convert center/size boxes to xyxy
- apply class-aware NMS
- reproduce the full post-processing path in C++17
- understand where GPU decode/NMS fit in a streaming detector
- distinguish model latency from end-to-end detector latency

## 1. Grid decoding

For grid cell $(g_x,g_y)$:

$$
c_x=\frac{\sigma(t_x)+g_x}{W_g}
$$

$$
c_y=\frac{\sigma(t_y)+g_y}{H_g}
$$

For anchor dimensions $(a_w,a_h)$:

$$
w=\frac{a_w e^{t_w}}{W_i}
$$

$$
h=\frac{a_h e^{t_h}}{H_i}
$$

The decoded normalized box becomes:

$$
[x_1,y_1,x_2,y_2]
=
[c_x-w/2,c_y-h/2,c_x+w/2,c_y+h/2]
$$

## 2. Confidence

This reference uses independent sigmoid class scores:

$$
score_k=\sigma(t_{obj})\sigma(t_k)
$$

The predicted class is the class with maximum score.

## 3. One-stage inference contract

The executable artifact fixes:

- image size
- grid size
- one anchor
- class count
- raw prediction values
- score threshold
- NMS threshold

Python and C++ must emit the same final detections.

## 4. Native streaming path

A production detector often looks like:

```text
camera frame
  ↓
GPU preprocessing
  ↓
GPU model inference
  ↓
GPU decode
  ↓
GPU/CPU NMS
  ↓
small detection list
  ↓
tracking / fusion
```

Keeping intermediate tensors resident on the GPU avoids unnecessary transfers.

## 5. Optional CUDA checkpoint

The module includes an optional CUDA decode kernel.

CPU CI does not require CUDA. When `CV9X_ENABLE_CUDA=ON` and a CUDA compiler is available, the native module can build the CUDA decode example.

The CUDA lesson is about:

- one thread per candidate
- device-resident raw predictions
- sigmoid/exp decode
- synchronization boundaries
- copying only compact outputs when appropriate

NMS optimization is intentionally left as an exercise because efficient GPU NMS has multiple implementation strategies.

## 6. Timing evidence

Detector timing should separate:

- preprocessing
- host-to-device transfer
- model inference
- decode
- NMS
- device-to-host transfer
- end-to-end latency
- warm-up behavior

CUDA timing must use GPU events or explicit synchronization around the measured region.

## 7. Failure modes

- wrong grid indexing
- wrong anchor units
- using softmax when the trained head expects independent sigmoids
- multiplying confidence differently from training/export convention
- clipping before versus after NMS inconsistently
- class-agnostic NMS by accident
- timing asynchronous GPU launches with CPU wall-clock only
- copying full dense outputs to CPU unnecessarily

## 8. Engineering rules

1. version decode equations with the model
2. preserve anchor units
3. preserve class ordering
4. define confidence composition
5. share one NMS contract across runtimes
6. compare final boxes, scores, and classes numerically
7. keep dense post-processing on-device when it materially reduces transfer cost

## 9. AV relevance

In surround perception, detector post-processing runs repeatedly across cameras. A seemingly small CPU decode or transfer cost can multiply by camera count and frame rate.

## 10. Book takeaway

> **A one-stage detector is not complete when the neural network returns a tensor. Decode, confidence composition, NMS, memory movement, and synchronization are part of the deployed detector.**

## Reproducible commands

CPU reference and parity:

```bash
python 19_YOLOOneStage/python/yolo_walkthrough.py
pytest 19_YOLOOneStage/tests
cmake -S 19_YOLOOneStage -B build-m19 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m19
ctest --test-dir build-m19 --output-on-failure
python 19_YOLOOneStage/python/yolo_native_parity.py \
  build-m19/cv9x_yolo_decode \
  19_YOLOOneStage/artifacts/yolo_head.txt
```

From the repository root, the optional CUDA checkpoint can be enabled on a CUDA-capable machine:

```bash
cmake -S . -B build-cuda -G Ninja -DCV9X_ENABLE_CUDA=ON
cmake --build build-cuda --target cv9x_yolo_decode_cuda
```

## Stage 1.5 applied lab

This module's deterministic implementation remains the mechanics reference. The real-data companion is:

- [Applied Lab 02 — Oxford-IIIT Pet Detection and Segmentation](../applied_labs/02_OxfordPetDetectionSegmentation/README.md)

The lab trains TinyYOLO, TinyDETR, and TinyUNet end to end using real public images and trimap masks while keeping dataset download optional.

## Next

Module 20 introduces region proposals and two-stage detection.
