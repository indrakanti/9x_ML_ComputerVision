# Module 25 — Keypoints and Pose Estimation

Pose estimation predicts structured landmark locations rather than boxes or dense class labels.

A common heatmap-based pipeline is:

```text
image
  ↓
backbone
  ↓
K heatmaps
  ↓
argmax / refinement
  ↓
(x, y, confidence) per keypoint
  ↓
skeleton / object geometry
```

## Learning objectives

You should be able to:

- explain heatmap-based keypoint prediction
- decode a keypoint from a heatmap
- understand confidence from peak response
- distinguish image coordinates from heatmap coordinates
- understand skeleton topology
- compute PCK-style keypoint accuracy
- reproduce heatmap decoding and PCK in C++17
- reason about missing/occluded keypoints

## 1. Heatmap representation

For $K$ keypoints, the model may emit:

$$
[K,H_h,W_h]
$$

heatmaps.

For keypoint $k$:

$$
(x_k,y_k)=\arg\max_{x,y} H_k(x,y)
$$

The peak value is commonly used as a confidence proxy.

## 2. Coordinate scaling

If heatmap resolution differs from image resolution, coordinates must be mapped carefully.

For scale factors:

$$
s_x=\frac{W_i}{W_h},\quad
s_y=\frac{H_i}{H_h}
$$

a simple mapping is:

$$
x_i=(x_h+0.5)s_x-0.5
$$

$$
y_i=(y_h+0.5)s_y-0.5
$$

The exact resize/alignment convention must match training.

## 3. Skeleton topology

Keypoints become a structured pose when connected by defined edges.

Examples:

- shoulder → elbow
- elbow → wrist
- hip → knee

Topology is metadata, not something argmax recovers automatically.

## 4. PCK

Percentage of Correct Keypoints measures whether predicted landmarks fall within a normalized distance threshold.

For normalization length $L$ and threshold $\alpha$:

$$
\|p_i-g_i\|_2 \le \alpha L
$$

counts as correct.

This module reports the fraction of valid keypoints satisfying that criterion.

## 5. Executable reference

The artifact contains:

- heatmap size
- image size
- three keypoint heatmaps
- ground-truth keypoints
- valid/visible flags
- PCK normalization length and threshold

Python and C++17 compute:

- decoded image-space keypoints
- confidence values
- PCK

## 6. Deployment implications

Keypoint systems require more than network inference:

- heatmap decode
- coordinate transforms
- confidence thresholding
- skeleton association
- temporal smoothing
- missing-keypoint handling

For multi-person pose, instance association becomes an additional problem.

## 7. Failure modes

- wrong heatmap/image scale
- half-pixel alignment mismatch
- swapping x/y
- treating low-confidence keypoints as valid
- evaluating invisible/ignored landmarks
- changing PCK normalization definition
- mixing top-down and bottom-up pose assumptions

## 8. Engineering rules

1. version keypoint order
2. version skeleton edges
3. version coordinate transform
4. preserve visibility semantics
5. report PCK threshold and normalization
6. compare decoded coordinates across runtimes

## 9. AV relevance

Keypoints are useful beyond human pose: articulated objects, trailer/hitch landmarks, wheel centers, corners, traffic-light elements, and calibration features can all be modeled as structured landmarks.

## 10. Book takeaway

> **Pose estimation turns feature maps into geometry. The learned heatmap is only half the contract; coordinate mapping, confidence, topology, and evaluation complete it.**

## Reproducible commands

```bash
python 01_Foundations_to_Modern_Vision/25_PoseEstimation/python/pose_walkthrough.py
pytest 01_Foundations_to_Modern_Vision/25_PoseEstimation/tests
cmake -S 01_Foundations_to_Modern_Vision/25_PoseEstimation -B build-m25 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m25
ctest --test-dir build-m25 --output-on-failure
python 01_Foundations_to_Modern_Vision/25_PoseEstimation/python/pose_native_parity.py \
  build-m25/cv9x_pose_decode \
  01_Foundations_to_Modern_Vision/25_PoseEstimation/artifacts/pose.txt
```

## Next

Module 26 introduces contrastive image-text representation learning.
