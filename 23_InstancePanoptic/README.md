# Module 23 — Instance and Panoptic Segmentation

Semantic segmentation answers **what class is this pixel?** Instance segmentation also answers **which object instance?**

Panoptic segmentation combines:

- **things**: countable instances such as cars and people
- **stuff**: amorphous regions such as road, sky, or vegetation

## Learning objectives

You should be able to:

- distinguish semantic, instance, and panoptic segmentation
- understand mask scores and per-instance identities
- resolve overlapping instance masks deterministically
- compose instance masks with semantic stuff regions
- explain panoptic segment IDs
- compute a compact panoptic-quality example
- reproduce panoptic composition in C++17
- reason about mask memory, overlap policy, and post-processing

## 1. Semantic vs instance vs panoptic

Semantic:

```text
pixel → class
```

Instance:

```text
pixel → class + instance identity
```

Panoptic:

```text
every pixel belongs to exactly one semantic segment,
and thing pixels also carry instance identity
```

## 2. Instance masks

An instance detector may emit:

- class ID
- confidence score
- mask probability/logit map

A threshold converts each mask to a binary region.

When masks overlap, the runtime must define which instance wins.

This module uses:

1. higher confidence first
2. later masks cannot overwrite already claimed pixels

That policy is deterministic and easy to test.

## 3. Panoptic segment IDs

The executable reference encodes a panoptic pixel as:

$$
segment\_id = class\_id \times 1000 + instance\_id
$$

For stuff classes, instance ID is zero.

Real systems should version the encoding convention explicitly.

## 4. Composition

The module begins with a semantic stuff map, then overlays valid thing instances in descending score order.

```text
semantic labels
      ↓
encode stuff segments
      ↓
sort instances by score
      ↓
threshold masks
      ↓
claim unassigned thing pixels
      ↓
panoptic map
```

## 5. Panoptic Quality

A simplified Panoptic Quality decomposition is:

$$
PQ = SQ \times RQ
$$

where:

- SQ measures segmentation quality of matched segments
- RQ measures recognition quality

A common full formulation is:

$$
PQ =
\frac{\sum_{(p,g)\in TP}IoU(p,g)}
{|TP|+\frac12|FP|+\frac12|FN|}
$$

This module includes a small deterministic example.

## 6. Native contract

The artifact contains:

- semantic labels
- instance class IDs
- instance IDs
- confidence scores
- binary mask logits
- mask threshold

Python and C++17 must emit the same panoptic ID map.

## 7. Deployment implications

Instance segmentation can be expensive because mask tensors are large.

Important runtime considerations include:

- number of candidate instances
- mask resolution
- overlap resolution
- thresholding
- upsampling
- class/instance metadata
- memory lifetime

Keeping mask post-processing on-device can materially reduce bandwidth.

## 8. Failure modes

- treating overlapping masks nondeterministically
- losing class-to-instance association
- changing mask threshold across runtimes
- using different panoptic-ID encodings
- overwriting high-confidence instances with lower-confidence masks
- including void/ignore pixels inconsistently in PQ

## 9. Engineering rules

1. version mask threshold
2. define overlap priority
3. define panoptic-ID encoding
4. preserve instance IDs
5. compare the complete panoptic map
6. report PQ definition and matching threshold
7. monitor mask-count and mask-resolution costs

## 10. AV relevance

Panoptic outputs can provide a unified scene representation: drivable stuff regions plus explicit traffic participants. That unified contract is useful for downstream fusion and planning, but the mask bandwidth can be substantial.

## 11. Book takeaway

> **Panoptic segmentation unifies dense semantics with object identity. The hard engineering part is not just predicting masks; it is composing them deterministically into one scene representation.**

## Reproducible commands

```bash
python 23_InstancePanoptic/python/panoptic_walkthrough.py
pytest 23_InstancePanoptic/tests
cmake -S 23_InstancePanoptic -B build-m23 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m23
ctest --test-dir build-m23 --output-on-failure
python 23_InstancePanoptic/python/panoptic_native_parity.py \
  build-m23/cv9x_panoptic_compose \
  23_InstancePanoptic/artifacts/panoptic.txt
```

## Next

Module 24 introduces promptable/foundation segmentation.
