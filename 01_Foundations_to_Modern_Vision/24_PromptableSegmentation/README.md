# Module 24 — Promptable / Foundation Segmentation

Promptable segmentation separates a reusable visual representation from lightweight user/task prompts.

A SAM-style system can be viewed as:

```text
image
  ↓
image encoder
  ↓
reusable image embedding
  +
point / box / mask prompt
  ↓
prompt encoder
  ↓
mask decoder
  ↓
multiple candidate masks + quality scores
```

## Learning objectives

You should be able to:

- explain image-encoder reuse
- distinguish image prompts from learned task heads
- understand positive and negative point prompts
- understand candidate-mask generation
- select a mask using predicted quality
- threshold mask logits consistently
- explain promptable-segmentation caching
- reproduce candidate-mask selection in C++17
- identify deployment constraints of large foundation segmenters

## 1. Prompt semantics

This module uses point prompts:

- positive point: selected pixel should belong to the object
- negative point: selected pixel should not belong to the object

Prompt coordinates and image-resize transforms are part of the model contract.

## 2. Candidate masks

Promptable segmenters may return multiple candidate masks.

The runtime needs an explicit policy:

```text
candidate masks
    +
predicted quality scores
    ↓
select best valid candidate
    ↓
threshold logits
    ↓
binary mask
```

The quality score is not the same as per-pixel probability.

## 3. Prompt-consistency validation

For teaching, this module adds a simple validity check:

- positive prompts must fall inside the thresholded mask
- negative prompts must fall outside it

Among valid candidates, choose the highest quality score.

This is not a replacement for a learned quality head. It is a transparent runtime check.

## 4. Reusable image embeddings

A key systems advantage is that the expensive image encoder may run once:

```text
image → image embedding  [expensive, cacheable]
                 ↓
              prompt 1 → mask
              prompt 2 → mask
              prompt 3 → mask
```

Interactive systems can therefore amortize image-encoder cost.

## 5. Foundation-model deployment constraints

Large promptable segmenters introduce:

- large encoder latency
- embedding memory
- resize/pad coordinate transforms
- prompt batching
- candidate-mask tensors
- quality-score interpretation
- decoder latency
- possible mixed-precision sensitivity

A production design may use a smaller/mobile encoder while preserving the same prompt/mask contract.

## 6. Native contract

The deterministic artifact contains:

- image height/width
- positive prompts
- negative prompts
- candidate mask logits
- candidate quality scores
- threshold

Python and C++17 must select the same candidate and emit the same binary mask.

## 7. Failure modes

- prompt coordinates transformed differently from the image
- threshold mismatch across runtimes
- interpreting quality score as pixel probability
- selecting a high-score mask that contradicts a prompt
- forgetting padding/crop offsets
- recomputing the image encoder unnecessarily for every prompt

## 8. Engineering rules

1. version coordinate transforms
2. version mask threshold
3. distinguish quality score from mask logits
4. cache image embeddings when the use case allows it
5. preserve prompt polarity
6. compare complete output masks across runtimes

## 9. AV relevance

Promptable segmentation is useful for annotation, interactive debugging, bootstrapping datasets, and open-ended scene analysis. In safety-critical runtime paths, latency and prompt provenance must be tightly controlled.

## 10. Book takeaway

> **Promptable segmentation decouples a reusable visual representation from the task specification. The runtime contract is therefore not just an image; it is image geometry + prompt geometry + mask-selection policy.**

## Reproducible commands

```bash
python 01_Foundations_to_Modern_Vision/24_PromptableSegmentation/python/promptable_walkthrough.py
pytest 01_Foundations_to_Modern_Vision/24_PromptableSegmentation/tests
cmake -S 01_Foundations_to_Modern_Vision/24_PromptableSegmentation -B build-m24 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m24
ctest --test-dir build-m24 --output-on-failure
python 01_Foundations_to_Modern_Vision/24_PromptableSegmentation/python/promptable_native_parity.py \
  build-m24/cv9x_promptable_mask \
  01_Foundations_to_Modern_Vision/24_PromptableSegmentation/artifacts/promptable.txt
```

## Next

Module 25 introduces keypoints and pose estimation.
