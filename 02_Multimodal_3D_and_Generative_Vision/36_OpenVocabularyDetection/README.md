# Module 36 — Open-Vocabulary Object Detection

## Objective

Connect a region-based vision encoder with text embeddings so a detector can score **user-supplied category names**, rather than a fixed class list. Apply box intersection-over-union (IoU) and class-aware non-maximum suppression (NMS).

```text
Image → region proposals → region embeddings ──┐
                                               ├─ cosine similarity → class score → NMS
Text prompts → text encoder → text embeddings ─┘
```

### Deterministic Python/C++17 lab

Three synthetic boxes, two text classes, region/text cosine scores and class-aware NMS. Two overlapping car boxes collapse to one detection, while a distinct person detection remains. The fixture checks numeric parity between Python and C++17. **This is not a trained detector**; its boxes and embeddings are provided rather than learned.

```bash
python 02_Multimodal_3D_and_Generative_Vision/36_OpenVocabularyDetection/python/open_vocab.py --synthetic
pytest 02_Multimodal_3D_and_Generative_Vision/36_OpenVocabularyDetection/python
cmake -S . -B build-native -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
```

### Optional real-image open-vocabulary detection

```bash
python -m pip install torch transformers pillow
python 02_Multimodal_3D_and_Generative_Vision/36_OpenVocabularyDetection/python/open_vocab.py \
  --image path/to/open-license-image.jpg --labels "a car" "a person" "a bicycle"
```

The optional pretrained pipeline uses `google/owlvit-base-patch32` and downloads model weights. No weights or dataset images are committed. Use appropriately licensed input images. Reported scores are model-dependent similarities, not calibrated probabilities. Test novel categories and domain shift before interpreting results.

**Next:** referring expressions, phrase grounding, and spatial localization in Module 37.
