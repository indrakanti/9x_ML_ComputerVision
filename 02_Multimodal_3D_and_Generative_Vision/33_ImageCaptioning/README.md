# Module 33 — Image Captioning and Multimodal Encoder–Decoders

**Learning objective:** connect image patch embeddings to a text decoder through scaled dot-product cross-attention, then run an optional pretrained captioning model on a real image.

## Architecture

```text
Image → vision encoder → patch embeddings (keys, values)
                                  ↓
Text tokens → decoder query → cross-attention → token prediction → caption
```

Cross-attention is `softmax(QKᵀ / sqrt(d)) V`. The tiny, deterministic Python and C++17 fixture implements a single-query, single-head attention step with two image patches. It is **not** a trained captioning model. The optional BLIP inference path runs an actual pretrained image-to-text model; no weights are checked into Git.

## Reproducible CPU fixture

```bash
python 02_Multimodal_3D_and_Generative_Vision/33_ImageCaptioning/python/captioning.py --synthetic
pytest 02_Multimodal_3D_and_Generative_Vision/33_ImageCaptioning/python
cmake -S . -B build-native -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
python 02_Multimodal_3D_and_Generative_Vision/33_ImageCaptioning/python/captioning_native_parity.py build-native/02_Multimodal_3D_and_Generative_Vision/33_ImageCaptioning/cpp/cv9x_module33_captioning
```

## Real-image applied example (optional, downloads pretrained weights)

```bash
python -m pip install transformers pillow torch
python 02_Multimodal_3D_and_Generative_Vision/33_ImageCaptioning/python/captioning.py --image path/to/open-license-photo.jpg
```

Default checkpoint: `Salesforce/blip-image-captioning-base`. Use an image whose license permits your use (e.g. an appropriately licensed Wikimedia Commons photo). Inspect generated captions for hallucinations and bias. The model path is optional, network-dependent, and excluded from CPU CI.

**Scope:** deterministic attention and pretrained inference; training an encoder–decoder from scratch and caption metrics (BLEU/CIDEr) are follow-up labs.
