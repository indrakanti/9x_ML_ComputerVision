# Module 35 — Vision-Language Model Architecture Patterns

## Goal

Understand the bridge between a vision encoder and a language decoder: **patch embeddings → learned projection → language embedding sequence → autoregressive generation**.

```text
image → vision encoder → patch embeddings ──┐
                                            ├─ projection → replace <image> placeholder
prompt → tokenizer → text embeddings ──────┘                    ↓
                                                   language decoder → tokens
```

### Deterministic lab

The Python and C++17 implementations project two 2D visual patches into a 3D language embedding space and replace an image placeholder in the text sequence. The test checks exact token order, dimensions, invalid inputs and native numerical parity. **This is an architectural fixture, not a trained VLM.**

```bash
python 02_Multimodal_3D_and_Generative_Vision/35_VisionLanguageModels/python/vlm.py --synthetic
pytest 02_Multimodal_3D_and_Generative_Vision/35_VisionLanguageModels/python
cmake -S . -B build-native -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
```

### Optional applied inference

```bash
python -m pip install torch pillow transformers
python 02_Multimodal_3D_and_Generative_Vision/35_VisionLanguageModels/python/vlm.py \
  --image path/to/open-license-image.jpg --prompt "Describe the objects in this scene."
```

The optional pretrained example uses `HuggingFaceTB/SmolVLM-256M-Instruct` and may require an updated Transformers version, model download, and adequate RAM. It is not part of offline CI. Use licensed images and verify generated claims: VLM outputs can hallucinate. The exact projector, tokenization, and image preprocessing depend on the checkpoint.

**Next extensions:** learned connector training, Q-Former/resampler, image-token budgeting, instruction tuning, quantization, and grounding benchmarks.
