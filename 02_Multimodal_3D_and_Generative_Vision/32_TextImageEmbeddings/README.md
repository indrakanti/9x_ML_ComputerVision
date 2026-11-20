# Module 32 — Text/Image Embeddings and Zero-Shot Classification

## Learning objectives
- Understand aligned image/text embedding spaces and contrastive learning.
- Compute normalized cosine similarity and temperature-scaled logits.
- Run an optional real pretrained CLIP zero-shot example with PyTorch/Transformers.
- Verify Python/C++ numerical parity with a deterministic synthetic fixture.

## Quick start (CPU, no dataset download)
```bash
python3 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/zero_shot.py --synthetic
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
```

## Real pretrained example (optional, requires internet on first run)
```bash
pip install torch transformers pillow
python3 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/zero_shot.py --image /path/to/image.jpg --labels "a dog" "a cat" "a car"
```

The pretrained checkpoint is downloaded from Hugging Face when invoked. Review the model card and license before reuse. No model weights or datasets are committed to this repository.

## Engineering contract
Embeddings are float32 row vectors; L2 normalization uses epsilon 1e-12; score is dot(normalize(image), normalize(text))/temperature. Default temperature is 1.0 for inference; it is not the learned CLIP logit scale. Predictions are ranked by descending score. Native C++ currently implements the deterministic scoring kernel, not the full CLIP encoder. Native encoder integration is a later checkpoint.
