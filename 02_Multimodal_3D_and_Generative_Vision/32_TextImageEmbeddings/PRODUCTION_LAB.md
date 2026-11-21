# Module 32 — Production Lab: Train, Evaluate, Export, Deploy

This lab supplements [Module 32's mathematical/scoring primer](README.md) rather than replacing it. It ships a **real, trainable PyTorch image/text dual encoder**, separate image/text ONNX artifacts, and a **real C++17 ONNX Runtime inference binary**. The dataset is CIFAR-10; no dataset or weights are committed.

## Architecture and equations

```text
RGB 32x32 → Conv→ReLU→Pool→Conv→GlobalPool→Linear ─── L2 norm ──┐
                                                              dot ×10 → 10 class logits
ASCII prompt → token embedding→masked mean→Linear ─── L2 norm ─┘
                              ↓
                     cross entropy + AdamW
```

Let `u=f_image(x)/||f_image(x)||₂` and `v_j=f_text(p_j)/||f_text(p_j)||₂`.
For ten CIFAR prompts the class logits are `z_j = 10 u·v_j`.
Minimize `L = -log(exp(z_y)/sum_j exp(z_j))`. Text/image towers both receive gradients.

**Model honesty:** the character-level text tower learns on ten class phrases. It accepts new printable ASCII strings but **has no demonstrated open-vocabulary or CLIP-quality generalization**. The optional pretrained CLIP script in the original lesson serves that different goal.

## Real dataset training

Dataset: **CIFAR-10**, from the TorchVision download utility. Check the dataset's original terms and attribution before redistribution; neither images nor downloaded copies should be committed. Use the official 50,000-image training and 10,000-image test splits. Set the seed and record the actual subset size.

```bash
# Run at repository root; requires torch torchvision numpy onnx onnxruntime
python 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/production.py train \
  --dataset cifar10 --download --data-root data/cifar10 --epochs 10 \
  --batch-size 64 --seed 42 --output runs/module32-cifar10
```

For limited hardware, append `--max-train 2000 --max-test 500`. These subset results are not full benchmark results. `metrics.json` records training loss, test top-1/top-5, per-class accuracy, 10% random chance baseline, seed and dataset sizes. Do not publish any performance number without actually running the dataset.

## Offline training/export smoke

```bash
python 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/production.py train \
  --dataset synthetic --epochs 1 --batch-size 20 --output runs/module32-smoke

python 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/production.py export \
  --checkpoint runs/module32-smoke/model.pt --output runs/module32-smoke/export
```

Synthetic patterns are **only CI smoke inputs**; their accuracy is not a real-world measure. Exports: `image.onnx` (float32 NCHW 32×32, variable batch), `text.onnx` (int64 [N,32], variable batch), input fixtures, `manifest.json` with SHA-256 checksums and preprocessing/tokenizer contracts. The export step validates ONNX and PyTorch–ONNX Runtime numeric parity.

## Real-image inference and CPU performance

After exporting a CIFAR-trained checkpoint, run:

```bash
python 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/infer_benchmark.py \
  --export-dir runs/module32-cifar10/export --image path/to/rgb.jpg \
  --warmup 5 --repeats 100
```

This produces ranked class probabilities, p50/p95/mean **warm image-tower CPU latency**, and peak process RSS. Preprocessing and text-embedding creation are excluded from the timed region and must be accounted for in end-to-end application benchmarks. These measurements are platform-specific; do not publish sample numbers as universal performance.

## Real C++17 inference

Use the native ONNX Runtime C++ SDK (tested CI target version 1.20.1 on Linux x86-64). This is a real execution backend, not an imitation of a neural network in C++.

```bash
cmake -S . -B build-native -G Ninja \
  -DCV9X_ONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-linux-x64-1.20.1
cmake --build build-native --parallel

python 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/production_native_parity.py \
  build-native/02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/cpp/cv9x_module32_onnx \
  runs/module32-smoke/export
```

The C++ binary loads both ONNX models as needed, accepts a precisely formatted float32 image tensor or int64 ASCII token tensor, and emits 32-dimensional embeddings. The parity script compares **both models** against ONNX Runtime Python. Your application can compute cosine logits from these embeddings, using the original Module 32 scoring kernel. For arbitrary production images, resize to RGB 32×32, scale bytes by 1/255, and convert HWC→CHW; identical preprocessing is essential.

### Deployment targets and evidence

| Target | Status |
|---|---|
| Linux x86-64 CPU with ONNX Runtime 1.20.1 | Automated training/export/native parity configured in GitHub Actions; consult CI result |
| GPU / CUDA / TensorRT | Not validated; requires operator compatibility, engine generation, and hardware timing |
| Jetson Orin / embedded ARM64 | Not validated; requires ARM64 runtime/toolchain and device measurements |
| TI / Qualcomm | Not validated; requires appropriate runtime/backend and hardware evaluation |

Native inference does not automatically mean automotive- or safety-qualified inference. Device-specific latency (warm and cold), resident memory, accuracy, calibration, and reproducibility still need hardware evidence.

## Reuse checklist

1. Keep label order, prompt strings, tokenizer, normalization and versioned artifact manifest together.
2. Measure top-1 and top-5 on the held-out test set and compare with the 10% chance baseline.
3. Verify exported models against PyTorch and native runtime before integration.
4. Store all metrics and build/runtime versions with each exported model.
5. Do not claim unseen-category generalization or hardware support without evaluation.
