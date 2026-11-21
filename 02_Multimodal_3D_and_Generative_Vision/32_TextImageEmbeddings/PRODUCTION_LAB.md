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

## Reusable native image classification (Module 32 completion)

The new **`cv9x_module32_classifier` C++17 library** exposes a reusable
`cv9x::module32::Classifier(export_directory)` and
`predict(image_path) → ranked labels/probabilities` API in
[`cpp/classifier.hpp`](cpp/classifier.hpp). Its `cv9x_module32_classify`
executable accepts real PNG/JPEG images. It decodes and resizes with OpenCV,
converts BGR→RGB and HWC→NCHW, executes both real ONNX encoders, and computes
ranked ten-class predictions.

Native dependencies: ONNX Runtime 1.20.1 C++ SDK, OpenCV (core/imgproc/imgcodecs),
OpenSSL Crypto (SHA-256), and nlohmann-json 3.10+.
On Ubuntu, install `libopencv-dev libssl-dev nlohmann-json3-dev` plus a
matching native ONNX Runtime SDK, then configure with `-DCV9X_ONNXRUNTIME_ROOT`.

```bash
cmake -S . -B build-native -DCV9X_ONNXRUNTIME_ROOT=/absolute/path/to/ort-sdk
cmake --build build-native --parallel

build-native/02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/cpp/cv9x_module32_classify \
  --export-dir runs/module32-cifar10/export \
  --image /path/to/real-photo.png --warmup 3 --repeats 30
```

The CLI outputs JSON containing the top prediction, all ten ranked
probabilities, and p50/p95/mean latency and process RSS. Latency includes
**image decode, preprocessing, image-tower inference and scoring**, with
session initialization and text-tower inference cached before timing.
Measurements are host-specific and not embedded-hardware evidence.

### Versioned artifact contract

Bundle `image.onnx`, `text.onnx` and `manifest.json` in one directory.
The manifest fixes schema version 1, ONNX opset, input/output dimensions,
`logit_scale=10`, ordered labels, exact prompts, ASCII tokenizer and RGB
preprocessing, and SHA-256 checksums for both model files. Native construction
**rejects a missing, modified or incompatible model before loading sessions**.

```bash
python 02_Multimodal_3D_and_Generative_Vision/32_TextImageEmbeddings/python/artifact_contract.py \
  --export-dir runs/module32-cifar10/export
```

CI creates a lossless 32×32 RGB PNG, compares the complete native ranked
probabilities against Python ONNX Runtime, and confirms corrupted ONNX bytes
are rejected. **Cross-language preprocessing parity for resized non-32×32
images is not yet guaranteed pixel-exact** (OpenCV versus PIL interpolation);
application-level acceptance tests should include target camera resolutions.

### Real-data evaluation evidence

The separately runnable **Module 32 Real CIFAR-10 Evidence** workflow now
also runs automatically on Module 32 PRs. It trains on a seeded
2,000-image CIFAR-10 training subset and evaluates a distinct 500-image
official test subset, then uploads the actual checkpoint, ONNX models,
`metrics.json`, a held-out-image CPU inference benchmark and artifact
manifest. `metrics.json` includes top-1, top-5, macro-F1, confusion matrix,
class accuracy, and representative misclassifications (sample indices are
within the evaluation subset).

These subset results are evidence of *execution and a small experiment*;
they are **not** full CIFAR-10 leaderboard metrics. To evaluate the entire
dataset, rerun with `--max-train 0 --max-test 0` and more epochs. Do not
claim an accuracy number until its GitHub artifact has actually been produced.

### First measured real-data result (November 21, 2026)

**GitHub Actions run [#44935121242](https://github.com/indrakanti/9x_ML_ComputerVision/actions/runs/44935121242) passed** on commit `d6ed407` using the pinned CIFAR-10 Python archive with TorchVision's original MD5 checksum verification. This is a **limited CPU reference experiment**, not a competitive CIFAR-10 model or a claim of unseen-category generalization.

| Measurement | Actual result |
|---|---:|
| Train / held-out test examples | 2,000 / 500 |
| Epochs / random seed | 3 / 42 |
| Top-1 test accuracy | **26.4%** |
| Top-5 test accuracy | 72.4% |
| Macro-F1 | 0.1842 |
| Uniform random chance top-1 | 10% |
| Training loss (epochs 1 → 3) | 2.3011 → 2.1132 |

The completed workflow also validated PyTorch → ONNX numerical parity, produced a real held-out-image ONNX Runtime CPU benchmark, validated the model bundle contract, and uploaded checkpoint, models and metrics as a time-limited Actions artifact. The 26.4% test result leaves considerable room for architecture, training-data, and augmentation improvements. To make stronger accuracy claims, run the full official split, include a train-only validation protocol, repeat seeds and compare a stronger image-only baseline.

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
