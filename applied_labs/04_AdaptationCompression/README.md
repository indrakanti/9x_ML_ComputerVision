# Applied Lab 04 — PEFT, Distillation, Pruning and Quantization

This lab adds the real-model layer for:

- **Module 30** — linear probing, adapters, LoRA / PEFT
- **Module 31** — distillation, pruning, quantization

Unlike the mechanics module, this lab operates on checkpoints trained on real CIFAR-10 images.

## Inputs

The primary input is a checkpoint produced by Applied Lab 01:

```text
runs/applied_lab_01/
├── tinycnn/best.pt
├── resnet18/best.pt
└── tinyvit/best.pt
```

The scripts do not download pretrained weights.

## Experiment A — LoRA domain adaptation

A TinyViT checkpoint trained on normal CIFAR-10 is adapted to a grayscale version of the same task.

The base model is frozen.

Every compatible linear layer receives a low-rank update:

$$
W' = W + \frac{\alpha}{r}BA
$$

Only LoRA A/B parameters are trainable.

This creates a real adaptation experiment:

```text
color CIFAR-trained TinyViT
          ↓
freeze base weights
          ↓
inject LoRA
          ↓
train on grayscale CIFAR
          ↓
evaluate grayscale test set
```

Run:

```bash
python applied_labs/04_AdaptationCompression/lora_adapt.py \
  --checkpoint runs/applied_lab_01/tinyvit/best.pt \
  --device auto \
  --epochs 10 \
  --rank 4 \
  --alpha 8 \
  --download
```

The report includes:

- total parameters
- trainable LoRA parameters
- trainable fraction
- baseline grayscale accuracy
- adapted grayscale accuracy

## Experiment B — Teacher/student distillation

Use a trained ResNet-18 teacher and distill into TinyCNN:

```bash
python applied_labs/04_AdaptationCompression/distill.py \
  --teacher-checkpoint runs/applied_lab_01/resnet18/best.pt \
  --device auto \
  --epochs 20 \
  --temperature 3.0 \
  --alpha 0.7 \
  --download
```

Loss:

$$
L =
\alpha L_{KD}
+
(1-\alpha)L_{CE}
$$

with:

$$
L_{KD}
=
T^2 KL(p_t^T \| p_s^T)
$$

The student remains a normal TinyCNN at deployment.

## Experiment C — Pruning and INT8 evaluation

Evaluate a trained checkpoint before and after compression:

```bash
python applied_labs/04_AdaptationCompression/compress_evaluate.py \
  --checkpoint runs/applied_lab_01/tinyvit/best.pt \
  --device cpu \
  --prune-amount 0.30 \
  --iterations 100 \
  --download
```

The report compares:

- baseline test accuracy
- pruned test accuracy
- achieved weight sparsity
- dynamically quantized INT8 test accuracy
- serialized state size
- CPU latency

Dynamic INT8 quantization applies to supported linear operators. This is a runtime experiment, not a claim that every operator is quantized.

## Important interpretation

Compression results are target-dependent.

A model can become:

- smaller but not faster
- sparser but slower
- faster but less accurate
- numerically close but operationally worse

Always report:

```text
accuracy
latency
model size
sparsity
device
precision
runtime
```

together.

## CI policy

CI does not train or download CIFAR-10.

Synthetic tests verify:

- LoRA injection preserves output at initialization
- only LoRA parameters are trainable after freezing
- distillation loss is finite
- pruning produces zeros
- model-size helper works

## Exercises

1. Compare LoRA rank 1, 2, 4, 8.
2. Adapt to blurred or color-jittered CIFAR instead of grayscale.
3. Distill TinyViT into TinyCNN.
4. Fine-tune after pruning and measure recovery.
5. Compare dynamic INT8 on TinyCNN vs TinyViT.
6. Add per-channel static quantization with calibration.
7. Export the compressed model and compare ONNX/runtime behavior.
