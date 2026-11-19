# Stage 1.5 — Applied Labs

Stage 1 teaches mechanics with tiny deterministic examples, PyTorch references, native C++ counterparts, and numerical parity.

Stage 1.5 adds a second layer:

```text
Stage 1 mechanics
  tiny deterministic examples
  math visible
  CPU CI
  Python ↔ C++ parity
          ↓
Stage 1.5 applied labs
  public research datasets
  real training
  real metrics
  optional GPU
  checkpoints / plots / reports
```

The applied labs are intentionally **optional**. A learner can complete the mechanics course on CPU without downloading large datasets.

## Principles

- Keep Stage-1 mechanics unchanged and checkable.
- Use public research datasets with recorded provenance.
- Never download datasets during CI.
- Provide `--device auto|cpu|cuda` where appropriate.
- Provide sample limits so a CPU learner can run a short experiment.
- Save machine-readable metrics in addition to plots.
- Record model, dataset, seed, transforms, device, and precision.
- Treat benchmark numbers as hardware-specific evidence, not universal truth.
- Link every applied lab back to the mechanics modules it extends.

## Lab plan

| Lab | Mechanics modules | Dataset | Goal |
|---|---|---|---|
| 01 | 09, 12, 14, 17 | CIFAR-10 | Train and compare CNN, ResNet, and ViT backbones |
| 02 | 18–23 | Oxford-IIIT Pet | Train TinyYOLO, TinyDETR and TinyUNet on real boxes/masks |
| 03 | 26–29 | CIFAR-10 | CLIP-style prompts, MIM, DINO and frozen linear probing |
| 04 | 30–31 | Lab-01 checkpoints | LoRA/PEFT, pruning, PTQ/QAT, accuracy-vs-latency comparison |

See [DATASETS.md](DATASETS.md) before downloading data.

## CI policy

Applied-lab code may be imported and smoke-tested with synthetic tensors, but CI must not require:

- network access
- dataset downloads
- pretrained-weight downloads
- CUDA
- long training runs

The real experiment is run explicitly by the learner.
