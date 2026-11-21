# Module Acceptance Standard — Understand → Train → Evaluate → Deploy

This checklist is the **definition of done** for applied engineering modules. A green lint/compile job alone never proves a usable deployment.

## Required deliverables

1. **Theory:** equations, architecture/dataflow diagrams, algorithm assumptions, numeric precision, and worked example.
2. **Train:** real PyTorch trainable model (or a clear distinction when a module is about a non-trainable method), public dataset download and license notes, deterministic seed, train/validation/test boundaries, optimizer/loss, checkpoint format and configuration.
3. **Evaluate:** held-out metrics, baseline, reproducible split, per-class/error analysis, and recorded run parameters. Synthetic smoke-test values **must not** be reported as real accuracy.
4. **Export:** standardized runtime artifact (e.g. ONNX) with input/output shape, dtype, preprocessing/tokenizer, operator/runtime version, SHA-256, and Python framework/runtime numerical parity.
5. **Native application:** C++17 real inference, not a reimplementation of a toy scoring-only kernel; native/runtime parity, invalid-input handling, linked dependency versions.
6. **Performance:** warm-up and measured p50/p95/mean inference latency, throughput where applicable, peak memory, hardware/runtime/build details, and which stages the timed region includes.
7. **CI and evidence:** offline CPU smoke train → export → inference → C++ parity, lint/types/tests; separately runnable real-dataset experiment and separately captured device measurements. Do not claim CUDA/Jetson/TI/Qualcomm deployment without evidence on that target.
8. **Reuse:** README exact commands, clean API, versioned outputs, artifact manifest, portability limitations, dataset/model redistribution terms, and known failure cases.

## Evidence vocabulary

- **Implemented**: source exists.
- **CI validated**: automated run successfully executed the functionality with a linked passing run.
- **Dataset validated**: completed real-dataset run with preserved results.
- **Hardware validated**: completed on specified device, toolchain and runtime with measurement evidence.
- **Not validated**: code/docs may exist, but no passing evidence; never relabel this as deployment-ready.

## Adoption plan

- Module **32** serves as the first train/evaluate/ONNX/native inference reference, beginning with CIFAR-10 and Linux x86-64.
- Retrofit Modules 33–36 to the same standard before advancing the module count.
- For foundation algorithms that are not independently trainable, name the consuming trainable system and verify its integration.
- Heavy GPU/hardware experiments can be opt-in but must yield a reproducible run report. Lightweight deterministic smoke tests stay mandatory on every PR.
