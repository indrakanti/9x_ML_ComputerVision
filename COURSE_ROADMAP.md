# 9x ML Computer Vision — Course Roadmap

## Scope

This roadmap is organized by **capability progression**, from ML fundamentals to current foundation/embodied vision concepts.

The course has three synchronized tracks:

- **Learning/training:** Python + PyTorch
- **Production runtime:** C++17/20 + CMake + exported-model runtimes
- **Acceleration:** CUDA/GPU execution, memory movement, synchronization, profiling, and optimized inference

CPU remains the portable correctness baseline. C++ and GPU are introduced early and revisited at every major model family rather than postponed until the end.

## Part I — ML Foundations for Vision

| Module | Topic | Status |
|---|---|---|
| 00 | Linux, Python/PyTorch, C++/CMake, GPU toolchain, testing, CI | **Foundation PR** |
| 01 | NumPy/PyTorch tensors + image/tensor memory contracts + C++ tensor/runtime concepts | Planned |
| 02 | Probability/statistics essentials for vision ML | Planned |
| 03 | Optimization: losses, gradients, SGD, momentum, Adam | Planned |
| 04 | Autograd from scalar graph intuition to PyTorch | Planned |
| 05 | Dataset splits, leakage, metrics, reproducibility | Planned |

## Part II — From Classical Features to Learned Features

| Module | Topic | Status |
|---|---|---|
| 06 | Linear classifiers for images | Planned |
| 07 | MLPs, activations, initialization, normalization | Planned |
| 08 | Convolution from first principles: NumPy, PyTorch and native/GPU execution concepts | Planned |
| 09 | CNN training loop from scratch + C++ inference smoke path | Planned |
| 10 | Regularization, augmentation, schedulers, checkpointing | Planned |
| 11 | Transfer learning + first ONNX export / native inference checkpoint | Planned |

## Part III — Modern Image Backbones

| Module | Topic | Status |
|---|---|---|
| 12 | ResNet and residual learning | Planned |
| 13 | Efficient CNNs: MobileNet/EfficientNet-style ideas | Planned |
| 14 | Vision Transformer fundamentals | Planned |
| 15 | Hierarchical/windowed transformers | Planned |
| 16 | ConvNet/Transformer hybrid design patterns | Planned |
| 17 | Modern backbone evaluation: accuracy + C++/GPU latency, memory and synchronization | Planned |

## Part IV — Detection, Segmentation and Pose

| Module | Topic | Status |
|---|---|---|
| 18 | Detection fundamentals: boxes, anchors, IoU, mAP | Planned |
| 19 | YOLO-family one-stage detection + native GPU inference checkpoint | Planned |
| 20 | Two-stage detection and region proposals | Planned |
| 21 | DETR-family set prediction and transformer detection | Planned |
| 22 | Semantic segmentation | Planned |
| 23 | Instance/panoptic segmentation | Planned |
| 24 | Promptable/foundation segmentation concepts + deployment constraints | Planned |
| 25 | Keypoints and human/object pose estimation | Planned |

## Part V — Representation Learning and Vision Foundation Models

| Module | Topic | Status |
|---|---|---|
| 26 | Contrastive learning and CLIP-style image/text alignment | Planned |
| 27 | Masked-image modeling | Planned |
| 28 | Self-supervised distillation and DINO-family ideas | Planned |
| 29 | Universal/foundation visual backbones and dense features | Planned |
| 30 | Linear probing, adapters, LoRA and parameter-efficient tuning | Planned |
| 31 | Distillation, pruning, quantization + native/GPU efficiency checkpoint | Planned |

## Part VI — Vision-Language and Open-Vocabulary Vision

| Module | Topic | Status |
|---|---|---|
| 32 | Text/image embeddings and zero-shot classification | Planned |
| 33 | Image captioning and multimodal encoders/decoders | Planned |
| 34 | Visual question answering and multimodal reasoning | Planned |
| 35 | Vision-language models / multimodal LLM architecture patterns | Planned |
| 36 | Open-vocabulary detection | Planned |
| 37 | Visual grounding / referring expressions | Planned |
| 38 | Retrieval-augmented multimodal systems | Planned |
| 39 | Tool-using / agentic visual systems | Planned |

## Part VII — Video and Temporal Intelligence

| Module | Topic | Status |
|---|---|---|
| 40 | Video datasets, clips, temporal sampling | Planned |
| 41 | 3-D convolution and temporal models | Planned |
| 42 | Video transformers | Planned |
| 43 | Video representation/foundation models | Planned |
| 44 | Learned optical flow / correspondence | Planned |
| 45 | Deep multi-object tracking and re-identification | Planned |
| 46 | Temporal grounding + streaming C++/GPU video inference checkpoint | Planned |

## Part VIII — 3-D Vision and Neural Scene Representation

| Module | Topic | Status |
|---|---|---|
| 47 | Depth estimation: monocular learned depth | Planned |
| 48 | Point clouds and learned 3-D representations | Planned |
| 49 | Multi-view learned geometry | Planned |
| 50 | NeRF fundamentals | Planned |
| 51 | 3-D Gaussian Splatting concepts | Planned |
| 52 | Neural rendering and novel-view synthesis | Planned |
| 53 | 3-D foundation / spatial reasoning models | Planned |

## Part IX — Generative Vision

| Module | Topic | Status |
|---|---|---|
| 54 | Autoencoders and variational autoencoders | Planned |
| 55 | GAN fundamentals and legacy relevance | Planned |
| 56 | Diffusion-model fundamentals | Planned |
| 57 | Latent diffusion and text-to-image systems | Planned |
| 58 | Flow matching / rectified-flow concepts | Planned |
| 59 | Image editing, inpainting, control/conditioning | Planned |
| 60 | Video generation concepts and temporal consistency | Planned |

## Part X — World Models, Embodied Vision and VLA

| Module | Topic | Status |
|---|---|---|
| 61 | World-model fundamentals: latent dynamics and prediction | Planned |
| 62 | Predictive video/visual representation learning | Planned |
| 63 | Spatial reasoning and embodied perception | Planned |
| 64 | Vision-language-action (VLA) architecture concepts | Planned |
| 65 | Action tokenization / continuous control interfaces | Planned |
| 66 | Robot datasets, imitation learning and behavior cloning | Planned |
| 67 | Safety, uncertainty and evaluation for embodied vision | Planned |
| 68 | On-device / edge embodied models + C++/GPU runtime and safety-monitoring checkpoint | Planned |

## Part XI — Training Systems and MLOps

| Module | Topic | Status |
|---|---|---|
| 69 | Experiment configuration and reproducibility | Planned |
| 70 | Mixed precision and gradient scaling | Planned |
| 71 | Multi-GPU / distributed training concepts | Planned |
| 72 | Data pipelines, caching and throughput | Planned |
| 73 | Checkpointing, resume and artifact versioning | Planned |
| 74 | TensorBoard / experiment tracking | Planned |
| 75 | Dataset/model cards, provenance and evaluation reports | Planned |

## Part XII — Advanced Deployment and Production Inference

| Module | Topic | Status |
|---|---|---|
| 76 | Torch compile/export concepts | Planned |
| 77 | ONNX export and graph inspection | Planned |
| 78 | ONNX Runtime CPU/GPU inference | Planned |
| 79 | TensorRT / accelerator concepts | Planned |
| 80 | Quantized inference | Planned |
| 81 | Batch vs streaming latency | Planned |
| 82 | Python service vs native C++ inference | Planned |
| 83 | C++ ONNX Runtime integration | Planned |
| 84 | Edge deployment and profiling | Planned |
| 85 | Production monitoring and model drift | Planned |

## Cross-Cutting C++ / GPU Checkpoints

These are not optional side notes. Each checkpoint revisits the same model from a systems perspective.

| After | Native/GPU milestone |
|---|---|
| Module 01 | C++ build, image/tensor layout, dtype, ownership, CPU/GPU device model |
| Module 08 | Convolution execution on CPU vs GPU; copies, synchronization, kernel-launch concepts |
| Module 11 | Export a trained model and execute it outside the Python training process |
| Module 17 | Measure C++/GPU latency, warm-up, memory, precision and synchronization |
| Module 19 | Detector inference in a streaming native pipeline |
| Module 31 | Quantization / pruning / precision tradeoffs in deployed inference |
| Module 46 | Streaming video inference, queues, backpressure and temporal state |
| Module 68 | Edge/embodied runtime, monitoring, timeout/fault behavior and safe degradation |
| Modules 76–85 | Full ONNX Runtime / TensorRT / C++ production integration |

### Safety-oriented engineering thread

The course does not claim that using C++ or a GPU by itself makes a system safe.

Instead, production lessons will explicitly separate:

~~~text
ML model correctness
runtime correctness
timing evidence
resource bounds
health monitoring
fault containment
fallback behavior
system-level safety argument
~~~

This thread is especially relevant to automotive and robotics deployment.

## Applied Projects

- image classifier with reproducible training/evaluation
- transfer-learning classifier
- YOLO-family detector
- semantic/instance segmentation
- pose/keypoint pipeline
- self-supervised representation probe
- CLIP-style zero-shot/retrieval application
- open-vocabulary grounded detector
- video understanding/tracking pipeline
- monocular depth / 3-D project
- generative image project
- multimodal visual assistant
- ONNX + C++ deployment project
- embodied/VLA conceptual capstone

## Course boundary philosophy

The course should teach durable concepts, while representative models can evolve.

For example, a module on self-supervised foundation backbones should teach:

- objectives
- data scaling
- feature quality
- dense vs global representation
- probing/adaptation
- distillation

rather than depending on one model release forever.

That lets the course stay current as model families change.
