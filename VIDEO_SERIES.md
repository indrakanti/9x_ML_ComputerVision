# 9x ML Computer Vision — Video Series

Each numbered roadmap module maps to a long-form episode by default. Some advanced modules may later split into multiple episodes if hands-on scope requires it.

## Foundation

| Episode | Topic | Status |
|---|---|---|
| 00 | ML Computer Vision Setup — Python/PyTorch + C++ + GPU on Linux | **Recording-ready** |
| 01 | NumPy to PyTorch Tensors for Vision | **Implementation-ready** |
| 02 | Probability & Statistics You Actually Need for Vision ML | **Implementation-ready** |
| 03 | Optimization: Losses, SGD, Momentum & Adam | **Implementation-ready** |
| 04 | Autograd from First Principles | **Implementation-ready** |
| 05 | Datasets, Splits, Leakage, Metrics & Reproducibility | **Implementation-ready** |
| 06 | Linear Classifiers for Images — Math to C++ Deployment | **Implementation-ready** |
| 07 | MLPs, Activations, Initialization & Normalization | **Implementation-ready** |
| 08 | Convolution from First Principles — Python, C++ & GPU Execution | **Implementation-ready** |
| 09 | CNN Training Loop — Forward, Backward, Optimizer, Checkpoint & C++ | **Implementation-ready** |
| 10 | Regularization, Augmentation, Schedulers & Checkpointing | **Implementation-ready** |
| 11 | Transfer Learning + First ONNX Export / Native Parity | **Implementation-ready** |
| 12 | ResNet and Residual Learning | **Implementation-ready** |
| 13 | Efficient CNNs — Depthwise Separable Convolution | **Implementation-ready** |
| 14 | Vision Transformer Fundamentals | **Implementation-ready** |
| 15 | Hierarchical and Windowed Transformers | **Implementation-ready** |
| 16 | Hybrid CNN–Transformer Backbones | **Implementation-ready** |
| 17 | Backbone Evaluation and Engineering Trade-Space | **Implementation-ready** |
| 18 | Detection Fundamentals — Boxes, IoU, NMS & mAP | **Implementation-ready** |
| 19 | YOLO-Style One-Stage Detection + GPU Checkpoint | **Implementation-ready** |
| 20 | Two-Stage Detection and Region Proposals | **Implementation-ready** |
| 21 | DETR-Style Set Prediction | **Implementation-ready** |
| 22 | Semantic Segmentation | **Implementation-ready** |
| 23 | Instance and Panoptic Segmentation | **Implementation-ready** |
| 24 | Promptable / Foundation Segmentation | **Implementation-ready** |
| 25 | Keypoints and Pose Estimation | **Implementation-ready** |
| 26 | Contrastive Learning and CLIP | **Implementation-ready** |

## Major series arcs

~~~text
06–11   Learned features and CNN foundations
12–17   Modern image backbones
18–25   Detection, segmentation and pose
26–31   Self-supervised / foundation visual representations
32–39   Vision-language and open-vocabulary systems
40–46   Video and temporal intelligence
47–53   3-D vision and neural scene representation
54–60   Generative vision
61–68   World models, embodied vision and VLA
69–75   Training systems and MLOps
76–85   Export, optimization, C++ and production inference
~~~

The active target is intentionally contemporary: the course should reach modern multimodal and embodied vision, not end at a legacy CNN/detector boundary.

Every major arc also revisits the native runtime path: Python training -> exported model -> C++ execution -> GPU acceleration -> profiling/monitoring.

See [COURSE_ROADMAP.md](COURSE_ROADMAP.md) for the module-by-module plan.
