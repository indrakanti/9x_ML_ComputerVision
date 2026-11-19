# Applied Lab 02 — Oxford-IIIT Pet Detection and Segmentation

This lab adds the real-data layer for:

- **Module 18** — boxes, IoU, NMS, AP concepts
- **Module 19** — YOLO-style one-stage detection
- **Module 20** — two-stage detection concepts
- **Module 21** — DETR-style set prediction
- **Module 22** — semantic segmentation
- **Module 23** — instance/panoptic segmentation concepts

The mechanics modules remain deterministic and native-parity focused. This lab trains small PyTorch models end to end on a real public dataset.

## Dataset

Oxford-IIIT Pet contains cat/dog images with:

- 37 breed labels
- head ROI annotations
- pixel-level trimap segmentation

The dataset is published under **CC BY-SA 4.0** by the Oxford Visual Geometry Group.

This lab uses the segmentation trimap and derives a full-pet bounding box from the foreground mask.

Trimap interpretation:

- 1 — pet foreground
- 2 — background
- 3 — boundary / ambiguous region

For this binary teaching lab, labels 1 and 3 are treated as pet foreground and label 2 as background.

## Why this dataset?

One dataset supports both tasks:

```text
image + trimap
      ↓
binary pet mask
      ├→ semantic segmentation target
      └→ foreground bounding box
             ↓
          detector target
```

That keeps detector and segmentation experiments on the same image/split contract.

## Models

### TinyYOLO

A small convolutional detector predicts a 4×4 grid.

Each cell emits:

```text
objectness + tx + ty + tw + th
```

The pet center selects one positive grid cell.

The lab trains:

- objectness BCE
- center/size regression loss

Inference decodes normalized xyxy boxes and reports IoU.

### TinyDETR

A small DETR-style model contains:

- CNN image encoder
- flattened spatial memory
- learned object queries
- Transformer decoder
- pet/no-object class head
- normalized box head

Because the dataset has one primary pet, matching selects the query with the lowest class + L1 + IoU cost for that target. Other queries learn the no-object class.

The full Hungarian algorithm remains in Module 21 mechanics; this real-data lab keeps the one-target assignment transparent.

### TinyUNet

A compact encoder-decoder predicts one pet foreground logit per pixel.

Training uses:

- BCEWithLogitsLoss
- soft Dice loss

Evaluation reports:

- pixel IoU
- Dice

## Quick CPU detector run

```bash
python applied_labs/02_OxfordPetDetectionSegmentation/train_detection.py \
  --model yolo \
  --device cpu \
  --epochs 2 \
  --max-train-samples 300 \
  --max-val-samples 80 \
  --max-test-samples 80 \
  --download
```

## TinyDETR

```bash
python applied_labs/02_OxfordPetDetectionSegmentation/train_detection.py \
  --model detr \
  --device auto \
  --epochs 15 \
  --download
```

## Segmentation

```bash
python applied_labs/02_OxfordPetDetectionSegmentation/train_segmentation.py \
  --device auto \
  --epochs 15 \
  --download
```

## Outputs

Detection:

```text
runs/applied_lab_02/detection/<model>/
├── best.pt
├── last.pt
└── metrics.json
```

Segmentation:

```text
runs/applied_lab_02/segmentation/
├── best.pt
├── last.pt
└── metrics.json
```

## Evaluation policy

Detection reports:

- mean box IoU
- recall@IoU 0.50

Segmentation reports:

- mean foreground IoU
- mean Dice

These are intentionally simple teaching metrics. They do not replace COCO mAP or full panoptic metrics.

## Important limitation

Oxford-IIIT Pet usually contains one primary annotated pet per image. Therefore this lab does **not** claim to be a realistic crowded multi-object benchmark.

It teaches real end-to-end training while keeping:

- assignment visible
- output shapes small
- CPU experiments possible

A later extension can use Penn-Fudan or a small COCO subset for multi-instance detection/segmentation.

## CI policy

CI never downloads Oxford-IIIT Pet.

The lab's tests use synthetic images/masks to verify:

- trimap → binary mask conversion
- mask → box conversion
- YOLO output shape
- DETR query output shape
- segmentation output shape

## Exercises

1. Add horizontal-flip augmentation.
2. Train at 192×192 instead of 128×128.
3. Add GIoU to detector training.
4. Increase DETR query count.
5. Compare YOLO vs DETR mean IoU and latency.
6. Keep trimap boundary as ignore rather than foreground.
7. Replace TinyUNet with torchvision FCN/DeepLab.
8. Extend the lab to Penn-Fudan for multiple pedestrian instances.
