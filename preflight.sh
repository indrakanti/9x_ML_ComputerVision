#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-preflight}"

echo "==> Ruff"
ruff check .

echo "==> mypy"
mypy [0-9][0-9]_*/python

echo "==> pytest"
pytest

echo "==> configure native C++"
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON

echo "==> build native C++"
cmake --build "${BUILD_DIR}" --parallel

echo "==> Python ↔ C++ parity"
python 02_ProbabilityStatistics/python/statistics_native_parity.py \
  "${BUILD_DIR}/02_ProbabilityStatistics/cv9x_statistics"
python 03_Optimization/python/optimization_native_parity.py \
  "${BUILD_DIR}/03_Optimization/cv9x_optimizer_state"
python 04_Autograd/python/autodiff_native_parity.py \
  "${BUILD_DIR}/04_Autograd/cv9x_autodiff"
python 05_DataDiscipline/python/data_discipline_native_parity.py \
  "${BUILD_DIR}/05_DataDiscipline/cv9x_data_discipline"
python 06_LinearClassifiers/python/linear_classifier_native_parity.py \
  "${BUILD_DIR}/06_LinearClassifiers/cv9x_linear_classifier" \
  06_LinearClassifiers/artifacts/linear_binary_model.txt
python 07_MLPs/python/mlp_native_parity.py \
  "${BUILD_DIR}/07_MLPs/cv9x_mlp" \
  07_MLPs/artifacts/mlp_binary_model.txt
python 08_Convolution/python/convolution_native_parity.py \
  "${BUILD_DIR}/08_Convolution/cv9x_convolution"
python 09_CNNTraining/python/cnn_training_native_parity.py \
  "${BUILD_DIR}/09_CNNTraining/cv9x_tiny_cnn" \
  09_CNNTraining/artifacts/tiny_cnn_smoke.txt
python 10_TrainingRobustness/python/training_robustness_native_parity.py \
  "${BUILD_DIR}/10_TrainingRobustness/cv9x_deployment_manifest" \
  10_TrainingRobustness/artifacts/deployment_manifest.txt
python 11_TransferLearningONNX/python/transfer_learning_native_parity.py \
  "${BUILD_DIR}/11_TransferLearningONNX/cv9x_transfer_model" \
  11_TransferLearningONNX/artifacts/transfer_model_smoke.txt
python 12_ResNet/python/residual_learning_native_parity.py \
  "${BUILD_DIR}/12_ResNet/cv9x_residual_block" \
  12_ResNet/artifacts/residual_block.txt
python 13_EfficientCNN/python/efficient_cnn_native_parity.py \
  "${BUILD_DIR}/13_EfficientCNN/cv9x_depthwise_separable" \
  13_EfficientCNN/artifacts/depthwise_separable.txt
python 14_VisionTransformer/python/vision_transformer_native_parity.py \
  "${BUILD_DIR}/14_VisionTransformer/cv9x_tiny_vit" \
  14_VisionTransformer/artifacts/tiny_vit.txt
python 15_WindowedTransformers/python/window_attention_native_parity.py \
  "${BUILD_DIR}/15_WindowedTransformers/cv9x_window_attention" \
  15_WindowedTransformers/artifacts/window_attention.txt
python 16_HybridBackbones/python/hybrid_backbone_native_parity.py \
  "${BUILD_DIR}/16_HybridBackbones/cv9x_hybrid_backbone" \
  16_HybridBackbones/artifacts/hybrid.txt
python 17_BackboneEvaluation/python/backbone_eval_native_parity.py \
  "${BUILD_DIR}/17_BackboneEvaluation/cv9x_backbone_eval" \
  17_BackboneEvaluation/artifacts/backbones.txt
python 18_DetectionFundamentals/python/detection_native_parity.py \
  "${BUILD_DIR}/18_DetectionFundamentals/cv9x_detection_fundamentals" \
  18_DetectionFundamentals/artifacts/detections.txt
python 19_YOLOOneStage/python/yolo_native_parity.py \
  "${BUILD_DIR}/19_YOLOOneStage/cv9x_yolo_decode" \
  19_YOLOOneStage/artifacts/yolo_head.txt
python 20_TwoStageDetection/python/two_stage_native_parity.py \
  "${BUILD_DIR}/20_TwoStageDetection/cv9x_two_stage" \
  20_TwoStageDetection/artifacts/proposals.txt
python 21_DETR/python/detr_native_parity.py \
  "${BUILD_DIR}/21_DETR/cv9x_detr_match" \
  21_DETR/artifacts/matching.txt
python 22_SemanticSegmentation/python/semantic_native_parity.py \
  "${BUILD_DIR}/22_SemanticSegmentation/cv9x_semantic_metrics" \
  22_SemanticSegmentation/artifacts/segmentation.txt
python 23_InstancePanoptic/python/panoptic_native_parity.py \
  "${BUILD_DIR}/23_InstancePanoptic/cv9x_panoptic_compose" \
  23_InstancePanoptic/artifacts/panoptic.txt

echo "==> CTest"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo
echo "PRE-PR PREFLIGHT: PASS"
