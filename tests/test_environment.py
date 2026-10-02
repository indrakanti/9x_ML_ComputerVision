from __future__ import annotations

import sys

from cv9x_mlcv.environment import (
    autograd_ok,
    environment_is_healthy,
    environment_report,
    opencv_ok,
    tensor_math_ok,
)


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 10)


def test_tensor_math() -> None:
    assert tensor_math_ok()


def test_autograd() -> None:
    assert autograd_ok()


def test_opencv() -> None:
    assert opencv_ok()


def test_environment_report_contains_core_packages() -> None:
    report = environment_report()

    for key in (
        "python",
        "numpy",
        "opencv",
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
        "onnxruntime_providers",
        "cuda_available",
    ):
        assert key in report


def test_environment_is_healthy() -> None:
    assert environment_is_healthy()
