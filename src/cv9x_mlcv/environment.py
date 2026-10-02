"""Environment validation helpers for the 9x ML Computer Vision course."""

from __future__ import annotations

import platform
import sys
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision


def tensor_math_ok() -> bool:
    """Return True when basic PyTorch tensor math works."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    result = a @ b
    expected = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
    return bool(torch.allclose(result, expected))


def autograd_ok() -> bool:
    """Return True when a deterministic autograd calculation is correct."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x**2).sum()
    loss.backward()

    if x.grad is None:
        return False

    expected = torch.tensor([2.0, 4.0, 6.0])
    return bool(torch.allclose(x.grad, expected))


def opencv_ok() -> bool:
    """Return True when a simple OpenCV image operation is correct."""
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 255

    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    return (
        blurred.shape == image.shape
        and blurred.dtype == image.dtype
        and int(blurred.max()) > 0
        and int(blurred.sum()) > 0
    )


def environment_report() -> dict[str, Any]:
    """Return a compact report of the course runtime environment."""
    cuda_available = torch.cuda.is_available()

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "tensor_math_ok": tensor_math_ok(),
        "autograd_ok": autograd_ok(),
        "opencv_ok": opencv_ok(),
    }


def environment_is_healthy() -> bool:
    """Return True when the minimum executable course environment is healthy."""
    report = environment_report()

    major, minor = sys.version_info[:2]

    return bool(
        (major, minor) >= (3, 10)
        and report["tensor_math_ok"]
        and report["autograd_ok"]
        and report["opencv_ok"]
    )
