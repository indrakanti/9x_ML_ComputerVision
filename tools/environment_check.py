#!/usr/bin/env python3
"""Print a human-readable ML Computer Vision environment report."""

from __future__ import annotations

import json

from cv9x_mlcv.environment import environment_is_healthy, environment_report


def main() -> int:
    report = environment_report()

    print("9x ML Computer Vision environment")
    print("=" * 40)

    ordered_keys = [
        "python",
        "python_executable",
        "platform",
        "numpy",
        "opencv",
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
        "cuda_available",
        "cuda_runtime",
        "gpu_name",
        "onnxruntime_providers",
        "tensor_math_ok",
        "autograd_ok",
        "opencv_ok",
    ]

    for key in ordered_keys:
        print(f"{key:24}: {report.get(key)}")

    print("\nJSON report")
    print(json.dumps(report, indent=2, default=str))

    healthy = environment_is_healthy()
    print(f"\nOverall status: {'PASS' if healthy else 'FAIL'}")

    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
