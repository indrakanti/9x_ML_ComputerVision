"""Module 01 walkthrough: image/tensor layout, strides, views, copies and devices."""

from __future__ import annotations

import json

import numpy as np
import torch

from tensor_contract import contract_from_tensor


def main() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    tensor_hwc = torch.from_numpy(image)

    print("NumPy HWC")
    print(" shape:", image.shape)
    print(" strides bytes:", image.strides)
    print()

    print("PyTorch HWC shared view")
    print(" shape:", tuple(tensor_hwc.shape))
    print(" stride elements:", tensor_hwc.stride())
    print(" contiguous:", tensor_hwc.is_contiguous())

    original = int(image[0, 0, 0])
    tensor_hwc[0, 0, 0] = original + 1
    print(" shared storage mutation visible in NumPy:", int(image[0, 0, 0]) == original + 1)
    print()

    tensor_chw = tensor_hwc.permute(2, 0, 1)
    print("CHW after permute")
    print(" shape:", tuple(tensor_chw.shape))
    print(" stride:", tensor_chw.stride())
    print(" contiguous:", tensor_chw.is_contiguous())

    tensor_chw_contiguous = tensor_chw.contiguous()
    print(" contiguous copy stride:", tensor_chw_contiguous.stride())
    print()

    tensor_float = tensor_chw_contiguous.to(torch.float32) / 255.0
    tensor_nchw = tensor_float.unsqueeze(0)

    print("NCHW model input")
    print(" shape:", tuple(tensor_nchw.shape))
    print(" dtype:", tensor_nchw.dtype)
    print(" device:", tensor_nchw.device)
    print(" min/max:", float(tensor_nchw.min()), float(tensor_nchw.max()))
    print()

    print("Tensor contract")
    print(json.dumps(contract_from_tensor(tensor_nchw), indent=2))

    if torch.cuda.is_available():
        cuda_tensor = tensor_nchw.to("cuda")
        print("\nCUDA copy complete:", cuda_tensor.device)
    else:
        print("\nCUDA unavailable: CPU correctness path is fully supported.")


if __name__ == "__main__":
    main()
