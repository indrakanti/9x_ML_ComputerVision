from __future__ import annotations

import numpy as np
import pytest
import torch
from tensor_contract import contract_from_tensor, hwc_uint8_to_nchw_float


def test_from_numpy_shares_cpu_memory() -> None:
    array = np.zeros((2, 3, 1), dtype=np.uint8)
    tensor = torch.from_numpy(array)

    tensor[0, 0, 0] = 7

    assert int(array[0, 0, 0]) == 7


def test_permute_changes_shape_and_strides_without_changing_values() -> None:
    hwc = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(2, 3, 4)
    chw = hwc.permute(2, 0, 1)

    assert tuple(chw.shape) == (4, 2, 3)
    assert chw[2, 1, 0].item() == hwc[1, 0, 2].item()
    assert not chw.is_contiguous()


def test_contiguous_materialization_preserves_values() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    permuted = source.permute(2, 0, 1)
    contiguous = permuted.contiguous()

    assert contiguous.is_contiguous()
    assert torch.equal(contiguous, permuted)
    assert contiguous.stride() != permuted.stride()


def test_hwc_uint8_to_nchw_float() -> None:
    image = torch.tensor(
        [[[0, 127, 255], [255, 0, 127]]],
        dtype=torch.uint8,
    )

    result = hwc_uint8_to_nchw_float(image)

    assert tuple(result.shape) == (1, 3, 1, 2)
    assert result.dtype == torch.float32
    assert result.is_contiguous()
    assert result.min().item() == pytest.approx(0.0)
    assert result.max().item() == pytest.approx(1.0)


def test_hwc_conversion_rejects_wrong_dtype() -> None:
    with pytest.raises(TypeError):
        hwc_uint8_to_nchw_float(torch.zeros((2, 2, 3), dtype=torch.float32))


def test_tensor_contract_reports_runtime_metadata() -> None:
    tensor = torch.zeros((1, 3, 4, 5), dtype=torch.float32)
    contract = contract_from_tensor(tensor)

    assert contract["shape"] == [1, 3, 4, 5]
    assert contract["strides"] == [60, 20, 5, 1]
    assert contract["dtype"] == "float32"
    assert contract["device"] == "cpu"
    assert contract["contiguous"] is True
