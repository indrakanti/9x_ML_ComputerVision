from __future__ import annotations

import pytest
import torch
from autodiff import Value, central_difference, graph_metadata


def test_addition_gradient() -> None:
    x = Value(2.0)
    y = Value(5.0)
    out = x + y
    out.backward()

    assert x.grad == pytest.approx(1.0)
    assert y.grad == pytest.approx(1.0)


def test_multiplication_gradient() -> None:
    x = Value(2.0)
    y = Value(3.0)
    out = x * y
    out.backward()

    assert x.grad == pytest.approx(3.0)
    assert y.grad == pytest.approx(2.0)


def test_chain_rule_and_shared_path_accumulation() -> None:
    x = Value(2.0)
    y = Value(3.0)
    loss = (x * y + x) ** 2
    loss.backward()

    assert loss.data == pytest.approx(64.0)
    assert x.grad == pytest.approx(64.0)
    assert y.grad == pytest.approx(32.0)


def test_power_gradient() -> None:
    x = Value(4.0)
    out = x**3
    out.backward()

    assert x.grad == pytest.approx(48.0)


def test_zero_grad_clears_graph() -> None:
    x = Value(2.0)
    y = x * x
    y.backward()
    assert x.grad != 0.0

    y.zero_grad()

    assert x.grad == 0.0
    assert y.grad == 0.0


def test_central_difference_matches_analytic_gradient() -> None:
    numerical = central_difference(lambda x: x**3 + 2.0 * x, 1.5)
    analytic = 3.0 * (1.5**2) + 2.0

    assert numerical == pytest.approx(analytic, rel=1e-6, abs=1e-6)


def test_tiny_engine_matches_pytorch() -> None:
    x = Value(2.0)
    y = Value(3.0)
    loss = (x * y + x) ** 2
    loss.backward()

    xt = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    loss_t = (xt * yt + xt) ** 2
    loss_t.backward()

    assert loss.data == pytest.approx(float(loss_t.detach()))
    assert x.grad == pytest.approx(float(xt.grad))
    assert y.grad == pytest.approx(float(yt.grad))


def test_graph_metadata() -> None:
    x = Value(2.0)
    y = Value(3.0)
    out = (x * y) ** 2

    metadata = graph_metadata(out)

    assert metadata["nodes"] == 4
    assert metadata["root_op"] == "**2"


def test_invalid_numerical_step_is_rejected() -> None:
    with pytest.raises(ValueError):
        central_difference(lambda x: x * x, 1.0, step=0.0)
