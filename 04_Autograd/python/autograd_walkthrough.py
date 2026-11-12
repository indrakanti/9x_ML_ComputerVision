"""Autograd walkthrough for Module 04."""

from __future__ import annotations

import torch
from autodiff import Value, central_difference


def main() -> None:
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")
    loss = (x * y + x) ** 2
    loss.backward()

    print("Tiny autodiff")
    print(" loss:", loss.data)
    print(" dL/dx:", x.grad)
    print(" dL/dy:", y.grad)

    xt = torch.tensor(2.0, requires_grad=True)
    yt = torch.tensor(3.0, requires_grad=True)
    loss_t = (xt * yt + xt) ** 2
    loss_t.backward()

    assert xt.grad is not None
    assert yt.grad is not None

    print("\nPyTorch autograd")
    print(" loss:", float(loss_t.detach()))
    print(" dL/dx:", float(xt.grad))
    print(" dL/dy:", float(yt.grad))

    numerical = central_difference(lambda v: (v * 3.0 + v) ** 2, 2.0)
    print("\nNumerical dL/dx:", numerical)


if __name__ == "__main__":
    main()
