"""Tiny scalar reverse-mode autodiff engine for Module 04."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

BackwardFn = Callable[[], None]


@dataclass(eq=False)
class Value:
    data: float
    label: str = ""
    grad: float = 0.0
    _prev: tuple[Value, ...] = field(default_factory=tuple, repr=False)
    _backward: BackwardFn = field(default=lambda: None, repr=False)
    _op: str = field(default="", repr=False)

    def __add__(self, other: float | Value) -> Value:
        rhs = other if isinstance(other, Value) else Value(float(other))
        out = Value(self.data + rhs.data, _prev=(self, rhs), _op="+")

        def backward() -> None:
            self.grad += out.grad
            rhs.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: float | Value) -> Value:
        return self + other

    def __mul__(self, other: float | Value) -> Value:
        rhs = other if isinstance(other, Value) else Value(float(other))
        out = Value(self.data * rhs.data, _prev=(self, rhs), _op="*")

        def backward() -> None:
            self.grad += rhs.data * out.grad
            rhs.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: float | Value) -> Value:
        return self * other

    def __pow__(self, exponent: int | float) -> Value:
        out = Value(self.data**exponent, _prev=(self,), _op=f"**{exponent}")

        def backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: float | Value) -> Value:
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other: float | Value) -> Value:
        lhs = other if isinstance(other, Value) else Value(float(other))
        return lhs - self

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        for node in topo:
            node.grad = 0.0


def central_difference(
    fn: Callable[[float], float],
    x: float,
    *,
    step: float = 1e-6,
) -> float:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    return (fn(x + step) - fn(x - step)) / (2.0 * step)


def graph_metadata(root: Value) -> dict[str, Any]:
    visited: set[Value] = set()

    def visit(node: Value) -> None:
        if node in visited:
            return
        visited.add(node)
        for parent in node._prev:
            visit(parent)

    visit(root)
    return {"nodes": len(visited), "root_op": root._op, "root_value": root.data}
