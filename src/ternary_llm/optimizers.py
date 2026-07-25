from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from ternary_llm.quantization import ternary_code


def _parameter_scale(parameter: Tensor) -> Tensor:
    dimensions: int | tuple[int, ...]
    dimensions = 0 if parameter.ndim <= 1 else tuple(range(1, parameter.ndim))
    return (
        parameter.detach()
        .abs()
        .mean(dim=dimensions, keepdim=True)
        .clamp_min(1e-5)
        .to(torch.float32)
    )


class TernaryCounterOptimizer(torch.optim.Optimizer):
    """Discrete ternary transitions driven by small signed evidence counters.

    PyTorch still requires floating tensors for autograd, but after every optimizer
    step each parameter is exactly a per-row scale times a code in {-1, 0, +1}.
    There is no full-precision shadow parameter or Adam moment state.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        counter_threshold: int = 8,
    ) -> None:
        if not 1 <= counter_threshold <= 127:
            raise ValueError("counter_threshold must be between 1 and 127")
        super().__init__(parameters, {"counter_threshold": counter_threshold})

    @torch.no_grad()
    def _initialize(self, parameter: Tensor, state: dict[str, Any]) -> None:
        scale = _parameter_scale(parameter)
        codes = ternary_code(parameter.detach() / scale)
        parameter.copy_(codes * scale.to(parameter.dtype))
        state["scale"] = scale
        state["counter"] = torch.zeros_like(parameter, dtype=torch.int8)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            threshold = int(group["counter_threshold"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    self._initialize(parameter, state)

                gradient = parameter.grad.detach()
                finite = torch.isfinite(gradient)
                evidence = -gradient.sign().to(torch.int16)
                evidence = torch.where(finite, evidence, torch.zeros_like(evidence))
                counter = state["counter"].to(torch.int16)
                counter = (counter + evidence).clamp(-threshold, threshold)

                scale = state["scale"].to(parameter.device)
                codes = torch.round(parameter.detach().to(torch.float32) / scale).clamp(-1, 1)
                move_up = counter >= threshold
                move_down = counter <= -threshold
                updated_codes = codes + move_up.to(codes.dtype) - move_down.to(codes.dtype)
                updated_codes.clamp_(-1, 1)
                transitioned = move_up | move_down
                counter = torch.where(transitioned, torch.zeros_like(counter), counter)

                parameter.copy_(updated_codes.to(parameter.dtype) * scale.to(parameter.dtype))
                state["counter"] = counter.to(torch.int8)
        return loss


class StochasticTernaryOptimizer(torch.optim.Optimizer):
    """Ternary state transitions sampled directly from normalized gradients."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        transition_rate: float = 0.002,
    ) -> None:
        if not 0.0 < transition_rate <= 1.0:
            raise ValueError("transition_rate must be in (0, 1]")
        super().__init__(parameters, {"transition_rate": transition_rate})

    @torch.no_grad()
    def _initialize(self, parameter: Tensor, state: dict[str, Any]) -> None:
        scale = _parameter_scale(parameter)
        codes = ternary_code(parameter.detach() / scale)
        parameter.copy_(codes * scale.to(parameter.dtype))
        state["scale"] = scale

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            transition_rate = float(group["transition_rate"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    self._initialize(parameter, state)

                gradient = parameter.grad.detach().to(torch.float32)
                finite_gradient = torch.where(
                    torch.isfinite(gradient),
                    gradient,
                    torch.zeros_like(gradient),
                )
                normalizer = finite_gradient.abs().mean().clamp_min(1e-12)
                probability = (
                    transition_rate * finite_gradient.abs() / normalizer
                ).clamp_max(1.0)
                transition = torch.rand_like(probability) < probability
                direction = -finite_gradient.sign()

                scale = state["scale"].to(parameter.device)
                codes = torch.round(parameter.detach().to(torch.float32) / scale).clamp(-1, 1)
                codes.add_(transition.to(codes.dtype) * direction).clamp_(-1, 1)
                parameter.copy_(codes.to(parameter.dtype) * scale.to(parameter.dtype))
        return loss


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor):
                total += value.numel() * value.element_size()
    return total
