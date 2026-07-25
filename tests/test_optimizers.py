import torch

from ternary_llm.optimizers import (
    StochasticTernaryOptimizer,
    TernaryCounterOptimizer,
    optimizer_state_bytes,
)


def test_counter_optimizer_keeps_scaled_ternary_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([[-0.2, 0.1, 0.8]]))
    optimizer = TernaryCounterOptimizer([parameter], counter_threshold=2)

    for _ in range(2):
        parameter.grad = torch.tensor([[-1.0, -1.0, -1.0]])
        optimizer.step()

    state = optimizer.state[parameter]
    scale = state["scale"]
    codes = torch.round(parameter.detach() / scale)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert state["counter"].dtype == torch.int8
    assert optimizer_state_bytes(optimizer) < parameter.numel() * 8


def test_counter_optimizer_transitions_after_threshold() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = TernaryCounterOptimizer([parameter], counter_threshold=2)

    parameter.grad = torch.tensor([-1.0])
    optimizer.step()
    assert parameter.item() == 0.0
    parameter.grad = torch.tensor([-1.0])
    optimizer.step()
    assert parameter.item() > 0.0


def test_stochastic_optimizer_has_no_per_weight_state() -> None:
    torch.manual_seed(1)
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.2, -0.2]))
    optimizer = StochasticTernaryOptimizer([parameter], transition_rate=1.0)
    parameter.grad = torch.tensor([-1.0, -1.0, 1.0])
    optimizer.step()

    state = optimizer.state[parameter]
    assert set(state) == {"scale"}
    assert parameter[0].item() > 0.0
    assert optimizer_state_bytes(optimizer) < parameter.numel() * 4
