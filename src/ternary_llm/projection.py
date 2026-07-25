from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def normalized_hadamard(size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Build a normalized Sylvester Hadamard matrix.

    The current experiments deliberately use power-of-two model widths so this
    exact, dependency-free construction is sufficient.
    """
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(size)


def coat_projection(covariance: Tensor) -> Tensor:
    """Return COAT's closed-form Q = U H projection for a covariance matrix."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    size = covariance.shape[0]
    covariance64 = covariance.detach().to(dtype=torch.float64, device="cpu")
    _, eigenvectors = torch.linalg.eigh(covariance64)

    # Eigenvector signs are mathematically arbitrary. Canonicalize them so the
    # saved projection is reproducible across equivalent eigendecompositions.
    pivots = eigenvectors.abs().argmax(dim=0)
    signs = eigenvectors[pivots, torch.arange(size)].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    eigenvectors = eigenvectors * signs

    hadamard = normalized_hadamard(size, dtype=torch.float64)
    return (eigenvectors @ hadamard).to(torch.float32)


def covariance_diagonal_cv(covariance: Tensor, projection: Tensor | None = None) -> float:
    """Coefficient of variation of marginal variances, before or after Q."""
    matrix = covariance.detach().to(torch.float64)
    if projection is not None:
        q = projection.detach().to(torch.float64)
        matrix = q.T @ matrix @ q
    diagonal = matrix.diagonal()
    return float((diagonal.std(unbiased=False) / diagonal.mean().clamp_min(1e-12)).item())


@dataclass
class CovarianceAccumulator:
    """Streaming, numerically stable covariance using parallel Welford updates."""

    size: int

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = torch.zeros(self.size, dtype=torch.float64)
        self.m2 = torch.zeros((self.size, self.size), dtype=torch.float64)

    @torch.no_grad()
    def update(self, samples: Tensor) -> None:
        values = samples.detach().reshape(-1, self.size).cpu().to(torch.float64)
        if values.shape[0] == 0:
            return
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + torch.outer(delta, delta) * (
            self.count * batch_count / total
        )
        self.mean += delta * (batch_count / total)
        self.count = total

    def covariance(self) -> Tensor:
        if self.count < 2:
            raise ValueError("at least two samples are required to estimate covariance")
        return self.m2 / (self.count - 1)
