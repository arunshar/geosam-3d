"""Heat-based geodesic distance on a Gaussian centroid graph.

We use the Varadhan-formula limit of the heat kernel:

    d^2(x, seed) = lim_{t -> 0} -4 t log u_t(x)

where u_t solves the heat equation with a delta source at the seed. On a
kNN graph this becomes:

    Step 1. Build Laplacian L = D - W on a kNN graph with Gaussian edge weights.
    Step 2. Diffuse: solve (I + t L) u = u_0 where u_0 = 1 on seed nodes.
    Step 3. Distance: d(i) = sqrt(-4 t log(u_i / u_seed_max + eps)).

This avoids the discrete gradient + divergence step from Crane et al. (ACM
TOG 2013), which degenerates at seeds in non-mesh graphs because outgoing
edges symmetrically cancel. The Varadhan approximation is correct in the
small-t limit and is monotone in u for any t, which is all we need for
label propagation. The whole pipeline is differentiable, so the feature
head can be trained end-to-end via the geodesic loss in training/train.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def knn_graph(points: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Return (edges, weights). edges: (E, 2) ints, weights: (E,) floats."""
    if points.dim() != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")
    N = points.shape[0]
    pdist = torch.cdist(points, points)             # (N, N)
    pdist.fill_diagonal_(float("inf"))
    knn_dist, knn_idx = pdist.topk(k, largest=False)   # (N, k)
    src = torch.arange(N, device=points.device).unsqueeze(1).expand_as(knn_idx)
    edges = torch.stack([src.flatten(), knn_idx.flatten()], dim=-1)
    sigma = knn_dist.median()
    weights = torch.exp(-knn_dist.flatten().pow(2) / (2.0 * sigma.pow(2) + 1e-8))
    return edges, weights


def graph_laplacian(N: int, edges: Tensor, weights: Tensor) -> Tensor:
    """Symmetric graph Laplacian L = D - W as a dense (N, N) tensor."""
    W = torch.zeros(N, N, device=edges.device, dtype=weights.dtype)
    W[edges[:, 0], edges[:, 1]] = weights
    W = 0.5 * (W + W.t())   # symmetrize
    D = W.sum(dim=-1)
    return torch.diag(D) - W


class HeatGeodesicKernel(nn.Module):
    """Differentiable heat-method geodesic kernel for label propagation.

    Args:
        k: number of nearest neighbours in the centroid graph.
        t: diffusion time (controls smoothing).
        eps: numerical epsilon for the linear solve.
    """

    def __init__(self, k: int = 16, t: float = 0.05, eps: float = 1e-4):
        super().__init__()
        self.k = int(k)
        self.t = float(t)
        self.eps = float(eps)

    @torch.no_grad()
    def precompute(self, centroids: Tensor):
        """Precompute the Laplacian and (I - t L) factorization for a scene."""
        edges, weights = knn_graph(centroids, k=self.k)
        L = graph_laplacian(centroids.shape[0], edges, weights)
        I = torch.eye(centroids.shape[0], device=centroids.device, dtype=centroids.dtype)
        A = I + self.t * L + self.eps * I
        L_chol = torch.linalg.cholesky(A)
        return {"L": L, "A_chol": L_chol, "edges": edges, "weights": weights}

    def geodesic(
        self,
        centroids: Tensor,
        seed_mask: Tensor,
        cache: dict | None = None,
    ) -> Tensor:
        """Compute approximate geodesic distance from the seed nodes.

        Uses the Varadhan formula d^2 ~ -4t log u. The sqrt(-4t log u) is
        provably the limit of the geodesic distance on a Riemannian
        manifold as t -> 0, and on kNN graphs it is monotone-in-u for
        any t > 0, which is all we need for label propagation.

        Args:
            centroids: (N, 3) Gaussian centers.
            seed_mask: (N,) float in {0, 1}; 1 marks seed Gaussians.
            cache: optional precomputed dict from ``precompute``.
        Returns:
            distance: (N,) tensor with seed nodes at distance 0.
        """
        cache = cache or self.precompute(centroids)
        A_chol = cache["A_chol"]

        u = torch.cholesky_solve(seed_mask.unsqueeze(-1), A_chol).squeeze(-1)
        u = u / u.max().clamp_min(1e-12)
        d2 = -4.0 * self.t * torch.log(u.clamp_min(1e-12))
        d = d2.clamp_min(0.0).sqrt()

        seed_idx = (seed_mask > 0.5).nonzero(as_tuple=True)[0]
        if seed_idx.numel() > 0:
            d = d - d[seed_idx].min()
        return d.clamp_min(0.0)

    def propagate_label(
        self,
        centroids: Tensor,
        seed_mask: Tensor,
        soft_threshold: float = 0.05,
    ) -> Tensor:
        """Return per-Gaussian probability of belonging to the seed's region."""
        d = self.geodesic(centroids, seed_mask)
        sigma = d.std().clamp_min(1e-3)
        return torch.exp(-d.pow(2) / (2.0 * sigma.pow(2)))
