"""Datasets for GeoSAM-3D.

STATUS: SYNTHETIC ONLY. The real ScanNet / Replica / ScanNet++ monocular
loaders are NOT implemented yet. The class below, ``ScanNetMonocularDataset``,
does not read any ScanNet data: it yields small random Gaussian-field tensors
of the right shapes so that ``python -m geosam3d.training.train`` imports and
runs a short smoke loop without external data or a GPU. Treat every tensor it
produces as random noise, not as a real scene.

When the real loader is written it should keep this same interface:
each ``__getitem__`` returns a ``SyntheticClip`` whose ``scene`` dict carries
per-Gaussian attributes with the keys the training loop and ``_gather_attrs``
expect: ``means`` (N, 3), ``colors`` (N, 3), ``scales`` (N, 3),
``opacities`` (N,), ``quats`` (N, 4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class SyntheticClip:
    """A single synthetic training item.

    Attributes:
        scene: name/id of the synthetic scene (used for the output subdir).
        frames: placeholder list of frame paths. EMPTY for synthetic data;
            the synthetic dataset never touches MonoGS, so no real frames
            exist. Kept for interface parity with the (unimplemented) real
            loader.
        scene_dict: per-Gaussian attribute tensors with keys
            ``means`` (N, 3), ``colors`` (N, 3), ``scales`` (N, 3),
            ``opacities`` (N,), ``quats`` (N, 4). All RANDOM.
    """

    scene: str
    frames: list = field(default_factory=list)
    scene_dict: dict = field(default_factory=dict)


class ScanNetMonocularDataset:
    """SYNTHETIC stand-in for a ScanNet monocular dataset.

    WARNING: This is NOT a real ScanNet loader. It ignores ``root`` and
    fabricates random Gaussian fields so the training command is runnable
    as a smoke test. The real loader (parse ScanNet monocular splits, run
    MonoGS, project SAM 2 masks) is not implemented; see README.

    Args:
        root: ignored (kept for interface parity).
        split: ignored except to vary the random seed a little.
        clip_length: ignored for synthetic data.
        num_scenes: how many synthetic scenes to yield.
        gaussians_per_scene: number of synthetic Gaussians per scene.
        num_masks: number of distinct synthetic SAM-2-style mask ids.
        seed: base RNG seed for reproducibility.
    """

    def __init__(
        self,
        root: str | Path = "data/scannet",
        split: str = "train",
        clip_length: int = 30,
        num_scenes: int = 2,
        gaussians_per_scene: int = 64,
        num_masks: int = 8,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.split = split
        self.clip_length = int(clip_length)
        self.num_scenes = int(num_scenes)
        self.gaussians_per_scene = int(gaussians_per_scene)
        self.num_masks = int(num_masks)
        self.seed = int(seed) + (1 if split == "val" else 0)

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> SyntheticClip:
        if not 0 <= idx < self.num_scenes:
            raise IndexError(idx)
        g = torch.Generator().manual_seed(self.seed + idx)
        n = self.gaussians_per_scene
        scene_dict = {
            "means": torch.randn(n, 3, generator=g),
            "colors": torch.rand(n, 3, generator=g),
            "scales": torch.rand(n, 3, generator=g) * 0.1,
            "opacities": torch.rand(n, generator=g),
            "quats": torch.randn(n, 4, generator=g),
            # SYNTHETIC pseudo-mask ids standing in for projected SAM 2 masks.
            "mask_ids": torch.randint(0, self.num_masks, (n,), generator=g),
        }
        return SyntheticClip(
            scene=f"synthetic_scene_{idx:03d}",
            frames=[],
            scene_dict=scene_dict,
        )


def swiss_roll_scene(
    n_gaussians: int = 700,
    n_classes: int = 6,
    turns: float = 2.0,
    height: float = 6.0,
    noise: float = 0.02,
    seed: int = 0,
) -> dict:
    """A synthetic *manifold* scene whose geometry has real structure.

    Points lie on a Swiss roll; the ground-truth segments are contiguous arcs
    along the roll (`labels`). Non-adjacent arcs come Euclidean-close across
    roll layers but stay geodesically far along the sheet, so geodesic label
    propagation should beat a Euclidean nearest-seed baseline (the paper's core
    claim). Unlike ``ScanNetMonocularDataset`` (pure noise), this scene is
    structured so that claim is measurable. Returns the usual per-Gaussian keys
    plus integer ``labels`` (N,) ground truth.

    This is SYNTHETIC manifold data, not ScanNet/Replica/ScanNet++.
    """
    g = torch.Generator().manual_seed(int(seed))
    u = torch.rand(n_gaussians, generator=g)                 # along-roll coord in [0, 1]
    t_min = 1.5 * math.pi
    t_max = (1.5 + 2.0 * float(turns)) * math.pi
    t = t_min + u * (t_max - t_min)
    means = torch.stack([t * torch.cos(t), height * torch.rand(n_gaussians, generator=g), t * torch.sin(t)], dim=-1)
    means = means + noise * t_max * torch.randn(n_gaussians, 3, generator=g)
    means = means - means.mean(dim=0, keepdim=True)
    means = means / means.abs().max().clamp_min(1e-6)        # ~[-1, 1] for stable kernel numerics
    labels = (u * n_classes).floor().clamp(0, n_classes - 1).long()
    quats = torch.randn(n_gaussians, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return {
        "means": means,
        "colors": torch.rand(n_gaussians, 3, generator=g),
        "scales": torch.rand(n_gaussians, 3, generator=g) * 0.1,
        "opacities": torch.rand(n_gaussians, generator=g),
        "quats": quats,
        "labels": labels,
        "mask_ids": labels.clone(),
    }


__all__ = ["ScanNetMonocularDataset", "SyntheticClip", "swiss_roll_scene"]
