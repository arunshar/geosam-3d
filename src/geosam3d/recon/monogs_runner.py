"""Thin wrapper around MonoGS for monocular 3DGS reconstruction.

We expect MonoGS to be installed as a sibling package. The runner exposes
a uniform API that the rest of GeoSAM-3D treats as a black-box scene
reconstructor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class MonoGSConfig:
    iters: int = 30000
    sh_degree: int = 3
    densify_until: int = 15000
    use_depth_prior: bool = True
    depth_prior_model: str = "depth-anything-v2-large"


class MonoGSRunner:
    def __init__(self, cfg: MonoGSConfig):
        self.cfg = cfg

    def reconstruct(self, frames: Iterable[Path], out_dir: Path) -> dict:
        """Run MonoGS on a sequence of monocular RGB frames.

        Returns:
            dict with keys ``means`` (N, 3), ``colors`` (N, 3), ``opacities``
            (N,), ``scales`` (N, 3), ``quats`` (N, 4).
        """
        try:
            from monogs.scripts.train import train as monogs_train  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "MonoGS not installed. See https://github.com/spla-tam/MonoGS"
            ) from exc

        out_dir.mkdir(parents=True, exist_ok=True)
        monogs_train(
            input=str(frames),
            output=str(out_dir),
            iters=self.cfg.iters,
            sh_degree=self.cfg.sh_degree,
            depth_prior=self.cfg.depth_prior_model if self.cfg.use_depth_prior else None,
        )
        return self._load_scene(out_dir)

    @staticmethod
    def _load_scene(path: Path) -> dict:
        ply = path / "point_cloud.ply"
        if not ply.exists():
            raise FileNotFoundError(f"MonoGS did not write {ply}")
        # Stub: in production, parse the .ply with plyfile.
        return {
            "means": torch.zeros(0, 3),
            "colors": torch.zeros(0, 3),
            "opacities": torch.zeros(0),
            "scales": torch.zeros(0, 3),
            "quats": torch.zeros(0, 4),
        }
