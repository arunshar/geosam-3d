"""ScanNet monocular split.

ScanNet ships RGB+D, but for our setup we treat depth as a held-out signal
and operate from RGB only. The dataset yields short clips (~1-2 sec) of
sequential frames per scene.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class ScanNetClip:
    scene: str
    frames: list[torch.Tensor]
    intrinsics: torch.Tensor


class ScanNetMonocularDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", clip_length: int = 30):
        self.root = Path(root)
        self.split = split
        self.clip_length = int(clip_length)
        self.entries = sorted(self.root.glob(f"{split}/*"))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> ScanNetClip:
        scene_dir = self.entries[idx]
        frames = sorted((scene_dir / "color").glob("*.jpg"))[:self.clip_length]
        if not frames:
            raise FileNotFoundError(f"no frames at {scene_dir / 'color'}")
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("imageio is required") from exc
        imgs = [torch.from_numpy(iio.imread(p)).permute(2, 0, 1).float() / 255.0 for p in frames]
        intrinsics = self._load_intrinsics(scene_dir)
        return ScanNetClip(scene=scene_dir.name, frames=imgs, intrinsics=intrinsics)

    def _load_intrinsics(self, scene_dir: Path) -> torch.Tensor:
        path = scene_dir / "intrinsic" / "intrinsic_color.txt"
        if not path.exists():
            return torch.eye(4)
        try:
            import numpy as np
            return torch.from_numpy(np.loadtxt(path).astype("float32"))
        except Exception:
            return torch.eye(4)
