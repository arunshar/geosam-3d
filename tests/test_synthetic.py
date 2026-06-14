"""Smoke tests for the SYNTHETIC training/eval path.

These verify only that the synthetic stand-ins are runnable and produce
correctly shaped tensors. They do NOT assert any benchmark accuracy, because
the real ScanNet/Replica/ScanNet++ pipeline is not implemented.
"""
from __future__ import annotations

import pytest
import torch

from geosam3d.data import ScanNetMonocularDataset, SyntheticClip


def test_synthetic_dataset_shapes():
    ds = ScanNetMonocularDataset(num_scenes=3, gaussians_per_scene=20, num_masks=5)
    assert len(ds) == 3
    clip = ds[0]
    assert isinstance(clip, SyntheticClip)
    assert clip.frames == []  # synthetic: never touches MonoGS
    sd = clip.scene_dict
    n = 20
    assert sd["means"].shape == (n, 3)
    assert sd["colors"].shape == (n, 3)
    assert sd["scales"].shape == (n, 3)
    assert sd["opacities"].shape == (n,)
    assert sd["quats"].shape == (n, 4)
    assert sd["mask_ids"].shape == (n,)
    assert int(sd["mask_ids"].max()) < 5


def test_gather_attrs_matches_in_dim():
    pytest.importorskip("hydra")  # train.py imports hydra at module load
    from geosam3d.training.train import _gather_attrs

    ds = ScanNetMonocularDataset(num_scenes=1, gaussians_per_scene=8)
    attrs = _gather_attrs(ds[0].scene_dict)
    assert attrs.shape == (8, 12)  # must match configs/default.yaml features.in_dim
    assert torch.isfinite(attrs).all()


def test_eval_demo_is_synthetic_and_runs():
    from geosam3d.eval.scannet_eval import _synthetic_demo

    result = _synthetic_demo(seed=0)
    summary = result["summary"]
    # Numbers exist but are explicitly labelled synthetic, not a benchmark.
    assert "SYNTHETIC" in summary["data"]
    assert 0.0 <= summary["mIoU_mean"] <= 1.0
    assert 0.0 <= summary["temporal_jaccard_mean"] <= 1.0
    assert result["per_scene"][0]["scene"].startswith("SYNTHETIC")
