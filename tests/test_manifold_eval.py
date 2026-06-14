"""Real manifold evaluation: geodesic label propagation should beat Euclidean.

The Swiss-roll scene has genuine manifold structure (arcs that are Euclidean-close
across roll layers but geodesically far), so this is a measurable check of the
paper's core claim, not a self-consistency check on noise.
"""
from __future__ import annotations

import torch

from geosam3d.data import swiss_roll_scene
from geosam3d.eval.scannet_eval import (
    euclidean_nearest_seed,
    evaluate_manifold,
    multiclass_miou,
)


def test_swiss_roll_scene_shapes_and_labels():
    scene = swiss_roll_scene(n_gaussians=300, n_classes=6, seed=0)
    assert scene["means"].shape == (300, 3)
    assert scene["labels"].shape == (300,)
    assert int(scene["labels"].min()) >= 0 and int(scene["labels"].max()) <= 5
    assert scene["means"].abs().max() <= 1.0 + 1e-5     # normalized
    # reproducible
    scene2 = swiss_roll_scene(n_gaussians=300, n_classes=6, seed=0)
    assert torch.allclose(scene["means"], scene2["means"])


def test_multiclass_miou_bounds():
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    assert multiclass_miou(target.clone(), target, 3) == 1.0          # perfect
    wrong = torch.tensor([1, 1, 0, 0, 1, 1])
    assert 0.0 <= multiclass_miou(wrong, target, 3) < 1.0


def test_euclidean_nearest_seed_recovers_seed_labels():
    centroids = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0]])
    seed_idx = torch.tensor([0, 2])
    seed_labels = torch.tensor([0, 1])
    pred = euclidean_nearest_seed(centroids, seed_idx, seed_labels)
    assert pred.tolist() == [0, 0, 1, 1]


def test_geodesic_beats_euclidean_on_manifold():
    scene = swiss_roll_scene(seed=0)
    res = evaluate_manifold(scene, seeds_per_class=3, k=10, t=0.05, seed=0)
    # the geodesic kernel must win clearly on the folded manifold, and be accurate
    assert res["geodesic_miou"] > res["euclidean_miou"]
    assert res["geodesic_miou"] > 0.6
    assert res["n_eval"] == res["n_points"] - res["n_seeds"]
