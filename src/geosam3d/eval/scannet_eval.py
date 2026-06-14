"""ScanNet 3D mIoU and temporal mask consistency for prompt-based 3D segmentation.

STATUS: there is NO real ScanNet / Replica / ScanNet++ benchmark here yet.
The dataset loaders and the SAM 2 / MonoGS pipeline are not implemented, so
this script cannot evaluate on real scenes. To avoid printing fabricated
leaderboard numbers, the default behaviour is to print a clear notice and
exit without numbers.

Pass ``--demo`` to instead run a tiny SYNTHETIC self-check: a random point
cloud is pushed through the real ``HeatGeodesicKernel``, a synthetic
ground-truth region is defined, and mIoU / temporal-Jaccard are computed on
THAT synthetic data. Those numbers measure the kernel's behaviour on noise,
not any benchmark, and are labelled as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geosam3d.propagate import HeatGeodesicKernel


def iou_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return float(inter / max(union, 1))


def multiclass_miou(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    """Mean IoU over the classes present in `target` (classes absent from both skipped)."""
    ious = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((p & t).sum().item() / union)
    return float(sum(ious) / max(len(ious), 1))


def _pick_seeds(labels: torch.Tensor, seeds_per_class: int, n_classes: int, generator) -> torch.Tensor:
    """A few labeled seed indices per class (the only supervision propagation gets)."""
    chunks = []
    for c in range(n_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        perm = torch.randperm(idx.numel(), generator=generator)
        chunks.append(idx[perm[: min(seeds_per_class, idx.numel())]])
    return torch.cat(chunks)


def euclidean_nearest_seed(centroids: torch.Tensor, seed_idx: torch.Tensor, seed_labels: torch.Tensor) -> torch.Tensor:
    """Baseline: each point takes the label of its nearest seed in 3D Euclidean space."""
    nearest = torch.cdist(centroids, centroids[seed_idx]).argmin(dim=1)
    return seed_labels[nearest]


def geodesic_nearest_seed(
    centroids: torch.Tensor, labels: torch.Tensor, seed_idx: torch.Tensor,
    kernel: HeatGeodesicKernel, n_classes: int,
) -> torch.Tensor:
    """Each point takes the class whose seeds are geodesically nearest (heat method)."""
    cache = kernel.precompute(centroids)
    dmat = torch.full((centroids.shape[0], n_classes), float("inf"))
    for c in range(n_classes):
        cls_seeds = seed_idx[labels[seed_idx] == c]
        if cls_seeds.numel() == 0:
            continue
        seed_mask = torch.zeros(centroids.shape[0])
        seed_mask[cls_seeds] = 1.0
        dmat[:, c] = kernel.geodesic(centroids, seed_mask, cache=cache)
    return dmat.argmin(dim=1)


def evaluate_manifold(
    scene: dict, *, seeds_per_class: int = 3, k: int = 10, t: float = 0.05, seed: int = 0
) -> dict:
    """Geodesic vs Euclidean label-propagation mIoU on a structured manifold scene.

    Picks a few seeds per class, propagates labels to the remaining points two
    ways, and scores both by mean IoU over the non-seed points. On a folded
    manifold the geodesic kernel should win, because Euclidean nearest-seed
    bleeds across layers that are close in 3D but far along the surface.
    """
    centroids, labels = scene["means"], scene["labels"]
    n_classes = int(labels.max().item()) + 1
    g = torch.Generator().manual_seed(int(seed))
    seed_idx = _pick_seeds(labels, seeds_per_class, n_classes, g)
    kernel = HeatGeodesicKernel(k=k, t=t)
    geo_pred = geodesic_nearest_seed(centroids, labels, seed_idx, kernel, n_classes)
    euc_pred = euclidean_nearest_seed(centroids, seed_idx, labels[seed_idx])

    nonseed = torch.ones(centroids.shape[0], dtype=torch.bool)
    nonseed[seed_idx] = False
    return {
        "geodesic_miou": round(multiclass_miou(geo_pred[nonseed], labels[nonseed], n_classes), 4),
        "euclidean_miou": round(multiclass_miou(euc_pred[nonseed], labels[nonseed], n_classes), 4),
        "n_classes": n_classes,
        "n_points": int(centroids.shape[0]),
        "n_seeds": int(seed_idx.numel()),
        "n_eval": int(nonseed.sum().item()),
        "data": "SYNTHETIC Swiss-roll manifold (NOT ScanNet/Replica/ScanNet++)",
        "note": "demonstrates geodesic > Euclidean label propagation; not a benchmark result",
    }


def _synthetic_demo(seed: int = 0) -> dict:
    """Run a real heat-geodesic propagation on a SYNTHETIC scene.

    Builds a random point cloud, defines a synthetic ground-truth region as
    the true k-nearest neighbours of a seed point, propagates a label with
    the actual kernel, thresholds it, and reports IoU plus a two-frame
    temporal Jaccard (label stability under a small jitter of the cloud).

    Every number here is computed on random data. It is a self-consistency
    check of the kernel, NOT a ScanNet/Replica/ScanNet++ result.
    """
    torch.manual_seed(seed)
    n = 200
    pts = torch.randn(n, 3)
    seed_idx = 0

    # Synthetic ground truth: the 25 points geodesically nearest the seed,
    # approximated by Euclidean nearest neighbours of the seed point.
    dists_to_seed = (pts - pts[seed_idx]).norm(dim=-1)
    k_region = 25
    gt = torch.zeros(n, dtype=torch.bool)
    gt[dists_to_seed.topk(k_region, largest=False).indices] = True

    kernel = HeatGeodesicKernel(k=12, t=0.05)
    seed_mask = torch.zeros(n)
    seed_mask[seed_idx] = 1.0

    prob = kernel.propagate_label(pts, seed_mask)
    pred = prob > 0.5
    miou = iou_3d(pred, gt)

    # Temporal Jaccard: jitter the cloud slightly (a synthetic "next frame")
    # and measure label-set stability of the prediction.
    pts2 = pts + 0.01 * torch.randn_like(pts)
    prob2 = kernel.propagate_label(pts2, seed_mask)
    pred2 = prob2 > 0.5
    temporal_jaccard = iou_3d(pred, pred2)

    rows = [
        {
            "scene": "SYNTHETIC_demo_scene",
            "mIoU": round(miou, 4),
            "temporal_jaccard": round(temporal_jaccard, 4),
        }
    ]
    return {
        "per_scene": rows,
        "summary": {
            "mIoU_mean": round(sum(r["mIoU"] for r in rows) / len(rows), 4),
            "temporal_jaccard_mean": round(
                sum(r["temporal_jaccard"] for r in rows) / len(rows), 4
            ),
            "n": len(rows),
            "data": "SYNTHETIC random point cloud (NOT ScanNet/Replica/ScanNet++)",
            "note": "kernel self-check on noise; not a benchmark result",
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="(unused; real eval not implemented)")
    p.add_argument("--root", default="data/scannet", help="(unused; real eval not implemented)")
    p.add_argument("--out", default="results/scannet.json")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run the synthetic kernel self-check instead of exiting.",
    )
    p.add_argument(
        "--manifold",
        action="store_true",
        help="Run the synthetic manifold eval: geodesic vs Euclidean mIoU on a Swiss roll.",
    )
    args = p.parse_args()

    if args.manifold:
        from geosam3d.data import swiss_roll_scene

        result = evaluate_manifold(swiss_roll_scene(seed=0), seeds_per_class=3, k=10, t=0.05, seed=0)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print("SYNTHETIC manifold eval (Swiss roll; NOT a ScanNet benchmark):")
        print(json.dumps(result, indent=2))
        print(
            f"geodesic mIoU {result['geodesic_miou']} vs Euclidean nearest-seed "
            f"{result['euclidean_miou']} over {result['n_classes']} classes"
        )
        return

    if not args.demo:
        print(
            "NO BENCHMARK RUN -- real ScanNet/Replica/ScanNet++ evaluation is not "
            "implemented (no dataset loader, no MonoGS/SAM 2 pipeline).\n"
            "No numbers are emitted. Use --demo for a SYNTHETIC kernel self-check, or "
            "--manifold for the geodesic-vs-Euclidean manifold eval (neither is a benchmark result)."
        )
        return

    result = _synthetic_demo()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("SYNTHETIC demo (random data, NOT a benchmark):")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
