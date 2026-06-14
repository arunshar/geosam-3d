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
    args = p.parse_args()

    if not args.demo:
        print(
            "NO BENCHMARK RUN -- real ScanNet/Replica/ScanNet++ evaluation is not "
            "implemented (no dataset loader, no MonoGS/SAM 2 pipeline).\n"
            "No numbers are emitted. Use --demo to run a SYNTHETIC kernel self-check "
            "on random data (those numbers are NOT a benchmark result)."
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
