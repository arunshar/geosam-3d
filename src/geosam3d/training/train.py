"""End-to-end training loop.

STATUS: runs only on SYNTHETIC data today.

Intended full pipeline (per scene), once the real components exist:
1. Run MonoGS to build a 3D Gaussian field.       <- recon stub, see recon/monogs_runner.py
2. Run SAM 2 on the input video to get per-frame masks.
3. Lift masks to per-Gaussian pseudo-labels by projecting and majority-voting.  <- stub below
4. Train the GaussianFeatureHead via contrastive loss using these pseudo-labels.  <- IMPLEMENTED + TESTED
5. Use HeatGeodesicKernel for label propagation at inference.                     <- IMPLEMENTED + TESTED

What actually runs right now: the dataset is the SYNTHETIC
``ScanNetMonocularDataset`` (random Gaussian fields, no ScanNet, no MonoGS,
no SAM 2). When a clip has no real frames the loop skips reconstruction and
trains the feature head directly on the synthetic Gaussian attributes and
synthetic pseudo-mask ids. This makes the command a runnable smoke test;
it does NOT train a usable model and produces NO benchmark numbers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(Path(__file__).parents[3] / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    logger.warning(
        "SYNTHETIC SMOKE RUN: training on random data from ScanNetMonocularDataset. "
        "No ScanNet/MonoGS/SAM 2 involved; results are not meaningful."
    )

    from geosam3d.data import ScanNetMonocularDataset
    from geosam3d.features import GaussianFeatureHead
    from geosam3d.features.gaussian_head import contrastive_loss
    from geosam3d.recon import MonoGSConfig, MonoGSRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    ds = ScanNetMonocularDataset(root=cfg.dataset.root, split=cfg.dataset.split, clip_length=cfg.dataset.clip_length)
    head = GaussianFeatureHead(
        in_dim=cfg.features.in_dim,
        hidden=cfg.features.hidden,
        out_dim=cfg.features.out_dim,
        depth=cfg.features.depth,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.train.lr)
    runner = MonoGSRunner(MonoGSConfig(iters=cfg.recon.iters))

    for scene_idx in range(len(ds)):
        clip = ds[scene_idx]

        if clip.frames:
            # Real path (NOT exercised by the synthetic dataset): reconstruct
            # with MonoGS, then project SAM 2 masks. Requires MonoGS + SAM 2.
            scene = runner.reconstruct(clip.frames, Path(cfg.train.out_dir) / clip.scene)
            mask_ids = _project_sam2_masks(scene, clip).to(device)
        else:
            # Synthetic path: use the random Gaussian field straight from the
            # dataset; no reconstruction, no SAM 2 projection.
            scene = clip.scene_dict
            mask_ids = scene["mask_ids"].to(device)

        gauss_attrs = _gather_attrs(scene).to(device)
        for step in range(cfg.train.steps_per_scene):
            opt.zero_grad()
            features = head(gauss_attrs)
            loss = contrastive_loss(features, mask_ids)
            loss.backward()
            opt.step()
            if step % cfg.logging.every == 0:
                logger.info("scene=%s step=%d loss=%.4f", clip.scene, step, loss.item())


def _gather_attrs(scene) -> torch.Tensor:
    """Stack per-Gaussian attributes into a (N, in_dim) tensor.

    Layout (in_dim=12): means(3) + colors(3) + scales(3) + opacities(1)
    + quats[:, :2](2). The two leading quaternion components are a
    placeholder feature; the full pose handling is future work.
    """
    return torch.cat([
        scene["means"],
        scene["colors"],
        scene["scales"],
        scene["opacities"].unsqueeze(-1),
        scene["quats"][:, :2],   # placeholder pose feature
    ], dim=-1)


def _project_sam2_masks(scene, clip) -> torch.Tensor:
    """STUB: project SAM 2 masks onto Gaussians.

    NOT IMPLEMENTED. The real version uses the camera intrinsics/extrinsics
    plus nearest-projection to assign each Gaussian the id of the SAM 2 mask
    it falls in. This stub returns random ids and is only reached on the
    (currently unused) real-frame path. The synthetic path bypasses it.
    """
    return torch.randint(0, 16, (scene["means"].shape[0],))


if __name__ == "__main__":
    main()
