"""End-to-end training loop.

For each scene:
1. Run MonoGS to build a 3D Gaussian field.
2. Run SAM 2 on the input video to get per-frame masks.
3. Lift masks to per-Gaussian pseudo-labels by projecting and majority-voting.
4. Train the GaussianFeatureHead via contrastive loss using these pseudo-labels.
5. Use HeatGeodesicKernel for label propagation at inference.
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

    from geosam3d.data import ScanNetMonocularDataset
    from geosam3d.features import GaussianFeatureHead
    from geosam3d.features.gaussian_head import contrastive_loss
    from geosam3d.recon import MonoGSConfig, MonoGSRunner

    ds = ScanNetMonocularDataset(root=cfg.dataset.root, split=cfg.dataset.split, clip_length=cfg.dataset.clip_length)
    head = GaussianFeatureHead(
        in_dim=cfg.features.in_dim,
        hidden=cfg.features.hidden,
        out_dim=cfg.features.out_dim,
        depth=cfg.features.depth,
    ).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=cfg.train.lr)
    runner = MonoGSRunner(MonoGSConfig(iters=cfg.recon.iters))

    for scene_idx in range(len(ds)):
        clip = ds[scene_idx]
        scene = runner.reconstruct(clip.frames, Path(cfg.train.out_dir) / clip.scene)
        gauss_attrs = _gather_attrs(scene).cuda()
        mask_ids = _project_sam2_masks(scene, clip).cuda()  # stub returns dummy ids
        for step in range(cfg.train.steps_per_scene):
            opt.zero_grad()
            features = head(gauss_attrs)
            loss = contrastive_loss(features, mask_ids)
            loss.backward()
            opt.step()
            if step % cfg.logging.every == 0:
                logger.info("scene=%s step=%d loss=%.4f", clip.scene, step, loss.item())


def _gather_attrs(scene) -> torch.Tensor:
    """Stack per-Gaussian attributes (means, colors, scales, opacities) into a (N, in_dim) tensor."""
    return torch.cat([
        scene["means"],
        scene["colors"],
        scene["scales"],
        scene["opacities"].unsqueeze(-1),
        scene["quats"][:, :1],   # filler
    ], dim=-1)


def _project_sam2_masks(scene, clip) -> torch.Tensor:
    """Stub: project SAM 2 masks onto Gaussians. Real impl uses the camera + nearest-projection."""
    return torch.randint(0, 16, (scene["means"].shape[0],))


if __name__ == "__main__":
    main()
