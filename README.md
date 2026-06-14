# GeoSAM-3D

> Geodesic-aware promptable 3D scene segmentation from monocular video. SAM 2 + Depth Anything V2 + monocular 3D Gaussian Splatting + heat-method geodesic propagation.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/Arun0808/geosam-3d)
![Status: research scaffold](https://img.shields.io/badge/status-research%20scaffold-orange)

> Status: work in preparation. The differentiable heat-method geodesic kernel and the per-Gaussian feature head are implemented and unit-tested. Reconstruction (MonoGS), SAM 2 mask projection, the dataset loaders, and the benchmarks are NOT implemented yet (see "Implementation status" below). There are no trained checkpoints and no benchmark results.

Open-vocabulary 3D scene segmentation tools (OpenMask3D, Gaussian Grouping) require multi-view RGB-D or pre-built meshes; SAM 2's video masks lifted to 3D via Euclidean kNN fragment around occlusions and concavities. GeoSAM-3D is a proposed design that would reconstruct a 3DGS scene from monocular video using Depth Anything V2 priors, lift SAM 2 masks to per-Gaussian feature embeddings, and propagate labels via a learned-feature + heat-method geodesic kernel. The geodesic kernel is the technical contribution and inherits directly from the manifold-distance machinery in TGARD; it is the part that exists today.

## Implementation status

| Component | State |
| --- | --- |
| Heat-method geodesic kernel (`propagate/heat_geodesic.py`) | Implemented, unit-tested (4 tests, CPU) |
| Per-Gaussian feature head + contrastive loss (`features/gaussian_head.py`) | Implemented, unit-tested |
| Training loop (`training/train.py`) | Runs as a SYNTHETIC smoke only (random data); no real training |
| Dataset loader (`data/__init__.py`) | SYNTHETIC stand-in only; no ScanNet/Replica/ScanNet++ reading |
| MonoGS reconstruction (`recon/monogs_runner.py`) | Stub; requires an external MonoGS install, untested |
| SAM 2 mask projection (`_project_sam2_masks` in `training/train.py`) | Stub; returns random ids |
| ScanNet / Replica / ScanNet++ benchmarks (`eval/scannet_eval.py`) | Not implemented; emits no numbers without `--demo` |
| Gradio Space (`space/app.py`) | Demo UI; segmentation callback is illustrative |

No leaderboard numbers are claimed or reproduced. `eval/scannet_eval.py --demo`
runs a self-consistency check of the kernel on a random point cloud; those
numbers are synthetic and are not a ScanNet/Replica/ScanNet++ result.

## Design (target pipeline)

The diagram below is the intended end-to-end design. Today only the geodesic
kernel and feature head exist; the reconstruction and SAM 2 stages are stubs.

## Method

```
phone video --[MonoGS + Depth Anything V2]--> 3D Gaussian field
                                                      |
                                                      v
       SAM 2 video masks ---[contrastive head]---> per-Gaussian features
                                                      |
                                                      v
       click prompt -> nearest features + heat-method geodesic propagation -> 3D mask
```

We compute approximate geodesic distances on the Gaussian centroid graph by
heat diffusion. The implementation (`propagate/heat_geodesic.py`) uses the
Varadhan small-time limit `d^2 ~= -4t log u_t` rather than the gradient +
divergence reconstruction of Crane, Weischedel, Wardetzky (ACM TOG 2013):
the divergence step degenerates at seeds on non-mesh kNN graphs, while the
Varadhan form is monotone in `u` for any `t`, which is all label propagation
needs.

1. Build kNN edges and a discrete Laplacian on the centroid graph.
2. Diffuse a heat impulse from the seed Gaussians: solve `(I + tL) u = u_0`.
3. Recover distance via `d = sqrt(-4t log(u / u_max))`.
4. Soft-threshold to produce the propagated mask label.

This kernel is differentiable for fine-tuning the feature head end-to-end,
and is covered by the unit tests in `tests/test_heat_geodesic.py`.

## Quickstart (synthetic smoke)

This runs end to end today on random data, with no datasets, MonoGS, or SAM 2:

```bash
git clone https://github.com/arunshar/geosam-3d
cd geosam-3d
pip install -e .

# Train the feature head for a couple of steps on a SYNTHETIC Gaussian field.
python -m geosam3d.training.train train.steps_per_scene=2

# Kernel self-check on a random point cloud (synthetic, NOT a benchmark).
python -m geosam3d.eval.scannet_eval --demo
```

The real monocular pipeline is not runnable yet: `scripts/download_scannet_mono.sh`,
the ScanNet/Replica/ScanNet++ loaders, MonoGS reconstruction, and SAM 2 mask
projection are not implemented. Running `eval` without `--demo` prints a notice
and emits no numbers.

## Smoke tests

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,space]"
pytest                                    # kernel + feature-head + Space tests
python /tmp/launch_smoke.py "$(pwd)" space/app.py
```

Verified status (CPU smoke):
- 4/4 heat-method geodesic tests pass (seed distance is zero, monotone-on-circle, label propagation in [0,1]).
- Space smoke tests require `gradio` installed; they cover kernel forward, feature head L2-norm, UI build, callback shape, requirements parseable, and HF README frontmatter.
- The synthetic training smoke (`train.steps_per_scene=2`) imports and runs on CPU.
- The Gradio Space UI builds and is illustrative; the segmentation callback is a placeholder, not a real 3D segmenter.

## Repository layout

```
geosam-3d/
├── src/geosam3d/
│   ├── recon/               # MonoGS wrapper (stub; needs external MonoGS)
│   ├── propagate/           # heat-method geodesic kernel (implemented + tested)
│   ├── features/            # per-Gaussian contrastive head (implemented + tested)
│   ├── data/                # SYNTHETIC dataset stand-in (no real ScanNet loader yet)
│   ├── training/train.py    # synthetic smoke loop
│   └── eval/scannet_eval.py # no real benchmark; --demo runs a synthetic self-check
├── space/app.py
├── configs/
├── tests/                   # heat-method correctness on toy graphs
└── paper/main.tex
```

## License

Apache 2.0.
