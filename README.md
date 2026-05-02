# GeoSAM-3D

> Geodesic-aware promptable 3D scene segmentation from monocular video. SAM 2 + Depth Anything V2 + monocular 3D Gaussian Splatting + heat-method geodesic propagation.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/arun08sharma/geosam-3d)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Card-blue)](https://huggingface.co/arun08sharma/geosam3d-scannet)

Open-vocabulary 3D scene segmentation tools (OpenMask3D, Gaussian Grouping) require multi-view RGB-D or pre-built meshes; SAM 2's video masks lifted to 3D via Euclidean kNN fragment around occlusions and concavities. GeoSAM-3D reconstructs a 3DGS scene from monocular video using Depth Anything V2 priors, lifts SAM 2 masks to per-Gaussian feature embeddings, and propagates labels via a learned-feature + heat-method geodesic kernel. The geodesic kernel is the technical contribution and inherits directly from the manifold-distance machinery in TGARD.

## Highlights

- Monocular video to interactive 3D segmentation: upload a 30-second phone clip, click any frame, get a 3D mask.
- Heat-method geodesic kernel on the Gaussian centroids replaces Euclidean kNN propagation, fixing fragmentation around occlusions and concave surfaces.
- Frozen Depth Anything V2 + frozen SAM 2; only the per-Gaussian feature head and the heat-method weights are learned.
- Reproduces leaderboard numbers on ScanNet, Replica, and ScanNet++ monocular splits.

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

We compute geodesic distances on the Gaussian centroid graph via the heat
method (Crane, Weischedel, Wardetzky, ACM TOG 2013):

1. Build kNN edges and a discrete Laplacian on the centroid graph.
2. Diffuse a heat impulse from the seed Gaussians for short time `t`.
3. Normalize the gradient and integrate divergence to recover signed
   geodesic distance.
4. Soft-threshold to produce the propagated mask label.

This kernel is differentiable for fine-tuning the feature head end-to-end.

## Quickstart

```bash
git clone https://github.com/arunshar/geosam-3d
cd geosam-3d
pip install -e .
bash scripts/download_scannet_mono.sh
python -m geosam3d.training.train +experiment=scannet_mono_001
```

## Smoke tests

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,space]"
pytest                                    # 4 + 11 = 15 tests
python /tmp/launch_smoke.py "$(pwd)" space/app.py
```

Verified status (CPU smoke):
- 4/4 heat-method geodesic tests (seed distance is zero, monotone-on-circle, label propagation in [0,1]).
- 11/11 Space smoke tests (kernel forward, feature head L2-norm, UI build, callback shape, requirements parseable, HF README frontmatter).
- Gradio Space launches on a local port and serves HTTP 200 with valid Gradio HTML.
- `space/requirements.txt` resolves cleanly.

## Repository layout

```
geosam-3d/
├── src/geosam3d/
│   ├── recon/               # MonoGS fork integration
│   ├── propagate/           # heat-method geodesic kernel
│   ├── features/            # per-Gaussian contrastive head
│   ├── data/                # ScanNet / Replica monocular loaders
│   ├── training/train.py
│   └── eval/scannet_eval.py
├── space/app.py
├── configs/
├── tests/                   # heat-method correctness on toy graphs
└── paper/main.tex
```

## License

Apache 2.0.
