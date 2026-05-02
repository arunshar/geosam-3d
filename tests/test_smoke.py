"""Smoke tests for HF Space deployment of geosam-3d."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT / "space" / "app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("geosam3d_space_app", SPACE_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["geosam3d_space_app"] = module
    spec.loader.exec_module(module)
    return module


# -- package imports ---------------------------------------------------------


def test_top_level_imports():
    import geosam3d
    from geosam3d import HeatGeodesicKernel
    assert geosam3d.__version__


def test_propagate_imports():
    from geosam3d.propagate import HeatGeodesicKernel
    from geosam3d.propagate.heat_geodesic import knn_graph, graph_laplacian


def test_features_imports():
    from geosam3d.features import GaussianFeatureHead
    from geosam3d.features.gaussian_head import contrastive_loss


# -- end-to-end heat geodesic + feature head --------------------------------


def test_geodesic_kernel_forward():
    from geosam3d.propagate import HeatGeodesicKernel
    pts = torch.randn(40, 3)
    kernel = HeatGeodesicKernel(k=8, t=0.05)
    seed = torch.zeros(40)
    seed[3] = 1.0
    d = kernel.geodesic(pts, seed)
    assert d.shape == (40,)
    assert torch.isfinite(d).all()
    assert (d >= 0.0).all()


def test_feature_head_l2_normalized_e2e():
    from geosam3d.features import GaussianFeatureHead
    head = GaussianFeatureHead(in_dim=12, hidden=32, out_dim=8, depth=1)
    attrs = torch.randn(20, 12)
    z = head(attrs)
    assert z.shape == (20, 8)
    assert torch.allclose(z.norm(dim=-1), torch.ones(20), atol=1e-4)


def test_propagate_label_pipeline():
    """End-to-end: features + heat-method label propagation."""
    from geosam3d.propagate import HeatGeodesicKernel
    pts = torch.randn(30, 3)
    kernel = HeatGeodesicKernel(k=6, t=0.05)
    seed = torch.zeros(30)
    seed[0] = 1.0
    p = kernel.propagate_label(pts, seed)
    assert p.shape == (30,)
    assert (p >= 0).all() and (p <= 1.0 + 1e-6).all()


# -- Gradio app smoke -------------------------------------------------------


def test_space_app_importable():
    module = _load_app_module()
    assert hasattr(module, "build_ui")
    assert hasattr(module, "segment")


def test_space_ui_builds():
    gr = pytest.importorskip("gradio")
    module = _load_app_module()
    ui = module.build_ui()
    assert isinstance(ui, gr.Blocks)


def test_space_callback_returns_two_values():
    """The segment callback returns (image, stats)."""
    module = _load_app_module()
    out = module.segment(None, 0, "0,0")
    assert isinstance(out, tuple) and len(out) == 2


# -- requirements + readme --------------------------------------------------


def test_space_requirements_parseable():
    req = REPO_ROOT / "space" / "requirements.txt"
    assert req.exists()
    text = req.read_text().lower()
    assert "gradio" in text
    assert "torch" in text


def test_space_readme_has_hf_frontmatter():
    readme = REPO_ROOT / "space" / "README.md"
    assert readme.exists()
    body = readme.read_text()
    assert body.startswith("---\n")
    assert "sdk: gradio" in body
    assert "app_file:" in body
