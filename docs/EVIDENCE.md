# geosam-3d — reproduced evidence

_Generated 2026-06-29T10:30:32Z by running the test suite on the real/tested code in this repo._

These are **reproduced** results: the code runs and every assertion below holds. Benchmark/leaderboard numbers in the paper (PSNR, mIoU, speedups) remain **targets, not reproduced**, and are labeled as such throughout.

## Test suite (`pytest -v`)

```
tests/test_heat_geodesic.py::test_seed_distance_is_zero PASSED           [  4%]
tests/test_heat_geodesic.py::test_distances_are_nonneg PASSED            [  9%]
tests/test_heat_geodesic.py::test_geodesic_on_circle_is_monotone PASSED  [ 13%]
tests/test_heat_geodesic.py::test_propagate_label_is_in_unit_interval PASSED [ 18%]
tests/test_manifold_eval.py::test_swiss_roll_scene_shapes_and_labels PASSED [ 22%]
tests/test_manifold_eval.py::test_multiclass_miou_bounds PASSED          [ 27%]
tests/test_manifold_eval.py::test_euclidean_nearest_seed_recovers_seed_labels PASSED [ 31%]
tests/test_manifold_eval.py::test_geodesic_beats_euclidean_on_manifold PASSED [ 36%]
tests/test_smoke.py::test_top_level_imports PASSED                       [ 40%]
tests/test_smoke.py::test_propagate_imports PASSED                       [ 45%]
tests/test_smoke.py::test_features_imports PASSED                        [ 50%]
tests/test_smoke.py::test_geodesic_kernel_forward PASSED                 [ 54%]
tests/test_smoke.py::test_feature_head_l2_normalized_e2e PASSED          [ 59%]
tests/test_smoke.py::test_propagate_label_pipeline PASSED                [ 63%]
tests/test_smoke.py::test_space_app_importable PASSED                    [ 68%]
tests/test_smoke.py::test_space_ui_builds PASSED                         [ 72%]
tests/test_smoke.py::test_space_callback_returns_two_values PASSED       [ 77%]
tests/test_smoke.py::test_space_requirements_parseable PASSED            [ 81%]
tests/test_smoke.py::test_space_readme_has_hf_frontmatter PASSED         [ 86%]
tests/test_synthetic.py::test_synthetic_dataset_shapes PASSED            [ 90%]
tests/test_synthetic.py::test_gather_attrs_matches_in_dim SKIPPED (could
tests/test_synthetic.py::test_eval_demo_is_synthetic_and_runs PASSED     [100%]

======================== 21 passed, 1 skipped in 1.54s =========================
```

## Reproduced demo (headline number)

On a sampled unit circle, geodesic distance from a seed to the antipode is 2.35 vs the Euclidean chord 2.0, the label propagates along the surface, not by straight-line proximity.
