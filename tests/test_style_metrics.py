"""画风深度特征指标（Gram / AdaIN / LPIPS / CSD）回归用例。

分两层：
- **快测**（默认跑）：数学口径、预处理形状、设备解析、容差表、CLI 契约。
  不加载 574MB VGG-19 与 2.4GB CSD 权重。
- **慢测**（``-m slow`` 或 ``STYLE_METRICS_SLOW=1``）：真实前向与数值验证，
  需要权重已就位。默认跳过，避免把 CI 拖成几分钟。
"""

from __future__ import annotations

import json
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_metrics import adain as adain_mod  # noqa: E402
from utils.style_metrics import devices, gram as gram_mod, imaging, inventory, runner  # noqa: E402
from utils.style_metrics.config import VGG19_ADAIN_LAYERS, VGG19_GRAM_LAYERS, WEIGHTS  # noqa: E402

SLOW = os.environ.get("STYLE_METRICS_SLOW", "").strip().lower() in ("1", "true", "yes", "on")
slow = pytest.mark.skipif(not SLOW, reason="需要真实权重的慢测；设 STYLE_METRICS_SLOW=1 开启")


# --------------------------------------------------------------------------- #
# 契约 / 配置
# --------------------------------------------------------------------------- #


def test_metric_kinds_are_declared():
    assert runner.METRIC_KIND == {
        "gram": "distance",
        "adain": "distance",
        "lpips": "distance",
        "csd": "similarity",
    }


def test_tolerance_table_covers_every_metric_and_precision():
    for metric in runner.METRIC_KIND:
        for precision in ("fp32", "fp16"):
            tol = runner.tolerance_for(metric, precision)
            assert tol["rtol"] > 0 and tol["atol"] > 0, (metric, precision)


def test_tolerance_is_declared_before_running():
    """容差必须是常量表，不允许由运行结果反推。"""
    import inspect

    src = inspect.getsource(runner)
    assert "TOLERANCES" in src
    # 容差表不允许出现运行时赋值（只允许字面量字典）
    assert src.index("TOLERANCES: dict") < src.index("def tolerance_for")


def test_vgg_layer_names_match_gram_and_adain_sets():
    for name in set(VGG19_GRAM_LAYERS) | set(VGG19_ADAIN_LAYERS):
        assert name.startswith("relu")


def test_weights_manifest_has_source_and_hash():
    for key, entry in WEIGHTS.items():
        assert entry["sha256"] and len(entry["sha256"]) == 64, key
        assert entry["source"].startswith("http"), key
        assert entry["bytes"] > 0, key


def test_inventory_reports_missing_without_crashing():
    info = inventory.describe_weight("lpips_alex")
    assert "expected_sha256" in info and "source_url" in info


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #


def test_imagenet_normalize_keeps_chw_shape():
    x = torch.rand(3, 8, 8)
    assert imaging.imagenet_normalize(x).shape == (3, 8, 8)


def test_imagenet_normalize_keeps_nchw_shape():
    x = torch.rand(2, 3, 8, 8)
    assert imaging.imagenet_normalize(x).shape == (2, 3, 8, 8)


def test_vgg_tensor_is_nchw_and_normalized():
    imgs = imaging.make_test_images()
    t = imaging.to_imagenet_vgg([imgs["base"], imgs["variant_hard"]], size=64)
    assert t.shape == (2, 3, 64, 64)
    assert t.dtype == torch.float32
    # ImageNet 归一化后不会是 [0,1] 区间内均匀分布
    assert float(t.min()) < 0.0


def test_lpips_tensor_range_is_minus_one_to_one():
    imgs = imaging.make_test_images()
    t = imaging.to_lpips([imgs["base"]], size=32)
    assert t.shape == (1, 3, 32, 32)
    assert float(t.min()) >= -1.0 - 1e-6 and float(t.max()) <= 1.0 + 1e-6


def test_test_images_are_deterministic_and_distinct():
    a = imaging.make_test_images()
    b = imaging.make_test_images()
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    base = np.asarray(a["base"], dtype=np.int16)
    hard = np.asarray(a["variant_hard"], dtype=np.int16)
    soft = np.asarray(a["variant_soft"], dtype=np.int16)
    assert np.abs(base - hard).mean() > 10, "hard 变体必须与基准明显不同"
    assert np.abs(base - soft).mean() > 3, "soft 变体必须与基准可测地不同"


def test_png_round_trip_is_identical():
    imgs = imaging.make_test_images()
    assert np.array_equal(np.asarray(imgs["base"]), np.asarray(imgs["base_png"]))


# --------------------------------------------------------------------------- #
# Gram / AdaIN 数学口径
# --------------------------------------------------------------------------- #


def test_gram_matrix_is_symmetric_and_scaled_by_chw():
    feat = torch.rand(2, 4, 6, 5)
    g = gram_mod.gram_matrix(feat, normalize=True)
    assert g.shape == (2, 4, 4)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6)
    raw = gram_mod.gram_matrix(feat, normalize=False)
    assert torch.allclose(g * (4 * 6 * 5), raw, atol=1e-5)


def test_gram_distance_is_zero_for_identical_features():
    feats = {n: torch.rand(1, 8, 4, 4) for n in VGG19_GRAM_LAYERS}
    res = gram_mod.gram_distance(feats, dict(feats))
    assert res["distance"] == pytest.approx(0.0, abs=1e-12)
    assert res["summary"]["mean_layer_cosine_distance"] == pytest.approx(0.0, abs=1e-9)


def test_gram_distance_grows_with_perturbation():
    feats = {n: torch.rand(1, 8, 4, 4) for n in VGG19_GRAM_LAYERS}
    small = {n: v + 0.01 for n, v in feats.items()}
    big = {n: v * 3.0 for n, v in feats.items()}
    d_small = gram_mod.gram_distance(feats, small)["distance"]
    d_big = gram_mod.gram_distance(feats, big)["distance"]
    assert d_small > 0 and d_big > d_small


def test_gram_distance_rejects_shape_mismatch():
    a = {n: torch.rand(1, 8, 4, 4) for n in VGG19_GRAM_LAYERS}
    b = {n: torch.rand(1, 8, 3, 3) for n in VGG19_GRAM_LAYERS}
    with pytest.raises(ValueError):
        gram_mod.gram_distance(a, b)


def test_adain_stats_are_channel_wise():
    feat = torch.zeros(1, 3, 5, 7)
    feat[0, 0] = 2.0
    feat[0, 1] = -2.0
    mu, sigma = adain_mod.feature_stats(feat)
    assert mu.shape == (1, 3) and sigma.shape == (1, 3)
    assert float(mu[0, 0]) == pytest.approx(2.0)
    assert float(mu[0, 2]) == pytest.approx(0.0)
    assert float(sigma[0, 2]) == pytest.approx(0.0)


def test_adain_distance_zero_for_identical_and_positive_for_shifted():
    feats = {n: torch.rand(1, 6, 4, 4) for n in VGG19_ADAIN_LAYERS}
    assert adain_mod.adain_distance(feats, dict(feats))["distance"] == pytest.approx(0.0, abs=1e-12)
    shifted = {n: v + 1.0 for n, v in feats.items()}
    res = adain_mod.adain_distance(feats, shifted)
    assert res["distance"] > 0
    # 平移只改均值，不该动标准差
    for layer in res["layers"].values():
        assert layer["std_l2"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 容差判定
# --------------------------------------------------------------------------- #


def test_within_tolerance_uses_declared_rule():
    tol = {"rtol": 1e-3, "atol": 1e-6}
    ok, limit = runner.within_tolerance(1.0005, 1.0, tol)
    assert ok and limit == pytest.approx(1e-3 + 1e-6)
    ok2, _ = runner.within_tolerance(1.01, 1.0, tol)
    assert not ok2


def test_fp16_tolerance_is_looser_than_fp32():
    for metric in runner.METRIC_KIND:
        strict = runner.tolerance_for(metric, "fp32")
        loose = runner.tolerance_for(metric, "fp16")
        assert loose["rtol"] >= strict["rtol"] and loose["atol"] >= strict["atol"]


# --------------------------------------------------------------------------- #
# 设备解析
# --------------------------------------------------------------------------- #


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        devices.resolve_device("tpu")


def test_cpu_device_is_always_available():
    backend = devices.resolve_device("cpu")
    assert backend.actual == "cpu" and backend.backend == "torch-cpu"
    assert backend.precision == "fp32"


def test_explicit_npu_failure_is_reported_not_silently_fallen_back(monkeypatch):
    monkeypatch.setattr(devices, "npu_probe", lambda *a, **k: {"available": False, "reason": "模拟不可用"})
    with pytest.raises(devices.DeviceUnavailableError) as exc:
        devices.resolve_device("npu")
    assert "模拟不可用" in str(exc.value)


def test_explicit_cuda_failure_is_reported(monkeypatch):
    monkeypatch.setattr(devices, "cuda_available", lambda: {"available": False, "reason": "无显卡"})
    with pytest.raises(devices.DeviceUnavailableError):
        devices.resolve_device("cuda")


def test_auto_prefers_cuda_when_probe_passes(monkeypatch):
    monkeypatch.setattr(
        devices, "cuda_available", lambda: {"available": True, "device_name": "FAKE-GPU"}
    )
    backend = devices.resolve_device("auto")
    assert backend.actual == "cuda" and backend.requested == "auto"
    assert "cuda" in backend.note


def test_auto_falls_back_to_cpu_and_says_so(monkeypatch):
    monkeypatch.setattr(devices, "cuda_available", lambda: {"available": False, "reason": "x"})
    monkeypatch.setattr(devices, "npu_probe", lambda *a, **k: {"available": False, "reason": "y"})
    backend = devices.resolve_device("auto")
    assert backend.actual == "cpu"
    assert "回落" in backend.note


def test_npu_probe_listing_is_not_treated_as_success():
    """设备列表里有 NPU 不等于可用：探针必须真的跑过一次前向。"""
    info = devices.npu_probe(use_cache=True)
    if info.get("available"):
        assert info.get("execution_devices"), "声称可用却没给出运行时执行设备证据"
        assert info.get("probe_output_shape")
    else:
        assert info.get("reason"), "不可用时必须给出原因"


def test_as_list_does_not_split_strings():
    assert devices._as_list("NPU") == ["NPU"]
    assert devices._as_list(["NPU", "CPU"]) == ["NPU", "CPU"]
    assert devices._as_list(None) == []


# --------------------------------------------------------------------------- #
# CSD 契约
# --------------------------------------------------------------------------- #


def test_csd_missing_weights_raises_actionable_error():
    """权重缺失要抛带来源/哈希/修复命令的 WeightsMissing，而不是一句干巴巴的 IO 错误。"""
    from utils.style_metrics import csd_metric, inventory

    with pytest.raises(inventory.WeightsMissing) as exc:
        csd_metric.load_csd_state_dict(Path("does-not-exist.bin"))
    assert exc.value.key == "csd"
    assert "huggingface.co" in exc.value.source_url
    assert "-m utils.style_metrics.fetch" in exc.value.remedy()
    # 保持向后兼容：老写法 except FileNotFoundError 仍能接住
    assert isinstance(exc.value, FileNotFoundError)


def test_csd_unavailable_exception_type_still_exists():
    from utils.style_metrics import csd_metric

    assert issubclass(csd_metric.CSDUnavailable, RuntimeError)


def test_csd_reports_unavailable_instead_of_faking_with_clip():
    from utils.style_metrics import runner as runner_mod

    outcome = runner_mod.run_csd(None, None, "not-a-backend")
    assert outcome.metric == "csd"
    assert outcome.status in ("unavailable", "error")
    assert outcome.value is None


def test_csd_npu_variant_uses_official_style_head_only():
    """NPU 变体必须挂官方 style 头，且明确声明 fp16 是独立版本。"""
    from utils.style_metrics import csd_metric

    assert hasattr(csd_metric, "NPUCSDMetric")
    src = inspect.getsource(csd_metric.NPUCSDMetric)
    assert "export_csd_onnx" in src
    assert "style_descriptor" in inspect.getsource(csd_metric.CSDCLIP)
    assert "不等价" in src, "NPU 变体必须公开声明精度不等价"


def test_lpips_npu_variant_is_declared_as_separate_precision():
    from utils.style_metrics import lpips_metric

    assert hasattr(lpips_metric, "NPULPIPSMetric")
    src = inspect.getsource(lpips_metric.NPULPIPSMetric)
    assert "不等价" in src
    # 必须来自官方实现，不许换成别的骨干
    assert "export_lpips_onnx" in src


# --------------------------------------------------------------------------- #
# CLI 契约
# --------------------------------------------------------------------------- #


def test_cli_inventory_runs_and_lists_weights():
    tool = PROJECT_ROOT / "tools" / "style_metrics_verify.py"
    assert tool.exists()
    proc = subprocess.run(
        [sys.executable, str(tool), "--inventory"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert "weights" in payload and "environment" in payload
    assert "vgg19" in payload["weights"]


def test_cli_rejects_unknown_metric():
    tool = PROJECT_ROOT / "tools" / "style_metrics_verify.py"
    proc = subprocess.run(
        [sys.executable, str(tool), "--metrics", "nope", "--device", "cpu"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert proc.returncode == 2


def test_cli_does_not_add_root_scripts():
    """根目录只允许启动入口（PROJECT_REQUIREMENTS §1）。"""
    allowed = {"app.py", "make-pic.py", "sd-make-pic.py", "publish_server.py"}
    root_py = {p.name for p in PROJECT_ROOT.glob("*.py")}
    assert root_py <= allowed, f"根目录出现未授权脚本: {sorted(root_py - allowed)}"


# --------------------------------------------------------------------------- #
# 慢测：真实前向
# --------------------------------------------------------------------------- #


@slow
def test_vgg19_torch_features_have_expected_shapes():
    backend = devices.resolve_device("cpu")
    feats = runner.vgg_features_torch([imaging.make_test_images()["base"]], backend)
    assert feats["relu1_1"].shape == (1, 64, 512, 512)
    assert feats["relu5_1"].shape == (1, 512, 32, 32)
    for v in feats.values():
        assert torch.isfinite(v).all()


@slow
def test_gram_and_adain_near_zero_on_same_image():
    backend = devices.resolve_device("cpu")
    imgs = imaging.make_test_images()
    fa = runner.vgg_features_torch([imgs["base"]], backend)
    fb = runner.vgg_features_torch([imgs["base_png"]], backend)
    assert runner.run_gram(fa, fb).value == pytest.approx(0.0, abs=1e-9)
    assert runner.run_adain(fa, fb).value == pytest.approx(0.0, abs=1e-9)


@slow
def test_gram_and_adain_separate_different_images():
    backend = devices.resolve_device("cpu")
    imgs = imaging.make_test_images()
    fa = runner.vgg_features_torch([imgs["base"]], backend)
    fh = runner.vgg_features_torch([imgs["variant_hard"]], backend)
    assert runner.run_gram(fa, fh).value > 0
    assert runner.run_adain(fa, fh).value > 0


@slow
def test_lpips_uses_calibrated_weights_and_is_zero_on_same_image():
    backend = devices.resolve_device("cpu")
    imgs = imaging.make_test_images()
    same = runner.run_lpips(imgs["base"], imgs["base_png"], backend)
    hard = runner.run_lpips(imgs["base"], imgs["variant_hard"], backend)
    assert same.value == pytest.approx(0.0, abs=1e-9)
    assert hard.value > 0.01


@slow
def test_csd_similarity_near_one_on_same_image():
    from utils.style_metrics import csd_metric

    if not csd_metric.DEFAULT_CSD_WEIGHTS.exists():
        pytest.skip("CSD 官方权重未就位")
    backend = devices.resolve_device("cpu")
    imgs = imaging.make_test_images()
    metric = csd_metric.CSDMetric(device="cpu")
    same, _, _ = metric.cosine(imgs["base"], imgs["base_png"])
    hard, _, _ = metric.cosine(imgs["base"], imgs["variant_hard"])
    assert same == pytest.approx(1.0, abs=1e-4)
    assert hard < same
