"""权重缺失 / 路径异常的容错回归用例。

要求（用户指定）：**模型不存在时要容错报错** —— 报清楚、能继续、不冒充成功。

- 不抛裸 `FileNotFoundError`，抛带「官方 URL + 字节数 + SHA-256 + 可直接执行的修复命令」的
  :class:`WeightsMissing`；
- 指标层面把缺失映射成 ``status="unavailable"``，**不是** ``"error"``、**更不是** 0 分；
- CLI 权重缺失时不崩、不写半个 JSON，逐条给出修复命令，退出码非零；
- 缺失的指标不影响其它指标照常跑完。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_metrics import config as cfg  # noqa: E402
from utils.style_metrics import inventory, runner  # noqa: E402


# --------------------------------------------------------------------------- #
# 错误对象本身
# --------------------------------------------------------------------------- #


def test_missing_weight_error_carries_full_remedy():
    exc = inventory.WeightsMissing("vgg19", "文件不存在")
    payload = exc.as_dict()
    assert payload["reason"] == "weights_missing"
    assert payload["key"] == "vgg19"
    assert payload["source_url"].startswith("https://download.pytorch.org/models/vgg19-")
    assert len(payload["expected_sha256"]) == 64
    assert payload["expected_bytes"] == 574673361
    # 修复命令必须可执行：包含 python、-m utils.style_metrics.fetch、URL、目标路径、哈希
    remedy = payload["remedy"]
    assert "-m utils.style_metrics.fetch" in remedy
    assert payload["source_url"] in remedy
    assert str(payload["expected_path"]) in remedy
    assert payload["expected_sha256"] in remedy


def test_missing_weight_message_is_human_readable():
    text = str(inventory.WeightsMissing("csd"))
    assert "缺少画风指标权重 [csd]" in text
    assert "官方来源" in text and "SHA-256" in text and "修复命令" in text


def test_weights_missing_is_a_filenotfounderror():
    """保持向后兼容：老代码 `except FileNotFoundError` 仍然能接住。"""
    assert issubclass(inventory.WeightsMissing, FileNotFoundError)


# --------------------------------------------------------------------------- #
# require_weight / preflight
# --------------------------------------------------------------------------- #


def test_require_weight_returns_path_when_present():
    path = inventory.require_weight("lpips_alex")
    assert path.exists()


def test_require_weight_raises_with_remedy_when_absent(monkeypatch):
    monkeypatch.setitem(
        cfg.WEIGHTS, "vgg19", {**cfg.WEIGHTS["vgg19"], "path": Path("no-such-weight.pth")}
    )
    with pytest.raises(inventory.WeightsMissing) as exc:
        inventory.require_weight("vgg19")
    assert "no-such-weight.pth" in str(exc.value)
    assert exc.value.source_url.startswith("https://")


def test_require_weight_detects_size_mismatch(monkeypatch, tmp_path):
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"too small")
    monkeypatch.setitem(cfg.WEIGHTS, "vgg19", {**cfg.WEIGHTS["vgg19"], "path": fake})
    with pytest.raises(inventory.WeightsMissing) as exc:
        inventory.require_weight("vgg19")
    assert "字节数不符" in str(exc.value)


def test_preflight_maps_metrics_to_their_missing_weights(monkeypatch):
    for key in ("vgg19", "lpips_alex", "lpips_alex_trunk", "csd", "clip_vit_l14"):
        monkeypatch.setitem(
            cfg.WEIGHTS, key, {**cfg.WEIGHTS[key], "path": Path(f"missing-{key}.bin")}
        )
    report = inventory.preflight(("gram", "adain", "lpips", "csd"))
    assert set(report) == {"gram", "adain", "lpips", "csd"}
    assert [p["key"] for p in report["gram"]] == ["vgg19"]
    assert {p["key"] for p in report["lpips"]} == {"lpips_alex", "lpips_alex_trunk"}
    assert {p["key"] for p in report["csd"]} == {"csd", "clip_vit_l14"}
    text = inventory.format_preflight(report)
    assert "-m utils.style_metrics.fetch" in text


def test_preflight_is_empty_when_everything_present():
    assert inventory.preflight(("gram", "adain", "lpips", "csd")) == {}


def test_metric_weights_cover_every_metric():
    for metric in runner.METRIC_KIND:
        assert inventory.METRIC_WEIGHTS.get(metric), metric


# --------------------------------------------------------------------------- #
# 指标层面：缺失 → unavailable，不是 error
# --------------------------------------------------------------------------- #


def test_run_lpips_reports_unavailable_for_missing_weights(monkeypatch):
    monkeypatch.setitem(
        cfg.WEIGHTS, "lpips_alex", {**cfg.WEIGHTS["lpips_alex"], "path": Path("nope.pth")}
    )
    from utils.style_metrics import lpips_metric

    lpips_metric.clear_cache()
    from utils.style_metrics import imaging, devices

    imgs = imaging.make_test_images()
    outcome = runner.run_lpips(imgs["base"], imgs["base_png"], devices.resolve_device("cpu"))
    assert outcome.status == "unavailable"
    assert outcome.value is None
    assert outcome.detail["reason"] == "weights_missing"
    assert outcome.detail["remedy"]


def test_vgg_encoder_raises_weights_missing_not_bare_oserror(monkeypatch):
    monkeypatch.setitem(
        cfg.WEIGHTS, "vgg19", {**cfg.WEIGHTS["vgg19"], "path": Path("nope-vgg.pth")}
    )
    from utils.style_metrics import vgg_encoder

    vgg_encoder.clear_cache()
    with pytest.raises(inventory.WeightsMissing):
        vgg_encoder.get_encoder("cpu")


def test_csd_missing_backbone_is_reported_before_torch_load(monkeypatch):
    """骨架缺失要在进 torch 之前报清楚，而不是让 clip 内部崩。"""
    monkeypatch.setitem(
        cfg.WEIGHTS, "clip_vit_l14", {**cfg.WEIGHTS["clip_vit_l14"], "path": Path("nope.pt")}
    )
    from utils.style_metrics import csd_metric

    with pytest.raises(inventory.WeightsMissing) as exc:
        csd_metric.CSDCLIP()
    assert exc.value.key == "clip_vit_l14"


def test_outcome_from_error_maps_missing_to_unavailable():
    a = runner.outcome_from_error("gram", inventory.WeightsMissing("vgg19"))
    assert a.status == "unavailable" and a.value is None
    b = runner.outcome_from_error("gram", ValueError("boom"))
    assert b.status == "error" and "boom" in b.error


# --------------------------------------------------------------------------- #
# CLI 端到端：权重目录为空时不崩、写 JSON、给修复命令、退出码非零
# --------------------------------------------------------------------------- #


def _run_cli_with_empty_model_root(tmp_path: Path, extra_args: list[str]):
    env = dict(os.environ)
    env["IMAGE_MAKER_STYLE_METRICS_HOME"] = str(tmp_path / "empty-models")
    env["IMAGE_MAKER_STYLE_METRICS_CACHE"] = str(tmp_path / "empty-cache")
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "style_metrics_verify.py"),
            "--device",
            "cpu",
            "--out",
            str(out),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=900,
        env=env,
    )
    return proc, out


def test_cli_gracefully_reports_every_missing_model(tmp_path):
    proc, out = _run_cli_with_empty_model_root(tmp_path, [])
    assert proc.returncode == 1, proc.stderr[-2000:]
    assert out.exists(), "权重缺失时仍应写出机器可读 JSON（如实记录 unavailable）"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["status_counts"].get("unavailable") == 4
    assert payload["summary"]["all_ok"] is False
    for record in payload["results"]:
        assert record["status"] == "unavailable"
        # 关键：不许出现「假数值」
        assert record["validation"].get("note")
        assert not record["validation"].get("self_distance_or_similarity")
        remedy = record["remediation"][0]["remedy"]
        assert "-m utils.style_metrics.fetch" in remedy
    # 终端上也要给出修复命令，且不能是 traceback
    assert "-m utils.style_metrics.fetch" in proc.stdout
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr


def test_cli_inventory_warns_and_exits_nonzero_when_models_absent(tmp_path):
    proc, _ = _run_cli_with_empty_model_root(tmp_path, ["--inventory"])
    assert proc.returncode == 1
    assert "缺失" in proc.stdout or "警告" in proc.stdout
    assert "-m utils.style_metrics.fetch" in proc.stdout


def test_cli_missing_pair_image_is_reported_not_traced(tmp_path):
    proc, out = _run_cli_with_empty_model_root(
        tmp_path, ["--pair", str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    )
    assert proc.returncode == 2
    assert "不存在" in proc.stdout
    assert not out.exists(), "输入图片就错了，不应该写出半份 JSON"


# --------------------------------------------------------------------------- #
# 环境变量覆盖本身
# --------------------------------------------------------------------------- #


def test_model_root_env_override_is_honoured(tmp_path, monkeypatch):
    """模块级常量在导入时求值，这里验证的是「用环境变量指路」这条契约可用。"""
    env = dict(os.environ)
    env["IMAGE_MAKER_STYLE_METRICS_HOME"] = str(tmp_path)
    code = (
        "import os,sys;"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r});"
        "from utils.style_metrics import config;"
        "print(config.MODEL_ROOT)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.stdout.strip() == str(tmp_path)
