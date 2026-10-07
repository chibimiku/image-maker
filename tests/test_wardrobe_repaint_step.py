# -*- coding: utf-8 -*-
"""「衣装重绘」= gpt-image 流水线里最后一道 Gemini 重绘（用户 2026-10-07 要求）。

用户原话与对比证据：
- 要求：*「gpt 生图最后加一个 prompts 的重绘：重绘图中角色的全部衣装，解决细节散乱的问题。
  保持设定逻辑不变。」*
- 证据：`data/20261007/Gemini_Generated_Image_chxafschxafschxa.jpg`（用户手动画完，更好）
  vs `...-final-tone+ink.png`（app 直出 + 「重绘提线」保守固件 + 本地色调/加墨）。
  两者全局亮度/饱和/清晰度接近（实测 151/84.9/234 vs 153/88.2/328），差在**衣装结构**：
  v5 固件的口径是"别动、只修线"，它不会主动把蕾丝/系带/靴袜画清楚。

本文件把三件事钉住：
1. 这一步（`wardrobe`）排在「局部重绘」之后、本地「色调校准/加墨」之前，且**用专用固件**；
2. 真正发出去的提示词里有用户那句内容 + 设定锁定条款，且**不是** v5 保守固件；
3. Tab 上的勾选框常驻可见、默认勾上、状态会被保存。
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_generation.gpt_image2_tab as gpt_image2_tab
from modules.image_generation.gpt_image2_tab import GptImage2Widget
from utils import analysis_gen, post_process
from utils.post_process import (
    WARDROBE_REPAIR_FIRMWARE,
    final_product_name,
    resolve_firmware_text,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_SENTENCE = "重绘图中角色的全部衣装，解决细节散乱的问题。保持设定逻辑不变。"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── 1. 步骤定义 ─────────────────────────────────────────────────────────────

def test_pipeline_flags_expose_the_wardrobe_step():
    steps = analysis_gen.pipeline_steps_from_flags(wardrobe=True, tone=True, ink=True)
    cfg = steps["wardrobe"]
    assert cfg["enabled"] is True
    assert cfg["resolution"] == "2K"
    assert cfg["reference_mode"] == "style"      # 与用户手动操作一致：手画图 + 画风参考图
    # 固件必须是**可读到的绝对路径**：给出文件名时 resolve_firmware_text 找不到文件，
    # 会把文件名当提示词发出去（= 裸重绘，衣装口径全丢）
    assert os.path.isabs(cfg["firmware"]) and os.path.isfile(cfg["firmware"])
    assert os.path.basename(cfg["firmware"]) == WARDROBE_REPAIR_FIRMWARE
    assert len(resolve_firmware_text(cfg["firmware"])) > 500


def test_wardrobe_step_is_off_when_not_requested():
    assert analysis_gen.pipeline_steps_from_flags(wardrobe=False)["wardrobe"]["enabled"] is False
    assert analysis_gen.pipeline_steps_from_flags()["wardrobe"]["enabled"] is False


def test_run_gpt_image_pipeline_counts_the_wardrobe_step_as_work(tmp_path, monkeypatch):
    """只勾「衣装重绘」也必须真的跑流水线（不能因为旧判据只看前 5 个键就直接返回原图）。"""
    calls = {}

    def fake_run_pipeline(paths, steps, **kwargs):
        calls["steps"] = steps
        return ["out.png"]

    monkeypatch.setattr(post_process, "run_pipeline", fake_run_pipeline)
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    steps = analysis_gen.pipeline_steps_from_flags(wardrobe=True)
    out = analysis_gen.run_gpt_image_pipeline([str(src)], steps, log_callback=lambda *_a: None)
    assert out == ["out.png"]
    assert calls["steps"]["wardrobe"]["enabled"] is True


# ── 2. 执行顺序与真正发出去的提示词 ─────────────────────────────────────────

def test_wardrobe_repaint_runs_after_local_and_before_tone_ink(tmp_path, monkeypatch):
    """顺序：重绘 → 结构线 → 局部重绘 → **衣装重绘** → 色调校准 → 加墨。"""
    order = []

    def fake_assemble(current, cfg, **kwargs):
        order.append(("assemble", os.path.basename(str(current)), kwargs.get("firmware")))
        return {"prompt": "P", "source_paths": [current], "extra_reference_paths": []}

    def fake_dispatch(request, **kwargs):
        out = os.path.join(order_dir["dir"], f"step-{len(order)}.png")
        with open(out, "wb") as handle:
            handle.write(b"png")
        return [out]

    order_dir = {"dir": str(tmp_path / "work")}
    os.makedirs(order_dir["dir"], exist_ok=True)
    monkeypatch.setattr(post_process, "assemble_repaint_request", fake_assemble)
    monkeypatch.setattr(post_process, "dispatch_repaint_request", fake_dispatch)
    monkeypatch.setattr(post_process, "log_repaint_call", lambda *a, **k: None)

    src = tmp_path / "base.png"
    src.write_bytes(b"png")
    final_dir = tmp_path / "final"
    steps = analysis_gen.pipeline_steps_from_flags(structure=True, local=True, wardrobe=True,
                                                   tone=True, ink=True)
    steps["local"]["enabled"] = False      # 局部重绘要裁切，这里只验证顺序
    steps["structure"]["enabled"] = False
    steps["repaint"]["enabled"] = True
    post_process.run_pipeline([str(src)], steps, firmware="V5-FIRMWARE", final_dir=str(final_dir),
                              log_callback=lambda *_a: None, work_dir=str(tmp_path / "steps"))

    keys = [row[0] for row in order]
    assert keys.count("assemble") == 2                     # 重绘 + 衣装重绘
    # 第一次是「重绘提线」（用通用固件），第二次是「衣装重绘」（用自己的专用固件）
    assert order[0][2] == "V5-FIRMWARE"
    assert os.path.basename(str(order[1][2])) == WARDROBE_REPAIR_FIRMWARE


def test_local_repaint_is_out_of_the_pipeline_order(tmp_path, monkeypatch):
    """局部重绘**已停用**（用户 2026-10-07）：即使旧配置把它打开，调度顺序里也不许出现它。"""
    order = []

    def fake_assemble(current, cfg, **kwargs):
        order.append(("assemble", kwargs.get("firmware")))
        return {"prompt": "P", "source_paths": [current], "extra_reference_paths": []}

    def fake_dispatch(request, **kwargs):
        out = os.path.join(str(tmp_path), f"step-{len(order)}.png")
        with open(out, "wb") as handle:
            handle.write(b"png")
        return [out]

    def fake_local(image_path, out_path, **kwargs):  # pragma: no cover - 不许被调用
        order.append(("local", None))
        raise AssertionError("已停用的局部重绘被调用了")

    monkeypatch.setattr(post_process, "assemble_repaint_request", fake_assemble)
    monkeypatch.setattr(post_process, "dispatch_repaint_request", fake_dispatch)
    monkeypatch.setattr(post_process, "log_repaint_call", lambda *a, **k: None)
    monkeypatch.setattr(post_process, "local_repaint_composite", fake_local)

    src = tmp_path / "base.png"
    src.write_bytes(b"png")
    steps = post_process.default_pipeline()
    steps["repaint"]["enabled"] = True
    steps["wardrobe"]["enabled"] = True
    steps["tone"]["enabled"] = True
    steps["ink"]["enabled"] = True
    steps["local"].update({"enabled": True, "regions": ["shoes_zoom", "waist"],
                           "region": "shoes_zoom"})
    post_process.run_pipeline([str(src)], steps, final_dir=str(tmp_path / "final"),
                              log_callback=lambda *_a: None, work_dir=str(tmp_path / "steps"))

    keys = [row[0] for row in order]
    assert "local" not in keys                      # 停用工序没有执行
    # 联网工序顺序：重绘 → 衣装重绘（本地色调/加墨不经过 assemble）
    assert keys.count("assemble") == 2
    assert os.path.basename(str(order[1][1])) == WARDROBE_REPAIR_FIRMWARE


def test_wardrobe_prompt_carries_the_user_wording_and_the_setting_lock(tmp_path, monkeypatch):
    """真正组装出来的提示词：有那句内容 + 锁定条款；**不是** v5 保守固件。"""
    captured = {}

    def fake_dispatch(request, **kwargs):
        captured["prompt"] = request.get("prompt") or ""
        captured["refs"] = list(request.get("extra_reference_paths") or [])
        out = tmp_path / "final-wrd.png"
        out.write_bytes(b"png")
        return [str(out)]

    monkeypatch.setattr(post_process, "dispatch_repaint_request", fake_dispatch)
    monkeypatch.setattr(post_process, "log_repaint_call", lambda *a, **k: None)

    src = tmp_path / "base.png"
    src.write_bytes(b"png")
    style_ref = tmp_path / "style.png"
    style_ref.write_bytes(b"png")
    steps = analysis_gen.pipeline_steps_from_flags(wardrobe=True, tone=False, ink=False)
    post_process.run_pipeline([str(src)], steps, final_dir=str(tmp_path / "final"),
                              style_ref_path=str(style_ref), log_callback=lambda *_a: None)

    prompt = captured["prompt"]
    assert USER_SENTENCE in prompt                       # 用户要求的那句，原样在
    assert "KEEP" in prompt and "FORBIDDEN" in prompt    # 设定逻辑锁 + 禁止项
    assert "蕾丝" in prompt and "靴" in prompt            # 明确点名的衣装部位
    v5 = resolve_firmware_text("repaint-system-conservative-v5.md")
    assert prompt != v5
    assert "CONSERVATIVE" not in prompt.upper() or "保守" not in prompt
    assert str(style_ref) in captured["refs"]            # 画风图仍是第二张参考


def test_final_product_name_marks_the_wardrobe_step():
    steps = analysis_gen.pipeline_steps_from_flags(wardrobe=True, tone=True, ink=True)
    name = final_product_name("out_000000", "120000-abcdef", steps)
    assert name.endswith("-final-wrd+tone+ink.png")
    steps = analysis_gen.pipeline_steps_from_flags(repaint=True, wardrobe=True, tone=True)
    assert final_product_name("o", "r", steps).endswith("-final-rp+wrd+tone.png")


# ── 3. Tab 勾选框 ───────────────────────────────────────────────────────────

def _make_tab(tmp_path, monkeypatch):
    config_path = tmp_path / "config-image.json"
    config_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(gpt_image2_tab, "CONFIG_IMAGE_FILE", str(config_path))
    return GptImage2Widget(), config_path


def test_wardrobe_switch_is_visible_and_on_by_default(qapp, tmp_path, monkeypatch):
    tab, _config = _make_tab(tmp_path, monkeypatch)
    check = tab.post_wardrobe_check
    assert check.parent() is tab.post_switch_row      # 常驻可见，不在折叠区
    assert not check.isHidden()
    assert check.isChecked() is True                  # 用户要求默认带上
    assert "衣装" in check.toolTip() and "散乱" in check.toolTip()
    tab.close()


def test_wardrobe_switch_is_wired_into_the_tab_pipeline(qapp, tmp_path, monkeypatch):
    tab, _config = _make_tab(tmp_path, monkeypatch)
    assert tab.post_pipeline_steps()["wardrobe"]["enabled"] is True
    # 勾选框 → 流水线步骤：关掉就不排这一步，勾回来就排
    tab.post_wardrobe_check.setChecked(False)
    assert tab.post_pipeline_steps()["wardrobe"]["enabled"] is False
    tab.post_wardrobe_check.setChecked(True)
    steps = tab.post_pipeline_steps()
    assert steps["wardrobe"]["enabled"] is True
    assert os.path.isfile(steps["wardrobe"]["firmware"])     # 专用固件能真读到
    tab.close()


def test_wardrobe_choice_is_remembered(qapp, tmp_path, monkeypatch):
    tab, config_path = _make_tab(tmp_path, monkeypatch)
    tab.post_wardrobe_check.setChecked(False)
    tab.save_defaults()
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    node = saved[gpt_image2_tab.CONFIG_NODE]["post_process"]
    assert node["wardrobe_enabled"] is False
    tab.close()

    # 同一个配置文件再开一个 Tab：刚才的「关」要被读回来
    tab2 = GptImage2Widget()
    try:
        assert tab2.post_wardrobe_check.isChecked() is False
    finally:
        tab2.close()


def test_reset_restores_the_wardrobe_default(qapp, tmp_path, monkeypatch):
    tab, _config = _make_tab(tmp_path, monkeypatch)
    tab.post_wardrobe_check.setChecked(False)
    tab.reset_post_defaults()
    assert tab.post_wardrobe_check.isChecked() is True
    tab.close()
