# -*- coding: utf-8 -*-
"""后处理流水线（utils/post_process.py）+ gpt-image-2 Tab 的勾选框联动。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境构造 Qt 控件必须

import cv2
import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils import post_process as pp  # noqa: E402


def _make_image(path, size=(360, 540)):
    h, w = size
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.ellipse(img, (w // 2, h // 5), (70, 90), 0, 0, 360, (190, 175, 165), -1)
    cv2.rectangle(img, (w // 2 - 80, h // 3), (w // 2 + 80, h - 40), (215, 210, 200), -1)
    for x in range(w // 2 - 80, w // 2 + 80, 18):
        cv2.line(img, (x, h // 3), (x + 9, h // 3), (110, 100, 95), 2)
    cv2.imwrite(path, img)
    return img


def test_region_presets_and_parse():
    assert pp.resolve_region_box("hair", 1000, 1500)[3] > 700
    assert pp.resolve_region_box("skirt", 1000, 1500)[1] > 500
    assert pp.resolve_region_box("0.1,0.2,0.3,0.4", 1000, 1000) == (100, 200, 300, 400)
    assert pp.resolve_region_box("10,20,30,40", 1000, 1000) == (10, 20, 30, 40)
    with pytest.raises(ValueError):
        pp.resolve_region_box("nope", 100, 100)


def test_subject_regions_cover_whole_figure(tmp_path):
    """整个人物 / 人物不含面部：自动框应覆盖大部分高度，且比上半身更大。"""
    img = _make_image(tmp_path / "sub.png", size=(420, 640))
    h, w = img.shape[:2]
    subject = pp.resolve_region_box("subject", w, h, image=img)
    no_face = pp.resolve_region_box("subject_no_face", w, h, image=img)
    upper = pp.resolve_region_box("upper", w, h)
    assert subject == no_face
    assert subject[3] > upper[3]                      # 比上半身更高
    assert (subject[3] - subject[1]) > 0.5 * h
    assert (subject[2] - subject[0]) > 0.25 * w
    # 框应包含画面中心（合成图的主体画在中间）
    assert subject[0] < w / 2 < subject[2] and subject[1] < h / 2 < subject[3]
    assert "subject" in pp.REGION_LABELS and "subject_no_face" in pp.REGION_LABELS
    assert "HAIR-STRUCTURE" not in pp.REGION_EMPHASIS.get("subject", "")


def test_subject_box_is_deterministic(tmp_path):
    """GrabCut 不设种子时同一张图会给出不同的人物框（实测三次三个框）→ 必须固定种子。

    框会漂移的话，鞋靴紧框/主体遮罩/局部裁切位置每次都不一样，拼接问题既无法复现也无法比较。
    """
    img = _make_image(tmp_path / "det.png", size=(420, 640))
    boxes = [pp.detect_subject_box(img) for _ in range(3)]
    assert boxes[0] == boxes[1] == boxes[2]
    assert pp.resolve_region_box("subject", 640, 420, image=img) == boxes[0]


def test_shoes_zoom_uses_tight_footwear_box(tmp_path, monkeypatch):
    """「鞋/靴紧框」必须真的用 detect_footwear_box 的紧框，而不是悄悄退回整个人物框。

    这条曾经是坏的：`shoes_zoom` 在 DYNAMIC_REGIONS 里，resolve 却一律返回人物框，
    于是鞋靴 FOCUS 条款被贴到全身裁切上（实测两个区域的 -crop.png 逐字节相同）。
    """
    img = _make_image(tmp_path / "shoe.png", size=(420, 640))
    sentinel = (10, 300, 630, 410)
    monkeypatch.setattr(pp, "detect_footwear_box", lambda image, **kw: sentinel)
    assert pp.resolve_region_box("shoes_zoom", 640, 420, image=img) == sentinel


def test_tight_region_restrict_mask_uses_subject_box(tmp_path, monkeypatch):
    """紧框区域算「只贴回主体」的遮罩时，要用整个人物框（紧框贴边会让 GrabCut 出空遮罩）。"""
    img = _make_image(tmp_path / "tight.png", size=(420, 640))
    seen = {}
    monkeypatch.setattr(pp, "detect_footwear_box", lambda image, **kw: (10, 300, 630, 410))

    def _mask(image, box=None, **kw):
        seen["box"] = box
        return np.full(image.shape[:2], 255, np.uint8)

    monkeypatch.setattr(pp, "subject_mask", _mask)

    def _fake(crop_path, firmware):
        patch = tmp_path / "patch.png"
        cv2.imwrite(str(patch), cv2.imread(crop_path))
        return str(patch)

    pp.local_repaint_composite(str(tmp_path / "tight.png"), str(tmp_path / "out.png"),
                               region="shoes_zoom", repaint_callable=_fake,
                               restrict_to_subject=True, log_callback=lambda m: None)
    assert seen.get("box") is None, "紧框区域必须用整个人物框（box=None → detect_subject_box）"


def test_subject_no_face_excludes_head_band(tmp_path, monkeypatch):
    """人物不含面部：排除框应落在人物框上半部（而不是画面下半部）。"""
    monkeypatch.setattr(pp, "detect_hand_boxes", lambda *a, **k: [])   # 这条只验脸
    img = _make_image(tmp_path / "sub2.png", size=(420, 640))
    h, w = img.shape[:2]
    box = pp.resolve_region_box("subject_no_face", w, h, image=img)
    ex = pp.resolve_exclusion_boxes("subject_no_face", img, box=box)
    assert len(ex) == 1
    fx0, fy0, fx1, fy1 = ex[0]
    assert fy0 < h * 0.25 and fy1 < h * 0.6          # 在头部一带
    assert (fx1 - fx0) < (box[2] - box[0])           # 比人物框窄
    assert pp.resolve_exclusion_boxes("subject", img, box=box) == []


def test_feather_blend_keeps_excluded_region_untouched(tmp_path):
    base = _make_image(tmp_path / "base.png", size=(400, 600))
    patch = np.full_like(base, 30)
    box = (40, 20, 340, 380)
    exclude = (140, 40, 250, 180)
    out = pp.feather_blend(base, patch, box, feather=20, exclude_boxes=[exclude])
    cy, cx = (exclude[1] + exclude[3]) // 2, (exclude[0] + exclude[2]) // 2
    assert out[cy, cx].tolist() == base[cy, cx].tolist()        # 排除区保留原像素
    # 框内、排除区之外的内部点应被 patch 覆盖（留足羽化边距，用容差比较）
    iy, ix = box[1] + 260, box[0] + 60
    assert abs(int(out[iy, ix][0]) - 30) <= 1


def test_emphasis_text_exists_for_key_regions():
    assert "HAIR-STRUCTURE" in pp.REGION_EMPHASIS["hair"]
    assert "FACE" in pp.REGION_EMPHASIS["face"]
    assert "DRESS/FOLD" in pp.REGION_EMPHASIS["skirt"]
    assert set(pp.REGION_EMPHASIS).issubset(set(pp.REGION_PRESETS))


def test_feather_blend_only_touches_box(tmp_path):
    base = _make_image(tmp_path / "base.png")
    patch = np.zeros_like(base)
    patch[:] = (10, 10, 10)
    box = (50, 60, 200, 300)
    out = pp.feather_blend(base, patch, box, feather=20)
    assert out.shape == base.shape
    # box 外完全不变
    assert np.array_equal(out[:40, :40], base[:40, :40])
    # box 中心被 patch 覆盖
    cy, cx = (box[1] + box[3]) // 2, (box[0] + box[2]) // 2
    assert out[cy, cx].tolist() == [10, 10, 10]


def test_structure_overlay_improves_or_keeps_lines(tmp_path):
    sys.path.insert(0, os.path.join(BASE, "tests"))
    import style_render_metrics as m
    src = tmp_path / "src.png"
    _make_image(src)
    before = m.analyze(str(src))
    info = pp.structure_overlay_file(str(src), str(tmp_path / "out.png"), strength=0.5, min_len=40)
    after = m.analyze(str(tmp_path / "out.png"))
    assert info["coverage"] > 0
    assert after["line_avg_len"] >= before["line_avg_len"] or \
        after["line_endpoint_density"] <= before["line_endpoint_density"]


def test_local_repaint_composite_with_injected_patch(tmp_path):
    base = _make_image(tmp_path / "base.png")
    patch = tmp_path / "patch.png"
    h, w = base.shape[:2]
    cv2.imwrite(str(patch), np.full((int(h * 0.52), w, 3), 120, np.uint8))
    out = tmp_path / "out.jpg"
    info = pp.local_repaint_composite(str(tmp_path / "base.png"), str(out), region="hair",
                                     patch_path=str(patch), feather=16, log_callback=lambda m: None)
    assert os.path.isfile(info["out"])
    merged = cv2.imread(str(out))
    assert merged.shape == base.shape


def test_local_repaint_uses_callable_and_emphasis(tmp_path):
    base = _make_image(tmp_path / "base.png")
    seen = {}

    def _fake_repaint(crop_path, firmware):
        seen["crop"] = crop_path
        crop = cv2.imread(crop_path)
        out = tmp_path / "patch.png"
        cv2.imwrite(str(out), crop)
        return str(out)

    out = tmp_path / "out.png"
    info = pp.local_repaint_composite(str(tmp_path / "base.png"), str(out), region="hair",
                                     repaint_callable=_fake_repaint, feather=12,
                                     log_callback=lambda m: None)
    assert os.path.isfile(info["out"])
    assert seen["crop"].endswith("-crop.png")
    assert info["box"][3] > info["box"][1]


def test_run_pipeline_respects_switches(tmp_path):
    img = tmp_path / "img.png"
    _make_image(img)
    # 全部关闭 → 原图不变
    steps = pp.default_pipeline()
    assert pp.run_pipeline([str(img)], steps, final_dir=str(tmp_path / "f0")) == [str(img)]
    # 只开结构线 → 产物是新文件
    steps["structure"]["enabled"] = True
    steps["structure"]["strength"] = 0.5
    out = pp.run_pipeline([str(img)], steps, final_dir=str(tmp_path / "f1"))
    assert len(out) == 1 and out[0] != str(img) and os.path.isfile(out[0])
    assert "-final-sline50" in out[0]


def test_run_pipeline_default_dir_is_isolated_under_pytest(tmp_path, monkeypatch):
    """不传 final_dir 时，pytest 里必须落到 data/test-result/<日期>/ 而不是真实日期目录。

    回归（用户 2026-09-23 反馈）：用例里 `run_pipeline([img], steps)` 没传 final_dir，
    默认落到 `data/<今天>/`，于是日期目录里堆了 90 多个
    `img-233500-a35670-final-sline50.png` 这种废文件。
    """
    img = tmp_path / "img.png"
    _make_image(img)
    steps = pp.default_pipeline()
    steps["structure"]["enabled"] = True
    monkeypatch.setenv("IMAGE_MAKER_TEST_OUTPUT", "1")
    out = pp.run_pipeline([str(img)], steps)
    assert out and out[0] != str(img)
    parts = out[0].replace("\\", "/").split("/")
    assert parts[:2] == ["data", "test-result"], out[0]
    assert os.path.basename(out[0]).startswith("test-")
    assert os.path.isfile(out[0])


def test_local_repaint_subject_no_face_keeps_face_pixels(tmp_path):
    """端到端：人物（不含面部）区域 + 注入的"重绘结果" → 脸区像素必须与原图完全一致。"""
    base_path = tmp_path / "base.png"
    base = _make_image(base_path, size=(420, 640))

    def _solid_repaint(crop_path, firmware):
        crop = cv2.imread(crop_path)
        out = tmp_path / "patch.png"
        cv2.imwrite(str(out), np.full_like(crop, 200))
        return str(out)

    out_path = tmp_path / "out.png"
    info = pp.local_repaint_composite(str(base_path), str(out_path), region="subject_no_face",
                                     repaint_callable=_solid_repaint, feather=24,
                                     log_callback=lambda m: None)
    merged = cv2.imread(str(out_path))
    assert merged.shape == base.shape
    assert info["exclude_boxes"], "应当有排除区（头部）"
    fx0, fy0, fx1, fy1 = info["exclude_boxes"][0]
    fcy, fcx = (fy0 + fy1) // 2, (fx0 + fx1) // 2
    assert merged[fcy, fcx].tolist() == base[fcy, fcx].tolist()
    # 人物框内、头部以下的位置应被贴回结果覆盖
    bx0, by0, bx1, by1 = info["box"]
    sy, sx = by1 - 40, (bx0 + bx1) // 2
    assert abs(int(merged[sy, sx][0]) - 200) <= 2


def test_run_pipeline_local_step_uses_callable(tmp_path, monkeypatch):
    img = tmp_path / "img2.png"
    _make_image(img)

    def _fake_local(image_path, out_path, **kwargs):
        os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
        cv2.imwrite(str(out_path), cv2.imread(image_path))
        return {"out": str(out_path), "box": (0, 0, 1, 1)}

    monkeypatch.setattr(pp, "local_repaint_composite", _fake_local)
    steps = pp.default_pipeline()
    steps["local"]["enabled"] = True
    steps["local"]["region"] = "face"
    out = pp.run_pipeline([str(img)], steps, final_dir=str(tmp_path / "final"))
    assert out and "-final-" in os.path.basename(out[0])


# ------------------------------------------------------- GUI 勾选框
# 注：GUI 相关用例放在 tests/test_gpt_image2_api.py（那边有可用的 tab fixture；
# 在无显示环境里跨模块临时构造控件会直接崩）。这里只测纯逻辑。



def test_final_goes_to_final_dir_and_intermediates_to_work_dir(tmp_path, monkeypatch):
    """多工序时：最后一道工序的产物落 final_dir（默认 data/<日期>），中间产物落 pipeline-steps。"""
    img = tmp_path / "shot.png"
    _make_image(img)
    final_dir = tmp_path / "data" / "20260923"
    work = tmp_path / "data" / "20260923" / "pipeline-steps"

    def _fake_local(image_path, out_path, **kwargs):
        import cv2 as _cv2
        _cv2.imwrite(str(out_path), _cv2.imread(image_path))
        return {"out": str(out_path), "box": (0, 0, 1, 1)}

    monkeypatch.setattr(pp, "local_repaint_composite", _fake_local)
    steps = pp.default_pipeline()
    steps["structure"]["enabled"] = True
    steps["local"]["enabled"] = True
    steps["local"]["region"] = "hair"
    out = pp.run_pipeline([str(img)], steps, work_dir=str(work), final_dir=str(final_dir))
    assert out and os.path.dirname(out[0]) == str(final_dir)          # 最终产物 → data/<日期>
    assert os.path.isfile(os.path.join(str(work), "pipeline-manifest.json"))
    # 结构线那步是中间产物，应留在 pipeline-steps
    assert [p for p in os.listdir(work) if "-sline" in p]


def test_single_step_pipeline_output_is_final(tmp_path):
    """只勾一道工序时，它就是最终产物 → 直接落 final_dir。"""
    img = tmp_path / "shot1.png"
    _make_image(img)
    final_dir = tmp_path / "out"
    steps = pp.default_pipeline()
    steps["structure"]["enabled"] = True
    out = pp.run_pipeline([str(img)], steps, final_dir=str(final_dir))
    assert out and os.path.dirname(out[0]) == str(final_dir)
    assert "-final-sline" in os.path.basename(out[0])


def test_date_output_dir_shape():
    assert pp.date_output_dir("20260923") == os.path.join("data", "20260923")


def test_run_pipeline_records_failure_and_resume_skips_success(tmp_path, monkeypatch):
    """失败步骤要记进清单；重试时跳过已成功步骤，只重跑失败那步。"""
    img = tmp_path / "shot2.png"
    _make_image(img)
    calls = []

    def _failing_local(image_path, out_path, **kwargs):
        calls.append("local")
        if len(calls) == 1:
            raise RuntimeError("模拟失败")
        cv2.imwrite(str(out_path), cv2.imread(image_path))     # 假的重绘：直接复制，不调 API
        return {"out": str(out_path), "box": (0, 0, 1, 1), "exclude_boxes": []}

    monkeypatch.setattr(pp, "local_repaint_composite", _failing_local)
    steps = pp.default_pipeline()
    steps["structure"]["enabled"] = True
    steps["local"]["enabled"] = True
    steps["local"]["region"] = "hair"
    work_dir = str(tmp_path / "work")
    out1 = pp.run_pipeline([str(img)], steps, log_callback=lambda m: None,
                           final_dir=str(tmp_path / "final"), work_dir=work_dir)
    failures = pp.pipeline_failures(work_dir)
    assert failures and failures[0]["step"] == "local"
    manifest = pp.load_manifest(work_dir)
    succeeded = [s["key"] for s in manifest["items"][0]["steps"] if s["status"] == "succeeded"]
    assert "structure" in succeeded            # 结构线那步成功了
    assert out1 and os.path.basename(out1[0]).endswith(".png")   # 失败后停在成功的最后一步

    # 重试：structure 不该再被重算（产物已存在 → 复用），只重跑 local
    overlay_calls = []
    real_overlay = pp.structure_overlay_file

    def _count_overlay(*a, **k):
        overlay_calls.append(1)
        return real_overlay(*a, **k)

    monkeypatch.setattr(pp, "structure_overlay_file", _count_overlay)
    out2 = pp.run_pipeline([str(img)], steps, log_callback=lambda m: None,
                           final_dir=str(tmp_path / "final"), work_dir=work_dir)
    assert overlay_calls == []                 # 跳过了已成功步骤
    assert out2 and "final-sline" in os.path.basename(out2[0]) and "local-hair" in os.path.basename(out2[0])


def test_pipeline_failures_empty_without_manifest(tmp_path):
    assert pp.pipeline_failures(str(tmp_path / "nope")) == []


def test_auto_repaint_scale_never_exceeds_api_cap():
    """自适应缩放：大裁切不放大（防"放大→接口压缩→再放大"），小裁切才放大。"""
    big = pp.auto_repaint_scale(1776, 3189)          # 超过 2048 上限
    assert big <= 2048 / 3189 + 1e-3 and big < 1.0
    mid = pp.auto_repaint_scale(1184, 2126)
    assert mid <= 2048 / 2126 + 1e-3
    small = pp.auto_repaint_scale(700, 900)
    assert 1.5 <= small <= 2.0                        # 小裁切放大但不超 max_scale
    assert pp.auto_repaint_scale(0, 0) == 1.0


def test_firmware_path_is_resolved_to_text(tmp_path):
    """固件传路径时必须读成文本（否则会拿文件名当提示词发出去）。"""
    fw = tmp_path / "firmware.md"
    fw.write_text("You are a CONSERVATIVE restoration model. Keep everything.", encoding="utf-8")
    assert pp.resolve_firmware_text(str(fw)).startswith("You are a CONSERVATIVE")
    assert pp.resolve_firmware_text("plain prompt text") == "plain prompt text"
    assert pp.resolve_firmware_text("") == ""


def test_local_repaint_gets_firmware_text_not_path(tmp_path):
    img = tmp_path / "in.png"
    _make_image(img)
    seen = {}
    fw = tmp_path / "fw.md"
    fw.write_text("FIRMWARE BODY " * 40, encoding="utf-8")

    def _fake(crop_path, firmware):
        seen["fw"] = firmware
        import cv2 as _cv2
        patch = tmp_path / "patch.png"
        _cv2.imwrite(str(patch), _cv2.imread(crop_path))
        return str(patch)

    pp.local_repaint_composite(str(img), str(tmp_path / "out.png"), region="hair",
                               firmware=str(fw), repaint_callable=_fake, log_callback=lambda m: None)
    assert seen["fw"].startswith("FIRMWARE BODY")


def test_run_pipeline_local_output_is_png(tmp_path, monkeypatch):
    img = tmp_path / "in2.png"
    _make_image(img)

    def _fake_local(image_path, out_path, **kwargs):
        import cv2 as _cv2
        _cv2.imwrite(str(out_path), _cv2.imread(image_path))
        return {"out": str(out_path), "box": (0, 0, 1, 1)}

    monkeypatch.setattr(pp, "local_repaint_composite", _fake_local)
    steps = pp.default_pipeline()
    steps["local"]["enabled"] = True
    out = pp.run_pipeline([str(img)], steps, final_dir=str(tmp_path / "final"))
    assert out and out[0].endswith(".png") and "-final-" in os.path.basename(out[0])


def test_multi_region_local_names_all_regions(tmp_path, monkeypatch):
    """多区域局部重绘：名字必须列出所有区域。

    修之前 `final_product_name` / 中间产物名只取 `regions[0]`，于是「跑了 4 个区域」的产物
    叫 `-local-subject_no_face.png`（最后一块贴回的中间文件还会覆盖同名的第一块），
    光看文件名分不清实际跑了哪几个区域 —— 排查「鞋/靴紧框没生效」时就卡在这上面。
    """
    img = tmp_path / "multi.png"
    _make_image(img)
    calls = []

    def _fake_local(image_path, out_path, **kwargs):
        calls.append(kwargs.get("region"))
        cv2.imwrite(str(out_path), cv2.imread(image_path))
        return {"out": str(out_path), "box": (0, 0, 1, 1), "exclude_boxes": []}

    monkeypatch.setattr(pp, "local_repaint_composite", _fake_local)
    steps = pp.default_pipeline()
    regions = ["subject_no_face", "shoes_zoom", "waist"]
    steps["local"].update({"enabled": True, "regions": regions, "region": regions[0], "feather": 48})
    steps["ink"]["enabled"] = True           # 让 local 变成中间步骤，走中间产物命名那条分支
    work = tmp_path / "work"
    out = pp.run_pipeline([str(img)], steps, final_dir=str(tmp_path / "final"),
                          work_dir=str(work), resume=False, log_callback=lambda m: None)
    assert calls == regions                                    # 每个区域都跑过、顺序不变
    name = os.path.basename(out[0])
    assert "final-local-" + ",".join(regions) in name
    mids = [p for p in os.listdir(str(work)) if p.endswith("-local-" + ",".join(regions) + ".png")]
    assert mids, "中间产物名也要列出所有区域"


def test_ink_lines_iterates_until_target_within_budget(monkeypatch):
    """加墨要迭代到「线-底对比」达标：单遍压深后重新抽线只回来一点点（实测目标 8 只走到 1.2）。

    这里把量尺与抽线换成假的：量尺依次给出 0/2/4/10（目标 8），第 4 次达标 → 应当压深 3 遍，
    且单像素累计压深不超过 max_darken（否则迭代会把线压成黑粗块）。
    """
    img = np.full((120, 160, 3), 200, np.uint8)
    mask = np.zeros((120, 160), np.uint8)
    mask[40:80, 10:150] = 255                      # 假装这些像素是"线"
    monkeypatch.setattr(pp, "extract_structure_lines", lambda *a, **k: mask.copy())
    values = iter([0.0, 2.0, 4.0, 10.0])
    calls = []

    def _fake_contrast(image, **kwargs):
        calls.append(1)
        return next(values, 10.0)

    monkeypatch.setattr(pp, "line_contrast", _fake_contrast)
    out = pp.ink_lines(img, target_sep=8.0, max_darken=40.0, max_passes=4)
    assert len(calls) == 4                          # 第 4 次量到 10 ≥ 8 才停
    delta = int(img[60, 80][0]) - int(out[60, 80][0])
    assert 0 < delta <= 40                          # 确实压深了，但没超预算
    assert np.array_equal(out[0, 0], img[0, 0])     # 非线像素不动


def test_ink_lines_noop_when_already_dark_enough(monkeypatch):
    """已经达标的产物再跑一遍加墨必须原样返回（可重复跑、不会越跑越黑）。"""
    img = np.full((60, 80, 3), 180, np.uint8)
    monkeypatch.setattr(pp, "line_contrast", lambda *a, **k: 12.0)
    assert np.array_equal(pp.ink_lines(img, target_sep=8.0), img)


def test_repaint_reference_mode_style_uses_style_image_not_line_anchor(tmp_path, monkeypatch):
    """`reference_mode="style"`：重绘的第二张参考是画风图，且**不生成线锚图**。

    线锚图被模型当成渲染语言时会把整张图塌成线稿/印上碎玻璃纹理（用户 2026-09-24 报的 bug），
    所以默认走画风图；要线锚图必须显式 `reference_mode="line_anchor"`（见上一条用例）。
    """
    from modules.others import api_backend
    img = tmp_path / "img.png"
    style = tmp_path / "style.png"
    _make_image(img)
    _make_image(style)
    seen = {}

    def _fake_repaint(**kwargs):
        seen.update(kwargs)
        out = tmp_path / "repainted.png"
        import cv2 as _cv2
        _cv2.imwrite(str(out), _cv2.imread(str(img)))
        return [str(out)]

    monkeypatch.setattr(api_backend, "generate_image_repaint", _fake_repaint)
    work = tmp_path / "work"
    steps = pp.default_pipeline()
    steps["repaint"].update({"enabled": True, "reference_mode": "style"})
    pp.run_pipeline([str(img)], steps, work_dir=str(work), final_dir=str(tmp_path / "final"),
                    style_ref_path=str(style), log_callback=lambda m: None, resume=False)
    refs = seen.get("extra_reference_paths") or []
    assert refs and os.path.basename(refs[0]) == "style.png"
    assert not [p for p in os.listdir(str(work)) if "lineanchor" in p], "style 模式不该生成线锚图"


def test_region_mask_falls_back_when_it_covers_nothing(tmp_path, monkeypatch):
    """主体遮罩在本次区域里几乎没覆盖 → 改为整框贴回（否则这次重绘等于白跑）。

    实测触发场景：亮蕾丝背景的全身插画上，整图 GrabCut 只框住了下半身，
    上半身的区域（含脸/手）贴回时被遮罩整块切掉，QA 报「改动像素占比 0.000」。
    """
    base_path = tmp_path / "maskless.png"
    base = _make_image(base_path, size=(420, 640))
    mask = np.zeros(base.shape[:2], np.uint8)
    mask[600:, :] = 255                     # 遮罩只在画面最底部 → 区域里覆盖≈0
    monkeypatch.setattr(pp, "subject_mask", lambda *a, **k: mask.copy())
    monkeypatch.setattr(pp, "detect_hand_boxes", lambda *a, **k: [])

    def _solid(crop_path, firmware):
        import cv2 as _cv2
        crop = _cv2.imread(crop_path)
        patch = tmp_path / "patch.png"
        _cv2.imwrite(str(patch), np.full_like(crop, 30))
        return str(patch)

    info = {}
    pp.local_repaint_composite(str(base_path), str(tmp_path / "out.png"), region="face",
                               repaint_callable=_solid, restrict_to_subject=True,
                               qa_callback=lambda d: info.update(d),
                               log_callback=lambda m: None)
    assert info["changed_ratio"] > 0.3, "遮罩没覆盖区域时应整框贴回，不能什么都不改"


def test_run_pipeline_uses_per_run_dir_and_unique_names(tmp_path, monkeypatch):
    """每次运行一个独立中间目录；最终产物名带 run-id；裁切图只留在中间目录。"""
    img = tmp_path / "shot3.png"
    _make_image(img)
    final_dir = tmp_path / "final"

    def _fake_local(image_path, out_path, **kwargs):
        import cv2 as _cv2
        scratch = kwargs.get("scratch_dir") or os.path.dirname(str(out_path))
        os.makedirs(scratch, exist_ok=True)
        _cv2.imwrite(os.path.join(scratch, "fake-crop.png"), _cv2.imread(image_path))
        _cv2.imwrite(str(out_path), _cv2.imread(image_path))
        return {"out": str(out_path), "box": (0, 0, 1, 1)}

    monkeypatch.setattr(pp, "local_repaint_composite", _fake_local)
    steps = pp.default_pipeline()
    steps["structure"]["enabled"] = True
    steps["local"]["enabled"] = True
    steps["local"]["region"] = "hair"

    out_a = pp.run_pipeline([str(img)], steps, final_dir=str(final_dir))
    out_b = pp.run_pipeline([str(img)], steps, final_dir=str(final_dir), resume=False)
    # 两次运行的中间目录不同、产物名不同（多线程/重复跑不会互相覆盖）
    dirs = [os.path.dirname(p) for p in out_a + out_b]
    assert len(set(dirs)) == 1                       # 最终产物都在 final_dir
    assert out_a[0] != out_b[0]                      # 名字带 run-id
    runs = os.listdir(os.path.join(str(final_dir), "pipeline-steps"))
    assert len(runs) == 2
    # 最终目录里不该出现裁切图
    assert not [p for p in os.listdir(str(final_dir)) if p.endswith("-crop.png")]


def test_local_scratch_dir_keeps_crop_out_of_output_dir(tmp_path):
    """裁切图写到 scratch_dir，不落在最终产物目录。"""
    img = tmp_path / "in3.png"
    _make_image(img)
    out_dir = tmp_path / "final2"
    scratch = tmp_path / "run"
    out_dir.mkdir()
    scratch.mkdir()

    def _fake(crop_path, firmware):
        import cv2 as _cv2
        assert os.path.dirname(str(crop_path)) == str(scratch)     # 裁切图在运行目录里
        patch = scratch / "patch.png"
        _cv2.imwrite(str(patch), _cv2.imread(crop_path))
        return str(patch)

    pp.local_repaint_composite(str(img), str(out_dir / "final.png"), region="hair",
                               repaint_callable=_fake, scratch_dir=str(scratch),
                               log_callback=lambda m: None)
    assert [p for p in os.listdir(str(scratch)) if p.endswith("-crop.png")]
    assert not [p for p in os.listdir(str(out_dir)) if p.endswith("-crop.png")]


def test_subject_no_face_keeps_hands_in_repaint(tmp_path, monkeypatch):
    """「人物（不含面部，含手）」：只排脸，**手仍然参与重绘**（贴合用户要求）。"""
    img = _make_image(tmp_path / "hand.png", size=(420, 640))
    monkeypatch.setattr(pp, "detect_hand_boxes", lambda image, subject_box=None, **kw: [(300, 400, 360, 460)])
    box = pp.resolve_region_box("subject_no_face", img.shape[1], img.shape[0], image=img)
    ex = pp.resolve_exclusion_boxes("subject_no_face", img, box=box)
    assert len(ex) == 1                          # 只有脸，手不排除
    assert (300, 400, 360, 460) not in ex


def test_subject_keep_hands_region_excludes_hands(tmp_path, monkeypatch):
    """想完全不重绘手时用「人物（不含面部·保留手部）」这个独立区域。"""
    img = _make_image(tmp_path / "keep.png", size=(420, 640))
    monkeypatch.setattr(pp, "detect_hand_boxes", lambda image, subject_box=None, **kw: [(300, 400, 360, 460)])
    box = pp.resolve_region_box("subject_keep_hands", img.shape[1], img.shape[0], image=img)
    ex = pp.resolve_exclusion_boxes("subject_keep_hands", img, box=box)
    assert (300, 400, 360, 460) in ex and len(ex) >= 2


def test_glove_state_clause_present_for_figure_regions():
    """人物区域必须带「戴手套状态不能被改」的条款（含手重绘时尤其重要）。"""
    assert "HANDS AND GLOVES" in pp.GLOVE_STATE_CLAUSE
    assert "gloved" in pp.GLOVE_STATE_CLAUSE and "bare hand" in pp.GLOVE_STATE_CLAUSE
    for key in ("subject", "subject_no_face", "subject_keep_hands"):
        assert "HANDS AND GLOVES" in pp.REGION_EMPHASIS[key], key
    assert "subject_keep_hands" in pp.REGION_LABELS


def test_exclusion_accepts_manual_boxes(tmp_path):
    img = _make_image(tmp_path / "manual.png", size=(300, 420))
    box = pp.resolve_region_box("subject", img.shape[1], img.shape[0], image=img)
    ex = pp.resolve_exclusion_boxes("subject", img, box=box, manual=[(10, 20, 110, 120)])
    assert (10, 20, 110, 120) in ex


def test_local_repaint_keeps_manual_exclusion_untouched(tmp_path, monkeypatch):
    """手动排除框内的像素必须原样保留。"""
    base_path = tmp_path / "base.png"
    base = _make_image(base_path, size=(400, 560))
    monkeypatch.setattr(pp, "detect_hand_boxes", lambda *a, **k: [])

    def _solid(crop_path, firmware):
        import cv2 as _cv2
        crop = _cv2.imread(crop_path)
        out = tmp_path / "patch.png"
        _cv2.imwrite(str(out), np.full_like(crop, 240))
        return str(out)

    out_path = tmp_path / "out.png"
    keep = (200, 300, 260, 360)
    info = pp.local_repaint_composite(str(base_path), str(out_path), region="subject",
                                      repaint_callable=_solid, feather=20,
                                      exclude_boxes=[keep], log_callback=lambda m: None)
    merged = cv2.imread(str(out_path))
    cy, cx = (keep[1] + keep[3]) // 2, (keep[0] + keep[2]) // 2
    assert merged[cy, cx].tolist() == base[cy, cx].tolist()
    assert keep in [tuple(b) for b in info["exclude_boxes"]]


def test_detect_hand_boxes_degrades_when_subprocess_unavailable(monkeypatch, tmp_path):
    """姿态子进程不可用（崩溃/超时/没装 torch）时要安全降级，不能抛错。"""
    import subprocess as _sp
    blank = np.full((240, 160, 3), 255, np.uint8)
    monkeypatch.setattr(_sp, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pp.detect_hand_boxes(blank, subject_box=(10, 10, 150, 230)) == []


def test_detect_hand_boxes_uses_cache(tmp_path, monkeypatch):
    """同一张图第二次调用应命中缓存（不再跑子进程）。"""
    import subprocess as _sp
    img = tmp_path / "hand_cache.png"
    out = _make_image(img, size=(300, 420))
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        import json as _json
        out_json = cmd[cmd.index("--out") + 1]
        _json.dump({"ok": True, "hands": [[10, 20, 60, 70]]}, open(out_json, "w", encoding="utf-8"))
        return None

    monkeypatch.setattr(_sp, "run", _fake_run)
    monkeypatch.setattr(pp, "_HAND_CACHE_PATH", str(tmp_path / "hand_boxes.json"))
    first = pp.detect_hand_boxes(out, subject_box=(0, 0, 300, 420), image_path=str(img))
    second = pp.detect_hand_boxes(out, subject_box=(0, 0, 300, 420), image_path=str(img))
    assert first == second == [(10, 20, 60, 70)]
    assert len(calls) == 1                      # 第二次命中缓存


def test_hand_pose_tool_exists():
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "detect_hands_pose.py")
    assert os.path.isfile(tool)


def test_dual_reference_builds_line_anchor_and_passes_it(tmp_path, monkeypatch):
    """重绘默认双参考：先本地生成线锚图，再作为第二张参考传给重绘。"""
    from modules.others import api_backend
    img = tmp_path / "src.png"
    _make_image(img)
    seen = {}

    def _fake_repaint(**kwargs):
        seen.update(kwargs)
        out = tmp_path / "repainted.png"
        import cv2 as _cv2
        _cv2.imwrite(str(out), _cv2.imread(str(img)))
        return [str(out)]

    monkeypatch.setattr(api_backend, "generate_image_repaint", _fake_repaint)
    steps = pp.default_pipeline()
    steps["repaint"]["enabled"] = True
    steps["repaint"]["dual_reference"] = True
    work = tmp_path / "work"
    pp.run_pipeline([str(img)], steps, work_dir=str(work), final_dir=str(tmp_path / "final"),
                    log_callback=lambda m: None, resume=False)
    refs = seen.get("extra_reference_paths") or []
    assert refs and os.path.isfile(refs[0])            # 线锚图已生成并传下去
    assert refs[0].endswith("-lineanchor.png")


def test_dual_reference_can_be_disabled(tmp_path, monkeypatch):
    from modules.others import api_backend
    img = tmp_path / "src2.png"
    _make_image(img)
    seen = {}

    def _fake_repaint(**kwargs):
        seen.update(kwargs)
        out = tmp_path / "r2.png"
        import cv2 as _cv2
        _cv2.imwrite(str(out), _cv2.imread(str(img)))
        return [str(out)]

    monkeypatch.setattr(api_backend, "generate_image_repaint", _fake_repaint)
    steps = pp.default_pipeline()
    steps["repaint"]["enabled"] = True
    steps["repaint"]["dual_reference"] = False
    pp.run_pipeline([str(img)], steps, work_dir=str(tmp_path / "w2"), final_dir=str(tmp_path / "f2"),
                    log_callback=lambda m: None, resume=False)
    assert not (seen.get("extra_reference_paths") or [])


def test_region_labels_are_explicit_about_face_and_hands():
    """区域命名要一眼看懂「保脸/保手」，不再用「不含面部」这种绕的说法。"""
    assert "保脸" in pp.REGION_LABELS["subject_no_face"]
    assert "手参与重绘" in pp.REGION_LABELS["subject_no_face"]
    assert "保手" in pp.REGION_LABELS["subject_keep_hands"]
    for key in ("subject_no_face", "subject_keep_hands"):
        assert key in pp.REGION_TOOLTIPS and pp.REGION_TOOLTIPS[key]


def test_tone_calibrate_darkens_toward_target():
    """色调校准：目标亮度更低时整体变暗，且高光裁剪比例不上升。"""
    import numpy as np
    import cv2
    base = np.full((300, 200, 3), 235, np.uint8)          # 一张很亮的图（模拟发白）
    base[50:150, 40:160] = 200
    cur = float(cv2.cvtColor(base, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
    out = pp.tone_calibrate(base, target_brightness=cur - 25, contrast=1.15, chroma=1.1)
    new = float(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
    assert new < cur - 8, (cur, new)                       # 确实压暗了


def test_tone_calibrate_does_not_wash_out_highlights():
    """高光只压缩不抬升：白裁剪（>250）比例不能明显变多。"""
    import numpy as np
    import cv2
    rng = np.random.default_rng(0)
    img = rng.integers(150, 256, size=(200, 200, 3), dtype=np.uint8)
    before = float((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 250).mean())
    out = pp.tone_calibrate(img, target_brightness=140, contrast=1.25, chroma=1.2)
    after = float((cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) > 250).mean())
    assert after <= before + 0.01


def test_reference_brightness_reads_photo(tmp_path):
    img = tmp_path / "ref.png"
    import numpy as np
    import cv2
    cv2.imwrite(str(img), np.full((100, 100, 3), 120, np.uint8))
    val = pp.reference_brightness(str(img))
    assert val is not None and 100 < val < 140
    assert pp.reference_brightness(str(tmp_path / "nope.png")) is None
