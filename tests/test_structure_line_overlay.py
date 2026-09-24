# -*- coding: utf-8 -*-
"""结构线叠加（tools/overlay_structure_lines.py）：Sol 提的"结构线图层叠加"后处理。

关键行为：叠加后线条指标应改善（段长↑或端点密度↓），而配色/饱和度基本不动。
"""
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tests"))

from utils.post_process import extract_structure_lines, overlay_structure_lines as overlay  # noqa: E402
import style_render_metrics as m  # noqa: E402


def _synthetic_drawing(path, size=(420, 620)):
    """造一张"线条不连续"的图：几个形状 + 断续边。"""
    h, w = size
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.ellipse(img, (w // 2, 150), (80, 100), 0, 0, 360, (200, 180, 170), -1)
    cv2.rectangle(img, (w // 2 - 90, 250), (w // 2 + 90, 520), (220, 215, 205), -1)
    # 断续轮廓：每 20px 画 10px
    for x in range(w // 2 - 90, w // 2 + 90, 20):
        cv2.line(img, (x, 250), (x + 10, 250), (120, 110, 100), 2)
    for y in range(250, 520, 20):
        cv2.line(img, (w // 2 - 90, y), (w // 2 - 90, y + 10), (120, 110, 100), 2)
    cv2.imwrite(path, img)
    return img


def test_extract_structure_lines_keeps_long_components(tmp_path):
    p = tmp_path / "draw.png"
    img = _synthetic_drawing(p)
    mask = extract_structure_lines(img, min_len=40)
    assert mask.max() == 255
    # 长连通域应被保留、极短噪声被丢弃
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    assert areas and min(areas) >= 40


def test_overlay_improves_line_metrics_without_colour_shift(tmp_path):
    src = tmp_path / "src.png"
    base = _synthetic_drawing(src)
    before = m.analyze(str(src))
    out, mask = overlay(base, base, strength=0.5, min_len=40, darken=0.45)
    dst = tmp_path / "out.png"
    cv2.imwrite(dst, out)
    after = m.analyze(str(dst))
    assert (mask > 0).mean() > 0.001
    # 线条连通性：段长上升或端点密度下降（至少一项明显改善）
    assert after["line_avg_len"] > before["line_avg_len"] or \
        after["line_endpoint_density"] < before["line_endpoint_density"]
    # 配色/色调基本不动
    assert abs(after["saturation_mean"] - before["saturation_mean"]) < 3.0


def test_overlay_zero_strength_is_noop(tmp_path):
    src = tmp_path / "src2.png"
    base = _synthetic_drawing(src)
    out, _mask = overlay(base, base, strength=0.0, min_len=40)
    assert np.abs(out.astype(int) - base.astype(int)).max() <= 1
