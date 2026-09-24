# -*- coding: utf-8 -*-
"""给画风参考图算一组「可量化」的渲染统计，作为 prompt_gpt 转换的客观提示。

输出（一行一段，直接塞进转换提示词）：
- 亮度/暗部占比/高光占比/白底占比/饱和度/色相分布/主色（k-means）/边缘密度/线色亮度/对比度

用法（仓库根目录）：
  python tools/style_ref_stats.py --style tid
  python tools/style_ref_stats.py --all --json-out cache/temp/style-stats.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.styles import ref_image_valid, style_ref_image  # noqa: E402

STYLES_FILE = os.path.join(PROJECT_ROOT, "conf", "config-styles.json")
HUE_NAMES = [
    ("red", 0, 8), ("orange", 8, 22), ("yellow", 22, 33), ("green", 33, 78),
    ("cyan", 78, 98), ("blue", 98, 128), ("purple", 128, 150), ("magenta", 150, 170),
]


def stats_for(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0]
    edges = cv2.Canny(gray, 50, 150)

    # 主要色相（按占比排序，排除近白/近灰的低饱和像素）
    colored = sat > 40
    hue_share = []
    total = int(colored.sum())
    for name, lo, hi in HUE_NAMES:
        cnt = int(((hue >= lo) & (hue < hi) & colored).sum())
        if total:
            hue_share.append((name, round(cnt / total, 3)))
    hue_share.sort(key=lambda x: -x[1])

    # 主色（k-means，取 5 个）
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    data = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 5, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=5)
    order = np.argsort(-counts)
    palette = []
    for i in order:
        b, g, r = centers[i]
        palette.append({"hex": "#%02x%02x%02x" % (int(r), int(g), int(b)),
                        "share": round(float(counts[i]) / counts.sum(), 3)})

    edge_mask = edges > 0
    line_brightness = float(gray[edge_mask].mean()) if edge_mask.any() else 0.0

    return {
        "size": f"{w}x{h}",
        "brightness_mean": round(float(gray.mean()), 1),
        "brightness_std": round(float(gray.std()), 1),
        "deep_shadow_ratio": round(float((gray < 60).mean()), 3),
        "highlight_ratio_230": round(float((gray > 230).mean()), 3),
        "near_white_ratio_245": round(float((gray > 245).mean()), 3),
        "saturation_mean": round(float(sat.mean()), 1),
        "saturation_std": round(float(sat.std()), 1),
        "high_sat_ratio": round(float((sat > 150).mean()), 3),
        "low_sat_ratio": round(float((sat < 40).mean()), 3),
        "value_mean": round(float(val.mean()), 1),
        "hue_shares": hue_share[:4],
        "dominant_palette": palette,
        "edge_density": round(float(edges.mean() / 255.0), 4),
        "line_brightness": round(line_brightness, 1),
        "contrast_std": round(float(gray.std()), 1),
    }


def to_text(name: str, st: dict) -> str:
    if not st:
        return f"[{name}] 无可用参考图"
    hues = ", ".join(f"{n} {int(s * 100)}%" for n, s in st["hue_shares"])
    pal = ", ".join(f"{p['hex']} ({int(p['share'] * 100)}%)" for p in st["dominant_palette"])
    return (
        f"[{name}] size {st['size']}; mean brightness {st['brightness_mean']}/255; "
        f"brightness std {st['brightness_std']}; pixels below 60 = {st['deep_shadow_ratio']}; "
        f"pixels above 230 = {st['highlight_ratio_230']}; above 245 = {st['near_white_ratio_245']}; "
        f"mean saturation {st['saturation_mean']}/255 (std {st['saturation_std']}); "
        f"highly saturated pixels = {st['high_sat_ratio']}; low saturation pixels = {st['low_sat_ratio']}; "
        f"hue distribution {hues}; dominant colours {pal}; "
        f"edge density {st['edge_density']}; edge pixel brightness {st['line_brightness']}"
    )


def main():
    parser = argparse.ArgumentParser(description="画风参考图的渲染统计")
    parser.add_argument("--style", help="单个画风名")
    parser.add_argument("--all", action="store_true", help="处理全部画风")
    parser.add_argument("--json-out", help="把结果写成 JSON")
    parser.add_argument("--styles-file", default=STYLES_FILE)
    args = parser.parse_args()

    with open(args.styles_file, "r", encoding="utf-8") as f:
        styles = json.load(f)

    names = [args.style] if args.style else (list(styles) if args.all else [])
    if not names:
        parser.error("需要 --style 或 --all")

    out = {}
    for name in names:
        ref = style_ref_image(styles, name)
        if not ref_image_valid(ref):
            print(f"[skip] {name}: 无可用参考图")
            continue
        st = stats_for(ref)
        out[name] = st
        print(to_text(name, st))

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nsaved {args.json_out}（{len(out)} 个画风）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
