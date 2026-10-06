# -*- coding: utf-8 -*-
"""线稿/发丝质量与「画面可读性」的自动指标（面向 gpt-image 画风生图）。

为什么不用全局亮度/白底/边缘密度当目标：这三个量高度受构图影响（人物占画幅比例一变，数字就不可比），
参考图与产物内容不同时，它们主要测到的是构图差而不是画风差。见
`docs/gpt-image-tid-style/README.md` §4.8。

这里实现 4 组更贴近「看着清不清楚 / 线连不连」的指标（全部本地可算）：

1. 分区明度层次 `tone`
   - 亮度分位 P10/P50/P90、局部 RMS 对比度（Laplacian 局部标准差，衡量"层次感"）
   - 亮部裁切率（>250）与暗部占比（<60）、以及**中灰带占比**（90~200）
   - 这些是"过曝/过灰"的直接度量，与主体占画幅关系较弱。

2. 多尺度边缘比例 `edges`
   - 粗/中/细三个尺度的 Canny 边缘能量占比（高斯预模糊 σ=2.0 / 1.0 / 0）
   - 目标是与参考图**比例匹配**，而不是总边缘越多越好；细尺度占比过高 = 碎线/噪声。

3. 线条连续性 `lines`
   - Canny → 形态学细化(skeleton) → 8 邻域连通域分析
   - 平均连通段长度、中位数、>60px 的长线占比、<15px 的碎线占比、端点密度（每 1000 骨架像素的端点数）
   - 碎线多 / 端点密 = 断线、糊边；长线多 = 结构连贯。

4. 负空间与主体占比 `space`
   - 低纹理高明度掩码（亮度>235 且局部标准差<6）的连通域：最大干净区占比、总占比
   - 主体占画幅（非留白且非深背景的中间调像素占比）

用法：
  python tests/style_render_metrics.py --ref data/style-ref/tid.png --images a.png b.png
  python tests/style_render_metrics.py --ref R.png --images X.png --json-out out.json
"""
import argparse
import json
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


from utils.style_render_metrics import (tone_stats, multiscale_edge_ratio, _skeleton, line_continuity, space_stats, analyze, _ratio, closeness)

def main():
    parser = argparse.ArgumentParser(description="线稿/发丝质量与可读性指标")
    parser.add_argument("--ref", required=True, help="参考图路径")
    parser.add_argument("--images", nargs="+", required=True, help="待评估图")
    parser.add_argument("--json-out", help="把结果写成 JSON")
    args = parser.parse_args()

    ref = analyze(args.ref)
    rows = []
    for p in args.images:
        cur = analyze(p)
        if not cur:
            print(f"[skip] 读不到 {p}")
            continue
        cur.update(closeness(ref, cur))
        rows.append(cur)

    cols = ["file", "render_score", "tone_score", "edge_score", "line_score", "space_score",
            "tone_range_p90_p10", "local_rms_contrast", "mid_band_ratio_90_200", "clip_high_ratio_250",
            "edge_share_fine", "line_avg_len", "line_long_ratio", "line_frag_ratio",
            "line_endpoint_density", "clean_space_ratio", "largest_clean_space_ratio"]
    print("\n参考图:", args.ref)
    print("  " + json.dumps({k: ref[k] for k in cols if k in ref and k != "file"}, ensure_ascii=False))
    print("\n" + " | ".join(f"{c[:14]:>14}" for c in cols))
    for r in sorted(rows, key=lambda x: -x["render_score"]):
        print(" | ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in cols))

    if args.json_out:
        payload = {"ref": ref, "rows": rows}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nsaved {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
