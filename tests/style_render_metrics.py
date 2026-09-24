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

import cv2
import numpy as np


# ---------------- 1. 分区明度层次 ----------------

def tone_stats(gray: np.ndarray) -> dict:
    p10, p50, p90 = (float(np.percentile(gray, q)) for q in (10, 50, 90))
    # 局部对比：Laplacian 的局部标准差（用 9x9 均值池化），刻画"有没有层次"
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    local_rms = float(np.sqrt(cv2.boxFilter(lap * lap, -1, (9, 9)).mean()))
    return {
        "tone_p10": round(p10, 1),
        "tone_p50": round(p50, 1),
        "tone_p90": round(p90, 1),
        "tone_range_p90_p10": round(p90 - p10, 1),
        "local_rms_contrast": round(local_rms, 3),
        "clip_high_ratio_250": round(float((gray > 250).mean()), 4),
        "shadow_ratio_60": round(float((gray < 60).mean()), 4),
        "mid_band_ratio_90_200": round(float(((gray >= 90) & (gray <= 200)).mean()), 4),
    }


# ---------------- 2. 多尺度边缘比例 ----------------

def multiscale_edge_ratio(gray: np.ndarray) -> dict:
    scales = {"coarse": 2.0, "mid": 1.0, "fine": 0.0}
    energy = {}
    for name, sigma in scales.items():
        src = gray if sigma <= 0 else cv2.GaussianBlur(gray, (0, 0), sigma)
        edges = cv2.Canny(src, 50, 150)
        energy[name] = float(edges.mean() / 255.0 * 100.0)
    total = sum(energy.values()) + 1e-9
    return {
        "edge_energy_coarse": round(energy["coarse"], 3),
        "edge_energy_mid": round(energy["mid"], 3),
        "edge_energy_fine": round(energy["fine"], 3),
        "edge_share_coarse": round(energy["coarse"] / total, 3),
        "edge_share_mid": round(energy["mid"] / total, 3),
        "edge_share_fine": round(energy["fine"] / total, 3),
    }


# ---------------- 3. 线条连续性 ----------------

def _skeleton(edges: np.ndarray) -> np.ndarray:
    """形态学细化；缺 ximgproc 时用腐蚀-开运算近似（够用）。"""
    try:
        return cv2.ximgproc.thinning(edges)
    except Exception:  # noqa: BLE001
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros_like(edges)
        img = edges.copy()
        while cv2.countNonZero(img) > 0:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
            img = eroded
        return skel


def line_continuity(gray: np.ndarray, blur_sigma: float = 1.0, min_px: int = 8) -> dict:
    src = gray if blur_sigma <= 0 else cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    edges = cv2.Canny(src, 50, 150)
    skel = _skeleton(edges)
    binary = (skel > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lengths = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    total_px = int(binary.sum())
    if not lengths:
        return {
            "line_total_px": total_px, "line_segments": 0, "line_avg_len": 0.0, "line_median_len": 0.0,
            "line_long_ratio": 0.0, "line_frag_ratio": 0.0, "line_endpoint_density": 0.0,
        }
    lengths_arr = np.array(lengths, dtype=np.float64)
    long_px = float(lengths_arr[lengths_arr >= 60].sum())
    frag_px = float(lengths_arr[lengths_arr < 15].sum())
    # 端点密度：8 邻域只有 1 个邻居的骨架像素
    nb = cv2.filter2D(binary, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    endpoints = int(((binary == 1) & (nb == 2)).sum())
    return {
        "line_total_px": total_px,
        "line_segments": len(lengths),
        "line_avg_len": round(float(lengths_arr.mean()), 1),
        "line_median_len": round(float(np.median(lengths_arr)), 1),
        "line_long_ratio": round(long_px / (lengths_arr.sum() + 1e-9), 3),
        "line_frag_ratio": round(frag_px / (lengths_arr.sum() + 1e-9), 3),
        "line_endpoint_density": round(endpoints / (total_px + 1e-9) * 1000.0, 2),
    }


# ---------------- 4. 负空间与主体占比 ----------------

def space_stats(gray: np.ndarray, bright_thr: int = 235, flat_thr: float = 6.0) -> dict:
    local_std = cv2.boxFilter(gray.astype(np.float32) ** 2, -1, (9, 9)) - \
        cv2.boxFilter(gray.astype(np.float32), -1, (9, 9)) ** 2
    local_std = np.sqrt(np.clip(local_std, 0, None))
    clean_mask = ((gray > bright_thr) & (local_std < flat_thr)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    total_px = clean_mask.size
    largest = 0
    if n_labels > 1:
        largest = int(max(stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)))
    return {
        "clean_space_ratio": round(float(clean_mask.mean()), 3),
        "largest_clean_space_ratio": round(largest / total_px, 3),
        "subject_mid_ratio": round(float(((gray >= 90) & (gray <= 235)).mean()), 3),
    }


def analyze(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {"file": os.path.basename(path), "size": f"{img.shape[1]}x{img.shape[0]}",
           "brightness_mean": round(float(gray.mean()), 1),
           "saturation_mean": round(float(hsv[:, :, 1].mean()), 1)}
    out.update(tone_stats(gray))
    out.update(multiscale_edge_ratio(gray))
    out.update(line_continuity(gray))
    out.update(space_stats(gray))
    return out


def _ratio(target: float, value: float) -> float:
    """1.0 = 完全一致，0 = 差一倍以上。"""
    if target <= 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(value - target) / target))


def closeness(ref: dict, cur: dict) -> dict:
    """与参考图的贴近度（0~1，1 = 完全一致）。

    权重按"用户实际在意的顺序"分配：
    - **明度层次 35%**（过曝/发灰是最刺眼的问题）
    - **线稿连通 25%**（断线/糊线是第二个抱怨点）
    - **多尺度边缘匹配 25%**（碎线 vs 有效轮廓）
    - **负空间 15%**（留白差异极易被构图差异污染，只作弱约束）
    注意：参考图与产物内容不同时，这仍是"渲染质量贴近度"，不是"画风身份相同"。
    """
    keys = [
        ("tone_range_p90_p10", 0.30), ("local_rms_contrast", 0.30),
        ("mid_band_ratio_90_200", 0.20), ("clip_high_ratio_250", 0.20),
    ]
    tone_score = sum(w * _ratio(ref[k], cur[k]) for k, w in keys if ref.get(k) is not None)
    edge_keys = [("edge_share_coarse", 1 / 3), ("edge_share_mid", 1 / 3), ("edge_share_fine", 1 / 3)]
    edge_score = sum(w * _ratio(ref[k], cur[k]) for k, w in edge_keys if ref.get(k) is not None)
    line_score = np.mean([
        _ratio(ref["line_long_ratio"], cur["line_long_ratio"]) if ref["line_long_ratio"] > 0 else 0.0,
        _ratio(ref["line_frag_ratio"], cur["line_frag_ratio"]) if ref["line_frag_ratio"] > 0 else 0.0,
    ])
    space_score = _ratio(ref["largest_clean_space_ratio"], cur["largest_clean_space_ratio"])
    total = 0.35 * tone_score + 0.25 * line_score + 0.25 * edge_score + 0.15 * space_score
    return {
        "tone_score": round(float(tone_score), 3),
        "edge_score": round(float(edge_score), 3),
        "line_score": round(float(line_score), 3),
        "space_score": round(float(space_score), 3),
        "render_score": round(float(total), 3),
    }


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
