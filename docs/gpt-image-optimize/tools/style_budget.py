# -*- coding: utf-8 -*-
"""线条预算 / 色彩预算分析：量化「平涂线稿」与「高细节厚涂」的差别。

对每张图统计：
  flat_ratio   局部梯度<0.02 的平坦像素占比（平涂色块比例，越高越"干净"）
  smooth_ratio 梯度 0.02~0.10（渐变/光斑过渡）
  edge_ratio   梯度 >0.20（硬边/线条）
  thin_frag / thin_mean_len  细线骨架碎片率 / 平均长度
  colors1k     色彩量化后使用的颜色数（16 灰阶级 × 3 通道）
  p50/p90/p99  梯度幅值分位
  comp_ratio   画面主体占比proxy：中心 50% 区域内的强边像素 / 全图强边像素
"""
import os
import sys
import cv2
import numpy as np


def analyze(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-9)
    total = mag.size
    p50, p90, p99 = np.percentile(mag, [50, 90, 99])

    thin = np.zeros(g.shape, np.uint8)
    strong = mag > 0.25
    ys, xs = np.nonzero(strong)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    for y, x in zip(ys, xs):
        a = ang[y, x] % 180
        if a < 22.5 or a >= 157.5:
            dy, dx = 0, 1
        elif 67.5 <= a < 112.5:
            dy, dx = 1, 0
        else:
            dy, dx = 1, 1
        v = mag[y, x]
        y1, y0 = min(y + dy, g.shape[0] - 1), max(y - dy, 0)
        x1, x0 = min(x + dx, g.shape[1] - 1), max(x - dx, 0)
        if v >= mag[y1, x1] and v >= mag[y0, x0]:
            thin[y, x] = 255
    num, _, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([1])

    q = (img // 16) * 16
    colors = len(np.unique(q.reshape(-1, 3), axis=0))
    h, w = g.shape
    inner = strong[h // 4:3 * h // 4, w // 4:3 * w // 4].sum()
    return dict(
        file=os.path.basename(path),
        kb=round(os.path.getsize(path) / 1024, 1),
        flat_ratio=float((mag < 0.02).mean()),
        smooth_ratio=float(((mag >= 0.02) & (mag <= 0.10)).mean()),
        edge_ratio=float((mag > 0.20).mean()),
        p50=p50, p90=p90, p99=p99,
        thin_frag=float((sizes < 5).sum() / max(sizes.size, 1)),
        thin_mean=float(sizes.mean()),
        colors=colors,
        center_edge_share=float(inner / max(strong.sum(), 1)),
    )


if __name__ == "__main__":
    paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    out = None
    for t in sys.argv[1:]:
        if t.startswith("--out="):
            out = t.split("=", 1)[1]
    rows = [analyze(p) for p in paths]
    head = ["file", "kb", "flat_ratio", "smooth_ratio", "edge_ratio", "p50",
            "p90", "p99", "thin_frag", "thin_mean", "colors", "center_edge_share"]
    lines = [" | ".join(head)]
    for r in rows:
        lines.append(" | ".join(
            f"{r[k]:.4f}" if isinstance(r[k], float) and k not in ("kb",) else str(r[k])
            for k in head))
    report = "\n".join(lines)
    print(report)
    if out:
        open(out, "w", encoding="utf-8").write(report + "\n")
