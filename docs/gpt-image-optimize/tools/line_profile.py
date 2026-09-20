# -*- coding: utf-8 -*-
"""线条锐利度/连续性剖面：梯度幅值分布 + 细线连续性（脊线分量的连通性）。"""
import os
import sys
import cv2
import numpy as np


def analysis(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    g8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = g8.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-9)
    p50, p90, p99 = np.percentile(mag, [50, 90, 99])
    # 细线连续性：非极大值抑制后取强边缘骨架，统计其连通域平均长度
    thin = np.zeros_like(g8)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    ys, xs = np.nonzero(mag > 0.25)
    keep = 0
    for y, x in zip(ys, xs):
        a = ang[y, x] % 180
        dy, dx = (0, 1) if (a < 22.5 or a >= 157.5) else ((1, 0) if 67.5 <= a < 112.5 else (1, 1))
        v = mag[y, x]
        if v >= mag[min(y + dy, mag.shape[0] - 1), min(x + dx, mag.shape[1] - 1)] and \
           v >= mag[max(y - dy, 0), max(x - dx, 0)]:
            thin[y, x] = 255
            keep += 1
    num, _, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([1])
    return dict(
        file=os.path.basename(path),
        p50=p50, p90=p90, p99=p99,
        frac_gt_20=float((mag > 0.20).mean()),
        frac_gt_30=float((mag > 0.30).mean()),
        thin_mean_len=float(sizes.mean()),
        thin_median_len=float(np.median(sizes)),
        thin_frag=float((sizes < 5).sum() / max(sizes.size, 1)),
        thin_max_len=int(sizes.max()),
    )


if __name__ == "__main__":
    paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    out = None
    for token in sys.argv[1:]:
        if token.startswith("--out="):
            out = token.split("=", 1)[1]
    rows = [analysis(p) for p in paths]
    head = ["file", "p50", "p90", "p99", "frac_gt_20", "frac_gt_30",
            "thin_mean_len", "thin_median_len", "thin_frag", "thin_max_len"]
    lines = [" | ".join(head)]
    for r in rows:
        lines.append(" | ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in head))
    report = "\n".join(lines)
    print(report)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
