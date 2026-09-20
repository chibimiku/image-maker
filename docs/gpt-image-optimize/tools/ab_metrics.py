# -*- coding: utf-8 -*-
"""A/B 实验统一指标：全部缩放到同一宽度再测，保证可比。

输出列：
  lap_var      拉普拉斯方差（锐度）
  hard_edge    硬边像素占比（梯度>20%）
  soft         半软过渡占比（梯度 2%~10%）—— 越低越"靠线定义"
  flat         平坦占比（<2%）
  edge_soft    硬边/半软 比值 —— 越高越"线稿化"
  frag         细线骨架碎片率（<5px 连通域占比）—— 越低线条越连续
  iso          孤立边缘像素占比 —— 越低越连续
  colors       量化色数 /1000
"""
import os
import sys

import cv2
import numpy as np

TARGET_W = 1024


def metrics(path, target_w=TARGET_W):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(path)
    h, w = img.shape[:2]
    if w != target_w:
        img = cv2.resize(img, (target_w, int(round(h * target_w / w))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag /= (mag.max() + 1e-9)

    hard = float((mag > 0.20).mean())
    soft = float(((mag >= 0.02) & (mag <= 0.10)).mean())
    flat = float((mag < 0.02).mean())

    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 200, L2gradient=True)
    total = int(edges.sum() // 255)
    if total:
        num, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        sizes = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([1])
        frag = float((sizes < 8).sum() / max(sizes.size, 1))
        binary = (edges > 0).astype(np.uint8)
        kern = np.ones((3, 3), np.uint8)
        kern[1, 1] = 0
        neigh = cv2.filter2D(binary, -1, kern, borderType=cv2.BORDER_CONSTANT)
        iso = float(((binary == 1) & (neigh <= 1)).sum() / total)
    else:
        frag, iso = 1.0, 1.0

    q = (img // 16) * 16
    colors = len(np.unique(q.reshape(-1, 3), axis=0))

    return {
        "lap_var": float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
        "hard_edge": hard,
        "soft": soft,
        "flat": flat,
        "edge_soft": hard / soft if soft else 0.0,
        "frag": frag,
        "iso": iso,
        "colors": colors,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
    names = [os.path.basename(p) for p in args]
    rows = [metrics(p) for p in args]
    keys = ["lap_var", "hard_edge", "soft", "flat", "edge_soft", "frag", "iso", "colors"]
    width = max(len(n) for n in names) + 2
    print("样本".ljust(width) + "".join(k.rjust(12) for k in keys))
    for n, r in zip(names, rows):
        print(n.ljust(width) + "".join(
            (f"{r[k]:.4f}" if k not in ("lap_var", "colors") else f"{r[k]:.0f}").rjust(12) for k in keys))
    if out:
        import json
        json.dump([{"name": n, **r} for n, r in zip(names, rows)],
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
