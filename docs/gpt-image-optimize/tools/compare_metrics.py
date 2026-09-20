# -*- coding: utf-8 -*-
"""线条连续性与细节清晰度量化对比（临时对比工具，非仓库代码）。

用法：
  python compare_metrics.py <img2.png> <flare.png> <sunburst.png> --out <报告路径>

指标：
  lap_var        拉普拉斯方差，越高=整体锐度/细节能量越强
  edge_density   Canny 边缘像素占比
  frag_ratio     边缘碎片化率：小连通域(<8px)占边缘比例，越低=线条越连续
  mean_comp      边缘连通域平均长度(px)，越高=线条越长越连贯
  iso_ratio      孤立边缘像素比例(3x3邻域内边缘邻居<=1)，越低越好
  hf_ratio       高频能量占比(FFT)，越高=细节纹理越丰富
  block_ratio    8px 网格块边界跳变占比，越低=越少压缩/网格伪影
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def load_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def edge_continuity(gray):
    edges = cv2.Canny(gray, 80, 200, L2gradient=True)
    total = int(edges.sum() // 255)
    if total == 0:
        return dict(edge_density=0.0, frag_ratio=1.0, mean_comp=0.0, iso_ratio=1.0, n_comp=0)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([], dtype=np.int32)
    small = int((sizes < 8).sum())
    frag_ratio = float(sizes[sizes < 8].sum() / total) if sizes.size else 0.0
    binary = (edges > 0).astype(np.uint8)
    kern = np.ones((3, 3), np.uint8)
    kern[1, 1] = 0
    neigh = cv2.filter2D(binary, -1, kern, borderType=cv2.BORDER_CONSTANT)
    iso_ratio = float(((binary == 1) & (neigh <= 1)).sum() / total)
    return dict(
        edge_density=float(total / edges.size),
        frag_ratio=frag_ratio,
        mean_comp=float(sizes.mean()) if sizes.size else 0.0,
        iso_ratio=iso_ratio,
        n_comp=int(max(num - 1, 0)),
        _small_comp=small,
    )


def hf_energy(gray):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / min(cy, cx)
    low = mag[r < 0.15].sum()
    mid = mag[(r >= 0.15) & (r < 0.4)].sum()
    high = mag[r >= 0.4].sum()
    tot = low + mid + high
    return float(high / tot) if tot else 0.0


def blockiness(gray):
    g = gray.astype(np.float32)
    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))
    bv = dv[:, 7::8].mean() if dv.shape[1] > 8 else 0.0
    nv = np.delete(dv, np.s_[7::8], axis=1).mean() if dv.shape[1] > 8 else 0.0
    bh = dh[7::8, :].mean() if dh.shape[0] > 8 else 0.0
    nh = np.delete(dh, np.s_[7::8], axis=0).mean() if dh.shape[0] > 8 else 0.0
    num = float((bv + bh) / 2)
    den = float((nv + nh) / 2)
    return float(num / den) if den else 0.0


def measure(path):
    img, gray = load_gray(path)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    res = {
        "file": os.path.basename(path),
        "size": [int(img.shape[1]), int(img.shape[0])],
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "tenengrad": float((cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3) ** 2
                            + cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3) ** 2).mean()),
        "hf_ratio": hf_energy(gray),
        "block_ratio": blockiness(gray),
        "file_kb": round(os.path.getsize(path) / 1024, 1),
    }
    res.update(edge_continuity(gray))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="把 JSON 报告写到这个路径")
    args = ap.parse_args()
    rows = [measure(p) for p in args.images]
    keys = ["file", "size", "file_kb", "lap_var", "tenengrad", "edge_density",
            "mean_comp", "n_comp", "frag_ratio", "iso_ratio", "hf_ratio", "block_ratio"]
    print(" | ".join(k for k in keys))
    for r in rows:
        print(" | ".join(str(r.get(k)) if k != "lap_var" else f"{r[k]:.1f}" for k in keys))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
