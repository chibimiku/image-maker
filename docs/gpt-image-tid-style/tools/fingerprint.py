# -*- coding: utf-8 -*-
"""一次性：统计每组产物的「画风指纹」指标（亮度/饱和度/边缘密度/亮部占比/色彩熵）。

用来量化 tid 画风是否被 gpt-image-2 吸收：tid 参考图是明亮、低饱和、高白底占比、
柔和细线的水彩风，所以「明亮度↑、白底占比↑、边缘密度↓、饱和度↓」= 更贴近。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(BASE, "data", "style-ref", "tid.png")
OUT_DIRS = [d for d in glob.glob(os.path.join(BASE, "data", "*", "gpt-image-tid"))
            if os.path.isdir(d)]


def fingerprint(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    bright = (gray > 235).mean()
    mid = (gray > 200).mean()
    # 颜色丰富度：色相直方图熵
    hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "file": os.path.basename(path),
        "size": f"{w}x{h}",
        "亮度均值": round(float(gray.mean()), 1),
        "亮度标准差": round(float(gray.std()), 1),
        "饱和度均值": round(float(sat.mean()), 1),
        "值(V)均值": round(float(val.mean()), 1),
        "白底占比>235": round(float(bright), 4),
        "浅底占比>200": round(float(mid), 4),
        "边缘密度": round(float(edges.mean() / 255.0), 4),
        "色相熵": round(entropy, 3),
    }


rows = []
ref_fp = fingerprint(REF)
ref_fp["组"] = "REF tid"
ref_fp["file"] = os.path.basename(REF)
rows.append(ref_fp)

for d in sorted(OUT_DIRS):
    for path in sorted(glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        fp = fingerprint(path)
        if fp:
            fp["组"] = os.path.basename(path).split("-")[1].split("_")[0] if "-" in os.path.basename(path) else "?"
            rows.append(fp)

if not rows:
    print("没有找到产物")
    sys.exit(0)

header = ["组", "size", "亮度均值", "亮度标准差", "饱和度均值", "白底占比>235", "浅底占比>200", "边缘密度", "色相熵"]
print(" | ".join(f"{h}" for h in header))
print("-" * 110)
for r in rows:
    print(" | ".join(str(r.get(h, "")) for h in header))

out = os.path.join(BASE, "data", "gpt-image-tid", "fingerprint.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\nsaved", out)
