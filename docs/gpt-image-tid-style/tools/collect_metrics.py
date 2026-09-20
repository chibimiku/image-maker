# -*- coding: utf-8 -*-
"""一次性：把两轮产物合并出「画风贴近度 + 画风指纹」两张表，写进 docs/gpt-image-tid-style/。

用法（仓库根目录）：
  & "C:\\Program Files\\Python310\\python.exe" docs/gpt-image-tid-style/tools/collect_metrics.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

DOCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.dirname(DOCS))
sys.path.insert(0, os.path.join(BASE, "tests"))
import calc_style_similarity as css  # noqa: E402

IMAGES = os.path.join(DOCS, "images")
REF = os.path.join(BASE, "data", "style-ref", "tid.png")

# 变体名 -> (画风模式说明, 文本字符数)
CHARGE = {
    "A1": ("off 全量指令（无图）", 11283),
    "A1_run2": ("off 全量指令（无图，复跑）", 11283),
    "A2": ("off 压缩指令（无图）", 1855),
    "B1": ("priority 参考优先", 4195),
    "B2": ("head 头部插入（全量）", 13232),
    "B3": ("interleave 图文交错（全量）", 13232),
    "B4": ("head 头部插入（全量，复跑）", 13232),
    "B5": ("压缩指令 + 参考图", 1855),
    "C": ("自定义短说明 + 参考图", 402),
    "C2": ("自定义短说明 + 参考图（复跑）", 402),
    "D": ("无画风对照", 95),
    "E1": ("仅参考图，无画风字", 95),
    "G1": ("校准版 678 + 参考图", 678),
    "G2": ("校准版 678 + 参考图（2.5-flare）", 678),
    "G3": ("校准版 678 + 参考图（2.5-sunburst）", 678),
    "G4_calibrated_no_ref": ("校准版 678 无参考图", 678),
    "H1_32k": ("31000 字符灌水", 31000),
    "H1_32k_run2": ("31000 字符灌水（复跑）", 31000),
    "H2_40k": ("40000 字符灌水", 40000),
}


def label_of(path):
    name = os.path.basename(path)
    key = name.replace("tid-", "").rsplit("_output", 1)[0]
    return key[:-4] if key.endswith(".png") else key


def fingerprint(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    p = hist / (hist.sum() + 1e-9)
    return {
        "亮度均值": round(float(gray.mean()), 1),
        "亮度标准差": round(float(gray.std()), 1),
        "饱和度均值": round(float(hsv[:, :, 1].mean()), 1),
        "白底占比>235": round(float((gray > 235).mean()), 4),
        "浅底占比>200": round(float((gray > 200).mean()), 4),
        "边缘密度": round(float(edges.mean() / 255.0), 4),
        "色相熵": round(float(-(p * np.log(p + 1e-12)).sum()), 3),
    }


files = sorted(glob.glob(os.path.join(IMAGES, "*.png")))
use_clip = css.load_clip()
ref_clip = css.clip_embedding(REF) if use_clip else None
ref_hsv = css.hsv_hist(REF)
ref_d, ref_w, ref_l = css.edge_stats(REF)
ref_lum = css.lum_contrast(REF)

rows = []
for p in files:
    key = label_of(p)
    mode, chars = CHARGE.get(key, ("?", 0))
    clip = css.cosine(ref_clip, css.clip_embedding(p)) if use_clip else float("nan")
    rows.append({
        "variant": key, "mode": mode, "chars": chars,
        "clip": round(clip, 4),
        "hsv": round(css.cosine(ref_hsv, css.hsv_hist(p)), 4),
        "edge_delta": round(abs(css.edge_stats(p)[0] - ref_d), 4),
        "warm_delta": round(abs(css.edge_stats(p)[1] - ref_w), 4),
        "light_delta": round(abs(css.edge_stats(p)[2] - ref_l), 4),
        "lum_delta": round(abs(css.lum_contrast(p) - ref_lum), 4),
        "file": os.path.basename(p),
        **fingerprint(p),
    })

rows.sort(key=lambda r: r["clip"], reverse=True)

# similarity_all.txt
header = ["variant", "clip", "hsv", "edge_delta", "warm_delta", "light_delta", "lum_delta", "file"]
lines = [" | ".join(header)]
for r in rows:
    lines.append(" | ".join(str(r.get(h, "-")) for h in header))
open(os.path.join(DOCS, "similarity_all.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

# fingerprint.json
with open(os.path.join(DOCS, "fingerprint.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print("\n".join(lines))
print(f"\n参考图 tid.png: {fingerprint(REF)}")
print(f"\n共 {len(rows)} 张产物 -> docs/gpt-image-tid-style/{{similarity_all.txt,fingerprint.json}}")
