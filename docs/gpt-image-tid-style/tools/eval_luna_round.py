# -*- coding: utf-8 -*-
"""评估 Luna 建议轮（data/*/tid-luna/*）：CLIP/HSV + 画风指纹，并与基线 G1 / H1 对比。"""
import glob
import json
import os
import sys

import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tests"))
import calc_style_similarity as css  # noqa: E402

REF = os.path.join(BASE, "data", "style-ref", "tid.png")
TARGETS = sorted(glob.glob(os.path.join(BASE, "data", "*", "tid-luna", "luna-*output*.png")))
BASELINES = {
    "G1_校准版+原图(基线)": os.path.join(BASE, "docs", "gpt-image-tid-style", "images", "tid-G1.png"),
    "A1_全量11k无图(基线)": os.path.join(BASE, "docs", "gpt-image-tid-style", "images", "tid-A1.png"),
}

use_clip = css.load_clip()
ref_clip = css.clip_embedding(REF) if use_clip else None
ref_hsv = css.hsv_hist(REF)
ref_d, ref_w, ref_l = css.edge_stats(REF)
ref_lum = css.lum_contrast(REF)

# 参考图自身「非主体区域」的配色基准：用 crop_left + crop_white 拼出的统计做参照
crop_stats = []
for c in ["crop_white.png", "crop_left.png"]:
    p = os.path.join(BASE, "data", "gpt-image-tid", "style-crops", c)
    if os.path.exists(p):
        g = cv2.imread(p)
        crop_stats.append((cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).mean(), cv2.cvtColor(g, cv2.COLOR_BGR2HSV)[:, :, 1].mean()))
if crop_stats:
    print(f"参考图无主体区域均值: 亮度 {sum(s[0] for s in crop_stats) / len(crop_stats):.1f} "
          f"饱和 {sum(s[1] for s in crop_stats) / len(crop_stats):.1f}")


def fp(p):
    img = cv2.imread(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return gray.mean(), hsv[:, :, 1].mean(), float((gray > 235).mean()), float(cv2.Canny(gray, 50, 150).mean() / 255.0)


rows = []
for p in TARGETS:
    key = os.path.basename(p).split("-")[1].split("_")[0].upper()
    bright, sat, white, edge = fp(p)
    rows.append({
        "variant": key,
        "clip": round(css.cosine(ref_clip, css.clip_embedding(p)) if use_clip else float("nan"), 4),
        "hsv": round(css.cosine(ref_hsv, css.hsv_hist(p)), 4),
        "bright": round(bright, 1), "sat": round(sat, 1), "white": round(white, 3),
        "edge": round(edge, 4), "edge_delta": round(abs(css.edge_stats(p)[0] - ref_d), 4),
        "file": os.path.basename(p),
    })
for label, p in BASELINES.items():
    if not os.path.exists(p):
        continue
    bright, sat, white, edge = fp(p)
    rows.append({
        "variant": label,
        "clip": round(css.cosine(ref_clip, css.clip_embedding(p)) if use_clip else float("nan"), 4),
        "hsv": round(css.cosine(ref_hsv, css.hsv_hist(p)), 4),
        "bright": round(bright, 1), "sat": round(sat, 1), "white": round(white, 3),
        "edge": round(edge, 4), "edge_delta": round(abs(css.edge_stats(p)[0] - ref_d), 4),
        "file": os.path.basename(p),
    })

header = f"{'variant':>22} {'clip':>7} {'hsv':>7} {'bright':>7} {'sat':>6} {'white':>6} {'edge':>7} {'edgeΔ':>7}"
print("\n" + header)
print("-" * len(header))
for r in rows:
    print(f"{r['variant']:>22} {r['clip']:>7.4f} {r['hsv']:>7.4f} {r['bright']:>7.1f} {r['sat']:>6.1f} "
          f"{r['white']:>6.3f} {r['edge']:>7.4f} {r['edge_delta']:>7.4f}   {r['file'][:34]}")
print("\n参考图 tid.png: 亮度 221.5 饱和 25.0 白底 0.667 边缘密度 0.0663")

out = os.path.join(BASE, "docs", "gpt-image-tid-style", "luna_round.json")
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved", out)
