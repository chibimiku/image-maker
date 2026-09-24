# -*- coding: utf-8 -*-
"""真实提示词验证集评估：CLIP/HSV 贴近度 + 画风指纹（对所有 tv-* 产物）。"""
import glob
import json
import os
import sys

import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tests"))
import calc_style_similarity as css  # noqa: E402

REF = os.path.join(BASE, "data", "style-ref", "tid.png")
use_clip = css.load_clip()
ref_clip = css.clip_embedding(REF) if use_clip else None
ref_hsv = css.hsv_hist(REF)
ref_d, ref_w, ref_l = css.edge_stats(REF)
ref_lum = css.lum_contrast(REF)


def fp(p):
    img = cv2.imread(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return gray.mean(), hsv[:, :, 1].mean(), (gray > 235).mean(), css.Canny(gray, 50, 150).mean() / 255.0


import glob as _g  # noqa: E402


def canny(gray, a, b):
    return cv2.Canny(gray, a, b)


files = sorted(_g.glob(os.path.join(BASE, "data", "*", "tid-verify", "tv-*output*.png")))
rows = []
for p in files:
    name = os.path.basename(p)
    parts = name.split("-")
    case = parts[1]
    variant = parts[2].split("_")[0]
    img = cv2.imread(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clip = css.cosine(ref_clip, css.clip_embedding(p)) if use_clip else float("nan")
    d, w, l = css.edge_stats(p)
    rows.append({
        "case": case, "variant": variant,
        "clip": round(clip, 4),
        "hsv": round(css.cosine(ref_hsv, css.hsv_hist(p)), 4),
        "bright": round(float(gray.mean()), 1),
        "sat": round(float(hsv[:, :, 1].mean()), 1),
        "white": round(float((gray > 235).mean()), 3),
        "edge": round(float(cv2.Canny(gray, 50, 150).mean() / 255.0), 4),
        "edge_delta": round(abs(d - ref_d), 4),
        "file": name,
    })

rows.sort(key=lambda r: (r["case"], r["variant"]))
print(f"{'case':>12} {'variant':>8} {'clip':>7} {'hsv':>7} {'bright':>7} {'sat':>6} {'white':>6} {'edge':>7} {'edgeΔ':>7}")
for r in rows:
    print(f"{r['case']:>12} {r['variant']:>8} {r['clip']:>7.4f} {r['hsv']:>7.4f} {r['bright']:>7.1f} "
          f"{r['sat']:>6.1f} {r['white']:>6.3f} {r['edge']:>7.4f} {r['edge_delta']:>7.4f}")

# 按变体求均值，看三种方式谁更贴
print("\n=== 按变体均值 ===")
for v in ["V1", "V2", "V3"]:
    sub = [r for r in rows if r["variant"] == v]
    if not sub:
        continue
    n = len(sub)
    print(f"{v}: n={n} clip={sum(r['clip'] for r in sub) / n:.4f} hsv={sum(r['hsv'] for r in sub) / n:.4f} "
          f"bright={sum(r['bright'] for r in sub) / n:.1f} sat={sum(r['sat'] for r in sub) / n:.1f} "
          f"white={sum(r['white'] for r in sub) / n:.3f} edgeΔ={sum(r['edge_delta'] for r in sub) / n:.4f}")

out = os.path.join(BASE, "docs", "gpt-image-tid-style", "verify_real_prompts.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\nsaved", out)
print("参考图 tid.png: 亮度 221.5 饱和 25.0 白底 0.667 边缘密度 0.0663")
