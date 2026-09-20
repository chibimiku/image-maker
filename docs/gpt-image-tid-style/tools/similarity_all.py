# -*- coding: utf-8 -*-
"""合并两轮产物，统一算「画风贴近度」（CLIP + HSV + 线条统计），输出到 CSV/文本。"""
import glob
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tests"))

import calc_style_similarity as css  # noqa: E402

REF = os.path.join(BASE, "data", "style-ref", "tid.png")

files = []
for d in sorted(glob.glob(os.path.join(BASE, "data", "*", "gpt-image-tid"))):
    files += sorted(glob.glob(os.path.join(d, "tid-*_output_*.png")))

use_clip = css.load_clip()
ref_hsv = css.hsv_hist(REF)
ref_clip = css.clip_embedding(REF) if use_clip else None
ref_density, ref_warmth, ref_light = css.edge_stats(REF)
ref_lum = css.lum_contrast(REF)

rows = []
for p in files:
    label = os.path.basename(p)
    variant = label.split("-")[1].split("_")[0]
    row = {"variant": variant, "file": label}
    if use_clip:
        row["clip"] = round(css.cosine(ref_clip, css.clip_embedding(p)), 4)
    row["hsv"] = round(css.cosine(ref_hsv, css.hsv_hist(p)), 4)
    d, w, l = css.edge_stats(p)
    row["edge_delta"] = round(abs(d - ref_density), 4)
    row["warm_delta"] = round(abs(w - ref_warmth), 4)
    row["light_delta"] = round(abs(l - ref_light), 4)
    row["lum_delta"] = round(abs(css.lum_contrast(p) - ref_lum), 4)
    rows.append(row)

rows.sort(key=lambda r: r.get("clip", 0), reverse=True)
header = ["variant", "clip", "hsv", "edge_delta", "warm_delta", "light_delta", "lum_delta", "file"]
out_lines = [" | ".join(header)]
for r in rows:
    out_lines.append(" | ".join(str(r.get(h, "-")) for h in header))

text = "\n".join(out_lines)
print(text)

out = os.path.join(BASE, "data", "gpt-image-tid", "similarity_all.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print("\nsaved", out)
