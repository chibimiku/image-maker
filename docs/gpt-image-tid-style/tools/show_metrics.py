# -*- coding: utf-8 -*-
"""从 fingerprint.json 打印指定变体的完整指纹（供文档取数）。"""
import json
import os
import sys

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fingerprint.json")
rows = json.load(open(p, encoding="utf-8"))
want = sys.argv[1:] or None
for r in rows:
    if want and r["variant"] not in want:
        continue
    print(f"{r['variant']:24s} clip={r['clip']:.4f} hsv={r['hsv']:.4f} 亮度={r['亮度均值']:6.1f} "
          f"饱和={r['饱和度均值']:6.1f} 白底={r['白底占比>235']:.3f} 边缘密度={r['边缘密度']:.4f} "
          f"边缘差={r['edge_delta']:.4f} 对比差={r['lum_delta']:.4f}")
