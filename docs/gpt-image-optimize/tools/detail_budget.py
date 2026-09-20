# -*- coding: utf-8 -*-
"""画面细节预算对比：同样 1024x1536 画布上，"需要刻画的东西"有多少。"""
import os
import cv2
import numpy as np

D = r"D:\code\image-maker\data\20260919"
CASES = [
    (r"anime_lolita_cmp\gpt-image-2_output_192911_0_ddd904.png", "192911 平涂线稿"),
    (r"anime_lolita_cmp_v2\gpt-image-2_output_192922_0_b000d3.png", "v2 高细节满构图"),
    (r"anime_lolita_cmp_v3\gpt-image-2_output_193138_0_d09c70.png", "v3 高细节满构图"),
    (r"anime_lolita_cmp_v3\gpt-image-2.5-flare_output_193215_0_75fc03.png", "v3 flare 高细节"),
]

print("样本                   强边(>20%)   强边像素数    平坦( <2%)   渐变(2-10%)   量化色数")
for rel, name in CASES:
    img = cv2.imdecode(np.fromfile(os.path.join(D, rel), np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-9)
    strong = (mag > 0.20).mean()
    px = strong * mag.size
    flat = (mag < 0.02).mean()
    smooth = ((mag >= 0.02) & (mag <= 0.10)).mean()
    q = (img // 16) * 16
    colors = len(np.unique(q.reshape(-1, 3), axis=0))
    print(f"{name:22s} {strong*100:9.2f}% {px:12.0f} {flat*100:11.2f}% {smooth*100:11.2f}% {colors:9d}")
