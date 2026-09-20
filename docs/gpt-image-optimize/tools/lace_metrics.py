# -*- coding: utf-8 -*-
"""蕾丝/褶皱保真度指标：只看裙摆区域（相对坐标裁剪），量化"细节有没有被合并"。

输出：
  lace_teng     裙摆区 tenengrad 均值（细节能量）
  lace_hard     裙摆区硬边像素占比
  lace_iso      裙摆区孤立边缘占比（线条是否连成结构）
  frill_cycles  沿水平扫描线的平均过零次数（= 裙摆上能数到的边/褶数）
  skirt_teng    裙身区 tenengrad 均值
"""
import os
import sys

import cv2
import numpy as np

TARGET_W = 1024
REG_LACE = (0.18, 0.44, 0.34, 0.20)   # 裙摆荷叶边
REG_SKIRT = (0.14, 0.34, 0.48, 0.24)  # 裙身整体


def prep(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if w != TARGET_W:
        img = cv2.resize(img, (TARGET_W, int(round(h * TARGET_W / w))), interpolation=cv2.INTER_AREA)
    return img


def crop(img, reg):
    H, W = img.shape[:2]
    x, y = int(reg[0] * W), int(reg[1] * H)
    w, h = int(reg[2] * W), int(reg[3] * H)
    return img[y:y + h, x:x + w]


def metrics(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    teng = float((gx ** 2 + gy ** 2).mean())
    mag = np.sqrt(gx * gx + gy * gy)
    mag /= (mag.max() + 1e-9)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 200, L2gradient=True)
    total = int(edges.sum() // 255)
    if total:
        binary = (edges > 0).astype(np.uint8)
        kern = np.ones((3, 3), np.uint8)
        kern[1, 1] = 0
        neigh = cv2.filter2D(binary, -1, kern, borderType=cv2.BORDER_CONSTANT)
        iso = float(((binary == 1) & (neigh <= 1)).sum() / total)
    else:
        iso = 1.0
    # 沿水平线数"边"：把每行灰度做差分后数符号变化次数，再取中位数
    rows = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    cycles = []
    for r in range(0, rows.shape[0], 4):
        line = cv2.GaussianBlur(rows[r].reshape(1, -1), (1, 5), 0).ravel()
        d = np.diff(line)
        cycles.append(int((np.diff(np.sign(d)) != 0).sum()) / max(1, len(d)))
    return {"teng": teng, "hard": float((mag > 0.20).mean()), "iso": iso,
            "cycles": float(np.median(cycles)) if cycles else 0.0}


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    head = f"{'样本':34s}{'lace_teng':>12s}{'lace_hard':>11s}{'lace_iso':>10s}{'frill_cycles':>13s}{'skirt_teng':>12s}"
    lines = [head]
    for p in paths:
        img = prep(p)
        l = metrics(crop(img, REG_LACE))
        s = metrics(crop(img, REG_SKIRT))
        row = (f"{os.path.basename(p)[:32]:34s}{l['teng']:12.0f}{l['hard']:11.4f}"
               f"{l['iso']:10.4f}{l['cycles']:13.4f}{s['teng']:12.0f}")
        lines.append(row)
    report = "\n".join(lines)
    print(report)
    if out:
        open(out, "w", encoding="utf-8").write(report + "\n")


if __name__ == "__main__":
    main()
