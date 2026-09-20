# -*- coding: utf-8 -*-
"""区域对比接触表：把每个 variant 的同一区域裁出来横排，用于评估手/发丝/蕾丝的修复效果。

用法：
  python crop_inspect.py --out <目录> --regions hands,hairr,face,lace --label 名=路径 ...
"""
import argparse
import os

import cv2
import numpy as np

# 相对 v3 源图（1024x1536）标定的区域，重绘图为 2:3 同构图，比例通用
REGIONS = {
    "handL": (0.70, 0.48, 0.24, 0.20),   # 撑在树枝上的左手
    "handL2": (0.71, 0.46, 0.13, 0.12),  # 左手手指数数用（更紧）
    "handR": (0.27, 0.14, 0.22, 0.17),   # 扶脸/托腮的右手
    "hairr": (0.62, 0.22, 0.30, 0.30),   # 画面右侧长发
    "face": (0.30, 0.08, 0.30, 0.20),    # 脸
    "lace": (0.16, 0.42, 0.42, 0.24),    # 裙摆蕾丝
    "hem": (0.20, 0.44, 0.34, 0.18),     # 裙摆荷叶边/花边（紧）
    "chest": (0.34, 0.20, 0.26, 0.18),   # 胸前蕾丝与蝴蝶结
    "skirt": (0.14, 0.36, 0.46, 0.22),   # 裙身整体层次
    "branch": (0.10, 0.58, 0.30, 0.28),  # 苔藓树枝
}


def load(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def label(img, text, size=0.8):
    h, w = img.shape[:2]
    c = np.full((h + 34, w, 3), 24, np.uint8)
    cv2.putText(c, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA)
    c[34:] = img
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", default="hands,hairr,face,lace")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--items", nargs="+", required=True, help="名=路径")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = []
    for it in a.items:
        name, _, path = it.partition("=")
        items.append((name, load(path)))
    for region in a.regions.split(","):
        rx, ry, rw, rh = REGIONS[region]
        tiles = []
        for name, img in items:
            H, W = img.shape[:2]
            x, y = int(rx * W), int(ry * H)
            w, h = int(rw * W), int(rh * H)
            crop = img[y:y + h, x:x + w]
            # 统一到相同高度，保证 hstack 可行（不同分辨率的重绘图都能比）
            target_h = int(h * a.scale)
            target_w = max(1, int(round(crop.shape[1] * target_h / crop.shape[0])))
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            tiles.append(label(crop, name, 0.8))
        target = max(t.shape[0] for t in tiles)
        tiles = [cv2.resize(t, (max(1, int(round(t.shape[1] * target / t.shape[0]))), target),
                            interpolation=cv2.INTER_AREA) if t.shape[0] != target else t for t in tiles]
        sheet = np.hstack(tiles)
        # 限制最长边，避免读图工具 8192px 上限
        max_side = 7600
        if max(sheet.shape[:2]) > max_side:
            k = max_side / max(sheet.shape[:2])
            sheet = cv2.resize(sheet, (int(sheet.shape[1] * k), int(sheet.shape[0] * k)), interpolation=cv2.INTER_AREA)
        out = os.path.join(a.out, f"inspect_{region}.jpg")
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("written", out, sheet.shape)


if __name__ == "__main__":
    main()
