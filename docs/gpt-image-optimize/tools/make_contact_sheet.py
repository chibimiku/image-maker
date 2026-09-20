# -*- coding: utf-8 -*-
"""生成三联对比图：原图缩略 + 放大局部裁切（脸/发/蕾丝）。"""
import os
import sys
import cv2
import numpy as np

def load(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def label(img, text, scale=1.0):
    h, w = img.shape[:2]
    canvas = np.zeros((h + 34, w, 3), np.uint8)
    canvas[:] = (28, 28, 28)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas[34:] = img
    return canvas

def main():
    out_dir = sys.argv[1]
    paths = sys.argv[2:]
    imgs = [load(p) for p in paths]
    names = [os.path.basename(p).split("_output")[0] for p in paths]
    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    thumbs = [cv2.resize(i, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for i in imgs]
    os.makedirs(out_dir, exist_ok=True)
    full = np.hstack([label(t, n) for t, n in zip(thumbs, names)])
    cv2.imwrite(os.path.join(out_dir, "cmp_full_three_up.png"), full)
    # 局部放大：四宫格裁切（左上叶、中上脸、中下蕾丝、右下裙摆）
    regions = {
        "crop_face": (0.30, 0.10, 0.45, 0.30),
        "crop_lace": (0.25, 0.45, 0.50, 0.35),
        "crop_hair": (0.10, 0.05, 0.30, 0.30),
        "crop_leaf": (0.60, 0.05, 0.35, 0.35),
    }
    for tag, (rx, ry, rw, rh) in regions.items():
        tiles = []
        for img, name in zip(imgs, names):
            H, W = img.shape[:2]
            x, y = int(rx * W), int(ry * H)
            cw, ch = int(rw * W), int(rh * H)
            crop = img[y:y + ch, x:x + cw]
            crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
            tiles.append(label(crop, name))
        cv2.imwrite(os.path.join(out_dir, f"cmp_{tag}_3x.png"), np.hstack(tiles))
    print("contact sheets written to", out_dir)

if __name__ == "__main__":
    main()
