# -*- coding: utf-8 -*-
"""重绘前后对比：统一缩放到同尺寸，输出并排图 + 局部放大 + 指标。"""
import os
import cv2
import numpy as np


def load(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def label(img, text):
    h, w = img.shape[:2]
    c = np.full((h + 36, w, 3), 24, np.uint8)
    cv2.putText(c, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    c[36:] = img
    return c


def metrics(img, tag):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / mag.max()
    lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    q = (img // 16) * 16
    colors = len(np.unique(q.reshape(-1, 3), axis=0))
    return (f"{tag}: {img.shape[1]}x{img.shape[0]}  lap_var={lap:.0f}  "
            f"边缘>20%={ (mag>0.2).mean()*100:.2f}%  平坦<2%={(mag<0.02).mean()*100:.2f}%  "
            f"渐变2-10%={((mag>=0.02)&(mag<=0.10)).mean()*100:.2f}%  色数={colors}")


def main():
    src, rep, out = None, None, None
    args = {}
    for a in os.sys.argv[1:]:
        k, _, v = a.partition("=")
        args[k.lstrip("-")] = v
    src, rep, out = args["src"], args["rep"], args["out"]
    a, b = load(src), load(rep)
    os.makedirs(out, exist_ok=True)
    H, W = 1536, 1024
    a2 = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
    side = np.hstack([label(a2, "gpt-image-2 original"), label(b2, "gemini-3-pro-image repaint")])
    cv2.imwrite(os.path.join(out, "side_by_side.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # 局部 3x：脸 / 蕾丝 / 树皮水面
    regions = {"face": (0.28, 0.10, 0.40, 0.26), "lace": (0.20, 0.45, 0.52, 0.32),
               "bg": (0.55, 0.08, 0.42, 0.34)}
    for tag, (rx, ry, rw, rh) in regions.items():
        tiles = []
        for img, name in ((a2, "original"), (b2, "repaint")):
            x, y, w, h = int(rx * W), int(ry * H), int(rw * W), int(rh * H)
            crop = img[y:y + h, x:x + w]
            crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
            tiles.append(label(crop, name))
        cv2.imwrite(os.path.join(out, f"crop_{tag}_2x.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(metrics(a, "原始      "))
    print(metrics(b, "Gemini重绘"))
    print("written to", out)


if __name__ == "__main__":
    main()
