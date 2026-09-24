# -*- coding: utf-8 -*-
"""面部/眼睛区域的画风对比 v2：人脸检测不可靠时用"上中部构图区"兜底。

用法与 v1 相同；v2 的改进：
- 人脸检测失败或检出的框过小/过大时，改用上中部区域（头部通常位于上 1/3 中部）
- 统一裁到相同尺寸后再比，避免尺寸差污染指标
- 输出对比拼图，便于肉眼确认（v1 的拼图在检测失败时会裁到身体，已修）
"""
import argparse
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def head_region(img, size=288):
    """返回头部/面部区域：优先人脸检测，失败则取上中部（半身/胸像构图的头所在位置）。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.08, 5, minSize=(int(h * 0.08), int(h * 0.08)))
    box = None
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # 合理的头部尺寸应占画面高度 8%~45%，否则视为误检
        if 0.06 <= fh / h <= 0.5 and fw / w <= 0.6:
            box = (x, y, fw, fh)
    if box is None:
        box = (int(w * 0.28), int(h * 0.03), int(w * 0.44), int(h * 0.30))
    x, y, bw, bh = box
    px, py = int(bw * 0.45), int(bh * 0.45)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img[: int(h * 0.4), :]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def edge_metrics(gray):
    edges = cv2.Canny(gray, 50, 150)
    n, _l, stats, _c = cv2.connectedComponentsWithStats(edges.astype(np.uint8), connectivity=8)
    lens = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]
    avg = float(np.mean(lens)) if lens else 0.0
    long_ratio = float(np.sum([x for x in lens if x >= 40]) / (sum(lens) + 1e-9))
    fine = cv2.Canny(gray, 50, 150).mean()
    coarse = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 2.0), 50, 150).mean()
    return {
        "edge_density": round(float(edges.mean() / 255.0), 4),
        "edge_fine_share": round(float(fine / (fine + coarse + 1e-9)), 3),
        "edge_avg_seg": round(avg, 1),
        "edge_long_ratio": round(long_ratio, 3),
    }


def face_stats(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    top = gray[: int(gray.shape[0] * 0.6), :]
    eye_edges = cv2.Canny(top, 50, 150)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    out = {
        "gray": gray,
        "hue_hist": hist,
        "eye_edge_density": round(float(eye_edges.mean() / 255.0), 4),
        "dark_ratio_80": round(float((gray < 80).mean()), 4),
        "brightness": round(float(gray.mean()), 1),
        "saturation": round(float(hsv[:, :, 1].mean()), 1),
    }
    out.update(edge_metrics(gray))
    return out


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load(p):
    """支持中文/空格路径。"""
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description="头部/面部区域的画风对比 v2")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out-dir", default=os.path.join(BASE, "cache", "temp", "face-compare2"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ref_img = _load(args.ref)
    if ref_img is None:
        print("[error] 读不到", args.ref)
        return 1
    ref_face = head_region(ref_img)
    ref_m = face_stats(ref_face)
    cv2.imwrite(os.path.join(args.out_dir, "ref_head.png"), ref_face)

    header = (f"{'图':>34} {'MAE↓':>7} {'配色相似↑':>9} {'边缘密度':>8} {'细/粗比':>8} "
              f"{'平均段长↑':>9} {'长线占比↑':>9} {'眼区边缘':>8} {'亮部均值':>8} {'饱和':>6}")
    print(f"\n基准: {os.path.basename(args.ref)}  头部区 亮度 {ref_m['brightness']:.1f} / 边缘 {ref_m['edge_density']:.4f}")
    print(header)
    print("-" * len(header))
    tiles = [("REF", ref_face)]
    for p in args.images:
        img = _load(p)
        if img is None:
            print(f"{os.path.basename(p):>34}  读不到")
            continue
        face = head_region(img)
        m = face_stats(face)
        mae = float(np.abs(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
                           - ref_m["gray"].astype(np.float32)).mean())
        name = os.path.basename(p)
        print(f"{name[:34]:>34} {mae:>7.1f} {cosine(ref_m['hue_hist'], m['hue_hist']):>9.4f} "
              f"{m['edge_density']:>8.4f} {m['edge_fine_share']:>8.3f} {m['edge_avg_seg']:>9.1f} "
              f"{m['edge_long_ratio']:>9.3f} {m['eye_edge_density']:>8.4f} {m['brightness']:>8.1f} {m['saturation']:>6.1f}")
        cv2.imwrite(os.path.join(args.out_dir, f"{os.path.splitext(name)[0]}_head.png"), face)
        tiles.append((name[:16], face))

    pad, tile = 8, 288
    contact = np.full((tile + 34 + pad * 2, (tile + pad) * len(tiles) + pad, 3), 255, np.uint8)
    for i, (label, face) in enumerate(tiles):
        x = pad + i * (tile + pad)
        contact[pad:pad + tile, x:x + tile] = face
        cv2.putText(contact, label, (x, tile + pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    out = os.path.join(args.out_dir, "heads_contact.png")
    cv2.imwrite(out, contact)
    print(f"\n头部对比拼图: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
