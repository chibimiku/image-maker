# -*- coding: utf-8 -*-
"""
画风贴近度算法计算：
1. CLIP 图像向量余弦相似度（openai/clip-vit-base-patch32，优先）——
   所有生成图使用同一正文 prompt（内容大致相同），因此该相似度主要反映画风/配色/构成差异。
2. HSV 色彩直方图余弦相似度（纯配色，与内容无关）。
3. 线条/渲染统计：边缘密度、边缘线色调（彩色线 vs 黑线）、亮度对比度。

用法：
  python tests/calc_style_similarity.py --ref-image d:\\sayhana-test.png \
      --images A.png B.png C.png D.png E.png
"""
import argparse
import os
import sys
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


# ---------------- CLIP 相似度（可选，失败则跳过） ----------------
_clip_model = None
_clip_processor = None


def load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return True
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        print(f"[info] 加载 CLIP 模型 {model_name} ...")
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model.eval()
        return True
    except Exception as e:
        print(f"[warn] CLIP 不可用（跳过该项）: {e}")
        return False


def clip_embedding(path):
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    inputs = _clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = _clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).numpy()


# ---------------- 风格统计特征 ----------------
def hsv_hist(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def edge_stats(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = float(edges.mean() / 255.0)
    # 边缘像素处的原始颜色（判断线条是彩色还是黑线）
    mask = edges > 0
    if mask.sum() == 0:
        return density, 0.5, 0.5
    bgr = img[mask]
    r, g, b = bgr[:, 2].mean(), bgr[:, 1].mean(), bgr[:, 0].mean()
    warmth = float(r / (r + b + 1e-6))          # >0.5 偏暖线，~0.5 中性
    lightness = float((r + g + b) / (3 * 255))  # 线色亮度（高=浅色线/低对比线）
    return density, warmth, lightness


def lum_contrast(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return float(gray.std())


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    parser = argparse.ArgumentParser(description="画风贴近度计算")
    parser.add_argument("--ref-image", required=True, help="参考图路径（画风基准）")
    parser.add_argument("--images", nargs="+", required=True, help="待对比的生成图路径")
    args = parser.parse_args()

    use_clip = load_clip()

    ref_hsv = hsv_hist(args.ref_image)
    ref_density, ref_warmth, ref_light = edge_stats(args.ref_image)
    ref_lum = lum_contrast(args.ref_image)
    ref_clip = clip_embedding(args.ref_image) if use_clip else None

    print(f"\n参考图: {args.ref_image}")
    print(f"  边缘密度={ref_density:.3f} 线色温暖度={ref_warmth:.3f} 线色亮度={ref_light:.3f} 亮度对比={ref_lum:.3f}")

    rows = []
    for p in args.images:
        name = os.path.basename(p).split("_")[0]
        row = {"组": name, "文件": os.path.basename(p)}
        if use_clip:
            row["CLIP相似度"] = cosine(ref_clip, clip_embedding(p))
        row["HSV配色相似度"] = cosine(ref_hsv, hsv_hist(p))
        d, w, l = edge_stats(p)
        row["边缘密度"] = d
        row["线色温暖度"] = w
        row["线色亮度"] = l
        row["亮度对比"] = lum_contrast(p)
        row["密度差"] = abs(d - ref_density)
        row["温暖差"] = abs(w - ref_warmth)
        row["亮度差"] = abs(l - ref_light)
        rows.append(row)

    # 排序键：CLIP 优先，否则用 HSV + 三个线条/对比差的负值组合
    if use_clip:
        rows.sort(key=lambda r: r["CLIP相似度"], reverse=True)
    else:
        rows.sort(key=lambda r: r["HSV配色相似度"] - 0.2 * (r["密度差"] + r["温暖差"] + r["亮度差"]), reverse=True)

    print("\n=== 贴近度排序（越高越接近参考图画风）===")
    header = ["组", "CLIP相似度", "HSV配色相似度", "边缘密度差", "线色温暖差", "线色亮度差", "亮度对比差"]
    print(" | ".join(f"{h:>10}" for h in header))
    for r in rows:
        clip_s = f"{r.get('CLIP相似度', 0):.4f}" if use_clip else "-"
        hsv_s = f"{r['HSV配色相似度']:.4f}"
        vals = [r["组"], clip_s, hsv_s,
                f"{r['密度差']:.4f}", f"{r['温暖差']:.4f}", f"{r['亮度差']:.4f}",
                f"{abs(r['亮度对比'] - ref_lum):.4f}"]
        print(" | ".join(f"{v:>10}" for v in vals) + "   " + r["文件"])


if __name__ == "__main__":
    main()
