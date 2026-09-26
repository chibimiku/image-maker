# -*- coding: utf-8 -*-
"""Content-resistant art-style similarity evaluation.

The final score deliberately does not use CLIP. Global CLIP image similarity is
useful context, but when reference and generated images contain different
subjects it rewards shared semantics and composition too strongly. The style
score combines:

* ResNet multi-layer Gram statistics (55%): texture, brush/edge and rendering
  correlations with most spatial/content layout discarded.
* HSV histogram (20%): palette and value distribution.
* Handcrafted render signature (25%): edge scale/direction, line colour,
  contrast and saturation.

CLIP is reported as a diagnostic only. ORB copy risk is also reported
separately; literal reference copying is a failure mode, not improved style.
"""
import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

_clip_model = None
_clip_processor = None
_resnet_model = None
_resnet_features = None


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return True
    try:
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        print(f"[info] loading diagnostic CLIP model {model_name} ...")
        _clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        _clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        _clip_model.eval()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] CLIP unavailable; diagnostic skipped: {exc}")
        return False


def clip_embedding(path):
    import torch
    from PIL import Image
    image = Image.open(path).convert("RGB")
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = _clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy()


def load_resnet_style_model():
    global _resnet_model, _resnet_features
    if _resnet_model is not None:
        return True
    try:
        from torchvision.models import ResNet50_Weights, resnet50
        model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        features = {}

        def capture(name):
            def hook(_module, _inputs, output):
                features[name] = output.detach()
            return hook

        model.relu.register_forward_hook(capture("stem"))
        model.layer1.register_forward_hook(capture("layer1"))
        model.layer2.register_forward_hook(capture("layer2"))
        _resnet_model = model
        _resnet_features = features
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ResNet style backbone unavailable: {exc}")
        return False


def resnet_style_signature(path):
    """Return spatially invariant channel correlations from early/mid layers."""
    import torch
    from PIL import Image
    from torchvision.models import ResNet50_Weights

    image = Image.open(path).convert("RGB")
    tensor = ResNet50_Weights.DEFAULT.transforms()(image).unsqueeze(0)
    _resnet_features.clear()
    with torch.no_grad():
        _resnet_model(tensor)
    signature = {}
    for name in ("stem", "layer1", "layer2"):
        feat = _resnet_features[name].squeeze(0).float()
        channels = feat.shape[0]
        flat = feat.reshape(channels, -1)
        flat = flat - flat.mean(dim=1, keepdim=True)
        flat = flat / (flat.std(dim=1, keepdim=True) + 1e-6)
        gram = flat @ flat.t() / max(1, flat.shape[1])
        upper = torch.triu_indices(channels, channels)
        gram_vec = gram[upper[0], upper[1]]
        raw = feat.reshape(channels, -1)
        moments = torch.cat((raw.mean(dim=1), raw.std(dim=1)))
        signature[name] = torch.cat((gram_vec, moments)).cpu().numpy()
    return signature


def resnet_style_similarity(ref, cur):
    layer_weights = {"stem": 0.25, "layer1": 0.35, "layer2": 0.40}
    return float(sum(layer_weights[k] * cosine(ref[k], cur[k]) for k in layer_weights))


def hsv_hist(path):
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 12, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def render_signature(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    values = []
    orientation = []
    for sigma in (0.0, 1.0, 2.0):
        src = gray if sigma == 0 else cv2.GaussianBlur(gray, (0, 0), sigma)
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        threshold = np.percentile(magnitude, 72)
        mask = magnitude >= threshold
        hist, _ = np.histogram(angle[mask] % 180.0, bins=12, range=(0, 180), weights=magnitude[mask])
        hist = hist.astype(np.float64)
        orientation.extend(hist / (hist.sum() + 1e-9))
        values.append(float((cv2.Canny(src, 50, 150) > 0).mean()))
    edge_mask = cv2.Canny(gray, 50, 150) > 0
    edge_pixels = image[edge_mask] if edge_mask.any() else image.reshape(-1, 3)
    b, g, r = (float(edge_pixels[:, i].mean()) for i in range(3))
    values.extend([
        r / (r + b + 1e-6),
        (r + g + b) / (3 * 255.0),
        float(gray.std() / 128.0),
        float(hsv[:, :, 1].mean() / 255.0),
        float(np.percentile(gray, 90) - np.percentile(gray, 10)) / 255.0,
    ])
    return {"orientation": np.array(orientation), "values": np.array(values)}


def _relative_closeness(a, b, floor=0.05):
    scale = np.maximum(np.abs(a), floor)
    return np.exp(-np.abs(a - b) / scale)


def render_similarity(ref, cur):
    orientation_score = max(0.0, cosine(ref["orientation"], cur["orientation"]))
    value_score = float(_relative_closeness(ref["values"], cur["values"]).mean())
    return 0.55 * orientation_score + 0.45 * value_score


def orb_copy_risk(ref_path, image_path):
    """Detect near-literal local copying; 0 is no evidence and 1 is strong evidence."""
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    cur = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or cur is None:
        return 0.0
    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(cur, None)
    if des1 is None or des2 is None or min(len(kp1), len(kp2)) < 8:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    match_score = min(1.0, len(good) / max(12.0, min(len(kp1), len(kp2)) * 0.08))
    inlier_score = 0.0
    if len(good) >= 8:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if inliers is not None:
            inlier_score = float(inliers.mean())
    return float(0.4 * match_score + 0.6 * inlier_score)


def evaluate(ref_path, image_paths, include_clip=True):
    has_resnet = load_resnet_style_model()
    has_clip = include_clip and load_clip()
    ref_hsv = hsv_hist(ref_path)
    ref_render = render_signature(ref_path)
    ref_resnet = resnet_style_signature(ref_path) if has_resnet else None
    ref_clip = clip_embedding(ref_path) if has_clip else None
    rows = []
    for path in image_paths:
        palette = max(0.0, cosine(ref_hsv, hsv_hist(path)))
        render = render_similarity(ref_render, render_signature(path))
        gram = resnet_style_similarity(ref_resnet, resnet_style_signature(path)) if has_resnet else None
        style_score = (0.45 * palette + 0.55 * render if gram is None else
                       0.55 * gram + 0.20 * palette + 0.25 * render)
        row = {
            "case": os.path.basename(path).split("_")[0],
            "file": os.path.abspath(path),
            "style_score": round(style_score, 4),
            "gram_style": None if gram is None else round(gram, 4),
            "palette": round(palette, 4),
            "render": round(render, 4),
            "copy_risk": round(orb_copy_risk(ref_path, path), 4),
        }
        if has_clip:
            row["clip_diagnostic"] = round(cosine(ref_clip, clip_embedding(path)), 4)
        rows.append(row)
    return sorted(rows, key=lambda row: row["style_score"], reverse=True)


def vision_audit(ref_path, image_paths, labels):
    """Use the configured multimodal text model as a rubric-based second judge."""
    from utils.analysis_gpt_prompt import call_text_model, load_text_api_config
    from utils.prompt_loader import read_prompt_file

    from PIL import Image

    cfg = load_text_api_config()
    user = ("Candidate labels in attachment order after Image 1: " + ", ".join(labels) +
            "\nRequested content shared by all candidates: one aqua-haired flower keeper holding a "
            "rose-and-peony bouquet in a moonlit conservatory, full-body vertical composition. "
            "Treat unrelated stages, crowds, signage, text, extra characters or replacement settings "
            "as content violations, not style similarities.")
    proxy_dir = os.path.join(BASE_DIR, "cache", "temp")
    os.makedirs(proxy_dir, exist_ok=True)
    proxies = []
    try:
        for index, path in enumerate([ref_path] + list(image_paths)):
            target = os.path.join(proxy_dir, f"style-audit-{os.getpid()}-{int(time.time() * 1000)}-{index}.jpg")
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                image.save(target, "JPEG", quality=90, optimize=True)
            proxies.append(target)
        raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"],
                              read_prompt_file("style-similarity/audit-system.md"), user,
                              timeout=300, max_tokens=4000, image_paths=proxies)
    finally:
        for path in proxies:
            try:
                os.remove(path)
            except OSError:
                pass
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.I).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        return json.loads(match.group(0)) if match else {"raw": raw}


def main():
    parser = argparse.ArgumentParser(description="Content-resistant art-style similarity evaluation")
    parser.add_argument("--ref-image", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--no-clip", action="store_true", help="Skip diagnostic CLIP embedding")
    parser.add_argument("--vision-audit", action="store_true",
                        help="Also rank candidates with the configured multimodal text model")
    args = parser.parse_args()
    rows = evaluate(args.ref_image, args.images, include_clip=not args.no_clip)
    print("\n=== Style similarity (higher is closer; CLIP is diagnostic only) ===")
    print("case | style | gram | palette | render | copy-risk | clip(diag) | file")
    for row in rows:
        print(f"{row['case']:>4} | {row['style_score']:.4f} | "
              f"{str(row.get('gram_style')):>6} | {row['palette']:.4f} | {row['render']:.4f} | "
              f"{row['copy_risk']:.4f} | {str(row.get('clip_diagnostic', '-')):>10} | "
              f"{os.path.basename(row['file'])}")
    audit = None
    if args.vision_audit:
        labels = [os.path.basename(path).split("_")[0] for path in args.images]
        audit = vision_audit(args.ref_image, args.images, labels)
        print("\n=== Vision rubric audit ===")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"reference": os.path.abspath(args.ref_image), "rows": rows,
                       "vision_audit": audit}, handle,
                      ensure_ascii=False, indent=2)
        print(f"saved {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
