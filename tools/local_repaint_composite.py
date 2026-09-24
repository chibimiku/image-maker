# -*- coding: utf-8 -*-
"""局部重绘 + 羽化贴回（CLI 薄壳，实现在 utils/post_process.py）。

按区域裁切 → 放大 → 走 Gemini 重绘（v5 固件 + 区域强调句）→ 羽化贴回。区域可选：
face / hair / head / skirt / upper / full，也可直接给 "left,top,right,bottom" 比例或 "x,y,w,h" 像素。

用法：
  python tools/local_repaint_composite.py --image <图> --region hair --feather 48 --scale 1.5 \
      [--firmware prompts/.../repaint-system-conservative-v5.md] --out <输出.jpg>
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.post_process import (DEFAULT_FEATHER, DEFAULT_SCALE, REGION_LABELS, REGION_PRESETS,  # noqa: E402
                                local_repaint_composite)


def _normalize_manual(items, image_path):
    """把比例/像素混写的框统一转成像素坐标。"""
    import cv2
    import numpy as _np
    img = cv2.imdecode(_np.fromfile(image_path, dtype=_np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return [tuple(int(v) for v in it) for it in items]
    h, w = img.shape[:2]
    out = []
    for it in items:
        if max(it) <= 1.5:
            out.append((int(it[0] * w), int(it[1] * h), int(it[2] * w), int(it[3] * h)))
        else:
            out.append(tuple(int(v) for v in it))
    return out


def main():
    ap = argparse.ArgumentParser(description="局部重绘 + 羽化贴回（近似遮罩式局部重绘）")
    ap.add_argument("--image", required=True)
    ap.add_argument("--region", default="upper",
                    help="区域：" + " / ".join(REGION_PRESETS) + "，或比例 left,top,right,bottom / 像素 x,y,w,h")
    ap.add_argument("--crop", default="", help="（兼容旧参数）等同 --region 的自定义坐标")
    ap.add_argument("--feather", type=int, default=DEFAULT_FEATHER)
    ap.add_argument("--firmware", default="prompts/gpt-image-optimize/repaint-system-conservative-v5.md")
    ap.add_argument("--resolution", default="2K")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="裁切图先放大再送重绘")
    ap.add_argument("--no-emphasis", action="store_true", help="不加区域强调句")
    ap.add_argument("--exclude", default="",
                    help='手动保留原像素的框，分号分隔：\"x,y,w,h;x2,y2,w2,h2\"（像素或 0~1 比例）')
    ap.add_argument("--no-hands", action="store_true", help="不自动排除手部（默认排除，戴手套也保留）")
    ap.add_argument("--patch", default="", help="已有局部重绘结果：跳过 API 调用，直接羽化合成")
    ap.add_argument("--keep-crop", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    region = args.crop or args.region
    manual = []
    for chunk in [c for c in str(args.exclude or "").split(";") if c.strip()]:
        vals = [float(v) for v in chunk.split(",")]
        if len(vals) == 4:
            manual.append(tuple(vals))
    if args.no_hands:
        from utils import post_process as _pp
        _pp.REGION_EXCLUSIONS["subject_no_face"] = [k for k in
                                                    _pp.REGION_EXCLUSIONS.get("subject_no_face", []) if k != "hands"]
    info = local_repaint_composite(
        args.image, args.out, region=region, firmware=args.firmware, resolution=args.resolution,
        scale=args.scale, feather=args.feather, keep_crop=args.keep_crop,
        emphasize=not args.no_emphasis, patch_path=args.patch,
        exclude_boxes=_normalize_manual(manual, args.image),
        log_callback=lambda m: print(m, flush=True))
    label = REGION_LABELS.get(str(region).lower(), region)
    print(f"区域 {label} box={info['box']} → {info['out']}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
