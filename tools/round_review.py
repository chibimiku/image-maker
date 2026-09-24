# -*- coding: utf-8 -*-
"""每轮固定流程：指标 + 我自己的裁切检查图 + 交 Sol 单图复核。

用法：
  python tools/round_review.py --image <产物.png> --tag r3 [--ask-sol]
输出：
  docs/gpt-image-tid-style/sol-cases/round-<tag>-*.png    （整图/下半/腰腹/靴子/脸部 裁切）
  docs/gpt-image-tid-style/sol-cases/round-<tag>-review.md（指标 + Sol 回答）
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tests"))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

OUT = os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases")
REF = os.path.join(BASE, "data", "style-ref", "tinkle-test.png")
PHOTO = os.path.join(BASE, "data", "20260921", "sucai", "2026-07-24 12_17_40_1.jpg")

CROPS = {
    "full": None,
    "bottom": (0.00, 0.52, 1.00, 1.00),
    "waist": (0.18, 0.28, 0.78, 0.62),
    "boots": (0.35, 0.72, 1.00, 1.00),
    "face": (0.05, 0.02, 0.55, 0.35),
}


def metrics(path):
    import style_render_metrics as m
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "（缺）"
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * 900 / h)), 900), interpolation=cv2.INTER_AREA)
    tmp = os.path.join(BASE, "data", "_rm.png")
    cv2.imwrite(tmp, small)
    cur = m.analyze(tmp)
    os.remove(tmp)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    return (f"{w}x{h} 亮度 {gray.mean():.1f} 饱和 {sat:.1f} 段长 {cur['line_avg_len']:.1f} "
            f"端点 {cur['line_endpoint_density']:.2f} 长线 {cur['line_long_ratio']:.3f} "
            f"碎线 {cur['line_frag_ratio']:.3f} 边缘占比 {cur['edge_share_mid']:.3f}")


def write_crops(path, tag):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    made = []
    for name, box in CROPS.items():
        if box is None:
            target = img
        else:
            target = img[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
        scale = min(2.0, 1400.0 / max(target.shape[:2]))
        if scale > 1.01:
            target = cv2.resize(target, (int(target.shape[1] * scale), int(target.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA)
        p = os.path.join(OUT, f"round-{tag}-{name}.png")
        cv2.imwrite(p, target)
        made.append(p)
    return made


def sol_uri(path, max_edge=768, quality=72):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def ask_sol(image_path, prompt):
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded()
    key = os.environ.get("IMAGE_MAKER_AIGC2D_API_KEY") or ""
    body = {"model": "gpt-5.6-sol", "max_completion_tokens": 9000,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": sol_uri(image_path)}}]}]}
    req = urllib.request.Request("https://new.aigc2d.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode())
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


PROMPT = """这是插画流水线本轮（第 3 轮）的最终产物。流程：真人照片 → gpt-image-2（tinkle 画风参考图 + 内容锚）→
Gemini 重绘（v5 保守固件 + 源图与 tinkle 参考图双参考 + 18 条画风条款）→ 结构线叠加（strength 0.28 / darken 0.18 / 1px 骨架 / 高频区减弱）
→ 局部重绘（先整身·保脸，再单独修鞋/靴带；两遍都**限制在人物主体遮罩内贴回**，背景保持原像素）。

上一轮你指出的问题：下半截有明显拼接错误（多出椅子腿、半截脚）、线条仍偏粗、束腰与袜带精度不足。

请只看这张图回答：
1. 下半截的拼接/重复结构问题是否消失？还有没有"多出来的家具腿""半截肢体""水平接缝"？
2. 线条粗细是否已经接近参考图那种细线？还有哪几处偏粗？
3. 束腰（corset）的绑带/系带、袜带（garter）/吊袜带、鞋带这三处的**结构可读性**如何？具体哪里不对？
4. 五官/面部/头发的"作画方法"（不含配色与角色固有特征）与参考图相比，还差什么？
5. 内容（姿势、服装、道具、背景布局）与输入照片是否一致？有没有跑偏？
6. 下一步最该做的 2 件事。"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ask-sol", action="store_true")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    crops = write_crops(args.image, args.tag)
    print("裁切:", [os.path.relpath(p, BASE) for p in crops])
    print("指标:", metrics(args.image))
    if args.ask_sol:
        answer = ask_sol(args.image, args.prompt)
    else:
        answer = "（本轮未调用 Sol）"
    md = os.path.join(OUT, f"round-{args.tag}-review.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# 第 {args.tag} 轮复核\n\n- 产物：`{os.path.relpath(args.image, BASE)}`\n\n## 指标\n\n"
                f"- 本轮：{metrics(args.image)}\n- tinkle 参考：{metrics(REF)}\n- 输入照片：{metrics(PHOTO)}\n\n"
                f"## 裁切\n\n" + "\n".join(f"- `{os.path.relpath(p, BASE)}`" for p in crops)
                + "\n\n## Sol 复核\n\n" + answer + "\n")
    print("复核文档:", os.path.relpath(md, BASE))
    print(answer[:2000])


if __name__ == "__main__":
    sys.exit(main())
