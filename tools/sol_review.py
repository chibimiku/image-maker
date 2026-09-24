# -*- coding: utf-8 -*-
"""把 Sol 的视觉审阅脚本固化成工具（可复用）：图片 + 简报 → gpt-5.6-sol 审阅 → 写入 md。

用法：
  python tools/sol_review.py --image data/<日期>/xxx-final-....png \
      --brief prompts-file-or-text --out docs/gpt-image-tid-style/sol-cases/answer_xxx.md
"""
import argparse
import base64
import json
import os
import sys
import urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

DEFAULT_PROMPT = """这是一张插画生成流水线的最终产物。请**只看这张图**，回答：
1. 肉眼可见的缺陷有哪些？按严重程度排序并给出具体位置。
2. 局部重绘/贴回有没有可见接缝或风格断层？在哪些位置？
3. 手与手套状态是否正确（手指数量、是否把戴手套的手画成裸手、手指是否融化）？
4. 线条是「连接过头」还是仍然发虚？发白/发灰严重吗？
5. 只能改 3 处时改哪 3 处性价比最高？
只描述你确实在图里看到的，不要泛泛而谈。"""


def _data_uri(path, max_edge=768, quality=72):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到图片: {path}")
    h, w = img.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise SystemExit("编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description="让 gpt-5.6-sol 审阅一张图（视觉）")
    ap.add_argument("--image", required=True)
    ap.add_argument("--brief", default="", help="额外简报：文件路径或直接文本")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-edge", type=int, default=768)
    args = ap.parse_args()

    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded()
    key = os.environ.get("IMAGE_MAKER_AIGC2D_API_KEY") or os.environ.get("AIGC2D_API_KEY") or ""
    if not key:
        raise SystemExit("缺少 IMAGE_MAKER_AIGC2D_API_KEY（写进 .env）")

    brief = args.brief
    if brief and os.path.isfile(brief):
        brief = open(brief, encoding="utf-8").read()
    prompt = (brief + "\n\n" if brief else "") + DEFAULT_PROMPT
    uri = _data_uri(args.image, max_edge=args.max_edge)
    body = {"model": args.model, "max_completion_tokens": 8000,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": uri}}]}]}
    req = urllib.request.Request(
        "https://new.aigc2d.com/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode())
    answer = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    out = args.out or os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases",
                                   f"answer_{os.path.splitext(os.path.basename(args.image))[0][:40]}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {args.model} 视觉审阅（{datetime.now():%Y-%m-%d %H:%M}）\n\n")
        f.write(f"- 图片：`{os.path.relpath(args.image, BASE)}`\n- 模型：{args.model}\n- 图片边长：{args.max_edge}px\n\n")
        if brief:
            f.write("## 简报\n\n" + brief + "\n\n")
        f.write("## 回答\n\n" + answer + "\n")
    print(f"✅ {os.path.relpath(out, BASE)}（{len(answer)} 字符）")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
