# -*- coding: utf-8 -*-
"""把 Sol 的审阅固化成工具（可复用）：图片或长文 + 简报 → gpt-5.6-sol / gpt-6-sol 审阅 → 写入 md。

两种模式：

  图片审阅（原行为）：
    python tools/sol_review.py --image data/<日期>/xxx-final-....png \
        --brief prompts-file-or-text --out docs/gpt-image-tid-style/sol-cases/answer_xxx.md

  长文审阅（2026-10-02 新增，给「工作指引/方案书」这类纯文本成品做技术审阅）：
    python tools/sol_review.py --text docs/261003-color-improve/AI_Color_Knowledge_Extraction_Workflow.md \
        --brief prompts/color-knowledge/workflow-review.md --model gpt-6-sol \
        --out docs/261003-color-improve/review-v1.md

模型名要实测确认（`GET {base}/v1/models`）：本仓库实测可用 `gpt-5.6-sol` / `gpt-5.6-sol-max` /
`gpt-5.6-sol-ultra` / `gpt-6-sol` / `gpt-6.1-sol`；`gpt-6-sol` 与 `gpt-6.1-sol` 均可正常返回。
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

DEFAULT_TEXT_PROMPT = """下面是一份**技术工作指引**（不是代码，是给另一个 AI 编码助手看的施工说明书）。
请以「严格的技术评审」身份读完，然后按这五节回答：

## 一、事实性错误
文中对现有系统的描述（模块名、函数名、文件路径、CLI 参数、字段名、行号、模型名、目录约定）
哪些与真实代码不符、或者无法验证？逐条列出「文中说的」→「实际是/无法确认」。
没有把握的地方就写「无法验证」，不要替我圆场。

## 二、可执行性缺口
文中给出的命令、参数、步骤，哪些直接跑会失败？哪些缺少前置条件（文件不存在、目录没建、
密钥没配、依赖没装、编码没设）？给出具体补齐方式。

## 三、设计与工序漏洞
工序顺序、缓存/续跑、失败处理、幂等性、成本、并发、人工验收环节上有什么漏洞？
如果某个环节的产出会污染生产链路（例如写进 conf/ 配置文件、往日期目录扔产物），明确指出来。

## 四、过度设计 / 无效复杂度
哪些步骤对「最终让生图变好」几乎没有贡献，可以砍掉？哪些是自我感动式的流程？

## 五、最小可行路径
如果只能做 5 步、其余全砍，你建议保留哪 5 步？按顺序写，每步一句话说清「跑什么、产出什么、
怎么判断这步成功了」。
"""


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


def _read_brief(brief):
    if brief and os.path.isfile(brief):
        return open(brief, encoding="utf-8").read()
    return brief or ""


def main():
    ap = argparse.ArgumentParser(description="让 sol 系列模型审阅一张图或一份长文")
    ap.add_argument("--image", default="", help="图片审阅模式的输入图")
    ap.add_argument("--text", default="", help="长文审阅模式的输入文件")
    ap.add_argument("--brief", default="", help="额外简报/评审要求：文件路径或直接文本")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-edge", type=int, default=768)
    ap.add_argument("--max-tokens", type=int, default=16000)
    args = ap.parse_args()
    if not args.image and not args.text:
        raise SystemExit("至少要给 --image 或 --text 之一")

    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded()
    key = os.environ.get("IMAGE_MAKER_AIGC2D_API_KEY") or os.environ.get("AIGC2D_API_KEY") or ""
    if not key:
        raise SystemExit("缺少 IMAGE_MAKER_AIGC2D_API_KEY（写进 .env）")

    brief = _read_brief(args.brief)
    content = []
    if args.text:
        path = args.text if os.path.isfile(args.text) else ""
        body = open(path, encoding="utf-8").read() if path else args.text
        label = os.path.relpath(path, BASE) if path else "(内联文本)"
        content.append({"type": "text", "text":
                        (brief or DEFAULT_TEXT_PROMPT) + f"\n\n===== 待审阅文档：{label} =====\n\n" + body})
    else:
        content.append({"type": "text", "text": (brief + "\n\n" if brief else "") + DEFAULT_PROMPT})
        content.append({"type": "image_url", "image_url": {"url": _data_uri(args.image, max_edge=args.max_edge)}})

    body = {"model": args.model, "max_completion_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        "https://new.aigc2d.com/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode())
    if data.get("error"):
        raise SystemExit(f"接口报错：{str(data['error'])[:300]}")
    answer = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    if not answer.strip():
        raise SystemExit("模型返回空内容（推理模型可能把预算烧在 reasoning 上，加大 --max-tokens 再试）")
    stem = os.path.splitext(os.path.basename(args.image or args.text))[0][:60]
    out = args.out or os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases", f"answer_{stem}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {args.model} 审阅（{datetime.now():%Y-%m-%d %H:%M}）\n\n")
        f.write(f"- 对象：`{os.path.relpath(args.text or args.image, BASE)}`\n- 模型：{args.model}\n")
        if args.image:
            f.write(f"- 图片边长：{args.max_edge}px\n")
        f.write(f"- tokens：{json.dumps(data.get('usage') or {}, ensure_ascii=False)}\n\n")
        if brief:
            f.write("## 简报\n\n" + brief + "\n\n")
        f.write("## 回答\n\n" + answer + "\n")
    print(f"✅ {os.path.relpath(out, BASE)}（{len(answer)} 字符）")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
