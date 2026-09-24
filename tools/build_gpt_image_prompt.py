# -*- coding: utf-8 -*-
"""为分析产物生成「gpt-image 专用短提示词」字段（gpt_image_prompt）。

为什么需要：分析产物是给 Gemini 通道用的（11k 画风 + 3.8~4.4k 长描述 ≈ 15k 字符），
gpt-image 通道在这个量级会断连（实测 ConnectionResetError），且长描述会让画风参考图失效。
所以单独出一份 ≤900 字符、**保留构图/姿势/服装/道具/镜头**、不含画风词的短描述。

用法：
  python tools/build_gpt_image_prompt.py --json "<分析产物.json>"           # 写入 gpt_image_prompt 字段
  python tools/build_gpt_image_prompt.py --json "<...json>" --max-chars 900 --show
  python tools/build_gpt_image_prompt.py --json "<...json>" --from-field short_description   # 不调模型，直接用已有短描述
"""
import argparse
import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from modules.others.api_backend import resolve_text_api_key  # noqa: E402
from utils import analysis_gpt_prompt as agp  # noqa: E402
from utils.prompt_loader import read_prompt_file  # noqa: E402

DEFAULT_MODEL = "gpt-5.6-luna"
URL = "https://new.aigc2d.com/v1/chat/completions"


def call_text(system_prompt, user_prompt, model=DEFAULT_MODEL, timeout=300, max_tokens=4000):
    cfg = json.load(open(os.path.join(BASE, "conf", "config.json"), encoding="utf-8"))
    key = resolve_text_api_key(cfg)
    payload = {"model": model,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}],
               "max_completion_tokens": max_tokens, "reasoning_effort": "low"}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(3):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              data=body, timeout=timeout)
            if r.status_code == 200:
                j = r.json()
                if j.get("error"):
                    last = str(j["error"])[:200]
                else:
                    return ((j["choices"][0].get("message") or {}).get("content") or "").strip()
            else:
                last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise RuntimeError(f"文本模型失败: {last}")


def sanitize(text: str) -> str:
    """去掉 markdown 围栏 / 前缀标签，压缩空白。"""
    t = str(text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    t = re.sub(r"^(short prompt|prompt|gpt-image prompt)\s*[:：]\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build(description: str, max_chars: int = 900) -> str:
    system = read_prompt_file("analysis-gpt-prompt-system.md")
    user = read_prompt_file("analysis-gpt-prompt-user.md").replace(
        "{description}", str(description or "").strip())
    out = sanitize(call_text(system, user))
    if len(out) > max_chars:
        # 超长就按句子/逗号收口，再不行直接硬截
        cut = out[:max_chars]
        for sep in (". ", "; ", ", "):
            if sep in cut:
                cut = cut[: cut.rfind(sep)]
                break
        out = cut.strip().rstrip(",;.")
    return out


def main():
    ap = argparse.ArgumentParser(description="生成分析产物的 gpt-image 短提示词字段")
    ap.add_argument("--json", required=True, help="分析产物 JSON 路径")
    ap.add_argument("--max-chars", type=int, default=0, help="默认：full=1400 / short=500")
    ap.add_argument("--tier", choices=["full", "short"], default="full",
                    help="full=完整内容锚(≤1400，不挂参考图用)；short=短内容锚(≤500，挂画风参考图用)")
    ap.add_argument("--from-field", default="", help="不调模型：直接用 JSON 里的某个字段（如 short_description）")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    key = agp.SHORT_FIELD_KEY if args.tier == "short" else agp.FIELD_KEY
    limit = args.max_chars or (agp.SHORT_FIELD_MAX_CHARS if args.tier == "short" else agp.FIELD_MAX_CHARS)

    data = json.load(open(args.json, encoding="utf-8"))
    if args.from_field:
        short = agp.sanitize_short_prompt(data.get(args.from_field) or "", limit)
    else:
        desc = data.get("english_description") or data.get("original_english_description") or ""
        if not desc:
            print("[错误] JSON 里没有 english_description")
            return 1
        short = agp.sanitize_short_prompt(agp.build_gpt_image_prompt(desc, max_chars=limit, tier=args.tier), limit)

    data[key] = short
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    suffix = "-gpt-image-short-prompts.txt" if args.tier == "short" else "-gpt-image-prompts.txt"
    txt = os.path.splitext(args.json)[0] + suffix
    with open(txt, "w", encoding="utf-8") as f:
        f.write(short)
    print(f"{key} = {len(short)} 字符（tier={args.tier}）")
    if args.show:
        print(short)
    print("已写入:", args.json)
    print("单独的 txt:", txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
