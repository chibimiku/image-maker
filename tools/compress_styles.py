# -*- coding: utf-8 -*-
"""批量把 config-styles.json 中每个画风预置的完整指令压缩为精简版（prompt_compressed）并固化。

用法:
  python tools/compress_styles.py                # 为缺少 prompt_compressed 的样式补全压缩版
  python tools/compress_styles.py --force        # 强制重新压缩所有样式
  python tools/compress_styles.py --only a,b     # 只压缩指定样式
  python tools/compress_styles.py --max-chars 800

使用 conf/config.json 中的文本 API（base_url / api_key / model）逐条调用 LLM，
要求返回 JSON: {"prompt_compressed": "..."}。
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.others.api_backend import fetch_llm_json, _extract_json_object

CONFIG_TEXT_FILE = os.path.join(PROJECT_ROOT, "conf", "config.json")
CONFIG_STYLES_FILE = os.path.join(PROJECT_ROOT, "conf", "config-styles.json")

COMPRESS_SYSTEM_PROMPT = (
    "You are an expert at condensing anime art-style specification texts for an image generation model, "
    "without losing any rule that affects the output.\n"
    "The user will paste the FULL style instruction. Produce a CONDENSED English style spec that:\n"
    "1. Keeps the overall artistic vibe, production modes, and every hard constraint "
    "(required framing, forbidden elements, fixed color/lighting rules).\n"
    "2. Keeps color & tonality rules, lighting/shadow treatment, line art and rendering conventions, detail density.\n"
    "3. Drops verbose prose, examples, repetitions and decorative phrasing.\n"
    "4. Target length: about 1/4 of the original, at most {max_chars} characters. "
    "If the original is already short, return it nearly unchanged.\n"
    "5. Respond ONLY with JSON: {{\"prompt_compressed\": \"<compressed text>\"}}"
)


def load_text_config():
    if not os.path.isfile(CONFIG_TEXT_FILE):
        return {}
    with open(CONFIG_TEXT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compress_one(base_url, api_key, model, prompt_text, max_chars, max_attempts=2):
    system_prompt = COMPRESS_SYSTEM_PROMPT.format(max_chars=max_chars)
    for attempt in range(1, max_attempts + 1):
        raw = fetch_llm_json(
            base_url, api_key, model,
            system_prompt, prompt_text,
            temperature=0.3, merge_system_prompt=False,
        )
        obj = _extract_json_object(raw)
        text = (obj.get("prompt_compressed") or obj.get("compressed") or "").strip()
        if text:
            return text
        print(f"  [warn] 第 {attempt} 次返回无法解析，重试...", flush=True)
        time.sleep(1)
    return ""


def main():
    parser = argparse.ArgumentParser(description="批量压缩画风预置指令到 prompt_compressed")
    parser.add_argument("--force", action="store_true", help="强制重新压缩（忽略已有 prompt_compressed）")
    parser.add_argument("--only", default="", help="只压缩这些样式，逗号分隔")
    parser.add_argument("--max-chars", type=int, default=700, help="压缩目标最大字符数（默认 700）")
    args = parser.parse_args()

    text_cfg = load_text_config()
    base_url = str(text_cfg.get("base_url") or "").strip()
    api_key = str(text_cfg.get("api_key") or "").strip()
    model = str(text_cfg.get("model") or "").strip()
    if not (base_url and api_key and model):
        print("[error] conf/config.json 中缺少文本 API 配置（base_url / api_key / model）")
        return 1

    with open(CONFIG_STYLES_FILE, "r", encoding="utf-8") as f:
        styles = json.load(f)

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    changed = 0
    skipped = 0
    failed = []
    for name in styles:
        entry = styles[name]
        if isinstance(entry, str):
            entry = {"prompt": entry}
            styles[name] = entry
        prompt_text = str(entry.get("prompt") or "").strip()
        if not prompt_text:
            print(f"[skip] {name}: 指令为空")
            skipped += 1
            continue
        if only is not None and name not in only:
            continue
        if not args.force and entry.get("prompt_compressed"):
            print(f"[keep] {name}: 已有 prompt_compressed，跳过（--force 可强制）")
            skipped += 1
            continue
        print(f"[run ] {name}: {len(prompt_text)} 字符 -> ", end="", flush=True)
        compressed = compress_one(base_url, api_key, model, prompt_text, args.max_chars)
        if compressed:
            entry["prompt_compressed"] = compressed
            changed += 1
            print(f"{len(compressed)} 字符 OK")
        else:
            print("失败")
            failed.append(name)
        time.sleep(0.3)

    if changed:
        tmp = CONFIG_STYLES_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(styles, f, ensure_ascii=False, indent=4)
        os.replace(tmp, CONFIG_STYLES_FILE)
        print(f"\n已更新 {changed} 个样式 -> {CONFIG_STYLES_FILE}")
    else:
        print("\n无变更")
    if failed:
        print("失败样式:", ", ".join(failed))
    print(f"跳过/保留: {skipped}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
