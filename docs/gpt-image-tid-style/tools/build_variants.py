# -*- coding: utf-8 -*-
"""重建 gpt-image 画风实验的提示词变体（docs/gpt-image-tid-style/prompts/）。

变体定义与 docs/gpt-image-tid-style/README.md §2.1 一致；直接复用项目统一的
`utils/styles.py` 组装函数，所以画风条目改动后重跑本脚本即可得到新变体。

用法（仓库根目录）：
  & "C:\\Program Files\\Python310\\python.exe" docs/gpt-image-tid-style/tools/build_variants.py --style tid
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE)

from utils.styles import (  # noqa: E402
    MODE_HEAD,
    MODE_INTERLEAVE,
    MODE_OFF,
    MODE_PRIORITY,
    build_ref_gen_params,
)

DEFAULT_OUT = os.path.join(BASE, "docs", "gpt-image-tid-style", "prompts")
STYLES_PATH = os.path.join(BASE, "conf", "config-styles.json")

SHORT_STYLE_TAG = (
    "You are given 1 reference image. Extract ONLY its art style: watercolor-like soft digital "
    "illustration, delicate fine lineart, luminous pastel palette, airy white negative space, floating "
    "droplets and bubbles, translucent hair strands, glossy detailed eyes. "
    "Do NOT copy its subject, pose, outfit or scene."
)

CALIBRATED_TEMPLATE = (
    "Use the art style of the attached reference image (Image 1) and draw a NEW image of the scene below. "
    "Keep from Image 1: <风格要点>. "
    "Change from Image 1: the character, pose, outfit, camera framing and background layout — all come "
    "only from the scene description. Do not copy the reference's person or composition. "
    "Scene: <主体描述>"
)


def build_variants(style: str):
    styles = json.load(open(STYLES_PATH, encoding="utf-8"))
    entry = styles.get(style) or {}
    variants = {}

    head, _post, _refs = build_ref_gen_params(styles, style, MODE_OFF)
    variants["A1_full_prompt_only"] = head
    variants["A2_full_prompt_only"] = entry.get("prompt_compressed", "")

    head, _post, _refs = build_ref_gen_params(styles, style, MODE_PRIORITY)
    variants["B1_compressed_priority"] = head

    head, _post, _refs = build_ref_gen_params(styles, style, MODE_HEAD)
    variants["B2_full_head"] = head

    head, post, _refs = build_ref_gen_params(styles, style, MODE_INTERLEAVE)
    variants["B3_full_interleave"] = (head + "\n\n" + post).strip()

    variants["C_short_style_tag"] = SHORT_STYLE_TAG
    variants["D_control_no_style"] = ""
    variants["G_calibrated_template"] = CALIBRATED_TEMPLATE
    return variants


def main():
    parser = argparse.ArgumentParser(description="重建 gpt-image 画风实验提示词变体")
    parser.add_argument("--style", default="tid", help="config-styles.json 中的画风名，默认 tid")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出目录（默认 docs/gpt-image-tid-style/prompts）")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    variants = build_variants(args.style)
    for name, text in variants.items():
        path = os.path.join(args.out, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        print(f"{name:28s} chars={len(text.strip()):6d}  {path}")
    print(f"\n画风 {args.style}：全量 {len((variants.get('A1_full_prompt_only') or ''))} 字符 / "
          f"压缩 {len((variants.get('A2_full_prompt_only') or ''))} 字符")


if __name__ == "__main__":
    main()
