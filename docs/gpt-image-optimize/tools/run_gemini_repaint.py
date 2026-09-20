# -*- coding: utf-8 -*-
"""在 gpt-image 产物上跑 Gemini 3 Pro Image 整体重绘（走项目既有 aigc2d 通道）。

用法：
  python run_gemini_repaint.py --source <输入png> --prompt-file <md/txt> --prefix <前缀> [--model ...]
"""
import argparse
import os
import re
import sys

BASE = r"D:\code\image-maker"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from modules.others.api_backend import generate_image_aigc2d  # noqa: E402


def load_prompt(path):
    text = open(path, encoding="utf-8").read()
    # 支持 md：只取 ``` 代码块里的正文（若有）
    blocks = re.findall(r"```(?:text)?\n(.*?)```", text, re.S)
    if blocks:
        return "\n".join(b.strip() for b in blocks).strip()
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model", default="gemini-3-pro-image-preview")
    ap.add_argument("--resolution", default="2K")
    ap.add_argument("--aspect", default="2:3")
    ap.add_argument("--subdir", default="gemini_repaint")
    ap.add_argument("--out", help="把最终 prompt 也写到这个文件，便于复用")
    args = ap.parse_args()

    prompt = load_prompt(args.prompt_file)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(prompt + "\n")
    print(f"model={args.model}  resolution={args.resolution}  aspect={args.aspect}", flush=True)
    print(f"source={args.source}", flush=True)
    print(f"prompt chars={len(prompt)}", flush=True)

    saved = generate_image_aigc2d(
        prompt=prompt,
        image_paths=[args.source],
        model=args.model,
        aspect_ratio=args.aspect,
        resolution=args.resolution,
        api_type="aigc2d",
        save_sub_dir=args.subdir,
        file_prefix=args.prefix,
        log_callback=lambda m: print("  " + str(m), flush=True),
    )
    print("SAVED", saved, flush=True)
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
