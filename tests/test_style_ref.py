# -*- coding: utf-8 -*-
"""
测试「艺术风格参考图（仅参考画风）」五组对照实验：
- A 对照组：仅样式指令（不附图）
- B 头部插入：样式指令 + 风格参考指令块，参考图作为附件（当前默认实现）
- C 参考优先：压缩样式指令为精简版 + 风格参考指令块，参考图主导画风
- D 交错模式：样式指令在前，参考图之后紧跟风格参考指令（图文交错）
- E 无样式引导：只有正文提示词（不附样式指令、不附参考图）
五组使用相同 prompt / 参考图 / 模型 / 比例，输出文件名带 A/B/C/D/E 前缀区分。

用法：
  python tests/test_style_ref.py --style puracotte-v3 --ref-image d:\\puracotte-test.jpg
  python tests/test_style_ref.py --style X --ref-image Y --prompt "自定义提示词"
"""
import argparse
import os
import json
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.others.api_backend import generate_image_aigc2d, generate_image_whatai, load_config
from utils.styles import style_prompt, build_style_ref_instruction, compress_style_text, REF_PRIORITY_PREAMBLE

STYLE_REF_IMAGE = r"d:\puracotte-test.jpg"
STYLE_NAME = "puracotte-v3"
TEST_PROMPT = (
    "a cute anime girl with long light-pink hair and big sparkling eyes, "
    "wearing a white frilly gothic lolita dress with lace and ribbons, "
    "holding a small porcelain doll, soft dreamy pastel background with "
    "floating flower petals and sparkles, upper body portrait, looking at viewer"
)
ASPECT_RATIO = "2:3"
OUT_DIR = os.path.join(BASE_DIR, "data", "test_style_ref")
os.makedirs(OUT_DIR, exist_ok=True)

# C 组：参考优先模式（声明与指令组装统一由 utils/styles.py 提供）


def load_styles():
    path = os.path.join(BASE_DIR, "conf", "config-styles.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_generate(api_type, model_name, prompt, instructions, image_paths, aspect_ratio, file_prefix, post_instructions=""):
    if api_type == "aigc2d":
        return generate_image_aigc2d(
            prompt=prompt,
            image_paths=image_paths,
            model=model_name,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            api_type=api_type,
            save_sub_dir="test_style_ref",
            file_prefix=file_prefix,
            return_metadata=True,
            post_instructions=post_instructions,
        )
    return generate_image_whatai(
        prompt=prompt,
        image_paths=image_paths,
        model=model_name,
        aspect_ratio=aspect_ratio,
        instructions=instructions,
        api_type=api_type,
        save_sub_dir="test_style_ref",
        file_prefix=file_prefix,
        return_metadata=True,
        post_instructions=post_instructions,
    )


def main():
    parser = argparse.ArgumentParser(description="艺术风格参考图 A/B/C/D 四组对照测试")
    parser.add_argument("--style", default=STYLE_NAME, help="config-styles.json 中的样式名")
    parser.add_argument("--ref-image", default=STYLE_REF_IMAGE, help="风格参考图路径")
    parser.add_argument("--prompt", default=TEST_PROMPT, help="正文提示词")
    parser.add_argument("--model", default="", help="模型名（留空则用当前配置）")
    args = parser.parse_args()

    styles = load_styles()
    if args.style not in styles:
        print(f"[error] 样式 [{args.style}] 不存在")
        return 1
    style_text = style_prompt(styles, args.style)
    if not os.path.exists(args.ref_image):
        print(f"[error] 测试图不存在: {args.ref_image}")
        return 1

    config = load_config()
    current_api = config.get("current_api", "aigc2d")
    api_cfg = config.get("apis", {}).get(current_api, {})
    model_name = args.model or api_cfg.get("model", "gemini-3-pro-image-preview")
    print(f"== 使用 API: {current_api}, 模型: {model_name} ==")
    print(f"== 样式: {args.style} ({len(style_text)} chars), 参考图: {args.ref_image} ==")

    style_ref = build_style_ref_instruction()
    print(f"== 风格参考指令块 ({len(style_ref)} chars) ==")
    print(style_ref[:400] + "...\n")

    compressed = compress_style_text(style_text)
    print(f"== C组压缩版样式指令 ({len(compressed)} chars) ==")

    cases = [
        ("A 对照-仅样式指令", "A", style_text, [], ""),
        ("B 头部插入-样式指令+风格参考图", "B", f"{style_ref}\n\n{style_text}", [args.ref_image], ""),
        ("C 参考优先-压缩样式+参考图主导", "C",
         f"{REF_PRIORITY_PREAMBLE}\n\n{style_ref}\n\n{compressed}", [args.ref_image], ""),
        ("D 交错-参考图后紧跟风格参考指令", "D", style_text, [args.ref_image], style_ref),
        ("E 无样式-仅正文提示词", "E", "", [], ""),
    ]

    results = {}
    for name, prefix, instructions, image_paths, post_instructions in cases:
        print(f"\n########## {name} ##########")
        print(f"instructions chars: {len(instructions)}, post_instructions chars: {len(post_instructions)}, image_paths: {image_paths}")
        t0 = time.time()
        try:
            result = call_generate(
                current_api, model_name, args.prompt, instructions, image_paths,
                ASPECT_RATIO, file_prefix=f"{prefix}_style_ref_test", post_instructions=post_instructions,
            )
            elapsed = time.time() - t0
            if isinstance(result, dict):
                files = result.get("saved_files", []) or []
                raw = result.get("raw_text", "")
                server_raw = result.get("server_response_raw", {}) or {}
                print(f"耗时 {elapsed:.1f}s, 保存 {len(files)} 张: {files}")
                if raw:
                    print(f"模型文本反馈: {raw[:500]}")
                if not files:
                    print(f"[warn] 服务器返回(截断): {str(server_raw)[:800]}")
                results[name] = {"files": files, "elapsed": elapsed}
            else:
                files = result or []
                print(f"耗时 {elapsed:.1f}s, 保存 {len(files)} 张: {files}")
                results[name] = {"files": files, "elapsed": elapsed}
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    print("\n================ 汇总 ================")
    for name, info in results.items():
        files = info.get("files", [])
        err = info.get("error", "")
        if files:
            print(f"[OK] {name}: {files}")
        elif err:
            print(f"[FAIL] {name}: {err}")
        else:
            print(f"[FAIL] {name}: 无返回")
    return 0


if __name__ == "__main__":
    sys.exit(main())
