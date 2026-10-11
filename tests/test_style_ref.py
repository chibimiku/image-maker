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
import copy
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.others.api_backend import generate_image_aigc2d, generate_image_whatai, load_config
from utils.styles import (MODE_HEAD, MODE_INTERLEAVE, MODE_PRIORITY, build_ref_gen_params,
                          build_style_ref_instruction, style_prompt, style_ref_image)

STYLE_NAME = "puracotte-style-v2"
TEST_PROMPT = (
    "a cute anime girl with long light-pink hair and big sparkling eyes, "
    "wearing a white frilly gothic lolita dress with lace and ribbons, "
    "holding a small porcelain doll, soft dreamy pastel background with "
    "floating flower petals and sparkles, upper body portrait, looking at viewer"
)
ASPECT_RATIO = "2:3"
# C 组：参考优先模式（声明与指令组装统一由 utils/styles.py 提供）


def run_plan(plan_path, dry_run=False):
    """Data-driven, bounded priority-mode comparison through the live assembler."""
    from modules.others.api_backend import get_api_config
    from utils.analysis_gen import build_gemini_content_postamble
    from utils import send_budget
    plan_path = os.path.abspath(plan_path)
    plan = json.load(open(plan_path, encoding="utf-8"))
    base = os.path.dirname(plan_path)
    styles = load_styles()
    style = plan["style"]
    cfg = get_api_config(api_type="aigc2d")
    model = cfg["model"]
    output = os.path.abspath(plan["output_dir"])
    os.makedirs(output, exist_ok=True)
    os.environ[send_budget.ENV_DIR] = os.path.join(output, "budget")
    os.environ[send_budget.ENV_LIMIT] = str(plan["max_calls"])
    prompt = open(os.path.join(base, plan["prompt_file"]), encoding="utf-8").read().strip()
    variants = plan["variants"]
    repeats = int(plan.get("repeat", 1))
    if repeats * len(variants) > int(plan["max_calls"]):
        raise ValueError("Plan exceeds image budget")
    for repeat in range(1, repeats + 1):
        for variant in variants:
            name = f"{variant['name']}-{repeat}"
            local = copy.deepcopy(styles)
            for field, filename in variant.get("field_files", {}).items():
                local[style][field] = open(os.path.join(base, filename), encoding="utf-8").read().strip()
            head, post, refs = build_ref_gen_params(local, style, MODE_PRIORITY, api_type="aigc2d")
            if variant.get("head_template"):
                head = open(os.path.join(base, variant["head_template"]), encoding="utf-8").read().strip()
            for filename in variant.get("extra_references", []):
                refs.append(os.path.abspath(os.path.join(base, filename)))
            if variant.get("reverse_references", False):
                refs.reverse()
            if plan.get("content_postamble", True):
                post = build_gemini_content_postamble(prompt, post)
            if variant.get("post_template"):
                template = open(os.path.join(base, variant["post_template"]), encoding="utf-8").read().strip()
                post = template.replace("{content}", prompt)
            full = head + "\n" + prompt + "\n" + post
            record = dict(name=name, model=model, resolution=cfg.get("resolution", "2K"),
                          aspect_ratio=plan["aspect_ratio"], instructions=head, prompt=prompt,
                          post_instructions=post, references=refs, retry_limit=0,
                          timeout=360, face_quality_boost=variant.get("face_quality_boost", True),
                          reference_sha256=[hashlib.sha256(open(p, "rb").read()).hexdigest() for p in refs])
            request_path = os.path.join(output, name + ".request.json")
            if os.path.exists(request_path) and not dry_run:
                raise ValueError(f"Existing attempt: {name}; inspect evidence before resuming")
            if dry_run:
                print(f"PREVIEW {name}: {model}, {len(full)} chars, {refs}", flush=True)
                continue
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            reservation = send_budget.reserve_image_attempt(
                run_id=name, operation=name, model=model, resolution=record["resolution"],
                aspect_ratio=plan["aspect_ratio"], prompt_sha256=hashlib.sha256(full.encode()).hexdigest(),
                prompt_chars=len(full), reference_images=refs)
            print(f"SEND {name}", flush=True)
            started = time.time()
            messages = []
            try:
                result = generate_image_aigc2d(
                    prompt=prompt, instructions=head, post_instructions=post, image_paths=refs,
                    model=model, aspect_ratio=plan["aspect_ratio"], resolution=record["resolution"],
                    api_type="aigc2d", save_sub_dir=output, file_prefix=name, return_metadata=True,
                    request_timeout=360, retry_limit=0, log_callback=messages.append,
                    face_quality_boost=record["face_quality_boost"])
                files = result.get("saved_files", []) if isinstance(result, dict) else (result or [])
                record.update(files=files, elapsed=time.time()-started,
                              messages=messages, error="" if files else
                              (result.get("raw_text", "No image; billing unknown") if isinstance(result, dict)
                               else "No image; billing unknown"))
                send_budget.settle_image_attempt(reservation, status="success" if files else "unknown",
                                                 output_path=files[0] if files else "", error=record["error"])
            except Exception as exc:
                record.update(files=[], elapsed=time.time()-started, error=str(exc))
                send_budget.settle_image_attempt(reservation, status="unknown", error=str(exc))
            with open(os.path.join(output, name + ".result.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"DONE {name}: {record['files']} {record['error']}", flush=True)
            if not record["files"]:
                return 1
    return 0


def load_styles():
    path = os.path.join(BASE_DIR, "conf", "config-styles.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_generate(api_type, model_name, prompt, instructions, image_paths, aspect_ratio, file_prefix,
                  post_instructions="", save_sub_dir="test_style_ref"):
    if api_type == "aigc2d":
        return generate_image_aigc2d(
            prompt=prompt,
            image_paths=image_paths,
            model=model_name,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            api_type=api_type,
            save_sub_dir=save_sub_dir,
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
        save_sub_dir=save_sub_dir,
        file_prefix=file_prefix,
        return_metadata=True,
        post_instructions=post_instructions,
    )


def main():
    parser = argparse.ArgumentParser(description="艺术风格参考图 A/B/C/D 四组对照测试")
    parser.add_argument("--plan", help="JSON plan for a bounded priority-mode comparison")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--style", default=STYLE_NAME, help="config-styles.json 中的样式名")
    parser.add_argument("--ref-image", default="", help="覆盖配置里的风格参考图路径")
    parser.add_argument("--prompt", default=TEST_PROMPT, help="正文提示词")
    parser.add_argument("--model", default="", help="模型名（留空则用当前配置）")
    parser.add_argument("--cases", default="ABCDE", help="只运行指定组，例如 C 或 BCD")
    parser.add_argument("--save-sub-dir", default="",
                        help="传给后端的输出子目录；留空时隔离到 data/test-result/<日期>/style-ref-horizontal/<画风>")
    args = parser.parse_args()
    if args.plan:
        return run_plan(args.plan, args.dry_run)

    styles = load_styles()
    if args.style not in styles:
        print(f"[error] 样式 [{args.style}] 不存在")
        return 1
    style_text = style_prompt(styles, args.style)
    ref_image = args.ref_image or style_ref_image(styles, args.style)
    if not os.path.exists(ref_image):
        print(f"[error] 测试图不存在: {ref_image}")
        return 1
    # Keep the CLI override authoritative for all reference-bearing cases.
    # Previously --ref-image was only existence-checked while B/C/D were still
    # assembled from the configured path, making reference A/B tests invalid.
    if args.ref_image:
        styles = dict(styles)
        styles[args.style] = dict(styles[args.style])
        styles[args.style]["ref_image"] = os.path.abspath(args.ref_image)

    config = load_config()
    current_api = config.get("current_api", "aigc2d")
    api_cfg = config.get("apis", {}).get(current_api, {})
    model_name = args.model or api_cfg.get("model", "gemini-3-pro-image-preview")
    print(f"== 使用 API: {current_api}, 模型: {model_name} ==")
    print(f"== 样式: {args.style} ({len(style_text)} chars), 参考图: {ref_image} ==")

    style_ref = build_style_ref_instruction()
    print(f"== 风格参考指令块 ({len(style_ref)} chars) ==")
    print(style_ref[:400] + "...\n")

    head_b, post_b, refs_b = build_ref_gen_params(styles, args.style, MODE_HEAD, api_type=current_api)
    priority_head, priority_post, priority_refs = build_ref_gen_params(
        styles, args.style, MODE_PRIORITY, api_type=current_api)
    head_d, post_d, refs_d = build_ref_gen_params(
        styles, args.style, MODE_INTERLEAVE, api_type=current_api)
    print(f"== C组参考优先指令 ({len(priority_head)} + 图后 {len(priority_post)} chars) ==")

    cases = [
        ("A 对照-仅样式指令", "A", style_text, [], ""),
        ("B 头部插入-样式指令+风格参考图", "B", head_b, refs_b, post_b),
        ("C 参考优先-压缩样式+参考图主导", "C",
         priority_head, priority_refs, priority_post),
        ("D 交错-参考图后紧跟风格参考指令", "D", head_d, refs_d, post_d),
        ("E 无样式-仅正文提示词", "E", "", [], ""),
    ]
    wanted = {c.upper() for c in str(args.cases or "ABCDE") if c.upper() in "ABCDE"}
    cases = [case for case in cases if case[1] in wanted]
    safe_style = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.style).strip("-") or "style"
    save_sub_dir = args.save_sub_dir or os.path.join(
        "..", "test-result", time.strftime("%Y%m%d"), "style-ref-horizontal", safe_style)

    results = {}
    for name, prefix, instructions, image_paths, post_instructions in cases:
        print(f"\n########## {name} ##########")
        print(f"instructions chars: {len(instructions)}, post_instructions chars: {len(post_instructions)}, image_paths: {image_paths}")
        t0 = time.time()
        try:
            result = call_generate(
                current_api, model_name, args.prompt, instructions, image_paths,
                ASPECT_RATIO, file_prefix=f"{prefix}_{safe_style}_style_ref_test",
                post_instructions=post_instructions, save_sub_dir=save_sub_dir,
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
