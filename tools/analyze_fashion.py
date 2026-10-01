# -*- coding: utf-8 -*-
"""Headless 单图全链路分析 CLI（分析功能在 modules/image_analysis/analysis_pipeline.py）。

对指定目录内的图片逐个执行 LLM 视觉分析 + 二次加工（refine / 服装搭配检查 /
去除照片风格 / 重算 pixiv_tags），并把分析结果 JSON（含 source_image_path）与
提示词 TXT 保存到各原图所在目录，结果 JSON 可直接拖入「投稿 Server」。

只做分析，不产生任何新图。

用法:
    python tools/analyze_fashion.py --dir data/20260903/fashion-generate
    python tools/analyze_fashion.py --dir data/20260905/fashion-backless --only backless_blonde_batch1 --skip temp
"""
from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.gui_entry import warm_up_optional_module  # noqa: E402

# Keep the GUI entry's import order: loading Qt first can break ONNX Runtime's DLL
# initialization on Windows, which the analysis workflow needs for local tagging.
warm_up_optional_module("onnxruntime", skip_env_var="IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD")

from modules.image_analysis.analysis_pipeline import (  # noqa: E402
    analyze_single_image,
    find_images,
    load_config,
    save_result_to_source,
)

DEFAULT_DIR = os.path.join(BASE_DIR, "data", "20260903", "fashion-generate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless 单图全链路分析（投稿格式落地）")
    parser.add_argument("--dir", default="", help=f"图片目录（默认 {DEFAULT_DIR}）")
    parser.add_argument("--image", action="append", default=[], help="单图输入；可重复。带此参数默认分析后生优化提示词图")
    parser.add_argument("--generate", choices=["none", "original", "refined", "both"], default=None,
                        help="运行与分析 Tab 相同的分析→生图链；none=只分析")
    parser.add_argument("--channel", choices=["gemini", "gpt"], default="gpt")
    parser.add_argument("--style", default="", help="画风名；省略则使用 UI 当前画风")
    parser.add_argument("--reference-mode", choices=["off", "head", "priority", "interleave"], default="off")
    parser.add_argument("--nsfw", action="store_true")
    parser.add_argument("--outfit-check", action="store_true")
    parser.add_argument("--remove-photo-style", action="store_true")
    parser.add_argument("--outfit-style", default="")
    parser.add_argument("--booru-tag-limit", type=int, default=30)
    ratios = ["不覆盖(沿用原逻辑)", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
    parser.add_argument("--aspect-ratio-first", choices=ratios, default=ratios[0],
                        help="分析后保存提示词的比例覆盖；默认保持分析比例")
    parser.add_argument("--aspect-ratio-second", choices=ratios, default=ratios[0],
                        help="Gemini 实际生图比例覆盖；默认保持分析比例")
    parser.add_argument("--save-to-source", action="store_true", help="只分析并将 JSON/TXT 存到原图目录")
    parser.add_argument("--jpg-upscale", action="store_true")
    parser.add_argument("--upscale-model", default="")
    parser.add_argument("--upscale-by", type=float, default=2.0)
    parser.add_argument("--webp-target-mb", type=float, default=10.0)
    parser.add_argument("--repaint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--local", action="store_true")
    from utils.post_process import REGION_LABELS
    parser.add_argument("--region", choices=["stable-four", *REGION_LABELS],
                        default="stable-four", help="局部重绘区域 key，或 stable-four（UI 默认）")
    parser.add_argument("--scope", choices=["full", "person_only", "person_noface", "details", "lines_only"], default="full")
    parser.add_argument("--first-pass-mode", choices=["generate", "edits"], default="generate")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--size-follow-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tone", action="store_true")
    parser.add_argument("--tone-target", choices=["style", "photo"], default="style")
    parser.add_argument("--ink", action="store_true")
    parser.add_argument("--text-model", default="", help="本次文本分析模型覆盖（不保存配置）")
    parser.add_argument("--styles-file", default="",
                        help="换用另一份画风表（整份替换 conf/config-styles.json）。受控实验用："
                             "把「正式条目 + 一个预先声明的改动」另存成实验画风表，避免改正式配置。")
    parser.add_argument("--request-dump", default="",
                        help="把 Gemini 生图线程的真实构造参数（prompt/instructions/参考图/比例/模型）"
                             "落到该目录，供受控实验核对请求差异。")
    parser.add_argument("--analysis-json", default="",
                        help="固定分析 JSON：跳过重新分析，直接用该 JSON 当内容锚生图（受控配对实验用）。"
                             "输出名沿用原 task hash。")
    parser.add_argument("--output-dir", default="",
                        help="产物目录（配合 --analysis-json：产物写到这里，不落 data/<日期>/）。")
    parser.add_argument("--image-api", default="", help="本次 Gemini 图片 API 节点覆盖")
    parser.add_argument("--image-model", default="", help="本次 Gemini 图片模型覆盖")
    parser.add_argument("--test-output", action="store_true", help="产物隔离到 data/test-result")
    parser.add_argument("--dry-run", action="store_true", help="只检查 GUI 工作流选项和输入，不调分析/生图接口")
    history_actions = parser.add_mutually_exclusive_group()
    history_actions.add_argument("--history-list", metavar="日志/断点/目录", help="列出历史任务与过程文件")
    history_actions.add_argument("--history-resume", metavar="日志/断点/目录", help="从选中过程图继续后续工序")
    history_actions.add_argument("--history-publish", metavar="日志/断点/目录", help="人工选图发布")
    parser.add_argument("--task", type=int, default=1, help="历史日志中的任务编号")
    parser.add_argument("--pick", type=int, default=0, help="历史过程文件编号")
    parser.add_argument("--history-json", action="store_true", help="历史列表输出 JSON")
    parser.add_argument("--only", default="", help="只处理文件名包含该子串的图片")
    parser.add_argument("--skip", default="", help="跳过文件名包含该子串的图片")
    parser.add_argument("--timeout", type=int, default=300, help="单步 LLM 请求超时秒数（默认 300）")
    args = parser.parse_args()

    if args.history_list or args.history_resume or args.history_publish:
        from tools.analysis_gpt_run import run_history_command
        action = "list" if args.history_list else "resume" if args.history_resume else "publish"
        target = args.history_list or args.history_resume or args.history_publish
        try:
            return run_history_command(action, target, args.task, args.pick, args.history_json)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"历史操作失败: {exc}", file=sys.stderr, flush=True)
            return 1

    if args.image or args.generate is not None:
        from tools.analysis_gpt_run import run_app_workflow
        sources = [os.path.abspath(p) for p in args.image]
        if args.dir:
            scan_dir = os.path.abspath(args.dir)
            if not os.path.isdir(scan_dir):
                parser.error(f"目录不存在: {scan_dir}")
            sources.extend(find_images(scan_dir))
        if args.only:
            sources = [p for p in sources if args.only in os.path.basename(p)]
        if args.skip:
            sources = [p for p in sources if args.skip not in os.path.basename(p)]
        sources = list(dict.fromkeys(sources))
        if not sources or any(not os.path.isfile(p) for p in sources):
            parser.error("至少指定一张存在的 --image，或用 --dir 指定有图片的目录")
        if args.save_to_source and (args.generate or "refined") != "none":
            parser.error("--save-to-source 必须与 --generate none 一起使用")
        if args.save_to_source and args.test_output:
            parser.error("--save-to-source 会写原图目录，不能与 --test-output 同用")
        if args.channel == "gpt" and (args.image_api or args.image_model):
            parser.error("GPT 通道使用 UI 固定的 gpt-image 节点；--image-api/--image-model 仅用于 Gemini")
        options = {"generate": args.generate or "refined", "channel": args.channel,
                   "style": args.style, "reference_mode": args.reference_mode,
                   "nsfw": args.nsfw, "outfit_check": args.outfit_check,
                   "remove_photo_style": args.remove_photo_style, "outfit_style": args.outfit_style,
                   "booru_tag_limit": args.booru_tag_limit, "save_to_source": args.save_to_source,
                   "aspect_ratio_first": args.aspect_ratio_first,
                   "aspect_ratio_second": args.aspect_ratio_second,
                   "jpg_upscale": args.jpg_upscale, "upscale_model": args.upscale_model,
                   "upscale_by": args.upscale_by, "webp_target_mb": args.webp_target_mb,
                   "repaint": args.repaint, "structure": args.structure, "local": args.local,
                   "region": args.region, "scope": args.scope,
                   "first_pass_mode": args.first_pass_mode, "quality": args.quality,
                   "size_follow_input": args.size_follow_input, "tone": args.tone,
                   "tone_target": args.tone_target, "ink": args.ink,
                   "text_model": args.text_model, "image_api": args.image_api,
                   "image_model": args.image_model, "timeout": args.timeout,
                   "styles_file": args.styles_file, "request_dump": args.request_dump,
                   "analysis_json": args.analysis_json, "output_dir": args.output_dir,
                   "test_output": args.test_output, "dry_run": args.dry_run}
        try:
            return run_app_workflow(sources, options)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"错误: {exc}", flush=True)
            return 1

    dir_env = os.environ.get("ANALYZE_DIR", "")
    images_dir = args.dir or dir_env or DEFAULT_DIR
    if not os.path.isabs(images_dir):
        images_dir = os.path.join(BASE_DIR, images_dir)
    if not os.path.isdir(images_dir):
        print(f"错误：目录不存在: {images_dir}")
        return 1

    config = load_config()
    images = find_images(images_dir)
    print(f"共发现 {len(images)} 张图片待分析：")
    for img in images:
        print(f"  - {os.path.basename(img)}")

    if args.only:
        images = [p for p in images if args.only in os.path.basename(p)]
    if args.skip:
        images = [p for p in images if args.skip not in os.path.basename(p)]

    for img_path in images:
        print("\n" + "=" * 78, flush=True)
        print(f"开始分析: {img_path}", flush=True)
        try:
            result = analyze_single_image(img_path, config, timeout_seconds=args.timeout,
                                          log_callback=print)
            if result:
                save_result_to_source(result, img_path, log_callback=print)
                print(f"✅ 完成: {os.path.basename(img_path)}", flush=True)
            else:
                print(f"❌ 未生成有效结果: {os.path.basename(img_path)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 分析异常 {os.path.basename(img_path)}: {exc}", flush=True)
    print("\n" + "=" * 78, flush=True)
    print("全部处理结束。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
