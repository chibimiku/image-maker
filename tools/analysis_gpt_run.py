# -*- coding: utf-8 -*-
"""分析产物 → gpt-image-2（带画风）→ 后处理工序：**无头命令行版**（与 GUI「分析图片并生图」同一套逻辑）。

例：
  python tools/analysis_gpt_run.py --json "data/20260921/sucai/xxx.json" --style tinkle \
      --quality high --size auto --steps repaint,structure,local --region subject_no_face

流程（调用 GUI 用的同一批函数）：
  1. 读分析 JSON → 短锚内容（`utils.analysis_gen.resolve_content_text`）
  2. 画风 `prompt_gpt` + 画风参考图（`utils.styles`）→ 组请求（`analysis_gen.build_gpt_image_request`）
  3. gpt-image-2 出首图（`api_backend.generate_image_aigc2d_gpt`），尺寸「auto」按输入图比例选
  4. 后处理：重绘（可选双参考）→ 结构线叠加 → 局部重绘+羽化贴回（`utils.post_process.run_pipeline`）
     最终产物落 `data/<日期>/`，中间产物落 `data/<日期>/pipeline-steps/<run-id>/`
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass


def _metrics(path):
    try:
        import cv2
        import numpy as np
        sys.path.insert(0, os.path.join(BASE, "tests"))
        import style_render_metrics as m
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * 900 / h)), 900), interpolation=cv2.INTER_AREA)
        tmp = os.path.join(os.path.dirname(path), "_m.png")
        cv2.imwrite(tmp, small)
        cur = m.analyze(tmp)
        os.remove(tmp)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        return (f"亮度 {gray.mean():.1f} 饱和 {sat:.1f} 段长 {cur['line_avg_len']:.1f} "
                f"端点 {cur['line_endpoint_density']:.2f} 长线 {cur['line_long_ratio']:.3f}")
    except Exception:  # noqa: BLE001
        return ""


def main():
    ap = argparse.ArgumentParser(description="分析产物 → gpt-image-2 → 工序（无头）")
    ap.add_argument("--json", required=True, help="分析产物 JSON（或同目录的 -prompts.txt 不必传）")
    ap.add_argument("--style", default="", help="画风名（config-styles.json 的键）")
    ap.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--size", default="auto", help="auto=按输入图比例挑；也可写 1024x1536 等")
    ap.add_argument("--steps", default="repaint,structure,local",
                    help="要跑的工序，逗号分隔：repaint,structure,local（空 = 只出首图）")
    ap.add_argument("--region", default="subject_no_face", help="局部重绘区域（默认 subject_no_face）")
    ap.add_argument("--feather", type=int, default=40)
    ap.add_argument("--structure-strength", type=float, default=0.5)
    ap.add_argument("--content-image", default="",
                    help="指定内容参考图（默认用分析原图）；可传一张更小的图以规避中转站大载荷 reset")
    ap.add_argument("--content-ref", default="off", choices=["on", "off"],
                    help="首图是否把**分析原图**当内容参考图挂上。off（默认）=分析图只用来出文字；"
                         "on=把原图当内容参考图（视觉锚定最强，代价是出图基本就是原图的画风重绘）")
    ap.add_argument("--use-source-base", action="store_true",
                    help="首图直接用分析原图（保线条，走 §⑲ E 路线），跳过 gpt-image 生成")
    ap.add_argument("--first-pass-mode", default="generate", choices=["generate", "edits"],
                    help="首图端点：generate（默认，/images/generations + image 字段，= app.py 当前契约）"
                         "或 edits（历史路径，把画风参考图当输入图编辑；实测画风注入更强但线条更碎）")
    ap.add_argument("--no-dual", action="store_true", help="重绘不用双参考（默认用）")
    ap.add_argument("--white-contrast", type=float, default=0.0,
                    help="白色/高亮材质区的局部对比增强量（0=关，建议 0.4~0.7；用于高光压平结构的情况）")
    ap.add_argument("--ink", action="store_true",
                    help="最后给线条加墨（只压线条像素），让线重新比底色深；线-底对比目标默认 12")
    ap.add_argument("--ink-target", type=float, default=10.0, help="线条与局部底色的目标分离度（亮度级）")
    ap.add_argument("--ink-max-darken", type=float, default=40.0)
    ap.add_argument("--preset", default="", choices=["", "quality", "fidelity"],
                    help="预设：quality=画质优先（色调目标取画风参考图 + 线条加墨，线-底对比 −20，色彩略偏参考图）；"
                         "fidelity=色彩保真（色调目标取输入照片、不加墨）")
    ap.add_argument("--tone-target", default="style", choices=["photo", "style"],
                    help="色调校准的目标取谁：style=画风参考图（默认，深色高对比氛围更像参考图，线-底对比 −16.7）、"
                         "photo=输入照片（更亮，线-底对比 −11.2）")
    ap.add_argument("--tone", action="store_true",
                    help="最后加一道确定性色调校准（压发白高光、提中对比与色度；目标亮度取输入图）")
    ap.add_argument("--tone-contrast", type=float, default=1.08)
    ap.add_argument("--tone-chroma", type=float, default=1.10,
                    help="色度增益上限；若给了 --tone-saturation 则按参考图饱和度求解（推荐）")
    ap.add_argument("--highlight-strength", type=float, default=0.85,
                    help="高光压缩比例（<1 越多压得越狠，建议 0.75~0.80 再压 3~5%%）")
    ap.add_argument("--skin-warm", type=float, default=0.0, help="肤色暖化强度（0~1，建议 0.3~0.5）")
    ap.add_argument("--sat-scale", type=float, default=1.0, help="目标饱和度系数（0.9~0.95 再降一点）")
    ap.add_argument("--tone-saturation", type=float, default=None,
                    help="目标饱和度（默认自动取输入照片的饱和度）")
    ap.add_argument("--local-resolution", default="2K", choices=["1K", "2K", "4K"],
                    help="局部重绘分辨率（默认全区域 2K，与 app.py 的工序契约一致）")
    ap.add_argument("--detail-boost", action="store_true",
                    help="把细节区（waist/thigh/shoes/face/hair）额外升到 4K。默认关 —— "
                         "2026-09-24 的契约是局部重绘统一 2K（4K 慢且不稳，实测收益不稳定）")
    ap.add_argument("--no-detail-boost", action="store_true", help="（保留参数）显式关闭细节区升分辨率")
    ap.add_argument("--repaint-ref", default="style", choices=["line", "style", "both", "none"],
                    help="重绘的第二张参考：line=线锚图 / style=画风参考图（Sol 建议，默认）/ both / none")
    ap.add_argument("--extra-region", default="", help="额外的局部重绘区域（逗号分隔），例如 shoes")
    ap.add_argument("--firmware", default="prompts/gpt-image-optimize/repaint-system-conservative-v5.md")
    ap.add_argument("--source-image", default="", help="覆盖分析 JSON 里的 original 图路径")
    args = ap.parse_args()

    from modules.others.api_backend import (generate_image_aigc2d_gpt,
                                            pick_gpt_image2_size_for_images)
    from utils import post_process as pp
    from utils.analysis_gen import build_first_pass_request, first_pass_sub_dir, resolve_content_text

    result = json.load(open(args.json, encoding="utf-8"))
    source = args.source_image or str(result.get("source_image_path") or "")
    print(f"[1/4] 分析产物: {os.path.basename(args.json)}")
    print(f"      原图: {source if os.path.isfile(source) else '(缺失)'}")
    print(f"      短锚: {len(resolve_content_text(result, 'short'))} 字符")

    styles = json.load(open(os.path.join(BASE, "conf", "config-styles.json"), encoding="utf-8"))
    content_ref = ""
    if args.content_ref == "on":
        content_ref = args.content_image if (args.content_image and os.path.isfile(args.content_image)) else source
        content_ref = content_ref if (content_ref and os.path.isfile(content_ref)) else ""
    # 首图请求与 GUI 走**同一个组装点**（画风说明 + 内容锚 + 排除句 + 渲染语言条款），
    # 内容锚沿用分析产物里的 gpt 字段（GUI 那条链用与 Gemini 同源的分析描述）。
    payload = build_first_pass_request(styles, args.style, result, tier="short",
                                       content_image_path=content_ref)
    ref = str(payload.get("style_ref_path") or "")
    style_clauses = list(payload.get("clauses") or [])
    print(f"[2/4] 画风 {args.style or '(无)'}：说明 {payload['style_chars']} 字符 / "
          f"内容锚 {payload['content_chars']} 字符 / 参考图 {'画风图' if ref else '无'} / "
          f"渲染条款 {len(style_clauses)} 条（{payload.get('clauses_source')}）")
    print(f"      提示词总长 {len(payload['prompt'])} 字符（软上限 2000 / 硬上限 15000）| "
          f"参考图 {len(payload['image_paths'])} 张（内容图 {'有' if content_ref else '无'} + 画风图 "
          f"{'有' if ref else '无'}）")

    if args.preset == "quality":
        args.tone = True
        args.tone_target = "style"
        args.ink = True
        args.ink_target = max(float(args.ink_target), 8.0)
    elif args.preset == "fidelity":
        args.tone = True
        args.tone_target = "photo"
        args.ink = False

    steps = pp.default_pipeline()
    wanted = [s.strip() for s in str(args.steps or "").split(",") if s.strip()]
    for key in ("repaint", "structure", "local"):
        steps[key]["enabled"] = key in wanted
    steps.setdefault("contrast", {})
    steps["contrast"].update({"enabled": float(args.white_contrast) > 0,
                              "amount": float(args.white_contrast), "protect_white": True})
    if float(args.white_contrast) > 0:
        wanted = wanted + ["contrast"]
    steps.setdefault("ink", {})
    steps["ink"].update({"enabled": bool(args.ink), "target_sep": args.ink_target,
                         "max_darken": args.ink_max_darken})
    if args.ink:
        wanted = wanted + ["ink"]
    steps.setdefault("tone", {})
    tone_ref = ref if (args.tone_target == "style" and ref) else source
    steps["tone"].update({"enabled": bool(args.tone), "reference_path": tone_ref,
                          "contrast": args.tone_contrast, "chroma": args.tone_chroma,
                          "highlight_strength": args.highlight_strength,
                          "skin_warm": args.skin_warm, "sat_target_scale": args.sat_scale})
    if args.tone:
        wanted = wanted + ["tone"]
    steps["structure"]["strength"] = args.structure_strength
    regions = [r.strip() for r in str(args.region or "").split(",") if r.strip()]
    regions += [r.strip() for r in str(args.extra_region or "").split(",") if r.strip()]
    steps["local"]["regions"] = regions or ["subject_no_face"]
    steps["local"]["region"] = (steps["local"]["regions"] or ["hair"])[0]
    steps["local"]["feather"] = args.feather
    steps["local"]["resolution"] = args.local_resolution
    steps["local"]["detail_boost"] = bool(args.detail_boost) and not bool(args.no_detail_boost)
    steps["repaint"]["reference_mode"] = ("none" if args.no_dual else
                                          {"line": "line_anchor", "style": "style",
                                           "both": "both", "none": "none"}[args.repaint_ref])
    style_clauses = [str(c) for c in (style_entry.get("repaint_clauses") or []) if str(c).strip()]

    day = pp.date_output_dir()
    if args.use_source_base:
        if not os.path.isfile(source):
            print("❌ --use-source-base 需要可用的原图")
            return 1
        base_path = source
        print(f"[3/4] 跳过 gpt-image 生成，直接用原图作为工序输入")
    else:
        size = args.size
        if str(size).lower() == "auto":
            refs_for_size = list(payload.get("image_paths") or [])
            if source and os.path.isfile(source):
                refs_for_size.insert(0, source)
            size = pick_gpt_image2_size_for_images(refs_for_size)
        print(f"[3/4] gpt-image-2 出首图（quality={args.quality}, size={size}, "
              f"端点={'/images/generations（新建图片）' if args.first_pass_mode == 'generate' else '/images/edits'}）…")
        saved = generate_image_aigc2d_gpt(
            prompt=payload["prompt"], image_paths=list(payload.get("image_paths") or []),
            model="gpt-image-2", size=size, quality=args.quality, output_format="png", n=1,
            api_type="aigc-2d-gpt", file_prefix=os.path.splitext(os.path.basename(args.json))[0][:24],
            save_sub_dir=first_pass_sub_dir(steps), return_metadata=False,
            mode=args.first_pass_mode,
            log_callback=lambda m: None) or []
        if not saved:
            print("❌ 首图生成失败（看 log/<日期>.log）")
            return 1
        base_path = saved[0]
        print(f"      首图: {os.path.relpath(base_path, BASE)}  {_metrics(base_path)}")

    print(f"[4/4] 工序: {', '.join(wanted) or '(无)'} | 区域 {regions} | "
          f"重绘参考 {steps['repaint']['reference_mode']} | 画风条款 {len(style_clauses)} 条")
    outs = pp.run_pipeline([base_path], steps, firmware=args.firmware,
                           style_ref_path=ref, style_clauses=style_clauses,
                           log_callback=lambda m: print("      ", m))
    print("\n最终产物:")
    for path in outs:
        print(f"  {os.path.relpath(path, BASE)}")
        print(f"    {_metrics(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
