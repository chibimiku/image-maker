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
     `--output-dir` 保存完整过程；加 `--publish-final` 时只把最终选中图复制到 `data/<日期>/`
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
    ap.add_argument("--steps", default="repaint",
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
    ap.add_argument("--repaint-ref", default="style",
                    choices=["line", "style", "style-neutral", "both", "none"],
                    help="重绘额外参考：style（默认，完整画风图）/ none / style-neutral / line / both")
    ap.add_argument("--repaint-style-ref", default="",
                    help="仅覆盖 Gemini 重绘使用的画风参考图；用于无角色风格板等受控实验，不改首图参考图")
    ap.add_argument("--repaint-no-style-clauses", action="store_true",
                    help="重绘仍可挂画风参考图，但不追加该画风的渲染条款；用于分离图片参考与文字条款的影响")
    ap.add_argument("--repaint-scope",
                    default="full",
                    choices=["full", "person_only", "person_noface", "details", "lines_only"],
                    help="重绘的「编辑范围」：不裁切不贴回，只在提示词里要求模型保留不该动的部分（§三十一）。"
                         "lines_only（默认）= 只把线条连通、不改色不改内容；person_only = 只重绘人物、背景保持；"
                         "person_noface = 人物可动但脸保持；details = 只修手/丝带/系带/鞋带/项链；full = 不额外限制")
    ap.add_argument("--extra-region", default="", help="额外的局部重绘区域（逗号分隔），例如 shoes")
    ap.add_argument("--firmware", default="prompts/gpt-image-optimize/repaint-system-conservative-v5.md")
    ap.add_argument("--source-image", default="", help="覆盖分析 JSON 里的 original 图路径")
    ap.add_argument("--content-field", default="gpt_image_prompt",
                    help="内容字段；默认身份完整锚 gpt_image_prompt，可指定 english_description 或短锚做对照")
    ap.add_argument("--prompt-file", default="", help="完整首图提示词快照（复现实验用）")
    ap.add_argument("--base-image", default="", help="跳过首图，对已有 GPT 产物跑工序")
    ap.add_argument("--output-dir", default="", help="隔离实验产物和请求清单的目录")
    ap.add_argument("--publish-final", action="store_true",
                    help="只把身份门禁后的最终选中图复制到 data/<当天>/；其余产物留在 --output-dir")
    ap.add_argument("--dry-run", action="store_true", help="保存请求清单，不调用图片接口")
    ap.add_argument("--prompt-recipe", choices=["legacy", "reference"], default="legacy")
    ap.add_argument("--repeat", type=int, default=1, help="同一首图独立复跑后处理次数")
    ap.add_argument("--identity-audit", action="store_true",
                    help="审计首图和最终图是否偏离分析出的角色/服装特征，并保存 JSON")
    ap.add_argument("--identity-correct", action="store_true",
                    help="最终图有高置信身份差异时，最多调用两次 Gemini 做定点修订，每次后重新审计")
    quality_group = ap.add_mutually_exclusive_group()
    quality_group.add_argument("--quality-refine", dest="quality_refine", action="store_true", default=True,
                               help="默认：重绘后做三图+九宫格质量审计，必要时用当前图+GPT首图修订一次")
    quality_group.add_argument("--no-quality-refine", dest="quality_refine", action="store_false",
                               help="关闭重绘后的质量审计与一次定点修订")
    args = ap.parse_args()

    from modules.others.api_backend import (generate_image_aigc2d_gpt,
                                            pick_gpt_image2_size_for_images)
    from utils import post_process as pp
    from utils.analysis_gen import (build_first_pass_request, first_pass_sub_dir,
                                    publish_final_output, resolve_content_text)

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
                                       content_text=str(result.get(args.content_field) or ""),
                                       prompt_recipe=args.prompt_recipe,
                                       content_image_path=content_ref)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            payload["prompt"] = f.read()
    ref = str(payload.get("style_ref_path") or "")
    repaint_style_ref = (os.path.abspath(args.repaint_style_ref)
                         if args.repaint_style_ref else ref)
    if args.repaint_style_ref and not os.path.isfile(repaint_style_ref):
        raise FileNotFoundError(repaint_style_ref)
    style_clauses = list(payload.get("clauses") or [])
    if args.repaint_no_style_clauses:
        style_clauses = []
    if args.repaint_ref == "style-neutral":
        from utils.style_gpt import resolve_neutral_repaint_clauses
        style_clauses, _ = resolve_neutral_repaint_clauses((styles or {}).get(args.style))
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
                                           "style-neutral": "style_neutral",
                                           "both": "both", "none": "none"}[args.repaint_ref])
    steps["repaint"]["scope"] = str(args.repaint_scope)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    size = args.size
    if str(size).lower() == "auto":
        size = pick_gpt_image2_size_for_images([source] if os.path.isfile(source) else payload["image_paths"])
    manifest = {"analysis_json": os.path.abspath(args.json), "content_field": args.content_field,
                "source": source, "style": args.style, "request": payload,
                "size": size, "quality": args.quality, "mode": args.first_pass_mode,
                "steps": steps, "firmware": args.firmware, "base": args.base_image,
                "repaint_style_ref": repaint_style_ref,
                "outputs": [], "quality_refine": {}, "identity": {},
                "published_final": "", "status": "planned"}
    def save_manifest():
        if output_dir:
            with open(os.path.join(output_dir, "request.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
    save_manifest()
    if args.dry_run:
        print("预检完成；未调用图片接口。")
        return 0
    if args.base_image:
        if not os.path.isfile(args.base_image):
            raise FileNotFoundError(args.base_image)
        base_path = os.path.abspath(args.base_image)
    elif args.use_source_base:
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
            save_sub_dir=output_dir or first_pass_sub_dir(steps), return_metadata=False,
            mode=args.first_pass_mode,
            log_callback=lambda m: None) or []
        if not saved:
            manifest["status"] = "first_pass_failed"
            save_manifest()
            print("❌ 首图生成失败（看 log/<日期>.log）")
            return 1
        base_path = saved[0]
        print(f"      首图: {os.path.relpath(base_path, BASE)}  {_metrics(base_path)}")

    manifest["base"] = base_path
    manifest["status"] = "first_pass_complete"
    if args.identity_audit or args.identity_correct:
        from utils.identity_audit import audit_image_identity, build_identity_correction_prompt
        try:
            first_audit = audit_image_identity(base_path, result, expected_prompt=payload["prompt"])
            first_audit["correction_prompt"] = build_identity_correction_prompt(first_audit, 1, 2)
        except Exception as exc:  # 审计是质量门，不应让已成功的生图链报废
            first_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                           "differences": [], "audit_error": f"{type(exc).__name__}: {exc}"}
        manifest["identity"]["first"] = first_audit
        if output_dir:
            with open(os.path.join(output_dir, "identity-first.json"), "w", encoding="utf-8") as f:
                json.dump(first_audit, f, ensure_ascii=False, indent=2)
        first_status = "审计失败" if first_audit.get("audit_error") else ("有差异" if first_audit["mismatch"] else "通过")
        print(f"      首图身份审计: {first_status} "
              f"({first_audit['severity']}, {first_audit['confidence']:.2f})")
    save_manifest()

    print(f"[4/4] 工序: {', '.join(wanted) or '(无)'} | 区域 {regions} | "
          f"重绘参考 {steps['repaint']['reference_mode']} | 画风条款 {len(style_clauses)} 条")
    outs = []
    for trial in range(max(1, args.repeat)):
        outs.extend(pp.run_pipeline([base_path], steps, firmware=args.firmware,
                           style_ref_path=repaint_style_ref, style_clauses=style_clauses,
                           final_dir=output_dir,
                           work_dir=os.path.join(output_dir, f"steps-{trial + 1}") if output_dir else None,
                           resume=False,
                           log_callback=lambda m: print("      ", m)))
    manifest["outputs"] = outs
    if args.quality_refine and outs and repaint_style_ref and os.path.isfile(repaint_style_ref):
        from modules.others.api_backend import generate_image_repaint
        from utils.refine_quality import (audit_refine_quality, build_quality_correction_prompt,
                                          should_refine_quality)
        try:
            quality_audit = audit_refine_quality(
                base_path, outs[-1], repaint_style_ref, first_pass_prompt=payload["prompt"])
            manifest["quality_refine"]["before"] = quality_audit
            if output_dir:
                with open(os.path.join(output_dir, "refine-quality-audit-0.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(quality_audit, f, ensure_ascii=False, indent=2)
            if should_refine_quality(quality_audit):
                quality_prompt = build_quality_correction_prompt(quality_audit)
                refined = generate_image_repaint(
                    [outs[-1]], resolution="2K", aspect_ratio=pp.snapped_aspect_ratio(outs[-1]),
                    prompt=quality_prompt,
                    use_detail_suffix=False, save_sub_dir=output_dir or os.path.dirname(outs[-1]),
                    file_prefix="quality-refine", extra_reference_paths=[base_path]) or []
                if refined:
                    outs.extend(refined)
                    quality_after = audit_refine_quality(
                        base_path, outs[-1], repaint_style_ref, first_pass_prompt=payload["prompt"])
                    manifest["quality_refine"]["after"] = quality_after
                    if output_dir:
                        with open(os.path.join(output_dir, "refine-quality-audit-1.json"),
                                  "w", encoding="utf-8") as f:
                            json.dump(quality_after, f, ensure_ascii=False, indent=2)
                    print("      质量门禁: 已用当前图 + GPT 首图做一次文字驱动修订（未再次发送画风图）")
            else:
                print("      质量门禁: 未发现需定点修复的高置信问题")
        except Exception as exc:
            manifest["quality_refine"]["audit_error"] = f"{type(exc).__name__}: {exc}"
            print(f"      质量门禁失败，保留当前图: {type(exc).__name__}: {exc}")
        manifest["outputs"] = outs
        manifest["selected_output"] = outs[-1]
        save_manifest()
    if (args.identity_audit or args.identity_correct) and outs:
        from utils.identity_audit import (audit_image_identity, build_identity_correction_prompt,
                                          identity_gate_action)
        from modules.others.api_backend import generate_image_repaint
        try:
            final_audit = audit_image_identity(outs[-1], result, expected_prompt=payload["prompt"])
            final_audit["correction_prompt"] = build_identity_correction_prompt(final_audit, 1, 2)
        except Exception as exc:  # 保留最终图并把审计故障写进清单
            final_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                           "differences": [], "audit_error": f"{type(exc).__name__}: {exc}",
                           "correction_prompt": ""}
        manifest["identity"]["final"] = final_audit
        if output_dir:
            with open(os.path.join(output_dir, "identity-final.json"), "w", encoding="utf-8") as f:
                json.dump(final_audit, f, ensure_ascii=False, indent=2)
        final_status = "审计失败" if final_audit.get("audit_error") else ("有差异" if final_audit["mismatch"] else "通过")
        print(f"      最终图身份审计: {final_status} "
              f"({final_audit['severity']}, {final_audit['confidence']:.2f})")
        action = identity_gate_action(final_audit)
        manifest["identity"]["action"] = action
        manifest["selected_output"] = outs[-1]
        current = outs[-1]
        current_audit = final_audit
        correction_rounds = []
        if args.identity_correct:
            for correction_round in range(1, 3):
                if identity_gate_action(current_audit) != "correct":
                    break
                prompt = build_identity_correction_prompt(
                    current_audit, iteration=correction_round, max_iterations=2)
                corrected = generate_image_repaint(
                    [current], resolution="2K", aspect_ratio=pp.snapped_aspect_ratio(current),
                    prompt=prompt,
                    use_detail_suffix=False, save_sub_dir=output_dir or os.path.dirname(current),
                    file_prefix=f"identity-correct-{correction_round}") or []
                if not corrected:
                    break
                current = corrected[-1]
                outs.extend(corrected)
                try:
                    current_audit = audit_image_identity(
                        current, result, expected_prompt=payload["prompt"])
                    current_audit["correction_prompt"] = build_identity_correction_prompt(
                        current_audit, min(2, correction_round + 1), 2)
                except Exception as exc:
                    current_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                                     "stable_anchors": [], "differences": [],
                                     "audit_error": f"{type(exc).__name__}: {exc}"}
                correction_rounds.append({"round": correction_round, "image": current,
                                          "audit": current_audit})
                if output_dir:
                    with open(os.path.join(output_dir, f"identity-corrected-{correction_round}.json"),
                              "w", encoding="utf-8") as f:
                        json.dump(current_audit, f, ensure_ascii=False, indent=2)
                print(f"      定点修订 {correction_round}/2 复审: "
                      f"{'仍有差异' if current_audit.get('mismatch') else '通过'} "
                      f"({current_audit.get('severity')}, {float(current_audit.get('confidence') or 0):.2f})")
        manifest["identity"]["correction_rounds"] = correction_rounds
        manifest["selected_output"] = current
        if correction_rounds and identity_gate_action(current_audit) != "accept":
            print("      身份门禁: 两轮后仍有差异，按不回退策略保留最后一轮修订图")
    manifest["status"] = "complete" if outs else "pipeline_failed"
    selected = str(manifest.get("selected_output") or (outs[-1] if outs else ""))
    if args.publish_final and selected and os.path.isfile(selected):
        published = publish_final_output(
            selected, style_name=args.style, process_dir=output_dir or os.path.dirname(selected))
        manifest["published_final"] = published
        print(f"      发布最终图: {os.path.relpath(published, BASE)}")
    save_manifest()
    print("\n最终产物:")
    for path in outs:
        print(f"  {os.path.relpath(path, BASE)}")
        print(f"    {_metrics(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
