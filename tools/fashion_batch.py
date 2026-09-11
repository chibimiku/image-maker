# -*- coding: utf-8 -*-
"""Fashion 批量采集-生图无头 CLI（原 data/fashion_pipeline_run|sexy|backless.py 的合并版）。

每个主题只改 prompts/fashion-batch-themes.json 里的画像（character_spec /
extra_prompt / save_subdir / collect_base / prefix / count / 是否出图后分析），
不再为单个实验复制脚本。

流程（与 FashionCollectorWidget._run_single_pipeline 一致）：
  随机品牌 → lolibrary 采集（dress/shoes/socks）→ 随机构图+画风+角色固定描述
  → generate_image_aigc2d → 可选出图后 Step 1 分析（<原图>_analysis.json）
  → 写入 data/fashion-collector/<base>/<prefix>_run_manifest.json

用法:
    python tools/fashion_batch.py --profile blonde_whitesilk
    python tools/fashion_batch.py --profile sexy_blonde --count 3
    python tools/fashion_batch.py --profile backless_blonde --no-proxy
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PROFILES_PATH = os.path.join(BASE_DIR, "prompts", "fashion-batch-themes.json")
DEFAULT_THEME = "甜美洛丽塔"
DEFAULT_PARTS = ["dress", "shoes", "socks"]


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_profiles() -> dict:
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("profiles", {}) or {}


def load_styles(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): v for k, v in json.load(f).items()}


def build_context(bundle, styles_data, profile: dict, theme: str) -> dict:
    from modules.fashion_collection.generation_plan import (
        build_reference_prompt,
        build_scene_and_character_description,
        generate_random_composition,
        resolve_prompt_and_instructions,
        resolve_style_bundle,
    )
    from modules.fashion_collection.theme_profiles import get_theme_profile

    theme_profile = get_theme_profile(theme)
    instructions_text, composition_ratio = generate_random_composition()
    _, style_text = resolve_style_bundle("", styles_data, theme_profile=theme_profile)
    final_prompt_base, final_instructions = resolve_prompt_and_instructions(
        profile.get("extra_prompt", ""), instructions_text,
        theme_profile=theme_profile, style_text=style_text,
    )
    scene_text, _ = build_scene_and_character_description(bundle, theme_profile, 1)
    character_text = profile["character_spec"]
    final_prompt = build_reference_prompt(final_prompt_base, bundle, scene_text, character_text)
    return {
        "final_prompt": final_prompt,
        "final_instructions": final_instructions,
        "scene_text": scene_text,
        "character_text": character_text,
        "composition_ratio": composition_ratio,
    }


def analyze_step1_save(image_path: str) -> str | None:
    from modules.image_analysis.analysis_pipeline import analyze_image_step1, save_step1_analysis

    log(f"[分析] Step 1 分析: {os.path.basename(image_path)}")
    try:
        result = analyze_image_step1(image_path, api_type="aigc2d", log_callback=log)
        return save_step1_analysis(result, image_path, log_callback=log)
    except Exception as exc:  # noqa: BLE001
        log(f"[分析] 失败: {exc}")
        return None


def run_one(index: int, brand_slug: str, styles_data, profile: dict, theme: str,
            output_base: str, use_proxy: bool, ratio: str, dry_run: bool) -> dict:
    from modules.fashion_collection.collection_service import FashionCollectionService
    from modules.others.api_backend import generate_image_aigc2d

    log(f"[run {index}] ============ 开始，品牌: {brand_slug} ============")
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{index}"
    output_dir = os.path.join(output_base, profile.get("collect_base", "lolibrary"), "lolibrary", now)
    parts = profile.get("parts") or DEFAULT_PARTS

    if dry_run:
        log(f"[run {index}] [dry-run] 计划: 采集 {parts} @ lolibrary（品牌 {brand_slug}）"
            f" → 生图到 data/<今天>/<{profile.get('save_subdir')}>/"
            f"<{profile.get('prefix')}>_batch{index}_*.png（比例 {ratio}）"
            f"{'，出图后 Step 1 分析' if profile.get('analyze_step1') else ''}")
        return {"index": index, "brand": brand_slug, "ok": True, "dry_run": True,
                "saved_files": [], "missing_parts": [], "output_dir": output_dir}

    service = FashionCollectionService(timeout=15)
    if use_proxy and profile.get("proxy"):
        service.set_proxy_url(profile["proxy"])
    service.enable_color_match = False  # 避免额外 LLM 颜色调用，保证批量稳定

    bundle = service.collect_bundle(
        site_key="lolibrary",
        brand_slug=brand_slug,
        output_dir=output_dir,
        max_pages=1,
        preferred_parts=parts,
        theme=theme,
        log_callback=log,
    )

    if not bundle.assets:
        log(f"[run {index}] 未采集到素材，跳过生成")
        return {"index": index, "brand": brand_slug, "ok": False, "reason": "no_assets",
                "missing_parts": list(bundle.missing_parts)}

    missing = list(bundle.missing_parts)
    log(f"[run {index}] 采集到 {len(bundle.assets)} 件素材，缺失={missing}")
    for a in bundle.assets:
        log(f"    [{a.part}] {a.item.title} -> {a.local_path}")

    ctx = build_context(bundle, styles_data, profile, theme)
    image_paths = [a.local_path for a in bundle.assets if os.path.isfile(a.local_path)]
    log(f"[run {index}] 构图比例(仅供参考): {ctx['composition_ratio']}")
    log(f"[run {index}] 开始生成，参考图 {len(image_paths)} 张，prompt {len(ctx['final_prompt'])} chars")

    result = generate_image_aigc2d(
        prompt=ctx["final_prompt"],
        image_paths=image_paths,
        aspect_ratio=ratio,
        instructions=ctx["final_instructions"],
        resolution=None,
        api_type="aigc2d",
        save_sub_dir=profile.get("save_subdir", "fashion-generate"),
        file_prefix=f"{profile.get('prefix', 'fashion')}_batch{index}",
        return_metadata=True,
        log_callback=log,
    )

    saved: list[str] = []
    if isinstance(result, dict):
        saved = list(result.get("saved_files") or [])
        raw = os.path.join(output_dir, f"{profile.get('prefix', 'fashion')}_batch{index}_result.json")
        try:
            with open(raw, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log(f"[run {index}] 原始结果: {raw}")
        except Exception as exc:  # noqa: BLE001
            log(f"[run {index}] 结果写入失败: {exc}")
    elif isinstance(result, list):
        saved = list(result or [])
    log(f"[run {index}] 保存图片: {saved}")

    analysis_files: list[str] = []
    if profile.get("analyze_step1"):
        for path in saved:
            analysis_path = analyze_step1_save(path)
            if analysis_path:
                analysis_files.append(analysis_path)

    return {
        "index": index, "brand": brand_slug, "ok": bool(saved),
        "saved_files": saved, "analysis_files": analysis_files,
        "missing_parts": missing, "output_dir": output_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fashion 批量采集-生图无头 CLI（主题画像见 prompts/fashion-batch-themes.json）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--profile", required=True, help="画像名（prompts/fashion-batch-themes.json 的 profiles 键）")
    parser.add_argument("--count", type=int, default=0, help="批量张数（0=用画像 count 或默认 2）")
    parser.add_argument("--ratio", default="3:4", help="出图比例（默认 3:4）")
    parser.add_argument("--theme", default="", help="覆盖画像主题（默认用画像 theme）")
    parser.add_argument("--no-proxy", action="store_true", help="不使用画像里的代理")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不采集不生图")
    args = parser.parse_args()

    profiles = load_profiles()
    profile = profiles.get(args.profile)
    if not profile:
        available = ", ".join(profiles.keys()) or "(空)"
        log(f"错误: 未找到画像 '{args.profile}'，可用: {available}")
        return 1

    theme = args.theme or profile.get("theme", DEFAULT_THEME)
    run_count = args.count or int(profile.get("count", 2))
    output_base = os.path.join(BASE_DIR, "data", "fashion-collector")
    os.makedirs(output_base, exist_ok=True)

    styles_data = load_styles(os.path.join(BASE_DIR, "conf", "config-styles.json"))

    from modules.fashion_collection.brand_scraper import pick_random_brands

    brands = pick_random_brands(run_count, project_root=BASE_DIR)
    if not brands:
        log("品牌缓存为空，回退固定品牌 angelic-pretty")
        brands = ["angelic-pretty"] * run_count
    log(f"随机品牌: {brands}")

    results = []
    for i, brand in enumerate(brands, start=1):
        try:
            results.append(run_one(i, brand, styles_data, profile, theme, output_base,
                                   use_proxy=not args.no_proxy, ratio=args.ratio, dry_run=args.dry_run))
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"[run {i}] 异常: {exc}")
            traceback.print_exc()
            results.append({"index": i, "brand": brand, "ok": False, "reason": str(exc)})

    manifest_dir = os.path.join(output_base, profile.get("collect_base", "lolibrary"))
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(),
        "profile": args.profile,
        "theme": theme,
        "character_spec": profile.get("character_spec", ""),
        "results": results,
    }
    manifest_path = os.path.join(manifest_dir, f"{profile.get('prefix', args.profile)}_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"清单已写入: {manifest_path}")

    print("\n========== 结果 ==========", flush=True)
    ok = 0
    for r in results:
        if r.get("ok") and not r.get("dry_run"):
            status = f"OK {r.get('saved_files')}"
            if r.get("analysis_files"):
                status += f" + 分析 {len(r['analysis_files'])} 张"
        elif r.get("dry_run"):
            status = "DRY-RUN"
        else:
            status = f"FAIL {r.get('reason')}"
        print(f"  [#{r['index']}] {status}", flush=True)
        if r.get("ok") and not r.get("dry_run"):
            ok += 1
    print(f"完成: {ok}/{len(results)} 张成功", flush=True)
    if args.dry_run:
        return 0
    return 0 if ok == run_count else 1


if __name__ == "__main__":
    sys.exit(main())
