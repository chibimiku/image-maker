#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpt-image-2 生图/编辑 无头 CLI（new.aigc2d 与 autodl 双站点）。

无 PyQt 依赖，纯 python 直接调用 `modules/others/api_backend.py` 的通道函数，
便于在命令行 / 脚本 / 无 GUI 环境（包括只能跑 python 出网的机器）里出图。

示例：
  # 文生图（aigc2d，纵向）
  python tools/gpt_image2_gen.py --prompt "一位少女站在海边，黄昏" --size 1024x1536

  # 参考图生图（aigc2d，自动走 /images/edits）
  python tools/gpt_image2_gen.py --prompt "换成海边背景" --image data/me.png

  # 切 autodl 站点做编辑
  python tools/gpt_image2_gen.py --site autodl --mode edit --prompt "只改天空" --image data/me.png

  # 批量：每行一条提示词，可用 `WxH|提示词` 覆盖单张尺寸
  python tools/gpt_image2_gen.py --prompt-file prompts.txt --output-subdir mytest --prefix demo

  # 只看解析结果，不发请求
  python tools/gpt_image2_gen.py --prompt "test" --dry-run

  # 看两个站点的配置状态
  python tools/gpt_image2_gen.py --list-sites

  # 看某站点当前可用的 gpt-image 系列模型（实时 GET /v1/models）
  python tools/gpt_image2_gen.py --list-models
  python tools/gpt_image2_gen.py --site autodl --list-models
"""
import argparse
import json
import os
import re
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.others.api_backend import (  # noqa: E402  (需要在 sys.path 之后导入)
    GPT_IMAGE2_MAX_REFERENCE_IMAGES,
    GPT_IMAGE2_MODEL_NOTES,
    GPT_IMAGE2_OUTPUT_FORMATS,
    GPT_IMAGE2_QUALITIES,
    GPT_IMAGE2_SITE_AIGC2D,
    GPT_IMAGE2_SITE_API_TYPES,
    GPT_IMAGE2_SITE_AUTODL,
    GPT_IMAGE2_SITE_DEFAULT_SIZES,
    GPT_IMAGE2_SITE_FILE_PREFIXES,
    GPT_IMAGE2_SITE_MODELS,
    GPT_IMAGE2_SITE_SAVE_SUB_DIRS,
    GPT_IMAGE2_SITE_SIZES,
    build_gpt_image2_payload,
    generate_image_aigc2d_gpt,
    generate_image_openai_image,
    generate_image_repaint,
    get_api_config,
    is_gpt_image_model,
    list_available_models,
    normalize_gpt_image2_size,
    resolve_images_endpoint,
    resolve_models_endpoint,
)
from utils.gpt_image_optimize import (  # noqa: E402
    ASPECT_RATIO_AUTO,
    build_repaint_prompt,
    load_config as load_repaint_config,
    resolve_aspect_ratio,
)

DEFAULT_CONFIG = os.path.join(BASE_DIR, "conf", "config-image.json")
SITE_CHOICES = (GPT_IMAGE2_SITE_AIGC2D, GPT_IMAGE2_SITE_AUTODL)
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
SIZE_LINE_RE = re.compile(r"^(auto|\d+x\d+)\s*$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt_image2_gen.py",
        description="gpt-image-2 生图/编辑 CLI（站点: new.aigc2d / autodl）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--site", choices=SITE_CHOICES, default=GPT_IMAGE2_SITE_AIGC2D,
                        help="站点，默认 new.aigc2d（apis.aigc-2d-gpt）")
    parser.add_argument("--mode", choices=("generate", "edit"), default="generate",
                        help="generate=生图（挂了 --image 会自动走 edits 垫图）; edit=编辑（必须有 --image）")
    parser.add_argument("--prompt", action="append", default=[], help="提示词，可重复")
    parser.add_argument("--prompt-file", help="提示词文件：每行一条，`#` 注释，可用 `WxH|提示词` 覆盖单张尺寸")
    parser.add_argument("--image", action="append", default=[], help=f"参考图/原图，可重复，最多 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张")
    parser.add_argument("--size", help="尺寸；aigc2d 仅 3 档(1024x1024/1536x1024/1024x1536)，autodl 另含 auto/1792x1024")
    parser.add_argument("--quality", choices=GPT_IMAGE2_QUALITIES, help="画质，默认取配置或 high")
    parser.add_argument("--output-format", choices=GPT_IMAGE2_OUTPUT_FORMATS, help="输出格式，默认取配置或 png")
    parser.add_argument("--n", type=int, help="张数，仅 new.aigc2d 站点有效（autodl 一次一张）")
    parser.add_argument("--model", help="模型名，默认取配置")
    parser.add_argument("--output-subdir", help="data/<日期>/ 下的子目录，默认按站点")
    parser.add_argument("--prefix", help="输出文件名前缀，默认按站点")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件，默认 conf/config-image.json")
    parser.add_argument("--timeout", type=int, help="覆盖该站点的请求超时秒数（不写回配置文件）")
    parser.add_argument("--dry-run", action="store_true", help="只打印解析出的端点与请求体，不发请求")
    parser.add_argument("--json", action="store_true", help="最后额外打印一段 JSON 汇总")
    parser.add_argument("--list-sites", action="store_true", help="打印两个站点的配置状态后退出")
    parser.add_argument("--list-models", action="store_true", help="实时拉取站点模型列表（GET /v1/models）后退出")
    # ---- gpt-image 产物优化（重绘提线）：与 GUI「重绘」模式共用 prompts/gpt-image-optimize/ ----
    parser.add_argument("--repaint", action="store_true",
                        help="对 --image 指定的产物跑 Gemini 重绘提线（修手/连通发丝/保住蕾丝结构）")
    parser.add_argument("--repaint-model", help="重绘模型，默认取 prompts/gpt-image-optimize/config.json")
    parser.add_argument("--repaint-resolution", help="重绘分辨率 1K/2K/4K，默认取配置")
    parser.add_argument("--repaint-aspect",
                        help=f"重绘输出宽高比，默认 auto（不传该字段，跟随输入图比例）；"
                             f"要强制可填 1:1/3:2/2:3/9:16/16:9 等")
    parser.add_argument("--repaint-repeat", type=int, help="每张源图重绘几次（抽卡用），默认取配置")
    parser.add_argument("--repaint-prompt-file", help="临时覆盖重绘提示词文件（相对仓库根或绝对路径）")
    parser.add_argument("--repaint-show-prompt", action="store_true", help="打印最终重绘提示词后退出")
    return parser


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _read_prompt_file(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            size_override = None
            if "|" in line:
                head, tail = line.split("|", 1)
                if SIZE_LINE_RE.match(head.strip()):
                    size_override = head.strip()
                    line = tail.strip()
            if line:
                items.append((line, size_override))
    return items


def _collect_prompts(args) -> list:
    """返回 [(prompt, size_override)]。"""
    items = [(text, None) for text in (args.prompt or []) if str(text).strip()]
    if args.prompt_file:
        if not os.path.isfile(args.prompt_file):
            raise ValueError(f"提示词文件不存在: {args.prompt_file}")
        items.extend(_read_prompt_file(args.prompt_file))
    return items


def _resolve_size(args, site: str, size_override: str = None) -> str:
    raw = size_override or args.size or ""
    if site == GPT_IMAGE2_SITE_AIGC2D:
        resolved = normalize_gpt_image2_size(size=raw) if raw else GPT_IMAGE2_SITE_DEFAULT_SIZES[site]
        if raw and str(raw).strip().lower() not in (resolved, *GPT_IMAGE2_SITE_SIZES[site]):
            _emit(f"[提示] aigc2d 只支持 3 档尺寸，'{raw}' 已收敛为 {resolved}")
        return resolved
    allowed = GPT_IMAGE2_SITE_SIZES[site]
    if not raw:
        return GPT_IMAGE2_SITE_DEFAULT_SIZES[site]
    if raw.strip().lower() not in allowed:
        _emit(f"[提示] autodl 站点未登记尺寸 '{raw}'，仍按原样发送（允许清单: {', '.join(allowed)}）")
    return raw.strip()


def _config_with_timeout(config_path: str, api_type: str, timeout: int):
    """把 timeout 临时覆盖进配置副本，返回 (可用路径, 清理函数)。"""
    if not timeout:
        return config_path, (lambda: None)
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    apis = data.setdefault("apis", {})
    node = apis.setdefault(api_type, {})
    node["timeout"] = int(timeout)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(data, handle, ensure_ascii=False, indent=4)
    finally:
        handle.close()
    return handle.name, (lambda: os.path.exists(handle.name) and os.remove(handle.name))


def cmd_list_sites(config_path: str) -> int:
    for site in SITE_CHOICES:
        api_type = GPT_IMAGE2_SITE_API_TYPES[site]
        cfg = get_api_config(config_path=config_path, api_type=api_type)
        key_state = "已配置" if str(cfg.get("api_key") or "").strip() else "未配置"
        _emit(f"站点 {site}")
        _emit(f"  api_type : {api_type}")
        _emit(f"  base_url : {cfg.get('base_url') or '(未配置)'}")
        _emit(f"  model    : {cfg.get('model') or '(未配置)'}")
        _emit(f"  api_key  : {key_state}")
        _emit(f"  timeout  : {cfg.get('timeout', '(默认)')}s")
        _emit(f"  尺寸     : {', '.join(GPT_IMAGE2_SITE_SIZES[site])}")
        _emit(f"  常用模型 : {', '.join(GPT_IMAGE2_SITE_MODELS[site])}")
        _emit(f"  输出目录 : data/<日期>/{GPT_IMAGE2_SITE_SAVE_SUB_DIRS[site]}/")
    return EXIT_OK


def cmd_list_models(config_path: str, site: str, timeout: int = 20) -> int:
    """实时拉取站点模型列表，只展示 gpt-image / dall-e 系列（其余是文本/视频模型）。"""
    api_type = GPT_IMAGE2_SITE_API_TYPES[site]
    cfg = get_api_config(config_path=config_path, api_type=api_type)
    if not str(cfg.get("api_key") or "").strip():
        _emit(f"[错误] apis.{api_type} 缺少 api_key（配置文件: {config_path}）")
        return EXIT_USAGE
    endpoint = resolve_models_endpoint(str(cfg.get("base_url") or ""), api_type=api_type)
    _emit(f"站点 {site}  {endpoint}")
    try:
        models = list_available_models(api_type=api_type, config_path=config_path, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - CLI 需要把异常转成退出码
        _emit(f"[错误] 获取模型列表失败: {type(exc).__name__}: {exc}")
        return EXIT_FAIL
    family = [name for name in models if is_gpt_image_model(name)]
    _emit(f"共 {len(models)} 个模型，gpt-image / dall-e 系列 {len(family)} 个：")
    for name in family:
        note = GPT_IMAGE2_MODEL_NOTES.get(name)
        _emit(f"  - {name}" + (f"    # {note}" if note else ""))
    others = [name for name in models if name not in family]
    if others:
        _emit(f"  （另有 {len(others)} 个文本/视频等模型未列出）")
    return EXIT_OK


def _describe_request(args, site: str, api_type: str, prompt: str, size: str, quality: str, output_format: str) -> dict:
    cfg = get_api_config(config_path=args.config, api_type=api_type)
    api_base = str(cfg.get("base_url") or "").rstrip("/")
    images = [p for p in args.image]
    endpoint = resolve_images_endpoint(api_base, has_images=bool(images), api_type=api_type)
    model = args.model or str(cfg.get("model") or "gpt-image-2")
    info = {
        "site": site,
        "api_type": api_type,
        "endpoint": endpoint,
        "mode": "edits" if images else "generations",
        "model": model,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "image_count": len(images),
    }
    if site == GPT_IMAGE2_SITE_AIGC2D:
        info["n"] = args.n or 1
        info["payload"] = build_gpt_image2_payload(
            prompt=prompt, model=model, size=size, quality=quality, output_format=output_format, n=args.n or 1
        )
    else:
        info["n"] = 1
        info["payload_note"] = "autodl: JSON/form 字段 model/prompt/size/quality/n=1/response_format=b64_json/output_format"
    return info


def _run_one(args, site: str, api_type: str, prompt: str, size: str) -> list:
    quality = args.quality or ""
    output_format = args.output_format or ""
    sub_dir = args.output_subdir or GPT_IMAGE2_SITE_SAVE_SUB_DIRS[site]
    prefix = args.prefix or GPT_IMAGE2_SITE_FILE_PREFIXES[site]
    images = list(args.image or [])
    config_path, cleanup = _config_with_timeout(args.config, api_type, args.timeout)
    try:
        if site == GPT_IMAGE2_SITE_AIGC2D:
            return generate_image_aigc2d_gpt(
                prompt=prompt,
                image_paths=images,
                model=args.model or None,
                size=size,
                quality=quality or None,
                output_format=output_format or None,
                n=args.n,
                mode=args.mode,
                api_type=api_type,
                save_sub_dir=sub_dir,
                file_prefix=prefix,
                config_path=config_path,
                log_callback=_emit,
            ) or []
        return generate_image_openai_image(
            prompt=prompt,
            image_paths=images,
            model=args.model or "gpt-image-2",
            aspect_ratio="1:1",
            instructions="",
            api_type=api_type,
            save_sub_dir=sub_dir,
            file_prefix=prefix,
            size=size,
            quality=quality or None,
            output_format=output_format or None,
            config_path=config_path,
        ) or []
    finally:
        cleanup()


def cmd_repaint(args) -> int:
    """gpt-image 产物优化：对 --image 逐张走 Gemini 重绘提线（提示词来自 prompts/gpt-image-optimize/）。"""
    conf = load_repaint_config()
    override_file = str(args.repaint_prompt_file or "").strip()
    if override_file:
        if not os.path.isabs(override_file):
            override_file = os.path.join(BASE_DIR, override_file)
        if not os.path.isfile(override_file):
            _emit(f"[错误] --repaint-prompt-file 不存在: {override_file}")
            return EXIT_USAGE
        with open(override_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        suffix_text = ""      # 覆盖文件视为完整提示词，不再叠加细节后缀
        prompt_label = override_file
    else:
        prompt_text = build_repaint_prompt(conf)
        suffix_text = None    # None = 由后端按配置再补后缀（保持与 GUI 一致）
        prompt_label = str(conf.get("system_prompt"))

    if args.repaint_show_prompt:
        _emit(f"提示词来源: {prompt_label}")
        _emit(f"提示词长度: {len(prompt_text)} 字符")
        _emit("-" * 60)
        _emit(prompt_text)
        if suffix_text is None:
            from utils.gpt_image_optimize import read_prompt_relative
            suffix_relative = str(conf.get("detail_suffix") or "").strip()
            if suffix_relative and conf.get("use_detail_suffix", True):
                try:
                    _emit("-" * 60)
                    _emit(f"细节后缀({suffix_relative}):")
                    _emit(read_prompt_relative(suffix_relative))
                except FileNotFoundError:
                    pass
        return EXIT_OK

    sources = [p for p in (args.image or []) if p]
    if not sources:
        _emit("[错误] --repaint 需要 --image 指定至少 1 张源图（gpt-image 的产物）")
        return EXIT_USAGE
    missing = [p for p in sources if not os.path.isfile(p)]
    if missing:
        _emit(f"[错误] 源图不存在: {', '.join(missing)}")
        return EXIT_USAGE

    model = args.repaint_model or str(conf.get("model") or "")
    resolution = args.repaint_resolution or str(conf.get("resolution") or "")
    repeat = args.repaint_repeat or int(conf.get("repeat") or 1)
    api_type = str(conf.get("api_type") or "")
    cfg = get_api_config(config_path=args.config, api_type=api_type)
    if not str(cfg.get("api_key") or "").strip():
        _emit(f"[错误] apis.{api_type} 缺少 api_key（配置文件: {args.config}）")
        return EXIT_USAGE

    _emit(f"重绘 {len(sources)} 张 -> 模型 {model} @{resolution}，每张 {repeat} 次")
    resolved_aspect = resolve_aspect_ratio(
        {"aspect_ratio": args.repaint_aspect} if args.repaint_aspect else conf, ASPECT_RATIO_AUTO
    )
    _emit(
        "输出比例: " + ("auto（跟随输入图，不下发 aspectRatio）" if resolved_aspect == ASPECT_RATIO_AUTO
                        else f"{resolved_aspect}（强制）")
    )
    if override_file:
        _emit(f"提示词来源: {prompt_label}（覆盖文件，{len(prompt_text)} 字符）")
    else:
        _emit(f"提示词来源: {prompt_label} + 细节后缀（合计 {len(prompt_text)} 字符）")
    if args.dry_run:
        for path in sources:
            _emit(f"DRY-RUN 将重绘: {path}")
        return EXIT_OK

    saved = generate_image_repaint(
        source_paths=sources,
        api_type=api_type,
        config_path=args.config,
        model=model,
        resolution=resolution,
        aspect_ratio=resolve_aspect_ratio(
            {"aspect_ratio": args.repaint_aspect} if args.repaint_aspect else conf, ASPECT_RATIO_AUTO
        ),
        prompt=prompt_text if override_file else None,
        prompt_suffix=suffix_text,
        repeat=repeat,
        save_sub_dir=args.output_subdir or None,
        file_prefix=args.prefix or None,
        log_callback=_emit,
    ) or []
    for path in saved:
        _emit(f"SAVED {path}")
    _emit(f"完成: 源图 {len(sources)} 张 -> 重绘产物 {len(saved)} 张")
    if args.json:
        _emit(json.dumps({"repaint": True, "sources": sources, "saved_files": saved},
                         ensure_ascii=False, indent=2))
    return EXIT_OK if saved else EXIT_FAIL


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.config):
        _emit(f"[错误] 找不到配置文件: {args.config}")
        return EXIT_USAGE
    if args.list_sites:
        return cmd_list_sites(args.config)
    if args.list_models:
        return cmd_list_models(args.config, args.site, timeout=args.timeout or 20)
    if args.repaint or args.repaint_show_prompt:
        return cmd_repaint(args)

    site = args.site
    api_type = GPT_IMAGE2_SITE_API_TYPES[site]

    try:
        prompts = _collect_prompts(args)
    except ValueError as exc:
        _emit(f"[错误] {exc}")
        return EXIT_USAGE
    if not prompts:
        _emit("[错误] 需要 --prompt 或 --prompt-file 提供提示词")
        return EXIT_USAGE

    images = list(args.image or [])
    missing = [p for p in images if not os.path.isfile(p)]
    if missing:
        _emit(f"[错误] 参考图不存在: {', '.join(missing)}")
        return EXIT_USAGE
    if len(images) > GPT_IMAGE2_MAX_REFERENCE_IMAGES:
        _emit(f"[提示] 参考图 {len(images)} 张，超过上限 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张，只发送前 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张")
        images = images[:GPT_IMAGE2_MAX_REFERENCE_IMAGES]
        args.image = images
    if args.mode == "edit" and not images:
        _emit("[错误] --mode edit 至少需要 1 张 --image")
        return EXIT_USAGE
    if args.n and site != GPT_IMAGE2_SITE_AIGC2D:
        _emit("[提示] autodl 站点一次只出一张，--n 已忽略")
        args.n = None

    cfg = get_api_config(config_path=args.config, api_type=api_type)
    if not str(cfg.get("api_key") or "").strip():
        _emit(f"[错误] apis.{api_type} 缺少 api_key（配置文件: {args.config}）")
        return EXIT_USAGE

    quality = args.quality or str(cfg.get("quality") or "")
    output_format = args.output_format or str(cfg.get("output_format") or "")

    results = []
    failures = 0
    for index, (prompt, size_override) in enumerate(prompts, start=1):
        size = _resolve_size(args, site, size_override)
        label = f"[{index}/{len(prompts)}]"
        if args.dry_run:
            info = _describe_request(args, site, api_type, prompt, size, quality, output_format)
            _emit(f"{label} DRY-RUN 站点={site} api_type={api_type} 端点={info['endpoint']} 尺寸={size} "
                  f"画质={quality or '(配置默认)'} 格式={output_format or '(配置默认)'} 参考图={info['image_count']}")
            _emit("  请求体: " + json.dumps(info.get("payload") or info.get("payload_note"), ensure_ascii=False))
            results.append({"prompt": prompt, "dry_run": True, **{k: v for k, v in info.items() if k != "payload"}})
            continue

        _emit(f"{label} 生成中: 站点={site} 尺寸={size} 参考图={len(images)} prompt={prompt[:60]}")
        try:
            saved = _run_one(args, site, api_type, prompt, size)
        except Exception as exc:  # noqa: BLE001 - CLI 需要把异常转成退出码
            failures += 1
            _emit(f"{label} 失败: {type(exc).__name__}: {exc}")
            results.append({"prompt": prompt, "saved_files": [], "error": str(exc)})
            continue
        if not saved:
            failures += 1
            _emit(f"{label} 失败: 未返回图片（看上面日志或 log/<日期>.log）")
            results.append({"prompt": prompt, "saved_files": [], "error": "no_image"})
            continue
        for path in saved:
            _emit(f"SAVED {path}")
        results.append({"prompt": prompt, "size": size, "saved_files": saved})

    saved_total = sum(len(item.get("saved_files") or []) for item in results)
    _emit(f"完成: 成功 {saved_total} 张 / 任务 {len(prompts)} 条 / 失败 {failures} 条")
    if args.json:
        _emit(json.dumps({"site": site, "api_type": api_type, "results": results}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return EXIT_OK
    return EXIT_FAIL if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
