# -*- coding: utf-8 -*-
"""SD-WebUI 批量生图统一引擎（无 PyQt，可直接跑）。

替代原先散落在根目录的 sd_gen_*.py 一次性脚本：引擎只此一份，
提示词以主题文件形式放在 prompts/sd-batch/*.txt，按主题复用。

用法示例:
    python tools/sd_batch_gen.py --theme-file prompts/sd-batch/yuri-cg.txt
    python tools/sd_batch_gen.py --theme-file prompts/sd-batch/loli-dynamic.txt --output-subdir lolidynamic
    python tools/sd_batch_gen.py --prompt "1girl, ..." --width 1024 --height 1024 --prefix test

主题文件格式（每行一条提示词）:
    # 注释行
    1824x1024|提示词文本        # 该张单独指定尺寸（长宽比灵活的批次）
    直接写提示词文本              # 使用 --width/--height 的默认尺寸

参数复用 conf/config-sd.json（sd_url / current_sd_group / sd_config_groups /
fixed_prompt / fixed_negative_prompt / last_used_style / webui_extra_payload），
负向模板默认取 data/negative_prompts/common-negative.txt。
输出：data/<YYYYMMDD>/<子目录>/<prefix>_<时间戳ms>_<idx>.png
"""
import argparse
import base64
import json
import os
import sys
import time

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CONFIG_SD_FILE = os.path.join(BASE_DIR, "conf", "config-sd.json")
DEFAULT_NEG_TEMPLATE_FILE = os.path.join(BASE_DIR, "data", "negative_prompts", "common-negative.txt")
DEFAULT_SD_URL = "http://127.0.0.1:7860"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config() -> dict:
    with open(CONFIG_SD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_theme_file(theme_file: str) -> tuple[list[tuple[str, int | None, int | None]], int, int]:
    """解析主题文件，返回 (提示词项列表, 默认宽, 默认高)。

    未单独指定尺寸的项按 (prompt, None, None) 返回，由调用方用默认尺寸补全。
    """
    default_w = 1024
    default_h = 1024
    items: list[tuple[str, int | None, int | None]] = []
    with open(theme_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                size_part, prompt_part = line.split("|", 1)
                prompt_part = prompt_part.strip()
                size_fields = [s.strip().lower() for s in size_part.split("x")]
                if len(size_fields) == 2 and prompt_part:
                    try:
                        w, h = int(size_fields[0]), int(size_fields[1])
                        items.append((prompt_part, w, h))
                        continue
                    except ValueError:
                        pass
                # 不是合法的 尺寸|提示词 语法，按整行提示词处理
                items.append((raw_line.strip(), None, None))
            else:
                items.append((line, None, None))
    return items, default_w, default_h


def build_payload(config: dict, prompt_text: str, negative_prompt_text: str, width: int, height: int,
                  group_name: str | None = None) -> dict:
    """镜像 sd_workflow_core.SdWorkflowThread._build_sd_payload 的参数拼装逻辑。"""
    group_name = group_name or config.get("current_sd_group", "Default")
    sd_settings = config.get("sd_config_groups", {}).get(group_name, {}) or {}
    style_prompt = config.get("last_used_style", "").strip()
    fixed_prompt = config.get("fixed_prompt", "").strip()
    final_prompt = ", ".join([p for p in [fixed_prompt, str(prompt_text or "").strip(), style_prompt] if p])
    fixed_neg_prompt = config.get("fixed_negative_prompt", "").strip()
    final_neg_prompt = ", ".join([p for p in [str(negative_prompt_text or "").strip(), fixed_neg_prompt] if p])

    payload = {
        "prompt": final_prompt,
        "negative_prompt": final_neg_prompt,
        "width": int(width),
        "height": int(height),
        "sampler_name": sd_settings.get("sampler", "Euler a"),
        "scheduler": sd_settings.get("scheduler", "Automatic"),
        "steps": sd_settings.get("steps", 20),
        "cfg_scale": sd_settings.get("cfg_scale", 7.0),
        "override_settings": {},
    }
    sd_model = str(sd_settings.get("sd_model", "") or "").strip()
    sd_vae_list = sd_settings.get("sd_vae", []) or []
    if sd_model:
        payload["override_settings"]["sd_model_checkpoint"] = sd_model
    final_modules = [v.strip() for v in sd_vae_list if v and v.strip() and v.strip().lower() != "automatic"]
    if final_modules:
        payload["override_settings"]["forge_additional_modules"] = final_modules
    else:
        payload["override_settings"].pop("forge_additional_modules", None)
        payload["override_settings"]["sd_vae"] = "Automatic"

    extra_payload_str = str(config.get("webui_extra_payload", "") or "").strip()
    if extra_payload_str:
        try:
            extra_payload = json.loads(extra_payload_str)
            if isinstance(extra_payload, dict):
                for key, value in extra_payload.items():
                    if key == "override_settings" and isinstance(value, dict):
                        payload["override_settings"].update(value)
                    else:
                        payload[key] = value
        except Exception as exc:  # noqa: BLE001
            log(f"警告：WebUI 附加字段 JSON 解析失败，已忽略 ({exc})")
    return payload


def save_images(images_base64: list[str], prefix: str, output_subdir: str,
                start_idx: int = 0) -> list[str]:
    saved_files = []
    date_str = time.strftime("%Y%m%d")
    output_dir = os.path.join(BASE_DIR, "data", date_str, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_b64 in enumerate(images_base64, start=start_idx):
        img_data = base64.b64decode(img_b64)
        timestamp = int(time.time() * 1000)
        filename = os.path.join(output_dir, f"{prefix}_{timestamp}_{idx}.png")
        with open(filename, "wb") as img_file:
            img_file.write(img_data)
        saved_files.append(filename)
    return saved_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SD-WebUI 批量生图统一引擎（读取 conf/config-sd.json，无 PyQt 依赖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--theme-file", default="", help="主题文件（prompts/sd-batch/*.txt，每行一条提示词）")
    parser.add_argument("--prompt", default="", help="单条提示词（与 --theme-file 二选一，优先 --theme-file）")
    parser.add_argument("--width", type=int, default=1024, help="默认宽（无逐张尺寸时用）")
    parser.add_argument("--height", type=int, default=1024, help="默认高（无逐张尺寸时用）")
    parser.add_argument("--output-subdir", default="sdbatch", help="data/<YYYYMMDD>/ 下的输出子目录")
    parser.add_argument("--prefix", default="sdbatch", help="输出文件名前缀")
    parser.add_argument("--sd-group", default="", help="覆盖 current_sd_group（用于指定配置组）")
    parser.add_argument("--negative-file", default=DEFAULT_NEG_TEMPLATE_FILE, help="负向模板文件")
    parser.add_argument("--start-idx", type=int, default=0, help="从该序号开始计数（补生成中断批次时用）")
    parser.add_argument("--timeout", type=int, default=600, help="单次请求超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只解析并打印请求信息，不调用 WebUI")
    args = parser.parse_args()

    if not args.theme_file and not args.prompt:
        log("错误：必须提供 --theme-file 或 --prompt 之一")
        return 1

    if args.theme_file:
        theme_path = args.theme_file if os.path.isabs(args.theme_file) else os.path.join(BASE_DIR, args.theme_file)
        if not os.path.isfile(theme_path):
            log(f"错误：主题文件不存在: {theme_path}")
            return 1
        items, _default_w, _default_h = load_theme_file(theme_path)
        prompts = [(prompt, w, h) for prompt, w, h in items]
        log(f"=== 主题文件: {theme_path} | 共 {len(prompts)} 张 ===")
    else:
        prompts = [(args.prompt, None, None)]

    if os.path.isfile(args.negative_file):
        with open(args.negative_file, "r", encoding="utf-8") as f:
            neg_template_text = f.read().strip()
    else:
        neg_template_text = ""
        log(f"警告：负向模板文件不存在，跳过: {args.negative_file}")

    config = load_config()
    sd_url = str(config.get("sd_url", DEFAULT_SD_URL)).rstrip("/")
    group = args.sd_group or config.get("current_sd_group", "Default")
    model_name = config.get("sd_config_groups", {}).get(group, {}).get("sd_model", "")
    log(f"=== SD-WebUI: {sd_url} | 配置组: {group} | 模型: {model_name}")
    log(f"=== 输出: data/<今天>/<{args.output_subdir}>/<{args.prefix}>_<时间戳>_<idx>.png ===\n")

    all_saved: list[str] = []
    for i, (prompt_text, w, h) in enumerate(prompts, start=1):
        width = w if w is not None else args.width
        height = h if h is not None else args.height
        payload = build_payload(config, prompt_text, neg_template_text, width, height, group_name=group)
        log(f"[{i}/{len(prompts)}] {width}x{height} | {prompt_text[:70]}...")
        if args.dry_run:
            log(f"    [dry-run] prompt={payload['prompt'][:120]}")
            continue
        try:
            resp = requests.post(f"{sd_url}/sdapi/v1/txt2img", json=payload, timeout=args.timeout)
            resp.raise_for_status()
            images_base64 = resp.json().get("images", [])
            if not images_base64:
                log("    [错误] 返回为空")
                continue
            saved = save_images(images_base64, prefix=args.prefix, output_subdir=args.output_subdir,
                                start_idx=args.start_idx + i - 1)
            all_saved.extend(saved)
            for path in saved:
                log(f"    已保存: {path}")
        except Exception as exc:  # noqa: BLE001
            log(f"    [错误] 生成失败: {exc}")

    log(f"\n=== 完成，共保存 {len(all_saved)} 张图片 ===")
    for path in all_saved:
        log(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
