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
    parser.add_argument("--only", default="", help="只处理文件名包含该子串的图片")
    parser.add_argument("--skip", default="", help="跳过文件名包含该子串的图片")
    parser.add_argument("--timeout", type=int, default=300, help="单步 LLM 请求超时秒数（默认 300）")
    args = parser.parse_args()

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
