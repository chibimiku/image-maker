# -*- coding: utf-8 -*-
"""材质表面定向过一遍（离线 CLI）：把一张已有的产物按材质重新渲染表面响应。

用途（2026-10-06 用户反馈）：Gemini 直出的图"皮革太平涂、没有参考图那种光泽"。
模型对 `leather` 这种单词只会给出一个中性材质解释，所以需要一次**只改表面材质**的定向重绘：

- 使用专用固件 `prompts/gpt-image-optimize/material-surface-base.md`，锁几何/身份/配色，
  只允许改光泽、
  反射结构、微观纹理与折痕内的局部明暗；
- 可再挂一张**材质参考图**（`--reference`）：只借它的材质处理方式，不搬它的内容
  （专用固件中的参考图条款明确限制用途）。

用法：
  python tools/repaint_material.py --image <产物> --material patent_leather --dry-run
  python tools/repaint_material.py --image <产物> --material patent_leather \
      --reference data/style-ref/satou_kuuki.jpg
  python tools/repaint_material.py --list

边界：一次调用只出一张、`repeat` 默认 1；重绘会重新编码整幅画面，几何保持靠固件约束，
不保证像素级不变。产物落 `data/<日期>/gpt_image_repaint-material/`（受
`IMAGE_MAKER_TEST_OUTPUT` 隔离）。
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.others.api_backend import generate_image_repaint  # noqa: E402
from utils.gpt_image_optimize import (  # noqa: E402
    MATERIAL_SPEC_KEYS, MATERIALS, build_material_prompt, describe_material, material_names,
    resolve_material_info,
)
from utils.output_isolation import resolve_output_target  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按材质定向重绘一张产物（只改表面材质响应）")
    parser.add_argument("--image", help="要处理的图片（必填）")
    parser.add_argument("--material", help="内置材质键：" + ", ".join(material_names()))
    parser.add_argument("--material-spec",
                        help="自定义材质：'material=<英文材质表述>; label=<中文名>; "
                             "clauses=条款1|条款2'（跨任务的材质不必改代码，直接在这里描述）")
    parser.add_argument("--targets", default="",
                        help="处理哪些表面（英文）。缺省按**材质外观**指认，不写死物件名；"
                             "要精确点读某个部位时才传（如 'the panel on the left chest'）")
    parser.add_argument("--reference", action="append", default=[],
                        help="材质参考图（可重复）：只借材质处理方式，不搬内容")
    parser.add_argument("--model", help="覆盖重绘模型（默认取 prompts/gpt-image-optimize/config.json）")
    parser.add_argument("--resolution", help="1K / 2K / 4K，默认取配置")
    parser.add_argument("--repeat", type=int, default=1, help="同一张出几次（抽卡用），默认 1")
    parser.add_argument("--out-dir", help="输出目录；缺省 data/<日期>/gpt_image_repaint-material/")
    parser.add_argument("--prefix", default="material", help="输出文件名前缀，默认 material")
    parser.add_argument("--show-prompt", action="store_true", help="只打印最终提示词后退出")
    parser.add_argument("--dry-run", action="store_true", help="只预检：打印提示词长度、模型、目标目录，不发请求")
    parser.add_argument("--list", action="store_true", help="列出可用材质后退出")
    parser.add_argument("--timeout", type=int, default=300, help="单张请求超时秒数，默认 300")
    return parser.parse_args(argv)


def resolve_out_dir(explicit: str = "") -> str:
    """输出目录（绝对路径，用于打印与预览）。

    `generate_image_repaint(save_sub_dir=...)` 支持相对 `data/<日期>/` 的子目录，
    也支持显式绝对路径；这里返回用于预览的绝对路径。
    """
    directory = str(explicit or "").strip() or os.path.join(
        "data", datetime.datetime.now().strftime("%Y%m%d"), DEFAULT_SUB_DIR)
    directory, _base = resolve_output_target(directory, "material")
    return os.path.abspath(directory)


DEFAULT_SUB_DIR = "gpt_image_repaint-material"


def run(args) -> int:
    if args.list:
        for key in material_names():
            info = MATERIALS[key]
            print(f"{key:<16} {info['label']}")
            print(f"{'':<16} {info['material']}")
            print(f"{'':<16} 条款 {len(info['clauses'])} 条；目标默认按材质外观指认（不认物件）")
        print("\n自定义材质：" + "--material-spec 'material=<英文材质表述>; label=<中文名>; clauses=条款1|条款2'")
        print("可用键：" + ", ".join(MATERIAL_SPEC_KEYS))
        return 0
    if not args.image:
        print("❌ 必须给 --image", file=sys.stderr)
        return 2
    image = os.path.abspath(args.image)
    if not os.path.isfile(image):
        print(f"❌ 找不到图片：{image}", file=sys.stderr)
        return 2
    if bool(args.material) == bool(args.material_spec):
        print("❌ --material 与 --material-spec 必须给且只能给一个；内置可用："
              + ", ".join(material_names()), file=sys.stderr)
        return 2
    try:
        info = resolve_material_info(args.material, args.material_spec)
        prompt = build_material_prompt(args.material, targets=args.targets, spec=args.material_spec)
    except ValueError as exc:
        print("❌ " + str(exc), file=sys.stderr)
        return 2
    references = []
    for path in args.reference:
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            print(f"❌ 找不到参考图：{resolved}", file=sys.stderr)
            return 2
        references.append(resolved)

    out_dir = resolve_out_dir(args.out_dir)
    # 传给后端的是相对子目录；--out-dir 显式给了就用绝对路径（resolve_save_dir 会原样使用）
    sub_dir = str(args.out_dir or "").strip() or DEFAULT_SUB_DIR
    print(f"材质：{info['label']}（{info['key']}）→ {info['material']}")
    print(f"条款：{len(info['clauses'])} 条")
    print(f"目标：{args.targets or '按材质外观指认（默认，不认物件）'}")
    print(f"提示词：{len(prompt)} 字符（材质固件 + 该材质的渲染条款）")
    print(f"输出目录：{out_dir}")
    print(f"参考图：{references or '无（只靠文字描述材质）'}")
    if args.show_prompt:
        print("\n" + prompt)
        return 0
    if args.dry_run:
        print("（--dry-run：未发请求）")
        return 0

    files = generate_image_repaint(
        [image], prompt=prompt, model=args.model, resolution=args.resolution,
        aspect_ratio="auto", repeat=max(1, int(args.repeat or 1)),
        save_sub_dir=sub_dir, file_prefix=str(args.prefix or "material"),
        extra_reference_paths=references or None, log_callback=lambda message: print("  " + str(message)),
        request_timeout=args.timeout,
    )
    files = [str(path) for path in (files or []) if path]
    if not files:
        print("❌ 没有产出（详见 log/<日期>.log）", file=sys.stderr)
        return 1
    for path in files:
        print("✅ " + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
