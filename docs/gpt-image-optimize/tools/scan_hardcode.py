# -*- coding: utf-8 -*-
"""扫描 gpt-image-optimize 相关代码里的「写死配置」：模型名 / 尺寸 / 比例 / 站点 / 目录 / 长 prompt。"""
import re

FILES = [
    r"D:\code\image-maker\utils\gpt_image_optimize.py",
    r"D:\code\image-maker\modules\image_generation\gpt_image2_tab.py",
    r"D:\code\image-maker\modules\others\api_backend.py",
    r"D:\code\image-maker\tools\gpt_image2_gen.py",
]

PATTERNS = {
    "模型名": r"gemini-3(\.\d)?-[a-z0-9.-]*image[a-z0-9.-]*",
    "分辨率常量": r"[\"'](1K|2K|4K)[\"']",
    "比例常量": r"[\"'](\d{1,2}:\d{1,2})[\"']",
    "站点/接口": r"[\"'](new\.aigc2d|aigc2d|aigc-2d-gpt|autodl)[\"']",
    "输出目录": r"[\"'](gpt_image_repaint|gpt-image-2-autodl)[\"']",
    "长 prompt 痕迹": r"detailed face|scallop|NO SIMPLIFICATION|reprint this image|Repaint this",
}
for path in FILES:
    text = open(path, encoding="utf-8").read()
    lines = text.splitlines()
    print("\n=====", path.split("\\")[-1], "=====")
    for label, pat in PATTERNS.items():
        hits = []
        for i, line in enumerate(lines, 1):
            if re.search(pat, line, re.I):
                hits.append((i, line.strip()[:150]))
        if hits:
            print(f"  [{label}] {len(hits)} 处")
            for i, line in hits[:12]:
                print(f"     L{i}: {line}")
