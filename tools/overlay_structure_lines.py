# -*- coding: utf-8 -*-
"""结构线叠加（CLI 薄壳，实现在 utils/post_process.py）。

用法：
  python tools/overlay_structure_lines.py --base <图> [--edge-source <图>] --strength 0.5 \
      [--min-len 120] [--darken 0.45] --out <输出.png>
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.post_process import (DEFAULT_DARKEN, DEFAULT_MIN_LEN, DEFAULT_STRUCTURE_STRENGTH,  # noqa: E402
                                structure_overlay_file)


def main():
    ap = argparse.ArgumentParser(description="结构线叠加（提升线稿连通性的后处理）")
    ap.add_argument("--base", required=True, help="底图（通常是重绘产物）")
    ap.add_argument("--edge-source", default="", help="抽线的图，默认同 --base")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRUCTURE_STRENGTH, help="线条不透明度 0~1")
    ap.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN, help="最短保留连通域（像素）")
    ap.add_argument("--darken", type=float, default=DEFAULT_DARKEN, help="线条相对局部色调的加深比例")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    info = structure_overlay_file(args.base, args.out, strength=args.strength,
                                 edge_source_path=args.edge_source or None,
                                 min_len=args.min_len, darken=args.darken)
    print(f"结构线覆盖率 {info['coverage']:.4f} -> {info['out']}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
