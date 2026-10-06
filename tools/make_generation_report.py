#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成「出图对比报告」HTML（参考图对照 + 泛化对照 + 推荐 prompt 组合）。

三种输入方式：
1) 实验轮次：`--label round21`（读 data/test-result/style-lab/round21.jsonl，按画风取参考图）
2) 显式指定：`--reference <画风名或图路径> --images <图...>`
3) 完整泛化对照：`--references <图...> --images <图...>`（参考图集合可多张，用于画风提取的 10+ 来源）

用法示例：
  python tools/make_generation_report.py --label round21
  python tools/make_generation_report.py --references a.jpg b.jpg c.jpg --images out1.jpg out2.jpg --out report.html
"""
import argparse
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.generation_report import build_payload, write_report  # noqa: E402
from utils.styles import normalize_style_entry  # noqa: E402

LAB = os.path.join(ROOT, "data", "test-result", "style-lab")


def style_refs(style):
    with io.open(os.path.join(ROOT, "conf", "config-styles.json"), encoding="utf-8") as f:
        styles = json.load(f)
    for name in (style, style.rsplit("-v", 1)[0] if "-v" in style else style):
        if name in styles:
            entry = normalize_style_entry(styles[name])
            if entry["ref_image"] and os.path.isfile(entry["ref_image"]):
                return [entry["ref_image"]]
    return []


def from_label(label):
    path = os.path.join(LAB, f"{label}.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path}")
    records = [json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip().startswith("{")]
    groups = {}
    for record in records:
        image = record.get("image")
        if not image:
            continue
        full = image if os.path.isabs(image) else os.path.join(ROOT, image)
        if not os.path.isfile(full):
            continue
        groups.setdefault(record.get("style"), {"refs": [], "candidates": []})
        refs = style_refs(record.get("style") or "")
        for ref in refs:
            if ref not in groups[record["style"]]["refs"]:
                groups[record["style"]]["refs"].append(ref)
        groups[record["style"]]["candidates"].append(
            {"id": f"{record.get('tag') or record.get('condition')} · {os.path.basename(full)}", "path": full})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="实验轮次标签（与 round*.jsonl 对应）")
    ap.add_argument("--style", help="只处理某个画风")
    ap.add_argument("--reference", help="画风名（查 config-styles.json）或单张参考图路径")
    ap.add_argument("--references", nargs="*", default=[], help="参考图集合（泛化对照用，可多张）")
    ap.add_argument("--images", nargs="*", default=[], help="候选图")
    ap.add_argument("--out", help="输出 HTML 路径")
    ap.add_argument("--device", default="auto-cpu", help="auto-cpu / cpu / cuda")
    args = ap.parse_args()

    os.environ.setdefault("IMAGE_MAKER_REPORT_DEVICE", args.device)

    if args.label:
        groups = from_label(args.label)
        if args.style:
            groups = {args.style: groups.get(args.style, {"refs": [], "candidates": []})}
        outdir = os.path.join(LAB, "reports")
        written = []
        for style, group in groups.items():
            if not group["refs"] or not group["candidates"]:
                print(f"  跳过 {style}（参考图或候选缺失）")
                continue
            payload = build_payload(group["refs"], group["candidates"],
                                    meta={"画风": style, "轮次": args.label,
                                          "参考图数": len(group["refs"]),
                                          "候选数": len(group["candidates"]),
                                          "指标口径": "style-similarity/2（整图八组）"})
            out = args.out or os.path.join(outdir, f"report-{args.label}-{style}.html")
            write_report(payload, out)
            written.append(out)
            print("written:", os.path.relpath(out, ROOT))
        if not written:
            raise SystemExit("没有任何可生成的报告")
        return

    if not args.images:
        raise SystemExit("需要 --images 或 --label")
    refs = list(args.references)
    if args.reference:
        if os.path.isfile(args.reference):
            refs.append(args.reference)
        else:
            refs.extend(style_refs(args.reference))
    if not refs:
        raise SystemExit("没有可用的参考图")
    candidates = [{"id": os.path.basename(p), "path": p if os.path.isabs(p) else os.path.join(ROOT, p)}
                  for p in args.images]
    refs = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in refs]
    payload = build_payload(refs, candidates,
                            meta={"参考图数": len(refs), "候选数": len(candidates),
                                  "指标口径": "style-similarity/2（整图八组）"})
    out = args.out or os.path.join(LAB, "reports", "report-manual.html")
    write_report(payload, out)
    print("written:", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
