#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多图对比 → 文字/视觉模型 0-100 画风相似度打分 CLI。

用法：
  # 单个画风（自动查参考图）
  python tools/llm_style_score.py --style komori-hikki-style --images a.jpg b.jpg

  # 多参考泛化对照（画风提取的 10+ 张来源）
  python tools/llm_style_score.py --references r1.jpg r2.jpg r3.jpg --images c1.jpg c2.jpg

  # 直接从实验轮次取候选
  python tools/llm_style_score.py --label round19 --style cccccanh

  # 先看会发什么、不发请求
  python tools/llm_style_score.py --style cccccanh --images a.jpg --dry-run

  # 顺带写进出图对比报告
  python tools/llm_style_score.py --style cccccanh --images a.jpg --report report.html
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

from utils import llm_style_score as llm  # noqa: E402
from utils.styles import normalize_style_entry  # noqa: E402

LAB = os.path.join(ROOT, "data", "test-result", "style-lab")


def style_refs(style):
    with io.open(os.path.join(ROOT, "conf", "config-styles.json"), encoding="utf-8") as handle:
        styles = json.load(handle)
    for name in (style, style.rsplit("-v", 1)[0] if "-v" in style else style):
        if name in styles:
            entry = normalize_style_entry(styles[name])
            if entry["ref_image"] and os.path.isfile(entry["ref_image"]):
                return [entry["ref_image"]]
    return []


def from_label(label, style):
    path = os.path.join(LAB, f"{label}.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path}")
    out = []
    for line in io.open(path, encoding="utf-8"):
        if not line.strip().startswith("{"):
            continue
        record = json.loads(line)
        if record.get("style") != style or not record.get("image"):
            continue
        full = record["image"] if os.path.isabs(record["image"]) else os.path.join(ROOT, record["image"])
        if os.path.isfile(full):
            out.append({"id": f"{record.get('tag') or record.get('condition')} · {os.path.basename(full)}",
                        "path": full})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style")
    ap.add_argument("--label")
    ap.add_argument("--references", nargs="*", default=[])
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=llm.BATCH_SIZE)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", help="把打分结果写成 JSON")
    ap.add_argument("--report", help="把打分写进已有的出图对比报告 HTML")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    refs = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in args.references]
    if args.style and not refs:
        refs = style_refs(args.style)
    candidates = []
    if args.label and args.style:
        candidates = from_label(args.label, args.style)
    for path in args.images:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        candidates.append({"id": os.path.basename(full), "path": full})

    if not refs or not candidates:
        raise SystemExit("需要参考图（--style 或 --references）与候选图（--images 或 --label）")

    print(f"参考图 {len(refs)} 张，候选 {len(candidates)} 个，批次 {args.batch_size}")
    if args.dry_run:
        for batch in llm.build_inputs(refs, candidates, batch_size=args.batch_size):
            print("-" * 60)
            print(llm.build_user_prompt(batch))
        print("-" * 60)
        print("SYSTEM:\n" + llm.SYSTEM_PROMPT[:600] + " ...")
        return

    result = llm.score_candidates(refs, candidates, timeout=args.timeout,
                                  log=lambda message: print("  " + message),
                                  batch_size=args.batch_size)
    print("\n参考集共性：" + (result.get("overall_note") or "-"))
    print(f"{'候选':<56}{'总分':>6}{'线条':>6}{'五官':>6}{'上色':>6}{'材质':>6}{'背景':>6}  判定")
    for item in result["assessments"]:
        print(f"{item['id'][:54]:<56}{item['style_score']:>6.0f}{item['linework']:>6.0f}"
              f"{item['face_hair']:>6.0f}{item['shading']:>6.0f}{item['texture']:>6.0f}"
              f"{item['background']:>6.0f}  {llm.score_to_verdict(item['style_score'])}")
    for item in result["assessments"]:
        print(f"  · {item['id'][:60]}: {item['reason']}｜最大差距：{item['biggest_gap']}")

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print("JSON:", args.out)

    if args.report:
        sys.path.insert(0, os.path.join(ROOT, "data", "test-result", "style-lab"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_report_patch", os.path.join(LAB, "llm_report_patch.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.patch_report(args.report, result)
        print("已写入报告:", args.report)


if __name__ == "__main__":
    main()
