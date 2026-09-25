# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def main():
    ap = argparse.ArgumentParser(description="审计重绘角色一致性，并可用 Gemini 做最多两轮定点修订")
    ap.add_argument("--json", required=True, help="初始图片分析 JSON")
    ap.add_argument("--image", required=True, help="要审计的 GPT/Gemini 产物")
    ap.add_argument("--out", required=True, help="审计 JSON 输出")
    ap.add_argument("--correct", action="store_true", help="发现高置信差异时调用 Gemini 修订")
    ap.add_argument("--correct-dir", default="", help="修订图片目录")
    ap.add_argument("--prompt-file", default="", help="实际 GPT 首图 prompt；提供时优先作为身份锚来源")
    args = ap.parse_args()
    from utils.identity_audit import (audit_image_identity, build_identity_correction_prompt,
                                      should_correct)
    analysis = json.load(open(args.json, encoding="utf-8"))
    expected_prompt = ""
    if args.prompt_file:
        expected_prompt = open(args.prompt_file, encoding="utf-8").read()
    audit = audit_image_identity(args.image, analysis, expected_prompt=expected_prompt)
    audit["correction_prompt"] = build_identity_correction_prompt(audit, 1, 2)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.correct and should_correct(audit):
        from modules.others.api_backend import generate_image_repaint
        current = args.image
        rounds = []
        for round_no in range(1, 3):
            if not should_correct(audit):
                break
            prompt = build_identity_correction_prompt(audit, round_no, 2)
            saved = generate_image_repaint(
                [current], resolution="2K", aspect_ratio="auto", prompt=prompt, use_detail_suffix=False,
                save_sub_dir=os.path.abspath(args.correct_dir or os.path.dirname(args.out)),
                file_prefix=f"identity-correct-{round_no}") or []
            if not saved:
                break
            current = saved[-1]
            audit = audit_image_identity(current, analysis, expected_prompt=expected_prompt)
            audit["correction_prompt"] = build_identity_correction_prompt(audit, min(2, round_no + 1), 2)
            rounds.append({"round": round_no, "images": saved, "audit": audit})
        audit["correction_rounds"] = rounds
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: audit[k] for k in ("mismatch", "severity", "confidence", "differences")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
