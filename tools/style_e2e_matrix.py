# -*- coding: utf-8 -*-
"""按 JSON 规格批量复跑「分析产物 → GPT 首图 → Gemini 重绘 → 身份审计/修订」。"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(text)).strip("-") or "case"


def _run_case(case: dict, root: str, common: dict) -> dict:
    style = str(case["style"])
    round_no = int(case.get("round") or 1)
    case_id = str(case.get("id") or f"{style}-r{round_no}")
    out_dir = os.path.join(root, _slug(case_id))
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(BASE, "tools", "analysis_gpt_run.py"),
           "--json", os.path.abspath(case["json"]), "--style", style,
           "--quality", str(common.get("quality", "high")), "--size", str(common.get("size", "auto")),
           "--first-pass-mode", str(common.get("first_pass_mode", "generate")),
           "--content-field", str(common.get("content_field", "gpt_image_prompt")),
           "--steps", str(common.get("steps", "repaint")),
           "--repaint-ref", str(common.get("repaint_ref", "none")),
           "--repaint-scope", str(common.get("repaint_scope", "full")),
           "--firmware", str(common.get("firmware", "prompts/gpt-image-optimize/repaint-system-conservative-v5.md")),
           "--output-dir", out_dir, "--identity-audit"]
    if case.get("base_image"):
        cmd.extend(["--base-image", os.path.abspath(str(case["base_image"]))])
    if common.get("identity_correct", True):
        cmd.append("--identity-correct")
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(cmd, cwd=BASE, text=True, encoding="utf-8", errors="replace",
                          capture_output=True)
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n" + proc.stderr)
    return {"id": case_id, "style": style, "round": round_no,
            "json": os.path.abspath(case["json"]), "output_dir": out_dir,
            "command": cmd, "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "returncode": proc.returncode}


def main() -> int:
    ap = argparse.ArgumentParser(description="5 画风等端到端矩阵复现实验")
    ap.add_argument("--spec", required=True, help="含 cases/common 的 JSON 规格")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=1, help="并行任务数；默认 1")
    args = ap.parse_args()
    spec = json.load(open(args.spec, encoding="utf-8"))
    root = os.path.abspath(args.output_dir)
    os.makedirs(root, exist_ok=True)
    cases = list(spec.get("cases") or [])
    common = dict(spec.get("common") or {})
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_run_case, case, root, common): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(cases)}] {result['id']}: rc={result['returncode']}", flush=True)
            with open(os.path.join(root, "matrix-progress.json"), "w", encoding="utf-8") as f:
                json.dump({"spec": spec, "results": results}, f, ensure_ascii=False, indent=2)
    order = {str(c.get("id") or f"{c['style']}-r{int(c.get('round') or 1)}"): i
             for i, c in enumerate(cases)}
    results.sort(key=lambda x: order.get(x["id"], 9999))
    with open(os.path.join(root, "matrix.json"), "w", encoding="utf-8") as f:
        json.dump({"spec": spec, "results": results}, f, ensure_ascii=False, indent=2)
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
