# -*- coding: utf-8 -*-
"""汇总第二轮衣装实验（tooling 侧的报告/图库生成，属于衣装实验族的一部分）。

用法：
    python tools/wardrobe_experiment.py --score ...   # 先产出 scores.json / pairs-blind.json
    python tools/wardrobe_report.py [--run-dir <目录>]

产物：summary.json、gallery-wardrobe-round2.html（单文件离线）、
      scoreboard.json（按 v3 通过/失败/未证实标准判定）。
"""
import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image  # noqa: E402

DEFAULT_RUN_DIR = os.path.join("data", "test-result", "20261005", "wardrobe-round2-r1")


def load(path, default=None):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return default if default is not None else {}


def thumb(path, max_side=620):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def verdict_from(record: dict) -> dict:
    """按任务书的通过标准判定一组评价：三项分别 pass/fail/uncertain。"""
    parsed = record.get("parsed") or {}
    roles = {}
    for item in parsed.get("images") or []:
        label = str(item.get("label") or "").upper()
        roles[label] = item
    assignment = record.get("assignment") or {}
    per_variant = {}
    for label, variant in assignment.items():
        item = roles.get(label)
        if not item:
            continue
        per_variant[variant] = {
            "clothing_action": item.get("clothing_action"),
            "framing": item.get("framing"),
            "protected_anchors": item.get("protected_anchors"),
            "full_outfit_verifiable": item.get("full_outfit_verifiable"),
            "observed_clothing": item.get("observed_clothing"),
            "observed_lower_frame_edge": item.get("observed_lower_frame_edge"),
            "evidence": item.get("evidence"),
        }
    return {"per_variant": per_variant, "preferred": parsed.get("preferred"),
            "reason": parsed.get("reason"), "status": record.get("status"),
            "error": record.get("error", ""), "model": record.get("model")}


def summarise(run_dir: str) -> dict:
    samples = load(os.path.join(run_dir, "samples.json"), {"samples": []}).get("samples") or []
    pairs = load(os.path.join(run_dir, "pairs-blind.json"), {"pairs": []}).get("pairs") or []
    scores = load(os.path.join(run_dir, "scores.json"), {"records": []}).get("records") or []
    ledger = load(os.path.join(run_dir, "ledger.json"), {})
    prepare = load(os.path.join(run_dir, "prepare-report.json"), {})
    resolver = load(os.path.join(run_dir, "resolver-result.json"), {})
    review = load(os.path.join(run_dir, "review-result.json"), {})
    score_by_pair = {item["pair_id"]: item for item in scores}

    per_channel = {}
    per_case = {}
    for sample in samples:
        per_channel.setdefault(sample["channel"], {"attempted": 0, "success": 0, "failed": 0})
        per_channel[sample["channel"]]["attempted"] += 1
        key = "success" if sample.get("status") == "success" else "failed"
        per_channel[sample["channel"]][key] += 1
        entry = per_case.setdefault(f"{sample['case_id']}-{sample['variant']}",
                                    {"attempted": 0, "success": 0, "files": []})
        entry["attempted"] += 1
        if sample.get("status") == "success":
            entry["success"] += 1
            entry["files"].extend(sample.get("saved_files") or [])

    requirements = {"C03": {"clothing_action": "pass", "framing": "pass", "protected_anchors": "pass"},
                    "C04": {"clothing_action": "pass", "framing": "pass", "protected_anchors": "pass"},
                    "C06": {"clothing_action": "pass", "framing": "pass", "protected_anchors": "pass"},
                    "C01": {"clothing_action": "pass", "protected_anchors": "pass"},
                    "C02": {"clothing_action": "pass", "protected_anchors": "pass"}}
    v3_verdicts = []
    for pair in pairs:
        record = score_by_pair.get(pair["pair_id"]) or {}
        verdict = verdict_from(record) if record.get("parsed") else {
            "per_variant": {}, "status": record.get("status", "no_record"),
            "error": record.get("error", ""), "model": record.get("model", "")}
        v3 = (verdict.get("per_variant") or {}).get("v3_routed") or {}
        v2 = (verdict.get("per_variant") or {}).get("v2_frozen") or {}
        needs = requirements.get(pair["case_id"], {})
        checks = {field: v3.get(field) for field in needs}
        missing = [field for field, value in checks.items() if value is None]
        failing = [field for field, value in checks.items() if value == "fail"]
        uncertain = [field for field, value in checks.items() if value == "uncertain"]
        if missing:
            outcome = "unproven"
        elif failing:
            outcome = "failed"
        elif uncertain:
            outcome = "unproven"
        else:
            outcome = "pass"
        v3_verdicts.append({"pair_id": pair["pair_id"], "case_id": pair["case_id"],
                            "channel": pair["channel"], "repeat": pair["repeat"],
                            "outcome": outcome, "v3": checks, "v2": v2,
                            "preferred": verdict.get("preferred"), "status": verdict.get("status"),
                            "error": verdict.get("error", ""),
                            "assignment": pair.get("assignment")})
    hard = [item for item in v3_verdicts if item["case_id"] in ("C03", "C04", "C06")]
    regression = [item for item in v3_verdicts if item["case_id"] in ("C01", "C02")]
    scoreboard = {
        "v3_all_hard_pass": all(item["outcome"] == "pass" for item in hard) and len(hard) == 12,
        "v3_hard": hard, "v3_regression": regression,
        "counts": {name: sum(1 for item in v3_verdicts if item["outcome"] == name)
                   for name in ("pass", "failed", "unproven")},
    }
    return {
        "run_dir": os.path.abspath(run_dir), "written_at": datetime.now().isoformat(),
        "models": {"generation": load(os.path.join(run_dir, "state.json"), {}).get("generation_params"),
                   "vision": load(os.path.join(run_dir, "state.json"), {}).get("vision_model")},
        "budget": {"budgets": ledger.get("budgets"), "attempts": ledger.get("attempts"),
                   "recovered": ledger.get("recovered")},
        "prepare_ok": prepare.get("ok"),
        "resolver": {"valid": (resolver.get("validation") or {}).get("valid"),
                     "matched": (resolver.get("validation") or {}).get("matched"),
                     "total": (resolver.get("validation") or {}).get("total")},
        "review": {"ready": review.get("ready"), "parsed_ready": (review.get("parsed") or {}).get("ready"),
                   "blocking": (review.get("parsed") or {}).get("blocking")},
        "calibration": load(os.path.join(run_dir, "state.json"), {}).get("stages", {}).get("calibrate"),
        "generation": {"per_channel": per_channel, "per_case_variant": per_case,
                       "total": len(samples),
                       "success": sum(1 for item in samples if item.get("status") == "success")},
        "pairs": {"total": len(pairs), "ready": sum(1 for item in pairs if item.get("status") == "ready"),
                  "a_v2": sum(1 for item in pairs if (item.get("assignment") or {}).get("A") == "v2_frozen")},
        "scoring": {"records": len(scores),
                    "scored": sum(1 for item in scores if item.get("status") == "scored"),
                    "unscored": sum(1 for item in scores if item.get("status") != "scored")},
        "scoreboard": scoreboard,
    }


def build_gallery(run_dir: str, summary: dict) -> str:
    samples = load(os.path.join(run_dir, "samples.json"), {"samples": []}).get("samples") or []
    pairs = load(os.path.join(run_dir, "pairs-blind.json"), {"pairs": []}).get("pairs") or []
    scores = {item["pair_id"]: item for item in
              load(os.path.join(run_dir, "scores.json"), {"records": []}).get("records") or []}
    by_id = {item["sample_id"]: item for item in samples}
    cards = []
    for pair in pairs:
        record = scores.get(pair["pair_id"]) or {}
        verdict = verdict_from(record) if record.get("parsed") else {}
        figures = []
        for label in ("A", "B"):
            variant = (pair.get("assignment") or {}).get(label)
            image = (pair.get("images") or {}).get(label)
            if image and os.path.isfile(image):
                figures.append(f'<figure><img src="{thumb(image)}" alt="{pair["pair_id"]}-{label}">'
                               f'<figcaption><b>{label}</b> · {variant}<br>'
                               f'<span class=path>{os.path.basename(image)}</span></figcaption></figure>')
            else:
                figures.append(f'<figure class=missing><div class=ph>无产物</div>'
                               f'<figcaption><b>{label}</b> · {variant}</figcaption></figure>')
        raw = record.get("raw") or record.get("error") or "（无评分记录）"
        cards.append(f"""
<section class=card>
  <h3>{pair['pair_id']} <small>{pair.get('role')} · gold={pair.get('gold_action')} ·
  framing={pair.get('framing')}</small></h3>
  <p class=meta>匿名映射：{'，'.join(f"{k}={v}" for k, v in (pair.get('assignment') or {}).items())}｜
  状态：{record.get('status', 'not_scored')}</p>
  <div class=row>{''.join(figures)}</div>
  <details><summary>原始评价响应</summary><pre>{raw}</pre></details>
</section>""")
    head = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>衣装第二轮 v2/v3 对照（{summary['generation']['success']}/{summary['generation']['total']}）</title>
<style>
body{{font-family:system-ui,'Segoe UI',sans-serif;margin:22px;background:#faf8f6;color:#222}}
h1{{font-size:20px}} h3{{margin:0 0 6px;font-size:15px}} small{{font-weight:400;color:#666}}
.card{{background:#fff;border:1px solid #e6e0da;border-radius:10px;padding:12px;margin:14px 0}}
.row{{display:flex;gap:12px;flex-wrap:wrap}} figure{{margin:0;width:300px}}
figure img{{width:300px;border-radius:8px;border:1px solid #ddd;display:block}}
figcaption{{font-size:12px;color:#444;margin-top:6px}} .path{{color:#888;font-size:11px}}
.meta{{font-size:12px;color:#555}} pre{{white-space:pre-wrap;font-size:12px;background:#f6f4f2;
padding:10px;border-radius:6px;max-height:380px;overflow:auto}} .missing .ph{{width:300px;height:400px;
display:flex;align-items:center;justify-content:center;background:#f1eeeb;color:#999;border-radius:8px}}
</style></head><body>
<h1>独立衣装第二轮：v2（首轮冻结控制）vs v3（路由修订候选）</h1>
<p class=meta>生图成功 {summary['generation']['success']}/{summary['generation']['total']}，
配对 {summary['pairs']['ready']}/{summary['pairs']['total']}，
有效评分 {summary['scoring']['scored']}/{summary['scoring']['records']}；
A=v2 的配对数 {summary['pairs']['a_v2']}（位置随机平衡）。</p>
<p class=meta>解析 {summary['resolver']['matched']}/{summary['resolver']['total']}，
就绪审查 ready={summary['review']['parsed_ready']}，
校准 {json.dumps(summary['calibration'], ensure_ascii=False)}</p>
"""
    html = head + "".join(cards) + "\n</body></html>"
    path = os.path.join(run_dir, "gallery-wardrobe-round2.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    summary = summarise(run_dir)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    gallery = build_gallery(run_dir, summary)
    print(json.dumps({key: summary[key] for key in ("generation", "pairs", "scoring", "resolver")},
                     ensure_ascii=False, indent=2))
    print("scoreboard v3_all_hard_pass:", summary["scoreboard"]["v3_all_hard_pass"])
    print("gallery:", gallery, os.path.getsize(gallery), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
