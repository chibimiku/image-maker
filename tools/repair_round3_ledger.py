"""一次性账本修复：放行记录归位 + 恢复被压回去的授权上限。

背景（2026-10-05 实测）：
1. `record_override` 曾把放行记录追加进 `budget_grants`，产生一条
   `budget=None / new_limit=None` 的条目；它属于 `gate_overrides`。
2. `Experiment` 启动时按 spec 上限「收紧」，把用户授权的 vision 31 压回 25，
   导致评分在第 10 组就断（少跑 6 次）。

只做这两件事：不改 attempts、不改 spent、不删任何记录、不换 run-id。
用法：python tools/repair_round3_ledger.py --run-dir <目录> [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.wardrobe_experiment import normalize_budgets  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="repair_round3_ledger.py")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--apply", action="store_true", help="真正写回（默认只打印计划）")
    args = parser.parse_args(argv)
    path = os.path.join(os.path.abspath(args.run_dir), "ledger.json")
    ledger = json.load(open(path, encoding="utf-8"))
    broke = [item for item in ledger.get("budget_grants") or [] if not item.get("budget")]
    kept = [item for item in ledger.get("budget_grants") or [] if item.get("budget")]
    overrides = list(ledger.get("gate_overrides") or [])
    # 补记被覆盖掉的那次追加（原记录已被同名错误写入覆盖，只能按账本自述重建）
    recorded = {record["budget"] for record in kept}
    rebuilt = []
    if "vision_http_attempts" not in recorded:
        rebuilt.append({
            "budget": "vision_http_attempts", "kind": "vision", "previous_limit": 25,
            "new_limit": 31, "delta": 6, "spent_at_grant": 15,
            "authorized_by": "user (conversation)",
            "reason": "校准实际用掉 15 槽（单图调用+单项重试），16 组双图评分还差 6 槽；用户已选同一方案",
            "source": "2026-10-05 用户答复：追加视觉额度，先修校准再跑完",
            "granted_at": "2026-10-05T18:28:50", "rebuilt_from": "账本 budget_grants 自述重建",
        })
    floor = normalize_budgets(None)
    for record in kept + rebuilt:
        floor[record["budget"]] = max(floor[record["budget"]], int(record["new_limit"]))
    before = dict(ledger["budgets"])
    plan = {
        "ledger": path,
        "budget_grants_before": len(ledger.get("budget_grants") or []),
        "moved_to_gate_overrides": len(broke),
        "rebuilt_grants": len(rebuilt),
        "budgets_before": before,
        "budgets_after": {key: max(int(before.get(key, 0)), int(value))
                          for key, value in floor.items()},
        "stop_reason_before": ledger.get("stop_reason", ""),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.apply:
        print("[dry-run] 未写盘")
        return 0
    backup = path + f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    ledger["budget_grants"] = kept + rebuilt
    ledger["gate_overrides"] = overrides + broke
    ledger["budgets"] = plan["budgets_after"]
    ledger["ledger_repairs"] = list(ledger.get("ledger_repairs") or []) + [{
        "at": datetime.now().isoformat(),
        "moved_override_records": len(broke), "rebuilt_grants": len(rebuilt),
        "budgets_before": before, "budgets_after": plan["budgets_after"],
        "note": ("放行记录从 budget_grants 归位到 gate_overrides；重建被覆盖掉的追加记录；"
                 "恢复账本里授权过的最高上限。未改 attempts、未改 spent、未删任何原记录。"),
        "backup": os.path.basename(backup)}]
    ledger.pop("stop_reason", None)
    ledger["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, ensure_ascii=False, indent=2)
    print(f"[applied] 备份={os.path.basename(backup)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
