# -*- coding: utf-8 -*-
"""把 conf/config-styles.json 的 prompt_gpt 等字段同步到子模块版本化文件（AGENTS 要求的步骤）。"""
import json
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

LOCAL = os.path.join(BASE, "conf", "config-styles.json")
SUB = os.path.join(BASE, "submodules", "image-maker-artstyle", "config-styles.json")

local = json.load(open(LOCAL, encoding="utf-8"))
sub = json.load(open(SUB, encoding="utf-8")) if os.path.isfile(SUB) else {}
shutil.copy(LOCAL, SUB + ".bak") if os.path.isfile(SUB) else None

changed = []
for name, entry in local.items():
    if not isinstance(entry, dict):
        continue
    target = sub.get(name)
    if not isinstance(target, dict):
        sub[name] = entry
        changed.append(name + "(新增)")
        continue
    for field in ("prompt_gpt", "prompt_compressed", "ref_image", "prompt", "repaint_clauses",
                  "enabled", "skip_quality_refine", "skip_identity_refine", "face_hair_refine", "generation_clauses",
                  "identity_correction_clauses", "post_adjustment", "motif_clauses", "motif_enabled"):
        value = entry.get(field)
        if value is not None and target.get(field) != value:
            target[field] = value
            changed.append(f"{name}.{field}")

with open(SUB, "w", encoding="utf-8") as f:
    json.dump(sub, f, ensure_ascii=False, indent=4)
    f.write("\n")

have = [n for n, e in sub.items() if isinstance(e, dict) and str(e.get("prompt_gpt") or "").strip()]
missing = [n for n, e in sub.items() if isinstance(e, dict) and not str(e.get("prompt_gpt") or "").strip()]
print(f"同步字段 {len(changed)} 处")
for c in changed[:12]:
    print("  -", c)
print(f"子模块现有 prompt_gpt: {len(have)}/{len(sub)} | 缺: {missing}")
