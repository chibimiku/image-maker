from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from .models import CollectionBundle


MAKE_PIC_SLOT_MAP = {
    "dress": "衣服1",
    "shoes": "鞋子",
    "socks": "袜子",
}


def build_make_pic_state(bundle: CollectionBundle, instructions: str = "", extra_prompt: str = "", aspect_ratio: str = "3:4") -> dict[str, Any]:
    slots = {}
    for asset in bundle.assets:
        slot_name = MAKE_PIC_SLOT_MAP.get(asset.part)
        if not slot_name:
            continue
        slots[slot_name] = {
            "filepath": asset.local_path,
            "prompt": asset.prompt_hint,
        }

    return {
        "instruction": instructions.strip(),
        "extra_prompt": extra_prompt.strip(),
        "aspect_ratio": aspect_ratio.strip() or "3:4",
        "slots": slots,
    }


def export_bundle_manifest(bundle: CollectionBundle, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "collection_bundle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def export_make_pic_state(bundle: CollectionBundle, state_path: str, instructions: str = "", extra_prompt: str = "", aspect_ratio: str = "3:4") -> str:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = build_make_pic_state(
        bundle,
        instructions=instructions,
        extra_prompt=extra_prompt,
        aspect_ratio=aspect_ratio,
    )
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return state_path


def launch_make_pic(project_root: str) -> subprocess.Popen:
    script_path = os.path.join(project_root, "make-pic.py")
    return subprocess.Popen([sys.executable, script_path], cwd=project_root)
