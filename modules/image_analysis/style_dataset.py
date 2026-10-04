"""外部识图 agent 的画风筛图清单契约；不进行联网或视觉评分。"""
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from PIL import Image

SCHEMA_VERSION = "image-maker.style-dataset.v1"
SCORE_KEYS = ("completeness", "style_consistency", "detail_readability", "artifact_cleanliness")
WEIGHTS = (0.30, 0.30, 0.25, 0.15)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("图片路径须为以 / 分隔的非空相对路径")
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or ":" in relative or str(value) != relative:
        raise ValueError(f"图片路径不合法: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"图片路径超出源目录: {relative}")
    return resolved


def inventory(root, recursive=False):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_root 必须是图片目录")
    paths = root.rglob("*") if recursive else root.iterdir()
    records = []
    for path in sorted(paths, key=lambda p: p.as_posix().casefold()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        actual = _inside(root, relative)
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        width, height, readable = 0, 0, False
        try:
            with Image.open(actual) as image:
                width, height = image.size
                image.load()
            readable = True
        except (OSError, ValueError, Image.DecompressionBombError):
            pass
        records.append({"path": relative, "sha256": digest, "width": width if readable else 0,
                        "height": height if readable else 0, "readable": readable,
                        "inspected": False, "decision": "needs_review", "reason": "待视觉审查",
                        "scores": None, "tags": [], "duplicate_of": None})
    return records


def make_inventory(root, style_name, requested_count=12, recursive=False):
    return {"schema_version": SCHEMA_VERSION, "style_name": style_name,
            "source_root": str(Path(root).resolve()), "recursive": recursive,
            "requested_count": requested_count,
            "agent": {"name": "", "model": "", "review_method": ""},
            "selection_summary": "", "shortfall_reason": "",
            "selected_order": [], "reference_image": "", "reference_reason": "",
            "images": inventory(root, recursive)}


def validate_manifest(document, source_override=None):
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的筛图清单格式；需要 " + SCHEMA_VERSION)
    required = {"schema_version", "style_name", "source_root", "recursive", "requested_count", "agent",
                "selection_summary", "shortfall_reason", "selected_order", "reference_image", "reference_reason", "images"}
    if required - set(document):
        raise ValueError("缺少清单字段: " + ", ".join(sorted(required - set(document))))
    if not isinstance(document["shortfall_reason"], str):
        raise ValueError("shortfall_reason 须为字符串")
    name = document.get("style_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("style_name 须为 1–80 位英文、数字、连字符或下划线")
    count = document.get("requested_count")
    if type(count) is not int or not 2 <= count <= 100:
        raise ValueError("requested_count 须为 2–100 的整数")
    recursive = document.get("recursive")
    if type(recursive) is not bool:
        raise ValueError("recursive 须为布尔值")
    source = source_override or document.get("source_root")
    if not isinstance(source, (str, os.PathLike)) or not Path(source).is_absolute():
        raise ValueError("source_root 须为绝对目录路径；跨机器可在导入时重新指定源目录")
    root = Path(source).resolve(strict=True)
    actual_records = {record["path"]: record for record in inventory(root, recursive)}
    records = document.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("images 须包含目录内所有图片的审查记录")
    by_path = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("每条 images 记录必须是对象")
        fields = {"path", "sha256", "width", "height", "readable", "inspected", "decision", "reason", "scores", "tags", "duplicate_of"}
        if fields - set(record):
            raise ValueError("图片审查记录缺少字段: " + ", ".join(sorted(fields - set(record))))
        path = record.get("path")
        _inside(root, path)
        if path in by_path:
            raise ValueError(f"图片记录重复: {path}")
        by_path[path] = record
    if set(by_path) != set(actual_records):
        missing, extra = sorted(set(actual_records) - set(by_path)), sorted(set(by_path) - set(actual_records))
        raise ValueError(f"目录图片覆盖不完整或已变动；漏记 {missing[:5]}；多记 {extra[:5]}")
    for key in ("selection_summary", "reference_reason"):
        if not isinstance(document.get(key), str) or not document[key].strip():
            raise ValueError(f"缺少 {key}")
    agent = document.get("agent")
    if not isinstance(agent, dict) or any(not isinstance(agent.get(k), str) or not agent[k].strip() for k in ("name", "model", "review_method")):
        raise ValueError("agent 须填写 name、model、review_method")
    scored = []
    for path, record in by_path.items():
        actual = actual_records[path]
        for key in ("sha256", "width", "height", "readable"):
            value = record.get(key)
            if type(value) is not type(actual[key]) or value != actual[key]:
                raise ValueError(f"文件已变动或元数据不匹配: {path} / {key}")
        decision = record.get("decision")
        if decision not in ("selected", "rejected"):
            raise ValueError(f"尚未完成审查: {path}；needs_review 不能进入训练")
        if type(record.get("inspected")) is not bool or (actual["readable"] and not record["inspected"]):
            raise ValueError(f"可读图片未实际查看: {path}")
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            raise ValueError(f"缺少入选/排除理由: {path}")
        tags = record.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f"tags 须为字符串列表: {path}")
        duplicate = record.get("duplicate_of")
        if duplicate is not None and (not isinstance(duplicate, str) or duplicate not in by_path or duplicate == path):
            raise ValueError(f"duplicate_of 应指向另一张目录图片: {path}")
        if decision == "selected" and (not actual["readable"] or duplicate is not None):
            raise ValueError(f"不可读或重复图片不能入选: {path}")
        scores = record.get("scores")
        if actual["readable"]:
            if not isinstance(scores, dict) or set(scores) != set(SCORE_KEYS):
                raise ValueError(f"scores 必须包含四个固定分项: {path}")
            for key in SCORE_KEYS:
                value = scores[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10:
                    raise ValueError(f"分项须为 0–10 数值: {path} / {key}")
        elif scores is not None or decision != "rejected":
            raise ValueError(f"不可读图片应 rejected，scores 为 null: {path}")
        total = round(sum(scores[key] * weight for key, weight in zip(SCORE_KEYS, WEIGHTS)), 3) if scores else None
        scored.append({**record, "selection_score": total})
    selected = document.get("selected_order")
    if not isinstance(selected, list) or any(not isinstance(path, str) for path in selected):
        raise ValueError("selected_order 须为有序相对路径列表")
    if len(set(selected)) != len(selected) or set(selected) != {p for p, r in by_path.items() if r["decision"] == "selected"}:
        raise ValueError("selected_order 必须恰好覆盖全部入选图片，不得重复")
    if len(selected) < 2 or len(selected) > count:
        raise ValueError("入选图片须至少 2 张，且不能超过 requested_count")
    shortfall = document.get("shortfall_reason", "")
    if len(selected) != count and (not isinstance(shortfall, str) or not shortfall.strip()):
        raise ValueError("入选不足目标数量时，必须填写 shortfall_reason，不得为凑数加入不合适图片")
    hashes = [by_path[path]["sha256"] for path in selected]
    if len(set(hashes)) != len(hashes):
        raise ValueError("入选图片存在完全重复的文件内容")
    ref = document.get("reference_image")
    if ref not in selected:
        raise ValueError("reference_image 必须是入选图片之一")
    return {"manifest": document, "source_root": str(root), "style_name": name,
            "selected_paths": [str(_inside(root, p)) for p in selected], "reference_path": str(_inside(root, ref)),
            "rows": scored, "reviewed_count": len(records), "selected_count": len(selected)}


def load_manifest(path, source_override=None):
    with open(path, encoding="utf-8-sig") as source:
        return validate_manifest(json.load(source), source_override)


def materialize_dataset(validated, output_root):
    """只复制，不移动源图；唯一目录避免覆盖既有训练数据。"""
    checked = validate_manifest(validated["manifest"], validated["source_root"])
    target = Path(output_root) / (checked["style_name"] + "-" + uuid.uuid4().hex[:12])
    target.mkdir(parents=True, exist_ok=False)
    copies = []
    selected = checked["manifest"]["selected_order"]
    for relative, source in zip(selected, checked["selected_paths"]):
        destination = target / "images" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        expected = next(r["sha256"] for r in checked["rows"] if r["path"] == relative)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError(f"复制过程中源文件发生变动: {relative}；该目录未导入训练")
        copies.append(str(destination.resolve()))
    saved = target / "selection-manifest.json"
    saved.write_text(json.dumps(checked["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"schema_version": SCHEMA_VERSION, "source_root": checked["source_root"],
             "rows": checked["rows"], "copied_paths": copies}
    (target / "import-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    ref = str((target / "images" / checked["manifest"]["reference_image"]).resolve())
    return {"directory": str(target.resolve()), "manifest_path": str(saved.resolve()),
            "image_paths": copies, "reference_path": ref, "style_name": checked["style_name"]}
