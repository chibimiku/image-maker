# -*- coding: utf-8 -*-
"""Prepare and review paired eye/face/hair style experiments.

All generated evidence stays in a dedicated data/test-result directory. This
tool never calls an image API or changes generation checkpoints.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_WORK = ROOT / "data" / "test-result" / "20260929" / "eye-style-eval"
STAGES = ("first", "pipeline", "quality", "identity", "hands", "final_review", "publish")
SCORE_FIELDS = ("eyes", "face", "hair", "identity", "scene", "structure")
REVIEW_FIELDS = ("run_id", "style", "strategy", "status", "crop_checked", *SCORE_FIELDS,
                 "reference_leak", "notes")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def init(args):
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"Manifest already exists: {manifest_path}")
    styles = read_json(args.styles)
    entries = {}
    for name, style in styles.items():
        if name == "默认(无附加)" or not isinstance(style, dict) or not style.get("enabled", True):
            continue
        raw_ref = style.get("ref_image") or ""
        ref = Path(raw_ref) if raw_ref else None
        if ref and not ref.is_absolute():
            ref = ROOT / ref
        entries[name] = {"reference": str(ref.resolve()) if ref else "",
                         "reference_sha256": digest(ref) if ref and ref.is_file() else None,
                         "reference_box": None, "runs": {}}
    write_json(manifest_path, {"schema": 1, "styles": entries})
    (work / "README.md").write_text(
        "# 眼部、面部与头发画法评估\n\n"
        "此目录只存评估索引、裁剪拼图和人工评分；原始图与断点保持原位。\n"
        "`manifest.json` 记录画风参考图 SHA-256 和各次运行的成功阶段。\n"
        "GPT 断点用 `refresh --runs-root <路径>` 导入；Gemini 直出在"
        " `gemini-import.csv` 中登记后用 `import-csv --csv <文件>` 导入。\n"
        "运行 `sheet` 生成拼图；在 `review.csv` 的 0–4 分栏填分，"
        "不可见填 `NA`。`crop_checked` 填 yes 后评分才进入 `summary`。\n"
        "自动头部框仅供定位，拼图上的 `AUTO` 或 `FALLBACK` 均需人工确认。"
        "若裁剪错误，用 `box` 命令为参考图或某个阶段设置归一化"
        " `[x0,y0,x1,y1]`，然后重跑 `sheet`。\n"
        "`reference_leak` 填 yes/no/uncertain；身份、场景和结构分数越高表示越保真。"
        "`metrics` 只对已确认裁剪调用本机已有的 ResNet/HSV 诊断，不是眼睛评分。\n",
        encoding="utf-8")
    with (work / "gemini-import.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerow(("style", "strategy", "image", "run_id", "source"))
    print(f"Prepared {len(entries)} styles in {work}")


def existing_output(stage):
    if not isinstance(stage, dict) or stage.get("status") != "success":
        return ""
    return next((str(p) for p in reversed(stage.get("outputs") or []) if Path(p).is_file()), "")


def refresh(args):
    work = Path(args.work).resolve()
    manifest_path = work / "manifest.json"
    manifest = read_json(manifest_path)
    counts = {"found": 0, "unknown_style": 0}
    for root_arg in args.runs_root:
        root = Path(root_arg).resolve()
        if not root.is_dir():
            raise SystemExit(f"Runs root does not exist: {root}")
        for checkpoint_path in root.rglob("generation-checkpoint.json"):
            # A generation process can be writing this file while we scan it.
            try:
                checkpoint = read_json(checkpoint_path)
            except (OSError, json.JSONDecodeError):
                continue
            payload = checkpoint.get("snapshot", {}).get("request_payload", {})
            style = payload.get("style_name")
            if style not in manifest["styles"]:
                counts["unknown_style"] += 1
                continue
            run_id = checkpoint_path.parent.name
            previous = manifest["styles"][style]["runs"].get(run_id, {})
            stages = checkpoint.get("stages") or {}
            outputs = {name: existing_output(stages.get(name)) for name in STAGES}
            manifest["styles"][style]["runs"][run_id] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_status": checkpoint.get("status", "unknown"),
                "updated_at": checkpoint.get("updated_at", ""),
                "source_image": payload.get("analysis_result", {}).get("source_image_path", ""),
                "outputs": outputs,
                "boxes": previous.get("boxes", {}),
                "strategy": previous.get("strategy", "gpt_pipeline"),
            }
            counts["found"] += 1
    write_json(manifest_path, manifest)
    print(f"Indexed {counts['found']} checkpoints; {counts['unknown_style']} unknown styles")


def add(args):
    """Register an existing Gemini image or another candidate with an explicit label."""
    work = Path(args.work).resolve()
    manifest_path = work / "manifest.json"
    manifest = read_json(manifest_path)
    if args.style not in manifest["styles"]:
        raise SystemExit(f"Unknown enabled style: {args.style}")
    image = Path(args.image).resolve()
    if not image.is_file():
        raise SystemExit(f"Image does not exist: {image}")
    runs = manifest["styles"][args.style]["runs"]
    if args.run_id in runs:
        raise SystemExit(f"Run already exists: {args.run_id}")
    runs[args.run_id] = {
        "checkpoint": "", "checkpoint_status": "success", "updated_at": "",
        "source_image": str(Path(args.source).resolve()) if args.source else "",
        "outputs": {"candidate": str(image)}, "boxes": {}, "strategy": args.strategy,
    }
    write_json(manifest_path, manifest)
    print(f"Registered {args.run_id}: {args.style} / {args.strategy}")


def import_csv(args):
    work = Path(args.work).resolve()
    manifest_path = work / "manifest.json"
    manifest = read_json(manifest_path)
    count = 0
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            style = (row.get("style") or "").strip()
            run_id = (row.get("run_id") or "").strip()
            image = Path(row.get("image") or "").resolve()
            if not style or not run_id:
                continue
            if style not in manifest["styles"] or not image.is_file():
                raise SystemExit(f"Invalid style or image in CSV row: {style} / {image}")
            runs = manifest["styles"][style]["runs"]
            if run_id in runs:
                continue
            source = (row.get("source") or "").strip()
            runs[run_id] = {
                "checkpoint": "", "checkpoint_status": "success", "updated_at": "",
                "source_image": str(Path(source).resolve()) if source else "",
                "outputs": {"candidate": str(image)}, "boxes": {},
                "strategy": (row.get("strategy") or "gemini_direct").strip(),
            }
            count += 1
    write_json(manifest_path, manifest)
    print(f"Imported {count} new image runs")


def box(args):
    work = Path(args.work).resolve()
    manifest_path = work / "manifest.json"
    manifest = read_json(manifest_path)
    entry = manifest["styles"].get(args.style)
    if entry is None:
        raise SystemExit(f"Unknown style: {args.style}")
    bounds = [float(part) for part in args.bounds.split(",")]
    if len(bounds) != 4 or not (0 <= bounds[0] < bounds[2] <= 1 and
                                0 <= bounds[1] < bounds[3] <= 1):
        raise SystemExit("Box must be normalized x0,y0,x1,y1 within 0–1")
    if args.stage == "reference":
        entry["reference_box"] = bounds
    else:
        if not args.run_id:
            raise SystemExit("--run-id is required for a generated image")
        run = entry["runs"].get(args.run_id)
        if not run:
            raise SystemExit(f"Unknown run: {args.run_id}")
        path = run["outputs"].get(args.stage)
        if not path:
            raise SystemExit(f"No successful {args.stage} image in {args.run_id}")
        run.setdefault("boxes", {})[path] = bounds
    write_json(manifest_path, manifest)
    print(f"Saved {args.style} / {args.stage} box: {bounds}")


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def head_box(image, manual_box=None):
    width, height = image.size
    if manual_box is not None:
        if len(manual_box) != 4 or not (0 <= manual_box[0] < manual_box[2] <= 1 and
                                         0 <= manual_box[1] < manual_box[3] <= 1):
            raise ValueError(f"Invalid normalized box: {manual_box}")
        return (int(manual_box[0] * width), int(manual_box[1] * height),
                int(manual_box[2] * width), int(manual_box[3] * height)), "MANUAL"
    import numpy as np
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5,
                                     minSize=(max(20, int(height * 0.05)),) * 2)
    plausible = [(x, y, w, h) for x, y, w, h in faces
                 if 0.05 <= h / height <= 0.5 and w / width <= 0.65]
    if plausible:
        x, y, w, h = max(plausible, key=lambda box: box[2] * box[3])
        return (max(0, int(x - w * 0.7)), max(0, int(y - h * 0.65)),
                min(width, int(x + w * 1.7)), min(height, int(y + h * 1.45))), "AUTO"
    return (int(width * 0.2), 0, int(width * 0.8), int(height * 0.38)), "FALLBACK"


def tile(path, label, manual_box=None, size=320):
    image = read_image(path)
    box, method = head_box(image, manual_box)
    crop = ImageOps.fit(image.crop(box), (size, size), method=Image.Resampling.LANCZOS)
    thumb = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, 2 * size + 56), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    canvas.paste(crop, (0, size + 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, size + 3), f"{label} | {method}", fill="black")
    draw.text((6, 2 * size + 27), Path(path).name[:42], fill="black")
    return canvas, method, crop


def sheet(args):
    work = Path(args.work).resolve()
    manifest = read_json(work / "manifest.json")
    output_dir = work / "sheets"
    output_dir.mkdir(exist_ok=True)
    crops_dir = work / "crops"
    crops_dir.mkdir(exist_ok=True)
    old_rows = {}
    review_path = work / "review.csv"
    if review_path.is_file():
        with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
            old_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    rows = []
    made = 0
    for style, entry in manifest["styles"].items():
        ref = entry["reference"]
        if not ref or not Path(ref).is_file():
            continue
        for run_id, run in entry["runs"].items():
            paths = [("REF", ref), ("FIRST", run["outputs"].get("first")),
                     ("REPAINT", run["outputs"].get("pipeline")),
                     ("CANDIDATE", run["outputs"].get("candidate"))]
            final = run["outputs"].get("publish") or run["outputs"].get("final_review")
            if final:
                paths.append(("FINAL", final))
            tiles = []
            crop_methods = []
            run_crops = crops_dir / run_id
            run_crops.mkdir(exist_ok=True)
            for label, path in paths:
                if path and Path(path).is_file():
                    try:
                        bounds = (entry.get("reference_box") if label == "REF" else
                                  run.get("boxes", {}).get(str(Path(path))))
                        item, method, crop = tile(path, label, bounds)
                    except (OSError, ValueError) as exc:
                        print(f"Skipped unreadable image {path}: {exc}")
                        continue
                    tiles.append(item)
                    crop.save(run_crops / f"{label.lower()}.png")
                    crop_methods.append(f"{label}:{method}")
            if len(tiles) < 2:
                continue
            canvas = Image.new("RGB", (320 * len(tiles), tiles[0].height), "white")
            for index, item in enumerate(tiles):
                canvas.paste(item, (320 * index, 0))
            canvas.save(output_dir / f"{run_id}.jpg", quality=90)
            row = {key: "" for key in REVIEW_FIELDS}
            row.update(old_rows.get(run_id, {}))
            row.update(run_id=run_id, style=style, strategy=run.get("strategy", ""),
                       status=run.get("checkpoint_status", ""))
            rows.append(row)
            made += 1
            print(f"{run_id}: {', '.join(crop_methods)}")
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Built {made} sheets; review table: {review_path}")


def metrics(args):
    """Run the existing local ResNet/HSV renderer on checked head crops."""
    import importlib.util
    work = Path(args.work).resolve()
    module_path = ROOT / "tests" / "calc_style_similarity.py"
    spec = importlib.util.spec_from_file_location("calc_style_similarity", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (work / "review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        checked = {row["run_id"] for row in csv.DictReader(handle)
                   if row.get("crop_checked", "").lower() == "yes"}
    result = {"note": "Head-crop texture/palette diagnostics only; not an eye-style score.",
              "runs": {}}
    for run_id in sorted(checked):
        folder = work / "crops" / run_id
        ref = folder / "ref.png"
        candidates = [folder / f"{name}.png" for name in
                      ("first", "repaint", "candidate", "final")]
        candidates = [str(path) for path in candidates if path.is_file()]
        if ref.is_file() and candidates:
            result["runs"][run_id] = module.evaluate(str(ref), candidates, include_clip=False)
    write_json(work / "head-metrics.json", result)
    print(f"Scored {len(result['runs'])} manually checked head-crop sets")


def exp_image(record):
    """Select one displayed artifact while preserving the release-gate status."""
    resume = record.get("resume") or {}
    groups = []
    if resume.get("status") == "success":
        groups.extend((resume.get("copied") or [], resume.get("outputs") or []))
    groups.extend((record.get("copied") or [], record.get("outputs") or []))
    best = record.get("besteffort") or {}
    groups.extend(([best.get("copy")], [best.get("source")], record.get("candidates") or []))
    for paths in groups:
        for raw in paths:
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file():
                return str(path.resolve())
    return ""


def exp_build(args):
    """Build reference-versus-result evidence for the 21-style experiment."""
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    state = read_json(args.state)
    cases = []
    for entry in manifest:
        number, style = entry["index"], entry["style"]
        ref = entry["style_config"]["ref_image"]
        if not Path(ref).is_file():
            raise SystemExit(f"Missing reference: {ref}")
        group = []
        for slot in ("a", "b"):
            source = entry["sources"][slot]["abs"]
            for channel in ("gemini", "gpt"):
                key = f"{number:02d}-{slot}-{channel}"
                record = state.get(key, {})
                image = exp_image(record)
                status = ("resumed" if (record.get("resume") or {}).get("status") == "success" else
                          "success" if record.get("status") == "success" else
                          "candidate" if image else "missing")
                case = {"key": key, "style": style, "slot": slot, "channel": channel,
                        "reference": ref, "source": source, "image": image,
                        "status": status, "failure_kind": record.get("failure_kind", ""),
                        "error": record.get("error", "")}
                group.append(case)
                cases.append(case)
        cell_w, cell_h = 390, 490
        canvas = Image.new("RGB", (cell_w * 3, cell_h * 2), "#1c1c20")
        draw = ImageDraw.Draw(canvas)
        for row, slot in enumerate(("a", "b")):
            row_items = [("REFERENCE", ref, "reference")]
            row_items.extend((f"{case['key']} [{case['status']}]", case["image"], case["status"])
                             for case in group if case["slot"] == slot)
            for column, (label, image_path, _status) in enumerate(row_items):
                x, y = column * cell_w, row * cell_h
                if image_path:
                    picture = ImageOps.contain(read_image(image_path),
                                               (cell_w - 12, cell_h - 38),
                                               method=Image.Resampling.LANCZOS)
                    canvas.paste(picture, (x + (cell_w - picture.width) // 2,
                                           y + (cell_h - 38 - picture.height) // 2))
                draw.text((x + 8, y + cell_h - 28), f"{style[:23]} | {label}", fill="white")
        out = work / f"reference-{number:02d}-{style}.jpg"
        canvas.save(out, quality=91)
        print(out.name)
    write_json(work / "cases.json", {"count": len(cases), "cases": cases})
    print(f"Built {len(manifest)} comparison sheets for {len(cases)} cases")


def exp_metrics(args):
    """Reuse the cached local ResNet style model as a secondary diagnostic."""
    import importlib.util
    work = Path(args.work).resolve()
    cases = read_json(work / "cases.json")["cases"]
    module_path = ROOT / "tests" / "calc_style_similarity.py"
    spec = importlib.util.spec_from_file_location("calc_style_similarity", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = {}
    for case in cases:
        groups.setdefault(case["style"], []).append(case)
    output = []
    for style, members in groups.items():
        images = [case["image"] for case in members if case["image"]]
        scores = module.evaluate(members[0]["reference"], images, include_clip=False)
        by_file = {row["file"]: row for row in scores}
        for case in members:
            row = {**case, "metrics": by_file.get(case["image"], {})}
            output.append(row)
        print(f"Measured {style}: {len(images)} images", flush=True)
    write_json(work / "global-style-metrics.json",
               {"note": "Global renderer/palette similarity only; content and face style need visual review.",
                "cases": output})


def exp_heads(args):
    """Make head enlargements, marking every automatically selected region."""
    import numpy as np
    work = Path(args.work).resolve()
    cases = read_json(work / "cases.json")["cases"]
    model_path = ROOT / "data" / "exp" / "_prepare" / "face_detection_yunet_2023mar.onnx"
    detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                         score_threshold=0.55, nms_threshold=0.3)

    def crop_head(path):
        image = read_image(path)
        width, height = image.size
        scale = min(1.0, 1024.0 / max(width, height))
        small = image.resize((round(width * scale), round(height * scale)))
        sw, sh = small.size
        detector.setInputSize((sw, sh))
        _count, faces = detector.detect(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
        usable = [] if faces is None else [face for face in faces
                                            if 0.04 <= face[2] / sw <= 0.7 and
                                            0.04 <= face[3] / sh <= 0.6]
        if usable:
            face = max(usable, key=lambda value: value[2] * value[3] * value[-1])
            x, y, fw, fh = [float(v) / scale for v in face[:4]]
            box = (max(0, int(x - 0.7 * fw)), max(0, int(y - 0.65 * fh)),
                   min(width, int(x + 1.7 * fw)), min(height, int(y + 1.45 * fh)))
            method = "YUNET"
        else:
            box, method = head_box(image)
        crop = ImageOps.fit(image.crop(box), (320, 320), method=Image.Resampling.LANCZOS)
        return crop, method

    groups = {}
    for case in cases:
        groups.setdefault(case["style"], []).append(case)
    for index, (style, members) in enumerate(groups.items(), 1):
        items = [("REF", members[0]["reference"])]
        items.extend((case["key"], case["image"]) for case in members)
        canvas = Image.new("RGB", (320 * len(items), 350), "#1c1c20")
        draw = ImageDraw.Draw(canvas)
        for column, (label, path) in enumerate(items):
            if not path:
                continue
            crop, method = crop_head(path)
            canvas.paste(crop, (column * 320, 0))
            draw.text((column * 320 + 5, 327), f"{label} {method}", fill="white")
        canvas.save(work / f"heads-{index:02d}-{style}.jpg", quality=92)
        print(f"Head sheet {index:02d} {style}", flush=True)


def exp_index(args):
    """Write an 84-case artifact/status/metric index for auditability."""
    work = Path(args.work).resolve()
    cases = read_json(work / "global-style-metrics.json")["cases"]
    lines = ["# 84 个组合的逐图索引", "",
             "分数是整图纹理/调色/边缘辅助指标，不代表眼睛、身份或发布合格。"
             "`candidate` 保留原流程失败状态。", "",
             "| 组合 | 画风 | 状态 | 整图分数 | 选中图 |", "|---|---|---|---:|---|"]
    for case in cases:
        path = case["image"]
        link = (f"[打开图]({Path(os.path.relpath(path, work)).as_posix()})" if path else "缺图")
        score = case.get("metrics", {}).get("style_score")
        lines.append(f"| `{case['key']}` | `{case['style']}` | {case['status']} | "
                     f"{score:.4f} | {link} |" if score is not None else
                     f"| `{case['key']}` | `{case['style']}` | {case['status']} | — | {link} |")
    (work / "CASE-INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Indexed {len(cases)} cases in {work / 'CASE-INDEX.md'}")


def _binding_rows(binding, styles=None):
    """把 authoritative-binding.json 摊平成逐 arm 的行。"""
    rows = []
    for style, entry in (binding.get("styles") or {}).items():
        if styles and style not in styles:
            continue
        for slot in STYLE_SLOTS:
            for arm in STYLE_ARMS:
                key = "%s-%s" % (slot, arm)
                gpt = (entry.get("gpt") or {}).get(key) or {}
                gem = (entry.get("gemini") or {}).get(key) or {}
                rows.append({
                    "style": style, "slot": slot, "arm": arm,
                    "gpt_image": gpt.get("selected_output") or "",
                    "gpt_sha256": gpt.get("selected_output_sha256") or "",
                    "gpt_status": gpt.get("recorded_status") or "",
                    "gpt_base": gpt.get("base_image") or "",
                    "gpt_base_sha256": gpt.get("base_sha256") or "",
                    "gpt_style_ref": gpt.get("style_ref") or "",
                    "gpt_analysis": gpt.get("analysis_json") or "",
                    "gpt_source": gpt.get("source") or "",
                    "gpt_binding": gpt.get("binding") or {},
                    "gemini_image": gem.get("selected_output") or "",
                    "gemini_sha256": gem.get("selected_output_sha256") or "",
                    "gemini_request": gem.get("request_snapshot") or "",
                    "gemini_request_paths": gem.get("request_image_paths") or [],
                    "gemini_request_sha": gem.get("request_image_sha256") or [],
                    "gemini_source": gem.get("source") or "",
                    "gemini_analysis": gem.get("analysis_json") or "",
                    "diffs": (entry.get("request_diffs") or {}).get(slot) or {},
                })
    return rows


def _put(canvas, image, box, label, draw, subtitle=""):
    x, y, width, height = box
    if image is not None:
        thumb = ImageOps.contain(image, (width - 8, height - 34), method=Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (width - thumb.width) // 2, y + (height - 34 - thumb.height) // 2))
    draw.text((x + 5, y + height - 30), label[:64], fill="white")
    if subtitle:
        draw.text((x + 5, y + height - 16), subtitle[:70], fill="#b9b9c0")


def round_compare(args):
    """出对照页：只画权威绑定表里选中的图，找不到就写缺失，不猜最后一张。"""
    binding = record_of(args.binding)
    out = Path(args.out).resolve()
    (out / "comparisons").mkdir(parents=True, exist_ok=True)
    rows = _binding_rows(binding, set(args.style) if args.style else None)
    cell = int(args.cell)
    sheet_h = cell + 34
    made = []
    index = []
    for style in sorted({r["style"] for r in rows}):
        ref = next((r["gpt_style_ref"] for r in rows if r["style"] == style and r["gpt_style_ref"]), "")
        for slot in STYLE_SLOTS:
            pair = [r for r in rows if r["style"] == style and r["slot"] == slot]
            by_arm = {r["arm"]: r for r in pair}
            for channel in STYLE_CHANNELS:
                key = "gpt_image" if channel == "gpt" else "gemini_image"
                skey = "gpt_sha256" if channel == "gpt" else "gemini_sha256"
                items = [("REFERENCE (style reference)", ref, "")]
                missing = []
                for arm in STYLE_ARMS:
                    rec = by_arm.get(arm) or {}
                    path = rec.get(key) or ""
                    status = (rec.get("gpt_status") or "") if channel == "gpt" else "gemini-direct"
                    if path and Path(path).is_file():
                        items.append(("%s [%s]" % (arm.upper(), status), path,
                                      str(rec.get(skey) or "")[:12]))
                    else:
                        items.append(("%s [MISSING]" % arm.upper(), "", ""))
                        missing.append(arm)
                canvas = Image.new("RGB", (cell * len(items), sheet_h), "#1c1c20")
                draw = ImageDraw.Draw(canvas)
                for column, (label, path, sha) in enumerate(items):
                    image = None
                    if path and Path(path).is_file():
                        try:
                            image = read_image(path)
                        except (OSError, ValueError):
                            image = None
                    _put(canvas, image, (column * cell, 0, cell, cell), label, draw, sha)
                target = out / "comparisons" / ("%s-%s-%s.jpg" % (style, slot, channel))
                canvas.save(target, quality=90)
                made.append(str(target))
                index.append({"style": style, "slot": slot, "channel": channel,
                              "sheet": str(target), "sheet_sha256": sha256_of(target),
                              "reference": ref, "missing_arms": missing,
                              "images": [{"arm": arm,
                                          "path": (by_arm.get(arm) or {}).get(key) or "",
                                          "sha256": (by_arm.get(arm) or {}).get(skey) or ""}
                                         for arm in STYLE_ARMS]})
    write_json(out / "comparison-index.json",
               {"generated_at": now_stamp(), "binding": str(Path(args.binding).resolve()),
                "sheets": index, "count": len(index)})
    print("对照页 %d 张 → %s" % (len(made), out / "comparisons"))
    for item in index:
        if item["missing_arms"]:
            print("  ! %s/%s/%s 缺失: %s" % (item["style"], item["slot"], item["channel"],
                                             ",".join(item["missing_arms"])))
    return 0


def head_box_auto(image, detector=None):
    """返回 (标准化框, 方法)；detector 为 None 时退回 Haar，仍标 AUTO/FALLBACK。

    检测器只提出候选框：`method` 是自动方法名，是否真是目标脸必须人工确认。
    """
    width, height = image.size
    import numpy as np
    if detector is not None:
        scale = min(1.0, 1024.0 / max(width, height))
        small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        sw, sh = small.size
        detector.setInputSize((sw, sh))
        _count, faces = detector.detect(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
        usable = [] if faces is None else [f for f in faces
                                           if 0.04 <= f[2] / sw <= 0.7 and 0.04 <= f[3] / sh <= 0.6]
        if usable:
            face = max(usable, key=lambda v: v[2] * v[3] * v[-1])
            x, y, w, h = [float(v) / scale for v in face[:4]]
            box = (max(0.0, x - 0.7 * w) / width, max(0.0, y - 0.65 * h) / height,
                   min(width, x + 1.7 * w) / width, min(height, y + 1.45 * h) / height)
            return box, "YUNET"
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5,
                                     minSize=(max(20, int(height * 0.05)),) * 2)
    plausible = [(x, y, w, h) for x, y, w, h in faces
                 if 0.05 <= h / height <= 0.5 and w / width <= 0.65]
    if plausible:
        x, y, w, h = max(plausible, key=lambda box: box[2] * box[3])
        return ((max(0, x - w * 0.7) / width, max(0, y - h * 0.65) / height,
                 min(width, x + w * 1.7) / width, min(height, y + h * 1.45) / height), "AUTO")
    return (0.2, 0.0, 0.8, 0.38), "FALLBACK"


def head_box_plausible(box, size):
    """候选框是否像一张脸：不能贴边、要有足够面积、宽高比不能离谱。"""
    width, height = size
    x0, y0, x1, y1 = _norm_box_to_px(box, size)
    w, h = x1 - x0, y1 - y0
    if w < 24 or h < 24:
        return False, "框太小（%dx%d）" % (w, h)
    if w > width * 0.98 or h > height * 0.98:
        return False, "框几乎覆盖整图"
    ratio = w / max(1, h)
    if not 0.4 <= ratio <= 2.6:
        return False, "宽高比 %.2f 不像头部框" % ratio
    if w * h < 0.002 * width * height:
        return False, "面积占比过小"
    return True, ""


def eye_band(box):
    """从头部框推出上下眼睑所在的窄带（原图坐标，1:1 不放大）。

    窄带先与头部框求交：模型给的框太紧时，按比例外扩的窄带可能落到框外，
    那样切出来的就可能是嘴或背景 —— 这种情况要显式报出来，不能当成有效裁剪。
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        return None
    width, height = x1 - x0, y1 - y0
    band = [x0 + 0.04 * width, y0 + 0.36 * height, x1 - 0.04 * width, y0 + 0.72 * height]
    clipped = [max(x0, min(x1, band[0])), max(y0, min(y1, band[1])),
               max(x0, min(x1, band[2])), max(y0, min(y1, band[3]))]
    outside = (clipped[0] > band[0] + 1e-6 or clipped[1] > band[1] + 1e-6
               or clipped[2] < band[2] - 1e-6 or clipped[3] < band[3] - 1e-6)
    return {"band": clipped, "requested": band, "clipped": outside}


def _norm_box_to_px(box, size):
    width, height = size
    return (max(0, round(box[0] * width)), max(0, round(box[1] * height)),
            min(width, round(box[2] * width)), min(height, round(box[3] * height)))


def round_crops(args):
    """切 1:1 全分辨率眼部/头部裁剪：无框版用于评分，带框版只用于定位检查。

    记录 detected / human_checked / contains_target_face / eye_readable 四个独立字段：
    检测器命中不是人工确认（第二轮把 YUNET 命中直接写成 crop_checked=yes，结论不成立）。
    """
    binding = record_of(args.binding)
    out = Path(args.out).resolve()
    (out / "crops").mkdir(parents=True, exist_ok=True)
    manual = record_of(args.manual_box) if args.manual_box else {}
    model_path = ROOT / "data" / "exp" / "_prepare" / "face_detection_yunet_2023mar.onnx"
    detector = None
    if model_path.is_file():
        try:
            detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                                 score_threshold=0.55, nms_threshold=0.3)
        except cv2.error:
            detector = None
    rows = _binding_rows(binding, set(args.style) if args.style else None)
    review = {"generated_at": now_stamp(), "binding": str(Path(args.binding).resolve()),
              "reviewer": {"human_checked": "agent 逐张看图并填写（见 evidence）",
                           "automated": "OpenCV YuNet 只提出候选框，自动结果不写成人工确认"},
              "note": "green box 只用于定位检查；评分请用 *_eye_clean.png（无框、1:1 原始像素）。",
              "crops": []}
    for row in rows:
        for channel, key, skey in (("gpt", "gpt_image", "gpt_sha256"),
                                   ("gemini", "gemini_image", "gemini_sha256")):
            path = row[key]
            if not path or not Path(path).is_file():
                review["crops"].append({"style": row["style"], "slot": row["slot"], "arm": row["arm"],
                                        "channel": channel, "status": "missing", "image": path or ""})
                continue
            image = read_image(path)
            override_key = "%s/%s/%s" % (row["style"], row["slot"], row["arm"])
            override = manual.get(override_key) or manual.get("%s/%s" % (row["style"], row["slot"]))
            if override:
                box, method = tuple(float(v) for v in override), "MANUAL"
            else:
                box, method = head_box_auto(image, detector)
            plausible, implausible_reason = head_box_plausible(box, image.size)
            band_info = eye_band(box)
            px_head = _norm_box_to_px(box, image.size)
            px_eye = _norm_box_to_px(band_info["band"], image.size) if band_info else px_head
            head_clean = image.crop(px_head)
            eye_clean = image.crop(px_eye)
            stem = "%s-%s-%s-%s" % (row["style"], row["slot"], row["arm"], channel)
            head_path = out / "crops" / (stem + "_head_clean.png")
            eye_path = out / "crops" / (stem + "_eye_clean.png")
            head_over = out / "crops" / (stem + "_head_boxed.jpg")
            head_clean.save(head_path)
            eye_clean.save(eye_path)
            over = head_clean.copy()
            draw = ImageDraw.Draw(over)
            outline = (0, 200, 0) if (method != "FALLBACK" and plausible) else (220, 60, 60)
            lx0 = px_eye[0] - px_head[0]
            ly0 = px_eye[1] - px_head[1]
            draw.rectangle([lx0, ly0, lx0 + eye_clean.width, ly0 + eye_clean.height],
                           outline=outline, width=3)
            draw.text((4, 4), "%s %s" % (method, row["arm"]), fill=outline)
            over.save(head_over, quality=92)
            status = "pending_human"
            if method == "FALLBACK" or not plausible or (band_info and band_info["clipped"]):
                status = "auto_box_suspect"
            review["crops"].append({
                "style": row["style"], "slot": row["slot"], "arm": row["arm"], "channel": channel,
                "image": path, "image_sha256": sha256_of(path),
                "image_size": list(image.size),
                "detection_method": method,
                "detected": method in ("YUNET", "AUTO"),
                "auto_box_plausible": plausible,
                "auto_box_reason": implausible_reason,
                "eye_band_clipped_to_head_box": bool(band_info and band_info["clipped"]),
                "box_normalized": [round(v, 5) for v in box],
                "box_pixels": list(px_head), "eye_band_pixels": list(px_eye),
                "eye_crop_size": list(eye_clean.size),
                "head_clean": str(head_path), "eye_clean": str(eye_path),
                "head_boxed": str(head_over),
                "human_checked": None, "contains_target_face": None, "eye_readable": None,
                "reviewer": "", "evidence": "",
                "status": status,
            })
    write_json(out / "crop-review.json", review)
    suspect = [c for c in review["crops"] if c.get("status") == "auto_box_suspect"]
    print("裁剪 %d 条 → %s" % (len(review["crops"]), out / "crops"))
    print("  自动框可疑 %d 条（需人工定框）:" % len(suspect))
    for item in suspect:
        print("    - %s/%s/%s %s method=%s reason=%s clipped=%s" % (
            item["style"], item["slot"], item["arm"], item["channel"], item["detection_method"],
            item.get("auto_box_reason") or "", item.get("eye_band_clipped_to_head_box")))
    print("  human_checked / contains_target_face / eye_readable 一律留空，需逐张看图后填写")
    return 0


def round_review(args):
    """逐 arm 的评分与门禁表：门禁来自权威绑定，分数一律留空等人工/看图填写。

    第二轮把「一组 pair 分数」和门禁结论同时复制给两臂，导致 control 未通过也写成 pass；
    这里改成一图一行，且不预填任何分数。
    """
    binding = record_of(args.binding)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    crops = record_of(args.crops) if args.crops else {}
    crop_by_key = {}
    for item in crops.get("crops") or []:
        crop_by_key[(item.get("style"), item.get("slot"), item.get("arm"), item.get("channel"))] = item
    previous = {}
    if args.from_round2 and Path(args.from_round2).is_file():
        with Path(args.from_round2).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                previous[(row.get("style"), row.get("slot"), row.get("channel"))] = row
    fields = ["run_id", "style", "slot", "arm", "channel", "image", "image_sha256",
              "crop_eye_clean", "crop_head_boxed", "detection_method", "detected",
              "human_checked", "contains_target_face", "eye_readable",
              "eye_0_4", "face_0_4", "hair_0_4", "medium_0_4",
              "identity", "anatomy", "scene", "quality", "final_review", "overall",
              "evaluation_method", "evidence",
              "recorded_status", "gate", "gate_reason", "final_review_sha256",
              "identity_audit_sha256", "anatomy_audit_sha256", "final_review_audit_sha256",
              "request_has_intervention", "request_changed_keys",
              "previous_round_score_reference", "previous_round_gate_reference"]
    rows = []
    verdicts = []
    for row in _binding_rows(binding):
        for channel in STYLE_CHANNELS:
            key = "gpt_image" if channel == "gpt" else "gemini_image"
            skey = "gpt_sha256" if channel == "gpt" else "gemini_sha256"
            image = row[key]
            if not image or not Path(image).is_file():
                rows.append({f: "" for f in fields} | {
                    "run_id": "%s-%s-%s-%s" % (row["style"], row["slot"], row["arm"], channel),
                    "style": row["style"], "slot": row["slot"], "arm": row["arm"], "channel": channel,
                    "gate": "invalid", "gate_reason": "权威产物缺失，无法评审",
                    "evaluation_method": "unavailable"})
                continue
            crop = crop_by_key.get((row["style"], row["slot"], row["arm"], channel)) or {}
            diff = row["diffs"].get(channel) or {}
            binding_info = row["gpt_binding"] if channel == "gpt" else {}
            if channel == "gpt":
                bound = binding_info.get("verdict") == "bound"
                qerr = row["gpt_binding"].get("quality_gate_failed")
                gate = "audits_bound" if (bound and not qerr) else "review_required"
                if bound and not qerr:
                    reason = ("身份/人体/终审都绑定了同一张最终图且自身通过；"
                              "这只说明审计链在这张图上闭合，不代表画风改法已被证明有效")
                elif bound and qerr:
                    reason = ("审计链已绑定最终图，但上游质量门禁报错，按判红处理："
                              + "; ".join(binding_info.get("reasons") or []))
                else:
                    reason = "; ".join(binding_info.get("reasons") or []) or "审计链未闭合"
                checks = binding_info.get("checks") or {}
            else:
                gate = "audited_readonly" if diff.get("has_intervention") else "negative_control"
                reason = ("直出请求确有差异：" + ", ".join(diff.get("changed_keys") or [])) if diff.get(
                    "has_intervention") else "直出请求与对照完全相同，属无干预重复，不能归因给变体"
                checks = {}
            run_id = "%s-%s-%s-%s" % (row["style"], row["slot"], row["arm"], channel)
            prev = previous.get((row["style"], row["slot"], channel)) or {}
            entry = {
                "run_id": run_id, "style": row["style"], "slot": row["slot"], "arm": row["arm"],
                "channel": channel, "image": image, "image_sha256": sha256_of(image),
                "crop_eye_clean": crop.get("eye_clean") or "",
                "crop_head_boxed": crop.get("head_boxed") or "",
                "detection_method": crop.get("detection_method") or "",
                "detected": "" if crop.get("detected") is None else str(crop.get("detected")).lower(),
                "human_checked": "", "contains_target_face": "", "eye_readable": "",
                "eye_0_4": "", "face_0_4": "", "hair_0_4": "", "medium_0_4": "",
                "identity": "", "anatomy": "", "scene": "", "quality": "", "final_review": "",
                "overall": "", "evaluation_method": "",
                "evidence": (crop.get("eye_clean") and ("评分依据 %s" % Path(crop["eye_clean"]).name)) or "",
                "recorded_status": row["gpt_status"] if channel == "gpt" else "gemini-direct",
                "gate": gate, "gate_reason": reason,
                "final_review_sha256": skey and row[skey] or "",
                "identity_audit_sha256": checks.get("identity", ""),
                "anatomy_audit_sha256": checks.get("anatomy", ""),
                "final_review_audit_sha256": checks.get("final_review", ""),
                "request_has_intervention": str(bool(diff.get("has_intervention"))).lower(),
                "request_changed_keys": ",".join(diff.get("changed_keys") or []),
                "previous_round_score_reference": (
                    "eye=%s face=%s hair=%s medium=%s（上一轮只记了一组，不能当两臂分数）" % (
                        prev.get("eye_0_4", "-"), prev.get("face_0_4", "-"),
                        prev.get("hair_0_4", "-"), prev.get("medium_0_4", "-")) if prev else ""),
                "previous_round_gate_reference": (
                    "/".join(x for x in (prev.get("identity", ""), prev.get("anatomy", ""),
                                         prev.get("final_review", "")) if x) if prev else ""),
            }
            rows.append(entry)
            verdicts.append({"run_id": run_id, "gate": gate, "reason": reason})
    with (out / "per-arm-review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "per-arm-gates.json",
               {"generated_at": now_stamp(), "binding": str(Path(args.binding).resolve()),
                "scoring_rule": "0–4 分与门禁逐图独立填写；NA=看不清；unknown 不等于 pass。"
                                "本次执行者逐张看图后填写，未填的保持空。",
                "rows": rows, "verdicts": verdicts})
    print("逐 arm 评分表 %d 行 → %s" % (len(rows), out / "per-arm-review.csv"))
    print("  分数列全部留空；门禁列由权威绑定给出，两臂不再共用一组结论")
    return 0


ROUND3_PLAN = {
    "round": "eye-face-hair-round3-20261001",
    "status": "prepared_not_run",
    "max_experiment_image_calls": 16,
    "automatic_repairs": False,
    "publishing": False,
    "production_style_writeback": False,
    "experiments": [
        {
            "id": "P1",
            "style": "goto-p",
            "stage": "single_gemini_repaint_from_frozen_gpt_base",
            "question": "同一正确首图与同一旧重绘文字下，把完整画风图一起送进生成请求，"
                        "是否引起参考角色/服装/姿势迁移？",
            "variable": "style_reference_image_in_generation_request",
            "arms": [
                {
                    "name": "full_ref",
                    "base": "round2:results/goto-p/{slot}/control-gpt/first/first-pass.png",
                    "style_ref": "style_entry:goto-p",
                    "style_reference_sent": True,
                    "repaint_ref": "style",
                    "expected_diff": [
                        "reference_image_paths: 多一张画风参考图（Image 2），两臂文字段落逐字相同",
                        "repaint_reference_mode: style vs none",
                    ],
                },
                {
                    "name": "source_only",
                    "base": "round2:results/goto-p/{slot}/control-gpt/first/first-pass.png",
                    "style_ref": "style_entry:goto-p",
                    "style_reference_sent": False,
                    "repaint_ref": "none",
                    "expected_diff": [
                        "reference_image_paths: 只发送冻结首图，两臂文字段落逐字相同",
                        "repaint_reference_mode: none vs style",
                    ],
                },
            ],
            "image_calls": 8,
            "primary_outcome": "复制参考角色（发色/瞳色/发型/衣装/姿势/道具/背景）次数、源身份通过数",
            "secondary_outcome": "发束与眼部画法、局部过白",
        },
        {
            "id": "P2",
            "style": "sheya-style",
            "stage": "single_source_only_eye_clause_repaint",
            "question": "不发送完整画风图、只替换一条眼睑/虹膜画法句，是否稳定改善上眼睑与虹膜组织？",
            "variable": "one_eye_rendering_sentence",
            "generation_style_reference": False,
            "arms": [
                {
                    "name": "control_eye_clause",
                    "base": "round2:results/sheya-style/{slot}/control-gpt/first/first-pass.png",
                    "base_override": {"b": "round2:results/sheya-style/b/variant-gpt/first/first-pass.png"},
                    "style_ref": "",
                    "style_reference_sent": False,
                    "repaint_ref": "none",
                    "clauses_only": True,
                    "clause_mode": "keep",
                    "expected_diff": [
                        "reference_image_paths: 两臂都只发送冻结首图",
                        "repaint prompt: 保留当前眼部画法句，另一臂替换该句",
                    ],
                },
                {
                    "name": "variant_eye_clause",
                    "base": "round2:results/sheya-style/{slot}/control-gpt/first/first-pass.png",
                    "base_override": {"b": "round2:results/sheya-style/b/variant-gpt/first/first-pass.png"},
                    "style_ref": "",
                    "style_reference_sent": False,
                    "repaint_ref": "none",
                    "clauses_only": True,
                    "clause_mode": "replace_eye_clause",
                    "replace_eye_clause": {
                        "old": "Keep the already approved sheya eye grammar unchanged: narrow alert anime eyes "
                               "with a tapered upper lash, compact iris and focused gaze. Do not further "
                               "sharpen, enlarge, round or redesign the eyes.",
                        "new": "Render the upper eyelid as a clear oblique upward-slanting line with a restrained "
                               "lower lid, a compact iris and grouped lashes; keep the current gaze, expression "
                               "and open/closed eye state, and build the light and small highlights inside the "
                               "iris colour the source already has.",
                    },
                    "expected_diff": [
                        "reference_image_paths: 两臂都只发送冻结首图",
                        "repaint prompt: 只用这 1 句替换保留臂的同位置句，其余段落逐字相同",
                    ],
                },
            ],
            "image_calls": 8,
            "primary_outcome": "上眼睑与虹膜组织是否改善，且源瞳色/睁闭眼状态/构图保持",
        },
    ],
    "required_per_arm": [
        "run_id", "experiment_id", "style", "slot", "arm", "repeat", "stage",
        "base_path", "base_sha256", "analysis_path", "analysis_sha256", "style_snapshot_sha256",
        "effective_request_path", "effective_request_sha256", "expected_diff", "actual_diff",
        "output_path", "output_sha256", "audit_image_sha256", "identity", "anatomy", "scene",
        "quality", "final_review", "overall", "eye_score", "face_score", "hair_score", "medium_score",
        "human_crop_review", "eye_readable", "evaluation_method", "evidence", "call_ids", "cost_status",
    ],
}


def _resolve_plan_base(plan, round_dir, row, spec_base, overrides):
    """把 `round2:<相对路径>` 解析成真实文件；支持按 slot 覆盖。"""
    slot = row["slot"]
    template = (overrides or {}).get(slot) or spec_base or ""
    if not template:
        return "", ""
    if template.startswith("round2:"):
        rel = template.split(":", 1)[1].format(slot=slot)
        return str((round_dir / rel).resolve()), "plan"
    return "", ""


def _style_reference_for(styles, style):
    entry = (styles or {}).get(style) or {}
    ref = str(entry.get("ref_image") or "")
    return ref if ref and Path(ref).is_file() else ""


def load_round_styles(round_dir, plan):
    """冻结本轮用到的画风条目：优先用本轮快照，没有就取当前正式配置并存快照。"""
    snapshot = round_dir / "snapshots" / "config-styles-frozen.json"
    if snapshot.is_file():
        frozen = record_of(snapshot)
        return frozen.get("styles") or {}, str(snapshot)
    source = ROOT / "conf" / "config-styles.json"
    styles = record_of(source)
    used = sorted({item["style"] for item in plan.get("experiments") or [] if item.get("style")})
    frozen = {"_source": str(source), "_frozen_at": now_stamp(), "_why":
              "第三轮 P1/P2 只用这几个条目；整份表按原样冻结，避免后续 conf 变动影响复现。",
              "_source_sha256": sha256_of(source),
              "styles": {name: styles.get(name) for name in used}}
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    write_json(snapshot, frozen)
    return frozen.get("styles") or {}, str(snapshot)


def _clip_sentence(prompt, clause, replacement):
    """只替换一条以句为单位写成的画法条款；命中数不是 1 就拒绝。"""
    text = str(prompt or "")
    hits = text.count(clause)
    if hits != 1:
        return None, "条款在提示词里命中 %d 次（要求恰好 1 次）" % hits
    return text.replace(clause, replacement), ""


def build_repaint_prompt(clauses, firmware_path, scope="full"):
    """复现 post_process 的重绘提示词拼装（style 模式），用于实验前的 dry-run 对账。

    必须与 `utils/post_process.run_pipeline` 的拼装顺序一致：固件 → 参考图条款 →
    STYLE LANGUAGE → 编辑范围句 → IDENTITY LOCK；少一段就跟实际送出的文字对不上。
    """
    from utils.post_process import (IDENTITY_LOCK_CLAUSE, REPAINT_SCOPE_CLAUSES,
                                    STYLE_REF_FACE_HAIR_GRAMMAR, STYLE_REF_ROLE_IN_REPAINT,
                                    resolve_firmware_text)
    prompt = (resolve_firmware_text(firmware_path) or "") + STYLE_REF_ROLE_IN_REPAINT + STYLE_REF_FACE_HAIR_GRAMMAR
    items = [str(c).strip() for c in (clauses or []) if str(c).strip()]
    if items:
        prompt += "\n\nSTYLE LANGUAGE (from the reference image):\n- " + "\n- ".join(items)
    scope_key = str(scope or "").strip().lower()
    scope_clause = REPAINT_SCOPE_CLAUSES.get(scope_key, "")
    if scope_clause:
        prompt += scope_clause
        prompt += IDENTITY_LOCK_CLAUSE
    return prompt


def round_prep(args):
    """冻结本轮输入、写 run-plan.json 与对账用的提示词快照（不调用任何接口）。"""
    round_dir = Path(args.round).resolve()
    plan = record_of(args.plan) if args.plan else ROUND3_PLAN
    out = Path(args.out).resolve() if args.out else round_dir
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    if args.dump_plan or not args.plan:
        write_json(out / "experiment-plan.effective.json", plan)
    binding = record_of(args.binding) if args.binding else {}
    styles, styles_snapshot = load_round_styles(round_dir, plan)
    runs = []
    problems = []
    for experiment in plan.get("experiments") or []:
        style = experiment["style"]
        entry = styles.get(style) or {}
        clauses = list(entry.get("repaint_clauses") or [])
        firmware = "prompts/gpt-image-optimize/repaint-system-conservative-v5.md"
        style_ref = _style_reference_for(styles, style)
        for slot in STYLE_SLOTS:
            # 先解析 base：binding 优先（权威产物），plan 里的 round2: 模板作为兜底
            manifest_path = round_dir.parent / "eye-face-hair-round2-20260930" / "results" / style / slot / "control-gpt" / "pipeline" / "request.json"
            base_from_binding = ""
            for row in _binding_rows(binding):
                if row["style"] == style and row["slot"] == slot and row["arm"] == "control":
                    base_from_binding = row["gpt_base"] or row["gpt_image"]
            for arm in experiment["arms"]:
                template = (arm.get("base_override") or {}).get(slot) or arm.get("base") or ""
                base = ""
                if template.startswith("round2:"):
                    rel = template.split(":", 1)[1].format(slot=slot)
                    base = str((round_dir.parent / "eye-face-hair-round2-20260930" / rel).resolve())
                elif base_from_binding:
                    base = base_from_binding
                if not base or not Path(base).is_file():
                    problems.append("%s/%s/%s: 找不到冻结首图 (%s)" % (style, slot, arm["name"], base or template))
                    continue
                stage_dir = out / "results" / experiment["id"] / style / slot / arm["name"]
                frozen_dir = out / "inputs" / ("%s-%s-%s" % (experiment["id"], style, slot))
                frozen_dir.mkdir(parents=True, exist_ok=True)
                frozen_base = frozen_dir / ("base-%s.png" % arm["name"] if arm.get("base_override") else "base.png")
                if not frozen_base.is_file():
                    frozen_base.write_bytes(Path(base).read_bytes())
                frozen_ref = ""
                if style_ref and arm.get("style_reference_sent"):
                    frozen_ref = str(frozen_dir / ("style-ref-%s" % Path(style_ref).name))
                    if not Path(frozen_ref).is_file():
                        Path(frozen_ref).write_bytes(Path(style_ref).read_bytes())
                prompt = build_repaint_prompt(clauses, firmware)
                clauses_only = bool(arm.get("clauses_only"))
                if arm.get("style_reference_sent"):
                    pass
                elif clauses_only:
                    # P2：不挂画风图、也不提参考图，只把条款当成文字规格。
                    from utils.post_process import IDENTITY_LOCK_CLAUSE, REPAINT_SCOPE_CLAUSES, resolve_firmware_text
                    prompt = resolve_firmware_text(firmware) or ""
                    items = [str(c).strip() for c in clauses if str(c).strip()]
                    if items:
                        prompt += ("\n\nSTYLE LANGUAGE (rendering targets for this repaint):\n- "
                                   + "\n- ".join(items))
                    prompt += REPAINT_SCOPE_CLAUSES.get("full", "")
                    prompt += IDENTITY_LOCK_CLAUSE
                elif style_ref:
                    # P1 source-only 臂：文字与挂图臂逐字相同，只把「Image 1/Image 2」
                    # 换成具名指代（与 utils/post_process.text_without_image_phrasing 同一函数）。
                    from utils.post_process import text_without_image_phrasing
                    prompt = text_without_image_phrasing(prompt, Path(style_ref).name)
                clause_note = ""
                if clauses_only:
                    clause_note = "不发送画风图；条款作为文字规格"
                if arm.get("clause_mode") == "replace_eye_clause":
                    old = (arm.get("replace_eye_clause") or {}).get("old") or ""
                    new = (arm.get("replace_eye_clause") or {}).get("new") or ""
                    prompt_after, error = _clip_sentence(prompt, old, new)
                    if error:
                        problems.append("%s/%s/%s: %s" % (style, slot, arm["name"], error))
                        continue
                    prompt = prompt_after
                    clause_note = "已用 1 句替换保留臂的同位置句"
                runs.append({
                    "run_id": "%s-%s-%s-%s-%s" % (plan.get("round", "round3"), experiment["id"], style,
                                                   slot, arm["name"]),
                    "experiment_id": experiment["id"], "style": style, "slot": slot,
                    "arm": arm["name"], "stage": experiment["stage"],
                    "base_path": str(frozen_base), "base_sha256": sha256_of(frozen_base),
                    "base_source": base, "base_source_sha256": sha256_of(base),
                    "analysis_path": "", "analysis_sha256": "",
                    "style_ref_path": frozen_ref, "style_ref_sha256": sha256_of(frozen_ref) if frozen_ref else "",
                    "style_ref_source": style_ref if arm.get("style_reference_sent") else "",
                    "style_reference_sent": bool(arm.get("style_reference_sent")),
                    "repaint_ref": arm.get("repaint_ref", "none"),
                    "clauses_only": bool(arm.get("clauses_only")),
                    "firmware": firmware, "firmware_sha256": sha256_of(ROOT / firmware),
                    "clauses": clauses, "clauses_source": str(entry.get("clauses_source") or ""),
                    "clause_note": clause_note,
                    "expected_diff": arm.get("expected_diff") or [],
                    "repaint_prompt_preview": prompt,
                    "repaint_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "repaint_prompt_chars": len(prompt),
                    "output_dir": str(stage_dir),
                    "repeat": int(plan.get("repeats", 2)),
                })
    run_plan = {"generated_at": now_stamp(), "round": str(round_dir), "plan": str(Path(args.plan).resolve()),
                "styles_snapshot": styles_snapshot, "styles_snapshot_sha256": sha256_of(styles_snapshot),
                "max_experiment_image_calls": plan.get("max_experiment_image_calls"),
                "audit_only": True, "automatic_repairs": False,
                "runs_total": len(runs) * int(plan.get("repeats", 2)),
                "runs": runs, "problems": problems}
    write_json(out / "run-plan.json", run_plan)
    print("运行计划 %d 个 arm × %d 次 = %d 次图片调用 → %s" % (
        len(runs), int(plan.get("repeats", 2)), len(runs) * int(plan.get("repeats", 2)),
        out / "run-plan.json"))
    for item in runs:
        print("  %-10s %-12s %s/%-2s ref=%-5s prompt=%d字 sha=%s" % (
            item["experiment_id"], item["arm"], item["style"], item["slot"],
            "yes" if item["style_reference_sent"] else "no", item["repaint_prompt_chars"],
            item["repaint_prompt_sha256"][:10]))
    if problems:
        print("  问题 %d 条:" % len(problems))
        for text in problems:
            print("    - " + text)
    return 1 if problems else 0


def summary(args):
    work = Path(args.work).resolve()
    with (work / "review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = [row for row in rows if row.get("crop_checked", "").lower() == "yes"]
    data = {"indexed": len(rows), "reviewed": len(reviewed), "by_strategy": {}}
    for row in reviewed:
        strategy = row.get("strategy") or "unspecified"
        group = data["by_strategy"].setdefault(strategy, {"count": 0, "scores": {}})
        group["count"] += 1
        for field in SCORE_FIELDS:
            try:
                score = float(row[field])
            except (ValueError, TypeError):
                continue
            if not 0 <= score <= 4:
                raise SystemExit(f"Score outside 0–4: {row['run_id']} {field}={score}")
            group["scores"].setdefault(field, []).append(score)
    for group in data["by_strategy"].values():
        group["scores"] = {name: {"mean": round(sum(values) / len(values), 3), "n": len(values)}
                           for name, values in group["scores"].items()}
    write_json(work / "summary.json", data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 受控轮次（round2 / round3）证据校正：权威绑定、对照、裁剪、评分、调用账本
#
# 第二轮把 15 个一次性脚本放进了 data/exp/<round>/tools/，违反仓库目录规则；
# 第三轮把这套「读记录、绑哈希、出对照、汇总账本」的能力合并进本 CLI，
# 全部按 --round 参数化，不再往 data/ 下放可执行脚本。
# ---------------------------------------------------------------------------

STYLE_SLOTS = ("a", "b")
STYLE_ARMS = ("control", "variant")
STYLE_CHANNELS = ("gpt", "gemini")
ROUND2_STYLES = ("dall-e-v2", "fuzichoco-v2", "goto-p", "renian", "sheya-style", "tid")
# 审计里允许忽略的非输入字段（输出路径、时间戳等）。
IGNORED_REQUEST_KEYS = ("output_dir", "work_dir", "final_dir", "save_sub_dir", "started_at",
                        "finished_at", "request_dump", "log", "outputs", "selected_output")


def sha256_of(path):
    if not path or not Path(path).is_file():
        return ""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def record_of(path):
    """读 JSON；读不到就返回 {}，让调用方显式报缺失。"""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def gpt_arm_facts(round_dir, style, slot, arm):
    """GPT 臂的权威事实：pipeline/request.json 的 selected_output + 各项审计绑定。"""
    base = Path(round_dir) / "results" / style / slot / f"{arm}-gpt"
    manifest_path = base / "pipeline" / "request.json"
    manifest = record_of(manifest_path)
    selected = str(manifest.get("selected_output") or "")
    identity = (manifest.get("identity") or {}).get("final") or {}
    anatomy = manifest.get("anatomy_review") or {}
    final_review = manifest.get("final_review") or {}
    quality = manifest.get("quality_refine") or {}
    run_record = record_of(base / "run-record.json")
    # 终审：清单里可能只留了 audit_error，而真实审计文件在 pipeline/ 下。
    # 两者不一致本身就是证据，单独记下来，不互相覆盖。
    final_audit_file = base / "pipeline" / "final-quality-audit.json"
    final_on_disk = record_of(final_audit_file)
    final_from = "manifest"
    if not final_review and final_on_disk:
        final_review = final_on_disk
        final_from = "pipeline/final-quality-audit.json"
    elif final_review and final_on_disk and str(final_review.get("candidate") or "") != str(final_on_disk.get("candidate") or ""):
        final_from = "manifest（与 pipeline/final-quality-audit.json 的候选不一致）"
    facts = {
        "channel": "gpt", "arm": arm, "style": style, "slot": slot,
        "manifest": str(manifest_path), "manifest_exists": manifest_path.is_file(),
        "manifest_sha256": sha256_of(manifest_path),
        "recorded_status": str(manifest.get("status") or ""),
        "selected_output": selected,
        "selected_output_exists": bool(selected) and Path(selected).is_file(),
        "selected_output_sha256": sha256_of(selected),
        "base_image": str(manifest.get("base") or ""),
        "base_sha256": sha256_of(str(manifest.get("base") or "")),
        "quality_audit_error": str(quality.get("audit_error") or ""),
        "quality_severity_before": (quality.get("before") or {}).get("severity"),
        "quality_severity_after": (quality.get("after") or {}).get("severity"),
        "identity": {"image": str(identity.get("image") or ""),
                     "sha256": str(identity.get("candidate_sha256") or "") or sha256_of(str(identity.get("image") or "")),
                     "mismatch": identity.get("mismatch"), "severity": identity.get("severity"),
                     "audit_error": str(identity.get("audit_error") or "")},
        "anatomy": {"image": str(anatomy.get("candidate") or ""),
                    "sha256": str(anatomy.get("candidate_sha256") or "") or sha256_of(str(anatomy.get("candidate") or "")),
                    "audit_error": str(anatomy.get("audit_error") or "")},
        "final_review": {"image": str(final_review.get("candidate") or ""),
                         "sha256": str(final_review.get("candidate_sha256") or "") or sha256_of(str(final_review.get("candidate") or "")),
                         "audit_error": str(final_review.get("audit_error") or ""),
                         "source": final_from,
                         "artifact": str(final_audit_file) if final_audit_file.is_file() else "",
                         "artifact_sha256": sha256_of(final_audit_file)},
        "run_record_selected": str(run_record.get("selected_output") or ""),
        "source": str(manifest.get("source") or ""), "analysis_json": str(manifest.get("analysis_json") or ""),
        "style_ref": str(manifest.get("repaint_style_ref") or (manifest.get("request") or {}).get("style_ref_path") or ""),
        "firmware": str(manifest.get("firmware") or ""),
        "steps": manifest.get("steps") or {},
        "repaint_reference_mode": (manifest.get("request") or {}).get("repaint_reference_mode"),
        "first_pass_prompt_sha256": hashlib.sha256(
            str((manifest.get("request") or {}).get("prompt") or "").encode("utf-8")).hexdigest(),
    }
    facts["binding"] = binding_verdict(facts)
    return facts


def gemini_arm_facts(round_dir, style, slot, arm):
    """Gemini 直出臂：run-record 的产物 + 真实请求快照（含图片顺序与角色）。"""
    base = Path(round_dir) / "results" / style / slot / f"{arm}-gemini"
    record_path = base / "run-record.json"
    record = record_of(record_path)
    first_dir = base / "first"
    snapshot_path = first_dir / "gemini-request-1.json"
    snapshot = record_of(snapshot_path)
    images = sorted([str(p) for p in first_dir.glob("*.jpg")] + [str(p) for p in first_dir.glob("*.png")])
    outputs = [str(p) for p in (record.get("outputs") or []) if str(p).strip()]
    note = ""
    if len(images) == 1:
        selected = images[0]
    elif outputs:
        selected = outputs[-1]
        note = "first/ 下有多张图，按 run-record.outputs 末项取值"
    else:
        selected = ""
        note = "没有可判定的直出产物"
    if len(outputs) != len(images) and images:
        note = (note + "；" if note else "") + "run-record.outputs(%d) 与 first/ 实际图片数(%d) 不一致" % (
            len(outputs), len(images))
    return {
        "channel": "gemini", "arm": arm, "style": style, "slot": slot,
        "manifest": str(record_path), "manifest_exists": record_path.is_file(),
        "manifest_sha256": sha256_of(record_path),
        "recorded_status": "rc=%s" % record.get("returncode") if record else "",
        "selected_output": selected, "selected_output_exists": bool(selected) and Path(selected).is_file(),
        "selected_output_sha256": sha256_of(selected),
        "request_snapshot": str(snapshot_path) if snapshot_path.is_file() else "",
        "request_snapshot_sha256": sha256_of(snapshot_path),
        "request_prompt": str(snapshot.get("prompt") or ""),
        "request_instructions": str(snapshot.get("instructions") or ""),
        "request_post_instructions": str(snapshot.get("post_instructions") or ""),
        "request_model": str(snapshot.get("model") or ""),
        "request_aspect_ratio": str(snapshot.get("aspect_ratio") or ""),
        "request_image_paths": [str(p) for p in (snapshot.get("image_paths") or [])],
        "request_image_sha256": [sha256_of(str(p)) for p in (snapshot.get("image_paths") or [])],
        "all_outputs": outputs, "images_in_first": images, "note": note,
        "source": str((record.get("filenames") or {}).get("source") or ""),
        "analysis_json": str((record.get("filenames") or {}).get("analysis_json") or ""),
    }


def binding_verdict(facts):
    """判断「选中的图」和「各项审计审的图」是不是同一份内容。

    只回答问题本身：能证明同一份 → bound；审计写在别的图上 → stale；
    没有这项审计 → missing；审计跑了但报错 → audit_failed；选中的图不存在 → invalid。
    不把「没有审计」和「审计判红」混为一谈，也不把 unknown 当通过。
    """
    selected_sha = facts.get("selected_output_sha256") or ""
    verdict = {"selected_sha256": selected_sha, "checks": {}, "verdict": "unknown", "reasons": []}
    if not selected_sha:
        verdict["verdict"] = "invalid"
        verdict["reasons"].append("selected_output 缺失或文件不存在")
        return verdict
    for name in ("identity", "anatomy", "final_review"):
        entry = facts.get(name) or {}
        audit_sha = str(entry.get("sha256") or "")
        if not audit_sha:
            # 这一轮的审计文件里没有内容哈希，因此无法证明它审的是哪一张。
            verdict["checks"][name] = "audit_failed" if entry.get("audit_error") else "missing"
            verdict["unbound"] = True
            verdict["reasons"].append(
                "%s：无法绑定（该轮没有记录被审图片的内容哈希）%s" % (
                    name, "；审计本身报错：" + str(entry["audit_error"])[:100] if entry.get("audit_error") else ""))
        elif entry.get("audit_error"):
            verdict["checks"][name] = "audit_failed"
            verdict["reasons"].append("%s：审计报错（%s）" % (name, str(entry["audit_error"])[:100]))
        elif audit_sha == selected_sha:
            verdict["checks"][name] = "bound"
        else:
            verdict["checks"][name] = "stale"
            verdict["reasons"].append("%s：审计绑定的是另一张图（%s…），不能用来证明最终图合格" % (
                name, audit_sha[:10]))
    checks = set(verdict["checks"].values())
    if checks == {"bound"}:
        verdict["verdict"] = "bound"
    elif "stale" in checks:
        verdict["verdict"] = "invalid"
    elif "audit_failed" in checks:
        verdict["verdict"] = "unknown"
    else:
        verdict["verdict"] = "unverifiable_selection"
    if facts.get("quality_audit_error"):
        verdict["reasons"].append("上游质量门禁报错：" + str(facts["quality_audit_error"])[:120])
        verdict["quality_gate_failed"] = True
    return verdict


def round_bind(args):
    """冻结本轮引用的上一轮证据，写出权威绑定表（不猜 mtime、不改原目录）。"""
    round_dir = Path(args.round).resolve()
    out = Path(args.out).resolve()
    (out / "frozen").mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at": now_stamp(), "round_dir": str(round_dir),
        "authority_rule": {
            "gpt": "results/<style>/<slot>/<arm>-gpt/pipeline/request.json 的 selected_output",
            "gemini": "results/<style>/<slot>/<arm>-gemini/run-record.json 的产物（first/ 唯一图；多图才回退 outputs 末项并标 note）",
            "forbidden": "按 mtime 猜最后一张图；本轮不复制、不改写上一轮任何文件",
        },
        "styles": {}, "frozen_inputs": [], "counts": {}, "mismatches": {},
    }
    styles = args.style or list(ROUND2_STYLES)
    for style in styles:
        entry = {"gpt": {}, "gemini": {}, "request_diffs": {}, "production_style_entry": None}
        for slot in STYLE_SLOTS:
            for arm in STYLE_ARMS:
                key = "%s-%s" % (slot, arm)
                entry["gpt"][key] = gpt_arm_facts(round_dir, style, slot, arm)
                entry["gemini"][key] = gemini_arm_facts(round_dir, style, slot, arm)
            # 每个 pair 的请求差异（只比输入字段）
            entry["request_diffs"][slot] = {
                "gpt": gpt_request_diff(round_dir, style, slot),
                "gemini": gemini_request_diff(round_dir, style, slot),
            }
        result["styles"][style] = entry
        # 冻结该画风实际用到的输入
        for slot in STYLE_SLOTS:
            for arm in STYLE_ARMS:
                facts = entry["gpt"][("%s-%s" % (slot, arm))]
                for label, path in (("source", facts["source"]), ("analysis_json", facts["analysis_json"]),
                                    ("style_ref", facts["style_ref"]), ("base_image", facts["base_image"])):
                    if path and Path(path).is_file():
                        result["frozen_inputs"].append({
                            "style": style, "slot": slot, "arm": arm, "role": label,
                            "path": path, "sha256": sha256_of(path),
                            "size": Path(path).stat().st_size})
    # 汇总口径
    gpt_arms = [(s, slot, arm) for s in styles for slot in STYLE_SLOTS for arm in STYLE_ARMS]
    rows = [result["styles"][s]["gpt"]["%s-%s" % (slot, arm)] for s, slot, arm in gpt_arms]
    result["counts"] = {
        "gpt_arms": len(rows),
        "recorded_complete": sum(1 for r in rows if r["recorded_status"] == "complete"),
        "recorded_review_required": sum(1 for r in rows if r["recorded_status"] == "review_required"),
        "quality_audit_error_arms": sum(1 for r in rows if r["quality_audit_error"]),
        "binding_not_proven_arms": sum(1 for r in rows if r["binding"]["verdict"] != "bound"),
        "binding_stale_arms": sum(1 for r in rows
                                  if "stale" in set(r["binding"]["checks"].values())),
        "binding_missing_arms": sum(1 for r in rows
                                    if "missing" in set(r["binding"]["checks"].values())),
        "final_review_audit_failed_arms": sum(
            1 for r in rows if (r["binding"]["checks"].get("final_review") == "audit_failed")),
        "complete_but_not_proven": sum(
            1 for r in rows if r["recorded_status"] == "complete" and r["binding"]["verdict"] != "bound"),
        "pairs_both_recorded_complete": sum(
            1 for s in styles for slot in STYLE_SLOTS
            if all(result["styles"][s]["gpt"]["%s-%s" % (slot, a)]["recorded_status"] == "complete"
                   for a in STYLE_ARMS)),
    }
    result["mismatches"] = {
        "gpt_arms_stale_or_missing_audit": [
            {"style": r["style"], "slot": r["slot"], "arm": r["arm"],
             "selected": Path(r["selected_output"]).name,
             "checks": r["binding"]["checks"], "reasons": r["binding"]["reasons"]}
            for r in rows if r["binding"]["verdict"] != "bound"],
        "gemini_non_intervention_pairs": [
            {"style": s, "slot": slot,
             "changed_keys": result["styles"][s]["request_diffs"][slot]["gemini"]["changed_keys"],
             "has_intervention": result["styles"][s]["request_diffs"][slot]["gemini"]["has_intervention"]}
            for s in styles for slot in STYLE_SLOTS],
    }
    write_json(out / "authoritative-binding.json", result)
    (out / "frozen" / "README.md").write_text(
        "# 冻结的上一轮证据\n\n"
        "本目录只记录路径与 SHA-256，不复制、不修改上一轮文件。"
        "完整清单见 `../authoritative-binding.json` 的 `frozen_inputs`。\n", encoding="utf-8")
    print("权威绑定: %s" % (out / "authoritative-binding.json"))
    print("  GPT 臂 %d：recorded complete %d / review_required %d / 质量门禁报错 %d" % (
        result["counts"]["gpt_arms"], result["counts"]["recorded_complete"],
        result["counts"]["recorded_review_required"], result["counts"]["quality_audit_error_arms"]))
    print("  两臂都 recorded complete 的 pair: %d" % result["counts"]["pairs_both_recorded_complete"])
    print("  审计与 selected_output 绑定不成立的 arm: %d（stale %d / missing %d / 终审审计报错 %d）" % (
        result["counts"]["binding_not_proven_arms"], result["counts"]["binding_stale_arms"],
        result["counts"]["binding_missing_arms"], result["counts"]["final_review_audit_failed_arms"]))
    print("  其中 recorded complete 却无法证明最终图合格的 arm: %d" % result["counts"]["complete_but_not_proven"])
    for item in result["mismatches"]["gpt_arms_stale_or_missing_audit"]:
        print("    - %s/%s/%s %s" % (item["style"], item["slot"], item["arm"], item["checks"]))
    return 0


def _request_input_view(payload):
    """只保留影响生成的输入字段，丢掉输出路径与时间戳。"""
    if not isinstance(payload, dict):
        return {}
    view = {}
    for key, value in payload.items():
        if key in IGNORED_REQUEST_KEYS:
            continue
        view[key] = value
    return view


def _first_diff(a, b, path=""):
    """递归比对，返回 [(字段路径, 控制臂值, 变体臂值)]。"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            out.extend(_first_diff(a.get(key), b.get(key), f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + ".length", len(a), len(b)))
        for index, (left, right) in enumerate(zip(a, b)):
            out.extend(_first_diff(left, right, f"{path}[{index}]"))
        return out
    if a != b:
        out.append((path, a, b))
    return out


def gpt_request_diff(round_dir, style, slot):
    control = record_of(Path(round_dir) / "results" / style / slot / "control-gpt" / "pipeline" / "request.json")
    variant = record_of(Path(round_dir) / "results" / style / slot / "variant-gpt" / "pipeline" / "request.json")
    diffs = _first_diff(_request_input_view(control.get("request") or {}),
                        _request_input_view(variant.get("request") or {}))
    steps_diffs = _first_diff(control.get("steps") or {}, variant.get("steps") or {})
    base_diff = (str(control.get("base") or "") != str(variant.get("base") or ""))
    return {
        "view": "pipeline/request.json 的 request 输入字段 + steps（忽略输出路径与时间戳）",
        "changed_fields": [d[0] for d in diffs] + ["steps." + d[0] for d in steps_diffs],
        "sample": [{"field": d[0], "control": str(d[1])[:220], "variant": str(d[2])[:220]} for d in diffs[:6]],
        "base_image_differs": base_diff,
        "has_intervention": bool(diffs or steps_diffs),
        "note": "受控实验里 base 必须相同；不同则不是同一输入的对照组" if base_diff else "",
    }


def gemini_request_diff(round_dir, style, slot):
    control = record_of(Path(round_dir) / "results" / style / slot / "control-gemini" / "first" / "gemini-request-1.json")
    variant = record_of(Path(round_dir) / "results" / style / slot / "variant-gemini" / "first" / "gemini-request-1.json")
    keys = ("prompt", "instructions", "post_instructions", "model", "aspect_ratio", "image_paths")
    changed = [k for k in keys if control.get(k) != variant.get(k)]
    sample = []
    for key in changed:
        left, right = str(control.get(key) or ""), str(variant.get(key) or "")
        if key == "image_paths":
            sample.append({"field": key, "control": left, "variant": right})
            continue
        index = 0
        while index < min(len(left), len(right)) and left[index] == right[index]:
            index += 1
        sample.append({"field": key, "first_diff_at": index,
                       "control": left[max(0, index - 60):index + 120],
                       "variant": right[max(0, index - 60):index + 120]})
    return {"view": "真实 Gemini 生图线程构造参数（prompt/instructions/post_instructions/图片顺序）",
            "changed_keys": changed, "sample": sample, "has_intervention": bool(changed),
            "control_snapshot_sha256": sha256_of(
                Path(round_dir) / "results" / style / slot / "control-gemini" / "first" / "gemini-request-1.json"),
            "variant_snapshot_sha256": sha256_of(
                Path(round_dir) / "results" / style / slot / "variant-gemini" / "first" / "gemini-request-1.json")}


def _arm_command(run, entry, repeat):
    """把这个 arm 的实参翻译成 `tools/analysis_gpt_run.py` 的命令行。

    受控实验只做一次整图 Gemini source 重绘：不加 tone/ink/post_adjustment/face_hair，
    自动修图全部关闭；`--audit-only` 让质量/身份/人体/终审照做但不产生任何新图。
    """
    steps_dir = Path(run["output_dir"]) / ("repeat-%d" % repeat)
    steps_dir.mkdir(parents=True, exist_ok=True)
    # 受控实验的提示词必须以 run-plan 里冻结的那一份为准（P2 的变体就是靠这一份生效）。
    # 只靠画风条目/固件拼装会拿到没改过的默认文字，变体就白跑了。
    # 路径带 arm 名：同 slot 的两臂不能互相覆盖快照。
    prompt_file = Path(run["output_dir"]) / ("repaint-prompt-%s.txt" % run["arm"])
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(str(run.get("repaint_prompt_preview") or ""), encoding="utf-8")
    run["prompt_file"] = str(prompt_file)
    cmd = [sys.executable, str(ROOT / "tools" / "analysis_gpt_run.py"),
           "--json", run["analysis_path"],
           "--style", run["style"],
           "--quality", "high",
           "--size", "auto",
           "--steps", "repaint",
           "--repaint-scope", "full",
           "--repaint-ref", run["repaint_ref"],
           "--firmware", str(ROOT / run["firmware"]),
           "--base-image", run["base_path"],
           "--identity-audit",
           "--audit-only",
           "--final-review-audit",
           "--output-dir", str(steps_dir)]
    cmd += ["--prompt-file", str(prompt_file)]
    if run.get("clauses_file"):
        cmd += ["--repaint-clauses-file", run["clauses_file"]]
    # P1 的唯一变量是「参考图是否送入生成请求」：文字段落两边逐字相同，
    # 所以不送图的那一臂要显式打开 text_without_image，否则它会连参考图条款一起丢掉，
    # 差异就不再是单一变量。
    if not run["style_reference_sent"] and run.get("style_ref_source"):
        cmd.append("--repaint-text-without-image")
    if run.get("clauses_only"):
        cmd.append("--repaint-clauses-only")
    if run["analysis_sha256"]:
        pass
    if run["style_ref_path"]:
        cmd += ["--repaint-style-ref", run["style_ref_path"]]
    if run.get("prompt_file"):
        cmd += ["--prompt-file", run["prompt_file"]]
    return cmd


def round_run(args):
    """执行一个 arm 的一次重复；声明与实际不一致时拒绝执行（不付费）。"""
    round_dir = Path(args.round).resolve()
    plan = record_of(args.plan)
    runs = plan.get("runs") or []
    match = [r for r in runs if r["experiment_id"] == args.experiment and r["slot"] == args.slot
             and r["arm"] == args.arm]
    if not match:
        raise SystemExit("run-plan.json 里没有 %s/%s/%s" % (args.experiment, args.slot, args.arm))
    run = match[0]
    # 声明检查：变体臂必须声明它改的是什么，否则不允许生图
    if not run.get("expected_diff"):
        raise SystemExit("拒绝执行：%s 没有声明 expected_diff，不能按变体生图" % run["run_id"])
    if run["analysis_path"] and not Path(run["analysis_path"]).is_file():
        raise SystemExit("固定分析 JSON 缺失：" + run["analysis_path"])
    total = plan.get("runs_total") or len(runs)
    if total > int(plan.get("max_experiment_image_calls") or 16):
        raise SystemExit("计划调用数 %d 超过本轮上限" % total)
    repeat = int(args.repeat)
    out_dir = Path(run["output_dir"]) / ("repeat-%d" % repeat)
    # 每次重复都用同一份冻结输入：不把上一次的输出当成下一次的输入
    base_sha = sha256_of(run["base_path"])
    if base_sha != run["base_sha256"]:
        raise SystemExit("冻结首图被改动：%s" % run["base_path"])
    arm_input = _analysis_input(round_dir, plan, run)
    if not arm_input:
        raise SystemExit("找不到 %s 固定分析 JSON 的源（round2 的 analysis_json）" % run["style"])
    run["analysis_path"] = arm_input
    run["analysis_sha256"] = sha256_of(arm_input)
    cmd = _arm_command(run, None, repeat)
    record = {"run_id": run["run_id"], "experiment_id": run["experiment_id"], "style": run["style"],
              "slot": run["slot"], "arm": run["arm"], "repeat": repeat, "stage": run["stage"],
              "base_path": run["base_path"], "base_sha256": run["base_sha256"],
              "analysis_path": run["analysis_path"], "analysis_sha256": run["analysis_sha256"],
              "style_snapshot_sha256": plan.get("styles_snapshot_sha256"),
              "style_reference_sent": run["style_reference_sent"],
              "style_ref_path": run["style_ref_path"], "repaint_ref": run["repaint_ref"],
              "expected_diff": run["expected_diff"],
              "planned_prompt_sha256": run["repaint_prompt_sha256"],
              "command": " ".join(cmd), "started_at": now_stamp(), "dry_run": bool(args.dry_run)}
    log_path = out_dir / "run.log"
    if args.dry_run:
        record["finished_at"] = now_stamp()
        write_json(out_dir / "run-record.json", record)
        print("[dry-run] " + record["command"])
        return 0
    started = datetime.datetime.now()
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL)
    record["returncode"] = proc.returncode
    record["seconds"] = round((datetime.datetime.now() - started).total_seconds(), 1)
    record["finished_at"] = now_stamp()
    record["log"] = str(log_path)
    manifest_path = out_dir / "request.json"
    manifest = record_of(manifest_path)
    record["manifest_status"] = manifest.get("status")
    record["selected_output"] = str(manifest.get("selected_output") or "")
    record["output_sha256"] = sha256_of(record["selected_output"])
    record["selected_output_exists"] = bool(record["output_sha256"])
    gate = manifest.get("final_gate") or {}
    record["final_gate"] = gate
    record["audit_image_sha256"] = {
        "identity": str(((manifest.get("identity") or {}).get("final") or {}).get("candidate_sha256") or ""),
        "anatomy": str((manifest.get("anatomy_review") or {}).get("candidate_sha256") or ""),
        "final_review": str((manifest.get("final_review") or {}).get("candidate_sha256") or ""),
    }
    record["actual_diff"] = _actual_call_diff(out_dir, run)
    record["call_ids"] = [row.get("operation") for row in record["actual_diff"].get("calls") or []]
    record["cost_status"] = "no_billing_export; estimate only (utils/cost_estimate.py)"
    write_json(out_dir / "run-record.json", record)
    print("%s repeat-%d rc=%s status=%s gate=%s" % (
        run["run_id"], repeat, proc.returncode, record["manifest_status"], gate.get("status")))
    return 0 if proc.returncode == 0 else 1


def _analysis_input(round_dir, plan, run):
    """固定分析 JSON：从上一轮同 style/slot 的 pipeline/request.json 读 analysis_json。"""
    candidates = []
    binding_path = round_dir / "round2-corrected" / "authoritative-binding.json"
    if binding_path.is_file():
        for row in _binding_rows(record_of(binding_path)):
            if row["style"] == run["style"] and row["slot"] == run["slot"]:
                candidates.append(row["gpt_analysis"])
    candidates.append(str(round_dir.parent / "eye-face-hair-round2-20260930" / "results" /
                          run["style"] / run["slot"] / "control-gpt" / "pipeline" / "request.json"))
    for path in candidates:
        if path and Path(path).is_file() and path.endswith(".json") and "request.json" not in path:
            return path
    for path in candidates:
        data = record_of(path)
        value = str(data.get("analysis_json") or "")
        if value and Path(value).is_file():
            return value
    return ""


def _actual_call_diff(out_dir, run):
    """把实际调用记录摊开：真正送出的图片、提示词哈希、以及它们与声明的差异。"""
    calls = []
    call_file = Path(out_dir) / "api-calls.jsonl"
    if call_file.is_file():
        for line in call_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    calls.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    result = {"calls": calls, "declared": run["expected_diff"], "notes": []}
    if not calls:
        result["notes"].append("没有 api-calls.jsonl：本次没有实际发出重绘请求（失败或未开始）")
        return result
    last = calls[-1]
    actual_refs = [item.get("path") for item in (last.get("reference_images") or [])]
    result.update({"actual_reference_images": actual_refs,
                   "actual_prompt_sha256": last.get("prompt_sha256"),
                   "actual_prompt_chars": last.get("prompt_chars"),
                   "actual_style_reference_sent": len(actual_refs) > 1,
                   "new_files": last.get("new_files") or []})
    if run["style_reference_sent"] and len(actual_refs) < 2:
        result["notes"].append("声明送参考图，但实际只送了源图")
    if (not run["style_reference_sent"]) and len(actual_refs) > 1:
        result["notes"].append("声明不送参考图，但实际送了 %d 张" % len(actual_refs))
    if last.get("prompt_sha256") != run.get("planned_prompt_sha256"):
        result["notes"].append("实际提示词与计划快照不一致（计划 %s / 实际 %s）" % (
            str(run.get("repaint_prompt_sha256"))[:10], str(last.get("prompt_sha256"))[:10]))
    return result


def round_rebind(args):
    """把一次已经付费产出的图重新绑定审计，不重新生图。

    用途：某次运行因为工具缺陷（而不是生成失败）拿不到有效绑定。此时图本身是
    合法产物，重新调用图片接口既浪费预算又改变样本，正确做法是对同一张图重跑
    只读审计，并把两次记录都留下（原记录移进 `pre-rebind/`）。
    """
    import shutil
    round_dir = Path(args.round).resolve()
    src = Path(args.run_dir).resolve()
    manifest_path = src / "request.json"
    if not manifest_path.is_file():
        raise SystemExit("没有 request.json：" + str(src))
    manifest = record_of(manifest_path)
    selected = str(manifest.get("selected_output") or "")
    if not selected or not Path(selected).is_file():
        raise SystemExit("该次运行没有可用的 selected_output")
    archive = src / "pre-rebind"
    archive.mkdir(exist_ok=True)
    for name in ("identity-final.json", "anatomy-audit.json", "final-quality-audit-0.json",
                 "final-quality-audit.json", "final-gate.json", "request.json", "api-calls.jsonl"):
        path = src / name
        if path.is_file() and not (archive / name).exists():
            shutil.copy2(path, archive / name)
    base_path = str(manifest.get("base") or "")
    style_ref = str(manifest.get("repaint_style_ref") or "")
    if not style_ref or not Path(style_ref).is_file():
        style_ref = base_path
    result = manifest.get("analysis_json") and record_of(str(manifest.get("analysis_json"))) or {}
    analysis = record_of(str(manifest.get("analysis_json") or ""))
    prompt = str((manifest.get("request") or {}).get("prompt") or "")
    from utils.refine_quality import (audit_hand_quality, audit_refine_quality, file_sha256,
                                      review_final_candidate, should_refine_quality)
    from utils.identity_audit import audit_image_identity, build_identity_correction_prompt
    identity = audit_image_identity(selected, analysis or result, expected_prompt=prompt)
    identity["correction_prompt"] = build_identity_correction_prompt(identity, 1, 2)
    identity["candidate_sha256"] = file_sha256(selected)
    identity["rebound_by"] = "tools/style_face_eval.py round-rebind（对同一张图重跑只读审计）"
    write_json(src / "identity-final.json", identity)
    anatomy = audit_hand_quality(selected)
    anatomy["rebound_by"] = identity["rebound_by"]
    write_json(src / "anatomy-audit.json", anatomy)
    selected, final_quality = review_final_candidate(
        selected, lambda candidate: audit_refine_quality(
            base_path, candidate, style_ref, first_pass_prompt=prompt, final_review=True),
        src, log_callback=print, audit_only=True)
    final_quality["rebound_by"] = identity["rebound_by"]
    write_json(src / "final-quality-audit.json", final_quality)
    manifest["identity"] = manifest.get("identity") or {}
    manifest["identity"]["final"] = identity
    manifest["identity"]["action"] = "accept" if not identity.get("mismatch") else "review"
    manifest["anatomy_review"] = anatomy
    manifest["final_review"] = final_quality
    manifest["selected_output"] = selected
    if should_refine_quality(final_quality) or final_quality.get("needs_review") or identity.get("mismatch"):
        manifest["status"] = "review_required"
    from utils.gate_status import evaluate_final_gate
    gate = evaluate_final_gate(final_image=selected, identity=identity, anatomy=anatomy,
                               quality=manifest.get("quality_refine") or {}, final_review=final_quality,
                               identity_action=manifest["identity"].get("action"),
                               quality_audit_error=str((manifest.get("quality_refine") or {}).get(
                                   "audit_error") or ""))
    manifest["final_gate"] = gate
    if gate["status"] != "complete":
        manifest["status"] = "review_required"
    else:
        manifest["status"] = "complete"
    write_json(src / "final-gate.json", gate)
    write_json(manifest_path, manifest)
    record_path = src / "run-record.json"
    record = record_of(record_path)
    if record:
        record["manifest_status"] = manifest["status"]
        record["output_sha256"] = file_sha256(selected)
        record["selected_output"] = selected
        record["final_gate"] = gate
        record["audit_image_sha256"] = {
            "identity": identity.get("candidate_sha256", ""),
            "anatomy": anatomy.get("candidate_sha256", ""),
            "final_review": final_quality.get("candidate_sha256", "")}
        record["rebind"] = {"at": now_stamp(), "archived_to": str(archive),
                            "note": "本次没有重新生图；对同一张图重跑了身份/人体/终审只读审计"}
        write_json(record_path, record)
    print("rebind 完成: %s" % src)
    print("  最终图 %s" % Path(selected).name)
    print("  门禁 %s | %s" % (gate["status"], gate["text"][:200]))
    return 0


def _actual_prompt(repeat_dir: Path) -> tuple:
    """取这次运行真正发出去的提示词：优先 api-calls.jsonl，其次 run.log 里的请求体。"""
    call_file = repeat_dir / "api-calls.jsonl"
    if call_file.is_file():
        for line in call_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("prompt"):
                    return str(entry["prompt"]), "api-calls.jsonl"
    log_file = repeat_dir / "run.log"
    if log_file.is_file():
        text = log_file.read_text(encoding="utf-8", errors="replace")
        marker = '"text": "'
        start = text.find("请求数据:")
        if start < 0:
            start = 0
        index = text.find(marker, start)
        if index >= 0:
            cursor = index + len(marker)
            chunks = []
            while cursor < len(text):
                char = text[cursor]
                if char == "\\" and cursor + 1 < len(text):
                    chunks.append(text[cursor:cursor + 2])
                    cursor += 2
                    continue
                if char == '"':
                    break
                chunks.append(char)
                cursor += 1
            try:
                return json.loads('"' + "".join(chunks) + '"'), "run.log 请求体"
            except json.JSONDecodeError:
                return "".join(chunks), "run.log 请求体(未转义)"
    return "", ""


def _round_planned_runs(plan):
    return [r for r in (plan.get("runs") or []) if r.get("run_id")]


def _round_pair_entries(plan, style, slot):
    """按计划声明顺序取这一对的臂（顺序即声明顺序，不靠字母序）。"""
    entries = [r for r in _round_planned_runs(plan)
               if r.get("style") == style and r.get("slot") == slot]
    order, first = [], {}
    for entry in entries:
        arm = str(entry.get("arm"))
        if arm not in first:
            first[arm] = entry
            order.append(arm)
    return order, first


def _arm_clauses(entry) -> list:
    """取这一臂的条款：计划里直接写了就用，否则读它声明的 `clauses_file`。"""
    clauses = [str(c) for c in (entry.get("clauses") or [])]
    if clauses:
        return clauses
    path = str(entry.get("clauses_file") or "")
    if path and Path(path).is_file():
        try:
            value = read_json(path)
        except (OSError, ValueError):
            return []
        if isinstance(value, list):
            return [str(c) for c in value]
    return []


def _round_pair_declaration(entry_a, entry_b):
    """从计划推导这一对**允许的**差异（只看声明字段，不看计划里的整段预览）。

    第四轮 P0.3：第三轮 P1 之所以被误判为「全声明成立」，是因为旧检查只比图片数量。
    这里改成把「声明」表达成一个可执行的变换，再要求实际提示词满足该变换：

    * ``single_sentence_clause``：两臂条款列表只有第 i 条不同（old → new）；实际提示词必须
      恰好命中 old 一次，且 ``A.replace(old, new) == B``；
    * ``phrasing_only``：声明差异只是「参考图是否送入」，文字只允许做 `Image N` →
      具名指代的替换（`utils.post_process.text_without_image_phrasing`）；
    * ``identical``：声明两臂请求完全相同；
    * ``undeclared``：计划没有声明 → cannot_compare（不允许按「变体已生效」解读）。
    """
    clauses_a = _arm_clauses(entry_a)
    clauses_b = _arm_clauses(entry_b)
    if clauses_a and len(clauses_a) == len(clauses_b):
        diff = [index for index in range(len(clauses_a)) if clauses_a[index] != clauses_b[index]]
        if len(diff) == 1:
            index = diff[0]
            return {"mode": "single_sentence_clause", "changed_index": index,
                    "old": clauses_a[index], "new": clauses_b[index],
                    "note": "计划声明两臂只有第 %d 条条款不同（唯一 eye 句）" % (index + 1)}
    sent_a = bool(entry_a.get("style_reference_sent"))
    sent_b = bool(entry_b.get("style_reference_sent"))
    if sent_a != sent_b and clauses_a == clauses_b:
        with_ref = entry_a if sent_a else entry_b
        without_ref = entry_b if sent_a else entry_a
        ref = str(with_ref.get("style_ref_path") or with_ref.get("style_ref_source") or "")
        return {"mode": "phrasing_only", "reference_name": os.path.basename(ref),
                "with_ref_arm": str(with_ref.get("arm")),
                "without_ref_arm": str(without_ref.get("arm")),
                "note": "计划声明唯一变量是参考图是否送入；文字只允许做具名指代替换"}
    if clauses_a == clauses_b and sent_a == sent_b:
        return {"mode": "identical", "note": "计划声明两臂请求完全相同"}
    return {"mode": "undeclared", "note": "计划没有声明这一对允许的差异"}


def _round_line_summary(left: str, right: str) -> dict:
    """可读的文字差异摘要：逐行比对，列出首个差异与增删行数。"""
    left_lines, right_lines = left.split("\n"), right.split("\n")
    changed = [(index, line_a, line_b)
               for index, (line_a, line_b) in enumerate(zip(left_lines, right_lines))
               if line_a != line_b]
    return {"left_lines": len(left_lines), "right_lines": len(right_lines),
            "changed_line_count": len(changed) + abs(len(left_lines) - len(right_lines)),
            "first_change": ({"line": changed[0][0] + 1, "left": changed[0][1][:200],
                              "right": changed[0][2][:200]} if changed else None),
            "sample_changes": [{"line": index + 1, "left": line_a[:160], "right": line_b[:160]}
                               for index, line_a, line_b in changed[:6]]}


def _round_compare_pair(declaration, arm_a, arm_b, prompt_a, prompt_b, refs_a, refs_b):
    """按声明比对两臂真实提示词；返回 (problems, evidence)。"""
    problems, evidence = [], []
    if not prompt_a or not prompt_b:
        problems.append("缺少可用提示词（没有 api-calls.jsonl，或日志被截断 / 未转义）")
        return problems, evidence
    mode = declaration.get("mode")
    summary = _round_line_summary(prompt_a, prompt_b)
    evidence.append({"kind": "text_summary", "arms": [arm_a, arm_b],
                     "chars": [len(prompt_a), len(prompt_b)], "line_diff": summary,
                     "identical": prompt_a == prompt_b})
    if mode == "single_sentence_clause":
        old = str(declaration.get("old") or "")
        new = str(declaration.get("new") or "")
        hits = prompt_a.count(old)
        if hits != 1:
            problems.append("%s 臂里声明的 eye 句命中 %d 次（要求恰好 1 次）：非声明行增删" % (arm_a, hits))
        elif prompt_a.replace(old, new) != prompt_b:
            problems.append("按声明替换唯一 eye 句后两臂仍不相同：存在非声明差异")
        evidence.append({"kind": "single_sentence", "arm_a": arm_a, "arm_b": arm_b,
                         "old_chars": len(old), "new_chars": len(new), "old_hits_in_a": hits})
    elif mode == "phrasing_only":
        from utils.post_process import text_without_image_phrasing
        sent_a, sent_b = bool(refs_a.get("count", 0) > 1), bool(refs_b.get("count", 0) > 1)
        name = str(declaration.get("reference_name") or "")
        if sent_a and not sent_b:
            expected = text_without_image_phrasing(prompt_a, name)
            compared = (arm_a, arm_b, expected, prompt_b)
        elif sent_b and not sent_a:
            expected = text_without_image_phrasing(prompt_b, name)
            compared = (arm_b, arm_a, expected, prompt_a)
        else:
            problems.append("两臂参考图数量相同，声明说唯一变量是「是否送参考图」")
            compared = None
        if compared:
            with_arm, without_arm, expected, actual = compared
            if expected != actual:
                problems.append("两臂文字差异不止「图片指代替换」：%s 臂相对 %s 臂少/多了整段文字"
                                % (without_arm, with_arm))
            evidence.append({"kind": "phrasing_only", "with_ref_arm": with_arm,
                             "without_ref_arm": without_arm,
                             "expected_chars": len(expected), "actual_chars": len(actual)})
    elif mode == "identical":
        if prompt_a != prompt_b:
            problems.append("计划声明两臂请求完全相同，实际提示词不同")
    else:
        problems.append("计划没有声明这一对允许的差异，不能按「变体已生效」解读")
    return problems, evidence


def round_diffcheck(args):
    """对账：按**声明**逐一比对每一次真实请求（第四轮 P0.3 重写）。

    旧版只检查参考图数量、且只比较每臂的第一次提示词，于是第三轮 P1 的 2,795 字符
    整段增删仍被标成 `all_declared_differences_confirmed`。现在：

    * 每个计划 run 必须有可用的真实提示词，否则 cannot_compare（缺请求 / 缺臂 / 截断日志）；
    * 同一臂的重复必须逐字相同，且同名文件的**内容哈希**必须一致（同名不同内容 → cannot_compare）；
    * 送到生成接口的参考图顺序与内容哈希必须与声明一致（顺序差 → cannot_compare）；
    * 两臂差异必须**恰好等于**已声明的变换（非声明行增删 → cannot_compare）；
    * 只有全部通过才写 `all_declared_differences_confirmed`，`--strict` 时非零退出。
    """
    round_dir = Path(args.round).resolve()
    plan = record_of(args.plan)
    groups, order = {}, []
    for run in _round_planned_runs(plan):
        key = (str(run.get("experiment_id")), str(run.get("style")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(run)
    report = {"generated_at": now_stamp(), "round": str(round_dir), "checks": [],
              "pairs": [], "problems": [], "cannot_compare": [], "summary": {}}
    verdicts = []

    def _record(problem, entry):
        report["problems"].append(problem)
        report["cannot_compare"].append(entry)

    for (exp_id, style) in order:
        entries = groups[(exp_id, style)]
        slots = []
        for entry in entries:
            if str(entry.get("slot")) not in slots:
                slots.append(str(entry.get("slot")))
        for slot in slots:
            arm_order, first = _round_pair_entries(plan, style, slot)
            per_arm = {}
            prompts = {}
            for arm in arm_order:
                runs = [r for r in entries if r.get("arm") == arm and r.get("slot") == slot]
                calls, prompt_rows, problems = [], [], []
                for run in runs:
                    output_dir = Path(str(run.get("output_dir") or ""))
                    repeat_dirs = sorted(output_dir.glob("repeat-*")) if output_dir.is_dir() else []
                    if not repeat_dirs:
                        repeat_dirs = [output_dir]
                    for repeat_dir in repeat_dirs:
                        prompt, source = _actual_prompt(repeat_dir)
                        if not prompt:
                            problems.append("缺少真实请求记录：%s" % repeat_dir)
                            prompt_rows.append({"repeat": repeat_dir.name, "prompt": "", "source": ""})
                            continue
                        if source != "api-calls.jsonl" or "未转义" in source:
                            problems.append("提示词来自日志（%s），可能被截断：%s" % (source, repeat_dir))
                        prompt_rows.append({"repeat": repeat_dir.name, "prompt": prompt, "source": source})
                        calls.extend([{"repeat": repeat_dir.name, **row}
                                      for row in _round4_read_jsonl(repeat_dir / "api-calls.jsonl")])
                per_arm[arm] = {"calls": calls, "rows": prompt_rows, "problems": problems}
                prompts[arm] = prompt_rows
            pair = {"experiment": exp_id, "style": style, "slot": slot, "arms": arm_order,
                    "checks": [], "problems": []}
            for arm in arm_order:
                rows = per_arm[arm]["rows"]
                pair["problems"].extend(per_arm[arm]["problems"])
                if not rows:
                    pair["problems"].append("%s 臂没有任何请求记录" % arm)
                    continue
                shas = {hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
                        for row in rows if row["prompt"]}
                if len(shas) > 1:
                    pair["problems"].append("%s 臂的重复之间提示词不一致（同名不同内容）" % arm)
                names = {row["repeat"] for row in rows}
                if len(names) != len(rows):
                    pair["problems"].append("%s 臂出现重复编号被多次执行" % arm)
                # 实际送出的参考图：顺序与内容哈希
                for call in per_arm[arm]["calls"]:
                    refs = call.get("reference_images") or []
                    plan_entry = first.get(arm) or {}
                    declared_sent = bool(plan_entry.get("style_reference_sent"))
                    actual_sent = len(refs) > 1
                    check = {"kind": "reference_images", "arm": arm, "repeat": call["repeat"],
                             "declared_sent": declared_sent, "actual_count": len(refs),
                             "paths": [item.get("path") for item in refs],
                             "sha256": [item.get("sha256") for item in refs],
                             "match": declared_sent == actual_sent}
                    pair["checks"].append(check)
                    if not check["match"]:
                        pair["problems"].append(
                            "%s/%s：声明送参考图=%s，实际送出 %d 张"
                            % (arm, call["repeat"], declared_sent, len(refs)))
                    base_sha = str(plan_entry.get("base_sha256") or "")
                    if base_sha and refs and str(refs[0].get("sha256") or "") != base_sha:
                        pair["problems"].append(
                            "%s/%s：首张参考图不是计划的冻结 base（顺序差或内容不同）"
                            % (arm, call["repeat"]))
            if len(arm_order) < 2:
                pair["problems"].append("这一对缺少另一条臂")
            else:
                arm_a, arm_b = arm_order[0], arm_order[1]
                declaration = _round_pair_declaration(first.get(arm_a) or {}, first.get(arm_b) or {})
                pair["declaration"] = declaration
                pair["declared_clauses"] = {"arm_a": [c[:120] for c in _arm_clauses(first.get(arm_a) or {})],
                                            "arm_b": [c[:120] for c in _arm_clauses(first.get(arm_b) or {})]}
                rows_a = {row["repeat"]: row for row in per_arm.get(arm_a, {}).get("rows", [])}
                rows_b = {row["repeat"]: row for row in per_arm.get(arm_b, {}).get("rows", [])}
                if set(rows_a) != set(rows_b):
                    pair["problems"].append("两臂的重复编号不对齐：%s vs %s"
                                            % (sorted(rows_a), sorted(rows_b)))
                calls_a = per_arm.get(arm_a, {}).get("calls", [])
                calls_b = per_arm.get(arm_b, {}).get("calls", [])
                refs_a = {"count": len((calls_a[0].get("reference_images") if calls_a else []) or [])}
                refs_b = {"count": len((calls_b[0].get("reference_images") if calls_b else []) or [])}
                for repeat in sorted(set(rows_a) & set(rows_b)):
                    problems, evidence = _round_compare_pair(
                        declaration, arm_a, arm_b, rows_a[repeat]["prompt"], rows_b[repeat]["prompt"],
                        refs_a, refs_b)
                    pair["checks"].append({"kind": "pair_text", "repeat": repeat,
                                           "declaration_mode": declaration.get("mode"),
                                           "problems": problems, "evidence": evidence})
                    pair["problems"].extend("%s：%s" % (repeat, text) for text in problems)
            pair["verdict"] = "can_compare" if not pair["problems"] else "cannot_compare"
            verdicts.append(pair["verdict"])
            for problem in pair["problems"]:
                _record("%s/%s/%s: %s" % (exp_id, style, slot, problem),
                        {"experiment": exp_id, "style": style, "slot": slot, "reason": problem})
            report["pairs"].append(pair)
    confirmed = bool(verdicts) and all(item == "can_compare" for item in verdicts)
    report["summary"] = {
        "pairs": len(verdicts),
        "pairs_can_compare": sum(1 for item in verdicts if item == "can_compare"),
        "pairs_cannot_compare": sum(1 for item in verdicts if item == "cannot_compare"),
        "problems": len(report["problems"]),
        "verdict": "all_declared_differences_confirmed" if confirmed
                   else "entry_must_be_fixed_before_trusting_variant",
    }
    out = Path(args.out).resolve() if args.out else round_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "request-diff-report.json", report)
    lines = ["# 实际请求差异对账（按声明逐一比对）", "",
             "逐个 arm、逐个重复从 `api-calls.jsonl` 读真实送出的提示词与参考图；",
             "先把计划里的「允许差异」表达成可执行的变换，再要求实际请求满足它。", ""]
    for pair in report["pairs"]:
        declaration = pair.get("declaration") or {}
        lines.append("- `%s/%s/%s`：%s（声明模式 `%s`）" % (
            pair["experiment"], pair["style"], pair["slot"], pair["verdict"],
            declaration.get("mode", "n/a")))
        if declaration.get("note"):
            lines.append("    - 声明：%s" % declaration["note"])
        for problem in pair["problems"]:
            lines.append("    - **%s**" % problem)
    lines += ["", "结论：`%s`" % report["summary"]["verdict"], "",
              "注意：这里只回答「实际请求是否符合声明」，不回答「变体是否有效」。"]
    (out / "request-diff-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("请求对账: %s（%d 对，%d 对不可比，%d 条问题）" % (
        out / "request-diff-report.json", len(verdicts), report["summary"]["pairs_cannot_compare"],
        len(report["problems"])))
    for text in report["problems"]:
        print("  ! " + text)
    if not verdicts:
        return 1 if args.strict else 0
    return 0 if confirmed or not args.strict else 1


def round_verdict(args):
    """按规程的成功条件逐条判定，并列出每条结论的证据文件。

    成功条件（PROTOCOL.md §P1/§P2）：
      P2：每个 slot 两次重复中至少一次目标 eye 分比对应 control 提升 ≥1，且两次平均不下降；
          所有 variant 的源身份/瞳色/睁闭眼、肢体、构图与最终门禁通过。
      P1：记录复制参考角色（发色/瞳色/发型/头饰/衣装/姿势/道具/背景）的次数与源身份通过数。
    这里只做机械汇总：分数来自 `scores.json`（人工/看图填写），身份与门禁来自 manifest。
    """
    round_dir = Path(args.round).resolve()
    plan = record_of(args.plan)
    scores = record_of(args.scores) if args.scores else {}
    reviews = scores.get("runs") or {}
    report = {"generated_at": now_stamp(), "round": str(round_dir),
              "score_source": str(Path(args.scores).resolve()) if args.scores else "",
              "experiments": {}}
    plan_runs = plan.get("runs") or []
    exp_ids = []
    for run in plan_runs:
        if run.get("experiment_id") not in exp_ids:
            exp_ids.append(run.get("experiment_id"))
    for exp_id in exp_ids:
        exp_runs = [r for r in plan_runs if r.get("experiment_id") == exp_id]
        arms = []
        for run in exp_runs:
            if run["arm"] not in arms:
                arms.append(run["arm"])
        rows = []
        for run in exp_runs:
            style, slot, arm = run["style"], run["slot"], run["arm"]
            arm_dir = Path(run["output_dir"])
            repeats = sorted(p for p in arm_dir.glob("repeat-*")) if arm_dir.is_dir() else []
            for repeat_dir in repeats:
                manifest = record_of(repeat_dir / "request.json")
                key = "%s|%s|%s|%s|%s" % (exp_id, style, slot, arm, repeat_dir.name)
                review = reviews.get(key) or {}
                gate = manifest.get("final_gate") or {}
                selected = str(manifest.get("selected_output") or "")
                rows.append({
                    "run_id": key, "experiment_id": exp_id, "style": style, "slot": slot, "arm": arm,
                    "repeat": repeat_dir.name,
                    "manifest_status": manifest.get("status"),
                    "gate_status": gate.get("status"),
                    "gate_text": gate.get("text", "")[:400],
                    "output_path": selected,
                    "output_sha256": sha256_of(selected),
                    "identity_gate": (manifest.get("identity") or {}).get("action"),
                    "identity_mismatch": ((manifest.get("identity") or {}).get("final") or {}).get("mismatch"),
                    "eye_score": review.get("eye_score"), "face_score": review.get("face_score"),
                    "hair_score": review.get("hair_score"), "medium_score": review.get("medium_score"),
                    "eye_readable": review.get("eye_readable"),
                    "source_identity_kept": review.get("source_identity_kept"),
                    "iris_colour_kept": review.get("iris_colour_kept"),
                    "eye_state_kept": review.get("eye_state_kept"),
                    "composition_kept": review.get("composition_kept"),
                    "reference_character_copy": review.get("reference_character_copy"),
                    "evidence": review.get("evidence", ""),
                    "caveat": review.get("caveat", ""),
                })
        verdict = {"rows": rows, "per_slot": {}, "conclusion": "insufficient_evidence", "reasons": []}
        for style in sorted({row["style"] for row in rows}):
            for slot in STYLE_SLOTS:
                group = [row for row in rows if row["style"] == style and row["slot"] == slot]
                if not group:
                    continue
                by_arm = {}
                for row in group:
                    by_arm.setdefault(row["arm"], []).append(row)
                entry = {"arms": {arm: [{"repeat": r["repeat"], "eye_score": r["eye_score"],
                                         "gate": r["gate_status"], "output": r["output_path"]}
                                        for r in sorted(items, key=lambda x: x["repeat"])]
                                  for arm, items in by_arm.items()}}
                if exp_id == "P2" and len(by_arm) == 2:
                    control_arm, variant_arm = arms[0], arms[1]
                    pairs = []
                    for control in by_arm.get(control_arm, []):
                        variant = next((v for v in by_arm.get(variant_arm, [])
                                        if v["repeat"] == control["repeat"]), None)
                        if not variant:
                            continue
                        try:
                            delta = float(variant["eye_score"]) - float(control["eye_score"])
                        except (TypeError, ValueError):
                            delta = None
                        pairs.append({"repeat": control["repeat"], "control_eye": control["eye_score"],
                                      "variant_eye": variant["eye_score"], "delta": delta,
                                      "variant_gate": variant["gate_status"],
                                      "variant_identity_kept": variant["source_identity_kept"],
                                      "variant_iris_kept": variant["iris_colour_kept"],
                                      "variant_eye_state_kept": variant["eye_state_kept"],
                                      "variant_composition_kept": variant["composition_kept"]})
                    deltas = [p["delta"] for p in pairs if p["delta"] is not None]
                    entry["pairs"] = pairs
                    entry["max_delta"] = max(deltas) if deltas else None
                    entry["mean_delta"] = round(sum(deltas) / len(deltas), 3) if deltas else None
                    ok = bool(deltas) and max(deltas) >= 1 and (sum(deltas) / len(deltas)) >= 0
                    gates_ok = all(p["variant_gate"] == "complete" for p in pairs) if pairs else False
                    kept_ok = all(p.get("variant_identity_kept") in (True, "yes", "true") and
                                  p.get("variant_iris_kept") in (True, "yes", "true") and
                                  p.get("variant_eye_state_kept") in (True, "yes", "true") for p in pairs) if pairs else False
                    entry.update({"improvement_condition": ok, "all_variant_gates_complete": gates_ok,
                                  "source_identity_and_eye_state_kept": kept_ok,
                                  "success": bool(ok and gates_ok and kept_ok)})
                verdict["per_slot"]["%s/%s" % (style, slot)] = entry
        successes = [k for k, v in verdict["per_slot"].items() if v.get("success")]
        if exp_id == "P2":
            verdict["conclusion"] = "success_on_all_slots" if successes and len(successes) == len(verdict["per_slot"]) \
                else "partial_success" if successes else "no_success"
            verdict["reasons"] = ["slot %s 满足提升条件且门禁通过" % k for k in successes]
            missing = [k for k, v in verdict["per_slot"].items() if not v.get("success")]
            verdict["reasons"] += ["slot %s 未满足：%s" % (
                k, "缺评分" if not verdict["per_slot"][k].get("pairs") else
                "提升/门禁/保持条件未同时成立") for k in missing]
        else:
            copies = [row for row in rows if str(row.get("reference_character_copy")).lower() in ("yes", "true")]
            verdict["reference_character_copy_count"] = len(copies)
            verdict["source_identity_pass_count"] = sum(
                1 for row in rows if row.get("source_identity_kept") in (True, "yes", "true"))
            verdict["conclusion"] = "observational_only"
            verdict["reasons"] = ["P1 是观察性对照：记录复制次数与身份通过数，不设统计显著结论"]
        report["experiments"][exp_id] = verdict
    write_json(Path(args.out).resolve() / "verdict.json", report)
    print("判定: %s" % (Path(args.out).resolve() / "verdict.json"))
    for exp_id, verdict in report["experiments"].items():
        print("  %s: %s" % (exp_id, verdict["conclusion"]))
        for key, entry in verdict["per_slot"].items():
            extra = ""
            if "max_delta" in entry:
                extra = " maxΔ=%s meanΔ=%s success=%s" % (entry.get("max_delta"), entry.get("mean_delta"),
                                                          entry.get("success"))
            print("    %-14s rows=%d%s" % (key, sum(len(v) for v in entry["arms"].values()), extra))
        for reason in verdict.get("reasons", [])[:8]:
            print("      - " + reason)
    return 0


def now_stamp():
    return datetime.datetime.now().isoformat(timespec="seconds")


# 受控实验里一次「图片 API 调用」留下的痕迹：文件名里的 `_HHMMSS-` 段是后端写入的调用时间戳。
CALL_ARTIFACT = re.compile(r"(_|^)(\d{6})-[0-9a-f]{6}(-canvas)?\.(png|jpg|jpeg|webp)$", re.I)
REPLAY_FILE = re.compile(r"_aigc-2d-gpt_(replay|server_response)_(\d{6})_([0-9a-f]{6})\.json$", re.I)
AUDIT_FILE = re.compile(r"(audit|refine-quality|identity-(first|final|corrected)|final-quality|final-repair)"
                        r".*\.(json|txt)$", re.I)


def _call_id(path, stamp):
    return "%s-%s" % (stamp, Path(path).stem[-6:])


def round_calls(args):
    """建账本：按调用事件去重，区分「有 usage 证据」与「只能估」的部分。"""
    round_dir = Path(args.round).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    unit = {"gpt_first_pass_high_usd": 0.1582159716, "gemini_image_call_usd": 0.03639735,
            "text_call_usd": 0.002283194736}
    if args.estimate and Path(args.estimate).is_file():
        estimate = record_of(args.estimate)
        unit.update({k + "_usd": v for k, v in (estimate.get("unit_cost_usd") or {}).items()
                     if isinstance(v, (int, float))})
        unit["source"] = str(Path(args.estimate).resolve())
    else:
        unit["source"] = "utils/cost_estimate.py 的按次/公式定价（第二轮 snapshots/cost-estimate.json 同源）"

    calls = []
    # 1) GPT 通道：每个 replay 文件 = 一次真实出图请求（含 prompt 摘要与 usage 配对）
    replay_files = sorted(round_dir.rglob("*_aigc-2d-gpt_replay_*.json"))
    responses = {}
    for path in sorted(round_dir.rglob("*_aigc-2d-gpt_server_response_*.json")):
        match = REPLAY_FILE.search(path.name)
        if match and match.group(2) not in responses:
            responses[match.group(2)] = path
    for path in replay_files:
        match = REPLAY_FILE.search(path.name)
        stamp, tail = (match.group(2), match.group(3)) if match else ("", "")
        replay = record_of(path)
        # 同一次调用写两个文件：replay 记录请求，server_response 记录返回（含 token usage）。
        # 文件名里的 6 位 HHMMSS 是唯一的配对键；同目录同一秒只会发生一次调用。
        response_path = responses.get(stamp)
        usage = (record_of(response_path).get("usage") or {}) if response_path else {}
        rel = str(path.relative_to(round_dir)).replace("\\", "/")
        parts = rel.split("/")
        style, slot, arm = parts[1], parts[2], parts[3].replace("-gpt", "")
        calls.append({
            "call_id": "gpt-%s-%s" % (stamp or "000000", tail or path.stem[-6:]),
            "kind": "image_generate", "channel": "gpt-image-2",
            "style": style, "slot": slot, "arm": arm,
            "operation": "first_pass" if replay.get("mode") == "generate" else str(replay.get("mode") or "generate"),
            "request_file": str(path), "request_sha256": sha256_of(path),
            "response_file": str(response_path) if response_path else "",
            "response_sha256": sha256_of(response_path) if response_path else "",
            "usage": usage, "usage_source": "server_response" if usage else "missing",
            "model": str(replay.get("model") or "gpt-image-2"),
            "size": str(replay.get("size") or ""), "quality": str(replay.get("quality") or ""),
            "prompt_sha256": hashlib.sha256(str(replay.get("prompt") or "").encode("utf-8")).hexdigest(),
            "prompt_chars": len(str(replay.get("prompt") or "")),
            "image_refs": len(replay.get("image_paths") or replay.get("images") or []),
            "started": stamp, "finished": stamp, "cached": False,
            "cost_status": "estimated", "cost_usd": unit["gpt_first_pass_high_usd"],
        })
    # 2) 图片调用痕迹（Gemini 直出 / 重绘 / 质量修订 / 身份修订 / 兜底 / 五官修订）
    for path in sorted(round_dir.rglob("*")):
        if not path.is_file():
            continue
        match = CALL_ARTIFACT.search(path.name)
        if not match:
            continue
        rel = str(path.relative_to(round_dir)).replace("\\", "/")
        parts = rel.split("/")
        style = slot = arm = ""
        if len(parts) > 3 and parts[0] == "results":
            style, slot = parts[1], parts[2]
            arm = parts[3].split("-")[0]
        name = path.name
        if "final-rescue" in name:
            operation, channel = "final_rescue_repair", "gemini-image"
        elif "quality-refine" in name:
            operation, channel = "quality_refine_repair", "gemini-image"
        elif "identity-correct" in name:
            operation, channel = "identity_correction", "gemini-image"
        elif "face-hair-style" in name:
            operation, channel = "face_hair_refine", "gemini-image"
        elif "final-rp" in name or "-pp" in name:
            operation, channel = "repaint", "gemini-image"
        elif name.rsplit("-", 1)[-1].split(".")[0].isdigit() or re.match(r"^[0-9a-f]{8}_\d{6}-", name):
            operation, channel = "gemini_direct", "gemini-image"
        else:
            operation, channel = "unclassified", "unknown"
        calls.append({
            "call_id": "img-%s-%s" % (match.group(2), Path(name).stem[-6:]),
            "kind": "image_generate", "channel": channel,
            "style": style, "slot": slot, "arm": arm, "operation": operation,
            "output_file": str(path), "output_sha256": sha256_of(path),
            "output_bytes": path.stat().st_size,
            "usage": {}, "usage_source": "missing", "model": "",
            "started": match.group(2), "finished": match.group(2), "cached": False,
            "cost_status": "estimated" if channel == "gemini-image" else "unknown",
            "cost_usd": unit["gemini_image_call_usd"] if channel == "gemini-image" else 0.0,
        })
    # 3) 审计调用：同一目录同一文件名后缀重复出现 = 同一次调用的多次落盘，去重
    audit_seen = {}
    for path in sorted(round_dir.rglob("*.json")):
        if not AUDIT_FILE.search(path.name):
            continue
        digest = sha256_of(path)
        key = (str(path.parent), path.name)
        audit_seen.setdefault(key, []).append(digest)
    audit_calls = []
    for (folder, name), digests in sorted(audit_seen.items()):
        rel = str(Path(folder).relative_to(round_dir)).replace("\\", "/") if folder.startswith(str(round_dir)) else folder
        audit_calls.append({"file": str(Path(folder) / name), "copies": len(digests),
                            "distinct_content": len(set(digests)), "rel": rel})
    deduped_audits = len(audit_calls)

    legacy = Path(round_dir) / "snapshots" / "actual-usage.json"
    legacy_data = record_of(legacy)
    ledger = {
        "generated_at": now_stamp(), "round_dir": str(round_dir),
        "unit_cost": unit,
        "counting_rule": {
            "image_call": "后端每次调用都会在产物文件名里写 `_HHMMSS` 调用戳；同一戳只计一次。",
            "gpt_attempt": "每个 `*_aigc-2d-gpt_replay_*.json` = 一次真实出图请求（含失败与归档 attempt）。",
            "audit_call": "按目录+文件名去重后的审计 JSON 份数；同一次审计写多份只计一次。",
            "audit_call_is_text": "审计走文本模型（三图审计），不是图片调用，单价按 text_call。",
        },
        "evidence_limits": [
            "Gemini 图片调用没有 usage 落盘，只能按产物文件名里的调用戳计一次；"
            "被丢弃或报错且没有留下文件的调用计不到，因此这是下界。",
            "审计 JSON 份数不等于调用次数：同一次审计可能写多份；本账本去重后单列，且审计是文本调用。",
            "没有账单导出，所有金额都是估算；有 usage 证据的只有 gpt 通道的 token 数。",
        ],
        "counts": {
            "image_calls_total": sum(1 for c in calls if c["kind"] == "image_generate"),
            "gpt_image_calls": sum(1 for c in calls if c["channel"] == "gpt-image-2"),
            "gemini_image_calls": sum(1 for c in calls if c["channel"] == "gemini-image"),
            "gemini_direct_calls": sum(1 for c in calls if c["operation"] == "gemini_direct"),
            "repaint_calls": sum(1 for c in calls if c["operation"] == "repaint"),
            "quality_refine_calls": sum(1 for c in calls if c["operation"] == "quality_refine_repair"),
            "identity_correction_calls": sum(1 for c in calls if c["operation"] == "identity_correction"),
            "final_rescue_calls": sum(1 for c in calls if c["operation"] == "final_rescue_repair"),
            "face_hair_refine_calls": sum(1 for c in calls if c["operation"] == "face_hair_refine"),
            "unclassified_image_calls": sum(1 for c in calls if c["operation"] == "unclassified"),
            "audit_json_files": deduped_audits,
            "gpt_calls_with_usage": sum(1 for c in calls if c["usage_source"] == "server_response"),
            "gpt_calls_total": len(replay_files),
        },
        "cost": {
            "known_usage_tokens": {
                "input": sum(int((c["usage"] or {}).get("input_tokens") or 0) for c in calls),
                "output": sum(int((c["usage"] or {}).get("output_tokens") or 0) for c in calls),
                "total": sum(int((c["usage"] or {}).get("total_tokens") or 0) for c in calls),
            },
            "known_usage_calls": sum(1 for c in calls if c["usage_source"] == "server_response"),
            "estimated_usd": round(sum(float(c.get("cost_usd") or 0) for c in calls), 4),
            "billed_usd": None,
            "billing_gap": "没有账单导出。gpt 出图按质量档单价估、Gemini 按按次单价估；"
                           "失败与丢弃的调用未计入，真实账单不可知。",
        },
        "calls": calls,
        "audit_inventory": audit_calls[:400],
    }
    # 分类账：金额按调用类别拆开，每一类标明证据等级
    buckets = {
        "gpt_image_first_pass": ("gpt-image-2", "estimated_high_unit", unit["gpt_first_pass_high_usd"]),
        "gemini_direct": ("gemini-image", "estimated_per_call", unit["gemini_image_call_usd"]),
        "repaint": ("gemini-image", "estimated_per_call", unit["gemini_image_call_usd"]),
        "quality_refine_repair": ("gemini-image", "estimated_per_call", unit["gemini_image_call_usd"]),
        "identity_correction": ("gemini-image", "estimated_per_call", unit["gemini_image_call_usd"]),
        "final_rescue_repair": ("gemini-image", "estimated_per_call", unit["gemini_image_call_usd"]),
        "face_hair_refine": ("gemini-image", "estimated_per_call", unit["gemini_image_call_usd"]),
        "unclassified": ("unknown", "unknown_unit", 0.0),
    }
    breakdown = []
    for name, (_channel, basis, price) in buckets.items():
        if name == "gpt_image_first_pass":
            count = ledger["counts"]["gpt_calls_total"]
        else:
            count = sum(1 for c in calls if c["operation"] == name)
        if not count:
            continue
        breakdown.append({"category": name, "count": count, "billing_basis": basis,
                          "unit_usd": price, "estimated_usd": round(count * price, 4),
                          "evidence": "server_response usage token 数（价格仍是估算）"
                                      if name == "gpt_image_first_pass" else "仅有产物文件时间戳"})
    if deduped_audits:
        breakdown.append({"category": "audit_text_calls", "count": deduped_audits,
                          "billing_basis": "estimated_text_unit", "unit_usd": unit["text_call_usd"],
                          "estimated_usd": round(deduped_audits * unit["text_call_usd"], 4),
                          "evidence": "审计 JSON 去重份数；不是图片调用"})
    ledger["cost"]["by_category"] = breakdown
    ledger["cost"]["estimated_usd_including_audits"] = round(
        sum(float(b["estimated_usd"]) for b in breakdown), 4)
    with (out / "calls.jsonl").open("w", encoding="utf-8") as handle:
        for call in calls:
            handle.write(json.dumps(call, ensure_ascii=False) + "\n")
    write_json(out / "calls-summary.json", {k: v for k, v in ledger.items() if k != "calls"})
    if legacy_data:
        write_json(out / "previous-usage-claim.json", legacy_data)
    print("调用账本: %s" % (out / "calls.jsonl"))
    print("  图片调用 %d（gpt-image %d / gemini %d：直出 %d、重绘 %d、质量修订 %d、身份修订 %d、兜底 %d）" % (
        ledger["counts"]["image_calls_total"], ledger["counts"]["gpt_image_calls"],
        ledger["counts"]["gemini_image_calls"], ledger["counts"]["gemini_direct_calls"],
        ledger["counts"]["repaint_calls"], ledger["counts"]["quality_refine_calls"],
        ledger["counts"]["identity_correction_calls"], ledger["counts"]["final_rescue_calls"]))
    print("  审计 JSON 去重后 %d 份；gpt 调用带 usage 的 %d/%d 次" % (
        ledger["counts"]["audit_json_files"], ledger["counts"]["gpt_calls_with_usage"],
        ledger["counts"]["gpt_calls_total"]))
    for row in ledger["cost"]["by_category"]:
        print("    %-22s %4d × $%.6f = $%.4f（%s）" % (
            row["category"], row["count"], row["unit_usd"], row["estimated_usd"], row["evidence"]))
    print("  估算合计 $%.4f（含审计；无账单证据，见 billing_gap）" % (
        ledger["cost"]["estimated_usd_including_audits"]))
    return 0


# ---------------------------------------------------------------------------
# 第四轮（eye-face-hair-round4-20261001）族
#
# 与 round2/round3 的两点关键差别（PROTOCOL.md P0.3/P0.4）：
#   1. 有效请求由业务模块的**同一个组装函数**产出，预检与实际发送共用一份结果；
#   2. 付费动作在真正发出 HTTP 之前先原子预留名额，硬上限跨进程累计。
# 所有功能都在本 CLI 内，不往 data/ 下放可执行脚本。
# ---------------------------------------------------------------------------

ROUND4_DEFAULT_DIR = ROOT / "data" / "exp" / "eye-face-hair-round4-20261001"
ROUND4_PROMPT_REL = "prompts/gpt-image-optimize/eye-round4-20261001"
ROUND4_ASSETS = ("common.md", "control-clauses.json", "variant-clauses.json")
ROUND4_CLAUSE_HEADER = "\n\nSTYLE LANGUAGE (rendering targets for this repaint):\n- "


def _round4_dir(value=""):
    return Path(value).resolve() if value else ROUND4_DEFAULT_DIR


def _round4_installed(name: str) -> Path:
    return ROOT / ROUND4_PROMPT_REL / name


def _round4_read_jsonl(path) -> list:
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def round4_install_assets(args):
    """把 `prompt-assets/` 的素材**逐字**安装到项目 prompts 目录（不解释、不改写）。"""
    round_dir = _round4_dir(args.round)
    assets = round_dir / "prompt-assets"
    target = ROOT / ROUND4_PROMPT_REL
    target.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for name in ROUND4_ASSETS:
        source = assets / name
        if not source.is_file():
            problems.append(f"素材缺失：{source}")
            continue
        destination = target / name
        source_bytes = source.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        replaced = False
        if destination.is_file() and destination.read_bytes() != source_bytes:
            if not args.force:
                problems.append(f"目标已存在且内容不同（拒绝覆盖，需要 --force）：{destination}")
                continue
            replaced = True
        if not destination.is_file() or replaced:
            destination.write_bytes(source_bytes)
        installed_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
        if installed_sha != source_sha:
            problems.append(f"安装后哈希不一致：{destination}")
        rows.append({"asset": name, "source": str(source), "installed": str(destination),
                     "sha256": source_sha, "bytes": len(source_bytes), "replaced": replaced})
    manifest = {"installed_at": now_stamp(), "round": str(round_dir), "directory": str(target),
                "files": rows, "problems": problems,
                "note": "运行时通过现有 prompt loader 读取；安装是逐字节复制，不做任何改写。"}
    write_json(round_dir / "prompt-install.json", manifest)
    for row in rows:
        print("安装 %-22s %s" % (row["asset"], row["sha256"][:16]))
    for text in problems:
        print("  ! " + text)
    return 1 if problems else 0


def _round4_asset_facts(round_dir) -> tuple:
    """比对 `prompt-assets/` 与项目 prompts 目录里的运行时副本。"""
    rows, problems = [], []
    for name in ROUND4_ASSETS:
        source = round_dir / "prompt-assets" / name
        installed = _round4_installed(name)
        source_sha = digest(source) if source.is_file() else ""
        installed_sha = digest(installed) if installed.is_file() else ""
        rows.append({"asset": name, "source_sha256": source_sha,
                     "installed_sha256": installed_sha, "installed": str(installed),
                     "identical": bool(source_sha) and source_sha == installed_sha})
        if not source_sha:
            problems.append(f"缺少素材 {source}")
        elif not installed_sha:
            problems.append(f"运行时副本未安装：{installed}（先跑 round4-install-assets）")
        elif source_sha != installed_sha:
            problems.append(f"运行时副本与素材不一致：{installed}")
    return rows, problems


def _round4_input_facts(round_dir, declared) -> tuple:
    """核验冻结输入的哈希与声明一致（不改任何文件）。"""
    rows, problems = [], []
    for item in declared.get("inputs") or []:
        slot = item.get("slot")
        for key, sha_key in (("base", "base_sha256"), ("analysis", "analysis_sha256"),
                             ("original_source", "original_source_sha256")):
            relative = str(item.get(key) or "")
            path = (round_dir / relative).resolve() if relative else None
            actual = digest(path) if path and path.is_file() else ""
            expected = str(item.get(sha_key) or "")
            ok = bool(actual) and actual == expected
            rows.append({"slot": slot, "role": key, "path": str(path or ""),
                         "expected_sha256": expected, "actual_sha256": actual, "match": ok,
                         "recorded_source": str(item.get(key + "_from") or "")})
            if not ok:
                problems.append(f"{slot}/{key}: 哈希与声明不符（声明 {expected[:12]} / 实际 "
                                f"{actual[:12] or '文件不存在'}）")
    return rows, problems


def _round4_clause_facts(round_dir) -> tuple:
    """两臂条款文件的差异必须**只有**那一条 eye 句。"""
    control_path = round_dir / "prompt-assets" / "control-clauses.json"
    variant_path = round_dir / "prompt-assets" / "variant-clauses.json"
    if not (control_path.is_file() and variant_path.is_file()):
        return {}, ["缺少 control/variant 条款文件"]
    control = read_json(control_path)
    variant = read_json(variant_path)
    if not (isinstance(control, list) and isinstance(variant, list)):
        return {}, ["条款文件必须是 JSON 数组"]
    if len(control) != 1 or len(variant) != 1:
        return {}, ["本轮每个条款文件必须恰好一条 eye 句（其余段落来自固件）"]
    facts = {"control_clause": control[0], "variant_clause": variant[0],
             "control_chars": len(control[0]), "variant_chars": len(variant[0]),
             "identical": control[0] == variant[0]}
    problems = []
    if facts["identical"]:
        problems.append("两臂条款完全相同，变体没有干预")
    return facts, problems


def _round4_argv(run) -> list:
    """把一条计划 run 翻译成 `tools/analysis_gpt_run.py` 的实参。

    受控实验只做**一次**整图 source 重绘：不送画风图（`--repaint-ref none`）、
    不做首图、不触发任何自动修图（`--audit-only`），审计只读。
    """
    return [
        "--json", run["analysis"],
        "--style", run["style"],
        "--styles-file", run["styles_file"],
        "--quality", "high",
        "--size", "auto",
        "--source-image", run["source_image"],
        "--steps", "repaint",
        "--repaint-scope", "full",
        "--repaint-ref", "none",
        "--repaint-clauses-only",
        "--repaint-clauses-file", run["clauses_file"],
        "--firmware", run["firmware"],
        "--base-image", run["base"],
        "--repaint-style-ref", run["audit_style_ref"],
        "--identity-audit",
        "--audit-only",
        "--final-review-audit",
        "--output-dir", run["output_dir"],
    ]


def _round4_call_cli(argv, cli_module=None):
    """在**本进程内**按给定实参跑一次 CLI（sys.argv 是它读参数的方式）。"""
    if cli_module is None:
        import tools.analysis_gpt_run as cli_module  # noqa: PLC0415
    saved = sys.argv
    sys.argv = ["analysis_gpt_run.py"] + list(argv)
    try:
        return cli_module.main()
    finally:
        sys.argv = saved


def round4_build_plan(args):
    """声明式任务清单 → 可执行的 `run-plan.json`（零图片调用）。

    `experiment-plan.json` **不是**现有 round-run schema：这里显式转换，并在转换时
    逐条核验冻结输入、素材安装、条款唯一差异、最终有效请求（≤3000 字符）与
    「删掉唯一 eye 句后剩余 prompt 逐字相同」。
    """
    round_dir = _round4_dir(args.round)
    declared_path = Path(args.plan) if args.plan else (round_dir / "experiment-plan.json")
    declared = read_json(declared_path)
    out = Path(args.out).resolve() if args.out else round_dir
    problems, checks = [], []
    if declared.get("schema") != "eye-round4-declarative-v1":
        problems.append("声明文件的 schema 不是 eye-round4-declarative-v1：" + str(declared.get("schema")))
    limit = int(declared.get("image_attempt_limit") or 0)
    declared_runs = declared.get("runs") or []
    repeats_expected = int(declared.get("repeats_per_arm") or 2)
    if len(declared_runs) != limit:
        problems.append(f"计划 run 数 {len(declared_runs)} 与图片尝试上限 {limit} 不一致")
    if int(declared.get("image_auto_retries") or 0) != 0:
        problems.append("声明文件要求 image_auto_retries=0")
    checks.append({"check": "declarative_schema", "ok": not problems, "limit": limit,
                   "declared_runs": len(declared_runs), "repeats_per_arm": repeats_expected})
    input_rows, input_problems = _round4_input_facts(round_dir, declared)
    asset_rows, asset_problems = _round4_asset_facts(round_dir)
    clause_facts, clause_problems = _round4_clause_facts(round_dir)
    problems += input_problems + asset_problems + clause_problems
    slots = {str(item.get("slot")): item for item in (declared.get("inputs") or [])}
    assets = {name: str(_round4_installed(name)) for name in ROUND4_ASSETS}
    firmware = str(_round4_installed("common.md"))
    budget_dir = str(out / "budget")
    runs = []
    for run in declared_runs:
        slot = str(run.get("slot"))
        item = slots.get(slot)
        if not item:
            problems.append(f"{run.get('run_id')}: 声明里没有 slot={slot} 的输入")
            continue
        entry = {
            "run_id": str(run.get("run_id") or ""),
            "experiment_id": str(run.get("experiment_id") or "E1"),
            "style": str(run.get("style") or "sheya-style"),
            "slot": slot, "arm": str(run.get("arm")), "repeat": int(run.get("repeat") or 1),
            "stage": "single_source_only_eye_clause_repaint",
            "base": str((round_dir / str(run.get("base"))).resolve()),
            "analysis": str((round_dir / str(run.get("analysis"))).resolve()),
            "source_image": str((round_dir / str(item.get("original_source"))).resolve()),
            "styles_file": str((round_dir / "inputs" / "config-styles-frozen.json").resolve()),
            "audit_style_ref": str((round_dir / str(declared.get("audit_style_reference")
                                                    or "inputs/sheya-style-reference.png")).resolve()),
            "clauses_asset": str(run.get("clauses_asset") or ""),
            "clauses_file": assets["control-clauses.json" if "control" in str(run.get("arm"))
                                    else "variant-clauses.json"],
            "firmware": firmware,
            "output_dir": str((round_dir / str(run.get("output_dir"))).resolve()),
            "declared_status": str(run.get("status") or "not_run"),
            "style_reference_sent": False,
            "generation_style_reference_sent": False,
            "audit_style_reference": str(declared.get("audit_style_reference") or ""),
            "expected_diff": ["唯一差异 = 那条 eye 句（STYLE LANGUAGE 里的 EYE DRAWING 句）；其余 prompt 逐字相同",
                              "两臂都不送画风参考图到生成接口；画风图只进审计"],
        }
        missing = [key for key in ("base", "analysis", "source_image", "styles_file",
                                   "audit_style_ref", "clauses_file", "firmware")
                   if not Path(entry[key]).is_file()]
        for key in missing:
            problems.append(f"{entry['run_id']}: 缺少 {key} → {entry[key]}")
        for key, field in (("base", "base_sha256"), ("analysis", "analysis_sha256"),
                           ("source_image", "source_image_sha256"), ("firmware", "firmware_sha256"),
                           ("clauses_file", "clauses_sha256")):
            entry[field] = digest(entry[key]) if Path(entry[key]).is_file() else ""
        entry["command"] = [sys.executable, str(ROOT / "tools" / "analysis_gpt_run.py")] + _round4_argv(entry)
        if missing:
            runs.append(entry)
            continue
        # 用**同一个组装函数**算出真正会被送出的有效请求，并落盘冻结。
        probe_argv = _round4_argv(entry) + ["--effective-request-only"]
        os.makedirs(entry["output_dir"], exist_ok=True)
        code = _round4_call_cli(probe_argv)
        probe_path = Path(entry["output_dir"]) / "effective-request.json"
        entry["effective_request_path"] = str(probe_path)
        entry["probe_returncode"] = code
        if not probe_path.is_file():
            problems.append(f"{entry['run_id']}: 有效请求组装失败（returncode={code}）")
            runs.append(entry)
            continue
        request = read_json(probe_path)
        # 冻结副本放在 `effective-requests/`：预检会把产物目录里的临时请求清掉，
        # 付费前要比对的那一份必须放在预检碰不到的地方。
        frozen_store = round_dir / "effective-requests"
        frozen_store.mkdir(parents=True, exist_ok=True)
        frozen_path = frozen_store / (entry["run_id"] + ".json")
        frozen_path.write_bytes(probe_path.read_bytes())
        entry["effective_request_path"] = str(frozen_path)
        entry["effective_request_probe_path"] = str(probe_path)
        entry["effective_request_sha256"] = digest(frozen_path)
        entry["planned_prompt_sha256"] = request.get("prompt_sha256")
        entry["prompt_chars"] = int(request.get("prompt_chars") or 0)
        entry["effective_request"] = {
            "prompt": request.get("prompt"),
            "source_paths": request.get("source_paths"),
            "extra_reference_paths": request.get("extra_reference_paths"),
            "reference_images": request.get("reference_images"),
            "model": request.get("model"), "resolution": request.get("resolution"),
            "aspect_ratio": request.get("aspect_ratio"), "use_detail_suffix": request.get("use_detail_suffix"),
            "detail_suffix_applied": request.get("detail_suffix_applied"),
            "n": request.get("n"), "repeat": request.get("repeat"),
            "save_sub_dir": request.get("save_sub_dir"), "file_prefix": request.get("file_prefix"),
        }
        runs.append(entry)
    # 有效请求约束：长度上限 + 两臂只差一条 eye 句
    max_chars = int(declared.get("effective_repaint_prompt_max_chars") or 3000)
    by_pair = {}
    for entry in runs:
        by_pair.setdefault((entry["slot"], entry["repeat"]), {})[entry["arm"]] = entry
    pair_rows = []
    for (slot, repeat), arms in sorted(by_pair.items()):
        if "control" not in arms or "variant" not in arms:
            problems.append(f"{slot}/repeat-{repeat}: 缺少 control 或 variant 臂")
            continue
        control_prompt = str((arms["control"].get("effective_request") or {}).get("prompt") or "")
        variant_prompt = str((arms["variant"].get("effective_request") or {}).get("prompt") or "")
        if not control_prompt or not variant_prompt:
            problems.append(f"{slot}/repeat-{repeat}: 有效请求缺少 prompt")
            continue
        control_clause = str(clause_facts.get("control_clause") or "")
        variant_clause = str(clause_facts.get("variant_clause") or "")
        row = {"slot": slot, "repeat": repeat,
               "control_prompt_chars": len(control_prompt), "variant_prompt_chars": len(variant_prompt),
               "limit": max_chars,
               "control_clause_present_once": control_prompt.count(control_clause) == 1,
               "variant_clause_present_once": variant_prompt.count(variant_clause) == 1}
        stripped_control = control_prompt.replace(ROUND4_CLAUSE_HEADER + control_clause, "", 1)
        stripped_variant = variant_prompt.replace(ROUND4_CLAUSE_HEADER + variant_clause, "", 1)
        row["remainder_identical"] = stripped_control == stripped_variant
        row["remainder_sha256"] = hashlib.sha256(stripped_control.encode("utf-8")).hexdigest()
        row["only_declared_clause_changed"] = bool(
            row["control_clause_present_once"] and row["variant_clause_present_once"]
            and row["remainder_identical"])
        row["within_limit"] = len(control_prompt) <= max_chars and len(variant_prompt) <= max_chars
        if not row["only_declared_clause_changed"]:
            problems.append(f"{slot}/repeat-{repeat}: 删除唯一 eye 句后剩余 prompt 不相同")
        if not row["within_limit"]:
            problems.append(f"{slot}/repeat-{repeat}: 有效 prompt 超过 {max_chars} 字符上限")
        for arm in ("control", "variant"):
            refs = (arms[arm].get("effective_request") or {}).get("extra_reference_paths") or []
            if refs:
                problems.append(f"{slot}/repeat-{repeat}/{arm}: 生成请求里出现了额外参考图 {refs}")
            sources = (arms[arm].get("effective_request") or {}).get("source_paths") or []
            if sources != [arms[arm].get("base")]:
                problems.append(f"{slot}/repeat-{repeat}/{arm}: 生成请求的源图不是冻结 base：{sources}")
        pair_rows.append(row)
    run_plan = {
        "schema": "eye-round4-run-plan-v1",
        "generated_at": now_stamp(),
        "round": str(round_dir),
        "declarative_plan": str(declared_path.resolve()),
        "declarative_sha256": digest(declared_path),
        "status": "ready_for_preflight" if not problems else "blocked",
        "image_attempt_limit": limit,
        "image_auto_retries": 0,
        "budget_dir": budget_dir,
        "text_audit_attempt_limit": int(declared.get("text_audit_attempt_limit") or 64),
        "text_audit_attempts_per_stage": 2,
        "prompt_directory": ROUND4_PROMPT_REL,
        "firmware": firmware, "firmware_sha256": digest(Path(firmware)) if Path(firmware).is_file() else "",
        "effective_repaint_prompt_max_chars": max_chars,
        "generation_style_reference_sent": False,
        "audit_style_reference": str((round_dir / str(declared.get("audit_style_reference") or "")).resolve()),
        "allowed_difference": declared.get("allowed_difference"),
        "ingredients": {"inputs": input_rows, "assets": asset_rows, "clauses": clause_facts},
        "prompt_pairs": pair_rows,
        "runs": runs,
        "runs_total": len(runs),
        "problems": problems,
        "notes": ["本文件由 round4-build-plan 从声明式清单转换而来；转换过程零图片调用。",
                  "模型/端点/分辨率来自现有配置，双方相同；预检记录脱敏快照。",
                  "不把上一张输出当作下一张输入；每次独立使用冻结 base。"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run-plan.json", run_plan)
    report = {"generated_at": run_plan["generated_at"], "round": str(round_dir),
              "checks": checks + [{"check": "inputs", "problems": input_problems},
                                  {"check": "assets", "problems": asset_problems},
                                  {"check": "clauses", "problems": clause_problems}],
              "prompt_pairs": pair_rows, "problems": problems,
              "verdict": "ready_for_preflight" if not problems else "blocked"}
    write_json(out / "run-plan-check.json", report)
    print("可执行计划 %d 个 run（图片尝试上限 %d）→ %s" % (len(runs), limit, out / "run-plan.json"))
    for entry in runs:
        print("  %-24s %s/%s r%d prompt=%d字 sha=%s" % (
            entry["run_id"], entry["slot"], entry["arm"], entry["repeat"],
            entry.get("prompt_chars") or 0, str(entry.get("planned_prompt_sha256"))[:10]))
    for text in problems:
        print("  ! " + text)
    return 1 if problems else 0


class _Round4FakeBackend:
    """受控实验的假图片后端：记录发送参数并在本地伪造产物，绝不发 HTTP。"""

    def __init__(self, log=None):
        self.calls = []
        self.tripwires = []
        self.log = log or (lambda m: None)

    def repaint(self, **kwargs):
        self.calls.append(kwargs)
        sources = [p for p in (kwargs.get("source_paths") or []) if p]
        sub_dir = str(kwargs.get("save_sub_dir") or "")
        prefix = str(kwargs.get("file_prefix") or "fake")
        os.makedirs(sub_dir, exist_ok=True)
        target = os.path.join(sub_dir, prefix + ".png")
        Image.open(sources[0]).convert("RGB").save(target)
        self.log("[fake] 重绘被调用一次 → " + target)
        return [target]

    def tripwire(self, name):
        def fail(*args, **kwargs):
            self.tripwires.append({"name": name, "args": str(args)[:200], "kwargs": str(kwargs)[:200]})
            raise AssertionError("受控实验出现未授权的图片调用：" + name)
        return fail


def _round4_fake_text_audits(counter):
    """假文本审计：返回**schema 完整**的结论，不做任何 HTTP，也不产生修图。"""
    import utils.refine_quality as quality

    def quality_audit(original, candidate, style, **kwargs):
        counter.append("quality-audit")
        if kwargs.get("final_review"):
            severity, gaps = "minor", [{"aspect": "line weight under the style target",
                                        "candidate": "soft", "target": "firmer tapered contours",
                                        "repair": "thicken local contours", "confidence": .8}]
        else:
            severity, gaps = "major", [{"aspect": "eye abstraction gap", "candidate": "soft eye edges",
                                        "target": "graphic eye construction",
                                        "repair": "tighten eye edges", "confidence": .9}]
        return {"needs_refine": True, "severity": severity, "confidence": .9,
                "structural_issues": [], "line_issues": [], "background_drift": [],
                "style_gaps": gaps, "ownership_uncertain": [], "needs_review": False,
                "candidate": os.path.abspath(candidate),
                "candidate_sha256": quality.file_sha256(candidate)}

    def anatomy_audit(candidate, **kwargs):
        counter.append("anatomy-audit")
        return {"needs_refine": False, "severity": "none", "confidence": .9,
                "structural_issues": [], "line_issues": [], "background_drift": [], "style_gaps": [],
                "ownership_uncertain": [], "needs_review": False,
                "candidate": os.path.abspath(candidate),
                "candidate_sha256": quality.file_sha256(candidate)}

    def identity_audit(image_path, analysis, **kwargs):
        counter.append("identity-audit")
        return {"mismatch": False, "severity": "none", "confidence": .9, "stable_anchors": [],
                "differences": [], "summary": "fake", "image": os.path.abspath(image_path),
                "candidate_sha256": quality.file_sha256(image_path)}

    return quality_audit, anatomy_audit, identity_audit


def round4_preflight(args):
    """假后端跑**完整请求组装路径**，截获真实发送参数（零 HTTP、零付费）。

    这是付费前置条件：证明每个 run 只有一次重绘请求，且没有任何 GPT 首图、
    质量/身份/人体/终审救援类图片请求；并把截获的发送参数与计划里冻结的
    有效请求逐字段比对。
    """
    from unittest.mock import patch
    import modules.others.api_backend as backend

    round_dir = _round4_dir(args.round)
    plan = read_json(Path(args.plan)) if args.plan else read_json(round_dir / "run-plan.json")
    if plan.get("problems"):
        print("计划本身有问题，先跑 round4-build-plan 修好：" + "; ".join(plan["problems"])[:400])
        return 1
    assertions, runs_report = [], []
    total_image_calls = 0
    for entry in plan.get("runs") or []:
        label = entry["run_id"]
        if args.run and args.run not in label:
            continue
        if (Path(entry["output_dir"]) / "run-record.json").is_file():
            # 已经有真实执行记录的 run 不允许被预检覆盖（输出目录不可覆盖）。
            runs_report.append({"run_id": label, "slot": entry["slot"], "arm": entry["arm"],
                                "repeat": entry["repeat"], "skipped": "already_executed",
                                "note": "该 run 已有 run-record.json，预检不触碰它的输出目录"})
            continue
        text_calls = []
        fake = _Round4FakeBackend(log=lambda m: None)
        quality_audit, anatomy_audit, identity_audit = _round4_fake_text_audits(text_calls)
        os.makedirs(entry["output_dir"], exist_ok=True)
        for name in ("effective-request.json", "request.json"):
            stale = Path(entry["output_dir"]) / name
            if stale.is_file():
                stale.unlink()
        captured = {}

        def _dispatch(request, **kwargs):
            captured["request"] = request
            return fake.repaint(**request["call_kwargs"])

        with patch.object(backend, "generate_image_repaint", fake.repaint), \
                patch.object(backend, "generate_image_aigc2d", fake.tripwire("generate_image_aigc2d")), \
                patch.object(backend, "generate_image_aigc2d_gpt", fake.tripwire("generate_image_aigc2d_gpt")), \
                patch.object(backend, "generate_image_openai_image", fake.tripwire("generate_image_openai_image")), \
                patch.object(backend, "generate_image_whatai", fake.tripwire("generate_image_whatai")), \
                patch("utils.post_process.dispatch_repaint_request", _dispatch), \
                patch("utils.refine_quality.audit_refine_quality", quality_audit), \
                patch("utils.refine_quality.audit_hand_quality", anatomy_audit), \
                patch("utils.identity_audit.audit_image_identity", identity_audit), \
                patch.dict(os.environ, {"IMAGE_MAKER_SEND_BUDGET_DIR": "",
                                        "IMAGE_MAKER_EFFECTIVE_REQUEST_DIR": ""}):
            code = _round4_call_cli(_round4_argv(entry))
        calls = fake.calls
        request = captured.get("request") or {}
        expected = entry.get("effective_request") or {}
        planned_sha = hashlib.sha256(str(expected.get("prompt") or "").encode("utf-8")).hexdigest()
        actual_sha = hashlib.sha256(str(request.get("prompt") or "").encode("utf-8")).hexdigest()
        checks = [
            {"assertion": "run 只有一次重绘图片请求", "ok": len(calls) == 1, "evidence": len(calls)},
            {"assertion": "没有 GPT 首图请求", "ok": not fake.tripwires,
             "evidence": [t["name"] for t in fake.tripwires]},
            {"assertion": "没有其他图片通道（openai/whatai/gemini 直出）请求",
             "ok": not fake.tripwires, "evidence": [t["name"] for t in fake.tripwires]},
            {"assertion": "最终 prompt 与冻结的有效请求逐字相同",
             "ok": bool(request.get("prompt")) and request.get("prompt") == expected.get("prompt"),
             "evidence": {"planned_sha256": planned_sha, "actual_sha256": actual_sha,
                          "planned_chars": len(str(expected.get("prompt") or "")),
                          "actual_chars": len(str(request.get("prompt") or ""))}},
            {"assertion": "生成请求只送冻结 base 一张图（顺序一致）",
             "ok": list(request.get("source_paths") or []) == list(expected.get("source_paths") or []),
             "evidence": {"actual": request.get("source_paths"), "planned": expected.get("source_paths")}},
            {"assertion": "没有额外参考图进入生成请求",
             "ok": not (request.get("extra_reference_paths") or []),
             "evidence": request.get("extra_reference_paths")},
            {"assertion": "模型/分辨率/比例与计划一致",
             "ok": [request.get("model"), request.get("resolution"), request.get("aspect_ratio")]
                   == [expected.get("model"), expected.get("resolution"), expected.get("aspect_ratio")],
             "evidence": {"actual": [request.get("model"), request.get("resolution"),
                                     request.get("aspect_ratio")],
                          "planned": [expected.get("model"), expected.get("resolution"),
                                      expected.get("aspect_ratio")]}},
            {"assertion": "use_detail_suffix=False（不追加细节后缀）",
             "ok": request.get("detail_suffix_applied") is False, "evidence": request.get("detail_suffix_applied")},
            {"assertion": "n=1（一次请求只出一张）", "ok": request.get("n") == 1, "evidence": request.get("n")},
            {"assertion": "源图内容哈希与冻结 base 一致",
             "ok": (request.get("reference_images") or [{}])[0].get("file_sha256") == entry.get("base_sha256"),
             "evidence": {"sent": (request.get("reference_images") or [{}])[0].get("file_sha256"),
                          "frozen": entry.get("base_sha256")}},
            {"assertion": "CLI 退出码在预期范围内", "ok": code in (0, 1), "evidence": code},
        ]
        for check in checks:
            check["run_id"] = label
        assertions.extend(checks)
        total_image_calls += len(calls)
        runs_report.append({
            "run_id": label, "slot": entry["slot"], "arm": entry["arm"], "repeat": entry["repeat"],
            "returncode": code, "image_dispatch_count": len(calls),
            "tripwire_hits": fake.tripwires, "text_audit_calls": len(text_calls),
            "text_audit_breakdown": {name: text_calls.count(name) for name in sorted(set(text_calls))},
            "effective_request_path": str(Path(entry["output_dir"]) / "effective-request.json"),
            "effective_request_sha256": digest(Path(entry["output_dir"]) / "effective-request.json")
            if (Path(entry["output_dir"]) / "effective-request.json").is_file() else "",
            "captured_prompt_sha256": actual_sha,
            "captured_reference_images": request.get("reference_images"),
            "captured_call_kwargs_keys": sorted((calls[0] if calls else {}).keys()),
            "checks": checks,
        })
    failed = [item for item in assertions if not item["ok"]]
    checked = [row for row in runs_report if not row.get("skipped")]
    report = {"generated_at": now_stamp(), "round": str(round_dir), "mode": "fake_backend_preflight",
              "real_http_requests": 0, "real_image_calls": 0,
              "runs": runs_report, "assertions": assertions, "failures": failed,
              "summary": {"runs_checked": len(checked), "runs_skipped": len(runs_report) - len(checked),
                          "image_dispatches": total_image_calls,
                          "assertions": len(assertions), "failed": len(failed)},
              "code_versions": code_versions(),
              "verdict": "pass" if (not failed and checked
                                    and total_image_calls == len(checked)) else "failed"}
    out = Path(args.out).resolve() if args.out else round_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "preflight-report.json", report)
    print("预检：%d 个 run，重绘派发 %d 次，断言 %d 条，失败 %d 条 → %s" % (
        len(runs_report), total_image_calls, len(assertions), len(failed), report["verdict"]))
    for item in failed:
        print("  ! %s / %s：%s" % (item["run_id"], item["assertion"], str(item["evidence"])[:240]))
    return 0 if report["verdict"] == "pass" else 1


def code_versions() -> dict:
    """本轮的代码版本快照（git 状态 + 关键文件哈希）。"""
    files = ("utils/gate_status.py", "utils/refine_quality.py", "utils/identity_audit.py",
             "utils/post_process.py", "utils/send_budget.py", "utils/gpt_image_optimize.py",
             "modules/others/api_backend.py", "tools/analysis_gpt_run.py", "tools/style_face_eval.py")
    hashes = {}
    for name in files:
        path = ROOT / name
        hashes[name] = digest(path) if path.is_file() else ""

    def _git(*command):
        try:
            done = subprocess.run(["git", "-C", str(ROOT), *command], capture_output=True,
                                  text=True, timeout=30)
            return done.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return "<git unavailable: %s>" % type(exc).__name__
    return {"generated_at": now_stamp(), "file_sha256": hashes, "git_head": _git("rev-parse", "HEAD"),
            "git_status_lines": _git("status", "--porcelain").splitlines()[:200]}


def round4_run(args):
    """按预算 wrapper 执行**一条**计划 run（真正付费的那一步）。

    付费之前：检查计划与冻结有效请求、拒绝重复派发、检查输出目录未被占用；
    然后设置预算环境变量并串行 spawn `tools/analysis_gpt_run.py`。
    """
    import subprocess as sp
    import utils.send_budget as send_budget

    round_dir = _round4_dir(args.round)
    plan = read_json(Path(args.plan)) if args.plan else read_json(round_dir / "run-plan.json")
    runs = plan.get("runs") or []
    match = [r for r in runs if r.get("run_id") == args.run_id]
    if not match:
        raise SystemExit("run-plan.json 里没有 run_id=" + str(args.run_id))
    entry = match[0]
    budget_dir = str(Path(plan.get("budget_dir") or (round_dir / "budget")).resolve())
    limit = int(plan.get("image_attempt_limit") or 8)
    out_dir = Path(entry["output_dir"])
    if (out_dir / "run-record.json").is_file():
        raise SystemExit("该 run 已有执行记录，拒绝重复派发：" + str(out_dir))
    reusable = send_budget.can_reuse_success(budget_dir, entry["run_id"])
    if reusable:
        print("断点恢复：该 run 已有成功产物，直接复用（不再发送）" + reusable)
        return 0
    state = send_budget.state(budget_dir, "image", limit)
    if state["used"] >= limit:
        raise SystemExit("图片尝试预算已用满（%d/%d），不发送" % (state["used"], limit))
    probe = Path(entry.get("effective_request_path") or "")
    if not probe.is_file():
        raise SystemExit("缺少冻结的有效请求，先跑 round4-build-plan：" + str(probe))
    if digest(probe) != entry.get("effective_request_sha256"):
        raise SystemExit("冻结的有效请求已被改动，拒绝派发：" + str(probe))
    command = entry.get("command") or []
    if not command:
        raise SystemExit("计划里没有可执行命令")
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "IMAGE_MAKER_SEND_BUDGET_DIR": budget_dir,
        "IMAGE_MAKER_SEND_BUDGET_RUN_ID": entry["run_id"],
        "IMAGE_MAKER_SEND_BUDGET_LIMIT": str(limit),
        "IMAGE_MAKER_TEXT_BUDGET_LIMIT": str(plan.get("text_audit_attempt_limit") or 64),
        "IMAGE_MAKER_TEXT_BUDGET_PER_STAGE": "2",
        "IMAGE_MAKER_IMAGE_MAX_RETRIES": "0",
        "IMAGE_MAKER_EFFECTIVE_REQUEST_DIR": str(out_dir),
        "PYTHONIOENCODING": "utf-8",
    })
    log_path = out_dir / "run.log"
    print("派发 %s（预算 %d/%d）" % (entry["run_id"], state["used"], limit))
    if args.dry_run:
        print("[dry-run] " + " ".join(command))
        return 0
    started = datetime.datetime.now()
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        proc = sp.run(command, cwd=str(ROOT), stdout=stream, stderr=sp.STDOUT,
                      stdin=sp.DEVNULL, env=env)
    record = {
        "run_id": entry["run_id"], "slot": entry["slot"], "arm": entry["arm"],
        "repeat": entry["repeat"], "stage": entry["stage"],
        "base_path": entry["base"], "base_sha256": entry["base_sha256"],
        "analysis_path": entry["analysis"], "analysis_sha256": entry["analysis_sha256"],
        "source_image": entry["source_image"], "source_image_sha256": entry["source_image_sha256"],
        "firmware": entry["firmware"], "firmware_sha256": entry["firmware_sha256"],
        "clauses_file": entry["clauses_file"], "clauses_sha256": entry["clauses_sha256"],
        "audit_style_ref": entry["audit_style_ref"], "generation_style_reference_sent": False,
        "planned_effective_request": entry.get("effective_request_path"),
        "planned_effective_request_sha256": entry.get("effective_request_sha256"),
        "planned_prompt_sha256": entry.get("planned_prompt_sha256"),
        "command": " ".join(command), "returncode": proc.returncode,
        "seconds": round((datetime.datetime.now() - started).total_seconds(), 1),
        "started_at": started.isoformat(timespec="seconds"), "finished_at": now_stamp(),
        "log": str(log_path),
        "budget_used_after": send_budget.state(budget_dir, "image", limit)["used"],
        "cost_status": "no_billing_export; estimate only (utils/cost_estimate.py)",
    }
    manifest = record_of(out_dir / "request.json")
    record["manifest_status"] = manifest.get("status")
    selected = str(manifest.get("selected_output") or "")
    record["selected_output"] = selected
    record["output_sha256"] = digest(selected) if selected and Path(selected).is_file() else ""
    gate = manifest.get("final_gate") or {}
    record["final_gate"] = gate
    record["gate_status"] = gate.get("status")
    record["gate_audits"] = gate.get("audits")
    record["gate_reasons"] = gate.get("reasons")
    record["audit_image_sha256"] = {
        "identity": str(((manifest.get("identity") or {}).get("final") or {}).get("candidate_sha256") or ""),
        "anatomy": str((manifest.get("anatomy_review") or {}).get("candidate_sha256") or ""),
        "final_review": str((manifest.get("final_review") or {}).get("candidate_sha256") or ""),
        "quality_latest": str((((manifest.get("quality_refine") or {}).get("after")
                                or (manifest.get("quality_refine") or {}).get("before")) or {})
                              .get("candidate_sha256") or ""),
    }
    sent = sorted(Path(out_dir).glob("effective-request-*.json"))
    actual_request = read_json(sent[-1]) if sent else {}
    record["actual_effective_request_sha256"] = digest(sent[-1]) if sent else ""
    record["actual_request_matches_plan"] = bool(
        actual_request and actual_request.get("prompt_sha256") == entry.get("planned_prompt_sha256"))
    record["actual_reference_images"] = actual_request.get("reference_images")
    calls = _round4_read_jsonl(out_dir / "api-calls.jsonl")
    record["repaint_calls"] = len(calls)
    record["repaint_prompt_sha256"] = [c.get("prompt_sha256") for c in calls]
    record["repaint_reference_paths"] = [[item.get("path") for item in (c.get("reference_images") or [])]
                                         for c in calls]
    write_json(out_dir / "run-record.json", record)
    print("%s rc=%s status=%s gate=%s → %s" % (
        entry["run_id"], proc.returncode, record["manifest_status"], record["gate_status"],
        str(record["output_sha256"])[:12]))
    return 0 if proc.returncode == 0 else 1


def round4_state(args):
    """读预算账本与逐 run 记录（不发送任何东西）。"""
    import utils.send_budget as send_budget
    round_dir = _round4_dir(args.round)
    plan = read_json(Path(args.plan)) if args.plan else read_json(round_dir / "run-plan.json")
    budget_dir = str(Path(plan.get("budget_dir") or (round_dir / "budget")).resolve())
    limit = int(plan.get("image_attempt_limit") or 8)
    state = send_budget.state(budget_dir, "image", limit)
    rows = []
    for entry in plan.get("runs") or []:
        record = record_of(Path(entry["output_dir"]) / "run-record.json")
        rows.append({"run_id": entry["run_id"], "slot": entry["slot"], "arm": entry["arm"],
                     "repeat": entry["repeat"], "executed": bool(record),
                     "status": record.get("manifest_status", "not_run"),
                     "gate": record.get("gate_status", ""),
                     "output_sha256": record.get("output_sha256", ""),
                     "seconds": record.get("seconds", "")})
    ledger = _round4_read_jsonl(Path(budget_dir) / "images.jsonl") if budget_dir else []
    summary = {"generated_at": now_stamp(), "budget_dir": budget_dir, "limit": limit,
               "used": state["used"], "remaining": state["remaining"], "runs": rows}
    write_json(round_dir / "calls-summary.json", summary)
    if budget_dir:
        os.makedirs(budget_dir, exist_ok=True)
        with (Path(budget_dir) / "calls.jsonl").open("w", encoding="utf-8") as handle:
            for row in ledger:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("预算 %d/%d（剩余 %d）" % (state["used"], limit, state["remaining"]))
    for row in rows:
        print("  %-24s %-18s %-16s %s" % (row["run_id"], row["status"], row["gate"],
                                          str(row["output_sha256"])[:12]))
    return 0


def round4_crop_single(args):
    """对**单张已有输出**重画裁剪（人工定框），并写 crop-review.json。

    第四轮 P0.1：第三轮 `eyes3-P2-sheya-style-a` 的 variant/r1 眼带大半落在水瓶上，
    那条裁剪不能支持可靠的眼部分数 —— 用权威 `selected_output` 重画，不重新生成旧图。

    * 机器（YuNet/Haar）只写 `detected` 与 `detection_method`，**不写进人工字段**；
    * `human_checked` / `contains_target_face` / `eye_readable` 只能由调用方显式给出
      （看不清就写 `NA`，不许用背景代评分）；
    * 输出落在 `--out/crops/<label>_*.png`，并把条目合并进 `--out/crop-review.json`。
    """
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise SystemExit("找不到图片：" + str(image_path))
    out = Path(args.out).resolve()
    (out / "crops").mkdir(parents=True, exist_ok=True)
    image = read_image(image_path)
    head_override = [float(v) for v in str(args.head_box).split(",")] if args.head_box else None
    eye_override = [float(v) for v in str(args.eye_box).split(",")] if args.eye_box else None
    auto_box, auto_method = head_box_auto(image)
    head_box = head_override or list(auto_box)
    plausible, reason = head_box_plausible(head_box, image.size)
    band_info = eye_band(head_box)
    eye_box = eye_override or (band_info["band"] if band_info else head_box)
    px_head = _norm_box_to_px(head_box, image.size)
    px_eye = _norm_box_to_px(eye_box, image.size)
    head_clean = image.crop(px_head)
    eye_clean = image.crop(px_eye)
    label = str(args.label or image_path.stem)
    head_path = out / "crops" / (label + "_head_clean.png")
    eye_path = out / "crops" / (label + "_eye_clean.png")
    over_path = out / "crops" / (label + "_head_boxed.jpg")
    head_clean.save(head_path)
    eye_clean.save(eye_path)
    over = head_clean.copy()
    draw = ImageDraw.Draw(over)
    lx0, ly0 = px_eye[0] - px_head[0], px_eye[1] - px_head[1]
    draw.rectangle([lx0, ly0, lx0 + eye_clean.width, ly0 + eye_clean.height],
                   outline=(0, 200, 0), width=3)
    draw.text((4, 4), "%s %s" % ("MANUAL" if head_override else auto_method, label), fill=(0, 200, 0))
    over.save(over_path, quality=92)
    entry = {
        "label": label,
        "round": str(getattr(args, "round", "") or ""),
        "experiment": str(getattr(args, "experiment", "") or ""),
        "image": str(image_path),
        "output_sha256": sha256_of(image_path),
        "image_size": list(image.size),
        "head_box_normalized": [round(float(v), 5) for v in head_box],
        "head_box_pixels": list(px_head),
        "eye_box_normalized": [round(float(v), 5) for v in eye_box],
        "eye_box_pixels": list(px_eye),
        "eye_box_source": "MANUAL" if eye_override else ("EYE_BAND_FROM_HEAD" if band_info else "HEAD"),
        "eye_band_clipped_to_head_box": bool(band_info and band_info["clipped"]),
        "head_box_auto_normalized": [round(float(v), 5) for v in auto_box],
        "detection_method": auto_method,
        "detected": auto_method in ("YUNET", "AUTO"),
        "auto_box_plausible": bool(plausible),
        "auto_box_reason": reason,
        "box_pixels": list(px_head),
        "box_normalized": [round(float(v), 5) for v in head_box],
        "human_checked": (str(args.human_checked) if args.human_checked else None),
        "contains_target_face": (str(args.contains_target_face) if args.contains_target_face else None),
        "eye_readable": (str(args.eye_readable) if args.eye_readable else None),
        "reviewer": str(args.reviewer or "agent"),
        "reason": str(args.reason or ""),
        "head_clean": str(head_path), "eye_clean": str(eye_path), "head_boxed": str(over_path),
        "head_crop_size": list(head_clean.size), "eye_crop_size": list(eye_clean.size),
        "superseded_crop": str(args.supersedes or ""),
        "note": "人工定框重画；机器只给 detected，不写人工确认字段。",
    }
    review_path = out / "crop-review.json"
    review = record_of(review_path) if review_path.is_file() else {
        "generated_at": now_stamp(),
        "reviewer": {"human_checked": str(args.reviewer or "agent")},
        "note": "每条的 human_checked / contains_target_face / eye_readable 由看图者填写；"
                "机器检出不写成人工确认。",
        "crops": []}
    review["crops"] = [item for item in (review.get("crops") or []) if item.get("label") != label]
    review["crops"].append(entry)
    review["updated_at"] = now_stamp()
    write_json(review_path, review)
    print("裁剪 %s → %s" % (label, eye_path))
    print("  头部框 %s（像素 %s）| 眼带 %s（像素 %s）| 方法 %s | 自动框 %s"
          % (entry["head_box_normalized"], entry["head_box_pixels"], entry["eye_box_normalized"],
             entry["eye_box_pixels"], auto_method, plausible))
    return 0


def round4_verify_inputs(args):
    """P0.1：核验冻结输入与旧证据一致，并解析**真实源图**（不因 original=null 另跑分析）。"""
    round_dir = _round4_dir(args.round)
    source_round = Path(args.source_round).resolve() if args.source_round else \
        (round_dir.parent / "eye-face-hair-round3-20261001")
    evidence_path = round_dir / "REVIEW-EVIDENCE.json"
    evidence = record_of(evidence_path)
    declared = record_of(round_dir / "experiment-plan.json")
    inputs = {str(item.get("slot")): item for item in (declared.get("inputs") or [])}
    old_inputs = {str(item.get("slot")): item for item in (evidence.get("inputs") or [])}
    rows, problems = [], []
    for slot, item in sorted(inputs.items()):
        old = old_inputs.get(slot) or {}
        facts = {"slot": slot}
        for role, relative, old_key in (("analysis", item.get("analysis"), "analysis_source"),
                                        ("base", item.get("base"), "base_source")):
            local = (round_dir / str(relative)).resolve()
            original = Path(str(old.get(old_key) or ""))
            facts[role] = {
                "local": str(local), "local_sha256": digest(local) if local.is_file() else "",
                "recorded_source": str(original), "recorded_exists": original.is_file(),
                "recorded_sha256": digest(original) if original.is_file() else "",
                "declared_sha256": str(item.get("%s_sha256" % role) or ""),
            }
            facts[role]["identical_to_recorded"] = bool(
                facts[role]["local_sha256"] and facts[role]["local_sha256"] == facts[role]["recorded_sha256"])
            facts[role]["matches_declaration"] = bool(
                facts[role]["local_sha256"] and facts[role]["local_sha256"] == facts[role]["declared_sha256"])
            if not facts[role]["identical_to_recorded"]:
                problems.append("%s/%s 与旧证据记录不一致：%s" % (slot, role, str(original)))
            # 旧 request.json 的 source 字段（冻结输入里的源图解析路径）
            manifest = record_of(Path(str(old.get("base_source") or "")).parent.parent / "request.json")
            if manifest:
                facts[role]["old_manifest_source"] = str(manifest.get("source") or "")
        analysis = record_of((round_dir / str(item.get("analysis"))).resolve())
        facts["analysis_fields"] = {
            "source_image_path": str(analysis.get("source_image_path") or ""),
            "original": analysis.get("original"),
            "gpt_image_prompt_chars": len(str(analysis.get("gpt_image_prompt") or "")),
            "english_description_chars": len(str(analysis.get("english_description") or "")),
        }
        facts["original_source"] = {
            "local": str((round_dir / str(item.get("original_source"))).resolve()),
            "sha256": digest((round_dir / str(item.get("original_source"))).resolve()),
            "from": str(item.get("original_source_from") or ""),
        }
        facts["original_source"]["matches_source_image_path"] = bool(
            facts["analysis_fields"]["source_image_path"]
            and Path(facts["analysis_fields"]["source_image_path"]).is_file()
            and digest(Path(facts["analysis_fields"]["source_image_path"]))
            == facts["original_source"]["sha256"])
        rows.append(facts)
    manifest = record_of(round_dir / "input-verification.json")
    if not manifest and source_round.is_dir():
        pass
    report = {"generated_at": now_stamp(), "scope": "P0.1 冻结输入核验",
              "rule": "源图路径由旧 request.json 的 source 或分析 JSON 的实际字段解析；"
                      "original=null 不构成重新分析的理由。",
              "evidence": str(evidence_path), "rows": rows, "problems": problems,
              "verdict": "verified" if not problems else "mismatch"}
    write_json(round_dir / "input-verification.json", report)
    for fact in rows:
        print("slot %s：analysis %s / base %s（与旧证据逐字相同=%s）" % (
            fact["slot"], str(fact["analysis"]["local_sha256"])[:12], str(fact["base"]["local_sha256"])[:12],
            fact["analysis"]["identical_to_recorded"] and fact["base"]["identical_to_recorded"]))
        print("   源图 %s（analysis.source_image_path 命中=%s）" % (
            fact["original_source"]["local"], fact["original_source"]["matches_source_image_path"]))
    for text in problems:
        print("  ! " + text)
    return 1 if problems else 0


def round4_replay_gate(args):
    """P0.1：用**当前**纯判定函数离线重算旧产物的门禁，旧状态/新状态/代码版本分列。

    只读：不重跑审计、不发新图、不覆盖旧证据。
    """
    from utils.gate_status import evaluate_final_gate
    target = Path(args.source_round).resolve() if args.source_round else _round4_dir(args.round)
    out = Path(args.out).resolve() if args.out else _round4_dir(args.round)
    records = []
    for manifest_path in sorted(target.rglob("request.json")):
        if any(part.startswith("pre-rebind") for part in manifest_path.parts):
            continue
        manifest = record_of(manifest_path)
        if not manifest.get("final_gate") and not manifest.get("selected_output"):
            continue
        records.append((manifest_path, manifest))
    rows, changed = [], []
    from utils.refine_quality import should_refine_quality
    for manifest_path, manifest in records:
        selected = str(manifest.get("selected_output") or "")
        # 忠实重现旧运行时的输入：`--audit-only` 下旧代码会把「质量审计仍有高置信缺陷、
        # 未自动修订」写进 quality_audit_error。重算时若不还原这一项，就会把上游判红
        # 当成不存在，得出「旧 review_required 其实是 complete」的假放宽。
        quality_audit_error = str((manifest.get("quality_refine") or {}).get("audit_error") or "")
        if manifest.get("audit_only") and should_refine_quality(
                (manifest.get("quality_refine") or {}).get("before") or {}):
            quality_audit_error = quality_audit_error or "审计模式：质量审计仍有高置信缺陷，未自动修订"
        gate = evaluate_final_gate(
            final_image=selected,
            identity=(manifest.get("identity") or {}).get("final"),
            anatomy=manifest.get("anatomy_review"),
            quality=manifest.get("quality_refine"),
            final_review=manifest.get("final_review"),
            identity_action=str((manifest.get("identity") or {}).get("action") or ""),
            quality_audit_error=quality_audit_error)
        old = manifest.get("final_gate") or {}
        row = {
            "manifest": str(manifest_path),
            "relative": str(manifest_path.parent.relative_to(target)),
            "selected_output": selected,
            "output_sha256": digest(selected) if selected and Path(selected).is_file() else "",
            "old_status": str(old.get("status") or manifest.get("status") or ""),
            "old_text": str(old.get("text") or "")[:400],
            "new_status": gate["status"],
            "new_text": gate["text"][:400],
            "new_audits": gate["audits"],
            "status_changed": str(old.get("status") or "") != gate["status"],
            "missing_audits": [item["audit"] for item in gate["audits"]
                               if item["verdict"] not in ("pass", "not_run")],
        }
        rows.append(row)
        if row["status_changed"]:
            changed.append(row)
    report = {"generated_at": now_stamp(), "scope": "P0.1 旧产物离线门禁重算",
              "target": str(target), "evaluator": "utils/gate_status.evaluate_final_gate（当前版本）",
              "code_versions": code_versions(),
              "counting": {"manifests": len(rows),
                           "status_changed": len(changed),
                           "old_complete_to_new_review": sum(
                               1 for row in rows if row["old_status"] == "complete"
                               and row["new_status"] != "complete"),
                           "old_review_stays_review": sum(
                               1 for row in rows if row["old_status"] != "complete"
                               and row["new_status"] != "complete")},
              "rows": rows,
              "note": "旧 gate 文本与旧产物保持原样；本文件只增加一列「当前代码重算」的结果。"}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "old-gate-recompute.json", report)
    print("离线重算 %d 份旧 manifest：%d 份状态变化" % (len(rows), len(changed)))
    for row in changed:
        print("  %-46s %s → %s" % (row["relative"], row["old_status"], row["new_status"]))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default=str(DEFAULT_WORK), help="Dedicated evaluation directory")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("init", help="Snapshot enabled styles and reference hashes")
    setup.add_argument("--styles", default=str(ROOT / "conf" / "config-styles.json"))
    collect = sub.add_parser("refresh", help="Index completed checkpoint stages without altering them")
    collect.add_argument("--runs-root", nargs="+", required=True)
    register = sub.add_parser("add", help="Register a Gemini direct image or another existing candidate")
    register.add_argument("--style", required=True)
    register.add_argument("--strategy", required=True, help="For example gemini_direct")
    register.add_argument("--image", required=True)
    register.add_argument("--run-id", required=True)
    register.add_argument("--source", default="", help="Optional content source image")
    bulk = sub.add_parser("import-csv", help="Register many Gemini direct images from a CSV")
    bulk.add_argument("--csv", required=True)
    locate = sub.add_parser("box", help="Correct a crop using normalized x0,y0,x1,y1")
    locate.add_argument("--style", required=True)
    locate.add_argument("--run-id", default="")
    locate.add_argument("--stage", required=True,
                        choices=("reference", "first", "pipeline", "quality", "identity",
                                 "hands", "final_review", "publish", "candidate"))
    locate.add_argument("--bounds", required=True, help="Normalized x0,y0,x1,y1")
    sub.add_parser("sheet", help="Build context/head comparison sheets and review.csv")
    sub.add_parser("metrics", help="Run existing local ResNet diagnostics on checked crops")
    sub.add_parser("summary", help="Summarize manually checked 0–4 scores")
    experiment = sub.add_parser("exp-build", help="Build reference comparison sheets for data/exp")
    experiment.add_argument("--manifest", default=str(ROOT / "data" / "exp" / "manifest.json"))
    experiment.add_argument("--state", default=str(ROOT / "data" / "exp" / "runs" / "state.json"))
    sub.add_parser("exp-metrics", help="Measure global style diagnostics for data/exp")
    sub.add_parser("exp-heads", help="Build marked head enlargements for data/exp")
    sub.add_parser("exp-index", help="Write a linked 84-case artifact/score index")
    bind = sub.add_parser("round-bind", help="Freeze a previous round and resolve authoritative outputs")
    bind.add_argument("--round", required=True, help="Round directory, e.g. data/exp/eye-face-hair-round2-20260930")
    bind.add_argument("--out", required=True, help="Where the corrected evidence is written")
    bind.add_argument("--style", action="append", default=[], help="Limit to one style (repeatable)")
    compare = sub.add_parser("round-compare", help="Build full-figure comparison sheets from authoritative binding")
    compare.add_argument("--binding", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--style", action="append", default=[])
    compare.add_argument("--cell", type=int, default=460)
    crops = sub.add_parser("round-crops", help="Cut full-resolution eye crops and record detector vs human checks")
    crops.add_argument("--binding", required=True)
    crops.add_argument("--out", required=True)
    crops.add_argument("--manual-box", default="", help="JSON file: {style/slot/arm: [x0,y0,x1,y1]} normalized")
    crops.add_argument("--style", action="append", default=[])
    review = sub.add_parser("round-review", help="Build the per-arm review table (scores + gates) for manual filling")
    review.add_argument("--binding", required=True)
    review.add_argument("--out", required=True)
    review.add_argument("--crops", default="", help="crop-review.json to carry human crop verdicts into the table")
    review.add_argument("--from-round2", default="", help="Previous round review.csv, imported as reference only")
    ledger = sub.add_parser("round-calls", help="Build a deduplicated image/audit call ledger with cost evidence")
    ledger.add_argument("--round", required=True)
    ledger.add_argument("--out", required=True)
    ledger.add_argument("--estimate", default="", help="Optional existing cost estimate JSON")
    prep = sub.add_parser("round-prep", help="Freeze a round's experiment inputs and emit the run plan")
    prep.add_argument("--round", required=True)
    prep.add_argument("--plan", default="", help="Optional plan JSON; omit to use the bundled round-3 plan")
    prep.add_argument("--out", default="", help="Where inputs/ and run-plan.json are written (default: round dir)")
    prep.add_argument("--binding", default="", help="authoritative-binding.json to resolve bases from")
    prep.add_argument("--dump-plan", action="store_true", help="Also write the effective plan next to run-plan.json")
    run_arm = sub.add_parser("round-run", help="Run one experiment arm through the audit-only CLI")
    run_arm.add_argument("--round", required=True)
    run_arm.add_argument("--plan", required=True)
    run_arm.add_argument("--experiment", required=True)
    run_arm.add_argument("--slot", required=True, choices=list(STYLE_SLOTS))
    run_arm.add_argument("--arm", required=True)
    run_arm.add_argument("--repeat", type=int, default=1)
    run_arm.add_argument("--dry-run", action="store_true")
    after = sub.add_parser("round-final-audit", help="Read-only final review bound to a produced image")
    after.add_argument("--round", required=True)
    after.add_argument("--plan", required=True)
    after.add_argument("--experiment", required=True)
    after.add_argument("--slot", required=True, choices=list(STYLE_SLOTS))
    after.add_argument("--arm", required=True)
    after.add_argument("--repeat", type=int, default=1)
    rebind = sub.add_parser("round-rebind", help="Re-audit an existing paid output without regenerating it")
    rebind.add_argument("--round", required=True)
    rebind.add_argument("--run-dir", required=True, help="The repeat-N directory holding request.json")
    diffcheck = sub.add_parser("round-diffcheck", help="Verify real requests against the declared differences")
    diffcheck.add_argument("--round", required=True)
    diffcheck.add_argument("--plan", required=True)
    diffcheck.add_argument("--out", default="", help="Where to write request-diff-report.json (default: round dir)")
    diffcheck.add_argument("--strict", action="store_true", help="Fail when no call records exist at all")
    verdict = sub.add_parser("round-verdict", help="Apply the protocol success conditions to the produced arms")
    verdict.add_argument("--round", required=True)
    verdict.add_argument("--plan", required=True)
    verdict.add_argument("--scores", default="", help="scores.json with per-run human scores and kept-flags")
    verdict.add_argument("--out", default="")
    r4_install = sub.add_parser("round4-install-assets",
                                help="第四轮：把 prompt-assets 逐字安装到 prompts 目录")
    r4_install.add_argument("--round", default="")
    r4_install.add_argument("--force", action="store_true", help="目标内容不同时也覆盖")
    r4_plan = sub.add_parser("round4-build-plan",
                             help="第四轮：声明式清单 → 可执行 run-plan.json（零图片调用）")
    r4_plan.add_argument("--round", default="")
    r4_plan.add_argument("--plan", default="", help="声明式 experiment-plan.json（默认取本轮目录）")
    r4_plan.add_argument("--out", default="")
    r4_pre = sub.add_parser("round4-preflight",
                            help="第四轮：假后端跑完整组装路径并截获发送参数（零 HTTP）")
    r4_pre.add_argument("--round", default="")
    r4_pre.add_argument("--plan", default="")
    r4_pre.add_argument("--out", default="")
    r4_pre.add_argument("--run", default="", help="只预检 run_id 含该子串的那一条")
    r4_run = sub.add_parser("round4-run", help="第四轮：按预算 wrapper 执行一条计划 run（付费）")
    r4_run.add_argument("--round", default="")
    r4_run.add_argument("--plan", default="")
    r4_run.add_argument("--run-id", required=True)
    r4_run.add_argument("--dry-run", action="store_true")
    r4_state = sub.add_parser("round4-state", help="第四轮：读预算账本与逐 run 记录")
    r4_state.add_argument("--round", default="")
    r4_state.add_argument("--plan", default="")
    r4_crop = sub.add_parser("round4-crop-single",
                             help="第四轮 P0.1：对单张已有输出人工定框重画裁剪并写 crop-review.json")
    r4_crop.add_argument("--image", required=True)
    r4_crop.add_argument("--out", required=True)
    r4_crop.add_argument("--label", default="")
    r4_crop.add_argument("--head-box", default="", help="归一化 x0,y0,x1,y1（人工定框）")
    r4_crop.add_argument("--eye-box", default="", help="归一化 x0,y0,x1,y1（不给则按头部框推眼带）")
    r4_crop.add_argument("--human-checked", default="")
    r4_crop.add_argument("--contains-target-face", default="")
    r4_crop.add_argument("--eye-readable", default="", help="yes / no / NA")
    r4_crop.add_argument("--reviewer", default="")
    r4_crop.add_argument("--reason", default="")
    r4_crop.add_argument("--supersedes", default="", help="被这次重画替代的旧裁剪路径")
    r4_crop.add_argument("--experiment", default="")
    r4_crop.add_argument("--round-label", default="")
    r4_verify = sub.add_parser("round4-verify-inputs",
                               help="第四轮 P0.1：核验冻结输入与旧证据一致并解析真实源图")
    r4_verify.add_argument("--round", default="")
    r4_verify.add_argument("--source-round", default="")
    r4_replay = sub.add_parser("round4-replay-gate",
                               help="第四轮 P0.1：用当前判定函数离线重算旧产物门禁（只读）")
    r4_replay.add_argument("--round", default="")
    r4_replay.add_argument("--source-round", default="")
    r4_replay.add_argument("--out", default="")
    args = parser.parse_args()
    return {"init": init, "refresh": refresh, "add": add, "import-csv": import_csv,
     "box": box, "sheet": sheet,
     "metrics": metrics, "summary": summary, "exp-build": exp_build,
     "exp-metrics": exp_metrics, "exp-heads": exp_heads,
     "exp-index": exp_index, "round-bind": round_bind, "round-calls": round_calls,
     "round-compare": round_compare, "round-crops": round_crops,
     "round-review": round_review, "round-prep": round_prep, "round-run": round_run,
     "round-rebind": round_rebind, "round-diffcheck": round_diffcheck,
     "round-verdict": round_verdict,
     "round4-install-assets": round4_install_assets,
     "round4-build-plan": round4_build_plan,
     "round4-preflight": round4_preflight,
     "round4-run": round4_run,
     "round4-crop-single": round4_crop_single,
     "round4-verify-inputs": round4_verify_inputs,
     "round4-replay-gate": round4_replay_gate,
     "round4-state": round4_state}[args.command](args)


if __name__ == "__main__":
    # 子命令的返回码要真的变成进程退出码（`--strict` 非零退出靠这一行生效）
    sys.exit(main() or 0)
