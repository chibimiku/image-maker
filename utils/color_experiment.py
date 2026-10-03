"""Reusable isolated color-profile pilot: snapshots, cached calls and visual review.

This is an experimental consumer of authored profiles, not a production GUI hook.
Long generation/selection/review prompts live under prompts/color-knowledge/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from urllib.parse import urlparse

from utils.atomic_io import write_json_atomic
from utils.prompt_loader import read_prompt_file

BASE = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = spec.get("profiles", [])
    ids = [p["id"] for p in profiles]
    if len(ids) != len(set(ids)) or "baseline" not in ids:
        raise ValueError("Profiles need unique IDs and baseline")
    if not spec.get("subject") or not spec.get("rendering"):
        raise ValueError("Missing subject/rendering")
    if not 1 <= int(spec.get("repeats", 2)) <= 4:
        raise ValueError("Pilot repeats must be 1..4")
    for profile in profiles:
        clauses = profile.get("clauses", [])
        if len(clauses) > 6 or sum(map(len, clauses)) > 1400:
            raise ValueError("Color clauses exceed pilot budget")
    return spec


def build_request(spec, profile, model=None):
    prompt = spec["subject"] + "\n\n" + spec["rendering"]
    if profile.get("clauses"):
        prompt += "\n\nCOLOR PLAN:\n- " + "\n- ".join(profile["clauses"])
    return {"prompt": prompt, "model": model or spec["model"],
            "resolution": spec["resolution"], "aspect_ratio": spec["aspect_ratio"],
            "api_type": spec["api_type"], "image_paths": [], "face_quality_boost": False}


def parse_object(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected nonempty JSON object")
    return value


def preflight(spec, model=None):
    from modules.others.api_backend import get_api_config
    cfg = get_api_config(api_type=spec["api_type"])
    if not cfg.get("api_key"):
        raise ValueError("Image API key unavailable")
    if urlparse(cfg["base_url"]).hostname != "new.aigc2d.com":
        raise ValueError("Pilot requires the authorized new.aigc2d host")
    selected = model or spec["model"]
    if not selected.startswith("gemini-"):
        raise ValueError("This pilot only uses Gemini models")
    return {"host": "new.aigc2d.com", "api_type": spec["api_type"], "model": selected,
            "resolution": spec["resolution"], "aspect_ratio": spec["aspect_ratio"],
            "requests": len(spec["profiles"]) * spec["repeats"],
            "key_source": cfg.get("_api_key_source"), "timeout": cfg.get("timeout"),
            "backend_max_retries": cfg.get("max_retries")}


def run_pilot(spec, out_dir, *, model=None, dry_run=False, retry_failed=False, log=print):
    out = Path(out_dir).resolve()
    if not out.is_relative_to(BASE / "data" / "test-result"):
        raise ValueError("Pilot output must be under data/test-result")
    config = preflight(spec, model)
    log(json.dumps(config, ensure_ascii=False))
    if dry_run:
        return config
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "experiment.json"
    protocol = {"spec": spec, "api": config}
    fingerprint = digest(protocol)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_hash"] != fingerprint:
            raise ValueError("Protocol changed; use a new output directory")
    else:
        manifest = {"protocol_hash": fingerprint, "protocol": protocol, "samples": [],
                    "seed": None, "limitations": ["Independent random draws; no paired seed", "No repaint or tone calibration", "No fixed style-reference image"]}
        write_json_atomic(str(manifest_path), manifest)
    # Interleave baseline and treatments across repeats to reduce order effects.
    for repeat in range(1, spec["repeats"] + 1):
        for profile in spec["profiles"]:
            sample_id = f"{profile['id']}-{repeat}"
            previous = next((s for s in manifest["samples"] if s["id"] == sample_id), None)
            if previous:
                if previous["status"] == "success" and all(Path(p).is_file() for p in previous["outputs"]):
                    log(f"[cached] {sample_id}")
                    continue
                if not retry_failed:
                    log(f"[not-retried] {sample_id}: {previous['status']}")
                    continue
                manifest.setdefault("previous_attempts", []).append(previous)
                manifest["samples"].remove(previous)
            request = build_request(spec, profile, config["model"])
            sample = {"id": sample_id, "profile_id": profile["id"], "repeat": repeat,
                      "request": request, "request_hash": digest(request), "status": "running", "outputs": []}
            manifest["samples"].append(sample)
            write_json_atomic(str(manifest_path), manifest)
            folder = out / sample_id
            folder.mkdir(exist_ok=True)
            write_json_atomic(str(folder / "request.json"), request)
            (folder / "prompt.txt").write_text(request["prompt"], encoding="utf-8")
            log(f"[generate] {sample_id}")
            started = time.perf_counter()
            try:
                from modules.others.api_backend import generate_image_aigc2d
                paths = generate_image_aigc2d(**request, save_sub_dir=str(folder),
                                               file_prefix=f"test-{sample_id}", log_callback=log)
                from PIL import Image
                for path in paths:
                    with Image.open(path) as image:
                        image.verify()
                sample["outputs"] = [str(Path(p).resolve()) for p in paths]
                sample["status"] = "success" if paths else "failed"
                if not paths:
                    sample["error"] = "Backend returned no image; see masked backend log"
            except Exception as exc:
                sample["status"] = "failed"
                sample["error"] = type(exc).__name__  # no headers or response bodies in snapshots
            sample["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            write_json_atomic(str(manifest_path), manifest)
            log(f"[{sample['status']}] {sample_id}, {sample['elapsed_seconds']}s")
    return manifest


def select_profile(spec, out_dir, *, text_cfg=None):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = read_prompt_file("color-knowledge/profile-select-system.md")
    user = json.dumps({"subject": spec["subject"], "protected": spec["protected"],
                       "candidates": [{k: p.get(k) for k in ("id", "label", "intent", "clauses")} for p in spec["profiles"]]}, ensure_ascii=False)
    cfg = text_cfg or load_text_api_config()
    request = {"system": system, "user": user, "model": cfg["model"], "base_url": cfg["base_url"]}
    target = out / "auto-selection.json"
    signature = digest(request)
    if target.exists():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("request_hash") == signature:
            return cached
        raise ValueError("Selection request changed; use new output directory")
    write_json_atomic(str(out / "auto-selection.request.json"), request)
    raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user, max_tokens=2500)
    (out / "auto-selection.raw.txt").write_text(raw, encoding="utf-8")
    value = parse_object(raw)
    ids = {p["id"] for p in spec["profiles"]}
    if value.get("selected_profile_id") not in ids or any(a.get("profile_id") not in ids for a in value.get("alternatives", [])):
        raise ValueError("Selector returned unknown profile")
    if not value.get("reason") or not isinstance(value.get("confidence"), (float, int)) or not 0 <= value["confidence"] <= 1:
        raise ValueError("Invalid selection reason/confidence")
    result = {"request_hash": signature, "response": value}
    write_json_atomic(str(target), result)
    return result


def summarize_pilot(out_dir):
    from PIL import Image, ImageDraw, ImageOps
    import numpy as np
    out = Path(out_dir).resolve()
    manifest = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    samples = [s for s in manifest["samples"] if s["status"] == "success"]
    cells, metrics = [], []
    for sample in samples:
        for index, path in enumerate(sample["outputs"]):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                pixels = np.asarray(rgb, dtype=float) / 255
                mx, mn = pixels.max(axis=2), pixels.min(axis=2)
                sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
                # Display-space luma is descriptive, not a calibrated perceptual measurement.
                luma = pixels @ np.array([0.2126, 0.7152, 0.0722])
                metrics.append({"id": sample["id"], "path": path, "size": list(rgb.size),
                                "mean_display_luma": float(luma.mean()), "mean_hsv_saturation": float(sat.mean()),
                                "luma_p10_p50_p90": [float(v) for v in np.percentile(luma, [10, 50, 90])],
                                "near_white_fraction": float((mn > 0.97).mean()),
                                "note": "Global descriptive metrics, not beauty/focus/content scores"})
                cell = Image.new("RGB", (600, 450), "#f4f4f4")
                tile = ImageOps.contain(rgb, (600, 400))
                cell.paste(tile, ((600 - tile.width) // 2, 35 + (400 - tile.height) // 2))
                ImageDraw.Draw(cell).text((12, 10), f"{sample['id']} / {index+1}", fill="black")
                cells.append(cell)
    if not cells:
        raise ValueError("No successful pilot images")
    sheet = Image.new("RGB", (1800, 450 * ((len(cells)+2)//3)), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % 3) * 600, (i // 3) * 450))
    sheet.save(out / "comparison.jpg", quality=94)
    result = {"samples": metrics, "limitations": ["No semantic masks", "No color histogram equals aesthetic success", "Scanned swatches are approximate"]}
    write_json_atomic(str(out / "color-metrics.json"), result)
    write_gallery(out, manifest)
    return result


def write_gallery(out, manifest):
    """Human-readable plan cards and original samples; no invented winner."""
    from html import escape
    review_path = out / "visual-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    rows = {}
    for row in review.get("response", {}).get("images", []):
        sample_id = review["mapping"][row["id"]]["sample_id"]
        rows[sample_id] = row
    profiles = manifest["protocol"]["spec"]["profiles"]
    cards = []
    score_labels = {"color_hierarchy": "色彩主次", "subject_focus": "主体焦点",
                    "tonal_depth": "明暗彩度层次", "spring_mood": "春日氛围",
                    "content_fidelity": "内容保持", "rendering_quality": "画法质量"}
    for profile in profiles:
        swatches = "".join(f'<span class="swatch" style="background:{escape(c)}" title="{escape(c)}"></span>' for c in profile.get("palette_preview", []))
        clauses = "".join(f"<li>{escape(c)}</li>" for c in profile.get("clauses", []))
        sources = "、".join(f"PDF {s['pdf_page']} / 印刷 {s['printed_page']}" for s in profile.get("sources", []))
        photos = []
        for sample in manifest["samples"]:
            if sample["profile_id"] != profile["id"]:
                continue
            row = rows.get(sample["id"], {})
            details = "".join(f"<li>{escape(x)}</li>" for x in row.get("observations", []))
            issues = "".join(f"<li>{escape(x)}</li>" for x in row.get("content_issues", []))
            issues += "".join(f"<li>不确定：{escape(x)}</li>" for x in row.get("uncertain", []))
            scores = " / ".join(f"{score_labels.get(k, k)}: {v}" for k, v in row.get("scores", {}).items())
            for path in sample["outputs"]:
                relative = Path(path).relative_to(out).as_posix()
                photos.append(f'<figure><a href="{escape(relative)}"><img src="{escape(relative)}" alt="{escape(sample["id"])}"></a><figcaption>{escape(sample["id"])} · {sample["elapsed_seconds"]} 秒</figcaption><p class="scores">{escape(scores)}</p><ul>{details}</ul><ul class="issue">{issues}</ul></figure>')
        cards.append(f'<section><h2>{escape(profile["label"])}</h2><p>{escape(profile["intent"])}</p><div>{swatches}</div><p class="muted">色板为设计近似预览，非书中实测。来源：{escape(sources) or "无附加策略"}</p><details><summary>实际配色条款</summary><ul>{clauses}</ul></details><div class="photos">{"".join(photos)}</div></section>')
    selection_path = out / "auto-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")).get("response", {}) if selection_path.exists() else {}
    selection_text = escape(str(selection.get("selected_profile_id", "尚未执行"))) + "：" + escape(str(selection.get("reason", "")))
    comparison = escape(str(review.get("response", {}).get("comparison", "视觉评价尚未执行")))
    spec = manifest["protocol"]["spec"]
    summary = escape(f"{manifest['protocol']['api']['model']} / {spec['resolution']} / {spec['aspect_ratio']} · {len(profiles)}组，每组{spec['repeats']}张 · 无重绘与 tone")
    issue_count = sum(bool(row.get("content_issues")) for row in rows.values())
    maxed_count = sum(bool(row.get("scores")) and all(v == 4 for v in row["scores"].values()) for row in rows.values())
    caution = f"探索性样本，不能据此宣称稳定收益。模型评价发现{issue_count}张有内容问题；{maxed_count}张所有维度满分，需注意评分区分度。模型选择与评级均不等于统计结论。"
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>配色方案对照</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#26352e;background:#edf1ec;margin:0}}main{{max-width:1450px;margin:auto;padding:28px}}section{{background:white;border-radius:12px;padding:24px;margin:20px 0}}h1,h2{{line-height:1.3}}.muted,.scores{{color:#657269;font-size:14px}}.swatch{{display:inline-block;width:64px;height:40px;margin:4px;border-radius:6px;border:1px solid #ddd}}.photos{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}figure{{margin:16px 0}}img{{width:100%;border-radius:8px}}.issue{{color:#a12e37}}aside{{padding:16px;background:#fff2d7;border-radius:10px}}@media(max-width:800px){{.photos{{grid-template-columns:1fr}}}}</style><main><h1>配色方案对照 · {escape(spec["id"])}</h1><p>{summary}</p><aside>{caution}</aside><p><b>文字接口自动选择：</b>{selection_text}</p><p><b>匿名模型比较：</b>{comparison}</p><p><a href="comparison.jpg">六图总览</a> · <a href="experiment.json">请求与运行记录</a> · <a href="color-metrics.json">本地统计</a> · <a href="visual-review.json">匿名评价原文</a></p>{"".join(cards)}</main></html>'''
    (out / "gallery.html").write_text(html, encoding="utf-8")


def evaluate_pilot(out_dir, *, text_cfg=None):
    from PIL import Image, ImageOps
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    out = Path(out_dir)
    manifest = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    paths = [(s["id"], p) for s in manifest["samples"] if s["status"] == "success" for p in s["outputs"]]
    if not paths:
        raise ValueError("No images for evaluation")
    random.Random(261003).shuffle(paths)
    mapping, proxies = {}, []
    for i, (sample_id, path) in enumerate(paths, 1):
        alias = f"V{i:02d}"
        mapping[alias] = {"sample_id": sample_id, "path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        proxy = out / f"review-{alias}.jpg"
        with Image.open(path) as image:
            ImageOps.contain(image.convert("RGB"), (1400, 1400)).save(proxy, quality=92)
        proxies.append(str(proxy))
    system = read_prompt_file("color-knowledge/pilot-evaluate-system.md")
    user = json.dumps({"subject": manifest["protocol"]["spec"]["subject"], "image_order": list(mapping)}, ensure_ascii=False)
    cfg = text_cfg or load_text_api_config()
    request = {"system": system, "user": user, "model": cfg["model"], "base_url": cfg["base_url"], "images": mapping}
    signature = digest(request)
    target = out / "visual-review.json"
    if target.exists():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("request_hash") == signature:
            return cached
        raise ValueError("Evaluation inputs changed; preserve old evaluation and use a new run")
    write_json_atomic(str(out / "visual-review.request.json"), request)
    raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user,
                          max_tokens=6500, timeout=240, image_paths=proxies)
    (out / "visual-review.raw.txt").write_text(raw, encoding="utf-8")
    value = parse_object(raw)
    rows = value.get("images", [])
    ids = [r.get("id") for r in rows]
    dimensions = {"color_hierarchy", "subject_focus", "tonal_depth", "spring_mood", "content_fidelity", "rendering_quality"}
    if len(ids) != len(mapping) or set(ids) != set(mapping):
        raise ValueError("Review must cover each image exactly once")
    for row in rows:
        scores = row.get("scores", {})
        if set(scores) != dimensions or any(type(v) is not int or not 0 <= v <= 4 for v in scores.values()):
            raise ValueError("Invalid review scores")
    result = {"request_hash": signature, "mapping": mapping, "response": value,
              "status": "provisional_model_review_not_gate"}
    write_json_atomic(str(target), result)
    return result
