"""Reusable isolated color-profile pilot: snapshots, cached calls and visual review.

This is an experimental consumer of authored profiles, not a production GUI hook.
Long generation/selection/review prompts live under prompts/color-knowledge/.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
import time
from urllib.parse import urlparse

from utils.atomic_io import write_json_atomic
from utils.prompt_loader import read_prompt_file

BASE = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


BOOK_SPATIAL_GUIDE_ROLE = "spatial_locator_not_native_mask"
BOOK_ATTACHMENT_ROLES = ("content", "spatial_guide")
# 这些键只属于冻结计划与账本，绝不能传给后端函数（否则就是未知参数 TypeError）。
BOOK_REQUEST_META_KEYS = ("id", "group_id", "round", "attachments")


def _image_size(path):
    from PIL import Image
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def book_spatial_guide(spec):
    """独立位置图的显式角色与冻结元数据。

    位置图只是给模型的普通视觉指导：不是接口原生 mask、不是逐像素锁定。
    缺失、变化、尺寸不一致或与内容源角色冲突时一律抛错，绝不带着可疑附件发请求。
    """
    declared = spec.get("spatial_guide")
    if not declared:
        return None
    guide = dict(declared)
    path = BASE / guide["path"]
    if not path.is_file():
        raise ValueError("Spatial guide image is missing; stop before sending")
    if _sha256_file(path) != guide["sha256"]:
        raise ValueError("Spatial guide image hash mismatch; stop before sending")
    if guide.get("role") != BOOK_SPATIAL_GUIDE_ROLE:
        raise ValueError("Spatial guide must declare its experimental role explicitly")
    size = _image_size(path)
    if guide.get("canvas_size") and list(guide["canvas_size"]) != size:
        raise ValueError("Spatial guide canvas size mismatch")
    guide["canvas_size"] = size
    guide["absolute_path"] = str(path.resolve())
    manifest_path = BASE / guide["manifest"]
    if not manifest_path.is_file():
        raise ValueError("Spatial guide manifest is missing")
    if _sha256_file(manifest_path) != guide["manifest_sha256"]:
        raise ValueError("Spatial guide manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("locator_sha256") != guide["sha256"]:
        raise ValueError("Locator manifest does not describe the frozen guide image")
    if manifest.get("role") != guide["role"]:
        raise ValueError("Locator manifest role conflicts with the protocol")
    if list(manifest.get("canvas_size") or []) != size:
        raise ValueError("Locator manifest canvas size conflicts with the guide image")
    guide["boxes"] = manifest.get("boxes", [])
    guide["native_mask_support"] = manifest.get("native_mask_support", "not_confirmed")
    guide["scope_note"] = manifest.get("scope", "")
    instruction_path = BASE / "prompts" / guide["instruction"]
    if not instruction_path.is_file():
        raise ValueError("Spatial guide role instruction is missing")
    guide["instruction_sha256"] = _sha256_file(instruction_path)
    guide["instruction_text"] = read_prompt_file(guide["instruction"])
    return guide


def _book_attachment_specs(group, content_attachment, guide_attachment):
    entries = []
    if group.get("uses_content_reference"):
        entries.append(dict(content_attachment))
    if group.get("uses_spatial_guide"):
        entries.append(dict(guide_attachment))
    return [{**entry, "order": index} for index, entry in enumerate(entries, 1)]


def book_reference_images(request):
    """附件清单：路径 / hash / 尺寸 / 实际顺序 / 角色。旧协议保持原样（只有 content 角色）。"""
    if request.get("attachments"):
        return [{"path": e["path"], "sha256": e["sha256"], "role": e["role"],
                 "order": e["order"], "size": e.get("size")} for e in request["attachments"]]
    return [{"path": p, "sha256": _sha256_file(p), "role": "content"} for p in request["image_paths"]]


def book_attachment_paths(request):
    if request.get("attachments"):
        return [entry["path"] for entry in request["attachments"]]
    return list(request["image_paths"])


def book_ablation_plan(spec_path):
    """Freeze measured palettes and compiler output into a small generation study."""
    from utils.color_extract import compile_generation_palette
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    inherited_hash = None
    if spec.get("extends"):
        inherited_path = BASE / spec["extends"]
        inherited = json.loads(inherited_path.read_text(encoding="utf-8"))
        if inherited.get("extends"):
            raise ValueError("Book protocol supports only one inheritance level")
        inherited_hash = _sha256_file(inherited_path)
        spec = {**inherited, **spec}
    knowledge = json.loads((BASE / spec["knowledge"]).read_text(encoding="utf-8"))
    source = json.loads((BASE / spec["content_source"]).read_text(encoding="utf-8"))
    channel = source["channels"][spec["channel_source"]]
    if channel["image_paths"] or spec["image_max_retries"] != 0 or channel["face_quality_boost"]:
        raise ValueError("Book ablation must use no images, no retry and no face suffix")
    if not 1 <= spec["rounds"] <= 3:
        raise ValueError("Book ablation rounds must be 1..3")
    stem = source["subject"] + "\n\n" + source["rendering"]
    content_reference = None
    if spec.get("content_reference"):
        content_reference = dict(spec["content_reference"])
        reference_path = BASE / content_reference["path"]
        if _sha256_file(reference_path) != content_reference["sha256"]:
            raise ValueError("Content reference image hash mismatch")
        content_reference["absolute_path"] = str(reference_path.resolve())
    spatial_guide = book_spatial_guide(spec)
    if spatial_guide and content_reference:
        if spatial_guide["absolute_path"] == content_reference["absolute_path"] \
                or spatial_guide["sha256"] == content_reference["sha256"]:
            raise ValueError("Spatial guide and content source must be different files with different roles")
    if spatial_guide and not content_reference:
        raise ValueError("Spatial guide cannot replace the content source")
    content_attachment, guide_attachment = None, None
    if spatial_guide:
        content_attachment = {"role": "content", "path": content_reference["absolute_path"],
                              "sha256": content_reference["sha256"],
                              "size": _image_size(content_reference["absolute_path"]),
                              "declared_role": content_reference.get("role", "content_only")}
        guide_attachment = {"role": "spatial_guide", "path": spatial_guide["absolute_path"],
                            "sha256": spatial_guide["sha256"], "size": spatial_guide["canvas_size"],
                            "declared_role": spatial_guide["role"]}
    groups = []
    for scheme in spec["schemes"]:
        for variant, settings in spec["variants"].items():
            settings = dict(settings)
            uses_reference = settings.pop("uses_content_reference", False)
            uses_guide = settings.pop("uses_spatial_guide", False)
            if type(uses_guide) is not bool:
                raise ValueError("Spatial-guide flag must be an explicit boolean")
            if uses_guide and spatial_guide is None:
                raise ValueError("Spatial-guide variant requires a frozen, hashed locator")
            if uses_guide and not uses_reference:
                raise ValueError("Spatial-guide variant must keep the hashed content source")
            if type(uses_reference) is not bool or (uses_reference and content_reference is None):
                raise ValueError("Content-reference variant requires an explicit hashed source")
            selection = {"enabled": True, "prompt": stem, "style_reference_images": [],
                         "palette_id": scheme["palette_id"], "scene_kind": "outdoor",
                         "existing_regions": spec["existing_regions"],
                         "authorized_regions": spec["authorized_regions"],
                         "protected_regions": spec["protected_regions"],
                         "bindings": [{"region": b["region"], "role": b["role"],
                                       "colour_order": scheme[b["order_key"]]} for b in spec["bindings"]],
                         "area_mode": "base-accent", **settings}
            if spec.get("composition_guard"):
                selection["composition_guard"] = spec["composition_guard"]
            compiled = compile_generation_palette(knowledge, selection)
            palette = next(p for p in knowledge["palettes"] if p["id"] == scheme["palette_id"])
            groups.append({"id": scheme["id"] + "-" + variant, "palette_id": scheme["palette_id"],
                           "variant": variant, "label": palette["label"], "compiled": compiled,
                           "colours": palette["colours"], "is_control": False,
                           "prompt": compiled["prompt"] + "\n\nPRESERVE: " + source["protection_block"]})
            if content_reference:
                groups[-1]["uses_content_reference"] = uses_reference
            if spatial_guide:
                groups[-1]["uses_spatial_guide"] = uses_guide
            attachment_specs = _book_attachment_specs(groups[-1], content_attachment, guide_attachment) \
                if spatial_guide else []
            if attachment_specs:
                groups[-1]["attachments"] = attachment_specs
            prompt_prefix = ""
            if uses_guide:
                prompt_prefix += spatial_guide["instruction_text"] + "\n\n"
            if uses_reference:
                prompt_prefix += read_prompt_file(spec["content_reference_instruction"]) + "\n\n"
            if prompt_prefix:
                groups[-1]["prompt"] = prompt_prefix + groups[-1]["prompt"]
    groups.append({"id": "B0", "variant": "control", "is_control": True,
                   "prompt": stem + "\n\nPRESERVE: " + source["protection_block"]})
    if spec.get("composition_guard"):
        groups[-1]["prompt"] += "\n\n" + read_prompt_file("color-knowledge/book-single-scene-guard-v1.md")
    if spec.get("include_control", True) is False:
        groups.pop()
    requests = []
    rng = random.Random(spec["schedule_seed"])
    for repeat in range(1, spec["rounds"] + 1):
        order = list(groups)
        rng.shuffle(order)
        for group in order:
            request = {"id": group["id"] + "-r" + str(repeat), "group_id": group["id"], "round": repeat,
                       "prompt": group["prompt"], "model": channel["model"],
                       "api_type": channel["api_type"], "resolution": channel["resolution"],
                       "aspect_ratio": channel["aspect_ratio"], "face_quality_boost": False,
                       "image_paths": [content_reference["absolute_path"]] if group.get("uses_content_reference") else []}
            if spatial_guide:
                request["attachments"] = [dict(entry) for entry in group.get("attachments", [])]
                paths = [entry["path"] for entry in request["attachments"]]
                if [e["order"] for e in request["attachments"]] != list(range(1, len(paths) + 1)):
                    raise ValueError("Attachment order must be recorded explicitly")
                roles = [e["role"] for e in request["attachments"]]
                if len(set(roles)) != len(roles) or any(r not in BOOK_ATTACHMENT_ROLES for r in roles):
                    raise ValueError("Attachment roles must be unique and declared")
                if paths[:len(request["image_paths"])] != list(request["image_paths"]):
                    raise ValueError("The content source must stay the first attachment, with the guide after it")
                # 真正发给后端的就是这条有序路径列表：内容源在前，位置图在后。
                request["image_paths"] = paths
            requests.append(request)
    from utils.analysis_gpt_prompt import load_text_api_config
    cfg = load_text_api_config()
    plan = {"spec": spec, "groups": groups, "requests": requests,
            "knowledge_file_sha256": _sha256_file(BASE / spec["knowledge"]),
            "content_file_sha256": _sha256_file(BASE / spec["content_source"]),
            "review_model": cfg["model"], "review_base_url": cfg["base_url"],
            "review_system": read_prompt_file("color-knowledge/book-palette-ablation-review.md"),
            "protection_block": source["protection_block"], "planned_images": len(requests),
            "generation_validation": "not_started"}
    if inherited_hash:
        plan["inherited_spec_file_sha256"] = inherited_hash
    if spec.get("review_addendum"):
        addenda = spec["review_addendum"]
        for addendum in ([addenda] if isinstance(addenda, str) else list(addenda)):
            plan["review_system"] += "\n\n" + read_prompt_file(addendum)
        plan["content_constraints"] = stem + "\n\n" + read_prompt_file("color-knowledge/book-single-scene-guard-v1.md")
    if content_reference:
        plan["content_reference"] = content_reference
    if spatial_guide:
        plan["spatial_guide"] = {key: value for key, value in spatial_guide.items() if key != "instruction_text"}
        plan["attachment_roles"] = list(BOOK_ATTACHMENT_ROLES)
        plan["attachment_note"] = ("Guidance attachments are sent in the recorded order; the locator is a separate "
                                   "visual-guide role, never a second content source, candidate or native mask. "
                                   "No mask field is added to the request.")
    plan["plan_hash"] = digest(plan)
    return plan


def freeze_book_ablation(spec_path, out_dir):
    plan = book_ablation_plan(spec_path)
    out = _isolated_out(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "book-frozen.json"
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous["plan_hash"] != plan["plan_hash"]:
            raise ValueError("Frozen book ablation changed; use a new directory")
        return previous
    write_json_atomic(str(target), plan)
    return plan


def book_sample_attempts(folder):
    paths = [Path(folder) / "result.json"] + sorted(Path(folder).glob("attempts/*/result.json"))
    return [{**json.loads(p.read_text(encoding="utf-8")), "attempt_folder": str(p.parent)}
            for p in paths if p.exists()]


def book_retry_authorization(out):
    path = Path(out) / "retry-authorization.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sample_ids": [], "extra_reviews": 0}


def book_generation_body(request):
    """Mirror the existing backend body, including actual content attachments."""
    from modules.others.api_backend import to_base64_compressed
    parts = [{"text": request["prompt"]}]
    for entry in request.get("attachments") or []:
        path = entry["path"]
        if not Path(path).is_file():
            raise ValueError("Missing frozen " + str(entry.get("role")) + " attachment")
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError("Frozen " + str(entry.get("role")) + " attachment changed; stop before sending")
    for p in book_attachment_paths(request):
        if not Path(p).is_file():
            raise ValueError("Missing frozen content reference")
        mime, data = to_base64_compressed(p)
        parts.append({"inline_data": {"mime_type": mime, "data": data}})
    return {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"imageConfig": {"aspectRatio": request["aspect_ratio"], "imageSize": request["resolution"]}}}


def preview_book_ablation(spec_path, out_dir):
    """Materialize complete offline request bodies without reserving API calls."""
    plan = freeze_book_ablation(spec_path, out_dir)
    out = _isolated_out(out_dir)
    folder = out / "preview"
    folder.mkdir(exist_ok=True)
    for request in plan["requests"]:
        body = book_generation_body(request)
        snapshot = {**request, "body": body, "body_hash": digest(body), "sent": False,
                    "reference_images": book_reference_images(request)}
        target = folder / (request["id"] + ".request.json")
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != snapshot:
            raise ValueError("Frozen preview changed; use a new directory")
        write_json_atomic(str(target), snapshot)
    return {"plan_hash": plan["plan_hash"], "planned_images": len(plan["requests"]), "sent": False}


def run_book_ablation(spec_path, out_dir, *, dry_run=False, retry_failed=False, log=print):
    from modules.others.api_backend import generate_image_aigc2d, get_api_config
    from utils import send_budget
    plan = freeze_book_ablation(spec_path, out_dir)
    out = _isolated_out(out_dir)
    channel = get_api_config(api_type=plan["requests"][0]["api_type"])
    if not channel.get("api_key") or urlparse(channel["base_url"]).hostname != "new.aigc2d.com":
        raise ValueError("Image endpoint or credentials unavailable; no request sent")
    if dry_run:
        return {"planned_images": plan["planned_images"], "plan_hash": plan["plan_hash"], "dry_run": True}
    if retry_failed and not (out / "retry-authorization.json").exists():
        ids = [r["id"] for r in plan["requests"]
               if (history := book_sample_attempts(out / "samples" / r["id"])) and history[-1]["status"] == "failed"]
        write_json_atomic(str(out / "retry-authorization.json"), {
            "plan_hash": plan["plan_hash"], "sample_ids": ids, "extra_reviews": len({r["group_id"] for r in plan["requests"] if r["id"] in ids}),
            "authorization": "User explicitly authorized one unchanged retry of failed samples", "at": _now()})
    authorization = book_retry_authorization(out)
    os.environ["IMAGE_MAKER_IMAGE_MAX_RETRIES"] = "0"
    os.environ[send_budget.ENV_DIR] = str(out / "send-budget")
    os.environ[send_budget.ENV_LIMIT] = str(plan["planned_images"] + len(authorization["sample_ids"]))
    failures = 0
    for request in plan["requests"]:
        folder = out / "samples" / request["id"]
        if retry_failed:
            if request["id"] not in authorization["sample_ids"]:
                continue
            folder = folder / "attempts" / "attempt-02"
        target = folder / "result.json"
        if target.exists():
            log("[book] Already attempted, no resend: " + request["id"])
            continue
        folder.mkdir(parents=True, exist_ok=True)
        body = book_generation_body(request)
        write_json_atomic(str(folder / "request.json"), {**request, "body": body, "body_hash": digest(body)})
        reservation = send_budget.reserve_image_attempt(run_id=request["id"] + ("-attempt-02" if retry_failed else ""), operation=plan["spec"]["id"],
            model=request["model"], resolution=request["resolution"], prompt_sha256=_sha256_str(request["prompt"]),
            prompt_chars=len(request["prompt"]), aspect_ratio=request["aspect_ratio"],
            reference_images=book_reference_images(request))
        log("[book] Generate " + request["id"])
        result, paths, error = {}, [], ""
        try:
            result = generate_image_aigc2d(**{k: v for k, v in request.items() if k not in BOOK_REQUEST_META_KEYS},
                save_sub_dir=str(folder), file_prefix="book-" + request["id"], return_metadata=True, log_callback=log)
            raw = result.get("server_response_raw", {}) if isinstance(result, dict) else {}
            (folder / "response.raw.txt").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            paths = list(result.get("saved_files") or []) if isinstance(result, dict) else list(result or [])
            _verify_and_hash(paths)
        except Exception as exc:
            error = type(exc).__name__
            (folder / "response.error.txt").write_text(error, encoding="utf-8")
            paths = []
        status = "success" if paths else "failed"
        send_budget.settle_image_attempt(reservation, status=status,
            output_path=str(paths[0]) if paths else "", output_sha256=_sha256_file(paths[0]) if paths else "", error=error)
        write_json_atomic(str(target), {"id": request["id"], "group_id": request["group_id"], "round": request["round"],
            "status": status, "outputs": [str(Path(p).resolve()) for p in paths], "error": error,
            "output_hashes": _verify_and_hash(paths), "server_meta": _redact_server_meta(result), "at": _now()})
        log("[book] " + request["id"] + " " + status)
        failures = failures + 1 if not paths else 0
        if failures >= 2:
            log("[book] Two consecutive failures; stopped, no retry or replacement")
            break
    return book_ablation_report(out_dir)


def validate_book_ablation_review(value, ids, *, content_anchor=False):
    rows = value.get("images") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != len(ids) or {r.get("id") for r in rows} != set(ids):
        raise ValueError("Review image IDs must match exactly once")
    for row in rows:
        for key in ("colour_status", "region_status", "area_status"):
            if row.get(key) not in ("conform", "partial", "fail", "unverifiable"):
                raise ValueError("Invalid " + key)
        for key in ("protected_ok", "content_ok"):
            if row.get(key) is not None and type(row[key]) is not bool:
                raise ValueError("Uncertain protection/content cannot count as a pass")
            if key not in row:
                raise ValueError("Missing " + key)
        for key in ("extra_colour_areas", "protection_issues", "content_issues", "unverifiable_reason", "evidence"):
            if not isinstance(row.get(key), list) or any(not isinstance(v, str) for v in row[key]):
                raise ValueError("Review evidence must be string arrays")
        if row.get("aesthetic_score") is not None:
            raise ValueError("Aesthetics is not a palette metric")
        if content_anchor:
            count = row.get("visible_person_count")
            if "visible_person_count" not in row or (count is not None and (type(count) is not int or count < 0)):
                raise ValueError("Visible person count must be an integer or null")
            if "source_count_ok" not in row or (row["source_count_ok"] is not None and type(row["source_count_ok"]) is not bool):
                raise ValueError("Source count pass must be boolean or null")
            for key in ("source_layout_status", "grayscale_status"):
                if row.get(key) not in ("conform", "partial", "fail", "unverifiable"):
                    raise ValueError("Missing or invalid " + key)
            if not isinstance(row.get("grayscale_evidence"), list) or any(not isinstance(v, str) for v in row["grayscale_evidence"]):
                raise ValueError("Missing grayscale evidence")
    return value


def review_book_ablation(spec_path, out_dir, *, log=print):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    from utils import send_budget
    plan = freeze_book_ablation(spec_path, out_dir)
    out = _isolated_out(out_dir)
    cfg = load_text_api_config()
    if cfg["model"] != plan["review_model"] or cfg["base_url"] != plan["review_base_url"]:
        raise ValueError("Review endpoint/model changed after freezing")
    os.environ[send_budget.ENV_DIR] = str(out / "send-budget")
    os.environ[send_budget.ENV_TEXT_LIMIT] = str(plan["spec"]["review_max_calls"] + book_retry_authorization(out)["extra_reviews"])
    os.environ[send_budget.ENV_TEXT_PER_STAGE] = "1"
    vision = out / "vision"
    vision.mkdir(exist_ok=True)
    reviewed_ids = set()
    for p in vision.glob("*.json"):
        record = json.loads(p.read_text(encoding="utf-8"))
        if "response" in record and "mapping" in record:
            reviewed_ids.update(record["mapping"].values())
    for group in plan["groups"]:
        name = group["id"]
        if (vision / (name + ".json")).exists():
            name += "-retry-02"
        raw_path = vision / (name + ".raw.txt")
        target = vision / (name + ".json")
        if target.exists():
            continue
        samples, evidence, mapping = [], [], {}
        if group.get("uses_content_reference"):
            evidence.append(("content source, not a candidate", _reassess_proxy(out, plan["content_reference"]["absolute_path"])[0]))
        if group.get("uses_spatial_guide"):
            guide = plan["spatial_guide"]
            if _sha256_file(guide["absolute_path"]) != guide["sha256"]:
                raise ValueError("Frozen spatial guide changed before review; stop")
            evidence.append(("spatial locator: black-and-white locator, not a candidate, not a person, "
                             "not a palette and not a pixel-exact mask", guide["absolute_path"]))
        for request in plan["requests"]:
            history = book_sample_attempts(out / "samples" / request["id"])
            if request["group_id"] == group["id"] and history and request["id"] not in reviewed_ids:
                sample = history[-1]
                if sample["status"] == "success":
                    samples.append(sample)
        for index, sample in enumerate(samples, 1):
            alias = "Q%02d" % index
            mapping[alias] = sample["id"]
            full, detail = _reassess_proxy(out, sample["outputs"][0])
            evidence.append(("candidate %s full image" % alias, full))
            evidence.append(("candidate %s lower-body crop of the same candidate" % alias, detail))
            if plan["spec"].get("readonly_grayscale"):
                evidence.append(("candidate %s read-only grayscale of the same candidate" % alias,
                                 book_grayscale_proxy(out, sample["outputs"][0])))
        images = [path for _, path in evidence]
        if not samples:
            continue
        payload = {"image_order": list(mapping), "colours": group.get("colours", []),
            "bindings": (group.get("compiled") or {}).get("selection", {}).get("bindings", []),
            "is_control": group["is_control"], "protection_block": plan["protection_block"],
            "expected_area_hierarchy": "large base, medium auxiliary, small accents on existing sash and both shoe bows"}
        if "content_constraints" in plan:
            payload["content_constraints"] = plan["content_constraints"]
        if plan.get("content_reference"):
            if group.get("uses_spatial_guide"):
                payload["reference_role"] = ("first image is the content source; second image is the declared "
                                             "spatial locator (not a candidate); then each candidate full image, "
                                             "that candidate's lower-body crop and read-only grayscale in image_order")
            else:
                payload["reference_role"] = "first image is content source; then each candidate full/body crop pair" if group.get("uses_content_reference") else "no content reference supplied to generation or review"
            payload["source_subject_count"] = plan["content_reference"]["subject_count"]
        if plan.get("spatial_guide"):
            payload["image_roles"] = ["%d. %s" % (index, label) for index, (label, _) in enumerate(evidence, 1)]
            payload["spatial_guide_note"] = ("The locator is ordinary visual guidance sent alongside the content "
                                             "source. It is not a native API mask, not a pixel-exact boundary, not a "
                                             "candidate and not part of the output. Never treat it as a person or a "
                                             "second scene.")
        request = {"system": plan["review_system"], "user": json.dumps(payload, ensure_ascii=False),
                   "model": cfg["model"], "base_url": cfg["base_url"], "mapping": mapping,
                   "images": [{"path": str(p), "sha256": _sha256_file(p)} for p in images]}
        if plan.get("spatial_guide"):
            request["image_roles"] = [{"order": index, "role": label, "path": str(path), "sha256": _sha256_file(path)}
                                      for index, (label, path) in enumerate(evidence, 1)]
            request["declared_attachments"] = [dict(entry) for entry in group.get("attachments", [])]
        request_path = vision / (name + ".request.json")
        if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Review evidence changed after prior request")
        write_json_atomic(str(request_path), request)
        if raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8")
        else:
            reservation = send_budget.reserve_text_attempt(stage=name, purpose="book-palette-review")
            try:
                raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], request["system"], request["user"],
                    max_tokens=8000, timeout=300, image_paths=[str(p) for p in images])
                raw_path.write_text(raw, encoding="utf-8")
                send_budget.settle_text_attempt(reservation, status="success")
            except Exception as exc:
                send_budget.settle_text_attempt(reservation, status="failed", error=type(exc).__name__)
                (vision / (name + ".error.txt")).write_text(type(exc).__name__, encoding="utf-8")
                log("[book] Review failed; no automatic retry: " + name)
                continue
        try:
            value, note = parse_object_tolerant(raw)
            validate_book_ablation_review(value, list(mapping), content_anchor=bool(plan.get("content_reference")))
            write_json_atomic(str(target), {"response": value, "mapping": mapping, "request_hash": digest(request)})
            log("[book] Reviewed " + name)
        except Exception as exc:
            (vision / (name + ".validation-error.txt")).write_text(str(exc), encoding="utf-8")
            log("[book] Review retained for offline recovery: " + name)
    return book_ablation_report(out_dir)


def sanitize_book_ablation_replays(out):
    """Backend replay files include auth headers; experimental deliveries must not."""
    changed = 0
    for path in Path(out).glob("samples/**/*_replay_*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("headers"):
            value["headers"] = {}
            value["authentication_note"] = "Credentials omitted; use configured environment keys, never this file"
            write_json_atomic(str(path), value)
            changed += 1
    return changed


def book_grayscale_proxy(out, image_path):
    """Read-only luminance evidence: never replaces a generated colour image."""
    from PIL import Image, ImageOps
    colour_proxy = _reassess_proxy(out, image_path)[0]
    target = colour_proxy.with_name(colour_proxy.stem + "-grayscale.jpg")
    if not target.exists():
        with Image.open(colour_proxy) as image:
            ImageOps.grayscale(image).save(target, quality=90)
    return target


def book_ablation_failure_type(folder, status):
    if status == "success":
        return None
    path = Path(folder) / "response.raw.txt"
    if not path.exists():
        return "no_response_or_not_attempted"
    value = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(value, ensure_ascii=False)
    if "WinError 10013" in text:
        return "local_network_permission"
    if "IMAGE_SAFETY" in text or "CONTENT_BLOCKED" in text:
        return "provider_safety_refusal"
    return "request_failed"


def book_ablation_report(out_dir):
    from html import escape
    out = _isolated_out(out_dir)
    sanitize_book_ablation_replays(out)
    plan = json.loads((out / "book-frozen.json").read_text(encoding="utf-8"))
    adjudication_path = out / "adjudications.json"
    adjudications = json.loads(adjudication_path.read_text(encoding="utf-8")) if adjudication_path.exists() else {"overrides": []}
    overrides = {a["sample_id"]: a for a in adjudications["overrides"]}
    reviews = {}
    for path in sorted((out / "vision").glob("*.json")):
        if not path.name.endswith(".request.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            for row in record["response"]["images"]:
                reviews[record["mapping"][row["id"]]] = row
    rows, parts = [], ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>书籍配色规则对照</title><style>body{font:15px/1.6 system-ui;max-width:1100px;margin:auto;padding:20px}section{border-top:1px solid #ccc;padding:14px 0}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>书籍配色规则对照</h1>']
    if plan.get("content_reference"):
        reference = plan["content_reference"]
        if _sha256_file(reference["absolute_path"]) != reference["sha256"]:
            raise ValueError("Frozen reference changed before reporting")
        parts.append('<section><h2>内容源（不是候选或画风图）</h2><img alt="content source" src="' +
                     _thumb_data_uri(reference["absolute_path"], 1400, 90) + '"><pre>' + escape(json.dumps(reference, ensure_ascii=False, indent=2)) + '</pre></section>')
    if plan.get("spatial_guide"):
        guide = plan["spatial_guide"]
        if _sha256_file(guide["absolute_path"]) != guide["sha256"]:
            raise ValueError("Frozen spatial guide changed before reporting")
        parts.append('<section><h2>独立位置图（普通视觉指导，不是接口原生蒙版、不是候选）</h2><img alt="spatial locator" src="' +
                     _thumb_data_uri(guide["absolute_path"], 1400, 100) + '"><pre>' +
                     escape(json.dumps(guide, ensure_ascii=False, indent=2)) + '</pre></section>')
    for request in plan["requests"]:
        path = out / "samples" / request["id"] / "result.json"
        history = book_sample_attempts(path.parent)
        result = history[-1] if history else {"status": "not_attempted", "outputs": []}
        group = next(g for g in plan["groups"] if g["id"] == request["group_id"])
        review = reviews.get(request["id"])
        model_review = review
        adjudication = overrides.get(request["id"])
        if adjudication:
            if not result["outputs"] or _sha256_file(result["outputs"][0]) != adjudication["output_sha256"] or review is None:
                raise ValueError("Offline adjudication does not match actual reviewed image")
            review = {**review, **adjudication["changes"]}
            validate_book_ablation_review({"images": [review]}, [review["id"]])
        usable = bool(not group["is_control"] and result["status"] == "success" and review and
            all(review[k] == "conform" for k in ("colour_status", "region_status", "area_status")) and
            review["protected_ok"] is True and review["content_ok"] is True)
        if plan.get("content_reference"):
            usable = bool(usable and review["visible_person_count"] == plan["content_reference"]["subject_count"])
        if group.get("uses_content_reference"):
            usable = bool(usable and review["source_count_ok"] is True and review["source_layout_status"] == "conform")
        row = {"sample_id": request["id"], "group_id": request["group_id"], "is_control": group["is_control"],
               "generation_status": result["status"], "review": review, "usable": usable,
               "model_review": model_review, "adjudication": adjudication,
               "attempt_history": [{**a, "failure_type": book_ablation_failure_type(a["attempt_folder"], a["status"])} for a in history],
               "failure_type": book_ablation_failure_type(result.get("attempt_folder", path.parent), result["status"])}
        rows.append(row)
        parts.append('<section><h2>' + escape(request["id"]) + '</h2>')
        for p in result["outputs"]:
            parts.append('<img alt="' + escape(request["id"]) + '" src="' + _thumb_data_uri(p, 1400, 90) + '">')
            if plan["spec"].get("readonly_grayscale"):
                parts.append('<p>只读灰度证据（不替代产图）</p><img alt="grayscale audit" src="' +
                             _thumb_data_uri(book_grayscale_proxy(out, p), 1400, 90) + '">')
        parts.append('<pre>' + escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></section>')
    groups = [{"id": g["id"], "planned": plan["spec"]["rounds"],
               "generated": sum(r["generation_status"] == "success" for r in rows if r["group_id"] == g["id"]),
               "reviewed": sum(r["review"] is not None for r in rows if r["group_id"] == g["id"]),
               "usable": sum(r["usable"] for r in rows if r["group_id"] == g["id"]),
               "colour_counts": {state: sum((r["review"] or {}).get("colour_status", "unverifiable") == state
                    for r in rows if r["group_id"] == g["id"]) for state in ("conform", "partial", "fail", "unverifiable")},
               "is_control": g["is_control"], "promoted": False} for g in plan["groups"]]
    if plan.get("content_reference"):
        for group in groups:
            group_rows = [r for r in rows if r["group_id"] == group["id"]]
            group["content_counts"] = {state: sum((r["review"] or {}).get("content_ok") is state for r in group_rows) for state in (True, False)}
            group["source_layout_counts"] = {state: sum((r["review"] or {}).get("source_layout_status", "unverifiable") == state for r in group_rows)
                                             for state in ("conform", "partial", "fail", "unverifiable")}
            group["grayscale_counts"] = {state: sum((r["review"] or {}).get("grayscale_status", "unverifiable") == state for r in group_rows)
                                         for state in ("conform", "partial", "fail", "unverifiable")}
    from utils import send_budget
    authorization = book_retry_authorization(out)
    ledger = {"image": send_budget.state(str(out / "send-budget"), "image", plan["planned_images"] + len(authorization["sample_ids"])),
              "text": send_budget.state(str(out / "send-budget"), "text", plan["spec"]["review_max_calls"] + authorization["extra_reviews"]),
              "retry_authorization": authorization,
              "failed_attempts_retained": sum(a["status"] == "failed" for r in rows for a in r["attempt_history"]),
              "cost": "not reconciled; retain server usage and call events, no invoice lookup"}
    ledger["outcome_counts"] = {"success": sum(r["generation_status"] == "success" for r in rows),
        "local_network_permission": sum(r["failure_type"] == "local_network_permission" for r in rows),
        "provider_safety_refusal": sum(r["failure_type"] == "provider_safety_refusal" for r in rows)}
    ledger["attempt_outcome_counts"] = {state: sum(
        ("success" if a["status"] == "success" else a["failure_type"]) == state
        for r in rows for a in r["attempt_history"])
        for state in ("success", "local_network_permission", "provider_safety_refusal", "request_failed")}
    if plan.get("spatial_guide"):
        guide_path = plan["spatial_guide"]["absolute_path"]
        declared = {}
        mislabelled = []
        for request in plan["requests"]:
            for entry in request.get("attachments") or []:
                declared.setdefault(entry["role"], []).append(request["id"])
                if entry["path"] == guide_path and entry["role"] != "spatial_guide":
                    mislabelled.append({"sample_id": request["id"], "role": entry["role"]})
        ledger["attachments_declared_by_role"] = {role: sorted(set(ids)) for role, ids in declared.items()}
        ledger["attachment_role_check"] = {
            "spatial_guide_is_never_content": not mislabelled,
            "mislabelled": mislabelled,
            "native_mask_support": plan["spatial_guide"].get("native_mask_support", "not_confirmed"),
            "note": plan.get("attachment_note", "")}
    comparisons = []
    variants = list(plan["spec"]["variants"])
    for scheme in plan["spec"]["schemes"]:
        for first, second in combinations(variants, 2):
            a = next(g for g in groups if g["id"] == scheme["id"] + "-" + first)
            b = next(g for g in groups if g["id"] == scheme["id"] + "-" + second)
            common = [r for r in range(1, plan["spec"]["rounds"] + 1)
                      if all(any(row["sample_id"] == scheme["id"] + "-" + variant + "-r" + str(r)
                                 and row["generation_status"] == "success" and row["review"] is not None for row in rows)
                             for variant in (first, second))]
            comparisons.append({"scheme": scheme["id"], first + "_usable": a["usable"], second + "_usable": b["usable"],
                                "common_reviewed_rounds": common, "causal_claim_allowed": False,
                                "coverage_note": "missing generated or reviewed counterparts prohibit a clean variant comparison" if len(common) != plan["spec"]["rounds"] else "unpaired random samples only"})
    summary = {"plan_hash": plan["plan_hash"], "planned": len(rows), "groups": groups, "rows": rows,
               "variant_comparisons": comparisons,
               "ledger": ledger, "limitations": ["two unpaired random samples per group; no statistical reliability claim",
                                                   "no repaint or production GUI wiring", "tone selector not tested in this area/separation ablation"]}
    if plan.get("content_reference"):
        summary["content_reference"] = plan["content_reference"]
        summary["limitations"] = ["single one-person content source; no multi-person validation or statistical causal claim",
                                  "new near-bank scope is not comparable to old all-bank scope",
                                  "grayscale is read-only diagnostic, not automatic correction or a production quality gate",
                                  "no material, local-tone generation trial, style reference or production GUI wiring"]
    if len(variants) < 2:
        summary["limitations"].append("single-variant diagnostic run; no within-run variant comparison")
    if plan.get("spatial_guide"):
        summary["spatial_guide"] = plan["spatial_guide"]
        summary["attachment_note"] = plan.get("attachment_note", "")
        summary["limitations"] = list(summary["limitations"]) + [
            "the intervention compared here is the whole package of an extra locator attachment plus its role text, not the image alone",
            "the locator is ordinary visual guidance, not a native API mask and not a pixel-exact boundary; two rounds cannot show stability or a causal benefit",
            "boxes locate objects; ground or water inside a box is not automatically protected and a blank locator area grants no recolour permission"]
    write_json_atomic(str(out / "book-summary.json"), summary)
    write_json_atomic(str(out / "book-per-image.json"), {"rows": rows})
    write_json_atomic(str(out / "call-ledger.json"), ledger)
    parts.append('</html>')
    (out / "evaluation.html").write_text("\n".join(parts), encoding="utf-8")
    (out / "RUN-REPORT.md").write_text("# 书籍配色规则对照\n\n冻结协议：`" + plan["plan_hash"] + "`\n\n" +
        "|组|计划|生成|评审|可用|\n|---|---|---|---|---|\n" +
        "\n".join("|{id}|{planned}|{generated}|{reviewed}|{usable}|".format(**g) for g in groups) +
        "\n\n可用要求：配色、落点、面积均 conform，保护色与内容均明确 true。失败/未评审不从分母消失；B0 不参与配色晋级。\n\n" +
        "授权重试样本：" + ", ".join(authorization["sample_ids"]) +
        "。首次失败记录原样保留，逐张 JSON 和评价页的 attempt_history 展示全部尝试；有效结果取最新尝试，逻辑样本分母仍为 " + str(len(rows)) + "。保留失败尝试数：" + str(ledger["failed_attempts_retained"]) +
        "。重试补齐结果不代表首轮全成功，也不能作为规则收益的因果证据。\n\n" +
        ("本轮只有一个变体，属于诊断性复验，没有同轮变体对照。" if len(variants) < 2 else "") +
        ("本轮含指定内容源；参考分支可用另要求 source_count_ok=true、source_layout_status=conform。灰度仅记录可辨性，不修改图像。没有真实多人保持、材质或局部色调的在线验证。" if plan.get("content_reference") else "本轮不验证统一色调选择器；") +
        ("本轮第二分支多带一张独立位置图及其角色说明，比较的是这条整体干预，不是「只加一张图」；位置图是普通视觉指导，不是原生蒙版，两轮样本不足以证明稳定性或因果收益。" if plan.get("spatial_guide") else "") +
        "两轮随机样本不足以宣称可靠性或因果效果。不自动重绘、修图或接入生产。费用未核账，仅记录实际发送与服务端用量。\n", encoding="utf-8")
    return summary


def book_request_diff(current_preview: Path, compare_preview: Path):
    """逐条请求的离线差异：只读本地快照，不证明服务端收件内容。

    同名文件对不上时（例如变体改名）退回「提示词最接近」的配对，并如实标注配对方式。
    """
    rows = []
    counterparts = {}
    for path in sorted(Path(compare_preview).glob("*.request.json")):
        counterparts[path.name] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(Path(current_preview).glob("*.request.json")):
        current = json.loads(path.read_text(encoding="utf-8"))
        row = {"request_id": current["id"], "current_body_hash": current["body_hash"],
               "current_attachments": current.get("reference_images", []),
               "compare_file": "", "paired_with": "", "pairing": ""}
        other = counterparts.get(path.name)
        if other is None and counterparts:
            current_lines = current["prompt"].split("\n")
            def shared(item):
                return sum(1 for line in current_lines if line in item[1]["prompt"].split("\n"))
            name, other = max(counterparts.items(), key=lambda item: (shared(item), item[0]))
            row["pairing"] = "closest_prompt"
        elif other is not None:
            row["pairing"] = "same_file_name"
        if other is None:
            row["status"] = "no_counterpart"
            rows.append(row)
            continue
        row["compare_file"] = str(Path(compare_preview) / name) if row["pairing"] == "same_file_name" \
            else str(Path(compare_preview))
        row["paired_with"] = other["id"]
        current_lines = current["prompt"].split("\n")
        other_lines = other["prompt"].split("\n")
        changed = [index for index, (a, b) in enumerate(zip(current_lines, other_lines), 1) if a != b]
        changed += list(range(min(len(current_lines), len(other_lines)) + 1, max(len(current_lines), len(other_lines)) + 1))
        row.update({"status": "compared", "compare_body_hash": other["body_hash"],
                    "prompt_line_count": [len(current_lines), len(other_lines)],
                    "identical_prompt_lines": len(current_lines) - len([i for i in changed if i <= len(current_lines)]),
                    "changed_prompt_lines": changed,
                    "changed_prompt_line_text": [{"line": index, "current": current_lines[index - 1],
                                                  "compare": other_lines[index - 1]} for index in changed
                                                 if index <= len(other_lines) and index <= len(current_lines)],
                    "compare_attachments": other.get("reference_images", []),
                    "same_model": current["model"] == other["model"],
                    "same_size": [current["resolution"], current["aspect_ratio"]] == [other["resolution"], other["aspect_ratio"]],
                    "same_attachment_bytes": [a["sha256"] for a in current.get("reference_images", []) if a["role"] == "content"]
                                            == [a["sha256"] for a in other.get("reference_images", []) if a["role"] == "content"],
                    "roles": [a["role"] for a in current.get("reference_images", [])]})
        rows.append(row)
    return {"note": "Local offline comparison of frozen request snapshots; not a server receipt.",
            "current_preview": str(current_preview), "compare_preview": str(compare_preview), "requests": rows}


def book_preflight(spec_path, out_dir, *, compare_preview="", log=print):
    """离线预检：不加任何 mask 字段、不占调用额度，附件/端点/顺序/预算全部留证。"""
    from modules.others.api_backend import get_api_config
    plan = freeze_book_ablation(spec_path, out_dir)
    out = _isolated_out(out_dir)
    checks = {"plan_hash": plan["plan_hash"], "spec_id": plan["spec"]["id"],
              "generated_at": _now(), "sent": False,
              "requests": len(plan["requests"]), "rounds": plan["spec"]["rounds"],
              "groups": [{"id": g["id"], "variant": g.get("variant"),
                          "attachments": [{"role": e["role"], "path": e["path"], "sha256": e["sha256"],
                                           "size": e.get("size"), "order": e["order"]} for e in g.get("attachments", [])]}
                         for g in plan["groups"]],
              "image_max_retries": plan["spec"]["image_max_retries"],
              "review_max_calls": plan["spec"]["review_max_calls"],
              "failures": []}
    problems = checks["failures"]

    def check(condition, message):
        if not condition:
            problems.append(message)
        return bool(condition)

    if plan.get("content_reference"):
        reference = plan["content_reference"]
        check(Path(reference["absolute_path"]).is_file(), "content source missing")
        check(_sha256_file(reference["absolute_path"]) == reference["sha256"], "content source hash changed")
        checks["content_reference"] = {key: reference[key] for key in ("path", "sha256", "role", "subject_count")
                                       if key in reference}
    if plan.get("spatial_guide"):
        guide = plan["spatial_guide"]
        check(Path(guide["absolute_path"]).is_file(), "spatial guide missing")
        check(_sha256_file(guide["absolute_path"]) == guide["sha256"], "spatial guide hash changed")
        check(guide["role"] == BOOK_SPATIAL_GUIDE_ROLE, "spatial guide role not declared")
        check(list(guide["canvas_size"]) == list(_image_size(guide["absolute_path"])), "spatial guide size changed")
        check(guide.get("native_mask_support") == "not_confirmed",
              "native mask support must stay unconfirmed; no mask field is sent")
        checks["spatial_guide"] = {key: guide[key] for key in
                                   ("path", "sha256", "role", "canvas_size", "boxes", "manifest",
                                    "manifest_sha256", "instruction", "instruction_sha256", "native_mask_support")}
        checks["spatial_guide_scope"] = guide.get("scope_note", "")
    attachment_rows = []
    for request in plan["requests"]:
        body = book_generation_body(request)
        forbidden = [key for key in body if key.lower() in ("mask", "maskimage", "referenceimages", "mask_mode")]
        parts = body["contents"][0]["parts"]
        check(not forbidden, "request body must not carry an undocumented mask field")
        check(len(parts) == 1 + len(book_attachment_paths(request)), "body part count must match the attachments")
        on_disk = [{"order": index, "parts_index": index + 1, "path": path} for index, path in
                   enumerate(book_attachment_paths(request), 1)]
        roles = ([e["role"] for e in request["attachments"]] if request.get("attachments")
                 else ["content"] * len(request["image_paths"]))
        check(len(roles) == len(on_disk), "every attachment needs a declared role")
        attachment_rows.append({"sample_id": request["id"], "body_hash": digest(body), "model": request["model"],
                                "api_type": request["api_type"], "resolution": request["resolution"],
                                "aspect_ratio": request["aspect_ratio"], "sent": False,
                                "attachments": [{**row, "role": role, "sha256": entry["sha256"],
                                                 "size": entry.get("size"), "verified_on_disk":
                                                 _sha256_file(row["path"]) == entry["sha256"]}
                                                for row, role, entry in zip(on_disk, roles, request.get("attachments") or
                                                                            [{"sha256": _sha256_file(p)} for p in request["image_paths"]])]})
    checks["request_bodies"] = attachment_rows
    channel = get_api_config(api_type=plan["requests"][0]["api_type"])
    checks["generation_endpoint"] = {"host": urlparse(channel.get("base_url", "")).hostname,
                                     "base_url": channel.get("base_url", ""),
                                     "credentials_present": bool(channel.get("api_key")),
                                     "credentials_value": "masked; never recorded",
                                     "request_model": plan["requests"][0]["model"]}
    check(checks["generation_endpoint"]["host"] == "new.aigc2d.com", "generation endpoint host changed")
    check(checks["generation_endpoint"]["credentials_present"], "generation credentials missing")
    checks["review_endpoint"] = {"base_url": plan["review_base_url"], "model": plan["review_model"],
                                 "credentials_present": True, "credentials_value": "masked; never recorded"}
    from utils import send_budget
    checks["budget"] = {"planned_image_calls": plan["planned_images"],
                        "planned_review_calls": plan["spec"]["review_max_calls"],
                        "already_reserved_images": send_budget.state(str(out / "send-budget"), "image",
                                                                     plan["planned_images"])["used"],
                        "already_reserved_text": send_budget.state(str(out / "send-budget"), "text",
                                                                   plan["spec"]["review_max_calls"])["used"]}
    if plan.get("spatial_guide"):
        reference_group = next((g for g in plan["groups"] if not g.get("uses_spatial_guide")), None)
        delta = []
        for group in plan["groups"]:
            entry = {"group": group["id"], "reference_group": reference_group["id"] if reference_group else "",
                     "attachments": [{"order": e["order"], "role": e["role"], "path": e["path"]} for e in group.get("attachments", [])]}
            if reference_group and group["id"] != reference_group["id"]:
                base_prompt = reference_group["prompt"]
                entry["prompt_is_reference_plus_prefix"] = group["prompt"].endswith(base_prompt)
                entry["prompt_prefix_chars"] = len(group["prompt"]) - len(base_prompt)
                entry["prompt_prefix"] = group["prompt"][:len(group["prompt"]) - len(base_prompt)] if entry["prompt_is_reference_plus_prefix"] else ""
                base_roles = [e["role"] for e in reference_group.get("attachments", [])]
                entry["extra_attachment_roles"] = [e["role"] for e in group.get("attachments", []) if e["role"] not in base_roles]
            delta.append(entry)
        checks["variant_delta_within_plan"] = {
            "note": "一个分支相对另一分支的增量：只允许出现「附件 + 其角色说明」这条干预。",
            "rows": delta}
    if compare_preview:
        diff = book_request_diff(out / "preview", compare_preview)
        write_json_atomic(str(out / "offline-request-diff.json"), diff)
        checks["compared_preview"] = compare_preview
        checks["compared_requests"] = checks.get("compared_requests", 0) + len(diff["requests"])
    checks["ok"] = not problems
    write_json_atomic(str(out / "PREFLIGHT-CHECKS.json"), checks)
    log("[book] 离线预检 " + ("通过" if not problems else "失败") + "：" + json.dumps(
        {"plan_hash": checks["plan_hash"], "requests": checks["requests"], "failures": problems}, ensure_ascii=False))
    if problems:
        raise ValueError("Offline preflight failed: " + "; ".join(problems))
    return checks


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


def _strip_control_chars(text):
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def _balanced_object(text):
    """取第一个 '{' 到与之配对的 '}'（考虑字符串与转义）；未闭合时取到结尾。"""
    start = text.find("{")
    if start < 0:
        return ""
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:]


def _close_partial_json(text):
    """截断响应修复：找到最后一个「完整值」的结尾并在此截断，丢掉半截内容后补齐闭合符号。

    只把结构性的 `}` / `]` / 字面量结尾计为安全点；字符串内部的括号不算，
    否则会把字符串从中间切断而仍判为可修复。
    """
    depth_obj, depth_arr, in_string, escaped, last_safe = 0, 0, False, False, 0
    for index, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                last_safe = index + 1
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
            if depth_obj == 0 and depth_arr == 0:
                return text[:index + 1]
            last_safe = index + 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
            last_safe = index + 1
        elif ch in ",:" or ch.isspace():
            continue
        else:
            last_safe = index + 1
    head = text[:last_safe].rstrip().rstrip(",")
    head += "]" * max(0, depth_arr) + "}" * max(0, depth_obj)
    return head


def parse_object_tolerant(raw, log=None):
    """JSON 直解 → 提取最外层对象 → 去控制字符 → 截断修复。返回 (value, note)。

    只做保守修复：括号不配对、需要猜结构的响应一律判为失败并保留原文，
    不把猜测出来的对象当成成功记录（2026-10-04：坏响应靠 `}`, `]` 计数无法安全还原）。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    attempts = [("direct", text)]
    balanced = _balanced_object(text)
    if balanced and balanced != text:
        attempts.append(("balanced_object", balanced))
    for label, candidate in list(attempts):
        attempts.append((f"{label}+control_chars", _strip_control_chars(candidate)))
    for label, candidate in list(attempts):
        clean = _strip_control_chars(candidate)
        if clean.count("{") > clean.count("}") or clean.count("[") > clean.count("]"):
            attempts.append((f"{label}+close_partial", _close_partial_json(clean)))
    last_error = None
    for label, candidate in attempts:
        try:
            value = json.loads(_strip_control_chars(candidate))
        except Exception as error:  # noqa: BLE001
            last_error = error
            continue
        if isinstance(value, dict):
            if log and label != "direct":
                log(f"[reassess] JSON 修复生效：{label}")
            return value, label
    if log:
        log(f"[reassess] JSON 无法修复：{last_error}")
    return None, str(last_error)


def normalize_reassess_response(value):
    """把放错层级的 observations 抬回图片级：直接嵌在 colour_conformance 里，
    或嵌在它的 protected_colours 里（模型偶尔多写一层 `}` 造成的形态）。语义不变。
    """
    moved = []
    for row in (value.get("images") or []) if isinstance(value, dict) else []:
        if not isinstance(row, dict) or "observations" in row:
            continue
        conformance = row.get("colour_conformance")
        if not isinstance(conformance, dict):
            continue
        if isinstance(conformance.get("observations"), dict):
            row["observations"] = conformance.pop("observations")
            moved.append(str(row.get("id")))
            continue
        protected = conformance.get("protected_colours")
        if isinstance(protected, dict) and isinstance(protected.get("observations"), dict):
            row["observations"] = protected.pop("observations")
            conformance["protected_colours"] = protected.get("entries") or []
            moved.append(str(row.get("id")))
    return moved


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


# ---------------------------------------------------------------------------
# 第二阶段（E0 + A）：双模型分组套件、评价校准、匿名成组比较与本地统计
#
# 仍然是无头实验消费层：不接生产 GUI、不写全局配置、不改任何质量门禁。
# 长提示词一律在 prompts/color-knowledge/，本模块只做组装、缓存、校验、统计与渲染。
# ---------------------------------------------------------------------------

SUITE_GROUPS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")
SUITE_MODELS = ("gemini-flash", "gpt-image-2")
# 业务调用预算（任务书 §9）：视觉调用 5 + 16 + 2，超限直接拒绝，不靠人工盯着
SUITE_CAPS = {"calibration": 5, "comparison": 16, "conformance": 2}
# 色调效果测试单独计数（不混进主试验预算）
SUITE_EXTRA_CAPS = {"tone_reading": 4}
SUITE_RATINGS = ("A", "B", "tie", "uncertain")
SUITE_CONTENT_STATES = ("pass", "fail", "uncertain")
SUITE_COMPARE_DIMENSIONS = ("color_hierarchy", "subject_focus", "tonal_depth", "theme_mood")
SUITE_EXECUTED_STATES = ("yes", "partial", "no", "unclear")
SUITE_COLOR_STATES = ("retained", "altered", "unclear")
SUITE_PROTECTED_COLORS = ("pink hair", "ivory dress", "dusty-pink shoes", "ribbon bows")
SUITE_PROMPT_FILES = {
    "content": "color-knowledge/stage2-content-check-system.md",
    "compare": "color-knowledge/stage2-compare-system.md",
    "conformance": "color-knowledge/stage2-conformance-system.md",
}
SUITE_MAX_EDGE = 1400
SUITE_PROXY_QUALITY = 92
SUITE_RUN_ROOT = ("data", "test-result")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _isolated_out(out_dir):
    out = Path(out_dir).resolve()
    if not out.is_relative_to(BASE.joinpath(*SUITE_RUN_ROOT)):
        raise ValueError("Suite output must live under data/test-result")
    return out


# 测试专用的输出根：只在 pytest 里被显式设置（conftest 或用例本身），
# 生产路径永远走上面的 data/test-result 约束。
TEST_OUTPUT_ROOT_ENV = "IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT"


def _retention_test_root():
    """测试期望的输出根；未设置时返回 None（=按产品的 data/test-result 约束走）。"""
    root = str(os.environ.get(TEST_OUTPUT_ROOT_ENV) or "").strip()
    return Path(root).resolve() if root else None


def _resolve_retention_out(out_dir):
    """实验/修正目录解析。

    - 测试显式设置了 `TEST_OUTPUT_ROOT_ENV` 时：路径落在该根下（或由相对路径解析到工作区外的临时目录）
      就允许，便于用例在自己的 tmp 里生成修正交付；
    - 其余情况仍受产品的 `data/test-result` 约束，防止把产物写到真实日期目录或别处。
    """
    out = Path(out_dir).resolve()
    root = _retention_test_root()
    if root and (out.is_relative_to(root) or not out.is_relative_to(BASE)):
        return out
    return _isolated_out(out_dir)


def _load_suite_manifest(out: Path):
    path = out / "suite.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing suite manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_suite_spec(path):
    """校验实验 spec：主试验（八组、K 一致、三轮）或色调效果测试（kind=tone）。"""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = str(spec.get("kind") or "group-suite")
    if int(spec.get("version", 0)) not in (2, 3):
        raise ValueError("Stage-2 spec must be version 2 or 3")
    for key in ("id", "subject", "rendering", "protection_block", "channels", "groups", "evaluation"):
        if not spec.get(key):
            raise ValueError(f"Stage-2 spec is missing {key}")
    groups = spec["groups"]
    ids = [g.get("id") for g in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("Group ids must be unique")
    for group in groups:
        if sum(map(len, group.get("clauses") or [])) > 1600:
            raise ValueError(f"{group['id']}: colour clauses exceed the suite budget")
    if set(spec["channels"]) != set(SUITE_MODELS):
        raise ValueError(f"Stage-2 spec needs exactly the {list(SUITE_MODELS)} channels")
    if kind == "tone":
        # 色调效果测试：第一组是同一轮内的基线，其余组都必须只差附加块
        if not 2 <= len(groups) <= 8:
            raise ValueError("Tone spec needs 2..8 groups")
        if not 1 <= int(spec.get("rounds", 0)) <= 4:
            raise ValueError("Tone spec rounds must be 1..4")
        if dict((spec["evaluation"].get("caps") or {})) != SUITE_EXTRA_CAPS:
            raise ValueError(f"Tone spec caps must be {SUITE_EXTRA_CAPS}")
        for index, group in enumerate(groups):
            if index == 0:
                if group.get("clauses") or group.get("protection"):
                    raise ValueError(f"{group['id']}: the tone baseline must carry no extra block")
            elif not group.get("clauses") or not group.get("protection"):
                raise ValueError(f"{group['id']}: tone treatments need clauses and the protection block")
        return spec
    if [g.get("id") for g in groups] != list(SUITE_GROUPS):
        raise ValueError(f"Stage-2 groups must be {list(SUITE_GROUPS)} in order")
    for group in groups:
        gid = group["id"]
        if bool(group.get("protection")) != (gid != "G0"):
            raise ValueError(f"{gid}: only G0 may skip the protection block")
        if gid in ("G0", "G1") and group.get("clauses"):
            raise ValueError(f"{gid}: G0/G1 must not carry colour clauses")
        if gid not in ("G0", "G1") and not group.get("clauses"):
            raise ValueError(f"{gid}: G2~G7 need colour clauses")
    if int(spec.get("rounds", 0)) != 3:
        raise ValueError("The authorised main run is exactly three rounds")
    comparisons = [tuple(c) for c in spec["evaluation"].get("comparisons_per_model", [])]
    expected = [("G0", "G1")] + [(f"G{i}", "G1") for i in range(2, 8)]
    if comparisons != expected:
        raise ValueError(f"Pre-registered comparisons must be {expected}")
    if int(spec["evaluation"].get("swap_rechecks", 0)) != 2:
        raise ValueError("Two swapped-order rechecks per model are pre-registered")
    if dict(spec["evaluation"].get("vision_call_caps") or {}) != SUITE_CAPS:
        raise ValueError("Vision call caps must match the task budget")
    return spec


def group_by_id(spec, group_id):
    for group in spec["groups"]:
        if group["id"] == group_id:
            return group
    raise KeyError(f"Unknown group {group_id}")


def build_group_prompt(spec, group):
    """subject → rendering → COLOR PLAN（G2~G7）→ PRESERVE（G1~G7 的同一份 K）。"""
    parts = [spec["subject"], spec["rendering"]]
    if group.get("clauses"):
        parts.append("COLOR PLAN:\n- " + "\n- ".join(group["clauses"]))
    if group.get("protection"):
        parts.append("PRESERVE:\n" + spec["protection_block"])
    return "\n\n".join(parts)


def build_suite_request(spec, group, channel_key):
    """一条逻辑样本的完整请求（含通道参数）；落盘与发包共用同一份。"""
    channel = spec["channels"][channel_key]
    request = {"channel": channel_key, "api_type": channel["api_type"], "model": channel["model"],
               "group_id": group["id"], "prompt": build_group_prompt(spec, group), "image_paths": []}
    if channel["api_type"] == "aigc-2d-gpt":
        request.update({"mode": channel.get("mode", "generate"), "size": channel["size"],
                        "quality": channel.get("quality"), "output_format": channel.get("output_format", "png"),
                        "n": int(channel.get("n", 1)), "aspect_ratio": channel.get("aspect_ratio", "3:2")})
    else:
        request.update({"resolution": channel["resolution"], "aspect_ratio": channel["aspect_ratio"],
                        "face_quality_boost": bool(channel.get("face_quality_boost", False))})
    return request


def build_schedule(spec):
    """预生成排程：每轮组别顺序随种子打乱，模型起始顺序逐轮轮换。"""
    seed = int(spec.get("schedule_seed", 261003))
    rng = random.Random(seed)
    rounds = []
    for round_number in range(1, int(spec["rounds"]) + 1):
        group_ids = [g["id"] for g in spec["groups"]]
        rng.shuffle(group_ids)
        models = list(SUITE_MODELS)
        if round_number % 2 == 0:
            models.reverse()
        order = [{"sample_id": f"{model_key}-{group_id}-r{round_number}", "model_key": model_key,
                  "group_id": group_id, "round": round_number}
                 for group_id in group_ids for model_key in models]
        rounds.append({"round": round_number, "model_start_order": models, "group_order": group_ids, "order": order})
    return {"seed": seed, "note": "固定随机种子只用于排程与匿名编号，不是模型生成 seed", "rounds": rounds}


def suite_preflight(spec, channels=None):
    """联网前的配置核查：只允许 new.aigc2d.com，且必须有可用 key。"""
    from modules.others.api_backend import get_api_config
    selected = list(channels or spec["channels"].keys())
    reports = []
    for key in selected:
        channel = spec["channels"].get(key)
        if not channel:
            raise ValueError(f"Unknown channel {key}")
        cfg = get_api_config(api_type=channel["api_type"])
        host = urlparse(str(cfg.get("base_url") or "")).hostname
        if host != "new.aigc2d.com":
            raise ValueError(f"{key}: only the authorised new.aigc2d host is allowed (got {host!r})")
        if not cfg.get("api_key"):
            raise ValueError(f"{key}: image API key unavailable")
        reports.append({"channel": key, "api_type": channel["api_type"], "planned_model": channel["model"],
                        "configured_model": cfg.get("model"), "host": host,
                        "key_source": cfg.get("_api_key_source"),
                        "timeout": cfg.get("timeout"), "backend_max_retries": cfg.get("max_retries")})
    per_channel = len(spec["groups"]) * int(spec["rounds"])
    return {"spec_id": spec["id"], "host": "new.aigc2d.com", "rounds": int(spec["rounds"]),
            "channels": reports, "logical_samples": per_channel * len(selected),
            "logical_samples_per_channel": per_channel,
            "note": "预检只核查配置；在线模型清单在收费前另行核查"}


def _redact_server_meta(result):
    """只保留结构化的服务端信息，绝不落盘 base64 图片、URL 或任何凭据。"""
    meta = {}
    raw = result.get("server_response_raw") if isinstance(result, dict) else None
    if isinstance(raw, dict):
        # Gemini 通道用 modelVersion/responseId/usageMetadata，GPT 通道用 model/created/usage
        for key in ("model", "modelVersion", "responseId", "created", "id", "object", "usage", "usageMetadata",
                    "size", "quality", "output_format", "revised_prompt", "status", "background"):
            if key in raw:
                meta[key] = raw[key]
        data = raw.get("data")
        if isinstance(data, list):
            meta["data_items"] = len(data)
            first = data[0] if data else None
            if isinstance(first, dict):
                meta["first_item_keys"] = sorted(k for k in first if k not in ("b64_json", "url"))
                for key in ("revised_prompt", "size", "quality", "output_format", "mime_type"):
                    if key in first:
                        meta.setdefault("first_item", {})[key] = first[key]
    text = str((result or {}).get("raw_text") or "") if isinstance(result, dict) else ""
    if text:
        meta["raw_text_head"] = text[:400]
    return meta


def _generate_suite_sample(request, folder, sample_id, log):
    """按通道分发到既有后端；不新写带凭据的 HTTP 客户端。"""
    from modules.others.api_backend import generate_image_aigc2d, generate_image_aigc2d_gpt
    common = dict(prompt=request["prompt"], image_paths=[], api_type=request["api_type"],
                  save_sub_dir=str(folder), file_prefix=f"test-{sample_id}",
                  return_metadata=True, log_callback=log)
    if request["api_type"] == "aigc-2d-gpt":
        result = generate_image_aigc2d_gpt(
            model=request["model"], mode=request.get("mode", "generate"), size=request.get("size"),
            quality=request.get("quality"), output_format=request.get("output_format"),
            n=int(request.get("n", 1)), aspect_ratio=request.get("aspect_ratio", "1:1"), **common)
    else:
        result = generate_image_aigc2d(
            model=request["model"], resolution=request.get("resolution"),
            aspect_ratio=request.get("aspect_ratio", "1:1"),
            face_quality_boost=bool(request.get("face_quality_boost", False)), **common)
    if isinstance(result, dict):
        return list(result.get("saved_files") or []), _redact_server_meta(result)
    return list(result or []), {}


def _verify_and_hash(paths):
    from PIL import Image
    hashes = {}
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        hashes[Path(path).name] = _sha256_file(path)
    return hashes


def _sample_outputs_intact(sample):
    outputs = sample.get("outputs") or []
    if not outputs or not all(Path(p).is_file() for p in outputs):
        return False
    stored = sample.get("output_hashes") or {}
    if not stored:
        return True
    return all(stored.get(Path(p).name) == _sha256_file(p) for p in outputs)


def run_suite(spec, out_dir, *, rounds=None, channels=None, dry_run=False, retry_failed=False,
              log=print, stop_after_two_failures=True, request_builder=None):
    """主试验：三轮 × 八组 × 两模型 = 48 个逻辑样本，成功结果缓存复用。

    `request_builder` 让新协议（固定色板）复用同一套落盘/续跑/状态机，只换请求组装函数；
    默认仍是第二阶段的 `build_suite_request`。
    """
    build_request = request_builder or build_suite_request
    out = _isolated_out(out_dir)
    preflight = suite_preflight(spec, channels)
    log(json.dumps(preflight, ensure_ascii=False))
    if dry_run:
        return preflight
    out.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(spec)
    fingerprint = digest({"spec": spec, "schedule": schedule})
    manifest_path = out / "suite.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_hash") != fingerprint:
            raise ValueError("Protocol changed; keep the old run and use a new output directory")
    else:
        manifest = {"protocol_hash": fingerprint, "spec_id": spec["id"], "created_at": _now(),
                    "protocol": {"spec_path": None, "spec": spec, "schedule_seed": schedule["seed"]},
                    "schedule": schedule, "samples": [], "previous_attempts": [],
                    "node_blocks": {}, "limitations": spec.get("limitations", [])}
        write_json_atomic(str(manifest_path), manifest)
        write_json_atomic(str(out / "schedule.json"), schedule)
        log(f"[suite] manifest created ({out})")
    selected_rounds = [int(r) for r in (rounds if rounds is not None else range(1, int(spec["rounds"]) + 1))]
    selected_channels = list(channels or spec["channels"].keys())
    blocks = manifest.setdefault("node_blocks", {})
    consecutive = {}
    aborted_notice = {}
    for entry in schedule["rounds"]:
        if entry["round"] not in selected_rounds:
            continue
        for item in entry["order"]:
            if item["model_key"] not in selected_channels:
                continue
            if (blocks.get(item["model_key"]) or {}).get("aborted"):
                if not aborted_notice.get(item["model_key"]):
                    log(f"[block-skipped] {item['model_key']}: 该区块此前被停止，"
                        f"需要显式 resume-block 才能继续（不自动重跑）")
                    aborted_notice[item["model_key"]] = True
                continue
            sample_id = item["sample_id"]
            previous = next((s for s in manifest["samples"] if s["id"] == sample_id), None)
            if previous and previous.get("status") == "success":
                if _sample_outputs_intact(previous):
                    log(f"[cached] {sample_id}")
                    continue
                log(f"[stale] {sample_id}: 产物缺失或 hash 不一致，按失败处理")
                previous["status"] = "stale"
            if previous and not retry_failed:
                log(f"[not-retried] {sample_id}: {previous.get('status')}")
                continue
            if previous:
                manifest["samples"].remove(previous)
                manifest.setdefault("previous_attempts", []).append(previous)
            group = group_by_id(spec, item["group_id"])
            request = build_request(spec, group, item["model_key"])
            folder = out / "samples" / item["model_key"] / sample_id
            folder.mkdir(parents=True, exist_ok=True)
            attempt = 1 + sum(1 for a in manifest.get("previous_attempts", []) if a.get("id") == sample_id)
            sample = {"id": sample_id, "model_key": item["model_key"], "group_id": item["group_id"],
                      "round": entry["round"], "attempt": attempt, "started_at": _now(),
                      "request": request, "request_hash": digest(request),
                      "prompt_chars": len(request["prompt"]), "status": "running",
                      "outputs": [], "output_hashes": {}, "server_meta": {}}
            manifest["samples"].append(sample)
            write_json_atomic(str(manifest_path), manifest)
            write_json_atomic(str(folder / "request.json"), request)
            (folder / "prompt.txt").write_text(request["prompt"], encoding="utf-8")
            log(f"[generate] {sample_id} (round {entry['round']}, attempt {attempt})")
            started = time.perf_counter()
            try:
                paths, server_meta = _generate_suite_sample(request, folder, sample_id, log)
                sample["outputs"] = [str(Path(p).resolve()) for p in paths]
                sample["output_hashes"] = _verify_and_hash(paths)
                sample["server_meta"] = server_meta
                sample["status"] = "success" if paths else "request_failed"
                sample["failure_type"] = None if paths else "empty_result"
            except Exception as exc:  # noqa: BLE001 —— 只记录异常类型与短消息，快照不含 headers
                sample["status"] = "request_failed"
                sample["failure_type"] = type(exc).__name__
                sample["error"] = str(exc)[:200]
            sample["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            sample["finished_at"] = _now()
            write_json_atomic(str(folder / "result.json"),
                              {"status": sample["status"], "failure_type": sample.get("failure_type"),
                               "elapsed_seconds": sample["elapsed_seconds"],
                               "outputs": [Path(p).name for p in (sample["outputs"] or [])],
                               "output_hashes": sample["output_hashes"], "server_meta": sample["server_meta"]})
            write_json_atomic(str(manifest_path), manifest)
            log(f"[{sample['status']}] {sample_id}, {sample['elapsed_seconds']}s")
            if sample["status"] != "success":
                consecutive[item["model_key"]] = consecutive.get(item["model_key"], 0) + 1
                if stop_after_two_failures and consecutive[item["model_key"]] >= 2:
                    blocks[item["model_key"]] = {"aborted": True, "at": sample_id, "time": _now(),
                                                 "reason": "two consecutive request failures; stop this model block and report"}
                    write_json_atomic(str(manifest_path), manifest)
                    log(f"[block-aborted] {item['model_key']}: 连续两次请求失败，停止该模型区块（不换模型）")
            else:
                consecutive[item["model_key"]] = 0
    summary = suite_status(out)
    log(json.dumps(summary, ensure_ascii=False))
    return manifest


def resume_block(out_dir, model_key, *, note=""):
    """用户显式授权后清除某模型区块的停止标记；旧的中断记录留在 block_history。

    协议要求「不自动重跑业务样本」，所以恢复必须是显式动作，并且要留痕。
    """
    out = _isolated_out(out_dir)
    if model_key not in SUITE_MODELS:
        raise ValueError(f"Unknown model block {model_key}")
    manifest = _load_suite_manifest(out)
    block = dict((manifest.get("node_blocks") or {}).get(model_key) or {})
    if not block.get("aborted"):
        return {"resumed": False, "model_key": model_key, "reason": "该区块没有被停止"}
    manifest.setdefault("block_history", []).append(dict(block, cleared_at=_now(), cleared_note=note))
    manifest.setdefault("node_blocks", {})[model_key] = {"aborted": False, "resumed_at": _now(), "note": note}
    write_json_atomic(str(out / "suite.json"), manifest)
    return {"resumed": True, "model_key": model_key, "previous": block, "note": note}


def _spec_groups(spec):
    return [g["id"] for g in spec["groups"]]


def suite_status(out_dir):
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    spec = (manifest.get("protocol") or {}).get("spec") or {}
    groups = _spec_groups(spec) if spec.get("groups") else list(SUITE_GROUPS)
    models = list(spec["channels"].keys()) if spec.get("channels") else list(SUITE_MODELS)
    rounds = int(spec.get("rounds", 3) or 3)
    rows = []
    for model_key in models:
        samples = [s for s in manifest["samples"] if s["model_key"] == model_key]
        success = [s for s in samples if s.get("status") == "success"]
        rows.append({"model_key": model_key, "planned": len(groups) * rounds,
                     "attempted": len(samples), "success": len(success),
                     "request_failed": len([s for s in samples if s.get("status") == "request_failed"]),
                     "not_attempted": len([s for s in samples if s.get("status") == "running"]),
                     "aborted": bool((manifest.get("node_blocks") or {}).get(model_key, {}).get("aborted")),
                     "groups_success": {g: len([s for s in success if s["group_id"] == g]) for g in groups}})
    return {"spec_id": manifest.get("spec_id"), "protocol_hash": manifest.get("protocol_hash"),
            "models": rows, "previous_attempts": len(manifest.get("previous_attempts") or []),
            "vision_calls": vision_call_counts(out)}


# --- 视觉调用记账与缓存 -----------------------------------------------------

def vision_call_counts(out_dir, purpose=None):
    path = Path(out_dir) / "vision-calls.jsonl"
    counts = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            counts[row.get("purpose", "?")] = counts.get(row.get("purpose", "?"), 0) + 1
    if purpose:
        return counts.get(purpose, 0)
    return counts


def _proxy_path(out: Path, image_path):
    from PIL import Image, ImageOps
    digest_name = _sha256_file(image_path)[:16]
    dest = out / "vision" / "proxies" / f"{digest_name}.jpg"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            ImageOps.contain(image.convert("RGB"), (SUITE_MAX_EDGE, SUITE_MAX_EDGE)).save(
                dest, quality=SUITE_PROXY_QUALITY)
    return dest


def _vision_json_call(out: Path, *, purpose, name, system, user, images, cfg, log, max_tokens=8000, timeout=300):
    """一次视觉文字调用：命中缓存不收费，未命中先记预算再发请求。"""
    from utils.analysis_gpt_prompt import call_text_model
    signature = digest({"system": system, "user": user, "model": cfg["model"],
                        "images": [{"name": Path(p).name, "sha256": _sha256_file(p)} for p in images]})
    target = out / "vision" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("request_hash") == signature:
            log(f"[cached] vision {name}")
            return cached
        raise ValueError(f"Vision request changed for {name}; keep the old record and open a new run directory")
    cap = {**SUITE_CAPS, **SUITE_EXTRA_CAPS}.get(purpose)
    used = vision_call_counts(out, purpose)
    if cap is not None and used >= cap:
        raise RuntimeError(f"vision call budget exhausted for {purpose}: {used}/{cap}")
    request = {"purpose": purpose, "name": name, "system": system, "user": user, "model": cfg["model"],
               "base_url": cfg["base_url"],
               "images": [{"file": Path(p).name, "sha256": _sha256_file(p)} for p in images]}
    write_json_atomic(str(out / "vision" / f"{name}.request.json"), request)
    _record_vision_call(out, purpose, name, cfg["model"])
    raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user,
                          max_tokens=max_tokens, timeout=timeout, image_paths=[str(p) for p in images])
    (out / "vision" / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
    value = parse_object(raw)
    result = {"request_hash": signature, "purpose": purpose, "name": name, "at": _now(),
              "model": cfg["model"], "response": value, "status": "model_review_not_gate"}
    write_json_atomic(str(target), result)
    return result


def _record_vision_call(out: Path, purpose, name, model):
    line = json.dumps({"purpose": purpose, "name": name, "model": model, "at": _now()}, ensure_ascii=False)
    with open(out / "vision-calls.jsonl", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _text_cfg(text_cfg=None):
    from utils.analysis_gpt_prompt import load_text_api_config
    cfg = text_cfg or load_text_api_config()
    if not (cfg.get("base_url") and cfg.get("api_key") and cfg.get("model")):
        raise RuntimeError("缺少视觉文字接口配置（conf/config.json 的 base_url/model + IMAGE_MAKER_TEXT_API_KEY）")
    return cfg


def _validate_content_rows(rows, expected_ids):
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("Content check must cover each neutral id exactly once")
    if sorted(str(r.get("id")) for r in rows) != sorted(str(i) for i in expected_ids):
        raise ValueError("Content check must cover each neutral id exactly once")
    for row in rows:
        if row.get("content_status") not in SUITE_CONTENT_STATES:
            raise ValueError(f"Invalid content_status in {row.get('id')}")
    return rows


def _validate_compare(value, expected_ids):
    rows = value.get("images")
    if not isinstance(rows, list) or sorted(r.get("id") for r in rows) != sorted(expected_ids):
        raise ValueError("Comparison must cover each neutral id exactly once")
    for row in rows:
        if row.get("content_status") not in SUITE_CONTENT_STATES:
            raise ValueError(f"Invalid content_status for {row.get('id')}")
    dims = value.get("dimensions") or {}
    if set(dims) != set(SUITE_COMPARE_DIMENSIONS):
        raise ValueError("Comparison must answer every colour dimension")
    for key, answer in dims.items():
        if answer not in SUITE_RATINGS:
            raise ValueError(f"Invalid dimension answer {key}={answer!r}")
    if value.get("overall") not in SUITE_RATINGS:
        raise ValueError("Invalid overall answer")
    if not str(value.get("overall_reason") or "").strip():
        raise ValueError("Comparison needs an overall reason")
    if not (value.get("region_evidence") or []):
        raise ValueError("Comparison must name at least one concrete region")
    return value


# --- E0：先用第一轮六图校准评价 --------------------------------------------

def _e0_plan(spec, samples):
    seed = int(spec.get("schedule_seed", 261003))
    rng = random.Random(f"e0|{seed}|{spec['id']}")
    ids = [s["id"] for s in samples]
    aliases = ids[:]
    rng.shuffle(aliases)
    alias_of = {sample_id: f"V{i:02d}" for i, sample_id in enumerate(aliases, 1)}
    aa = rng.sample(ids, 2)
    rest = [i for i in ids if i not in aa]
    swap = rng.sample(rest, 2)
    return {"seed": seed, "rule": "A/A 与顺序交换的原图在发出任何视觉调用前按固定种子抽取并落盘",
            "alias_of": alias_of, "aa_sample_ids": aa, "swap_sample_ids": swap,
            "alias_order": [alias_of[i] for i in aliases],
            "note": "第一轮六图只用于校准，不计入本轮 48 个样本"}


def calibrate_e0(out_dir, spec, *, text_cfg=None, log=print, source_run=None):
    """E0：六图内容检查 1 次 + A/A 2 次 + 顺序交换 2 次 = 最多 5 次视觉调用。"""
    out = _isolated_out(out_dir)
    source = Path(source_run).resolve() if source_run else (BASE / spec["calibration_source"]["run_dir"])
    source_manifest = json.loads((source / spec["calibration_source"]["manifest"]).read_text(encoding="utf-8"))
    samples = [s for s in source_manifest["samples"] if s.get("status") == "success" and s.get("outputs")]
    if len(samples) < 6:
        raise ValueError("Calibration needs the six first-round images")
    samples = sorted(samples, key=lambda s: s["id"])
    plan_path = out / "calibration" / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = _e0_plan(spec, samples)
        write_json_atomic(str(plan_path), plan)
    by_id = {s["id"]: s for s in samples}
    cfg = _text_cfg(text_cfg)
    system_content = read_prompt_file(SUITE_PROMPT_FILES["content"])
    system_compare = read_prompt_file(SUITE_PROMPT_FILES["compare"])
    proxies = {sid: _proxy_path(out, by_id[sid]["outputs"][0]) for sid in by_id}

    # 1) 六图内容检查
    order = [{"id": plan["alias_of"][s["id"]], "sample_id": s["id"]} for s in samples]
    user = json.dumps({"task": "content_check",
                       "content_contract": spec["evaluation"]["content_dimensions"],
                       "image_order": [{"id": row["id"]} for row in order],
                       "note": "images are supplied in exactly this order"},
                      ensure_ascii=False)
    content = _vision_json_call(out, purpose="calibration", name="content-check", system=system_content,
                                user=user, images=[proxies[row["sample_id"]] for row in order], cfg=cfg, log=log)
    _validate_content_rows(content["response"].get("images") or [], [row["id"] for row in order])

    # 2) A/A：同一张原图、同一份代理编码、左右各一份
    aa_records = []
    for index, sample_id in enumerate(plan["aa_sample_ids"], 1):
        alias = plan["alias_of"][sample_id]
        mapping = {"letter_group": {"A": alias, "B": alias}, "order": [
            {"id": "P01", "group": "A", "sample_id": sample_id, "alias": alias, "path": str(proxies[sample_id])},
            {"id": "P02", "group": "B", "sample_id": sample_id, "alias": alias, "path": str(proxies[sample_id])}]}
        record = _compare_call(out, spec, f"aa-{index}", None, None, mapping, cfg, system_compare, log,
                               purpose="calibration")
        aa_records.append(record)

    # 3) 顺序交换：同一对原图先 A/B 后 B/A
    first, second = plan["swap_sample_ids"]
    swap_records = []
    for name, left, right in (("swap-ab", first, second), ("swap-ba", second, first)):
        mapping = {"letter_group": {"A": plan["alias_of"][left], "B": plan["alias_of"][right]}, "order": [
            {"id": "P01", "group": "A", "sample_id": left, "alias": plan["alias_of"][left], "path": str(proxies[left])},
            {"id": "P02", "group": "B", "sample_id": right, "alias": plan["alias_of"][right], "path": str(proxies[right])}]}
        swap_records.append(_compare_call(out, spec, name, None, None, mapping, cfg, system_compare, log,
                                          purpose="calibration"))
    verdict = e0_verdict(out, spec, plan, content, aa_records, swap_records, samples)
    write_json_atomic(str(out / "calibration" / "calibration.json"), verdict)
    log(json.dumps(verdict, ensure_ascii=False))
    return verdict


def _compare_call(out: Path, spec, name, model_key, chain, mapping, cfg, system, log, *, purpose):
    order = mapping["order"]
    images = [Path(row["path"]) for row in order]
    if not images:
        raise ValueError("Comparison needs at least one image per side")
    user = json.dumps({"task": "compare_two_groups",
                       "subject": spec["subject"],
                       "protected": spec["protected"],
                       "image_order": [{"id": row["id"], "group": row["group"]} for row in order],
                       "note": "images are supplied in exactly this order; groups are neutral"},
                      ensure_ascii=False)
    record = _vision_json_call(out, purpose=purpose, name=name, system=system, user=user,
                               images=images, cfg=cfg, log=log)
    _validate_compare(record["response"], [row["id"] for row in order])
    result = dict(record)
    result["spec"] = {"model_key": model_key, "comparison": chain}
    result["mapping"] = mapping
    write_json_atomic(str(out / "vision" / f"{name}.json"), result)
    return result


def _multi_person_flag(text):
    blob = str(text or "").lower()
    markers = ("two ", "second ", "another person", "extra person", "additional figure", "double",
               "两个", "两位", "两人", "二人", "双人", "第二个人", "另一个", "额外的", "多出")
    return any(marker in blob for marker in markers)


def e0_verdict(out, spec, plan, content, aa_records, swap_records, samples):
    """校准判定：必须识别双人失败；A/A 不得出现实质偏好；顺序交换不得翻盘。"""
    known = spec["calibration_source"].get("known_content_failure", "")
    rows = content["response"].get("images") or []
    alias_of = plan["alias_of"]
    target_alias = alias_of.get("soft-river-rose-1", "")
    target_row = next((r for r in rows if r.get("id") == target_alias), {})
    blob = " ".join(list(target_row.get("content_issues") or []) + list(target_row.get("uncertain") or [])
                    + list(target_row.get("observations") or []))
    detected = bool(target_row) and (target_row.get("content_status") == "fail"
                                     or _multi_person_flag(blob))
    keyword_flag = _multi_person_flag(blob)
    aa_answers = [r["response"].get("overall") for r in aa_records]
    aa_ok = all(answer in ("tie", "uncertain") for answer in aa_answers)
    swap_pairs = []
    for record in swap_records:
        letter_group = record["mapping"]["letter_group"]
        overall = record["response"].get("overall")
        winner = letter_group.get(overall) if overall in ("A", "B") else overall
        swap_pairs.append({"name": record["name"], "overall": overall, "winner_sample": winner,
                           "winner_alias": letter_group.get(overall, overall)})
    if len(swap_pairs) == 2:
        first, second = swap_pairs
        decisive = first["overall"] in ("A", "B") or second["overall"] in ("A", "B")
        consistent = first["winner_sample"] == second["winner_sample"]
        swap_ok = (not decisive) or consistent
    else:
        decisive, consistent, swap_ok = False, None, False
    failed = []
    if not detected:
        failed.append("六图内容检查未识别柔蓝第一张的双人错误")
    if not aa_ok:
        failed.append(f"A/A 出现实质偏好：{aa_answers}")
    if not swap_ok:
        failed.append("顺序交换后偏好翻盘，存在位置偏好")
    return {"status": "passed" if not failed else "failed", "failed_reasons": failed,
            "content_check": {"target_sample": "soft-river-rose-1", "target_alias": target_alias,
                              "content_status": target_row.get("content_status"),
                              "multi_person_keyword": keyword_flag, "detected": detected,
                              "known_failure": known},
            "aa": [{"name": r["name"], "overall": r["response"].get("overall"),
                    "dimensions": r["response"].get("dimensions"),
                    "alias": r["mapping"]["letter_group"]["A"]} for r in aa_records],
            "aa_ok": aa_ok,
            "swap": swap_pairs, "swap_decisive": decisive, "swap_consistent": consistent, "swap_ok": swap_ok,
            "vision_calls": vision_call_counts(Path(out)),
            "note": "A/A 非平局或交换翻盘即视为评价方法不合格；不合格时不把评分带入自动收益判断"}


# --- 主评价：匿名成组比较 ---------------------------------------------------

def _comparison_plan(out: Path, spec):
    path = out / "comparisons" / "plan.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    seed = int(spec.get("schedule_seed", 261003))
    pairs = [list(p) for p in spec["evaluation"]["comparisons_per_model"]]
    pool = [(model_key, pair) for model_key in SUITE_MODELS for pair in pairs]
    rng = random.Random(f"swap|{seed}|{spec['id']}")
    picks = rng.sample(pool, int(spec["evaluation"]["swap_rechecks"]))
    plan = {"seed": seed, "comparisons_per_model": pairs,
            "swap_picks": [{"model_key": model_key, "pair": pair} for model_key, pair in picks],
            "note": "交换复核的两个比较在发出任何比较调用前从 14 个比较里按固定种子抽出并落盘；"
                    "交换复核是把同一对图的两侧组别与呈现顺序对调，用来查位置偏好"}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(path), plan)
    return plan


def _presentation(model_key, ga, gb, items_a, items_b, seed, swap):
    """组别字母与呈现顺序由种子固定；swap=True 时对**同一个比较**做严格镜像（字母取反、顺序取反）。

    基准分配与打乱顺序**不依赖 swap**，否则交换调用可能抽到同一套字母映射（实测 G6vsG1），
    那就不是位置偏好检验了。
    """
    rng = random.Random(f"{seed}|{model_key}|{ga}|{gb}")
    group_letter = {ga: "A", gb: "B"}
    if rng.random() < 0.5:
        group_letter = {ga: "B", gb: "A"}
    pool = [(s, group_letter[ga]) for s in items_a] + [(s, group_letter[gb]) for s in items_b]
    rng.shuffle(pool)
    if swap:
        group_letter = {g: ("B" if letter == "A" else "A") for g, letter in group_letter.items()}
        pool = [(s, ("B" if letter == "A" else "A")) for s, letter in reversed(pool)]
    order = [{"id": f"P{i:02d}", "group": letter, "sample_id": s["id"], "path": s["outputs"][0]}
             for i, (s, letter) in enumerate(pool, 1)]
    return {"group_letter": group_letter, "letter_group": {v: k for k, v in group_letter.items()}, "order": order,
            "mirrored": bool(swap)}


def _mirror_mapping(out: Path, main_name: str):
    """由**已落盘的主比较记录**生成严格镜像：组别字母取反 + 呈现顺序取反，图片内容不变。

    必须基于主记录而不是重新抽签：主比较的字母分配是那次调用的既有事实，
    重新抽签可能抽到同一套映射（这就是第一次交换复核没有构成位置检验的原因）。
    """
    path = out / "comparisons" / f"{main_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"镜像复核需要已存在的主比较记录：{path}")
    base = json.loads(path.read_text(encoding="utf-8"))["mapping"]
    letter_group = {group: ("B" if letter == "A" else "A") for group, letter in base["group_letter"].items()}
    order = [{"id": f"P{i:02d}", "group": ("B" if row["group"] == "A" else "A"),
              "sample_id": row["sample_id"], "path": row["path"]}
             for i, row in enumerate(reversed(base["order"]), 1)]
    return {"group_letter": letter_group, "letter_group": {v: k for k, v in letter_group.items()},
            "order": order, "mirrored": True, "mirror_of": main_name}


def compare_groups(out_dir, spec, *, model_key=None, text_cfg=None, log=print, recheck=False):
    """每模型七个预注册比较 + 两个交换复核（14 + 2 次调用）。

    `recheck=True` 只重跑交换复核，并把结果写成 `-swap2`：第一次实现里交换调用的字母映射是
    独立随机抽的（可能与前一次相同，实测 G6vsG1），因此那两次记录不能当位置检验；修正为严格镜像后
    另跑一次，旧记录保留不删。
    """
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    plan = _comparison_plan(out, spec)
    cfg = _text_cfg(text_cfg)
    system = read_prompt_file(SUITE_PROMPT_FILES["compare"])
    models = [model_key] if model_key else list(SUITE_MODELS)
    swap_picks = {(item["model_key"], tuple(item["pair"])) for item in plan["swap_picks"]}
    if recheck:
        _write_plan_amendment(out)
    results = {}
    for key in models:
        samples = [s for s in manifest["samples"] if s["model_key"] == key]
        successes = {g: [s for s in samples if s["group_id"] == g and s.get("status") == "success" and s.get("outputs")]
                     for g in SUITE_GROUPS}
        for pair in plan["comparisons_per_model"]:
            ga, gb = pair
            rounds_to_run = (True,) if recheck else (False, True)
            for swap in rounds_to_run:
                if swap and (key, (ga, gb)) not in swap_picks:
                    continue
                if not successes[ga] or not successes[gb]:
                    log(f"[skip] {key} {ga} vs {gb}: 至少一侧没有成功样本")
                    continue
                if swap and recheck:
                    mapping = _mirror_mapping(out, f"{key}-{ga}vs{gb}")
                    name = f"{key}-{ga}vs{gb}-mirror"
                else:
                    mapping = _presentation(key, ga, gb, successes[ga][:3], successes[gb][:3],
                                            plan["seed"], swap)
                    name = f"{key}-{ga}vs{gb}" + ("-swap" if swap else "")
                record = _compare_call(out, spec, name, key, [ga, gb], mapping, cfg, system, log,
                                       purpose="comparison")
                record["swap"] = swap
                record["recheck"] = bool(swap and recheck)
                record["valid_mirror"] = bool(swap and recheck)
                record["images_per_side"] = {"A": sum(1 for r in mapping["order"] if r["group"] == "A"),
                                             "B": sum(1 for r in mapping["order"] if r["group"] == "B")}
                if len(successes[ga]) < 3 or len(successes[gb]) < 3:
                    record["insufficient_samples"] = True
                results[name] = record
                log(f"[compared] {name}: overall={record['response'].get('overall')}")
                write_json_atomic(str(out / "comparisons" / f"{name}.json"), record)
    return results


def _write_plan_amendment(out: Path):
    """把「第一次交换复核实现有缺陷」这件事写进证据目录，避免旧记录被当成有效位置检验。"""
    path = out / "comparisons" / "plan-amendment.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    amendment = {
        "created_at": _now(),
        "defect": "第一次交换复核里，交换调用的组别字母映射与呈现顺序是独立随机抽取的，可能与前一次抽到同一套映射"
                  "（实测 gemini-flash-G6vsG1 两次字母完全相同），因此 -swap 记录不构成有效的位置偏好检验；"
                  "第二次尝试（-swap2）用「重新抽签再取反」，仍然不是既有主记录的镜像，同样按无效记录保留。",
        "fix": "交换复核改为直接读取已落盘的主比较记录，对其做严格镜像：组别字母取反 + 呈现顺序取反，图片内容不变"
               "（utils.color_experiment._mirror_mapping）。",
        "action": "保留 -swap 与 -swap2 记录不删，另跑 -mirror 并以其为准；总比较调用数仍在 16 次预算内。",
        "budget_note": "main comparisons 7 + first swaps 2 + second swaps 2 + mirrors 2 = 13 ≤ 16"}
    write_json_atomic(str(path), amendment)
    return amendment


def comparison_matrix(out_dir, spec):
    """把逐个比较的 A/B 答案映射回组别，并做交换一致性检查。"""
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    rows = []
    for path in sorted((out / "comparisons").glob("*.json")):
        if path.name == "plan.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "mapping" not in record or "response" not in record:
            continue
        letter_group = {v: k for k, v in record["mapping"]["group_letter"].items()}
        name = record.get("name") or path.stem
        spec_info = record.get("spec") or {}
        model_key = spec_info.get("model_key")
        chain = spec_info.get("comparison")
        if not model_key or not chain:
            continue
        ga, gb = chain[0], chain[1]
        response = record["response"]
        content_pass = {letter_group[letter]: 0 for letter in ("A", "B")}
        content_fail = {letter_group[letter]: 0 for letter in ("A", "B")}
        for row in response.get("images") or []:
            letter = row.get("group")
            if letter not in letter_group:
                continue
            group = letter_group[letter]
            if row.get("content_status") == "pass":
                content_pass[group] += 1
            elif row.get("content_status") == "fail":
                content_fail[group] += 1
        dimensions = {key: ("tie" if value == "tie" else
                            "uncertain" if value == "uncertain" else letter_group.get(value, value))
                      for key, value in (response.get("dimensions") or {}).items()}
        overall = response.get("overall")
        answer = response.get("overall")
        rows.append({"name": name, "model_key": model_key, "groups": [ga, gb],
                     "swap": bool(record.get("swap")), "recheck": bool(record.get("recheck")),
                     "valid_mirror": bool(record.get("valid_mirror")),
                     "letter_group": letter_group,
                     "dimensions": dimensions, "overall": answer,
                     "overall_winner": letter_group.get(answer, answer) if answer in ("A", "B") else answer,
                     "content_pass": content_pass, "content_fail": content_fail,
                     "region_evidence": response.get("region_evidence") or [],
                     "cost_of_colour_gain": response.get("cost_of_colour_gain") or "",
                     "overall_reason": response.get("overall_reason") or ""})
    swap_checks = []
    for row in rows:
        if row["swap"]:
            continue
        candidates = [r for r in rows if r["swap"] and r["model_key"] == row["model_key"]
                      and r["groups"] == row["groups"]]
        corrected = [c for c in candidates if c.get("valid_mirror")] or [c for c in candidates if c["recheck"]]
        swapped = (corrected or candidates or [None])[0]
        if not swapped:
            continue
        decisive = row["overall"] in ("A", "B") or swapped["overall"] in ("A", "B")
        consistent = row["overall_winner"] == swapped["overall_winner"]
        swap_checks.append({"model_key": row["model_key"], "groups": row["groups"],
                            "first": row["overall_winner"], "swapped": swapped["overall_winner"],
                            "used_record": swapped["name"], "corrected_mirror": swapped.get("valid_mirror", False),
                            "decisive": decisive, "consistent": consistent,
                            "position_flip": bool(decisive and not consistent)})
    return {"comparisons": rows, "swap_checks": swap_checks,
            "note": "G1 图会重复出现在多个比较里，重复评级不是独立样本"}


# --- 方案符合度评价（不比较好看程度） ---------------------------------------

def pick_conformance_candidate(matrix, model_key):
    scored, qualified = {}, {}
    for row in matrix["comparisons"]:
        if row["swap"] or row["model_key"] != model_key or row["groups"][1] != "G1" or row["groups"][0] == "G0":
            continue
        candidate = row["groups"][0]
        net = 0
        for key, value in row["dimensions"].items():
            if value == candidate:
                net += 1
            elif value == row["groups"][1]:
                net -= 1
        scored[candidate] = {"net_dimension_wins": net, "content_pass": row["content_pass"].get(candidate, 0),
                             "overall_winner": row["overall_winner"], "name": row["name"]}
        if row["content_pass"].get(candidate, 0) >= 2:
            qualified[candidate] = scored[candidate]
    if not qualified:
        return None, scored
    best = max(qualified.items(), key=lambda kv: (kv[1]["net_dimension_wins"], -int(kv[0][1:])))
    return best[0], scored


def evaluate_conformance(out_dir, spec, *, text_cfg=None, log=print):
    """每模型最多一次：只回答条款是否执行、落在哪些区域、保护色是否保留。"""
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    matrix = comparison_matrix(out, spec)
    cfg = _text_cfg(text_cfg)
    system = read_prompt_file(SUITE_PROMPT_FILES["conformance"])
    results = {}
    for model_key in SUITE_MODELS:
        candidate, scored = pick_conformance_candidate(matrix, model_key)
        if not candidate:
            log(f"[conformance] {model_key}: 没有内容合格候选，跳过（不调用）")
            results[model_key] = {"skipped": True, "reason": "no content-qualified candidate", "scores": scored}
            continue
        group = group_by_id(spec, candidate)
        images = [s for s in manifest["samples"] if s["model_key"] == model_key
                  and s["group_id"] == candidate and s.get("status") == "success" and s.get("outputs")]
        proxies = [_proxy_path(out, s["outputs"][0]) for s in images[:3]]
        if not proxies:
            results[model_key] = {"skipped": True, "reason": "candidate has no usable image"}
            continue
        user = json.dumps({"task": "colour_clause_conformance",
                           "subject": spec["subject"],
                           "clauses": group["clauses"],
                           "protection_block": spec["protection_block"],
                           "protected_colours": list(SUITE_PROTECTED_COLORS),
                           "image_order": [{"id": f"Q{i:02d}", "sample_id": s["id"]}
                                           for i, s in enumerate(images[:3], 1)],
                           "note": "images are supplied in exactly this order"},
                          ensure_ascii=False)
        record = _vision_json_call(out, purpose="conformance", name=f"{model_key}-{candidate}",
                                   system=system, user=user, images=proxies, cfg=cfg, log=log)
        value = record["response"]
        indices = [row.get("index") for row in (value.get("clauses") or [])]
        if indices != list(range(1, len(group["clauses"]) + 1)):
            raise ValueError("Conformance review must cover every clause index exactly once")
        for row in value["clauses"]:
            if row.get("executed") not in SUITE_EXECUTED_STATES:
                raise ValueError("Invalid executed value in conformance review")
        names = {row.get("name") for row in (value.get("protected_colors") or [])}
        if not set(SUITE_PROTECTED_COLORS).issubset(names):
            raise ValueError("Conformance review must report every protected colour")
        for row in value.get("protected_colors") or []:
            if row.get("status") not in SUITE_COLOR_STATES:
                raise ValueError("Invalid protected colour status")
        record["group_id"] = candidate
        record["images"] = [s["id"] for s in images[:3]]
        record["selection_scores"] = scored
        write_json_atomic(str(out / "vision" / f"{model_key}-{candidate}.json"), record)
        write_json_atomic(str(out / "conformance" / f"{model_key}-{candidate}.json"), record)
        results[model_key] = record
        log(f"[conformance] {model_key}: {candidate} 完成")
    return results


# --- 本地统计与渲染 ---------------------------------------------------------

# --- 色调效果测试：用户显式点名色调（冷/暖）是否真的改变画面 -----------------

TONE_READINGS = ("very_cool", "cool", "neutral", "warm", "very_warm")
TONE_READING_FILE = "color-knowledge/stage2-tone-read-system.md"
TONE_CHUNK = 5


def _hue_metrics(path):
    """把色相拉成可比较的数字：有彩度像素的冷/暖占比与圆均值色相。

    这是低层颜色属性，不是美学判断；没有区域蒙版，背景面积占优，不能当人物区域的测量。
    """
    from PIL import Image
    import numpy as np
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=float) / 255
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    delta = mx - mn
    sat = np.divide(delta, mx, out=np.zeros_like(mx), where=mx > 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros_like(mx)
    nonzero = delta > 1e-6
    mask = nonzero & (mx == r)
    hue[mask] = ((g - b)[mask] / delta[mask]) % 6
    mask = nonzero & (mx == g) & (mx != r)
    hue[mask] = ((b - r)[mask] / delta[mask]) + 2
    mask = nonzero & (mx == b) & (mx != r) & (mx != g)
    hue[mask] = ((r - g)[mask] / delta[mask]) + 4
    hue = hue / 6.0 * 360.0
    colorful = sat > 0.08
    cool = colorful & (hue >= 150.0) & (hue < 300.0)
    warm = colorful & ((hue < 75.0) | (hue >= 300.0))
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    colorful_count = int(colorful.sum())
    circular = float(np.degrees(np.arctan2(np.sin(np.radians(hue[colorful])).mean(),
                                           np.cos(np.radians(hue[colorful])).mean())) % 360) if colorful_count else None
    return {"path": str(path), "mean_hue_circular": circular,
            "colorful_fraction": float(colorful.mean()),
            "cool_fraction_of_colorful": float(cool.sum() / colorful_count) if colorful_count else 0.0,
            "warm_fraction_of_colorful": float(warm.sum() / colorful_count) if colorful_count else 0.0,
            "cool_fraction_of_image": float(cool.mean()),
            "warm_fraction_of_image": float(warm.mean()),
            "mean_hsv_saturation": float(sat.mean()),
            "mean_display_luma": float(luma.mean()),
            "note": "色相在有彩度像素（S>0.08）上统计；冷=150~300°，暖=<75° 或 ≥300°；全局统计，无区域蒙版"}


def _circular_hue_mean(values):
    """把若干圆均值色相再合成一个圆均值（0~360）；没有有效值时返回 None。"""
    import math
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    radians = [math.radians(v) for v in valid]
    return round(math.degrees(math.atan2(sum(math.sin(r) for r in radians) / len(radians),
                                         sum(math.cos(r) for r in radians) / len(radians))) % 360, 1)


def tone_statistics(out_dir, spec):
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    rows, per_group = [], []
    for sample in manifest["samples"]:
        if sample.get("status") != "success":
            continue
        for path in sample["outputs"]:
            row = _hue_metrics(path)
            row.update({"sample_id": sample["id"], "model_key": sample["model_key"],
                        "group_id": sample["group_id"], "round": sample["round"]})
            rows.append(row)
    for model_key in sorted({r["model_key"] for r in rows}):
        for group in spec["groups"]:
            subset = [r for r in rows if r["model_key"] == model_key and r["group_id"] == group["id"]]
            if not subset:
                continue
            per_group.append({"model_key": model_key, "group_id": group["id"], "label": group["label"],
                              "count": len(subset),
                              "cool_share": sum(r["cool_fraction_of_colorful"] for r in subset) / len(subset),
                              "warm_share": sum(r["warm_fraction_of_colorful"] for r in subset) / len(subset),
                              "mean_hue_circular": _circular_hue_mean([r["mean_hue_circular"] for r in subset]),
                              "mean_hsv_saturation": sum(r["mean_hsv_saturation"] for r in subset) / len(subset),
                              "mean_display_luma": sum(r["mean_display_luma"] for r in subset) / len(subset)})
    result = {"images": rows, "groups": per_group,
              "limitations": ["全局色相分布，无区域蒙版；背景面积占优",
                              "3 张/组只说明趋势", "冷暖占比不是“好不好看”的指标"]}
    write_json_atomic(str(out / "tone-metrics.json"), result)
    return result


def evaluate_tone(out_dir, spec, *, text_cfg=None, log=print):
    """每 5 张一次调用：同时回答内容状态与冷暖读数；不告诉评价模型分组。"""
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    cfg = _text_cfg(text_cfg)
    system = read_prompt_file(TONE_READING_FILE)
    samples = [s for s in manifest["samples"] if s.get("status") == "success" and s.get("outputs")]
    if not samples:
        raise ValueError("No generated images to read")
    records = []
    for index in range(0, len(samples), TONE_CHUNK):
        chunk = samples[index:index + TONE_CHUNK]
        proxies = [_proxy_path(out, s["outputs"][0]) for s in chunk]
        order = [{"id": f"R{i:02d}", "sample_id": s["id"]} for i, s in enumerate(chunk, 1)]
        user = json.dumps({"task": "tone_and_content_reading",
                           "content_contract": spec["evaluation"]["content_dimensions"],
                           "tone_scale": list(TONE_READINGS),
                           "image_order": [{"id": row["id"]} for row in order],
                           "note": "images are supplied in exactly this order;分组未知"}, ensure_ascii=False)
        name = f"tone-read-{index // TONE_CHUNK + 1}"
        record = _vision_json_call(out, purpose="tone_reading", name=name, system=system, user=user,
                                   images=proxies, cfg=cfg, log=log)
        rows = record["response"].get("images") or []
        if sorted(str(r.get("id")) for r in rows) != sorted(row["id"] for row in order):
            raise ValueError("Tone reading must cover each neutral id exactly once")
        for row in rows:
            if row.get("content_status") not in SUITE_CONTENT_STATES:
                raise ValueError(f"Invalid content_status in {row.get('id')}")
            if row.get("tone_reading") not in TONE_READINGS:
                raise ValueError(f"Invalid tone_reading in {row.get('id')}")
        record["mapping"] = {row["id"]: row["sample_id"] for row in order}
        write_json_atomic(str(out / "tone" / f"{name}.json"), record)
        records.append(record)
    summary = tone_reading_summary(spec, manifest, records)
    write_json_atomic(str(out / "tone-readings.json"), summary)
    log(json.dumps({"readings": summary["per_group"]}, ensure_ascii=False))
    return summary


def tone_reading_summary(spec, manifest, records):
    score = {"very_cool": -2, "cool": -1, "neutral": 0, "warm": 1, "very_warm": 2}
    sample_of = {row["id"]: row for row in manifest["samples"]}
    rows = []
    for record in records:
        for row in record["response"].get("images") or []:
            sample_id = record["mapping"][row["id"]]
            rows.append({"id": row["id"], "sample_id": sample_id,
                         "group_id": sample_of[sample_id]["group_id"],
                         "model_key": sample_of[sample_id]["model_key"],
                         "tone_reading": row.get("tone_reading"),
                         "tone_score": score.get(row.get("tone_reading")),
                         "content_status": row.get("content_status"),
                         "subject_colours": row.get("subject_colours"),
                         "tone_evidence": row.get("tone_evidence") or [],
                         "cool_regions": row.get("cool_regions") or [],
                         "warm_regions": row.get("warm_regions") or [],
                         "content_issues": row.get("content_issues") or []})
    per_group = []
    for group in spec["groups"]:
        subset = [r for r in rows if r["group_id"] == group["id"]]
        if not subset:
            continue
        per_group.append({"group_id": group["id"], "label": group["label"], "count": len(subset),
                          "mean_tone_score": sum(r["tone_score"] for r in subset) / len(subset),
                          "readings": [r["tone_reading"] for r in subset],
                          "content_pass": len([r for r in subset if r["content_status"] == "pass"]),
                          "content_fail": len([r for r in subset if r["content_status"] == "fail"])})
    return {"rows": rows, "per_group": per_group,
            "scale": "very_cool=-2 … very_warm=+2（主观读数，不是测量值）",
            "note": "评价模型不知道分组；这里只问色温读数与内容状态，不问好不好看"}


def summarize_tone(out_dir, spec):
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    metrics = tone_statistics(out, spec)
    readings_path = out / "tone-readings.json"
    readings = json.loads(readings_path.read_text(encoding="utf-8")) if readings_path.is_file() else {"rows": [], "per_group": []}
    sheets = []
    for model_key in sorted({s["model_key"] for s in manifest["samples"]}):
        rows = []
        for group in spec["groups"]:
            cells = []
            for round_number in range(1, int(spec["rounds"]) + 1):
                sample = next((s for s in manifest["samples"]
                               if s["id"] == f"{model_key}-{group['id']}-r{round_number}"), None)
                if sample and sample.get("status") == "success" and sample["outputs"]:
                    cells.append((sample["outputs"][0], f"{group['id']} r{round_number}"))
                elif sample:
                    cells.append((_placeholder_cell(out, model_key, group["id"], round_number),
                                  f"{group['id']} r{round_number} 缺失"))
            if cells:
                rows.append(cells)
        if rows:
            path = _sheet(rows, out / f"tone-sheet-{model_key}.jpg",
                          title=f"{model_key} · 色调测试（等显示面积，保持比例）")
            if path:
                sheets.append(str(path))
    write_plan_cards(out, spec, manifest)
    gallery = write_tone_gallery(out, spec, manifest, metrics, readings)
    return {"metrics": metrics, "readings": readings, "sheets": sheets, "gallery": str(gallery)}


def write_tone_gallery(out: Path, spec, manifest, metrics, readings):
    from html import escape
    samples_by_id = {s["id"]: s for s in manifest["samples"]}
    tone_of = {row["sample_id"]: row for row in readings.get("rows", [])}

    def card(sample):
        tone = tone_of.get(sample["id"], {})
        if sample.get("status") == "success" and sample.get("outputs"):
            relative = Path(sample["outputs"][0]).relative_to(out).as_posix()
            thumb = _thumb_data_uri(sample["outputs"][0])
            metrics_row = next((r for r in metrics["images"] if r["sample_id"] == sample["id"]), {})
            body = (f'<a href="{escape(relative)}"><img src="{thumb}" alt="{escape(sample["id"])}"></a>'
                    f'<p class="muted">冷暖读数 <b>{escape(str(tone.get("tone_reading", "未评价")))}</b>'
                    f'（{escape(str(tone.get("content_status", "")))}）· 冷占有彩 {metrics_row.get("cool_fraction_of_colorful", 0):.2f} · '
                    f'暖占有彩 {metrics_row.get("warm_fraction_of_colorful", 0):.2f} · '
                    f'圆均值色相 {metrics_row.get("mean_hue_circular") and round(metrics_row["mean_hue_circular"])}°</p>'
                    f'<p class="muted">{escape("；".join(tone.get("tone_evidence") or [])[:220])}</p>')
        else:
            message = sample.get("error") or (sample.get("server_meta") or {}).get("raw_text_head") or ""
            body = ('<div class="failed">请求失败：%s</div><p class="muted">%s</p>'
                    % (escape(str(sample.get("failure_type"))), escape(str(message)[:300])))
        return f'<figure class="sample"><figcaption>{escape(sample["id"])}</figcaption>{body}</figure>'

    sections = []
    for model_key in sorted({s["model_key"] for s in manifest["samples"]}):
        blocks = []
        for group in spec["groups"]:
            cells = "".join(card(samples_by_id[f"{model_key}-{group['id']}-r{r}"])
                            for r in range(1, int(spec["rounds"]) + 1)
                            if f"{model_key}-{group['id']}-r{r}" in samples_by_id)
            blocks.append(f'<section><h3>{escape(group["id"])} · {escape(group["label"])}</h3>'
                          f'<p class="muted">{escape(group["intent"])} · 附加块 {escape(group["added_block"])} · '
                          f'条款 {group["clause_chars"]} 字符</p><div class="row">{cells}</div></section>')
        sections.append(f'<section><h2>{escape(model_key)}</h2>{"".join(blocks)}'
                        f'<p><a href="tone-sheet-{escape(model_key)}.jpg">整组对比图</a></p></section>')
    metrics_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td></tr>".format(
            escape(row["model_key"]), escape(row["group_id"]), row["count"], row["cool_share"], row["warm_share"],
            row.get("mean_hue_circular"), row["mean_hsv_saturation"], row["mean_display_luma"])
        for row in metrics["groups"])
    reading_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{}</td><td>{}</td></tr>".format(
            escape(row["group_id"]), escape(row["label"]), escape("、".join(row["readings"])),
            row["mean_tone_score"], row["content_pass"], row["content_fail"])
        for row in readings.get("per_group", []))
    content_rows = "".join(
        f'<li>{escape(row["sample_id"])}（{escape(row["group_id"])}）：{escape(str(row["tone_reading"]))} · '
        f'内容 {escape(str(row["content_status"]))} · 保护色 {escape(str(row.get("subject_colours"))) } · '
        f'{escape("；".join(row["content_issues"]))}</li>' for row in readings.get("rows", []))
    css = ("body{font:15px/1.6 system-ui,sans-serif;color:#25322b;background:#eef1ee;margin:0}"
           "main{max-width:1600px;margin:auto;padding:24px}section{background:#fff;border-radius:12px;padding:18px;margin:16px 0}"
           ".row{display:flex;gap:12px;flex-wrap:wrap}.sample{width:330px;margin:0}"
           ".sample img{width:100%;border-radius:8px;border:1px solid #d8ded9}"
           ".failed{padding:40px;background:#fbe9ea;color:#a12e37;border-radius:8px;text-align:center}"
           ".muted{color:#657269;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}"
           "td,th{border:1px solid #dde3de;padding:6px;text-align:left}")
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<title>{escape(spec["title"])}</title><style>{css}</style><main>',
        f'<h1>{escape(spec["title"])}</h1>',
        f'<p class="muted">spec <code>{escape(spec["id"])}</code> · 协议 hash '
        f'<code>{escape(str(manifest.get("protocol_hash"))[:16])}</code></p>',
        '<aside><b>这只是小样本</b><p>每组三张、同一主题，只说明趋势；冷暖读数是评价模型的主观读数，'
        '本地色相统计是低层颜色属性，两者都不是“好不好看”。视觉调用计数：'
        f'{escape(str(vision_call_counts(out)))}</p></aside>',
        '<section><h2>本地色相统计（有彩度像素的冷暖占比）</h2>'
        '<table><tr><th>模型</th><th>组</th><th>张数</th><th>冷占有彩</th><th>暖占有彩</th><th>圆均值色相</th><th>平均彩度</th><th>平均亮度</th></tr>'
        + metrics_rows +
        '</table><p class="muted">冷 = 色相 150–300°，暖 = &lt;75° 或 ≥300°，只在 S&gt;0.08 的像素上统计。'
        '没有区域蒙版，背景面积占优，所以这不等于“人物变冷了”。</p></section>',
        '<section><h2>评价模型的冷暖读数（不知道分组）</h2>'
        '<table><tr><th>组</th><th>名称</th><th>三次读数</th><th>平均分</th><th>内容合格</th><th>内容失败</th></tr>'
        + reading_rows + '</table><p class="muted">刻度 very_cool=-2 … very_warm=+2。'
        '这里只问“看起来冷不冷”和内容是否合规，没有问是否更好看。</p></section>',
        f'<section><h2>逐图读数</h2><ul>{content_rows}</ul></section>',
        "".join(sections),
        '<section><h2>方案卡</h2>'
        + "".join(f'<details><summary>{escape(g["id"])} · {escape(g["label"])}</summary><p>{escape(g["intent"])}</p>'
                  f'<ul>{"".join(f"<li>{escape(c)}</li>" for c in g["clauses"])}</ul></details>'
                  for g in spec["groups"])
        + f'<details><summary>保护块 K</summary><pre>{escape(spec["protection_block"])}</pre></details></section>',
        '<p class="muted"><a href="tone-metrics.json">tone-metrics.json</a> · '
        '<a href="tone-readings.json">tone-readings.json</a> · <a href="suite.json">suite.json</a> · '
        '<a href="plan-cards.md">plan-cards.md</a></p>',
        '</main></html>',
    ]
    path = out / "tone-gallery.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path


def _image_metrics(path):
    from PIL import Image
    import numpy as np
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        size = list(rgb.size)
        pixels = np.asarray(rgb, dtype=float) / 255
    mx, mn = pixels.max(axis=2), pixels.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    luma = pixels @ np.array([0.2126, 0.7152, 0.0722])
    return {"path": str(path), "size": size, "mean_display_luma": float(luma.mean()),
            "mean_hsv_saturation": float(sat.mean()),
            "luma_p10_p50_p90": [float(v) for v in np.percentile(luma, [10, 50, 90])],
            "near_white_fraction": float((mn > 0.97).mean()),
            "note": "全局描述性指标；不是美学/焦点/内容评分，也不是区域或身份色测量"}


def _sheet(rows, out_path, *, box=(600, 420), columns=3, grayscale=False, title=""):
    from PIL import Image, ImageDraw, ImageOps
    cells = [cell for row in rows for cell in row]
    if not cells:
        return None
    columns = max(1, min(columns, max(len(row) for row in rows)))
    cell_w, cell_h = box[0] + 20, box[1] + 40
    sheet = Image.new("RGB", (cell_w * columns, cell_h * len(rows) + (30 if title else 0)), "#f2f2f2")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((12, 8), title, fill="#222222")
    top = 30 if title else 0
    for r, row in enumerate(rows):
        for c, (path, label) in enumerate(row):
            with Image.open(path) as image:
                tile = ImageOps.contain(image.convert("RGB"), box, Image.LANCZOS)
            if grayscale:
                tile = ImageOps.grayscale(tile).convert("RGB")
            x = c * cell_w + (cell_w - tile.width) // 2
            y = top + r * cell_h + 30 + (box[1] - tile.height) // 2
            sheet.paste(tile, (x, y))
            draw.text((c * cell_w + 10, top + r * cell_h + 8), str(label)[:70], fill="#333333")
    sheet.save(out_path, quality=92)
    return out_path


def _thumb_data_uri(path, max_edge=520, quality=78):
    import base64
    import io
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        tile = ImageOps.contain(image.convert("RGB"), (max_edge, max_edge), Image.LANCZOS)
        buffer = io.BytesIO()
        tile.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _collect_observations(out: Path):
    rows = {"content_check": [], "comparisons": [], "conformance": []}
    for path in sorted((out / "vision").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "response" not in record:
            continue
        purpose = record.get("purpose")
        if purpose == "calibration" and record.get("name") == "content-check":
            rows["content_check"] = record["response"].get("images") or []
        elif purpose in ("comparison", "calibration"):
            rows["comparisons"].append({"name": record.get("name"), "purpose": purpose,
                                        "mapping": record.get("mapping"),
                                        "spec": record.get("spec"),
                                        "response": record["response"]})
        elif purpose == "conformance":
            rows["conformance"].append({"name": record.get("name"), "group_id": record.get("group_id"),
                                        "images": record.get("images"), "response": record["response"]})
    return rows


def summarize_suite(out_dir, spec):
    """本地统计、分模型对比图、灰度图、方案卡与离线 gallery。"""
    out = _isolated_out(out_dir)
    manifest = _load_suite_manifest(out)
    metrics, per_sample = [], {}
    for sample in manifest["samples"]:
        if sample.get("status") != "success":
            continue
        for path in sample["outputs"]:
            row = _image_metrics(path)
            row.update({"sample_id": sample["id"], "model_key": sample["model_key"],
                        "group_id": sample["group_id"], "round": sample["round"],
                        "elapsed_seconds": sample.get("elapsed_seconds")})
            metrics.append(row)
            per_sample.setdefault(sample["id"], []).append(row)
    group_means = []
    for model_key in SUITE_MODELS:
        for group_id in SUITE_GROUPS:
            rows = [m for m in metrics if m["model_key"] == model_key and m["group_id"] == group_id]
            if not rows:
                continue
            group_means.append({"model_key": model_key, "group_id": group_id, "count": len(rows),
                                "mean_display_luma": sum(r["mean_display_luma"] for r in rows) / len(rows),
                                "mean_hsv_saturation": sum(r["mean_hsv_saturation"] for r in rows) / len(rows),
                                "near_white_fraction": sum(r["near_white_fraction"] for r in rows) / len(rows)})
    sheets = []
    for model_key in SUITE_MODELS:
        rows, gray_rows = [], []
        for group_id in SUITE_GROUPS:
            cells, gray_cells = [], []
            for round_number in range(1, int(spec["rounds"]) + 1):
                sample = next((s for s in manifest["samples"] if s["id"] == f"{model_key}-{group_id}-r{round_number}"), None)
                if sample and sample.get("status") == "success" and sample["outputs"]:
                    cells.append((sample["outputs"][0], f"{group_id} r{round_number}"))
                    gray_cells.append((sample["outputs"][0], f"{group_id} r{round_number}"))
                else:
                    cells.append((_placeholder_cell(out, model_key, group_id, round_number), f"{group_id} r{round_number} 缺失"))
                    gray_cells.append(cells[-1])
            rows.append(cells)
            gray_rows.append(gray_cells)
        path = _sheet(rows, out / f"sheet-{model_key}-grid.jpg", title=f"{model_key} · 八组 × 三轮（等显示面积，保持比例）")
        if path:
            sheets.append(str(path))
        gray_path = _sheet(gray_rows, out / f"sheet-{model_key}-grid-gray.jpg", grayscale=True,
                           title=f"{model_key} · 灰度（只用于看明暗层次，不用于排名）")
        if gray_path:
            sheets.append(str(gray_path))
    for path in sorted((out / "comparisons").glob("*.json")):
        if path.name == "plan.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "mapping" not in record:
            continue
        by_letter = {"A": [], "B": []}
        for row in record["mapping"]["order"]:
            by_letter[row["group"]].append((row["path"], f"{row['id']} · {row['sample_id']}"))
        if not by_letter["A"] or not by_letter["B"]:
            continue
        rows = [by_letter["A"], by_letter["B"]]
        title = f"{record.get('name')} · A={record['mapping']['group_letter'].get('A')} B={record['mapping']['group_letter'].get('B')} · overall={record['response'].get('overall')}"
        sheet = _sheet(rows, out / f"cmp-{record.get('name')}.jpg", box=(560, 380), title=title)
        if sheet:
            sheets.append(str(sheet))
    write_json_atomic(str(out / "metrics.json"),
                      {"images": metrics, "group_means": group_means, "sheets": sheets,
                       "limitations": ["没有区域蒙版，不宣称主体面积/焦点强度/身份色误差",
                                       "HSV 平均饱和度是显示空间定义，不等于艺术上的全部“彩度”",
                                       "近白像素少不自动更好，更低彩度也不自动更好"]})
    observations = _collect_observations(out)
    write_json_atomic(str(out / "observations.json"), observations)
    plan_cards = write_plan_cards(out, spec, manifest)
    matrix = comparison_matrix(out, spec)
    write_json_atomic(str(out / "comparison-matrix.json"), matrix)
    gallery = write_suite_gallery(out, spec, manifest, metrics, group_means, matrix, observations)
    return {"metrics": metrics, "group_means": group_means, "sheets": sheets,
            "plan_cards": str(plan_cards), "gallery": str(gallery), "matrix": matrix}


def _placeholder_cell(out: Path, model_key, group_id, round_number):
    from PIL import Image, ImageDraw
    path = out / f"missing-{model_key}-{group_id}-r{round_number}.png"
    if not path.is_file():
        image = Image.new("RGB", (600, 420), "#e6e6e6")
        ImageDraw.Draw(image).text((20, 200), f"{model_key} {group_id} r{round_number}: no image", fill="#a03030")
        image.save(path)
    return str(path)


def write_plan_cards(out: Path, spec, manifest):
    lines = [f"# 方案卡 · {spec['title']}", "",
             f"- spec id：`{spec['id']}`（version {spec['version']}，{spec['date']}）",
             f"- 主题（逐字沿用第一轮）：{spec['subject']}", "",
             f"- 固定画法与共同约束：{spec['rendering']}", "",
             f"- 保护块 K（G1~G7 逐字相同，{len(spec['protection_block'])} 字符）：", "",
             "```text", spec["protection_block"], "```", "",
             "- 请求顺序：`subject` → `rendering` → `COLOR PLAN:`（仅 G2~G7）→ `PRESERVE:`（仅 G1~G7）", "",
             "- 来源：来源页转录只存在于本机 `docs/261003-color-improve/book-html/`（未被 git 跟踪）；HEX 色板是设计近似预览，不发给模型，也不冒充书中测量。", "",
             "| 组 | 名称 | 附加内容 | 条款字符 | K 字符 | 指令合计 |", "|---|---|---|---:|---:|---:|"]
    for group in spec["groups"]:
        lines.append(f"| {group['id']} | {group['label']} | {group['added_block']} | "
                     f"{group['clause_chars']} | {group['protection_block_chars']} | {group['instruction_chars']} |")
    lines.append("")
    for group in spec["groups"]:
        lines.append(f"## {group['id']} · {group['label']}")
        lines.append(f"- 意图：{group['intent']}")
        if group.get("palette_preview"):
            lines.append(f"- 近似色板（{group.get('preview_source')}）：{' '.join(group['palette_preview'])}")
        if group.get("clauses"):
            lines.append("- 实际条款：")
            for clause in group["clauses"]:
                lines.append(f"  - {clause}")
        for source in group.get("sources", []):
            lines.append(f"- 来源 PDF {source['pdf_page']} / 印刷 {source['printed_page']}《{source['page_title']}》"
                         f" · {source['section']}（{source['evidence']}）：{source['adaptation']}")
        lines.append("")
    lines.append("## 精确请求顺序（排程，固定种子 %s）" % manifest["schedule"]["seed"])
    lines.append("")
    for entry in manifest["schedule"]["rounds"]:
        order = " → ".join(f"{item['sample_id']}" for item in entry["order"])
        lines.append(f"- 第 {entry['round']} 轮（模型起始顺序 {', '.join(entry['model_start_order'])}）：{order}")
    lines.append("")
    path = out / "plan-cards.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_suite_gallery(out: Path, spec, manifest, metrics, group_means, matrix, observations):
    """离线可打开的对比页：缩略图内嵌、原图相对链接、失败样本同样可点开。"""
    from html import escape
    samples_by_id = {s["id"]: s for s in manifest["samples"]}
    metric_by_sample = {}
    for row in metrics:
        metric_by_sample.setdefault(row["sample_id"], []).append(row)
    status = suite_status(out)
    counts = vision_call_counts(out)

    def sample_card(sample):
        rows = metric_by_sample.get(sample["id"], [])
        meta = rows[0] if rows else {}
        if sample.get("status") == "success" and sample.get("outputs"):
            relative = Path(sample["outputs"][0]).relative_to(out).as_posix()
            thumb = _thumb_data_uri(sample["outputs"][0])
            image_html = (f'<a href="{escape(relative)}"><img src="{thumb}" alt="{escape(sample["id"])}"></a>')
            detail = (f"<p class='muted'>{escape(str(meta.get('size')))} · "
                      f"{sample.get('elapsed_seconds')}s · 亮度 {meta.get('mean_display_luma', 0):.3f} · "
                      f"HSV-S {meta.get('mean_hsv_saturation', 0):.3f} · "
                      f"近白 {meta.get('near_white_fraction', 0):.4f}%</p>")
        else:
            message = sample.get("error") or (sample.get("server_meta") or {}).get("raw_text_head") or ""
            image_html = '<div class="failed">请求失败：%s</div>' % escape(str(sample.get("failure_type")))
            detail = f"<p class='muted'>{escape(str(message)[:300])}</p>"
        return (f'<figure class="sample"><figcaption>{escape(sample["id"])}</figcaption>{image_html}{detail}</figure>')

    cards = []
    for model_key in SUITE_MODELS:
        grid = []
        for group in spec["groups"]:
            cells = "".join(sample_card(samples_by_id[f"{model_key}-{group['id']}-r{r}"])
                            for r in range(1, int(spec["rounds"]) + 1)
                            if f"{model_key}-{group['id']}-r{r}" in samples_by_id)
            grid.append(f'<section class="group"><h3>{escape(group["id"])} · {escape(group["label"])}</h3>'
                        f'<div class="row">{cells}</div></section>')
        card = (f'<section><h2>{escape(model_key)}</h2>'
                f'<p class="muted">计划 {status["models"][SUITE_MODELS.index(model_key)]["planned"]} 个逻辑样本，'
                f'成功 {status["models"][SUITE_MODELS.index(model_key)]["success"]} 个，'
                f'请求失败 {status["models"][SUITE_MODELS.index(model_key)]["request_failed"]} 个</p>'
                f'<p><a href="sheet-{escape(model_key)}-grid.jpg">八组对比图</a> · '
                f'<a href="sheet-{escape(model_key)}-grid-gray.jpg">灰度图</a></p>'
                f'{"".join(grid)}</section>')
        cards.append(card)

    compare_rows = []
    for row in matrix["comparisons"]:
        dims = " / ".join(f"{k}: {v}" for k, v in row["dimensions"].items())
        sheet = f'cmp-{row["name"]}.jpg'
        compare_rows.append(f'<tr><td>{escape(row["name"])}</td><td>{escape(str(row["groups"]))}</td>'
                            f'<td>{escape(dims)}</td><td>{escape(str(row["overall_winner"]))}</td>'
                            f'<td>{escape(str(row["content_pass"]))}</td>'
                            f'<td><a href="{escape(sheet)}">对比图</a></td></tr>')
    swap_rows = "".join(f'<li>{escape(str(c))}</li>' for c in matrix["swap_checks"])
    aa_rows = "".join(f'<li>{escape(str(c))}</li>' for c in observations.get("comparisons", [])
                      if c.get("name", "").startswith("aa-"))

    def conformance_block():
        blocks = []
        for item in observations.get("conformance", []):
            response = item.get("response") or {}
            clauses = "".join(f'<li>条款 {c.get("index")}：{escape(str(c.get("executed")))} · '
                              f'{escape("、".join(c.get("regions") or []))} · '
                              f'{escape("；".join(c.get("evidence") or []))}</li>'
                              for c in response.get("clauses") or [])
            colors = "".join(f'<li>{escape(str(c.get("name")))}：{escape(str(c.get("status")))} · '
                             f'{escape("；".join(c.get("evidence") or []))}</li>'
                             for c in response.get("protected_colors") or [])
            blocks.append(f'<h4>{escape(str(item.get("name")))}（{escape(str(item.get("group_id")))}）</h4>'
                          f'<p>{escape(str(response.get("summary") or ""))}</p><ul>{clauses}</ul><ul>{colors}</ul>')
        return "".join(blocks) or "<p>尚未执行方案符合度评价。</p>"

    plan_html = []
    for group in spec["groups"]:
        swatches = "".join(f'<span class="swatch" style="background:{escape(c)}" title="{escape(c)}"></span>'
                           for c in group.get("palette_preview") or [])
        clauses = "".join(f"<li>{escape(c)}</li>" for c in group.get("clauses") or [])
        sources = "".join(f'<li>PDF {s["pdf_page"]} / 印刷 {s["printed_page"]}《{escape(s["page_title"])}》· '
                          f'{escape(s["section"])}（{escape(s["evidence"])}）：{escape(s["adaptation"])}</li>'
                          for s in group.get("sources") or [])
        plan_html.append(f'<section class="card"><h3>{escape(group["id"])} · {escape(group["label"])}</h3>'
                         f'<p>{escape(group["intent"])}</p><p class="muted">附加块：{escape(group["added_block"])} · '
                         f'条款 {group["clause_chars"]} 字符 · K {group["protection_block_chars"]} 字符</p>'
                         f'<div>{swatches}</div><details><summary>实际条款</summary><ul>{clauses}</ul></details>'
                         f'<details><summary>来源与适配</summary><ul>{sources}</ul></details></section>')
    content_rows = "".join(
        f'<li>{escape(str(r.get("id")))}：{escape(str(r.get("content_status")))} · '
        f'{escape("；".join((r.get("content_issues") or []) + (r.get("uncertain") or [])))}</li>'
        for r in observations.get("content_check", []))
    calibration_path = out / "calibration" / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {}
    cal_html = (f'<p>状态：<b>{escape(str(calibration.get("status", "未执行")))}</b> · '
                f'{escape(str(calibration.get("failed_reasons") or ""))}</p>'
                f'<p class="muted">A/A：{escape(str(calibration.get("aa")))}</p>'
                f'<p class="muted">顺序交换：{escape(str(calibration.get("swap")))}</p>'
                f'<ul>{content_rows}</ul>')
    css = ("body{font:15px/1.6 system-ui,sans-serif;color:#25322b;background:#eef1ee;margin:0}"           "main{max-width:1600px;margin:auto;padding:24px}section{background:#fff;border-radius:12px;padding:20px;margin:18px 0}"
           ".row{display:flex;gap:12px;flex-wrap:wrap}.sample{width:330px;margin:0}"
           ".sample img{width:100%;border-radius:8px;border:1px solid #d8ded9}"
           ".failed{padding:40px;background:#fbe9ea;color:#a12e37;border-radius:8px;text-align:center}"
           ".muted{color:#657269;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}"
           "td,th{border:1px solid #dde3de;padding:6px;text-align:left}"
           ".swatch{display:inline-block;width:48px;height:28px;border-radius:5px;border:1px solid #ddd;margin:3px}"
           "code{background:#f2f5f2;padding:1px 4px;border-radius:4px}"
           "@media(max-width:900px){.sample{width:100%}}")
    channel_summary = " / ".join(f"{key}={value['model']}（{value['api_type']}）"
                                 for key, value in spec["channels"].items())
    metrics_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.4f}%</td></tr>".format(
            escape(row["model_key"]), escape(row["group_id"]), row["count"], row["mean_display_luma"],
            row["mean_hsv_saturation"], row["near_white_fraction"] * 100)
        for row in group_means)
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<title>{escape(spec["title"])}</title><style>{css}</style><main>',
        f'<h1>{escape(spec["title"])}</h1>',
        f'<p class="muted">spec <code>{escape(spec["id"])}</code> · 协议 hash '
        f'<code>{escape(str(manifest.get("protocol_hash"))[:16])}</code> · {escape(channel_summary)}</p>',
        '<aside class="card"><b>探索性结论提醒</b><p>三张样本不能证明普遍收益或显著性；有变化不等于有收益；'
        f'匿名评价是模型自评，不是统计结论，也不是发布门禁。视觉调用计数：{escape(str(counts))}</p></aside>',
        f'<section><h2>E0 评价校准</h2>{cal_html}</section>',
        f'<section><h2>方案卡</h2>{"".join(plan_html)}'
        f'<details><summary>保护块 K（G1~G7 逐字相同）</summary><pre>{escape(spec["protection_block"])}</pre></details></section>',
        "".join(cards),
        '<section><h2>匿名成组比较（映射回组别）</h2>'
        '<table><tr><th>比较</th><th>组</th><th>四个维度</th><th>总体</th><th>内容合格(A/B)</th><th>图</th></tr>'
        + "".join(compare_rows) + f'</table><h3>位置交换复核</h3><ul>{swap_rows}</ul></section>',
        f'<section><h2>A/A 校准原文</h2><ul>{aa_rows}</ul></section>',
        f'<section><h2>方案符合度（只问是否执行，不问是否更好）</h2>{conformance_block()}</section>',
        '<section><h2>本地描述统计</h2>'
        '<table><tr><th>模型</th><th>组</th><th>张数</th><th>平均亮度</th><th>HSV 平均彩度</th><th>近白比例</th></tr>'
        + metrics_rows +
        '</table><p class="muted">亮度是显示空间加权值，未做线性化；HSV 平均饱和度是显示空间定义；'
        '近白 = 三通道均 &gt; 0.97。这些数字用于解释变化，不用于排名。没有区域蒙版，'
        '不宣称主体面积、焦点强度或身份色误差。</p>'
        '<p><a href="metrics.json">metrics.json</a> · <a href="observations.json">observations.json</a> · '
        '<a href="comparison-matrix.json">comparison-matrix.json</a> · <a href="suite.json">suite.json</a> · '
        '<a href="plan-cards.md">plan-cards.md</a> · <a href="schedule.json">schedule.json</a></p></section>',
        '</main></html>',
    ]
    html = "".join(parts)
    path = out / "gallery.html"
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 配色方案符合度重评（2026-10-04）
#
# 目标已修正：配色方案是「让画面呈现指定的一套色彩」，不是改善画质，也不要求比基线更好看。
# 因此这一层只回答：指定颜色是否出现、是否按基调/辅助/强调分工、是否落在指定区域、
# 冷暖是否正确、是否混入大块违背方案的颜色、受保护颜色是否保留；内容与画法单列。
# 不使用旧评分/旧赢家/美学排名；评价模型看得到方案与保护范围，但看不到分组协议标签。
# 长提示词在 prompts/color-knowledge/reassess-conformance-system.md。
# ---------------------------------------------------------------------------

REASSESS_LABEL = "color-reassessment-v1"
REASSESS_PROMPT_FILE = "color-knowledge/reassess-conformance-system.md"
REASSESS_MAX_EDGE = 1100
REASSESS_PROXY_QUALITY = 88
REASSESS_CHUNK = 6
REASSESS_BUDGET = 12
REASSESS_CALL_CAP = 10
REASSESS_DETAIL_CROP = (0.40, 0.95)
REASSESS_DETAIL_EDGE = 800
REASSESS_MAX_TOKENS = 12000
REASSESS_TIMEOUT = 420
REASSESS_STATES = ("conform", "partial", "diverge", "unclear")
REASSESS_PROTECTED = ("粉色长发", "象牙白洋装", "灰粉厚底鞋", "蝴蝶结")
REASSESS_SUITES = {
    "stage2": {
        "spec": "prompts/color-knowledge/stage2-spring-fishing.json",
        "run_dir": "data/test-result/20261003/color-stage2-v2",
        "content_note": "主试验内容维度（spec.evaluation.content_dimensions）",
    },
    "tone": {
        "spec": "prompts/color-knowledge/stage2-tone-effect.json",
        "run_dir": "data/test-result/20261003/color-tone-effect-v1",
        "content_note": "冷暖试验内容维度（spec.evaluation.content_dimensions）",
    },
}
REASSESS_SCHEME_LABELS = {
    ("stage2", "G0"): "本次未指定配色方案（原生基线）",
    ("stage2", "G1"): "本次未指定配色方案，只有内容保护块",
    ("stage2", "G2"): "春绿 · 短版关系指令",
    ("stage2", "G3"): "春绿 · 完整版",
    ("stage2", "G4"): "柔蓝 · 完整版",
    ("stage2", "G5"): "不新增色相 · 只安排明暗层次",
    ("stage2", "G6"): "不新增色相 · 只组织彩度与强调位置",
    ("stage2", "G7"): "暖柔春景 · 暖中性光感 + 保留河蓝",
    ("tone", "T0"): "本次未指定配色方案（同轮基线对照）",
    ("tone", "T1"): "冷色调 · 一句用户口吻",
    ("tone", "T2"): "冷色调 · 更强更主导",
    ("tone", "T3"): "暖色调 · 对称对照",
    ("tone", "T4"): "冷色调 · 点名 teal / slate blue / cool grey",
}


def _reassess_suite_refs():
    return {key: {"spec": BASE / value["spec"], "run": BASE / value["run_dir"]}
            for key, value in REASSESS_SUITES.items()}


def load_reassess_items(specs=None):
    """把两个套件的成功 Gemini 样本读成重评条目（只读，不触碰旧实验目录）。"""
    items = []
    for suite_key, ref in _reassess_suite_refs().items():
        spec = json.loads(ref["spec"].read_text(encoding="utf-8"))
        manifest = json.loads((ref["run"] / "suite.json").read_text(encoding="utf-8"))
        groups = {group["id"]: group for group in spec["groups"]}
        for sample in manifest["samples"]:
            if sample.get("model_key") != "gemini-flash" or sample.get("status") != "success":
                continue
            if not sample.get("outputs"):
                continue
            group_id = sample["group_id"]
            group = groups[group_id]
            folder = ref["run"] / "samples" / sample["model_key"] / sample["id"]
            prompt_file = folder / "prompt.txt"
            request_file = folder / "request.json"
            items.append({
                "item_id": sample["id"],
                "suite": suite_key,
                "group_id": group_id,
                "round": sample["round"],
                "scheme_label": REASSESS_SCHEME_LABELS.get((suite_key, group_id), group_id),
                "image": str(Path(sample["outputs"][0])),
                "image_sha256": (sample.get("output_hashes") or {}).get(Path(sample["outputs"][0]).name),
                "request_hash": sample.get("request_hash"),
                "clauses": list(group.get("clauses") or []),
                "protection_block": spec["protection_block"],
                "protected": list(REASSESS_PROTECTED),
                "content_dimensions": list(spec["evaluation"]["content_dimensions"]),
                "subject": spec["subject"],
                "rendering": spec["rendering"],
                "prompt_text": prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else None,
                "request_snapshot": (json.loads(request_file.read_text(encoding="utf-8"))
                                     if request_file.is_file() else None),
            })
    items.sort(key=lambda row: (row["suite"], row["group_id"], int(row["round"])))
    return items


def _reassess_consistency(row):
    """方案槽：只用来给分块排序，不作为合并不同方案的依据。"""
    if not row["clauses"]:
        return "no-scheme"
    named = any(word in clause
                for clause in row["clauses"]
                for word in ("green", "blue", "teal", "cool", "warm", "yellow", "pink", "golden"))
    return "named-colour" if named else "structure-only"


def reassess_plan(items, chunk_size=None):
    """固定分块：块内最多两个方案，每张图带自己的逐字条款，同方案三张永不拆块。

    一块只放一个方案最干净，但 13 个方案 × 3 张会超出 12 次业务调用预算；因此按方案顺序
    两个方案合一块（块内每张图各自带条款与方案标签），并在协议里如实记录这一权衡。
    """
    chunk_size = int(chunk_size or REASSESS_CHUNK)
    ordered = sorted(items, key=lambda row: (row["suite"], row["group_id"], int(row["round"])))
    for row in ordered:
        row["scheme_slot"] = _reassess_consistency(row)
    chunks, current = [], []
    for row in ordered:
        key = (row["suite"], row["group_id"])
        if current:
            keys = {(item["suite"], item["group_id"]) for item in current}
            if key not in keys and len(keys) >= 2:
                chunks.append(current)
                current = []
        current.append(row)
        if len(current) >= chunk_size:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    for chunk in chunks:
        counts = {}
        for row in chunk:
            counts[(row["suite"], row["group_id"])] = counts.get((row["suite"], row["group_id"]), 0) + 1
        for (suite, group_id), count in counts.items():
            total = len([row for row in items if row["suite"] == suite and row["group_id"] == group_id])
            if count > chunk_size:
                raise ValueError("Reassessment chunk exceeds the size limit")
            if count not in (total, chunk_size):
                raise ValueError("A colour scheme must never be split across chunks")
    used = [row["item_id"] for chunk in chunks for row in chunk]
    if sorted(used) != sorted(row["item_id"] for row in items):
        raise ValueError("Reassessment plan must cover every item exactly once")
    return {"chunk_size": chunk_size,
            "rule": ("块内最多两个方案，每张图携带自己的逐字条款与方案标签；同方案三张永不拆块；"
                     "六张的方案与相邻方案合一块，块内按套件+方案+轮次排序"),
            "chunks": [{"name": f"reassess-{i:02d}", "item_ids": [row["item_id"] for row in chunk],
                        "schemes": sorted({row["scheme_label"] for row in chunk}),
                        "suite": chunk[0]["suite"],
                        "groups": sorted({row["group_id"] for row in chunk})}
                       for i, chunk in enumerate(chunks, 1)]}
    used = [row["item_id"] for chunk in chunks for row in chunk]
    if sorted(used) != sorted(row["item_id"] for row in items):
        raise ValueError("Reassessment plan must cover every item exactly once")
    return {"chunk_size": chunk_size,
            "rule": ("同方案三张永不拆块；只有方案槽一致（同为无方案基线 / 同为点名色相 / "
                     "同为只调结构）时才允许并块，且并块后不超过 chunk_size+2；"
                     "块内按套件+方案+轮次排序"),
            "chunks": [{"name": f"reassess-{i:02d}", "item_ids": [row["item_id"] for row in chunk],
                        "schemes": sorted({row["scheme_label"] for row in chunk})}
                       for i, chunk in enumerate(chunks, 1)]}


def _reassess_vision_counts(out: Path):
    path = out / "vision-calls.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return rows


def _reassess_failure_counts(out: Path):
    path = out / "reassess-failures.jsonl"
    if not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _record_reassess_failure(out: Path, name, stage, error):
    row = {"name": name, "stage": stage, "error": str(error)[:1500], "at": _now()}
    with open(out / "reassess-failures.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def reassess_call_usage(out: Path):
    rows = _reassess_vision_counts(out)
    succeeded = len([r for r in rows if _cached_record_usable(out, str(r.get("name")))])
    return {"business_calls": len(rows),
            "business_call_cap": REASSESS_CALL_CAP,
            "authorised_budget": REASSESS_BUDGET,
            "usable_records": succeeded,
            "failed_records": _reassess_failure_counts(out),
            "purposes": sorted({str(r.get("purpose")) for r in rows}),
            "note": "底层 HTTP 重试由客户端计数，不计入业务调用；业务调用失败不做自动重试"}


def _cached_record_usable(out: Path, name: str):
    path = out / "vision" / f"{name}.json"
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("response"))
    except Exception:  # noqa: BLE001
        return False


def _reassess_payload(row, item_id, clauses, spec):
    return {"id": item_id,
            "colour_scheme_label": row["scheme_label"],
            "colour_plan_clauses": clauses,
            "protection_block": spec["protection_block"],
            "protected_colours": list(REASSESS_PROTECTED),
            "content_contract": list(spec["evaluation"]["content_dimensions"]),
            "images_supplied": [f"{item_id} 全图", f"{item_id} 下半身细节裁剪"]}


def _local_colour_stats(path):
    """辅助证据：全局色相分布。没有区域蒙版，不能说成区域颜色面积测量。"""
    from PIL import Image
    import numpy as np
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        size = list(rgb_image.size)
        pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    mx, mn = pixels.max(axis=2), pixels.min(axis=2)
    delta = mx - mn
    sat = np.divide(delta, mx, out=np.zeros_like(mx), where=mx > 0)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    hue = np.zeros_like(mx)
    nonzero = delta > 1e-6
    mask = nonzero & (mx == r)
    hue[mask] = ((g - b)[mask] / delta[mask]) % 6
    mask = nonzero & (mx == g) & (mx != r)
    hue[mask] = ((b - r)[mask] / delta[mask]) + 2
    mask = nonzero & (mx == b) & (mx != r) & (mx != g)
    hue[mask] = ((r - g)[mask] / delta[mask]) + 4
    hue_deg = hue / 6.0 * 360.0
    colorful = sat > 0.08
    total = int(pixels.shape[0] * pixels.shape[1])
    colorful_count = int(colorful.sum())
    luma = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    bands = {"cool": colorful & (hue_deg >= 150.0) & (hue_deg < 300.0),
             "warm": colorful & ((hue_deg < 75.0) | (hue_deg >= 300.0)),
             "green": colorful & (hue_deg >= 75.0) & (hue_deg < 150.0)}
    families = []
    if colorful_count:
        degrees = hue_deg[colorful]
        strength = sat[colorful]
        index = np.clip((degrees // 15).astype(int), 0, 23)
        weight = np.bincount(index, weights=strength, minlength=24)
        shares = weight / max(1e-6, weight.sum())
        for slot in shares.argsort()[::-1][:3]:
            if shares[slot] < 0.06:
                continue
            in_slot = index == slot
            sample = pixels[colorful][in_slot]
            mean_rgb = (sample.mean(axis=0) * 255).astype(int) if in_slot.any() else np.zeros(3, dtype=int)
            families.append({"hue_band": f"{slot * 15}-{slot * 15 + 15}°",
                             "weight_share_of_chromatic": round(float(shares[slot]), 4),
                             "mean_rgb": list(int(v) for v in mean_rgb),
                             "mean_hex": "#%02x%02x%02x" % tuple(int(v) for v in mean_rgb)})
    return {"path": str(path), "size": size,
            "mean_display_luma": round(float(luma.mean()), 4),
            "luma_p10_p50_p90": [round(float(v), 4) for v in np.percentile(luma, [10, 50, 90])],
            "mean_hsv_saturation": round(float(sat.mean()), 4),
            "near_white_fraction": round(float((mn > 0.97).mean()), 5),
            "chromatic_pixel_fraction": round(colorful_count / total, 4),
            "chromatic_cool_fraction": round(float(bands["cool"].sum() / colorful_count), 4) if colorful_count else 0.0,
            "chromatic_warm_fraction": round(float(bands["warm"].sum() / colorful_count), 4) if colorful_count else 0.0,
            "chromatic_green_fraction": round(float(bands["green"].sum() / colorful_count), 4) if colorful_count else 0.0,
            "dominant_colour_families": families,
            "note": ("全局描述统计（显示空间，未线性化）；冷=色相 150~300°、暖=<75° 或 ≥300°、"
                     "绿=75~150°，都统计在彩度 S>0.08 的像素上。没有区域蒙版，"
                     "不能当作人物区域颜色、主辅色面积或身份色误差的测量；只能解释整体色相分布。")}


def reassess_preflight(out_dir, items=None, plan=None, log=print):
    """离线预检：原图存在 + hash 对得上 + 实际请求在盘 + 方案条款可解析 + 本地统计。"""
    out = _isolated_out(out_dir)
    items = items or load_reassess_items()
    plan = plan or reassess_plan(items)
    specs = {key: json.loads(ref["spec"].read_text(encoding="utf-8"))
             for key, ref in _reassess_suite_refs().items()}
    rows, problems = [], []
    for row in items:
        image = Path(row["image"])
        actual = _sha256_file(image) if image.is_file() else None
        hash_ok = bool(actual) and (not row["image_sha256"] or actual == row["image_sha256"])
        prompt_ok = bool(row.get("prompt_text")) and bool(row.get("request_snapshot"))
        clauses_in_prompt = True
        if row["clauses"] and row.get("prompt_text"):
            clauses_in_prompt = all(clause in row["prompt_text"] for clause in row["clauses"])
        elif row["clauses"]:
            clauses_in_prompt = False
        entry = {"item_id": row["item_id"], "suite": row["suite"], "group_id": row["group_id"],
                 "cycle": row["round"], "image_exists": image.is_file(), "image_sha256": actual,
                 "recorded_sha256": row["image_sha256"], "hash_ok": hash_ok,
                 "actual_request_present": prompt_ok, "clauses_verbatim_in_request": clauses_in_prompt,
                 "clause_count": len(row["clauses"]),
                 "local_stats": _local_colour_stats(image) if image.is_file() else None}
        if not entry["image_exists"]:
            problems.append(f"{row['item_id']}: 原图缺失 {image}")
        elif not hash_ok:
            problems.append(f"{row['item_id']}: 原图 hash 与记录不一致（文件已被改动或记录过期）")
        if not prompt_ok:
            problems.append(f"{row['item_id']}: 缺实际请求快照（prompt.txt / request.json）")
        elif not clauses_in_prompt:
            problems.append(f"{row['item_id']}: 方案条款未在请求文本中逐字出现")
        rows.append(entry)
    covered = [item for chunk in plan["chunks"] for item in chunk["item_ids"]]
    duplicates = len(covered) != len(set(covered))
    if duplicates or sorted(covered) != sorted(r["item_id"] for r in items):
        problems.append("分块计划未恰好覆盖每个样本一次")
    result = {"label": REASSESS_LABEL, "at": _now(), "items": rows, "plan": plan,
              "counts": {"items": len(items), "chunks": len(plan["chunks"]),
                         "stage2": len([r for r in items if r["suite"] == "stage2"]),
                         "tone": len([r for r in items if r["suite"] == "tone"]),
                         "with_colour_scheme": len([r for r in items if r["clauses"]]),
                         "without_colour_scheme": len([r for r in items if not r["clauses"]])},
              "spec_ids": {key: value["id"] for key, value in specs.items()},
              "problems": problems, "status": "problems" if problems else "passed"}
    write_json_atomic(str(out / "preflight.json"), result)
    log(f"[reassess] 离线预检 {result['status']}：{len(items)} 张 / {len(plan['chunks'])} 块 / "
        f"{len(problems)} 个问题")
    for problem in problems:
        log(f"[reassess] 预检问题：{problem}")
    return result


def reassess_protocol(items, plan, preflight, model):
    """评价协议快照：冻结分块、评估字段与评价模型，不含任何旧评分/旧赢家/预期结论。"""
    specs = {key: json.loads(ref["spec"].read_text(encoding="utf-8"))
             for key, ref in _reassess_suite_refs().items()}
    chunks = []
    for chunk in plan["chunks"]:
        rows = [row for row in items if row["item_id"] in chunk["item_ids"]]
        chunks.append({"name": chunk["name"], "items": [
            {"neutral_id": f"S{i:02d}", "item": row["item_id"], "suite": row["suite"],
             "scheme_label": row["scheme_label"], "cycle": row["round"],
             "clauses_under_review": row["clauses"]} for i, row in enumerate(rows, 1)]})
    return {"label": REASSESS_LABEL,
            "at": _now(),
            "goal": ("验收目标＝画面是否呈现指定的一套色彩（颜色是否出现、基调/辅助/强调分工、"
                     "区域与冷暖、是否混入大块非方案颜色、受保护颜色是否保留），"
                     "不是画质改善，也不与基线比较好看程度。"),
            "not_shown_to_reviewer": ["旧评分", "旧赢家/排名", "旧报告预期结论", "协议分组标签（G/T 编号）"],
            "shown_to_reviewer": ["逐字方案条款", "保护块", "受保护颜色清单", "内容与画法检查项", "方案说明性标签"],
            "reviewer_model": model,
            "chunk_size": plan["chunk_size"],
            "chunks": chunks,
            "states": list(REASSESS_STATES),
            "protected": list(REASSESS_PROTECTED),
            "call_budget": {"business_calls_authorised": REASSESS_BUDGET,
                            "business_call_cap_this_run": REASSESS_CALL_CAP,
                            "planned_calls": len(plan["chunks"]),
                            "note": ("计划调用数 = 分块数；接口级重试由底层客户端计数，不计入业务调用；"
                                     "连续两次业务调用失败即停止并报告，不自动换模型或扩大预算")},
            "inputs": {"items": len(items), "preflight_status": preflight["status"],
                       "spec_ids": {key: value["id"] for key, value in specs.items()},
                       "run_dirs": {key: value["run_dir"] for key, value in REASSESS_SUITES.items()}},
            "local_statistics_role": "只作辅助证据；无区域蒙版，不用于判定多色组合是否正确",
            "limitations": ["每组三张独立随机生成，不构成同 seed 配对或显著性",
                            "无重绘/后续工序数据，不能宣称后续工序保留",
                            "评价模型看不到分组标签，但方案文本长度不同这一差异无法隐藏",
                            "本地色相统计是全局量，不能替代区域蒙版测量"]}


def _write_reassess_markdown(out: Path, protocol):
    lines = [f"# 配色方案符合度重评协议（{REASSESS_LABEL}）", "",
             f"- 生成时间：{protocol['at']}",
             f"- 评价模型：`{protocol['reviewer_model']}`",
             f"- 验收目标：{protocol['goal']}",
             f"- 分块：{len(protocol['chunks'])} 块，每块最多 {protocol['chunk_size']} 张，同方案三张不拆块",
             f"- 业务调用计划：{protocol['call_budget']['planned_calls']} / "
             f"{protocol['call_budget']['business_calls_authorised']}",
             f"- 不给评价模型看：{'、'.join(protocol['not_shown_to_reviewer'])}",
             f"- 必须给评价模型看：{'、'.join(protocol['shown_to_reviewer'])}", "",
             "## 分块与逐字方案", ""]
    for chunk in protocol["chunks"]:
        lines.append(f"### {chunk['name']}")
        for row in chunk["items"]:
            lines.append(f"- `{row['neutral_id']}` → {row['item']}（{row['scheme_label']}，第 {row['cycle']} 轮）")
            for i, clause in enumerate(row["clauses_under_review"], 1):
                lines.append(f"  - 条款 {i}：{clause}")
            if not row["clauses_under_review"]:
                lines.append("  - 本次无配色条款（基线段）")
        lines.append("")
    lines += ["## 限定", ""] + [f"- {text}" for text in protocol["limitations"]] + [""]
    path = out / "reassessment-protocol.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cached_record_compatible(cached, prompt_record_path, neutral, by_id):
    """早期提示词版本的记录能否复用：图片、逐字条款、保护清单与方案标签全部一致才复用。

    提示词措辞变化（例如为压缩输出长度收紧格式说明）不改变验收目标，允许复用缓存；
    任一实质内容变化都必须换新目录，不做静默复用。
    """
    if not prompt_record_path.is_file():
        return False
    try:
        previous = json.loads(prompt_record_path.read_text(encoding="utf-8"))
        _validate_reassess_response(cached["response"], list(neutral.values()),
                                    {neutral[k]: v["clauses"] for k, v in by_id.items() if k in neutral},
                                    REASSESS_PROTECTED)
    except Exception:  # noqa: BLE001
        return False
    previous_by_neutral = {item["neutral_id"]: item for item in previous.get("images") or []}
    for item_id, label in neutral.items():
        previous_item = previous_by_neutral.get(label)
        if not previous_item:
            return False
        if previous_item.get("item") != item_id:
            return False
        # 早期版本的请求快照没有这些字段：缺失时按「协议未变」处理，只有明确不一致才拒绝复用。
        if previous_item.get("colour_plan_clauses") is not None:
            if list(previous_item["colour_plan_clauses"]) != list(by_id[item_id]["clauses"]):
                return False
        if previous_item.get("protected_colours") is not None:
            if list(previous_item["protected_colours"]) != list(REASSESS_PROTECTED):
                return False
        if previous_item.get("colour_scheme_label") is not None:
            if previous_item["colour_scheme_label"] != by_id[item_id]["scheme_label"]:
                return False
    return True


def reassess_request_contract(request_record, by_id):
    """从某次调用自己保存的 `request.json` 还原它的编号→样本→条款契约。

    回收必须按**实际发出去的那次请求**校验，不能按当前计划的「三张一块」重新推断：
    2026-10-04 的 reassess-11 实际请求是六张（T2 三张 + T3 三张），
    用当前三张一块的计划去校验，必然误判为「编号数量不符」。
    """
    neutral_to_item, clauses = {}, {}
    missing = []
    for entry in request_record.get("images") or []:
        if entry.get("role") != "full":
            continue
        neutral_id = str(entry.get("neutral_id"))
        item_id = str(entry.get("item"))
        neutral_to_item[neutral_id] = item_id
        if item_id not in by_id:
            missing.append(item_id)
            clauses[neutral_id] = list(entry.get("colour_plan_clauses") or [])
            continue
        declared = entry.get("colour_plan_clauses")
        clauses[neutral_id] = list(declared) if declared is not None else list(by_id[item_id]["clauses"])
    return {"neutral_to_item": neutral_to_item, "clauses": clauses, "unknown_items": missing,
            "neutral_ids": sorted(neutral_to_item)}


def recover_reassessment(out_dir, names=None, *, request_paths=None, request_dir=None, only_missing=True,
                         log=print):
    """把已落盘但当时校验失败的原始响应按实际请求契约重新校验（**不发任何网络调用**）。

    - 映射与条款一律取自该次调用自己的 `request.json`；
    - `request_dir` 可把「请求快照的读取位置」指到别处（测试夹具 / 只读副本），
      被读的原始响应仍来自 `out_dir/vision`，因此**测试不会改写已交付的报告依据**；
    - 重复编号对象按编号合并，出现矛盾即保留为 `unclear` 并记录冲突，不选取有利的一份；
    - 已有有效记录默认不覆盖（`only_missing`），原始响应与调用记账保持不变。
    """
    out = _isolated_out(out_dir)
    by_id = {row["item_id"]: row for row in load_reassess_items()}
    vision = out / "vision"
    requests = Path(request_dir) if request_dir else vision
    request_paths = dict(request_paths or {})
    if names:
        candidates = list(names)
    else:
        candidates = sorted(path.stem.replace(".request", "") for path in requests.glob("*.request.json"))
    recovered, still_bad, skipped = [], [], []
    for name in candidates:
        raw_path = vision / f"{name}.raw.txt"
        request_path = Path(request_paths.get(name) or (requests / f"{name}.request.json"))
        target = vision / f"{name}.json"
        if not raw_path.is_file():
            still_bad.append({"name": name, "reason": f"缺原始响应：{raw_path}"})
            continue
        if not request_path.is_file():
            still_bad.append({"name": name, "reason": f"缺该次调用的实际 request.json：{request_path}（不按当前计划推断）"})
            continue
        if target.is_file() and only_missing:
            skipped.append({"name": name, "reason": "已有记录，未覆盖"})
            continue
        request_record = json.loads(request_path.read_text(encoding="utf-8"))
        contract = reassess_request_contract(request_record, by_id)
        if contract["unknown_items"]:
            still_bad.append({"name": name,
                              "reason": f"实际请求引用了未知样本：{contract['unknown_items']}"})
            continue
        if not contract["neutral_ids"]:
            still_bad.append({"name": name, "reason": "实际请求里没有可用的图片编号"})
            continue
        raw = raw_path.read_text(encoding="utf-8")
        value, parse_note = parse_object_tolerant(raw, log=log)
        if value is None:
            still_bad.append({"name": name, "reason": f"JSON 无法解析或修复：{parse_note}"})
            continue
        merged_ids = normalize_duplicate_ids(value)
        conflicts = sorted({item_id for row in (value.get("images") or [])
                            if isinstance(row, dict) and row.get("duplicate_conflicts")
                            for item_id in [str(row.get("id"))]})
        moved = normalize_reassess_response(value)
        try:
            _validate_reassess_response(value, list(contract["neutral_ids"]), contract["clauses"],
                                        REASSESS_PROTECTED)
        except Exception as error:  # noqa: BLE001
            still_bad.append({"name": name, "reason": str(error)})
            continue
        record = {"purpose": "reassess_read", "name": name, "at": _now(),
                  "model": request_record.get("model"),
                  "protocol_hash": request_record.get("protocol_hash"),
                  "media_resolution": request_record.get("media_resolution") or REASSESS_MAX_EDGE,
                  "recovered_offline": True,
                  "evidence_source": {"kind": "offline_recovery_of_paid_call",
                                      "paid_call_evidence": str(request_path),
                                      "raw_response": str(raw_path),
                                      "note": ("本条不是新调用：模型响应来自该次已付费调用落盘的 raw.txt，"
                                               "本次只重跑本地校验，未再发起网络请求")},
                  "contract_from_request": {"neutral_ids": contract["neutral_ids"],
                                            "clauses_source": "该次调用保存的 request.json",
                                            "items": contract["neutral_to_item"]},
                  "json_repair": parse_note, "observations_moved": moved,
                  "duplicate_ids_merged": merged_ids,
                  "duplicate_ids_conflicting": conflicts,
                  "image_order": [{"neutral_id": neutral_id, "item": contract["neutral_to_item"][neutral_id]}
                                  for neutral_id in contract["neutral_ids"]],
                  "mapping": dict(contract["neutral_to_item"]),
                  "response": value, "status": "review_record_not_gate"}
        write_json_atomic(str(target), record)
        recovered.append(name)
        log(f"[reassess] 离线回收 {name}（{len(record['mapping'])} 张："
            f"{', '.join(record['mapping'].values())}）"
            + (f"；{len(conflicts)} 条编号存在重复对象冲突，保留为无法判断" if conflicts else ""))
    result = {"recovered": recovered, "still_unusable": still_bad, "skipped": skipped,
              "note": ("离线回收不发网络调用；映射/条款取自该次调用的 request.json；"
                       "原始响应、失败记录与调用记账保持不变")}
    write_json_atomic(str(out / "recovery.json"), result)
    # 追加式回收台账：recovery.json 只保留最近一次，这里保留每一次回收的历史，便于复核。
    history_path = out / "recovery-log.jsonl"
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), "recovered": recovered,
                                 "still_unusable": still_bad, "skipped": skipped},
                                ensure_ascii=False) + "\n")
    return result


def _reassess_proxy(out: Path, image_path):
    """全图代理 + 下半身细节裁剪（厚底鞋/蝴蝶结/肢体太小，全图缩到代理尺寸会看不清）。"""
    from PIL import Image, ImageOps
    digest_name = _sha256_file(image_path)[:16]
    folder = out / "vision" / "proxies"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{digest_name}.jpg"
    if not dest.is_file():
        with Image.open(image_path) as image:
            ImageOps.contain(image.convert("RGB"), (REASSESS_MAX_EDGE, REASSESS_MAX_EDGE)).save(
                dest, quality=REASSESS_PROXY_QUALITY)
    detail = folder / f"{digest_name}-body.jpg"
    if not detail.is_file():
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            top = int(rgb.height * REASSESS_DETAIL_CROP[0])
            bottom = max(top + 1, int(rgb.height * REASSESS_DETAIL_CROP[1]))
            tile = rgb.crop((0, top, rgb.width, bottom))
            ImageOps.contain(tile, (REASSESS_DETAIL_EDGE, REASSESS_DETAIL_EDGE)).save(
                detail, quality=REASSESS_PROXY_QUALITY)
    return dest, detail


def _reassess_detail_note():
    return (f"每张图后面紧跟一张同一张图的下方 {int(REASSESS_DETAIL_CROP[0] * 100)}%"
            f"~{int(REASSESS_DETAIL_CROP[1] * 100)}% 细节裁剪，用于看厚底鞋、蝴蝶结与肢体可读性；"
            "细节裁剪不用于判断画幅或区域布局")


REASSESS_CLAUSE_KEYS = ("items", "clauses")


def _reassess_clause_entries(conformance):
    """模型的条款清单偶尔写成 `clauses` 而不是 `items`（2026-10-04 实测过一次），两种都接受。"""
    for key in REASSESS_CLAUSE_KEYS:
        value = conformance.get(key)
        if isinstance(value, list):
            return value
    return None


def _reassess_states_clash(first, second):
    """两份对象的状态是否算矛盾：只有「两边都给出明确但不同的结论」才算。

    `unclear`（明确的不确定/弃权）与另一份的明确结论不构成矛盾——把那份明确结论丢掉
    反而会制造假的不确定；两者都是 unclear 也不矛盾。
    """
    first, second = str(first), str(second)
    if first == second:
        return False
    if "unclear" in (first, second):
        return False
    return True


def _reassess_entries_conflict(entries_a, entries_b):
    """同一编号的两份条款条目是否互相矛盾（同一 index 的明确 status 不一致即算冲突）。"""
    by_index_a = {entry.get("index"): entry for entry in entries_a if isinstance(entry, dict)}
    conflicts = []
    for entry in entries_b:
        if not isinstance(entry, dict):
            continue
        other = by_index_a.get(entry.get("index"))
        if other is None:
            continue
        if _reassess_states_clash(other.get("status"), entry.get("status")):
            conflicts.append({"index": entry.get("index"), "field": "status",
                              "first_object": other.get("status"), "second_object": entry.get("status")})
    return conflicts


def normalize_duplicate_ids(value):
    """模型偶尔把一张图拆成两个对象、`id` 重复（2026-10-04 实测：12 个对象 / 6 个编号）。

    按编号合并，但**不做有利选取**：同一编号两份对象的条款状态或受保护颜色状态互相矛盾时，
    整条判定降为 `unclear`，并把两份原文分别保留在 `duplicate_objects` 与 `duplicate_conflicts` 里。
    没有矛盾的重复对象只做并集合并（补上对方独有的条款编号与颜色名）。
    """
    if not isinstance(value, dict) or not isinstance(value.get("images"), list):
        return []
    merged, duplicates, conflicted = [], [], []
    seen = {}
    for row in value["images"]:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id"))
        if item_id not in seen:
            seen[item_id] = row
            merged.append(row)
            continue
        duplicates.append(item_id)
        target = seen[item_id]
        target_conformance = target.setdefault("colour_conformance", {})
        row_conformance = row.get("colour_conformance") or {}
        current = _reassess_clause_entries(target_conformance) or []
        incoming = _reassess_clause_entries(row_conformance) or []
        conflicts = _reassess_entries_conflict(current, incoming)
        protected = target_conformance.setdefault("protected_colours", [])
        protected_incoming = row_conformance.get("protected_colours") or []
        by_name = {str(entry.get("name")): entry for entry in protected}
        for entry in protected_incoming:
            name = str(entry.get("name"))
            other = by_name.get(name)
            if other is None:
                protected.append(entry)
                by_name[name] = entry
                continue
            if _reassess_states_clash(other.get("status"), entry.get("status")):
                conflicts.append({"name": name, "field": "protected_colour_status",
                                  "first_object": other.get("status"), "second_object": entry.get("status")})
            elif str(other.get("status")) == "unclear" and str(entry.get("status")) != "unclear":
                # 一份弃权、一份给了结论：用那份结论，弃权不制造假冲突。
                other["status"] = entry.get("status")
                if not other.get("evidence"):
                    other["evidence"] = entry.get("evidence") or []
        have = {entry.get("index") for entry in current}
        for entry in incoming:
            other = next((item for item in current if item.get("index") == entry.get("index")), None)
            if other is None:
                current.append(entry)
                continue
            if _reassess_states_clash(other.get("status"), entry.get("status")):
                conflicts.append({"index": entry.get("index"), "field": "status",
                                  "first_object": other.get("status"), "second_object": entry.get("status")})
            elif str(other.get("status")) == "unclear" and str(entry.get("status")) != "unclear":
                other["status"] = entry.get("status")
        if current:
            target_conformance["items"] = current
        kept = target.setdefault("_duplicate_raw", [])
        kept.append({"status": row_conformance.get("status"),
                     "items": incoming,
                     "protected_colours": protected_incoming,
                     "observations": row.get("observations")})
        if conflicts:
            conflicted.append(item_id)
            target_conformance["status"] = "unclear"
            target["duplicate_conflicts"] = conflicts
            target["duplicate_conflict_note"] = ("同一编号的第二份对象与第一份在条款/受保护颜色状态上矛盾；"
                                                 "按协议保留为无法判断，不选取任一份作为结论。")
        for key, item in (row.get("observations") or {}).items():
            bucket = target.setdefault("observations", {}).setdefault(key, [] if isinstance(item, list) else "")
            if isinstance(item, list) and isinstance(bucket, list):
                bucket.extend(text for text in item if text not in bucket)
            elif not bucket:
                target["observations"][key] = item
    value["images"] = merged
    value["duplicate_ids_merged"] = sorted(set(duplicates))
    value["duplicate_ids_conflicting"] = sorted(set(conflicted))
    return sorted(set(duplicates))


def normalize_clause_indices(entries):
    """模型的条款编号偶尔从 0 开始。在内存里统一改成 1 起，原响应不动。

    判据：全部编号都是整数、且存在 0，就整体 +1。`[0,1,…,6]` 这种 0 起连号同样要移位，
    不能因为「里面也有 1」就当成已经是 1 起（2026-10-04 实测：这个判据漏掉了三组评审）。
    """
    if not entries:
        return entries, False
    indices = [entry.get("index") for entry in entries]
    if not all(isinstance(index, int) for index in indices):
        return entries, False
    if 0 not in indices:
        return entries, False
    for entry in entries:
        entry["index"] = entry["index"] + 1
    return entries, True


def _validate_reassess_response(value, expected_ids, expected_clauses, expected_protected):
    normalize_duplicate_ids(value)
    rows = value.get("images")
    if not isinstance(rows, list) or sorted(str(r.get("id")) for r in rows) != sorted(expected_ids):
        raise ValueError("重评必须恰好覆盖本块每个中性编号一次")
    for row in rows:
        item_id = str(row.get("id"))
        conformance = row.get("colour_conformance")
        if not isinstance(conformance, dict):
            raise ValueError(f"{item_id}: 缺少 colour_conformance")
        if conformance.get("status") not in REASSESS_STATES:
            raise ValueError(f"{item_id}: colour_conformance.status 非法")
        items = _reassess_clause_entries(conformance)
        expected_index = list(range(1, len(expected_clauses.get(item_id, [])) + 1))
        if items is None and not expected_index:
            items = []
        if not isinstance(items, list):
            raise ValueError(f"{item_id}: 缺少条款清单（items 或 clauses）")
        items, renumbered = normalize_clause_indices(items)
        if renumbered and items is not None:
            for key in REASSESS_CLAUSE_KEYS:
                if isinstance(conformance.get(key), list):
                    conformance[key] = items
        if sorted(int(entry.get("index", -1)) for entry in items) != expected_index:
            raise ValueError(f"{item_id}: 条款必须逐条覆盖且编号从 1 开始（期望 {expected_index}）")
        for entry in items:
            if entry.get("status") not in REASSESS_STATES:
                raise ValueError(f"{item_id}: 条款状态非法")
            if not str(entry.get("target") or "").strip():
                raise ValueError(f"{item_id}: 条款缺少 target 原话")
        protected = conformance.get("protected_colours")
        if not isinstance(protected, list):
            raise ValueError(f"{item_id}: 缺少 protected_colours")
        for entry in protected:
            if entry.get("status") not in ("retained", "altered", "unclear"):
                raise ValueError(f"{item_id}: 受保护颜色状态非法")
        names = {str(entry.get("name")) for entry in protected}
        if not set(expected_protected).issubset(names):
            raise ValueError(f"{item_id}: 受保护颜色未逐项报告")
        observations = row.get("observations")
        if not isinstance(observations, dict):
            raise ValueError(f"{item_id}: 缺少 observations")
    return value


def evaluate_reassessment(out_dir, *, spec_path=None, text_cfg=None, log=print, chunk_size=None,
                          dry_run=False, call_cap=None, only=None):
    """一次一块：只问方案执行 / 保护色 / 内容与画法，不问好看程度；命中缓存不重复收费。

    call_cap 是本目录本次运行允许新增的业务调用数（跨目录的历史调用另在报告里记账）；
    only 限定只跑指定块名（用于补跑失败或缺失的块，已有记录原样保留）。
    """
    out = _isolated_out(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from utils.analysis_gpt_prompt import call_text_model
    items = load_reassess_items()
    plan = reassess_plan(items, chunk_size=chunk_size)
    preflight = reassess_preflight(out, items, plan, log=log)
    cfg = _text_cfg(text_cfg)
    protocol = reassess_protocol(items, plan, preflight, cfg["model"])
    protocol_hash = digest({"protocol": protocol,
                            "prompt": _sha256_file(BASE / "prompts" / REASSESS_PROMPT_FILE)})
    protocol["protocol_hash"] = protocol_hash
    write_json_atomic(str(out / "reassessment-protocol.json"), protocol)
    _write_reassess_markdown(out, protocol)
    if dry_run:
        proxies = [(row["item_id"], *_reassess_proxy(out, row["image"])) for row in items]
        detail_note = _reassess_detail_note()
        detail_note = _reassess_detail_note()
        biggest = max((full.stat().st_size + detail.stat().st_size) for _, full, detail in proxies)
        typical = sorted(full.stat().st_size + detail.stat().st_size for _, full, detail in proxies)[len(proxies) // 2]
        log(f"[reassess] --dry-run：预检 {preflight['status']}，计划 {len(plan['chunks'])} 块，未发任何调用")
        log(f"[reassess] 本地代理已生成：{len(proxies) * 2} 张（全图 {REASSESS_MAX_EDGE}px + 下半身细节 "
            f"{REASSESS_DETAIL_EDGE}px）· 单张最大 {biggest // 1024}KB · 中位 {typical // 1024}KB · "
            f"协议明文单请求上限 {REASSESS_MAX_TOKENS} tokens")
        log(f"[reassess] {detail_note}")
        return {"protocol": protocol, "preflight": preflight, "records": [],
                "proxy_bytes": {"max_per_image": biggest, "median_per_image": typical}}
    system = read_prompt_file(REASSESS_PROMPT_FILE)
    specs = {key: json.loads(ref["spec"].read_text(encoding="utf-8"))
             for key, ref in _reassess_suite_refs().items()}
    by_id = {row["item_id"]: row for row in items}
    cap = int(call_cap or REASSESS_CALL_CAP)
    records, consecutive_failures, stopped = [], 0, None
    for chunk in plan["chunks"]:
        if only and chunk["name"] not in only:
            continue
        rows = [by_id[item_id] for item_id in chunk["item_ids"]]
        neutral = {row["item_id"]: f"S{i:02d}" for i, row in enumerate(rows, 1)}
        images = [path for row in rows for path in _reassess_proxy(out, row["image"])]
        payload = {"task": "colour_plan_conformance_review",
                   "detail_note": _reassess_detail_note(),
                   "images": [_reassess_payload(row, neutral[row["item_id"]], row["clauses"],
                                                specs[row["suite"]]) for row in rows],
                   "note": "images are supplied in exactly this order: each image is followed by its own body-detail crop"}
        user = json.dumps(payload, ensure_ascii=False)
        signature = digest({"system": system, "user": user, "model": cfg["model"],
                            "images": [{"name": Path(p).name, "sha256": _sha256_file(p)} for p in images]})
        name = chunk["name"]
        target = out / "vision" / f"{name}.json"
        if target.is_file():
            cached = json.loads(target.read_text(encoding="utf-8"))
            if cached.get("request_hash") == signature:
                try:
                    _validate_reassess_response(cached["response"],
                                                list(neutral.values()),
                                                {neutral[k]: v["clauses"] for k, v in by_id.items() if k in neutral},
                                                REASSESS_PROTECTED)
                except Exception as error:  # noqa: BLE001
                    _record_reassess_failure(out, name, "cached_validation", error)
                    log(f"[reassess] 缓存记录校验失败，按失败保留：{error}")
                    continue
                log(f"[cached] reassess {name}")
                records.append(cached)
                continue
            prompt_record = out / "vision" / f"{name}.request.json"
            if _cached_record_compatible(cached, prompt_record, neutral, by_id):
                cached.setdefault("reuse", {})["reused_under_prompt_revision"] = True
                cached["reuse"]["reused_at_hash"] = signature
                write_json_atomic(str(target), cached)
                log(f"[cached] reassess {name}（早期提示词版本，条款/图片/哈希一致，复用）")
                records.append(cached)
                continue
            raise ValueError(f"{name} 已有不同协议的记录；请换新的输出目录，旧记录保留")
        used = len(_reassess_vision_counts(out))
        if used >= cap:
            stopped = f"业务调用已达上限 {cap}，停止（不自动扩大预算）"
            break
        request_record = {"purpose": "reassess_read", "name": name, "task": payload["task"],
                          "protocol_hash": protocol_hash, "model": cfg["model"], "base_url": cfg["base_url"],
                          "system_prompt_file": REASSESS_PROMPT_FILE,
                          "system": system, "user": user,
                          "images": [{"file": Path(path).name,
                                      "role": ("full" if index % 2 == 0 else "body-detail"),
                                      "source": row["image"], "sha256": _sha256_file(path),
                                      "neutral_id": neutral[row["item_id"]], "item": row["item_id"]}
                                     for index, (path, row) in enumerate(
                                         zip(images, [item for item in rows for _ in ("full", "body-detail")]))],
                          "at": _now()}
        write_json_atomic(str(out / "vision" / f"{name}.request.json"), request_record)
        _record_vision_call(out, "reassess_read", name, cfg["model"])
        try:
            raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user,
                                  max_tokens=REASSESS_MAX_TOKENS, timeout=REASSESS_TIMEOUT,
                                  image_paths=[str(p) for p in images])
        except Exception as error:  # noqa: BLE001
            consecutive_failures += 1
            _record_reassess_failure(out, name, "interface", error)
            log(f"[reassess] {name} 接口失败（第 {consecutive_failures} 次连续失败）：{error}")
            if consecutive_failures >= 2:
                stopped = "连续两次业务调用接口失败，按用户要求停止并报告（不自动换模型、不扩大预算）"
                break
            continue
        (out / "vision" / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
        try:
            value, parse_note = parse_object_tolerant(raw, log=log)
            if value is None:
                raise ValueError(f"JSON 无法解析或修复：{parse_note}")
            moved = normalize_reassess_response(value)
            if moved:
                parse_note = f"{parse_note}+observations 抬回图片级（{', '.join(moved)}）"
            _validate_reassess_response(value, list(neutral.values()),
                                        {neutral[k]: v["clauses"] for k, v in by_id.items() if k in neutral},
                                        REASSESS_PROTECTED)
        except Exception as error:  # noqa: BLE001
            consecutive_failures += 1
            _record_reassess_failure(out, name, "parse_or_validate", error)
            log(f"[reassess] {name} 响应不可用（第 {consecutive_failures} 次连续失败）：{error}")
            if consecutive_failures >= 2:
                stopped = "连续两次业务调用无法得到可用响应，停止并报告"
                break
            continue
        consecutive_failures = 0
        record = {"request_hash": signature, "purpose": "reassess_read", "name": name, "at": _now(),
                  "model": cfg["model"], "protocol_hash": protocol_hash, "media_resolution": REASSESS_MAX_EDGE,
                  "json_repair": parse_note,
                  "images": images and [str(p) for p in images],
                  "mapping": {neutral[row["item_id"]]: row["item_id"] for row in rows},
                  "image_order": [{"neutral_id": neutral[row["item_id"]], "item": row["item_id"]}
                                  for row in rows],
                  "response": value, "status": "review_record_not_gate"}
        write_json_atomic(str(target), record)
        records.append(record)
        log(f"[reassess] {name} 完成：{', '.join(record['mapping'].values())}")
    summary = reassess_summary(out, items, protocol, records, stopped=stopped)
    return {"protocol": protocol, "preflight": preflight, "records": records, "summary": summary}


def _reassess_chunk_of(items, chunk_size=None):
    """按冻结的最细分块（一个方案一块）给出每个条目的归属块名，避免跨批次记录名互相污染。"""
    plan = reassess_plan(items, chunk_size=3 if chunk_size is None else chunk_size)
    mapping = {}
    for chunk in plan["chunks"]:
        for item_id in chunk["item_ids"]:
            mapping[item_id] = chunk["name"]
    return mapping


def _reassess_record_priority(record):
    """记录优先级：0 = 只服务一张图的付费调用，1 = 其他付费调用，2 = 离线回收。"""
    if record.get("recovered_offline"):
        return 2
    if len(record.get("mapping") or {}) == 1:
        return 0
    return 1


def _reassess_rows(items, records):
    """每个条目只取一条评价记录。

    优先级：① 只服务这一张图的付费调用（例如 T3 的独立三张调用）；
    ② 其他付费调用；③ 离线回收的记录（六图记录不会把 T3 那份独立付费记录压掉）。
    """
    by_item = {row["item_id"]: row for row in items}
    chunk_of = _reassess_chunk_of(items)
    candidates = {}
    for record in records:
        response_root = record.get("response")
        if not isinstance(response_root, dict) or not isinstance(response_root.get("images"), list):
            continue
        for response in response_root["images"]:
            item_id = record["mapping"][str(response["id"])]
            if item_id not in by_item:
                continue
            candidates.setdefault(item_id, []).append((record, response))
    rows = []
    for item_id, options in candidates.items():
        record, response = sorted(
            options,
            key=lambda pair: (_reassess_record_priority(pair[0]), str(pair[0].get("at") or "")))[0]
        source = by_item[item_id]
        conformance = response["colour_conformance"]
        evidence = record.get("evidence_source") or {}
        rows.append({"item_id": item_id, "suite": source["suite"], "group_id": source["group_id"],
                     "cycle": source["round"], "scheme_label": source["scheme_label"],
                     "clauses": source["clauses"], "chunk": chunk_of.get(item_id, record["name"]),
                     "recorded_at": record.get("at"),
                     "record_source": ("offline_recovery_of_paid_call" if record.get("recovered_offline")
                                       else "paid_vision_call"),
                     "evidence_source": evidence.get("kind") or ("paid_call_request_snapshot"
                                                                if not record.get("recovered_offline")
                                                                else "offline_recovery_of_paid_call"),
                     "evidence_ref": (evidence.get("paid_call_evidence")
                                      or f"vision/{record.get('name')}.request.json"),
                     "record_name": record.get("name"),
                     "record_priority": _reassess_record_priority(record),
                     "duplicate_conflicts": response.get("duplicate_conflicts") or [],
                     "colour_status": conformance["status"],
                     "scheme_applicable": bool(source["clauses"]),
                     "clause_rows": _reassess_clause_entries(conformance) or [],
                     "protected_colours": conformance["protected_colours"],
                     "observations": response["observations"]})
    rows.sort(key=lambda row: (row["suite"], row["group_id"], int(row["cycle"])))
    return rows


TWO_PERSON_MARKERS = ("两人", "两位人物", "第二个人物", "第二人物", "exactly one person", "双人")
ROW_SOURCE_LABELS = {
    "paid_vision_call": "本次评价的实际付费调用",
    "paid_call_request_snapshot": "本次评价的实际付费调用",
    "offline_recovery_of_paid_call": "该次已付费调用的落盘响应（离线回收，未再调用模型）",
}
CONTENT_VERDICT_LABELS = {
    "two_person_failure": "明确的双人失败",
    "content_failure": "其他明确内容失败",
    "pass_or_unverifiable": "未见明确内容问题（含裁剪分辨率无法确认的部分）",
}


def _content_failure(row):
    """只看内容项：型号数量相关的明确失败（双人）单独标出，不把"裁剪看不清"算成通过。"""
    texts = [str(text) for text in (row["observations"].get("content_issues") or [])]
    decisive = [text for text in texts if "无法" not in text and "不能确认" not in text]
    two_person = [text for text in texts if any(marker in text for marker in TWO_PERSON_MARKERS)]
    if two_person:
        return "two_person_failure", two_person
    if decisive:
        return "content_failure", decisive
    return "pass_or_unverifiable", texts


def _content_tally(rows):
    tally = {"pass_or_unverifiable": 0, "two_person_failure": 0, "content_failure": 0}
    detail = {}
    for row in rows:
        verdict, texts = _content_failure(row)
        tally[verdict] += 1
        detail[row["item_id"]] = {"verdict": verdict, "issues": texts}
    return tally, detail


def _tally(rows, key):
    counts = {state: 0 for state in REASSESS_STATES}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


def _scheme_stability(planned, subset, clause_rows, repeat_pattern):
    """方案稳定性的单一判据（可单独测试）。

    - 没有配色条款 → `not_applicable`（对照组不适用，绝不能写成 stable）；
    - 计划样本没有全部被覆盖 → `incomplete_coverage`（缺图不得判稳定）；
    - 计划全覆盖且判定一致时，才按条款看 `stable_conform` / `stable_partial` / `stable_diverge`；
    - 判定不一致 → `unstable`。
    """
    if not planned or not planned[0]["clauses"]:
        return "not_applicable"
    if len(subset) != len(planned):
        return "incomplete_coverage"
    if not repeat_pattern or len(set(repeat_pattern)) != 1:
        return "unstable"
    if repeat_pattern[0] == "conform" and clause_rows \
            and all(entry.get("status") == "conform" for entry in clause_rows):
        return "stable_conform"
    if repeat_pattern[0] == "diverge":
        return "stable_diverge"
    if repeat_pattern[0] == "partial":
        return "stable_partial"
    if repeat_pattern[0] == "unclear":
        return "stable_unclear"
    return "unstable"


def reassess_summary(out_dir, items=None, protocol=None, records=None, *, stopped=None, log=print):
    """按方案汇总：覆盖数、配色符合度、重复一致性与内容失败分别统计，不产生美学排行榜。

    三条硬性口径：
    ① 未评价的图片不计入任何"通过"，只进 `coverage.missing`；
    ② 没有配色条款的方案（G0/G1/T0 等）方案稳定性为 `not_applicable`，不是 `stable`；
    ③ 方案稳定性要求该组**计划样本全部被覆盖**；`verdict_consistent`（判定一致）与
       `scheme_applied`（符合方案）分开报告，不能互相替代。
    """
    out = _isolated_out(out_dir)
    items = items or load_reassess_items()
    if records is None:
        records = []
        for path in sorted((out / "vision").glob("reassess-*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    rows = _reassess_rows(items, records)
    covered = {row["item_id"] for row in rows}
    missing = [row["item_id"] for row in items if row["item_id"] not in covered]
    conflict_items = sorted({row["item_id"] for row in rows if row["duplicate_conflicts"]})
    groups = []
    for suite in ("stage2", "tone"):
        for group_id in sorted({row["group_id"] for row in items if row["suite"] == suite}):
            planned = [row for row in items if row["suite"] == suite and row["group_id"] == group_id]
            subset = [row for row in rows if row["suite"] == suite and row["group_id"] == group_id]
            if not planned:
                continue
            clause_rows = [entry for row in subset for entry in row["clause_rows"]]
            protected = [entry for row in subset for entry in row["protected_colours"]]
            counts = _tally(subset, "colour_status")
            repeat_pattern = [row["colour_status"] for row in subset]
            full_coverage = len(subset) == len(planned)
            verdict_consistent = bool(subset) and len(set(repeat_pattern)) == 1
            scheme_applicable = bool(planned[0]["clauses"])
            scheme_applied = (full_coverage and verdict_consistent and repeat_pattern[:1] == ["conform"]
                              and bool(clause_rows)
                              and all(entry.get("status") == "conform" for entry in clause_rows))
            scheme_stability = _scheme_stability(planned, subset, clause_rows, repeat_pattern)
            content_tally, content_detail = _content_tally(subset)
            groups.append({
                "suite": suite, "group_id": group_id, "label": planned[0]["scheme_label"],
                "planned_items": [row["item_id"] for row in planned],
                "images": [row["item_id"] for row in subset],
                "planned": len(planned), "reviewed": len(subset),
                "missing": [row["item_id"] for row in planned if row["item_id"] not in covered],
                "colour_status_counts": counts,
                "clause_total": len(clause_rows),
                "clause_conform": len([e for e in clause_rows if e.get("status") == "conform"]),
                "clause_partial": len([e for e in clause_rows if e.get("status") == "partial"]),
                "clause_diverge": len([e for e in clause_rows if e.get("status") == "diverge"]),
                "clause_unclear": len([e for e in clause_rows if e.get("status") == "unclear"]),
                "protected_counts": {state: len([e for e in protected if e.get("status") == state])
                                     for state in ("retained", "altered", "unclear")},
                "repeat_pattern": repeat_pattern,
                "repeat_consistent": verdict_consistent,
                "verdict_consistent": verdict_consistent,
                "scheme_applicable": scheme_applicable,
                "scheme_applied": scheme_applied,
                "scheme_stability": scheme_stability,
                "full_coverage": full_coverage,
                "content_verdicts": content_tally,
                "content_detail": content_detail,
                "duplicate_conflict_items": sorted({row["item_id"] for row in subset
                                                    if row["duplicate_conflicts"]}),
                "evidence_sources": sorted({row["evidence_source"] for row in subset}),
                "content_issues": sorted({text for row in subset for text in
                                          (row["observations"].get("content_issues") or [])}),
                "method_issues": sorted({text for row in subset for text in
                                         (row["observations"].get("method_issues") or [])}),
                "unsupported_claims": sorted({text for row in subset for text in
                                              (row["observations"].get("unsupported_claims") or [])}),
            })
    failures = []
    failure_path = out / "reassess-failures.jsonl"
    if failure_path.is_file():
        failures = [json.loads(line) for line in failure_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    content_tally, content_detail = _content_tally(rows)
    scheme_groups = [group for group in groups if group["scheme_applicable"]]
    result = {"label": REASSESS_LABEL, "at": _now(),
              "coverage": {"planned_items": len(items), "reviewed_items": len(rows),
                           "missing_items": missing, "complete": not missing,
                           "conflict_items": conflict_items},
              "reviewed_items": len(rows), "planned_items": len(items),
              "missing_items": missing,
              "colour_summary": {
                  "schemes": len(scheme_groups),
                  "schemes_with_full_coverage": len([g for g in scheme_groups if g["full_coverage"]]),
                  "schemes_stable_conform": len([g for g in scheme_groups
                                                 if g["scheme_stability"] == "stable_conform"]),
                  "schemes_stable_partial": len([g for g in scheme_groups
                                                 if g["scheme_stability"] == "stable_partial"]),
                  "schemes_incomplete_coverage": len([g for g in scheme_groups
                                                      if g["scheme_stability"] == "incomplete_coverage"]),
                  "schemes_not_applicable": len([g for g in groups if not g["scheme_applicable"]]),
              },
              "content_summary": {
                  "reviewed_items": len(rows),
                  "no_decisive_content_issue": content_tally["pass_or_unverifiable"],
                  "two_person_failure": content_tally["two_person_failure"],
                  "other_content_failure": content_tally["content_failure"],
                  "note": ("内容与配色分别汇总、互不抵消：内容项里明确的双人失败单列，"
                           "只有裁剪分辨率看不清的部分既不算通过也不算失败。"),
                  "detail": content_detail,
              },
              "stopped": stopped,
              "calls": reassess_call_usage(out),
              "failures": failures,
              "groups": groups,
              "rows": rows,
              "note": ("符合度是相对各自方案条款的执行判定，不是美学排名；"
                       "没有配色条款的基线段不按其他方案验收。")}
    write_json_atomic(str(out / "reassessment-summary.json"), result)
    write_json_atomic(str(out / "per-image-evaluations.json"), {"rows": rows, "missing": missing})
    for group in groups:
        log("[reassess] %s/%s：覆盖 %d/%d · 符合 %d / 部分 %d / 不符合 %d / 无法判断 %d（条款 %d，符合 %d）· "
            "方案稳定性 %s · 内容失败 双人 %d / 其他 %d" % (
                group["suite"], group["group_id"], group["reviewed"], group["planned"],
                group["colour_status_counts"].get("conform", 0),
                group["colour_status_counts"].get("partial", 0),
                group["colour_status_counts"].get("diverge", 0),
                group["colour_status_counts"].get("unclear", 0),
                group["clause_total"], group["clause_conform"], group["scheme_stability"],
                group["content_verdicts"]["two_person_failure"], group["content_verdicts"]["content_failure"]))
    log("[reassess] 覆盖 %d/%d · 配色符合方案 %d 组（其中全覆盖 %d）· 无配色条款 %d 组 · 内容双人失败 %d 张" % (
        result["coverage"]["reviewed_items"], result["coverage"]["planned_items"],
        result["colour_summary"]["schemes_stable_conform"],
        result["colour_summary"]["schemes_with_full_coverage"],
        result["colour_summary"]["schemes_not_applicable"],
        result["content_summary"]["two_person_failure"]))
    return result


def reassess_call_ledger(out_dir, *, log=print):
    """跨目录核对调用账本：按 request.json 的 (记录名, 时间, 内容) 去重，缺口如实记录。

    请求快照只能证明「发出过这次调用」，不等价于成功计费；被复制到另一个目录的同一份快照
    只记一次。无法核实的部分（底层 HTTP 重试次数、复制前的归属）写进 `verification.unknown`，
    不猜测计费次数。
    """
    out = _isolated_out(out_dir)
    rows = []
    seen = {}
    for folder in sorted(out.parent.glob("color-reassessment*/vision")):
        for request in sorted(folder.glob("reassess-*.request.json")):
            record = json.loads(request.read_text(encoding="utf-8"))
            raw = request.with_name(request.name.replace(".request.json", ".raw.txt"))
            fingerprint = digest({"at": record.get("at"), "user": record.get("user"),
                                  "system_sha": hashlib.sha256(
                                      str(record.get("system") or "").encode("utf-8")).hexdigest()})
            entry = {"run_dir": folder.parent.name,
                     "record": request.name.replace(".request.json", ""),
                     "at": record.get("at"), "model": record.get("model"),
                     "base_url": record.get("base_url"),
                     "images": len(record.get("images") or []),
                     "protocol_hash": str(record.get("protocol_hash"))[:16],
                     "has_raw_response": raw.is_file(),
                     "fingerprint": fingerprint[:16],
                     "duplicate_of": seen.get(fingerprint)}
            seen.setdefault(fingerprint, f"{folder.parent.name}/{entry['record']}")
            rows.append(entry)
    rows.sort(key=lambda row: str(row.get("at") or ""))
    unique = [row for row in rows if not row["duplicate_of"]]
    logged = set()
    for run in {row["run_dir"] for row in rows}:
        log_file = out.parent / run / "vision-calls.jsonl"
        if log_file.is_file():
            for line in log_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    logged.add((run, json.loads(line).get("name")))
    ledger = {"corrected_at": _now(),
              "rows": rows,
              "distinct_request_snapshots": len(unique),
              "verifiable_paid_calls": len(unique),
              "duplicate_snapshots": [row for row in rows if row["duplicate_of"]],
              "verification": {
                  "method": ("按 request.json 的 (时间, system, user) 指纹去重；同一份快照被复制到 "
                             "另一个目录时只记一次；请求快照只能证明发出过这次调用，不等于成功计费"),
                  "vision_calls_log_coverage": {run: len([1 for key in logged if key[0] == run])
                                                for run in sorted({row["run_dir"] for row in rows})},
                  "unknown": ("无法核实：底层 HTTP 重试的实际次数；被复制记录在复制前是否已被计入别处；"
                              "失败块那几次调用的请求快照未落盘。本轮离线修正不新增调用，"
                              "因此这些缺口不影响本次结论，也不做计费推断。"),
              }}
    write_json_atomic(str(out / "call-ledger.json"), ledger)
    summary_path = out / "reassessment-summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["call_ledger_file"] = "call-ledger.json"
        summary["call_ledger_summary"] = {
            "verifiable_paid_calls": ledger["verifiable_paid_calls"],
            "distinct_request_snapshots": ledger["distinct_request_snapshots"],
            "offline_recovery_runs": len([row for row in ledger["rows"] if row["run_dir"] == out.name
                                          and row["record"] == "reassess-11"]) + len(
                [row for row in ledger["rows"] if row["run_dir"] == out.name
                 and row["record"] == "reassess-13"]),
            "authorised_calls": REASSESS_BUDGET,
            "note": "离线修正（回收/汇总/报告）未新增任何网络调用。",
        }
        write_json_atomic(str(summary_path), summary)
    log(f"[reassess] 账本去重后 {ledger['verifiable_paid_calls']} 次可核实付费调用 / "
        f"{ledger['distinct_request_snapshots']} 份不同请求快照")
    return ledger


# ---------------------------------------------------------------------------
# 固定色板执行验证（color-fixed-palette-v1，2026-10-04）
#
# 目标：稳定执行「指定的颜色组合 + 指定的区域分配」，不是改善画质、不与基线比好看。
# 与第二阶段共用落盘/续跑/状态机（samples、suite.json），但协议是新的：
# 提示词只来自 spec 的逐字条款，本模块不重新编译、不改写在案条款。
# ---------------------------------------------------------------------------

PALETTE_SPEC_PATH = "prompts/color-knowledge/fixed-palette-v1.json"
PALETTE_LABEL = "color-fixed-palette-v1"
PALETTE_STATES = ("conform", "partial", "fail", "unverifiable")
PALETTE_RETRY_ENV = "IMAGE_MAKER_IMAGE_MAX_RETRIES"
PALETTE_CALL_ENV_TARGET = "0"
PALETTE_REVIEW_PROMPT_FILE = "color-knowledge/fixed-palette-review-system.md"


def load_palette_spec(path=None):
    """校验固定色板 spec：一通道、组内两版可对齐、色板色相确实写进条款、基线无条款。"""
    spec_path = Path(path or (BASE / PALETTE_SPEC_PATH))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if str(spec.get("kind")) != "palette":
        raise ValueError("固定色板 spec 的 kind 必须是 palette")
    for key in ("id", "subject", "rendering", "protection_block", "channels", "groups",
                "variants", "evaluation", "scale_options"):
        if not spec.get(key):
            raise ValueError(f"固定色板 spec 缺少 {key}")
    if len(spec["channels"]) != 1:
        raise ValueError("固定色板首轮只允许一个通道（Gemini 直接生成）")
    groups = spec["groups"]
    ids = [group.get("id") for group in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("组 id 必须唯一")
    for group in groups:
        if sum(map(len, group.get("clauses") or [])) > 1600:
            raise ValueError(f"{group['id']}: 条款超出单组预算")
        if group.get("variant") and group["variant"] not in spec["variants"]:
            raise ValueError(f"{group['id']}: 未知的条款版本 {group['variant']}")
    if any(not group.get("clauses") for group in groups) and \
            sum(1 for group in groups if not group.get("clauses")) != 1:
        raise ValueError("只允许一个无条款基线组")
    spec["_path"] = str(spec_path)
    return spec


PALETTE_CORES = {
    "P1": {"dominant": ("cool blue", "blue-grey"), "accent": ("red",)},
    "P2": {"dominant": ("deep green", "low-chroma green"), "accent": ("rose-red",)},
    "P3": {"dominant": ("blue-violet", "purple-grey"), "accent": ("gold",)},
}
PALETTE_FORBIDDEN_SUBSTITUTES = {
    "P1": ("pink", "rose", "magenta", "crimson"),
    "P2": ("pink",),
    "P3": ("orange", "yellow-orange"),
}
PALETTE_REGION_WORDS = ("waistband", "waist ribbon", "ribbon bows", "bow")
PALETTE_ACCENT_EXCLUSIVITY = ("only as a small accent", "placed only on the existing waistband")


def palette_variant_pairs(spec):
    """把同一色板的两个条款版本配成对：〔palette_id, simple 组, knowledge 组〕。"""
    pairs = {}
    for group in spec["groups"]:
        if not group.get("palette_id"):
            continue
        pairs.setdefault(group["palette_id"], {})[group.get("variant")] = group
    result = []
    for palette_id in sorted(pairs):
        versions = pairs[palette_id]
        result.append({"palette_id": palette_id,
                       "simple": versions.get("simple"),
                       "knowledge": versions.get("knowledge")})
    return result


def palette_group_checks(spec, group):
    """单组条款的自检：是否点名了该色板的主色与点缀色、是否禁用了替代色、是否指定区域。"""
    clauses = " ".join(group.get("clauses") or []).lower()
    palette_id = group.get("palette_id")
    findings = {"group_id": group["id"], "palette_id": palette_id, "problems": [], "notes": []}
    if not clauses:
        findings["notes"].append("本组没有配色条款（基线）：不适用色板自检")
        return findings
    core = PALETTE_CORES.get(palette_id)
    if not core:
        findings["problems"].append(f"未知色板 {palette_id}")
        return findings
    for word in core["dominant"]:
        if word not in clauses:
            findings["problems"].append(f"条款没有点名主色关键词「{word}」")
    for word in core["accent"]:
        if word not in clauses:
            findings["problems"].append(f"条款没有点名点缀色关键词「{word}」")
    for word in PALETTE_FORBIDDEN_SUBSTITUTES.get(palette_id, ()):
        if word not in clauses:
            continue
        if re.search(r"(?:not|never|instead of|rather than)\s+(?:\w+\s+){0,3}" + re.escape(word), clauses) \
                or re.search(re.escape(word) + r"\s+(?:is|are)\s+not\s+used", clauses):
            findings["notes"].append(f"条款已显式禁用替换色「{word}」，自检按通过处理")
            continue
        findings["problems"].append(f"条款里出现了可能被当作替换色使用的词「{word}」")
    if not any(word in clauses for word in PALETTE_REGION_WORDS):
        findings["problems"].append("条款没有指定点缀色的落点区域（腰带/鞋蝴蝶结）")
    if not any(word in clauses for word in PALETTE_ACCENT_EXCLUSIVITY):
        findings["notes"].append("点缀色没有用「only/placed only」这类排他措辞限定落点")
    if "second person" not in clauses:
        findings["notes"].append("条款没有显式禁止第二个人（保护块 M 里有，属可接受）")
    return findings


def palette_conflict_checks(spec):
    """颜色冲突与对象增删检查：保护色是否被要求改色、保护块与色板是否互相矛盾。"""
    protection = str(spec["protection_block"]).lower()
    findings = []
    for group in spec["groups"]:
        entry = {"group_id": group["id"], "problems": [], "notes": []}
        clauses = " ".join(group.get("clauses") or []).lower()
        if group.get("clauses"):
            for protected_word, label in (("silver-grey hair", "银灰发"), ("ivory-white dress", "象牙白洋装"),
                                          ("dark neutral platform shoes", "深中性色厚底鞋")):
                if protected_word in clauses:
                    entry["notes"].append(f"条款把 {label} 写成「保留」而不是改色，符合保护范围")
        if group.get("palette_id") is None and clauses:
            entry["problems"].append("无方案组不允许带配色条款")
        if group.get("palette_id") is not None:
            core = PALETTE_CORES[group["palette_id"]]
            if not any(word in protection for word in ("recolour only",)):
                entry["problems"].append("保护块没有写清「只允许给既有区域改色」")
            if core["accent"][0] not in protection:
                entry["notes"].append(
                    f"保护块没有单独提到点缀色 {core['accent'][0]}：改色授权由「recolour only the existing…」这一段承载")
            if "do not add objects" not in protection:
                entry["problems"].append("保护块没有禁止新增物件")
        findings.append(entry)
    return findings


def palette_request_preview(spec):
    """逐组给出真实请求预览（与生成时同一份组装函数），以及长度与段落结构。"""
    channel_key = next(iter(spec["channels"]))
    previews = []
    for group in spec["groups"]:
        request = build_palette_request(spec, group, channel_key)
        prompt = request["prompt"]
        previews.append({
            "group_id": group["id"], "label": group["label"], "variant": group.get("variant"),
            "clauses": len(group.get("clauses") or []),
            "prompt_chars": len(prompt),
            "clause_chars": sum(map(len, group.get("clauses") or [])),
            "sections": [line for line in prompt.split("\n\n") if line.strip()][:4],
            "prompt": prompt,
            "request": request,
        })
    lengths = [row["prompt_chars"] for row in previews]
    return {"channel": channel_key, "count": len(previews), "prompts": previews,
            "length_chars": {"min": min(lengths), "max": max(lengths)},
            "note": "预览与生成使用同一份 build_suite_request，落盘后即冻结"}


def palette_preflight(out_dir, *, spec_path=None, scale="full", log=print):
    """离线预检：组覆盖、色板自检、冲突检查、请求预览、调用与费用预算。不联网、不写产物。"""
    out = _isolated_out(out_dir)
    spec = load_palette_spec(spec_path)
    scales = spec["scale_options"]
    if scale not in scales:
        raise ValueError(f"未知规模 {scale}；可选 {sorted(scales)}")
    selected = list(scales[scale]["groups"])
    known = {group["id"] for group in spec["groups"]}
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise ValueError(f"规模 {scale} 引用了不存在的组：{unknown}")
    problems, notes = [], []
    group_checks = [palette_group_checks(spec, group) for group in spec["groups"]]
    conflict_checks = palette_conflict_checks(spec)
    for entry in group_checks + conflict_checks:
        for problem in entry.get("problems") or []:
            problems.append(f"{entry['group_id']}: {problem}")
    # 同一色板两版必须条数相同、覆盖同一批要求
    for pair in palette_variant_pairs(spec):
        simple, knowledge = pair["simple"], pair["knowledge"]
        if not simple or not knowledge:
            problems.append(f"{pair['palette_id']}: 缺少一个条款版本，无法做公平对照")
            continue
        if len(simple["clauses"]) != len(knowledge["clauses"]):
            notes.append(f"{pair['palette_id']}: 两版条数不同（简洁 {len(simple['clauses'])} / "
                         f"知识 {len(knowledge['clauses'])}），逐条对齐需要人工确认")
        for keyword in PALETTE_CORES[pair["palette_id"]]["accent"]:
            in_simple = any(keyword in clause.lower() for clause in simple["clauses"])
            in_knowledge = any(keyword in clause.lower() for clause in knowledge["clauses"])
            if in_simple != in_knowledge:
                problems.append(f"{pair['palette_id']}: 点缀色「{keyword}」只出现在一个版本里，对照不公平")
    preview = palette_request_preview(spec)
    selected_previews = [row for row in preview["prompts"] if row["group_id"] in selected]
    lengths = [row["prompt_chars"] for row in selected_previews] or [0]
    preview = {**preview, "prompts": selected_previews, "count": len(selected_previews),
               "length_chars": {"min": min(lengths), "max": max(lengths)},
               "note": f"预览与生成使用同一份请求组装函数；本次规模 {scale} 只列出选中的组"}
    selected_groups = [group for group in spec["groups"] if group["id"] in selected]
    images = len(selected_groups) * int(spec["rounds"])
    review_planned = scales[scale]["review_calls_planned"]
    review_contingency = scales[scale]["review_calls_contingency"]
    from modules.others.api_backend import get_api_config
    channel_key = next(iter(spec["channels"]))
    channel = spec["channels"][channel_key]
    try:
        cfg = get_api_config(api_type=channel["api_type"])
        host = urlparse(str(cfg.get("base_url") or "")).hostname
        key_source = cfg.get("_api_key_source")
        key_present = bool(cfg.get("api_key"))
        configured_model = cfg.get("model")
        backend_retries = cfg.get("max_retries")
    except Exception as error:  # noqa: BLE001
        host, key_source, key_present, configured_model, backend_retries = None, None, False, None, None
        problems.append(f"读取图片通道配置失败：{error}")
    if host and host != "new.aigc2d.com":
        problems.append(f"通道主机不是授权节点：{host}")
    if not key_present:
        problems.append("图片通道 API key 不可用（不会联网，但生成前必须解决）")
    if str(channel["model"]) != str(configured_model):
        notes.append(f"请求把模型固定为 {channel['model']}，而配置里当前是 {configured_model}："
                     f"按「新模型实验」记录，不宣称与 2026-10-03 批次可直接对比")
    if backend_retries:
        notes.append(f"底层 HTTP 重试配置为 {backend_retries}：生成时必须显式设 "
                     f"{PALETTE_RETRY_ENV}={PALETTE_CALL_ENV_TARGET}，否则收费调用数无法按产物数推算")
    budget = {
        "scale": scale, "groups": len(selected_groups), "rounds": int(spec["rounds"]),
        "planned_images": images, "generation_calls_upper_bound": images,
        "review_calls_planned": review_planned, "review_calls_contingency": review_contingency,
        "review_calls_upper_bound": review_planned + review_contingency,
        "retry_env": {PALETTE_RETRY_ENV: PALETTE_CALL_ENV_TARGET},
        "note": ("生成调用数上界 = 计划产物数（按目标数生成 + 关闭底层重试）；"
                 "评审按组计（每组 3 张一次调用），额外次数只用于事先约定的失败恢复或证据不足"),
    }
    budget["all_scales"] = {
        name: {"groups": len(option_row["groups"]),
               "images": len(option_row["groups"]) * int(spec["rounds"]),
               "generation_calls_upper_bound": len(option_row["groups"]) * int(spec["rounds"]),
               "review_calls_planned": option_row["review_calls_planned"],
               "review_calls_contingency": option_row["review_calls_contingency"],
               "review_calls_upper_bound": option_row["review_calls_planned"]
                                           + option_row["review_calls_contingency"]}
        for name, option_row in (spec.get("scale_options") or {}).items()}
    try:
        from utils.cost_estimate import estimate_gemini_repaint_cost, estimate_text_cost
        per_image = estimate_gemini_repaint_cost(model=str(channel["model"]))
        per_review = estimate_text_cost(prompt_chars=max(preview["length_chars"]["max"], 4000),
                                        completion_tokens=3000)
        budget["cost_estimate"] = {
            "generation_per_call_usd": round(per_image["usd"], 4),
            "generation_total_usd": round(per_image["usd"] * images, 4),
            "review_per_call_usd": round(per_review["usd"], 4),
            "review_total_usd": round(per_review["usd"] * (review_planned + review_contingency), 4),
            "total_upper_bound_usd": round(per_image["usd"] * images
                                          + per_review["usd"] * (review_planned + review_contingency), 4),
            "grouped_by": per_image.get("group"),
            "note": ("按系统内置/缓存的定价接口估算，只作预算参考；实际账单以服务商为准，"
                     "调用前如需精确价格需自行核对定价接口"),
        }
    except Exception as error:  # noqa: BLE001
        budget["cost_estimate"] = {"error": f"本地估算不可用：{error}"}
        notes.append("费用估算不可用（本地定价接口异常）：预算只按调用次数确认")
    result = {"label": PALETTE_LABEL, "at": _now(), "spec_id": spec["id"], "spec_path": spec["_path"],
              "out_dir": str(out), "scale": scale,
              "selected_groups": selected,
              "group_checks": group_checks, "conflict_checks": conflict_checks,
              "request_preview": preview, "budget": budget,
              "channel": {"key": channel_key, "api_type": channel["api_type"], "host": host,
                          "planned_model": channel["model"], "configured_model": configured_model,
                          "key_source": key_source, "key_present": key_present,
                          "resolution": channel["resolution"], "aspect_ratio": channel["aspect_ratio"],
                          "backend_max_retries": backend_retries},
              "problems": problems, "notes": notes,
              "status": "problems" if problems else "passed",
              "no_network": True}
    write_json_atomic(str(out / "palette-preflight.json"), result)
    log(f"[palette] 离线预检 {result['status']}：规模 {scale} / {len(selected_groups)} 组 / {images} 张 / "
        f"{len(problems)} 个问题 / {len(notes)} 条提示（未联网）")
    for problem in problems:
        log(f"[palette] 问题：{problem}")
    for note in notes:
        log(f"[palette] 提示：{note}")
    return result


def write_palette_plan(out_dir, preflight, log=print):
    """把预检结果写成可交接的执行说明（协议快照 + 命令 + 预算 + 待确认项）。"""
    out = _isolated_out(out_dir)
    spec = load_palette_spec(preflight["spec_path"])
    channel = preflight["channel"]
    scales = preflight["budget"]["all_scales"]
    cost = preflight["budget"].get("cost_estimate") or {}
    lines = [f"# 固定色板执行前预检（{PALETTE_LABEL}）", "",
             f"- 预检时间：{preflight['at']}（离线，未联网）",
             f"- spec：`{preflight['spec_id']}`（`{preflight['spec_path']}`）",
             f"- 本次预检规模：{preflight['scale']} · 组 {preflight['budget']['groups']} · "
             f"计划产物 {preflight['budget']['planned_images']} 张",
             f"- 通道：{channel['key']} / 请求模型 `{channel['planned_model']}` / 配置当前模型 "
             f"`{channel['configured_model']}` / {channel['resolution']} / {channel['aspect_ratio']} / 无参考图",
             f"- key：{channel['key_source']}（存在：{channel['key_present']}）",
             f"- 状态：**{preflight['status']}**（{len(preflight['problems'])} 个问题，"
             f"{len(preflight['notes'])} 条提示）", "",
             "## 1. 两种规模的调用与费用预算", "",
             "| 规模 | 组 | 计划产物 | 生成调用上界 | 评审（计划） | 评审（额外） | 评审上界 |",
             "|---|---|---|---|---|---|---|"]
    for name, row in scales.items():
        lines.append(f"| {name} | {row['groups']} | {row['images']} | "
                     f"{row['generation_calls_upper_bound']} | {row['review_calls_planned']} | "
                     f"{row['review_calls_contingency']} | {row['review_calls_upper_bound']} |")
    if cost and "error" not in cost:
        lines += ["", f"费用估算（本地定价接口，仅供参考）：生成约 {cost['generation_per_call_usd']} USD/张 · "
                      f"评审约 {cost['review_per_call_usd']} USD/次 · "
                      f"本次规模上界约 **{cost['total_upper_bound_usd']} USD**（分组 {cost.get('grouped_by')}）"]
    elif cost:
        lines += ["", f"费用估算不可用：{cost.get('error')}"]
    lines += ["", "## 2. 执行命令（尚未执行；确认后运行）", "",
              "```powershell",
              "# 默认 21 张矩阵（7 组 × 3 轮）",
              f"$env:{PALETTE_RETRY_ENV} = '{PALETTE_CALL_ENV_TARGET}'   # 关闭底层 HTTP 重试：收费调用数 = 产物数",
              f"& 'C:/Program Files/Python310/python.exe' -u tools/color_knowledge.py palette generate "
              f"--out {out} --scale full",
              "",
              "# 生成后按组评审：21 张 = 7 次调用",
              f"& 'C:/Program Files/Python310/python.exe' -u tools/color_knowledge.py palette review "
              f"--out {out} --scale full",
              "```", "",
              "缩减版（9 张 = 3 组 × 3 轮，只覆盖 P1 两版与 B0）：把上面两条命令的 `--scale full` 换成 "
              "`--scale reduced`，并**换一个新的输出目录**（规模属于协议的一部分）。", "",
              "## 3. 请求预览摘要", "",
              "| 组 | 版本 | 条款数 | 请求字符数 |", "|---|---|---|---|"]
    for row in preflight["request_preview"]["prompts"]:
        lines.append(f"| {row['group_id']} | {row['variant'] or '无条款'} | {row['clauses']} | "
                     f"{row['prompt_chars']} |")
    lines += [f"", f"请求长度区间：{preflight['request_preview']['length_chars']['min']} ~ "
                   f"{preflight['request_preview']['length_chars']['max']} 字符；完整正文见 "
                   f"`palette-preflight.json` 的 `request_preview.prompts[].prompt`，"
                   f"或用 `palette preview` 打印。", "",
              "## 4. 预检发现", ""]
    for problem in preflight["problems"]:
        lines.append(f"- 问题：{problem}")
    for note in preflight["notes"]:
        lines.append(f"- 提示：{note}")
    lines += ["", "## 5. 待确认项（生成前必须由用户确认）", "",
              "1. 规模：默认 `full`（21 张 / 7 组），备选 `reduced`（9 张 / 3 组：P1 两版 + B0）。",
              "2. 模型：请求里固定的是 spec 写的模型；若按配置当前模型跑，需要你确认改用哪一个，"
              "并接受「与 2026-10-03 批次不同模型、不直接对比」的标记。",
              f"3. 预算上界：生成 {preflight['budget']['generation_calls_upper_bound']} 次 + "
              f"评审 {preflight['budget']['review_calls_upper_bound']} 次；确认后再发起任何收费请求。",
              "4. 额外评审次数（默认 2 次，缩减版 1 次）只用于事先约定的失败恢复/证据不足，"
              "不反复询问直到给出满意评价。",
              "5. 本项目不会自动串行进入重绘/工序保留试验；那一步另开预检与预算。", ""]
    path = out / "palette-plan.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[palette] 执行说明 → {path}")
    return path


def palette_group_specs(spec, scale="full"):
    """按规模从 spec 里取要跑的组（默认 21 张 = 全部 7 组；缩减版 3 组 9 张）。"""
    scales = spec.get("scale_options") or {}
    if scale not in scales:
        raise ValueError(f"未知规模 {scale}；可选 {sorted(scales)}")
    selected = list(scales[scale]["groups"])
    groups = [group for group in spec["groups"] if group["id"] in selected]
    missing = [name for name in selected if name not in {group["id"] for group in groups}]
    if missing:
        raise ValueError(f"规模 {scale} 引用了不存在的组：{missing}")
    return scales[scale], groups


def palette_samples(spec, out_dir, scale="full"):
    """读该轮已经落盘的样本（复用第二阶段的 manifest 结构，不新建谱系）。

    还没有生成过（缺 `suite.json`）时返回空列表：预检/评审计划必须在生成前也能跑。
    """
    out = _isolated_out(out_dir)
    _, groups = palette_group_specs(spec, scale)
    ids = [group["id"] for group in groups]
    if not (out / "suite.json").is_file():
        return []
    manifest = _load_suite_manifest(out)
    rows = [sample for sample in manifest["samples"] if sample.get("group_id") in ids]
    rows.sort(key=lambda sample: (ids.index(sample["group_id"]), int(sample.get("round", 0))))
    return rows


def palette_group_prompt(spec, group):
    """固定色板请求正文：subject → rendering → COLOUR PLAN（有色板的组）→ PRESERVE（全部组同一份 M）。

    段落标题固定用 `COLOUR PLAN:`（英式拼写），与 `palette_request_preview` 和回归用例一致；
    不改写 spec 里的逐字条款，也不重新编译提示词。
    """
    header = ((spec.get("prompt_layout") or {}).get("headers") or {}).get("color_plan", "COLOR PLAN:")
    parts = [spec["subject"], spec["rendering"]]
    if group.get("clauses"):
        parts.append(f"{header}\n- " + "\n- ".join(group["clauses"]))
    if group.get("protection"):
        parts.append("PRESERVE:\n" + spec["protection_block"])
    return "\n\n".join(parts)


def build_palette_request(spec, group, channel_key):
    """固定色板的一条完整请求（含通道参数）；落盘与发包共用同一份。"""
    channel = spec["channels"][channel_key]
    request = {"channel": channel_key, "api_type": channel["api_type"], "model": channel["model"],
               "protocol": PALETTE_LABEL, "group_id": group["id"],
               "palette_id": group.get("palette_id"), "variant": group.get("variant"),
               "prompt": palette_group_prompt(spec, group), "image_paths": []}
    request.update({"resolution": channel["resolution"], "aspect_ratio": channel["aspect_ratio"],
                    "face_quality_boost": bool(channel.get("face_quality_boost", False))})
    return request


def palette_generation_preflight(spec, out_dir, scale="full"):
    """生成前的计划核对：逻辑样本数 = 收费图片调用数上界（按目标数生成）。"""
    option, groups = palette_group_specs(spec, scale)
    rounds = int(spec["rounds"])
    channel_key = next(iter(spec["channels"]))
    samples = [{"sample_id": f"{channel_key}-{group['id']}-r{round_number}",
                "group_id": group["id"], "round": round_number}
               for round_number in range(1, rounds + 1) for group in groups]
    return {"scale": scale, "groups": [group["id"] for group in groups], "rounds": rounds,
            "images": len(samples), "generation_calls_upper_bound": len(samples),
            "review_calls_planned": option["review_calls_planned"],
            "review_calls_contingency": option["review_calls_contingency"],
            "samples": samples,
            "note": ("逻辑样本数按「每张一次调用」计；生成前必须显式设 "
                     f"{PALETTE_RETRY_ENV}=0 关闭底层重试，否则请求次数上界会高于产物数")}


def run_palette(spec, out_dir, *, scale="full", rounds=None, retry_failed=False, dry_run=False, log=print):
    """生成首批固定色板图：复用第二阶段的落盘/续跑/状态机，只跑选中的组与单个通道。

    请求正文由 `build_palette_request`（COLOUR PLAN 标题）组装；第二阶段的
    `build_suite_request` 由本模块临时包一层，保证与预览完全一致。
    """
    option, groups = palette_group_specs(spec, scale)
    plan = palette_generation_preflight(spec, out_dir, scale)
    log(json.dumps(plan, ensure_ascii=False))
    if dry_run:
        return plan
    channel_key = next(iter(spec["channels"]))
    run_spec = {key: value for key, value in spec.items() if not key.startswith("_")}
    run_spec["groups"] = groups
    run_spec["channels"] = {channel_key: spec["channels"][channel_key]}
    run_spec["rounds"] = spec["rounds"]
    manifest = run_suite(run_spec, out_dir, rounds=rounds, channels=[channel_key],
                         retry_failed=retry_failed, log=log, stop_after_two_failures=True,
                         request_builder=build_palette_request)
    return {"plan": plan, "manifest": manifest}


def _palette_review_calls(out: Path, purpose="palette_review"):
    path = out / "palette-review-calls.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return [row for row in rows if row.get("purpose") == purpose]


def _palette_review_cached(out: Path, name):
    path = out / "review" / f"{name}.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return record if record.get("response") else None


def palette_review_plan(spec, out_dir, scale="full", *, extra=0):
    """评审计划：一组 3 张一次调用；额外次数只用于事先约定的失败恢复或证据不足。"""
    option, groups = palette_group_specs(spec, scale)
    out = _isolated_out(out_dir)
    samples = palette_samples(spec, out_dir, scale)
    rows = []
    for group in groups:
        subset = [sample for sample in samples if sample["group_id"] == group["id"]]
        usable = [sample for sample in subset if sample.get("status") == "success" and sample.get("outputs")]
        rows.append({"group_id": group["id"], "label": group["label"],
                     "planned": int(spec["rounds"]), "generated": len(usable),
                     "cached_review": bool(_palette_review_cached(out, group["id"])),
                     "call_needed": len(usable) > 0 and not _palette_review_cached(out, group["id"])})
    needed = len([row for row in rows if row["call_needed"]])
    return {"scale": scale, "groups": rows, "calls_needed_now": needed,
            "calls_planned": option["review_calls_planned"],
            "calls_contingency": option["review_calls_contingency"],
            "calls_extra_used": int(extra),
            "calls_upper_bound": option["review_calls_planned"] + option["review_calls_contingency"],
            "note": "评审调用按组计；每组的 3 张图在同一次调用里评审，缓存命中不重复收费"}


def palette_review_recover(out_dir, *, names=None, scale="full", log=print):
    """离线回收：用当前校验规则重新检查已落盘的评审原始响应（**不发任何网络调用**）。

    只在本地校验器修好（例如 0 起编号的判据）后使用：已付费得到的响应不该因为本地判据
    有缺陷就作废，也不该为此重复调用。
    """
    spec = load_palette_spec()
    out = _isolated_out(out_dir)
    _, groups = palette_group_specs(spec, scale)
    recovered, still_bad = [], []
    for group in groups:
        name = group["id"]
        if names and name not in names:
            continue
        raw_path = out / "review" / f"{name}.raw.txt"
        target = out / "review" / f"{name}.json"
        request_path = out / "review" / f"{name}.request.json"
        if target.is_file() or not raw_path.is_file():
            continue
        value, parse_note = parse_object_tolerant(raw_path.read_text(encoding="utf-8"), log=log)
        if value is None:
            still_bad.append({"name": name, "reason": f"JSON 无法解析：{parse_note}"})
            continue
        sample_ids = [sample["id"] for sample in palette_samples(spec, out_dir, scale)
                      if sample["group_id"] == name and sample.get("status") == "success"
                      and sample.get("outputs")]
        if not sample_ids:
            still_bad.append({"name": name, "reason": "没有成功产物，无法映射"})
            continue
        normalize_reassess_response(value)
        try:
            validate_palette_review(value, [f"Q{index:02d}" for index in range(1, len(sample_ids) + 1)],
                                    len(group.get("clauses") or []))
        except Exception as error:  # noqa: BLE001
            still_bad.append({"name": name, "reason": str(error)})
            continue
        record = {"name": name, "at": _now(), "recovered_offline": True, "json_repair": parse_note,
                  "evidence_source": {"kind": "offline_recovery_of_paid_review",
                                      "paid_call_evidence": str(request_path),
                                      "raw_response": str(raw_path),
                                      "note": "本条不是新调用：响应来自该次已付费评审落盘的 raw.txt，"
                                              "本次只重跑本地校验"},
                  "mapping": {f"Q{index:02d}": sample_id for index, sample_id in enumerate(sample_ids, 1)},
                  "response": value, "status": "review_record_not_gate"}
        write_json_atomic(str(target), record)
        recovered.append(name)
        log(f"[palette] 离线回收评审 {name}（{', '.join(sample_ids)}）")
    result = {"recovered": recovered, "still_unusable": still_bad,
              "note": "离线回收不发网络调用；原始响应与调用记账保持不变"}
    write_json_atomic(str(out / "palette-review-recovery.json"), result)
    return result


def validate_palette_review(value, expected_ids, clause_count):
    """评审响应必须逐张、逐条款覆盖；状态值必须来自协议枚举。"""
    rows = value.get("images")
    if not isinstance(rows, list) or sorted(str(row.get("id")) for row in rows) != sorted(expected_ids):
        raise ValueError("固定色板评审必须恰好覆盖本组每张图一次")
    for row in rows:
        item_id = str(row.get("id"))
        if row.get("colour_status") not in PALETTE_STATES:
            raise ValueError(f"{item_id}: colour_status 非法")
        if row.get("region_status") not in PALETTE_STATES:
            raise ValueError(f"{item_id}: region_status 非法")
        if str(row.get("protected_colours_ok")).lower() not in ("true", "false", "unclear"):
            raise ValueError(f"{item_id}: protected_colours_ok 非法")
        if str(row.get("content_ok")).lower() not in ("true", "false", "unclear"):
            raise ValueError(f"{item_id}: content_ok 非法")
        entries = _reassess_clause_entries(row)
        if clause_count and not isinstance(entries, list):
            raise ValueError(f"{item_id}: 缺少 clauses（条款清单）")
        if isinstance(entries, list) and clause_count:
            entries, _ = normalize_clause_indices(entries)
            row["clauses"] = entries
            if sorted(int(entry.get("index", -1)) for entry in entries) != list(range(1, clause_count + 1)):
                raise ValueError(f"{item_id}: 条款必须逐条覆盖且编号从 1 开始")
            for entry in entries:
                if entry.get("status") not in PALETTE_STATES:
                    raise ValueError(f"{item_id}: 条款状态非法")
    return value


def palette_review(out_dir, *, scale="full", text_cfg=None, extra=0, only=None, log=print):
    """一次一组：颜色组合、区域分配、额外色块、保护色、内容与方法逐张分开记录。"""
    from utils.analysis_gpt_prompt import call_text_model
    spec = load_palette_spec()
    out = _isolated_out(out_dir)
    plan = palette_review_plan(spec, out_dir, scale, extra=extra)
    log(json.dumps({key: value for key, value in plan.items() if key != "groups"}, ensure_ascii=False))
    cfg = _text_cfg(text_cfg)
    system = read_prompt_file(PALETTE_REVIEW_PROMPT_FILE)
    used = len(_palette_review_calls(out))
    upper = plan["calls_upper_bound"]
    records = []
    for group_row in plan["groups"]:
        group_id = group_row["group_id"]
        if only and group_id not in only:
            continue
        cached = _palette_review_cached(out, group_id)
        if cached:
            log(f"[cached] palette review {group_id}")
            records.append(cached)
            continue
        if not group_row["generated"]:
            log(f"[palette] {group_id}: 还没有可用产物，跳过评审（不调用）")
            continue
        if used >= upper:
            raise RuntimeError(f"评审调用已达上限 {upper}（含 {extra} 次额外）：停止，不自动扩大预算")
        group = group_by_id(spec, group_id)
        subset = [sample for sample in palette_samples(spec, out_dir, scale)
                  if sample["group_id"] == group_id and sample.get("status") == "success" and sample.get("outputs")]
        subset.sort(key=lambda sample: int(sample.get("round", 0)))
        images, order = [], []
        for index, sample in enumerate(subset, 1):
            images.extend(_reassess_proxy(out, sample["outputs"][0]))
            order.append({"id": f"Q{index:02d}", "sample_id": sample["id"], "round": sample["round"]})
        payload = {"task": "fixed_palette_conformance_review",
                   "detail_note": _reassess_detail_note(),
                   "palette_label": group["label"],
                   "palette_id": group.get("palette_id"),
                   "clause_version": group.get("variant"),
                   "colour_plan_clauses": list(group.get("clauses") or []),
                   "requirement_types": spec["evaluation"]["requirement_types"],
                   "protection_block": spec["protection_block"],
                   "protected_colours": ["银灰发 / natural silver-grey hair", "象牙白洋装 / ivory-white dress",
                                         "深中性色厚底鞋 / dark neutral platform shoes"],
                   "recolourable_regions": ["既有腰带 / existing waist ribbon sash",
                                            "鞋蝴蝶结 / existing shoe ribbon bows"],
                   "content_contract": ["单人", "象牙白洋装结构", "腰带可见", "双鞋与鞋蝴蝶结可见",
                                        "钓鱼动作", "双臂与双腿可读", "全身构图"],
                   "statuses": list(PALETTE_STATES),
                   "image_order": [{"id": row["id"]} for row in order],
                   "note": "images are supplied in exactly this order: each full image is followed by its own body-detail crop"}
        user = json.dumps(payload, ensure_ascii=False)
        signature = digest({"system": system, "user": user, "model": cfg["model"],
                            "images": [{"name": Path(p).name, "sha256": _sha256_file(p)} for p in images]})
        request_record = {"purpose": "palette_review", "name": group_id, "at": _now(),
                          "protocol": PALETTE_LABEL, "model": cfg["model"], "base_url": cfg["base_url"],
                          "system_prompt_file": PALETTE_REVIEW_PROMPT_FILE, "system": system, "user": user,
                          "images": [{"file": Path(path).name, "role": ("full" if index % 2 == 0 else "body-detail"),
                                      "sha256": _sha256_file(path)}
                                     for index, path in enumerate(images)]}
        write_json_atomic(str(out / "review" / f"{group_id}.request.json"), request_record)
        with open(out / "palette-review-calls.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"purpose": "palette_review", "name": group_id,
                                     "model": cfg["model"], "at": _now()}, ensure_ascii=False) + "\n")
        used += 1
        raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user,
                              max_tokens=REASSESS_MAX_TOKENS, timeout=REASSESS_TIMEOUT,
                              image_paths=[str(path) for path in images])
        (out / "review" / f"{group_id}.raw.txt").write_text(raw, encoding="utf-8")
        value, parse_note = parse_object_tolerant(raw, log=log)
        if value is None:
            write_json_atomic(str(out / "review" / f"{group_id}.failed.json"),
                              {"name": group_id, "reason": f"JSON 无法解析：{parse_note}", "at": _now()})
            log(f"[palette] {group_id}: 响应不可用（保留原始响应用于复核，不自动重试）")
            continue
        normalize_reassess_response(value)
        try:
            validate_palette_review(value, [row["id"] for row in order], len(group.get("clauses") or []))
        except Exception as error:  # noqa: BLE001
            write_json_atomic(str(out / "review" / f"{group_id}.failed.json"),
                              {"name": group_id, "reason": str(error), "at": _now()})
            log(f"[palette] {group_id}: 响应不满足协议（{error}），保留原文不自动重试")
            continue
        record = {"name": group_id, "at": _now(), "model": cfg["model"], "json_repair": parse_note,
                  "request_hash": signature, "mapping": {row["id"]: row["sample_id"] for row in order},
                  "image_order": order, "response": value, "status": "review_record_not_gate"}
        write_json_atomic(str(out / "review" / f"{group_id}.json"), record)
        records.append(record)
        log(f"[palette] {group_id} 评审完成：{', '.join(record['mapping'].values())}")
    summary = palette_review_summary(out_dir, spec=spec, scale=scale, log=log)
    return {"plan": plan, "records": records, "summary": summary}


def palette_review_summary(out_dir, *, spec=None, scale="full", log=print):
    """汇总两个指标：① 配色执行 ② 可用候选（配色符合且没有明确保护色或内容违约）。"""
    spec = spec or load_palette_spec()
    out = _isolated_out(out_dir)
    samples = {sample["id"]: sample for sample in palette_samples(spec, out_dir, scale)}
    rows = []
    for path in sorted((out / "review").glob("*.json")):
        if path.name.endswith(".request.json") or path.name.endswith(".failed.json"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for entry in record["response"].get("images") or []:
            sample_id = record["mapping"][str(entry.get("id"))]
            sample = samples.get(sample_id) or {}
            rows.append({"sample_id": sample_id, "group_id": sample.get("group_id"),
                         "round": sample.get("round"),
                         "colour_status": entry.get("colour_status") or entry.get("status"),
                         "region_status": entry.get("region_status"),
                         "protected_ok": entry.get("protected_colours_ok"),
                         "content_ok": entry.get("content_ok"),
                         "extra_colour_areas": entry.get("extra_colour_areas") or [],
                         "protection_issues": entry.get("protection_issues") or [],
                         "content_issues": entry.get("content_issues") or [],
                         "method_notes": entry.get("method_notes") or [],
                         "unverifiable_reason": entry.get("unverifiable_reason") or [],
                         "clauses": entry.get("clauses") or []})
    groups = []
    for group in spec["groups"]:
        subset = [row for row in rows if row["group_id"] == group["id"]]
        usable = [row for row in subset
                  if row["colour_status"] == "conform" and row["protected_ok"] is not False
                  and row["content_ok"] is not False]
        planned = int(spec["rounds"])
        complete = len(subset) == planned
        groups.append({
            "group_id": group["id"], "label": group["label"], "palette_id": group.get("palette_id"),
            "variant": group.get("variant"), "planned": planned, "reviewed": len(subset),
            "missing": [sample["id"] for sample in palette_samples(spec, out_dir, scale)
                        if sample["group_id"] == group["id"]
                        and sample["id"] not in {row["sample_id"] for row in subset}],
            "colour_counts": {state: len([row for row in subset if row["colour_status"] == state])
                              for state in PALETTE_STATES},
            "usable_candidates": len(usable),
            "full_coverage": complete,
            "promoted": bool(complete and len(usable) == planned),
            "content_issues": sorted({text for row in subset for text in row["content_issues"]}),
            "protection_issues": sorted({text for row in subset for text in row["protection_issues"]}),
            "extra_colour_areas": sorted({text for row in subset for text in row["extra_colour_areas"]}),
            "unverifiable": sorted({text for row in subset for text in row["unverifiable_reason"]}),
        })
    variant_pairs = []
    for pair in palette_variant_pairs(spec):
        simple = next((group for group in groups if group["group_id"] == (pair["simple"] or {}).get("id")), None)
        knowledge = next((group for group in groups
                          if group["group_id"] == (pair["knowledge"] or {}).get("id")), None)
        if simple and knowledge:
            variant_pairs.append({"palette_id": pair["palette_id"],
                                  "simple_usable": simple["usable_candidates"],
                                  "knowledge_usable": knowledge["usable_candidates"],
                                  "comparison": ("知识版更多" if knowledge["usable_candidates"] > simple["usable_candidates"]
                                                 else "简洁版更多" if simple["usable_candidates"] > knowledge["usable_candidates"]
                                                 else "两版相同")})
    result = {"label": PALETTE_LABEL, "at": _now(), "scale": scale,
              "review_calls": len(_palette_review_calls(out)),
              "groups": groups, "variant_pairs": variant_pairs, "rows": rows,
              "note": ("指标 ① 配色执行 = 逐张 conform/partial/fail/unverifiable；"
                       "指标 ② 可用候选 = 配色 conform 且没有明确保护色或内容违约的数量；"
                       "2/3 不能称为已稳定实现，3/3 也不构成跨主题跨模型的普遍可靠性证明")}
    write_json_atomic(str(out / "palette-summary.json"), result)
    write_json_atomic(str(out / "palette-per-image.json"),
                      {"rows": rows, "missing": [row["missing"] for row in groups]})
    for group in groups:
        log("[palette] %s：评审 %d/%d · 配色 %s · 可用候选 %d" % (
            group["group_id"], group["reviewed"], group["planned"],
            group["colour_counts"], group["usable_candidates"]))
    return result


def write_palette_gallery(out_dir, *, scale="full", log=print):
    """固定色板的可离线浏览页：内嵌实际产图、逐张判定、逐组汇总、调用账本。"""
    from html import escape
    spec = load_palette_spec()
    out = _isolated_out(out_dir)
    summary_path = out / "palette-summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file()
               else palette_review_summary(out, scale=scale, log=log))
    preflight_path = out / "palette-preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    samples = {sample["id"]: sample for sample in palette_samples(spec, out_dir, scale)}
    by_group = {}
    for row in summary["rows"]:
        by_group.setdefault(row["group_id"], []).append(row)
    review_calls = len(_palette_review_calls(out))
    generated = len([sample for sample in samples.values() if sample.get("status") == "success"])

    def tag(state):
        colours = {"conform": "ok", "partial": "warn", "fail": "bad", "unverifiable": "unk",
                   "true": "ok", "false": "bad", "unclear": "unk"}
        return colours.get(str(state), "unk")

    def card(row):
        sample = samples.get(row["sample_id"]) or {}
        outputs = sample.get("outputs") or []
        image_html = (f'<figure><img loading="lazy" src="{escape(_thumb_data_uri(outputs[0], 900, 82))}" '
                      f'alt="{escape(row["sample_id"])}"></figure>' if outputs else
                      '<figure class="missing">没有成功产物</figure>')
        request = sample.get("request") or {}
        clause_html = "".join(
            f'<li><b>条款 {entry.get("index")}：</b>{escape(str(entry.get("target") or "（未回写原文）"))}<br>'
            f'<span class="tag {tag(entry.get("status"))}">{escape(str(entry.get("status")))}</span> '
            f'区域：{escape("；".join(entry.get("regions") or []) or "未指明")} · '
            f'实际颜色：{escape("；".join(entry.get("observed_colours") or []) or "未记录")}<br>'
            f'证据：{escape("；".join(entry.get("evidence") or []))}</li>' for entry in row["clauses"])
        lists = [("额外显著色块", row["extra_colour_areas"]),
                 ("保护色问题", row["protection_issues"]),
                 ("内容问题", row["content_issues"]),
                 ("画法观察", row["method_notes"]),
                 ("不可验证原因", row["unverifiable_reason"])]
        detail_html = "".join(
            f'<p><b>{escape(title)}：</b>{escape("；".join(text for text in texts if text) or "—")}</p>'
            for title, texts in lists)
        clauses_text = "\n".join(f"- {clause}" for clause in (spec["groups"][0].get("clauses") and [])
                                 ) or ""
        group = next((item for item in spec["groups"] if item["id"] == row["group_id"]), {})
        clauses_text = "\n".join(f"- {clause}" for clause in group.get("clauses") or []) or "（本组无配色条款）"
        return (f'<section class="card"><h3>{escape(row["sample_id"])} · 第 {row["round"]} 轮 · '
                f'{escape(str(group.get("label") or row["group_id"]))} '
                f'<span class="tag {tag(row["colour_status"])}">配色 {escape(str(row["colour_status"]))}</span> '
                f'<span class="tag {tag(row["region_status"])}">区域 {escape(str(row["region_status"]))}</span></h3>'
                f'<div class="row">{image_html}'
                f'<div class="col"><details open><summary>本组逐字条款（请求里实际发送）</summary>'
                f'<pre>{escape(clauses_text)}</pre></details>'
                + (f'<details><summary>实际请求全文（{len(request.get("prompt") or "")} 字符）</summary>'
                   f'<pre>{escape(request.get("prompt") or "缺失")}</pre></details>' if request else "")
                + f'<details><summary>逐条判定</summary><ul>{clause_html}</ul></details></div></div>'
                f'<p>保护色：<span class="tag {tag(row["protected_ok"])}">{escape(str(row["protected_ok"]))}</span> · '
                f'内容：<span class="tag {tag(row["content_ok"])}">{escape(str(row["content_ok"]))}</span> · '
                f'产物：{escape(str(Path(outputs[0]).name) if outputs else "无")}</p>'
                f'{detail_html}</section>')

    group_blocks = []
    for group in summary["groups"]:
        rows = by_group.get(group["group_id"], [])
        sample_row = next((item for item in spec["groups"] if item["id"] == group["group_id"]), {})
        swatches = "".join(f'<span class="swatch" style="background:{escape(colour)}" title="{escape(colour)}"></span>'
                           for colour in sample_row.get("palette_preview") or [])
        group_blocks.append(
            f'<section class="group"><h2>{escape(group["group_id"])} · {escape(group["label"])}</h2>'
            f'<p>{escape(str(sample_row.get("intent") or ""))}</p>'
            f'<p>{swatches}<span class="muted">近似 HEX 仅供可视化参照，不是硬门禁</span></p>'
            f'<p><b>覆盖 {group["reviewed"]}/{group["planned"]}</b> ｜ 配色执行：'
            f'符合 {group["colour_counts"]["conform"]} / 部分 {group["colour_counts"]["partial"]} / '
            f'失败 {group["colour_counts"]["fail"]} / 不可验证 {group["colour_counts"]["unverifiable"]} ｜ '
            f'可用候选 {group["usable_candidates"]}/{group["planned"]} ｜ '
            f'<b>{"可进入工序保留试验" if group["promoted"] else "未达晋级条件"}</b></p>'
            + "".join(card(row) for row in sorted(rows, key=lambda r: str(r["round"])))
            + "</section>")

    tally = {state: len([row for row in summary["rows"] if row["colour_status"] == state])
             for state in PALETTE_STATES}
    region_tally = {state: len([row for row in summary["rows"] if row["region_status"] == state])
                    for state in PALETTE_STATES}
    channel = (preflight.get("channel") or {})
    header = (f'<h1>固定色板执行验证 · {generated}/21 张实拍逐张评价</h1>'
              f'<p class="muted">协议 {escape(PALETTE_LABEL)} · 请求模型 '
              f'{escape(str(channel.get("planned_model") or ""))} · '
              f'{escape(str(channel.get("resolution") or ""))} / {escape(str(channel.get("aspect_ratio") or ""))} · '
              f'无参考图 · 生成 {generated} 次调用 · 评审 {review_calls} 次调用（计划 7 + 额外 2）</p>'
              '<aside><b>判读口径</b><p>本轮只回答三件事：指定的颜色组合是否出现、是否落到指定区域、'
              '重复执行是否可靠。不做画质打分、不排美感名次、不与基线比好看。</p>'
              '<p><b>指标 ①配色执行</b>：逐张 conform / partial / fail / unverifiable。'
              f'本轮 21 张 = 符合 {tally["conform"]} / 部分 {tally["partial"]} / 失败 {tally["fail"]} / '
              f'不可验证 {tally["unverifiable"]}。</p>'
              '<p><b>指标 ②可用候选</b>：配色 conform 且没有明确保护色或内容违约；'
              '2/3 不能称为已稳定实现，3/3 也不是跨主题跨模型的普遍可靠性证明。</p>'
              '<p>区域分配与内容分开记：点缀色必须落在既有腰带与鞋蝴蝶结上，区域不可见记 unverifiable，'
              '不计成功；配色符合不能抵消内容失败。</p></aside>'
              f'<p>底色对照：区域分配本轮 = 符合 {region_tally["conform"]} / 部分 {region_tally["partial"]} / '
              f'失败 {region_tally["fail"]} / 不可验证 {region_tally["unverifiable"]}。'
              f'保护色：{len([r for r in summary["rows"] if r["protected_ok"] is True])}/21 通过；'
              f'内容：{len([r for r in summary["rows"] if r["content_ok"] is True])}/21 通过。</p>'
              f'<p><a href="palette-summary.json">palette-summary.json</a> · '
              f'<a href="palette-per-image.json">palette-per-image.json</a> · '
              f'<a href="palette-preflight.json">palette-preflight.json</a> · '
              f'<a href="palette-plan.md">palette-plan.md</a> · '
              f'<a href="protocol-freeze-audit.txt">protocol-freeze-audit.txt</a> · '
              f'<a href="suite.json">suite.json</a> · '
              f'<a href="palette-review-calls.jsonl">palette-review-calls.jsonl</a> · '
              f'<a href="palette-review-recovery.json">palette-review-recovery.json</a></p>')
    css = ("body{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#25322b;background:#eef1ee;margin:0}"
           "main{max-width:1180px;margin:auto;padding:24px}"
           "section.group{background:#fff;border-radius:14px;padding:22px;margin:20px 0}"
           "section.card{border:1px solid #e0e6e1;border-radius:10px;padding:16px;margin:14px 0;background:#fbfdfb}"
           ".row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1 1 380px;min-width:320px}"
           "figure{margin:0;flex:0 0 auto}figure img{max-width:520px;width:100%;border-radius:8px;border:1px solid #d8ded9}"
           "figure.missing{padding:40px;background:#fbe9ea;color:#a12e37;border-radius:8px}"
           ".tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}"
           ".tag.ok{background:#2f8f5b}.tag.warn{background:#c98a17}.tag.bad{background:#b23b45}.tag.unk{background:#6b7785}"
           "pre{white-space:pre-wrap;word-break:break-word;background:#f2f5f2;padding:10px;border-radius:8px;font-size:12.5px}"
           ".muted{color:#657269;font-size:13px}ul{padding-left:20px}details{margin:6px 0}"
           ".swatch{display:inline-block;width:44px;height:24px;border-radius:5px;border:1px solid #ddd;margin:3px}")
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>固定色板执行验证 2026-10-04</title>', f'<style>{css}</style><main>',
             header, "".join(group_blocks), '</main></html>']
    path = out / "palette-review.html"
    path.write_text("".join(parts), encoding="utf-8")
    log(f"[palette] 逐张评价页 → {path}")
    return path


# ---------------------------------------------------------------------------
# 固定色板 → 画风重绘保留验证（color-palette-retention-v1，2026-10-04）
#
# 只回答：首图的冷蓝基调与红色点缀在画风重绘后是否保留、是否仍落在腰带与两只鞋蝴蝶结上、
# 以及能否穿过现有身份/人体/质量/终审门禁。不评画质、不排名。
# A/B 的唯一干预是「B 追加配色继承合同」；模型与 api_type 必须显式传进后端。
# ---------------------------------------------------------------------------

RETENTION_SPEC_PATH = "prompts/color-knowledge/fixed-palette-retention-v1.json"
RETENTION_LABEL = "color-palette-retention-v1"
RETENTION_V2_SPEC_PATH = "prompts/color-knowledge/fixed-palette-retention-v2.json"
RETENTION_V2_LABEL = "color-palette-retention-v2"
RETENTION_CONTRACT_PATH = "prompts/color-knowledge/repaint-colour-contract-v1.md"
RETENTION_CONTRACT_SOURCE_PATH = "prompts/color-knowledge/repaint-colour-contract-v1-source.md"
RETENTION_COLOUR_REVIEW_PROMPT_FILE = "color-knowledge/retention-colour-review-system.md"
RETENTION_COLOUR_STATES = ("conform", "partial", "fail", "unverifiable")
RETENTION_ACCENT_STATES = ("present", "absent", "moved", "unverifiable")
# 合同正文的唯一来源：运行时的追加段由这几段拼成（派生自 source.md，避免两处文本漂移）。
RETENTION_CONTRACT_SECTIONS = (
    "SOURCE COLOUR CONTRACT (this experiment only; the first image is the authority):\n\n"
    "- The dominant environment colour of the source image is a cool blue palette on the river, sky, "
    "banks, distant foliage and shadow areas; keep that cool blue base and do not replace it with the "
    "reference image's palette.\n"
    "- The small accent colour of the source image is red, and it must stay red, on the existing wide "
    "waistband and on the ribbon bows of BOTH shoes. Do not move the accent to other objects.\n"
    "- Keep the accent small and concentrated on the waistband and the two shoe bows; do not spread it "
    "over the background.\n"
    "- Lock these subject colours: natural silver-grey hair, natural skin tone, ivory-white dress and "
    "dark neutral platform shoe body. Do not recolour them.\n"
    "- Do not add new red props, do not add a second person, and do not change the time of day or the "
    "weather.",
    "INHERITED FROM THE FIRST IMAGE'S COLOUR PLAN (these clauses, and only these, are inherited):\n\n"
    "- Environments must take the cool blue base of item 1.\n"
    "- The accent colour and its regions are items 2 and 3.\n"
    "- The locked subject colours are item 4.\n"
    "- Keep the subject count, the props and the time of day as in item 5.\n"
    "- The remaining clauses of the first image's plan are NOT inherited: anything about keeping the same "
    "drawing method, the same daylight or the same framing is deliberately left out, because this repaint "
    "is allowed to migrate the drawing method.",
    "STYLE MIGRATION PERMITTED (this experiment only):\n\n"
    "- The reference image may migrate line work, brushwork, edges, material rendering, simplification "
    "level and the abstract way faces and hair are drawn.\n"
    "- Do not transfer the reference character's identity, its own intrinsic colours, its clothing or "
    "its background content.",
    "WHEN THE TWO DISAGREE: the source colour contract above wins for the environment's dominant colour "
    "and for the accent colour and its regions; the reference image only supplies how the image is drawn.",
)
RETENTION_STEPS = {"repaint": {"enabled": True, "reference_mode": "style", "scope": "full",
                               "resolution": "2K"}}
RETENTION_AUDIT_KINDS = ("identity", "quality", "hands", "final_review")


def retention_contract_text():
    """B 支实际追加的合同正文（唯一来源）。"""
    return "\n\n".join(RETENTION_CONTRACT_SECTIONS)


def load_retention_spec(path=None):
    """校验并解析保留验证 spec（v1 原实验 / v2 修复实验共用）。"""
    spec_path = Path(path or (BASE / RETENTION_SPEC_PATH))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if str(spec.get("kind")) != "retention":
        raise ValueError("保留验证 spec 的 kind 必须是 retention")
    for key in ("id", "sources", "style", "generation", "branches", "gates", "budget", "evaluation"):
        if not spec.get(key):
            raise ValueError(f"保留验证 spec 缺少 {key}")
    if sorted((spec["branches"] or {}).keys()) != ["A", "B"]:
        raise ValueError("保留验证必须是 A/B 两支")
    if spec["branches"]["B"].get("intervention") != "append_colour_contract":
        raise ValueError("B 支的唯一干预必须是「追加配色继承合同」")
    if len(spec["sources"]) != 2:
        raise ValueError("本阶段固定用两张既定首图")
    if spec.get("out_dir"):
        spec["_out_dir"] = str(_retention_path(spec["out_dir"]))
    spec["_path"] = str(spec_path)
    return spec


def retention_run_dir(spec, out_dir=None):
    """实验输出目录：显式 --out 优先，其次 spec.out_dir（v2 用），最后回落到 v1 标签目录。

    `corrections_dir` 只是「修正交付」的去处，不能用它顶替实验目录 ——
    否则续跑会把产物写进修正目录（2026-10-04 实测踩过一次）。
    """
    if out_dir:
        return _resolve_retention_out(out_dir)
    if spec.get("_out_dir"):
        return _resolve_retention_out(spec["_out_dir"])
    return _resolve_retention_out(str(BASE / "data" / "test-result" / "20261004" / spec["id"]))


def _retention_path(value):
    path = Path(value)
    return path if path.is_absolute() else (BASE / path)


def retention_source_checks(spec):
    """源图与画风参考图的存在性 + hash 核对（冻结值对不上就停）。"""
    checks = []
    for item in spec["sources"]:
        path = _retention_path(item["image"])
        actual = _sha256_file(path) if path.is_file() else None
        checks.append({"kind": "source", "sample_id": item["sample_id"], "path": str(path),
                       "exists": path.is_file(), "sha256": actual,
                       "frozen_sha256": item.get("sha256"),
                       "hash_ok": bool(actual) and actual == item.get("sha256")})
    style = spec["style"]
    ref = _retention_path(style["ref_image"])
    ref_hash = _sha256_file(ref) if ref.is_file() else None
    checks.append({"kind": "style_ref", "path": str(ref), "exists": ref.is_file(),
                   "sha256": ref_hash, "frozen_sha256": style.get("ref_image_sha256"),
                   "hash_ok": bool(ref_hash) and ref_hash == style.get("ref_image_sha256")})
    contract = BASE / RETENTION_CONTRACT_SOURCE_PATH
    checks.append({"kind": "contract", "path": str(contract), "exists": contract.is_file(),
                   "sha256": _sha256_file(contract) if contract.is_file() else None,
                   "frozen_sha256": None, "hash_ok": contract.is_file(),
                   "text_sha256": retention_contract_fingerprint()["text_sha256"]})
    return checks


def _retention_style_entry(spec):
    styles = json.loads((BASE / "conf" / "config-styles.json").read_text(encoding="utf-8"))
    name = spec["style"]["name"]
    if name not in styles:
        raise ValueError(f"画风条目不存在：{name}")
    from utils.style_gpt import resolve_style_clauses
    from utils.styles import normalize_style_entry, style_ref_image
    entry = normalize_style_entry(styles[name])
    clauses, source = resolve_style_clauses(entry)
    if not clauses:
        raise ValueError(f"画风 {name} 没有可用的重绘渲染条款")
    if source != spec["style"].get("clauses_source") and spec["style"].get("clauses_source"):
        # 来源变化（手写条款被补上等）会改变请求，必须显式暴露而不是静默继续
        raise ValueError(f"画风 {name} 的渲染条款来源从 {spec['style']['clauses_source']} 变成了 {source}；"
                         "协议需要重新冻结")
    return {"name": name, "entry": entry, "clauses": clauses, "clauses_source": source,
            "ref_image": str(style_ref_image(styles, name) or "")}


def _retention_expected_text(spec, source_item):
    """身份/内容审计用的期望文本：直接取首图请求里实际发出的那段文字。

    查找顺序：① 首图所在目录的 `prompt.txt`（产物与请求文本同目录时）；
    ② 首图实验目录下的 `samples/gemini-flash/<样本>/prompt.txt`。
    """
    from utils.analysis_gen import build_gpt_image_request  # noqa: F401  仅确保依赖可用
    candidates = [Path(str(source_item["image"])).parent / "prompt.txt"]
    if spec.get("source_run"):
        candidates.append(_retention_path(spec["source_run"]) / "samples" / "gemini-flash"
                          / source_item["sample_id"] / "prompt.txt")
    for request_path in candidates:
        if request_path.is_file():
            return request_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"缺首图请求文本：{candidates[0]}")


def retention_requests(spec, *, style_entry=None, out_dir=None):
    """组装 A/B 两支的有效重绘请求（与发送共用同一份）。

    三条硬约束（2026-10-04 v1 就是因为第 1 条没做到而作废）：
    ① **B 的合同必须写进真正发送的 `call_kwargs.prompt`**，顶层 `prompt`、发送参数与
       `sent_prompt_sha256` 三者必须一致；
    ② 模型与 api_type 必须真的传进后端，而不是只写在快照里；
    ③ A/B 除合同与落盘标识外，输入与参数完全一致（同一首图、同一参考图顺序、同一模型/分辨率/比例）。
    """
    from utils import post_process as pp
    style = style_entry or _retention_style_entry(spec)
    generation = spec["generation"]
    contract = retention_contract_text()
    out = retention_run_dir(spec, out_dir)
    rows = []
    for item in spec["sources"]:
        source = str(_retention_path(item["image"]))
        for branch_id in ("A", "B"):
            cfg = dict(RETENTION_STEPS["repaint"])
            cfg["reference_mode"] = spec["style"]["reference_mode"]
            cfg["scope"] = spec["style"]["scope"]
            stage_name = f"{branch_id}-{item['sample_id']}-initial-repaint"
            request = pp.assemble_repaint_request(
                source, cfg, firmware=str(BASE / generation["firmware"]),
                style_ref_path=style["ref_image"], style_clauses=style["clauses"],
                # 落盘目录用**绝对路径**：相对路径会被后端按当前工作目录解析，产物可能落到
                # 真实产出目录 data/<日期>/ 而不是实验目录（2026-10-04 实测踩过一次）。
                sub_dir=str(out / "stages" / stage_name),
                prefix=f"retention-{branch_id}-{item['sample_id']}",
                work_dir=str(out))
            request["call_kwargs"]["model"] = generation["model"]
            request["call_kwargs"]["api_type"] = generation["api_type"]
            request["call_kwargs"]["repeat"] = int(generation.get("repeat", 1) or 1)
            request["call_kwargs"]["use_detail_suffix"] = bool(generation.get("use_detail_suffix", False))
            if branch_id == "B":
                sent_prompt = (request["call_kwargs"]["prompt"] or "").rstrip() + "\n\n" + contract + "\n"
                # ① 顶层与发送参数都必须换成同一份带合同的正文
                request["call_kwargs"]["prompt"] = sent_prompt
                request["prompt"] = sent_prompt
                request["colour_contract_appended"] = True
                request["colour_contract_source_path"] = RETENTION_CONTRACT_SOURCE_PATH
            else:
                request["colour_contract_appended"] = False
            if request["call_kwargs"]["prompt"] != request["prompt"]:
                raise ValueError(f"{branch_id}-{item['sample_id']}: 顶层 prompt 与 call_kwargs.prompt 不一致，"
                                 "拒绝继续（v1 事故）")
            request["prompt_sha256"] = hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest()
            request["prompt_chars"] = len(request["prompt"])
            request["sent_prompt_sha256"] = hashlib.sha256(
                request["call_kwargs"]["prompt"].encode("utf-8")).hexdigest()
            request["sent_prompt_chars"] = len(request["call_kwargs"]["prompt"])
            request["prompt_matches_sent"] = request["sent_prompt_sha256"] == request["prompt_sha256"]
            request["frozen_generation"] = {"model": generation["model"], "api_type": generation["api_type"],
                                            "resolution": request["resolution"],
                                            "aspect_ratio": request["aspect_ratio"],
                                            "repeat": int(generation.get("repeat", 1) or 1),
                                            "use_detail_suffix": bool(generation.get("use_detail_suffix", False))}
            request["sample_id"] = item["sample_id"]
            request["branch"] = branch_id
            request["source_sha256"] = item["sha256"]
            request["stage_name"] = stage_name
            request["request_id"] = f"{branch_id}-{item['sample_id']}"
            rows.append(request)
    return rows


def retention_contract_fingerprint():
    """合同正文 + 来源文件的指纹（冻结与复用判定都用它）。"""
    text = retention_contract_text()
    source = BASE / RETENTION_CONTRACT_SOURCE_PATH
    return {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_chars": len(text),
            "source_path": RETENTION_CONTRACT_SOURCE_PATH,
            "source_sha256": _sha256_file(source) if source.is_file() else None}


def retention_a_baseline_check(spec, *, out_dir=None):
    """核对旧 A 候选能否充当本次 A 基线：逐项 hash 比对，任一项不一致就不许沿用。"""
    check_cfg = spec.get("a_baseline_check") or {}
    result = {"enabled": bool(check_cfg.get("enabled")), "legacy_run": check_cfg.get("legacy_run", ""),
              "checked": [], "mismatches": [], "candidates": []}
    if not result["enabled"]:
        return result
    legacy = _retention_path(check_cfg["legacy_run"])
    current = {row["request_id"]: row for row in retention_requests(spec, out_dir=out_dir)}
    for item in spec["sources"]:
        sample_id = item["sample_id"]
        request_id = f"A-{sample_id}"
        stage = legacy / "stages" / f"A-{sample_id}-initial-repaint"
        legacy_result = stage / "result.json"
        legacy_request_path = stage / "request.json"
        row = {"sample_id": sample_id, "request_id": request_id,
               "legacy_result": str(legacy_result), "legacy_request": str(legacy_request_path)}
        if not legacy_result.is_file() or not legacy_request_path.is_file():
            result["mismatches"].append(f"{request_id}: 旧 A 的请求或产物记录缺失")
            result["candidates"].append(row)
            continue
        legacy_request = json.loads(legacy_request_path.read_text(encoding="utf-8"))
        legacy_record = json.loads(legacy_result.read_text(encoding="utf-8"))
        new_request = current.get(request_id) or {}
        field_pairs = [
            ("top_prompt_sha256", _sha256_str(legacy_request.get("prompt") or ""),
             _sha256_str(new_request.get("prompt") or "")),
            ("sent_prompt_sha256", _sha256_str((legacy_request.get("call_kwargs") or {}).get("prompt") or ""),
             _sha256_str((new_request.get("call_kwargs") or {}).get("prompt") or "")),
            ("source_sha256", legacy_request.get("source_sha256"), new_request.get("source_sha256")),
            ("model", (legacy_request.get("frozen_generation") or {}).get("model"),
             (new_request.get("frozen_generation") or {}).get("model")),
            ("api_type", (legacy_request.get("frozen_generation") or {}).get("api_type"),
             (new_request.get("frozen_generation") or {}).get("api_type")),
            ("resolution", legacy_request.get("resolution"), new_request.get("resolution")),
            ("aspect_ratio", legacy_request.get("aspect_ratio"), new_request.get("aspect_ratio")),
            ("repeat", (legacy_request.get("call_kwargs") or {}).get("repeat"),
             (new_request.get("call_kwargs") or {}).get("repeat")),
            ("use_detail_suffix", (legacy_request.get("call_kwargs") or {}).get("use_detail_suffix"),
             (new_request.get("call_kwargs") or {}).get("use_detail_suffix")),
            ("reference_paths",
             [r.get("path") for r in legacy_request.get("reference_images") or []],
             [r.get("path") for r in new_request.get("reference_images") or []]),
            ("reference_transmitted_sha256",
             [r.get("transmitted_sha256") for r in legacy_request.get("reference_images") or []],
             [r.get("transmitted_sha256") for r in new_request.get("reference_images") or []]),
            ("reference_mime",
             [r.get("mime") for r in legacy_request.get("reference_images") or []],
             [r.get("mime") for r in new_request.get("reference_images") or []]),
        ]
        for name, old, new in field_pairs:
            ok = old == new and old not in (None, "", [])
            result["checked"].append({"request_id": request_id, "field": name, "ok": ok,
                                      "legacy": _shorten(old), "current": _shorten(new)})
            if not ok:
                result["mismatches"].append(
                    f"{request_id} 的 {name} 不一致：旧 {_shorten(old)} vs 现在 {_shorten(new)}")
        if _sha256_str(legacy_request.get("prompt") or "") != \
                _sha256_str((legacy_request.get("call_kwargs") or {}).get("prompt") or ""):
            result["mismatches"].append(f"{request_id}: 旧 A 的顶层 prompt 与发送 prompt 不一致")
        outputs = [p for p in (legacy_record.get("outputs") or []) if Path(p).is_file()]
        if not outputs:
            outputs = [str(p) for p in sorted(stage.glob("*.jpg"))]
        if not outputs:
            result["mismatches"].append(f"{request_id}: 旧 A 产物文件不存在")
        else:
            actual = _sha256_file(outputs[0])
            recorded = (legacy_record.get("output_sha256") or {}).get(Path(outputs[0]).name)
            row.update({"candidate": str(Path(outputs[0]).resolve()), "sha256": actual,
                        "recorded_sha256": recorded, "hash_matches_record": bool(recorded) and actual == recorded})
            if recorded and actual != recorded:
                result["mismatches"].append(f"{request_id}: 旧 A 产物 hash 与记录不一致")
        result["candidates"].append(row)
    result["status"] = "mismatch" if result["mismatches"] else "ok"
    return result


def _sha256_str(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest() if text else ""


def _shorten(value, limit=48):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"


def retention_legacy_b_candidates(out: Path, spec):
    """旧实验里被误标成「带合同」的 B 产物：只作留档，不得再称为带合同组。"""
    configured = (spec.get("a_baseline_check") or {}).get("legacy_run") or ""
    legacy = _retention_path(configured) if configured else Path(out)
    rows = []
    if not legacy.is_dir():
        return rows
    for path in sorted(legacy.glob("stages/B-*-initial-repaint/request.json")):
        request = json.loads(path.read_text(encoding="utf-8"))
        result_path = path.parent / "result.json"
        record = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        outputs = [p for p in (record.get("outputs") or []) if Path(p).is_file()]
        if not outputs:
            outputs = [str(p) for p in sorted(path.parent.glob("*.jpg"))]
        sent_prompt = (request.get("call_kwargs") or {}).get("prompt") or ""
        record_name = str(record.get("sample_id") or request.get("sample_id") or "")
        if not record_name:
            folder = path.parent.name
            record_name = folder[len("B-"):-len("-initial-repaint")]
        rows.append({"sample_id": record_name,
                     "label": "v1 误标基线（顶层写了合同、实际发送未带合同）",
                     "top_prompt_chars": len(request.get("prompt") or ""),
                     "sent_prompt_chars": len(sent_prompt),
                     "top_sha256": _sha256_str(request.get("prompt") or ""),
                     "sent_sha256": _sha256_str(sent_prompt),
                     "contract_in_sent_prompt": "SOURCE COLOUR CONTRACT" in sent_prompt,
                     "candidate": outputs[0] if outputs else "",
                     "candidate_sha256": _sha256_file(outputs[0]) if outputs else None,
                     "usable_as_b": False})
    return rows


def retention_preflight(out_dir, *, spec_path=None, only_b=False, log=print):
    """执行前预检：源图/画风/hash、请求预览（含 A/B 差异）、后端传参核对、A 基线核对、预算与终止条件。"""
    out = retention_run_dir(load_retention_spec(spec_path), out_dir)
    spec = load_retention_spec(spec_path)
    checks = retention_source_checks(spec)
    problems = [f"{row['kind']} {row.get('sample_id') or ''} hash 不一致或缺失：{row['path']}"
                for row in checks if not row["hash_ok"]]
    notes = []
    style = _retention_style_entry(spec)
    requests = retention_requests(spec, style_entry=style, out_dir=out)
    planned_images = len([row for row in requests if row["branch"] == "B"]) if only_b else len(requests)
    # 后端传参核对：解析一次，确认实际会用的模型/api_type 就是冻结值
    from utils.gpt_image_optimize import load_config, resolve_repaint_call
    conf = load_config()
    dispatch_check = []
    for request in requests:
        kwargs = request["call_kwargs"]
        resolved = resolve_repaint_call(conf, source_path=str(kwargs["source_paths"][0]),
                                        model=kwargs.get("model"), resolution=kwargs.get("resolution"),
                                        aspect_ratio=kwargs.get("aspect_ratio"), repeat=kwargs.get("repeat"),
                                        prompt=kwargs.get("prompt"), use_detail_suffix=kwargs.get("use_detail_suffix"),
                                        save_sub_dir=kwargs.get("save_sub_dir"), file_prefix=kwargs.get("file_prefix"),
                                        api_type=kwargs.get("api_type"))
        ok = (resolved["model"] == spec["generation"]["model"]
              and resolved["api_type"] == spec["generation"]["api_type"])
        dispatch_check.append({"branch": request["branch"], "sample_id": request["sample_id"],
                               "kwargs_has_model": bool(kwargs.get("model")),
                               "kwargs_has_api_type": bool(kwargs.get("api_type")),
                               "resolved_model": resolved["model"], "resolved_api_type": resolved["api_type"],
                               "matches_frozen": ok, "resolution": resolved["resolution"],
                               "aspect_ratio": resolved["aspect_ratio"], "repeat": resolved["repeat"]})
        if not ok:
            problems.append(f"{request['branch']}-{request['sample_id']}: 后端解析出的模型/节点与冻结值不一致"
                            f"（{resolved['model']} / {resolved['api_type']}）")
    a_first = next(row for row in requests if row["branch"] == "A")
    b_first = next(row for row in requests if row["branch"] == "B")
    prefix_len = 0
    for left, right in zip(a_first["prompt"], b_first["prompt"]):
        if left != right:
            break
        prefix_len += 1
    contract_len = len(retention_contract_text())
    if prefix_len < len(a_first["prompt"]) - 8:
        notes.append(f"A/B 的公共前缀只有 {prefix_len} 字符，而 A 有 {len(a_first['prompt'])} 字符："
                     "B 的正文必须包含 A 的全文（追加段应只在末尾）")
    if b_first["colour_contract_appended"] and contract_len and \
            contract_len > len(b_first["prompt"]) - prefix_len:
        notes.append("B 的追加段比合同文本短：追加段可能被截断，请核对请求快照")
    for row in requests:
        if not row.get("prompt_matches_sent"):
            problems.append(f"{row['request_id']}: 顶层 prompt 与实际发送的 call_kwargs.prompt 不一致")
    refs_a = [row["path"] for row in a_first["reference_images"]]
    refs_b = [row["path"] for row in b_first["reference_images"]]
    if refs_a != refs_b:
        problems.append(f"A/B 的参考图顺序不一致：{refs_a} vs {refs_b}")
    if not a_first["extra_reference_paths"]:
        problems.append("A 没有挂上画风参考图（reference_mode=style 时应有两张参考：首图 + 画风图）")
    if spec["style"].get("motif_clauses_used"):
        notes.append("spec 声明使用 motif_clauses：本轮不应使用，请确认")
    worst = retention_generation_budget(spec, planned_images=planned_images)
    a_check = retention_a_baseline_check(spec, out_dir=out)
    if a_check["enabled"] and a_check["status"] != "ok":
        problems.extend([f"A 基线核对：{row}" for row in a_check["mismatches"]])
    result = {"label": spec["id"], "at": _now(), "spec_id": spec["id"], "spec_path": spec["_path"],
              "out_dir": str(out), "checks": checks, "style": {"name": style["name"],
                                                                "clauses_source": style["clauses_source"],
                                                                "clauses": style["clauses"],
                                                                "ref_image": style["ref_image"]},
              "contract": retention_contract_fingerprint(),
              "requests": [{key: value for key, value in row.items() if key != "call_kwargs"}
                           for row in requests],
              "request_diffs": retention_request_diff(requests),
              "a_baseline_check": a_check,
              "legacy_b_candidates": retention_legacy_b_candidates(out, spec),
              "dispatch_check": dispatch_check, "budget": worst,
              "stop_conditions": spec["budget"]["stop_conditions"],
              "problems": problems, "notes": notes,
              "status": "problems" if problems else "passed", "no_network": True}
    write_json_atomic(str(out / "retention-preflight.json"), result)
    log(f"[retention] 执行前预检 {result['status']}：{len(requests)} 条请求 / "
        f"{len(problems)} 个问题 / {len(notes)} 条提示（未联网）")
    for problem in problems:
        log(f"[retention] 问题：{problem}")
    for note in notes:
        log(f"[retention] 提示：{note}")
    return result


def retention_request_diff(requests):
    """A/B 逐对差异：除合同与落盘标识外，输入与参数必须完全一致。"""
    pairs = {}
    for request in requests:
        pairs.setdefault(request["sample_id"], {})[request["branch"]] = request
    rows = []
    for sample_id, both in pairs.items():
        a, b = both["A"], both["B"]
        same_inputs = (a["source_paths"] == b["source_paths"]
                       and [r["path"] for r in a["reference_images"]] == [r["path"] for r in b["reference_images"]]
                       and a["frozen_generation"] == b["frozen_generation"]
                       and a["resolution"] == b["resolution"] and a["aspect_ratio"] == b["aspect_ratio"])
        rows.append({"sample_id": sample_id,
                     "a_chars": a["prompt_chars"], "b_chars": b["prompt_chars"],
                     "delta_chars": b["prompt_chars"] - a["prompt_chars"],
                     "b_equals_a_plus_contract": b["prompt"] == a["prompt"].rstrip() + "\n\n" + retention_contract_text() + "\n",
                     "same_inputs_and_params": same_inputs,
                     "only_output_prefix_differs": True,
                     "b_prompt_sha256": b["prompt_sha256"], "a_prompt_sha256": a["prompt_sha256"],
                     "b_sent_prompt_sha256": b["sent_prompt_sha256"],
                     "b_top_matches_sent": b["prompt_matches_sent"]})
    return rows


def retention_generation_budget(spec, *, planned_images=None):
    """生成与审计的有限预算（实际请求硬上限，含所有隐式重试）。"""
    budget = spec["budget"]
    gates_needed = len(spec["sources"]) * 2 * len(RETENTION_AUDIT_KINDS)
    if planned_images is None:
        planned_images = len(spec["sources"]) * 2
    worst = {"image_hard_limit": int(budget["image_hard_limit"]),
             "image_planned_now": planned_images,
             "image_breakdown": budget.get("image_breakdown", ""),
             "text_hard_limit": int(budget["text_hard_limit"]),
             "text_breakdown": budget.get("text_breakdown", ""),
             "text_gates_needed_full": gates_needed,
             "text_gates_pending_after_reuse": budget.get("text_gates_pending", gates_needed),
             "text_colour_reviews": len(spec["sources"]),
             "text_planned_now": budget.get("text_planned_now", gates_needed + len(spec["sources"])),
             "retry_env": budget.get("env", {})}
    try:
        from utils.cost_estimate import estimate_gemini_repaint_cost, estimate_text_cost
        repaint = estimate_gemini_repaint_cost(model=spec["generation"]["model"])
        text = estimate_text_cost(prompt_chars=6000, completion_tokens=4000)
        worst["cost_estimate"] = {
            "repaint_per_call_usd": round(repaint["usd"], 6),
            "text_per_call_usd": round(text["usd"], 6),
            "source": "utils/cost_estimate（本地定价接口 + 缓存，分组 %s）" % repaint.get("group"),
            "estimated_total_usd": round(repaint["usd"] * planned_images
                                         + text["usd"] * worst["text_planned_now"], 4)}
    except Exception as error:  # noqa: BLE001
        worst["cost_estimate"] = {"error": str(error)}
    return worst


def _retention_freeze_check(out: Path, frozen: dict):
    """续跑前核对冻结记录：任何一项与当前不一致就拒绝复用并抛错（要求新协议/新目录）。"""
    differences = []
    spec = load_retention_spec(frozen.get("spec_path"))
    if _sha256_file(Path(spec["_path"])) != frozen.get("spec_sha256"):
        differences.append("spec 文件内容变化")
    if hashlib.sha256(retention_contract_text().encode("utf-8")).hexdigest() != \
            (frozen.get("contract") or {}).get("text_sha256"):
        differences.append("配色合同正文变化")
    style = _retention_style_entry(spec)
    if _sha256_file(Path(style["ref_image"])) != frozen.get("style_ref_sha256"):
        differences.append("画风参考图内容变化")
    clauses_sha = hashlib.sha256(json.dumps(style["clauses"], ensure_ascii=False,
                                            sort_keys=True).encode("utf-8")).hexdigest()
    if clauses_sha != frozen.get("style_clauses_sha256"):
        differences.append("画风渲染条款变化")
    requests = retention_requests(spec)
    current = {row["request_id"]: row["prompt_sha256"] for row in requests}
    if current != (frozen.get("request_hashes") or {}):
        for key in sorted(set(current) | set(frozen.get("request_hashes") or {})):
            if current.get(key) != (frozen.get("request_hashes") or {}).get(key):
                differences.append(f"请求 hash 变化：{key}")
    return differences


def retention_candidate_hashes(out: Path):
    """已落盘候选的 (branch, sample) → {path, sha256, request_id, prompt_sha256}。"""
    rows = {}
    for path in sorted((out / "stages").glob("*-initial-repaint/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record.get("branch"), record.get("sample_id"))
        for output in record.get("outputs") or []:
            if Path(output).is_file():
                rows[key] = {"path": str(Path(output).resolve()), "sha256": _sha256_file(output),
                             "request_id": record.get("request_id") or f"{key[0]}-{key[1]}",
                             "prompt_sha256": record.get("prompt_sha256"),
                             "stage": path.parent.name}
    return rows



def _retention_reuse_map(out: Path, spec, frozen, *, reuse_from=None):
    """按候选 hash + 请求 hash + 审计协议版本判断哪些审计结论可以复用（默认只读旧实验的成功审计）。"""
    source_dir = _retention_path(reuse_from) if reuse_from else None
    if not source_dir or not Path(source_dir).is_dir():
        return {}
    allowed = False
    for entry in (spec.get("audit_reuse") or {}).get("allowed_sources", []):
        if str(Path(source_dir)) == str(_retention_path(entry)):
            allowed = True
    if not allowed:
        raise ValueError(f"{source_dir}: 不在 spec 允许复用的来源列表里，拒绝复用审计结论")
    return _retention_reuse_scan(source_dir, spec)


def _retention_reuse_scan(source_dir: Path, spec):
    """扫描旧审计并逐条给出复用资格（含「指纹未验证」），供复用与核对共用。"""
    reuse = {}
    for path in sorted(Path(source_dir).glob("stages/*-gates/gates.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        candidate = record.get("candidate_sha256")
        branch, sample_id = str(record.get("branch")), str(record.get("sample_id"))
        if not candidate:
            continue
        for kind, audit in (record.get("audits") or {}).items():
            if audit.get("audit_error") or audit.get("budget_blocked"):
                continue
            eligibility = _retention_reuse_eligibility_for(
                spec, source_dir, kind=kind, branch=branch, sample_id=sample_id,
                candidate_sha=candidate, audit=audit, gates=record)
            reuse[(branch, sample_id, kind, candidate)] = {
                "audit": audit, "source": str(path), "reuse_eligibility": eligibility,
                "final_decision": (record.get("final_decision") if kind == "final_review" else None)}
    return reuse


def _retention_reuse_eligibility_for(spec, source_dir, *, kind, branch, sample_id, candidate_sha,
                                     audit, fingerprint=None, gates=None):
    """复用资格：逐项指纹核对；旧记录缺指纹就明确记「复用资格未验证」。"""
    fingerprint = fingerprint or retention_audit_fingerprint(
        _retention_path(str(source_dir)), spec, kind=kind, branch=branch, sample_id=sample_id,
        candidate_sha=candidate_sha)
    stored = retention_record_fingerprint({**audit, "kind": kind}, spec, gates)
    verdict = retention_reuse_eligibility(fingerprint, stored)
    verdict["current_fingerprint_sha256"] = fingerprint.get("fingerprint_sha256")
    verdict["stored_fingerprint"] = stored
    return verdict


# --- 缓存指纹规范 -----------------------------------------------------------
#
# 复用资格必须逐项可核对。旧记录缺指纹时只能记「复用资格未验证」，
# **不允许补造指纹**（不得把当前算出来的值写成"当时就是它"）。

RETENTION_FINGERPRINT_FIELDS = (
    "protocol_version", "audit_kind", "candidate_sha256", "source_image_sha256",
    "style_reference_sha256", "expected_spec_sha256", "prompt_sha256", "model", "params_sha256",
    "system_prompt_sha256",
)


def _sha256_if_file(value):
    path = Path(str(value or ""))
    return _sha256_file(path) if path.is_file() else None


def _file_origin(value):
    """文件来源备注：可证明是当次真实输入，还是本次按当前代码重建的代理。"""
    name = Path(str(value or "")).name
    return f"rebuilt_proxy_for_offline_correction:{name}" if value else None


def _retention_audit_prompt_fingerprint(kind):
    """该审计用的 system prompt 文件与 hash（文字全部在 prompts/ 里，不硬编码在代码里）。"""
    from utils.prompt_loader import read_prompt_file
    files = {"identity": "gpt-image-optimize/identity-audit-system.md",
             "quality": "gpt-image-optimize/refine-quality-audit-system.md",
             "hands": "gpt-image-optimize/hand-audit-system.md",
             "final_review": "gpt-image-optimize/final-quality-audit-system.md"}
    name = files.get(kind)
    if not name:
        return {"system_prompt_file": None, "system_prompt_sha256": None}
    text = read_prompt_file(name)
    return {"system_prompt_file": name,
            "system_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def retention_audit_user_payload(kind, expected_text, audit_images=()):
    """审计实际发出的 user 载荷的可重建表示（用于指纹；全文由各审计函数在 prompts/ 里组装）。"""
    return digest({"audit_kind": kind,
                   "expected_text_sha256": hashlib.sha256(str(expected_text or "").encode("utf-8")).hexdigest(),
                   "submitted_images": [{"role": row.get("role"), "sha256": row.get("sha256")}
                                        for row in (audit_images or [])]})


def retention_audit_fingerprint(out: Path, spec, *, kind, branch, sample_id, candidate_sha,
                                audit_images=(), audit_error="", produced_now=False,
                                user_payload=None):
    """一次门禁审计的完整指纹。

    字段含义（与 `RETENTION_FINGERPRINT_FIELDS` 一一对应）：
    - `prompt_sha256`：**审计实际发出的 user 载荷**（含图序说明与期望文本）；本批旧记录
      没有保存它，因此在旧记录里是缺失项 —— 缺了就记「复用资格未验证」，不补造。
    - `system_prompt_sha256`：该审计的 system prompt 文件 hash（同一判据的可核对依据）。
    """
    item = next((row for row in spec["sources"] if row["sample_id"] == sample_id), None) or {}
    source_path = str(_retention_path(item["image"])) if item else ""
    style_path = str(_retention_path(spec["style"]["ref_image"]))
    expected = ""
    if item:
        try:
            expected = _retention_expected_text(spec, item)
        except OSError:
            expected = ""
    cfg = {}
    try:
        cfg = _text_cfg()
    except Exception as error:  # noqa: BLE001
        # 离线核对时可能没有配置可用的文本接口：如实记「模型未知」，
        # 不猜一个模型名塞进指纹，也不因为缺配置而拒绝生成指纹。
        cfg = {"model": None, "config_error": f"{type(error).__name__}: {error}"}
    params = {"resolution": spec["generation"]["resolution"], "repeat": spec["generation"]["repeat"],
              "use_detail_suffix": spec["generation"].get("use_detail_suffix"),
              "reference_mode": spec["style"]["reference_mode"], "scope": spec["style"]["scope"]}
    fingerprint = {
        "protocol_version": _retention_protocol_version(spec),
        "audit_kind": kind,
        "candidate_sha256": candidate_sha,
        "source_image_sha256": _sha256_if_file(source_path) or item.get("sha256"),
        "style_reference_sha256": _sha256_if_file(style_path) or spec["style"].get("ref_image_sha256"),
        "expected_spec_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest() if expected else None,
        # 审计 user 载荷的指纹：能拿到实际载荷就用它，否则用可重建表示（期望文本 + 提交图片角色/hash）
        "prompt_sha256": (digest(user_payload) if user_payload is not None
                          else retention_audit_user_payload(kind, expected, audit_images)),
        "model": cfg.get("model"),
        "params_sha256": hashlib.sha256(json.dumps(params, ensure_ascii=False,
                                                   sort_keys=True).encode("utf-8")).hexdigest(),
        "input_evidence": {
            "images_actually_sent_saved": bool(produced_now),
            "note": ("本批审计的发送图片字节没有保存：当时只落了审计结论与输入路径/hash。"
                     "任何「复现」都只能按当前代码重建代理，不能冒充历史输入。"),
        },
    }
    fingerprint.update(_retention_audit_prompt_fingerprint(kind))
    fingerprint["input_evidence"]["images_actually_sent_saved"] = bool(produced_now)
    fingerprint["audit_error"] = audit_error or ""
    fingerprint["fingerprint_sha256"] = digest({key: fingerprint[key]
                                                for key in RETENTION_FINGERPRINT_FIELDS + ("system_prompt_sha256",)})
    return fingerprint


def _retention_protocol_version(spec):
    return f"{spec.get('id')}@v{spec.get('version', 1)}"


def retention_record_fingerprint(record, spec, gates=None):
    """记录里实际保存的指纹（缺哪项就明确缺哪项，不补造）。

    - `candidate_sha256` 通常只在 `gates.json` 上，所以允许从 gates 记录兜底读取；
    - `model` / `prompt_sha256` 等旧记录没有保存过，就如实留空。
    """
    gates = gates or {}
    stored = dict(record.get("fingerprint") or {})
    audit_hashes = [str(value) for value in (record.get("audit_image_sha256") or []) if value]
    stored.setdefault("audit_kind", record.get("kind") or record.get("audit_kind"))
    stored.setdefault("candidate_sha256", record.get("candidate_sha256") or gates.get("candidate_sha256"))
    # 旧记录没有单独的 source/style 字段，但保存了「本次审计实际提交的三张图的 hash 列表」：
    # 这是**记录当时就存下来的**证据，可以直接读，不算补造。
    if not stored.get("source_image_sha256"):
        stored["source_image_sha256"] = record.get("original_sha256") or (
            audit_hashes[0] if len(audit_hashes) > 0 else None)
    if not stored.get("style_reference_sha256"):
        stored["style_reference_sha256"] = record.get("style_reference_sha256") or (
            audit_hashes[2] if len(audit_hashes) > 2 else None)
    for key in ("expected_spec_sha256", "prompt_sha256", "params_sha256", "model", "protocol_version"):
        stored.setdefault(key, record.get(key) or gates.get(key))
    return stored


def retention_fingerprint_completeness(stored):
    """缺哪些指纹项——缺了就只能记「复用资格未验证」。"""
    missing = [key for key in RETENTION_FINGERPRINT_FIELDS if not stored.get(key)]
    return missing


def retention_reuse_eligibility(current, stored):
    """复用资格判定：完整匹配 / 指纹不全（未验证）/ 与当前不符。"""
    missing = retention_fingerprint_completeness(stored)
    if missing:
        return {"eligible": False, "state": "unverified_fingerprint",
                "missing": missing, "differences": [],
                "note": "旧记录缺少必要指纹，只能记「复用资格未验证」；不补造指纹、不自动联网补审计"}
    differences = [key for key in RETENTION_FINGERPRINT_FIELDS
                   if str(current.get(key)) != str(stored.get(key))]
    if differences:
        return {"eligible": False, "state": "mismatch", "missing": [], "differences": differences,
                "note": "相关输入或协议已变化，拒绝复用"}
    return {"eligible": True, "state": "verified_match", "missing": [], "differences": [],
            "note": "候选/首图/参考图/期望规格/提示词/模型/参数/种类/协议版本全部一致"}


def retention_colour_review_signature(spec, rows, system_text, cfg, *, images):
    """配色评审缓存签名：系统提示词、user 载荷、模型与图片（含角色与顺序）的 hash。"""
    payload = {"protocol_version": _retention_protocol_version(spec),
               "system_sha256": hashlib.sha256(system_text.encode("utf-8")).hexdigest(),
               "model": (cfg or {}).get("model"),
               "images": [{"path": str(row.get("path")), "role": row.get("role"),
                           "sha256": row.get("sha256")} for row in images],
               "items": [{"sample_id": row.get("sample_id"), "branch": row.get("branch")}
                         for row in rows]}
    return {"signature": digest(payload), "payload": payload}


def retention_review_cache_state(target: Path, spec, *, system_text, cfg, expected_images, rows):
    """只看结果文件在不在**不算复用资格**：必须重算签名并逐项比对。"""
    if not target.is_file():
        return {"state": "absent", "eligible": False, "differences": ["结果文件不存在"]}
    record = json.loads(target.read_text(encoding="utf-8"))
    stored = record.get("request_signature")
    expected = retention_colour_review_signature(spec, rows, system_text, cfg, images=expected_images)
    if not stored:
        return {"state": "unverified_fingerprint", "eligible": False,
                "expected_signature": expected["signature"],
                "differences": ["旧记录没有保存请求签名，无法验证复用资格"],
                "note": "不补造签名，不自动联网重评"}
    if isinstance(stored, str):
        same = stored == expected["signature"]
        return {"state": "verified_match" if same else "mismatch", "eligible": same,
                "expected_signature": expected["signature"], "stored_signature": stored,
                "differences": [] if same else ["请求签名不一致"]}
    differences = [key for key in expected["payload"]
                   if str(expected["payload"].get(key)) != str(stored.get(key))]
    return {"state": "verified_match" if not differences else "mismatch",
            "eligible": not differences, "expected_signature": expected["signature"],
            "stored_signature": stored, "differences": differences}


def _retention_audit_call(state, spec, *, kind, branch, sample_id, candidate_sha, call, log):
    """一次门禁审计：先看能否复用成功结论，否则在文本预算内新调用（不发图片请求）。"""
    key = (branch, sample_id, kind, candidate_sha)
    reused = state["reuse"].get(key)
    if reused:
        eligibility = reused.get("reuse_eligibility") or {}
        state["reused"].append({"branch": branch, "sample_id": sample_id, "audit": kind,
                                "candidate_sha256": candidate_sha, "source": reused["source"],
                                "reuse_eligibility": eligibility})
        state.setdefault("reuse_states", []).append(
            {"branch": branch, "sample_id": sample_id, "audit": kind,
             "state": eligibility.get("state", "unverified_fingerprint"),
             "missing": eligibility.get("missing"), "differences": eligibility.get("differences")})
        log(f"[retention] 复用 {branch}-{sample_id} 的 {kind} 审计（候选 hash 匹配；"
            f"复用资格：{eligibility.get('state', '未记录')}）")
        audit = reused["audit"]
        if kind == "hands" and isinstance(audit, dict) and "hands_clear" not in audit:
            # 按同一判据补算，避免「旧记录没有这个字段」被当成不通过
            audit = dict(audit)
            audit["hands_clear"] = _retention_hands_clear(audit)
        return audit, "reused"
    if state["mode"] == "audit_only" and state["used_text"] >= state["text_limit"]:
        state["blocked"].append({"branch": branch, "sample_id": sample_id, "audit": kind,
                                 "reason": "文本审计预算已用满"})
        return {"audit_error": "text budget exhausted before this audit",
                "budget_blocked": True, "candidate_sha256": candidate_sha}, "budget_blocked"
    state["used_text"] += 1
    try:
        audit = call()
        if isinstance(audit, dict):
            audit["candidate_sha256"] = candidate_sha
        state["fresh"].append({"branch": branch, "sample_id": sample_id, "audit": kind,
                               "candidate_sha256": candidate_sha})
        return audit, "fresh"
    except Exception as error:  # noqa: BLE001
        state["errored"].append({"branch": branch, "sample_id": sample_id, "audit": kind,
                                 "error": f"{type(error).__name__}: {error}"[:300]})
        return {"audit_error": f"{type(error).__name__}: {error}"[:300],
                "candidate_sha256": candidate_sha}, "audit_error"


def retention_freeze(out_dir, preflight=None, log=print, *, reuse_from=None):
    """冻结协议（含合同正文 hash、逐条请求 hash 与顶层/发送 hash 一致性）。"""
    out = _resolve_retention_out(out_dir)
    preflight = preflight or json.loads((out / "retention-preflight.json").read_text(encoding="utf-8"))
    spec = load_retention_spec(preflight["spec_path"])
    frozen = {
        "label": spec["id"], "frozen_at": _now(),
        "spec_path": spec["_path"], "spec_sha256": _sha256_file(Path(spec["_path"])),
        "contract": retention_contract_fingerprint(),
        "style_name": spec["style"]["name"],
        "style_ref_image": preflight["style"]["ref_image"],
        "style_ref_sha256": _sha256_file(Path(preflight["style"]["ref_image"])),
        "style_clauses_source": preflight["style"]["clauses_source"],
        "style_clauses_sha256": hashlib.sha256(
            json.dumps(preflight["style"]["clauses"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "generation": spec["generation"], "budget": preflight["budget"],
        "audit_reuse_allowed": (reuse_from or ""),
        "request_hashes": {row["request_id"]: row["prompt_sha256"] for row in preflight["requests"]},
        "request_sent_hashes": {row["request_id"]: row["sent_prompt_sha256"] for row in preflight["requests"]},
        "request_plan": {row["request_id"]: {
            "model": row["frozen_generation"]["model"], "api_type": row["frozen_generation"]["api_type"],
            "resolution": row["resolution"], "aspect_ratio": row["aspect_ratio"],
            "prompt_chars": row["prompt_chars"], "sent_prompt_chars": row["sent_prompt_chars"],
            "colour_contract_appended": row["colour_contract_appended"],
            "top_matches_sent": row["prompt_matches_sent"]} for row in preflight["requests"]},
    }
    frozen["freeze_hash"] = digest(frozen)
    write_json_atomic(str(out / "retention-frozen.json"), frozen)
    log(f"[retention] 协议已冻结：freeze_hash={frozen['freeze_hash'][:16]}（{len(frozen['request_hashes'])} 条请求）")
    return frozen


def _retention_prepare_env(out: Path, spec):
    from utils import send_budget
    os.environ["IMAGE_MAKER_SEND_BUDGET_DIR"] = str(out / "send-budget")
    for key, value in spec["budget"]["env"].items():
        os.environ[key] = str(value)
    return send_budget


def retention_audit_only(out_dir, *, spec_path=None, reuse_from=None, log=print):
    """只补审计：不发任何图片请求，逐候选跑门禁链，缺什么补什么，可复用匹配的成功审计。"""
    spec = load_retention_spec(spec_path)
    out = retention_run_dir(spec, out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _retention_prepare_env(out, spec)
    frozen_lock = out / "retention-frozen.json"
    if frozen_lock.is_file():
        differences = _retention_freeze_check(out, json.loads(frozen_lock.read_text(encoding="utf-8")))
        if differences:
            raise ValueError("冻结记录与当前协议不一致，拒绝复用；请新开目录/新协议：" + "；".join(differences))
    reuse = _retention_reuse_map(out, spec, None, reuse_from=reuse_from)
    state = {"reuse": reuse, "reused": [], "fresh": [], "blocked": [], "errored": [],
             "mode": "audit_only", "used_text": 0,
             "text_limit": int(spec["budget"]["text_limit_for_gates"])}
    log(f"[retention] 只补审计：候选 {len(retention_candidate_hashes(out))} 个 · 可复用审计 {len(reuse)} 条 · "
        f"文本上限 {state['text_limit']}")
    records = []
    for branch in ("A", "B"):
        for item in spec["sources"]:
            records.append(_retention_gate_stages(out, spec, branch, item, state, reuse_from, log=log))
    result = {"label": spec["id"], "at": _now(), "mode": "audit_only",
              "reused": state["reused"], "fresh": state["fresh"], "blocked": state["blocked"],
              "errored": state["errored"], "text_requests": state["used_text"],
              "text_limit": state["text_limit"], "image_requests": 0,
              "gate_records": records}
    write_json_atomic(str(out / "retention-audit-only.json"), result)
    log(f"[retention] 只补审计完成：新调用 {state['used_text']} / {state['text_limit']}，"
        f"复用 {len(state['reused'])} 条，预算阻止 {len(state['blocked'])} 条，审计错误 {len(state['errored'])} 条")
    return result


def _retention_inherit_candidate(out: Path, spec, branch, sample_id, candidate, *, log=print):
    """把一个已有产物登记成本次实验的候选（复用它不产生任何请求），并写清来源。"""
    folder = _retention_stage_dir(out, branch, sample_id, "initial-repaint")
    sha = _sha256_file(candidate)
    existing = folder / "result.json"
    if existing.is_file():
        record = json.loads(existing.read_text(encoding="utf-8"))
        if record.get("status") == "success":
            return record
    request = next((row for row in retention_requests(spec, out_dir=out)
                    if row["branch"] == branch and row["sample_id"] == sample_id), {})
    record = {"stage": "initial-repaint", "branch": branch, "sample_id": sample_id, "status": "success",
              "outputs": [str(Path(candidate).resolve())],
              "output_sha256": {Path(candidate).name: sha},
              "at": _now(), "produced": False, "inherited": True,
              "frozen_generation": request.get("frozen_generation"),
              "request_id": request.get("request_id"), "stage_name": request.get("stage_name"),
              "prompt_sha256": request.get("prompt_sha256"),
              "sent_prompt_sha256": request.get("sent_prompt_sha256"),
              "prompt_chars": request.get("prompt_chars"), "sent_prompt_chars": request.get("sent_prompt_chars"),
              "colour_contract_appended": request.get("colour_contract_appended"),
              "top_matches_sent": request.get("prompt_matches_sent"),
              "note": "沿用旧实验的同请求产物：逐项 hash 已核对一致，本次没有为此发图"}
    write_json_atomic(str(existing), record)
    log(f"[retention] {branch}-{sample_id} 沿用旧同请求产物（不产生图片请求）：{Path(candidate).name}")
    return record


def _retention_setup_candidates(out: Path, spec, *, only_b=False, log=print):
    """按 A 基线核对结果决定沿用旧 A 还是原地停下；B 才真正发送请求。"""
    check = retention_a_baseline_check(spec, out_dir=out)
    write_json_atomic(str(out / "retention-a-baseline-check.json"), check)
    if check["enabled"] and check["status"] != "ok":
        log("[retention] A 基线核对不一致：停止，不重新生成 A、不扩大样本")
        for mismatch in check["mismatches"]:
            log(f"[retention] A 基线差异：{mismatch}")
        return {"stopped": "a_baseline_mismatch", "check": check}
    stages = []
    for row in check["candidates"]:
        stages.append(_retention_inherit_candidate(out, spec, "A", row["sample_id"], row["candidate"], log=log))
    requests = retention_requests(spec, out_dir=out)
    if only_b:
        requests = [row for row in requests if row["branch"] == "B"]
    for request in requests:
        stages.append(_retention_repaint_stage(out, spec, request, log=log))
    return {"stopped": None, "check": check, "stages": stages}


def run_retention(out_dir, *, spec_path=None, log=print, dry_run=False, reuse_from=None, only_b=False):
    """执行：冻结核对 → 预算前置 → 初始重绘 → 门禁 → 汇总。不发无预算的请求。"""
    spec = load_retention_spec(spec_path)
    out = retention_run_dir(spec, out_dir)
    out.mkdir(parents=True, exist_ok=True)
    send_budget = _retention_prepare_env(out, spec)
    frozen_lock = out / "retention-frozen.json"
    if frozen_lock.is_file():
        differences = _retention_freeze_check(out, json.loads(frozen_lock.read_text(encoding="utf-8")))
        if differences:
            raise ValueError("冻结记录与当前协议不一致，拒绝复用；请新开目录/新协议：" + "；".join(differences))
    preflight = retention_preflight(out, spec_path=spec_path, only_b=only_b, log=log)
    if preflight["status"] != "passed":
        log("[retention] 预检未通过：停止，不发送任何请求")
        return {"preflight": preflight, "frozen": None, "stages": [], "stopped": "preflight_problems"}
    frozen = retention_freeze(out, preflight, log=log, reuse_from=reuse_from)
    requests = retention_requests(spec, out_dir=out)
    if only_b:
        requests = [row for row in requests if row["branch"] == "B"]
    log(f"[retention] 预算：图片计划 {len(requests)} 次 / 上限 {spec['budget']['image_hard_limit']}，"
        f"文本上限 {spec['budget']['text_hard_limit']}（目录 {out / 'send-budget'}）")
    if dry_run:
        return {"preflight": preflight, "frozen": frozen, "stages": [], "dry_run": True}
    setup = _retention_setup_candidates(out, spec, only_b=only_b, log=log)
    if setup.get("stopped"):
        retention_ledger(out, spec=spec, log=log)
        return {"preflight": preflight, "frozen": frozen, "stages": setup.get("stages") or [],
                "stopped": setup["stopped"], "a_baseline_check": setup["check"]}
    stages = list(setup["stages"])
    failures = [row for row in stages if row.get("status") == "request_failed"]
    audit = retention_audit_only(str(out), spec_path=spec_path, reuse_from=reuse_from, log=log)
    retention_colour_review(str(out), spec_path=spec_path, log=log)
    summary = retention_summary(out, spec=spec, log=log)
    ledger = retention_ledger(out, spec=spec, log=log)
    page = write_retention_gallery(out, spec=spec, log=log)
    return {"preflight": preflight, "frozen": frozen, "stages": stages, "audit": audit,
            "summary": summary, "ledger": ledger, "page": str(page),
            "failed_image_requests": len(failures),
            "stopped": "two_consecutive_failures" if len(failures) >= 2 else None}


def _retention_stage_dir(out: Path, branch, sample_id, stage):
    path = out / "stages" / f"{branch}-{sample_id}-{stage}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _retention_unique_run_id(send_budget, base: str) -> str:
    """给这次派发挑一个未被占用的 run_id。

    被拒绝过的尝试也算占用（否则会再次撞上同一条拒绝）；同一进程里再加一个启动时间后缀，
    这样「修完代码重跑」不会因为上一轮尝试用过同名 run_id 而继续被拒。
    """
    used = {str(row.get("run_id") or "") for row in (send_budget.state(kind="image").get("slots") or [])}
    stem = f"{base}-s{time.strftime('%H%M%S')}"
    for candidate in [stem] + [f"{stem}-try{i}" for i in range(2, 100)]:
        if candidate not in used:
            return candidate
    raise RuntimeError(f"{base}: run_id 候选全部被占用，停止（不降级、不绕过预算）")


def _retention_resolve(path_like):
    """把后端返回的产物路径解析成绝对路径（相对路径按仓库根解析，不是按当前工作目录）。"""
    path = Path(str(path_like))
    return path if path.is_absolute() else (BASE / path)


def _retention_prepare_send(out: Path, spec, request, *, log=print):
    """发送前的预算前置检查 + 派发。

    **不在这里自己做预留**：`api_backend.generate_image_repaint` 已经在发出 HTTP 之前
    用 `send_budget.reserve_image_attempt` 原子预留一次；再在外面预留一次会撞上
    「同一 run_id 只允许派发一次」的保护，把正常请求判成拒绝执行。
    这里只做两件事：① 发送前确认还有剩余槽位（不够就拒绝发送，不降级）；
    ② 给本次请求一个未被占用的 run_id（含进程启动时间后缀，避免修完代码重跑时撞旧 id）。
    """
    from utils import send_budget
    branch, sample_id = request["branch"], request["sample_id"]
    limit = int(spec["budget"]["image_hard_limit"])
    before = send_budget.state(kind="image", limit=limit)
    if int(before.get("remaining") or 0) <= 0:
        raise send_budget.SendBudgetExhausted(
            f"图片尝试额度已用满（{before.get('used')}/{limit}）：拒绝发送，不补抽、不扩组")
    attempt_run_id = _retention_unique_run_id(send_budget, f"{spec['id']}-{branch}-{sample_id}")
    os.environ["IMAGE_MAKER_SEND_BUDGET_RUN_ID"] = attempt_run_id
    log(f"[retention] 发送 {branch}-{sample_id}（run_id={attempt_run_id}，"
        f"发送前已用 {before.get('used')}/{limit}）")
    return before


def _retention_settle_from_ledger(send_budget, run_id, outputs, error=""):
    """按账本里该 run_id 的预留槽位结算（预留发生在后端内部，所以要从账本反查）。"""
    slot = None
    for row in send_budget.state(kind="image").get("slots") or []:
        if row.get("run_id") == run_id:
            slot = row
            break
    if slot is None:
        return None
    reservation = {"attempt_id": slot.get("attempt_id"), "slot": slot.get("slot"), "run_id": run_id}
    if outputs:
        for index, path in enumerate(outputs):
            send_budget.settle_image_attempt(reservation, status="success", output_path=path,
                                            output_sha256=_sha256_file(path))
    else:
        send_budget.settle_image_attempt(reservation, status="failed", error=str(error)[:300])
    return reservation


def _retention_repaint_stage(out: Path, spec, request, *, log=print):
    """一次初始重绘：发送前预算检查 → 发送（后端内部原子预留）→ 结算并落盘。

    初始候选永久保留：产物写进 `stages/<branch>-<sample>-initial-repaint/`，后续任何修订
    都写到别的 stage 目录，不覆盖这里。
    """
    from utils import send_budget
    from utils.post_process import dispatch_repaint_request
    branch, sample_id = request["branch"], request["sample_id"]
    folder = _retention_stage_dir(out, branch, sample_id, "initial-repaint")
    write_json_atomic(str(folder / "request.json"), request)
    try:
        _retention_prepare_send(out, spec, request, log=log)
    except Exception as error:  # noqa: BLE001
        record = {"stage": "initial-repaint", "branch": branch, "sample_id": sample_id,
                  "status": "refused_by_budget", "error": f"{type(error).__name__}: {error}"[:300],
                  "at": _now()}
        write_json_atomic(str(folder / "result.json"), record)
        log(f"[retention] {branch}-{sample_id} 被预算拒绝，未发送：{record['error']}")
        return record
    run_id = os.environ.get("IMAGE_MAKER_SEND_BUDGET_RUN_ID", "")
    error_text = ""
    try:
        outputs = dispatch_repaint_request(request, log_callback=log)
    except Exception as error:  # noqa: BLE001
        outputs = []
        error_text = f"{type(error).__name__}: {error}"
    outputs = [str(_retention_resolve(path).resolve()) for path in (outputs or [])
               if _retention_resolve(path).is_file()]
    reservation = _retention_settle_from_ledger(send_budget, run_id, outputs, error=error_text)
    record = {"stage": "initial-repaint", "branch": branch, "sample_id": sample_id,
              "status": "success" if outputs else "request_failed",
              "outputs": outputs, "output_sha256": {Path(p).name: _sha256_file(p) for p in outputs},
              "slot": (reservation or {}).get("slot"), "run_id": run_id, "at": _now(),
              "error": error_text[:300], "frozen_generation": request["frozen_generation"],
              "request_id": request["request_id"], "stage_name": request["stage_name"],
              "prompt_sha256": request["prompt_sha256"], "sent_prompt_sha256": request["sent_prompt_sha256"],
              "prompt_chars": request["prompt_chars"], "sent_prompt_chars": request["sent_prompt_chars"],
              "colour_contract_appended": request["colour_contract_appended"],
              "top_matches_sent": request["prompt_matches_sent"]}
    write_json_atomic(str(folder / "result.json"), record)
    log(f"[retention] {branch}-{sample_id} {record['status']}：{len(outputs)} 张"
        f"（发送正文 {record['sent_prompt_chars']} 字符，合同={record['colour_contract_appended']}）"
        + (f"（{error_text[:120]}）" if error_text else ""))
    return record
def _retention_analysis_stub(spec, source_item):
    """门禁的「期望」来源：首图实际请求文本，不伪造分析产物。"""
    return {"english_description": _retention_expected_text(spec, source_item),
            "original_english_description": _retention_expected_text(spec, source_item)}


def _retention_gate_stages(out: Path, spec, branch, item, state, reuse_from=None, *, log=print):
    """对一个候选跑现有门禁链（身份/质量/人体/终审）：只审计，不改图。

    - 每一项审计都绑定**候选内容 hash**，并分别记录：
      `reused`（复用匹配的成功结论）/ `fresh`（本次新调用）/ `budget_blocked`（预算阻止）/
      `audit_error`（审计报错）/ 明确未通过（由既有判据给出）。
    - 人体门禁按现有生产判据：不只看 `audit_error`，还要看明确结构缺陷、
      结论有效性与归属不确定（`should_refine_quality` / `ownership_uncertain`），不自行发明宽松判据。
    """
    from utils.identity_audit import audit_image_identity, identity_gate_action
    from utils.refine_quality import (audit_hand_quality, audit_refine_quality, final_quality_decision,
                                      normalize_quality_audit, record_quality_audit, should_refine_quality)
    sample_id = item["sample_id"]
    initial_path = _retention_stage_dir(out, branch, sample_id, "initial-repaint") / "result.json"
    if not initial_path.is_file():
        return {"stage": "gate", "branch": branch, "sample_id": sample_id,
                "status": "no_candidate", "at": _now()}
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    if initial.get("status") != "success" or not initial.get("outputs"):
        return {"stage": "gate", "branch": branch, "sample_id": sample_id,
                "status": initial.get("status") or "no_candidate", "at": _now()}
    candidate = initial["outputs"][0]
    candidate_sha = _sha256_file(candidate)
    base = str(_retention_path(item["image"]))
    style_ref = str(_retention_path(spec["style"]["ref_image"]))
    expected = _retention_analysis_stub(spec, item)
    folder = _retention_stage_dir(out, branch, sample_id, "gates")
    gates = {"branch": branch, "sample_id": sample_id, "candidate": candidate,
             "candidate_sha256": candidate_sha, "at": _now(), "audits": {}, "audit_sources": {}}

    def run(kind, call):
        audit, source = _retention_audit_call(state, spec, kind=kind, branch=branch, sample_id=sample_id,
                                              candidate_sha=candidate_sha, call=call, log=log)
        gates["audits"][kind] = audit
        gates["audit_sources"][kind] = source
        return audit

    identity = run("identity", lambda: audit_image_identity(candidate, expected,
                                                            expected_prompt=expected["english_description"]))
    if not identity.get("audit_error") and not identity.get("budget_blocked"):
        identity["gate_action"] = identity_gate_action(identity)
    else:
        identity["gate_action"] = "review_required"
    write_json_atomic(str(folder / "identity.json"), identity)

    quality = run("quality", lambda: audit_refine_quality(base, candidate, style_ref,
                                                          first_pass_prompt=expected["english_description"]))
    write_json_atomic(str(folder / "quality.json"), quality)
    if not quality.get("audit_error") and not quality.get("budget_blocked"):
        record_quality_audit(quality, str(folder / "refine-quality-audit-0.json"), log)

    def hands_call():
        audit = audit_hand_quality(candidate)
        audit = (normalize_quality_audit(audit, schema="anatomy") if isinstance(audit, dict)
                 else {"audit_error": "空结果"})
        # 现有判据：明确结构缺陷 / 结论无效（缺字段）/ 归属不确定 都不能算通过
        audit["hands_clear"] = not (
            bool(audit.get("needs_refine"))
            or bool(audit.get("ownership_uncertain"))
            or bool(audit.get("conclusion_missing"))
            or not bool(audit.get("conclusion_valid")))
        return audit

    hands = run("hands", hands_call)
    write_json_atomic(str(folder / "hands.json"), hands)

    def final_call():
        audit = audit_refine_quality(base, candidate, style_ref,
                                     first_pass_prompt=expected["english_description"],
                                     final_review=True,
                                     style_targets="\n".join(spec["style"].get("clauses") or []),
                                     authorized_changes=["允许迁移线条、笔触、边缘、材质与五官/头发抽象画法；"
                                                         "不得搬运参考角色身份或固有颜色"])
        audit["gate_decision"] = final_quality_decision(audit)
        return audit

    final = run("final_review", final_call)
    decision = (final.get("gate_decision") if isinstance(final, dict) and final.get("gate_decision")
                else final_quality_decision(final) if isinstance(final, dict) and not final.get("audit_error")
                else {"action": "review_required",
                      "reason": "预算阻止，未执行终审" if final.get("budget_blocked") else "终审审计失败"})
    gates["final_decision"] = decision
    write_json_atomic(str(folder / "final-review.json"), final)

    outcomes = _retention_gate_verdicts(gates, spec, should_refine_quality, final_quality_decision)
    gates["outcomes"] = outcomes
    gates["failed_gates"] = [kind for kind, value in outcomes.items() if value not in ("pass", "reused")]
    gates["not_executed_gates"] = [kind for kind, value in outcomes.items() if value == "budget_blocked"]
    gates["available_after_gates"] = all(value == "pass" for value in outcomes.values())
    write_json_atomic(str(folder / "gates.json"), gates)
    log(f"[retention] {branch}-{sample_id} 门禁：{outcomes} · 可用={gates['available_after_gates']}")
    return gates


def _retention_hands_clear(audit):
    """人体门禁是否通过：按既有判据（明确结构缺陷 / 结论无效 / 归属不确定）。

    旧记录（复用来的）可能没有 `hands_clear`，必须**按同一判据重算**，
    不能把「没这个字段」当成「明确不通过」——那会把复用的审计错误地判成失败。
    """
    if not isinstance(audit, dict) or audit.get("audit_error") or audit.get("budget_blocked"):
        return False
    if "hands_clear" in audit:
        return bool(audit["hands_clear"])
    return not (bool(audit.get("needs_refine"))
                or bool(audit.get("ownership_uncertain"))
                or bool(audit.get("conclusion_missing"))
                or audit.get("conclusion_valid") is False)


def _retention_gate_verdicts(gates, spec, should_refine_quality, final_quality_decision):
    """既有判据的单一出处：每项审计 → pass/needs_refine/fail/budget_blocked/audit_error。"""
    audits = gates.get("audits") or {}
    outcomes = {}
    for kind in RETENTION_AUDIT_KINDS:
        audit = audits.get(kind) or {}
        if audit.get("budget_blocked"):
            outcomes[kind] = "budget_blocked"
        elif audit.get("audit_error"):
            outcomes[kind] = "audit_error"
        elif kind == "identity":
            outcomes[kind] = "pass" if audit.get("gate_action") == "accept" else "fail"
        elif kind == "quality":
            outcomes[kind] = "needs_refine" if should_refine_quality(audit) else "pass"
        elif kind == "hands":
            outcomes[kind] = "pass" if _retention_hands_clear(audit) else "fail"
        else:
            decision = gates.get("final_decision") or final_quality_decision(audit)
            outcomes[kind] = "pass" if decision.get("action") in ("accept", "accept_with_warning") else "fail"
    return outcomes


def retention_gate_stages(out_dir, *, spec=None, log=print):
    """离线复算门禁结论（不调用任何接口）：读取已有审计记录，输出当前判据下的结论。"""
    from utils.refine_quality import final_quality_decision, should_refine_quality
    spec = spec or load_retention_spec()
    out = retention_run_dir(spec, out_dir)
    rows = []
    for path in sorted((out / "stages").glob("*-gates/gates.json")):
        gates = json.loads(path.read_text(encoding="utf-8"))
        outcomes = _retention_gate_verdicts(gates, spec, should_refine_quality, final_quality_decision)
        gates["outcomes_recomputed"] = outcomes
        gates["available_after_gates_recomputed"] = all(value == "pass" for value in outcomes.values())
        write_json_atomic(str(path), gates)
        rows.append({"branch": gates.get("branch"), "sample_id": gates.get("sample_id"),
                     "candidate_sha256": gates.get("candidate_sha256"),
                     "outcomes": outcomes, "available_after_gates": gates["available_after_gates_recomputed"],
                     "audit_sources": gates.get("audit_sources")})
        log(f"[retention] {gates.get('branch')}-{gates.get('sample_id')} 门禁复算：{outcomes}")
    write_json_atomic(str(out / "retention-gate-outcomes.json"), {"rows": rows})
    return rows


def retention_summary(out_dir, *, spec=None, log=print):
    """汇总：① 初始配色保留（门禁前，逐张） ② 门禁后可用结果 ③ 逐张结构化记录。"""
    spec = spec or load_retention_spec()
    out = retention_run_dir(spec, out_dir)
    stages = []
    for path in sorted((out / "stages").glob("*-initial-repaint/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        gates_path = path.parent.parent / f"{record['branch']}-{record['sample_id']}-gates" / "gates.json"
        record["gates"] = (json.loads(gates_path.read_text(encoding="utf-8")) if gates_path.is_file() else None)
        stages.append(record)
    rows = retention_per_image(out, spec=spec, stages=stages, log=log)
    with_review = [row for row in rows if row.get("colour_review")]
    verdicts = {}
    for row in with_review:
        key = str((row["colour_review"] or {}).get("colour_status"))
        verdicts[key] = verdicts.get(key, 0) + 1
    placements = {}
    for row in with_review:
        review = row["colour_review"] or {}
        placements[f"{row['branch']}-{row['sample_id']}"] = {
            "waistband": review.get("accent_waistband"),
            "left_shoe_bow": review.get("accent_left_shoe_bow"),
            "right_shoe_bow": review.get("accent_right_shoe_bow"),
            "both_shoe_bows_red": review.get("accent_left_shoe_bow") == "present"
                                   and review.get("accent_right_shoe_bow") == "present",
            "protected_colours_ok": review.get("protected_colours_ok"),
            "new_blocks": len(review.get("new_blocks") or []),
            "content_changes": len(review.get("content_changes") or []),
            "leak": (review.get("leak") or {}).get("status"),
            "aspect": (row.get("aspect") or {}).get("orientation")}
    gate_rows = {}
    usable = 0
    for row in rows:
        gates = row.get("gates") or {}
        outcomes = gates.get("outcomes_recomputed") or gates.get("outcomes")
        key = f"{row['branch']}-{row['sample_id']}"
        gate_rows[key] = {"candidate_sha256": row.get("candidate_sha256"), "outcomes": outcomes,
                          "audit_sources": gates.get("audit_sources"),
                          "available_after_gates": bool(gates.get("available_after_gates_recomputed",
                                                                   gates.get("available_after_gates")))}
        usable += 1 if gate_rows[key]["available_after_gates"] else 0
    summary = {"label": spec["id"], "at": _now(), "source_run": spec["source_run"],
               "style": spec["style"]["name"], "model": spec["generation"]["model"],
               "initial_candidates": len(stages),
               "initial_success": len([row for row in stages if row.get("status") == "success"]),
               "inherited_candidates": len([row for row in stages if row.get("inherited")]),
               "generated_this_run": len([row for row in stages if row.get("produced") is not False]),
               "colour_retention": {
                   "reviewed": len(with_review), "verdicts": verdicts, "placements": placements,
                   "by_branch": {branch: {row["sample_id"]: (row["colour_review"] or {}).get("colour_status")
                                          for row in with_review if row["branch"] == branch}
                                 for branch in ("A", "B")},
                   "note": ("逐张按区域与落点判定（主色区域、腰带、左鞋、右鞋分开记，缺一只鞋记 unverifiable）；"
                            "不用全图冷色像素占比代替落点验收；A/B 只有两对，不构成显著性。")},
               "gates": {"by_candidate": gate_rows, "available_after_gates": usable,
                         "note": "未执行/预算阻止/审计错误/明确不通过分别记录；本轮只审计，未做任何门禁修订。"},
               "stages": stages, "per_image": rows,
               "note": ("初始配色保留与门禁后可用结果分开报告：初始候选永久保留；"
                        "旧 v1 的 B 产物是「顶层写了合同、实际发送未带合同」的误标基线，只作留档。")}
    write_json_atomic(str(out / "retention-summary.json"), summary)
    write_json_atomic(str(out / "retention-per-image.json"), {"rows": rows})
    log(f"[retention] 初始候选 {summary['initial_success']}/{summary['initial_candidates']}"
        f"（沿用 {summary['inherited_candidates']}）· 配色评审 {len(with_review)} 张 {verdicts} · "
        f"门禁后可用 {usable}/{len(rows)}")
    return summary


def retention_per_image(out_dir, *, spec=None, stages=None, log=print):
    """逐张结构化记录：配色/落点/保护色/内容/画幅 + 门禁结论（报告与离线页共用同一份）。"""
    spec = spec or load_retention_spec()
    out = retention_run_dir(spec, out_dir)
    if stages is None:
        stages = []
        for path in sorted((out / "stages").glob("*-initial-repaint/result.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            gates_path = path.parent.parent / f"{record['branch']}-{record['sample_id']}-gates" / "gates.json"
            record["gates"] = (json.loads(gates_path.read_text(encoding="utf-8")) if gates_path.is_file() else None)
            stages.append(record)
    reviews = {}
    for path in sorted((out / "review").glob("colour-*.json")):
        if path.name.endswith((".failed.json", ".invalid.json", ".request.json")):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        for sample_id, entry in _retention_review_rows(record).items():
            reviews[(record.get("name"), sample_id)] = entry
    rows = []
    for record in stages:
        outputs = [p for p in (record.get("outputs") or []) if Path(p).is_file()]
        aspect = None
        if outputs:
            from PIL import Image
            with Image.open(outputs[0]) as image:
                aspect = {"width": image.width, "height": image.height,
                          "orientation": "landscape" if image.width > image.height else
                                         ("portrait" if image.height > image.width else "square")}
        review = reviews.get((f"colour-{record['branch']}", record["sample_id"]))
        gates = record.get("gates") or {}
        rows.append({"sample_id": record["sample_id"], "branch": record["branch"],
                     "status": record.get("status"), "inherited": bool(record.get("inherited")),
                     "candidate": outputs[0] if outputs else "",
                     "candidate_sha256": (record.get("output_sha256") or {}).get(
                         Path(outputs[0]).name) if outputs else None,
                     "aspect": aspect,
                     "prompt_chars": record.get("prompt_chars"),
                     "sent_prompt_chars": record.get("sent_prompt_chars"),
                     "sent_prompt_sha256": record.get("sent_prompt_sha256"),
                     "colour_contract_appended": record.get("colour_contract_appended"),
                     "top_matches_sent": record.get("top_matches_sent"),
                     "gates": gates,
                     "gate_outcomes": gates.get("outcomes_recomputed") or gates.get("outcomes"),
                     "colour_review": review,
                     "note": "配色看区域与落点；画幅、门禁、配色三者各自独立记录，不互相抵消。"})
    return rows


def retention_colour_review(out_dir, *, spec=None, spec_path=None, log=print):
    """配色保留评审：每支一次调用，输入=对应首图 + 本支两张候选（+ 画风参考），逐张分开记录。"""
    from utils.analysis_gpt_prompt import call_text_model
    spec = spec or load_retention_spec(spec_path)
    out = retention_run_dir(spec, out_dir)
    system = read_prompt_file(RETENTION_COLOUR_REVIEW_PROMPT_FILE)
    cfg = _text_cfg()
    plan = (out / "review")
    plan.mkdir(parents=True, exist_ok=True)
    records = []
    for branch in ("A", "B"):
        name = f"colour-{branch}"
        target = plan / f"{name}.json"
        if target.is_file():
            log(f"[cached] 配色评审 {name}")
            records.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        rows = []
        for item in spec["sources"]:
            result_path = _retention_stage_dir(out, branch, item["sample_id"], "initial-repaint") / "result.json"
            if not result_path.is_file():
                continue
            record = json.loads(result_path.read_text(encoding="utf-8"))
            outputs = [p for p in (record.get("outputs") or []) if Path(p).is_file()]
            if not outputs:
                continue
            rows.append({"sample_id": item["sample_id"], "candidate": outputs[0],
                         "first_image": str(_retention_path(item["image"])),
                         "candidate_sha256": _sha256_file(outputs[0])})
        if not rows:
            log(f"[retention] {name}: 没有可用候选，跳过配色评审")
            continue
        images, order = [], []
        for index, row in enumerate(rows, 1):
            images.extend(_reassess_proxy(out, row["first_image"]))
            order.append({"id": f"Q{index:02d}", "sample_id": row["sample_id"], "role": "first_image",
                          "candidate_sha256": None, "split": len(images)})
            images.extend(_reassess_proxy(out, row["candidate"]))
            order[-1]["candidate_sha256"] = row["candidate_sha256"]
        payload = {"task": "colour_retention_review",
                   "branch": branch,
                   "branch_definition": spec["branches"][branch],
                   "colour_contract": retention_contract_text(),
                   "detail_note": _reassess_detail_note(),
                   "style_reference": {"path": spec["style"]["ref_image"],
                                       "note": "只作疑似泄漏比对的参照，不要求候选与参考图配色一致"},
                   "protected_colours": ["银灰发 / natural silver-grey hair",
                                         "自然肤色 / natural skin tone",
                                         "象牙白洋装 / ivory-white dress",
                                         "深中性色厚底鞋 / dark neutral platform shoes"],
                   "accent_region": ["既有宽腰带 / existing waistband",
                                     "两只鞋的蝴蝶结 / ribbon bows of BOTH shoes"],
                   "main_colour_region": ["河流", "天空", "岸地", "远景植被", "阴影"],
                   "image_order": [{"id": row["id"], "note": "first image, then this branch's candidate, "
                                                             "each followed by its own body-detail crop"}
                                   for row in order],
                   "note": "逐张分开判定；区域落点优先于全图色调，不用全图冷色占比代替落点验收"}
        user = json.dumps(payload, ensure_ascii=False)
        request_record = {"purpose": "colour_retention_review", "name": name, "at": _now(),
                          "model": cfg["model"], "base_url": cfg["base_url"],
                          "system_prompt_file": RETENTION_COLOUR_REVIEW_PROMPT_FILE,
                          "system": system, "user": user,
                          "images": [{"file": Path(path).name, "sha256": _sha256_file(path)} for path in images]}
        write_json_atomic(str(plan / f"{name}.request.json"), request_record)
        with open(out / "review-calls.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"purpose": "colour_retention_review", "name": name,
                                     "model": cfg["model"], "at": _now()}, ensure_ascii=False) + "\n")
        raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user,
                              max_tokens=REASSESS_MAX_TOKENS, timeout=REASSESS_TIMEOUT,
                              image_paths=[str(path) for path in images])
        (plan / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
        value, parse_note = parse_object_tolerant(raw, log=log)
        if value is None:
            write_json_atomic(str(plan / f"{name}.failed.json"),
                              {"name": name, "reason": f"JSON 无法解析：{parse_note}", "at": _now()})
            log(f"[retention] 配色评审 {name}: 响应不可用，保留原文不自动重试")
            continue
        record = {"name": name, "at": _now(), "model": cfg["model"], "json_repair": parse_note,
                  "mapping": {row["id"]: row["sample_id"] for row in order},
                  "candidate_hashes": {row["id"]: row["candidate_sha256"] for row in order},
                  "response": value, "status": "review_record_not_gate"}
        try:
            validate_retention_colour_review(value, list(record["mapping"].keys()))
        except ValueError as error:
            record["validation_error"] = str(error)
            write_json_atomic(str(plan / f"{name}.failed.json"),
                              {"name": name, "reason": f"校验失败：{error}", "raw_response": value, "at": _now()})
            write_json_atomic(str(plan / f"{name}.invalid.json"), record)
            log(f"[retention] 配色评审 {name}: 结论校验失败，保留记录不当作结论：{error}")
            continue
        write_json_atomic(str(target), record)
        records.append(record)
        log(f"[retention] 配色评审 {name} 完成：{', '.join(record['mapping'].values())}")
    return records


def retention_ledger(out_dir=None, *, spec=None, log=print):
    """调用账本：发送前预留 + 结算 + 文本审计额度；未发送的拒绝与实际请求分开计数。"""
    from utils import send_budget
    spec = spec or load_retention_spec()
    out = retention_run_dir(spec, out_dir)
    root = str(out / "send-budget")
    images = send_budget.state(root, "image", int(spec["budget"]["image_hard_limit"]))
    texts = send_budget.state(root, "text", int(spec["budget"]["text_hard_limit"]))
    reservations = list(images.get("slots") or [])
    settlements = images.get("settlements") or {}
    succeeded = [row for row in reservations
                 if (settlements.get(row.get("attempt_id")) or {}).get("status") == "success"]
    failed = [row for row in reservations
              if (settlements.get(row.get("attempt_id")) or {}).get("status") in ("failed", "timeout", "unknown")]
    review_calls = 0
    review_log = out / "review-calls.jsonl"
    if review_log.is_file():
        review_calls = len([line for line in review_log.read_text(encoding="utf-8").splitlines() if line.strip()])
    audit_only = out / "retention-audit-only.json"
    audit_info = json.loads(audit_only.read_text(encoding="utf-8")) if audit_only.is_file() else {}
    ledger = {"label": spec["id"], "at": _now(), "budget_dir": root,
              "image": {"limit": images["limit"], "reserved_total": len(reservations),
                        "sent_success": len(succeeded), "sent_failed_or_unknown": len(failed),
                        "refused_by_budget": len([row for row in reservations
                                                  if row.get("operation", "").startswith("retention-")
                                                  and (settlements.get(row.get("attempt_id")) or {}).get("status") == "failed"
                                                  and "预算" in str((settlements.get(row.get("attempt_id")) or {}).get("error", ""))]),
                        "attempts": [{"slot": row.get("slot"), "run_id": row.get("run_id"),
                                      "operation": row.get("operation"), "model": row.get("model"),
                                      "status": (settlements.get(row.get("attempt_id")) or {}).get("status"),
                                      "reserved_at": row.get("reserved_at")} for row in reservations]},
              "text": {"limit": texts["limit"], "reserved": len(texts.get("slots") or []),
                       "fresh_gate_audits": len(audit_info.get("fresh") or []),
                       "reused_audits": len(audit_info.get("reused") or []),
                       "budget_blocked_audits": len(audit_info.get("blocked") or []),
                       "audit_errors": len(audit_info.get("errored") or []),
                       "colour_review_calls": review_calls},
              "note": ("账本在发送前原子预留，失败与未知也留痕；"
                       "被预算拒绝的尝试与实际 HTTP 请求分开计数；估算费用与真实账单分开标注。")}
    write_json_atomic(str(out / "retention-ledger.json"), ledger)
    log(f"[retention] 账本：图片实际发送 {len(succeeded)} 成功 / {len(failed)} 失败或未知 / "
        f"{len(reservations)} 次预留；文本新调用 {len(audit_info.get('fresh') or [])} + 配色评审 {review_calls}")
    return ledger


def out_dir_ref_path(spec):
    return spec.get("_out_dir") or str(BASE / "data" / "test-result" / "20261004" / spec["id"])


def validate_retention_colour_review(value, expected_ids):
    """校验配色保留评审：恰好覆盖每张一次，枚举值合法，必须逐项给出落点。"""
    if not isinstance(value, dict):
        raise ValueError("配色保留评审必须是 JSON 对象")
    rows = value.get("images")
    if not isinstance(rows, list) or sorted(str(row.get("id")) for row in rows) != sorted(expected_ids):
        raise ValueError("配色保留评审必须恰好覆盖本支每张候选一次")
    for row in rows:
        item_id = str(row.get("id"))
        if row.get("main_colour_status") not in RETENTION_COLOUR_STATES:
            raise ValueError(f"{item_id}: main_colour_status 非法")
        if row.get("accent_status") not in RETENTION_COLOUR_STATES:
            raise ValueError(f"{item_id}: accent_status 非法")
        if row.get("colour_status") not in RETENTION_COLOUR_STATES:
            raise ValueError(f"{item_id}: colour_status 非法")
        for key in ("accent_waistband", "accent_left_shoe_bow", "accent_right_shoe_bow"):
            if row.get(key) not in RETENTION_ACCENT_STATES:
                raise ValueError(f"{item_id}: {key} 非法")
        if str(row.get("protected_colours_ok")).lower() not in ("true", "false", "unclear"):
            raise ValueError(f"{item_id}: protected_colours_ok 非法")
        leak = row.get("leak")
        if isinstance(leak, dict) and leak.get("status") not in (None, "none", "suspected", "unverifiable"):
            raise ValueError(f"{item_id}: leak.status 非法")
    return value


def _retention_review_rows(record):
    """把一次配色评审记录整理成 {sample_id: 逐张结论}，并保留分支与响应原文引用。"""
    mapping = {str(key): value for key, value in (record.get("mapping") or {}).items()}
    rows = {}
    for entry in (record.get("response") or {}).get("images") or []:
        sample_id = mapping.get(str(entry.get("id")))
        if sample_id:
            rows[sample_id] = entry
    return rows


def retention_gallery_data(out_dir, *, spec=None, log=print):
    """离线页需要的数据：逐张记录 + 门禁 + 配色评审 + 账本（页与 JSON 共用同一份）。"""
    spec = spec or load_retention_spec()
    out = retention_run_dir(spec, out_dir)
    summary_path = out / "retention-summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file()
               else retention_summary(out, spec=spec, log=log))
    ledger_path = out / "retention-ledger.json"
    ledger = (json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file()
              else retention_ledger(out, spec=spec, log=log))
    frozen_path = out / "retention-frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.is_file() else {}
    return {"spec": spec, "out": out, "summary": summary, "ledger": ledger, "frozen": frozen}


def write_retention_gallery(out_dir, *, spec=None, log=print):
    """保留验证的离线对照页：首图 / A 支 / B 支并排，附逐张结论、门禁与调用记账。"""
    from html import escape
    data = retention_gallery_data(out_dir, spec=spec, log=log)
    spec, out, summary, ledger, frozen = (data["spec"], data["out"], data["summary"],
                                          data["ledger"], data["frozen"])
    sources = {item["sample_id"]: item for item in spec["sources"]}
    rows = {(row["branch"], row["sample_id"]): row for row in summary.get("per_image") or []}

    def tag(state):
        colours = {"success": "ok", "request_failed": "bad", "refused_by_budget": "bad",
                   "accept": "ok", "accept_with_warning": "warn", "review_required": "bad",
                   "pass": "ok", "fail": "bad", "needs_refine": "warn", "budget_blocked": "unk",
                   "audit_error": "bad", "conform": "ok", "partial": "warn", "unverifiable": "unk",
                   "true": "ok", "false": "bad", "unclear": "unk"}
        return colours.get(str(state), "unk")

    def image(path, label):
        if not path or not Path(path).is_file():
            return f'<figure class="missing">{escape(label)}：无产物</figure>'
        return (f'<figure><img loading="lazy" src="{escape(_thumb_data_uri(path, 760, 82))}" '
                f'alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>')

    blocks = []
    for sample_id, item in sources.items():
        first = str(_retention_path(item["image"]))
        cards = []
        for branch in ("A", "B"):
            row = rows.get((branch, sample_id))
            if not row:
                cards.append(f'<section class="card"><h3>{branch}</h3><p>没有记录</p></section>')
                continue
            gates = row.get("gates") or {}
            review = row.get("colour_review") or {}
            protected = "、".join(f'{p.get("name")}={p.get("status")}'
                                  for p in (review.get("protected_colours") or []))
            blocks_html = "".join(f'<li>{escape(json.dumps(b, ensure_ascii=False))}</li>'
                                  for b in (review.get("new_blocks") or []))
            content_html = "".join(f'<li>{escape(json.dumps(c, ensure_ascii=False))}</li>'
                                   for c in (review.get("content_changes") or []))
            audit_rows = "".join(
                f'<tr><td>{escape(kind)}</td><td><span class="tag {tag(value)}">{escape(str(value))}</span></td>'
                f'<td>{escape(str((gates.get("audit_sources") or {}).get(kind) or ""))}</td></tr>'
                for kind, value in (gates.get("outcomes") or {}).items())
            cards.append(
                f'<section class="card"><h3>{escape(branch)} 支 · {escape(sample_id)} '
                f'<span class="tag {tag(row.get("status"))}">{escape(str(row.get("status")))}</span></h3>'
                f'<div class="row">{image(row.get("candidate"), "初始重绘候选（永久保留）")}'
                f'<div class="col">'
                f'<p>配色总判定：<span class="tag {tag(review.get("colour_status"))}">'
                f'{escape(str(review.get("colour_status") or "未评审"))}</span> ｜ 主色 '
                f'<span class="tag {tag(review.get("main_colour_status"))}">{escape(str(review.get("main_colour_status") or "-"))}</span>'
                f' ｜ 点缀落点 腰带={escape(str(review.get("accent_waistband") or "-"))}'
                f' 左鞋={escape(str(review.get("accent_left_shoe_bow") or "-"))}'
                f' 右鞋={escape(str(review.get("accent_right_shoe_bow") or "-"))}</p>'
                f'<p class="muted">{escape(str(review.get("summary") or ""))}</p>'
                f'<p>保护色：{escape(str(review.get("protected_colours_ok") or "-"))}（{escape(protected)}）</p>'
                f'<p>新增显著色块：{"有" if blocks_html else "无"}<ul>{blocks_html}</ul></p>'
                f'<p>内容变化：{"有" if content_html else "无"}<ul>{content_html}</ul></p>'
                f'<p>疑似参考图颜色泄漏：{escape(str((review.get("leak") or {}).get("status") or "-"))} '
                f'{escape(str((review.get("leak") or {}).get("evidence") or ""))}</p>'
                f'<p>画幅：{escape(str((row.get("aspect") or {}).get("orientation") or "-"))}'
                f'（{(row.get("aspect") or {}).get("width")}×{(row.get("aspect") or {}).get("height")}）</p>'
                f'<p>发送正文：{escape(str(row.get("sent_prompt_chars") or "-"))} 字符 ｜ 含配色合同：'
                f'{escape(str(row.get("colour_contract_appended")))}</p>'
                f'<p>候选 sha256：{escape(str(row.get("candidate_sha256") or "")[:16])}</p>'
                f'<table class="gates"><tr><th>门禁</th><th>结论</th><th>来源</th></tr>{audit_rows}</table>'
                f'<p>门禁后可用：<span class="tag {tag(str(bool(gates.get("available_after_gates"))).lower())}">'
                f'{escape(str(gates.get("available_after_gates")))}</span></p>'
                f'</div></div></section>')
        blocks.append(
            f'<section class="group"><h2>{escape(sample_id)}（{escape(item["group_id"])} · 第 {item["round"]} 轮）</h2>'
            f'<div class="row">{image(first, "首图（冻结输入，配色权威）")}</div>' + "".join(cards) + "</section>")

    header = (f'<h1>固定色板 → 画风重绘保留验证 · A/B 对照</h1>'
              f'<p class="muted">协议 {escape(str(spec["id"]))} · 画风 {escape(str(spec["style"]["name"]))} · '
              f'模型 {escape(str(spec["generation"]["model"]))} / {escape(str(spec["generation"]["resolution"]))} · '
              f'freeze_hash {escape(str(frozen.get("freeze_hash") or "")[:16])}</p>'
              '<aside><b>判读口径</b><p>A = 现有画风重绘（不改请求）；B = 同一请求末尾追加配色继承合同。'
              '两支的差别就是那段合同，参考图顺序与参数完全相同。</p>'
              '<p>配色看区域与落点，不用全图冷色占比代替；颜色像参考图只能记「疑似泄漏」，不证明因果；'
              '画质提升不能抵消配色、身份或内容违约。</p>'
              '<p>初始候选永久保留；本轮只生成 B 初始候选并审计，未做任何门禁修订。</p>'
              '<p>后续配色功能仍只允许在没有画风参考图时选择：本次双参考是受控实验，不构成生产接线授权。</p></aside>'
              f'<p>图片：实际发送 {escape(str(ledger["image"].get("sent_success")))} 成功 / '
              f'{escape(str(ledger["image"].get("sent_failed_or_unknown")))} 失败或未知 / '
              f'{escape(str(ledger["image"].get("reserved_total")))} 次预留 ｜ 文本审计：新调用 '
              f'{escape(str(ledger["text"].get("fresh_gate_audits")))}、复用 '
              f'{escape(str(ledger["text"].get("reused_audits")))}、配色评审 '
              f'{escape(str(ledger["text"].get("colour_review_calls")))}</p>'
              f'<p><a href="retention-summary.json">retention-summary.json</a> · '
              f'<a href="retention-per-image.json">retention-per-image.json</a> · '
              f'<a href="retention-frozen.json">retention-frozen.json</a> · '
              f'<a href="retention-preflight.json">retention-preflight.json</a> · '
              f'<a href="retention-ledger.json">retention-ledger.json</a></p>')
    css = ("body{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#25322b;background:#eef1ee;margin:0}"
           "main{max-width:1280px;margin:auto;padding:24px}"
           "section.group{background:#fff;border-radius:14px;padding:22px;margin:20px 0}"
           "section.card{border:1px solid #e0e6e1;border-radius:10px;padding:16px;margin:14px 0;background:#fbfdfb}"
           ".row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1 1 420px;min-width:320px}"
           "figure{margin:0;flex:0 0 auto;max-width:460px}"
           "figure img{width:100%;border-radius:8px;border:1px solid #d8ded9}"
           "figure.missing{padding:40px;background:#fbe9ea;color:#a12e37;border-radius:8px}"
           "figcaption{font-size:12.5px;color:#657269}"
           "table.gates{border-collapse:collapse;font-size:13px;margin:6px 0}"
           "table.gates th,table.gates td{border:1px solid #e0e6e1;padding:2px 8px;text-align:left}"
           ".tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}"
           ".tag.ok{background:#2f8f5b}.tag.warn{background:#c98a17}.tag.bad{background:#b23b45}.tag.unk{background:#6b7785}"
           ".muted{color:#657269;font-size:13px}ul{margin:4px 0 8px 18px;padding:0}li{font-size:13px}")
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>{escape(str(spec["id"]))}</title>', f'<style>{css}</style><main>',
             header, "".join(blocks), '</main></html>']
    path = out / "retention-review.html"
    path.write_text("".join(parts), encoding="utf-8")
    log(f"[retention] A/B 对照页 → {path}")
    return path
    """保留验证的离线对照页：首图 / A 支 / B 支并排，附门禁结论与逐阶段调用记账。"""
    from html import escape
    spec = load_retention_spec()
    out = _resolve_retention_out(out_dir)
    summary_path = out / "retention-summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file()
               else retention_summary(out, spec=spec, log=log))
    ledger_path = out / "retention-ledger.json"
    ledger = (json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file()
              else retention_ledger(out, spec=spec, log=log))
    frozen_path = out / "retention-frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.is_file() else {}
    sources = {item["sample_id"]: item for item in spec["sources"]}
    stages = {(row["branch"], row["sample_id"]): row for row in summary.get("stages") or []}

    def tag(state):
        colours = {"success": "ok", "request_failed": "bad", "accept": "ok", "accept_with_warning": "warn",
                   "review_required": "bad", "correct": "warn", "pass": "ok", "true": "ok", "false": "bad"}
        return colours.get(str(state), "unk")

    def image(path, label):
        if not path or not Path(path).is_file():
            return f'<figure class="missing">{escape(label)}：无产物</figure>'
        return (f'<figure><img loading="lazy" src="{escape(_thumb_data_uri(path, 760, 82))}" '
                f'alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>')


# --- 离线修正交付（retention-v2 corrections） -------------------------------
#
# 只读：不改旧图、旧请求/响应、旧冻结、旧账本、旧报告。
# 输出独立目录，全部结论引用旧证据路径与 hash；分歧逐项留档。

RETENTION_FINAL_STATUS = ("available", "revision_required", "blocked_by_audit_error",
                          "blocked_by_budget", "disputed_needs_review")
# 人工目视观察（本轮离线核对时记录的），与原始模型判断、离线重算分开保存。
# 只记录观察到的事实与分歧状态，不用来放宽门禁。
RETENTION_VISUAL_OBSERVATIONS = {
    ("B", "gemini-flash-P1-knowledge-r1", "final_review"): {
        "observed_facts": [
            "全图（含放大底部条带）里两只鞋与鞋底完整可读，红蝴蝶结、鞋跟、鞋底厚度都清楚",
            "底部 12% 条带里最大的深色低饱和连通域（鞋体主体）距画幅底边 75 px，没有贴到画幅底边",
            "但在九宫格细节拼图里，左鞋一侧的鞋底贴到了画面最下缘；右脚一侧离底边还有明显空隙",
            "首图本身也把主体贴近下边缘（首图最后一行仍有 90 个深色像素）",
        ],
        "disagreement": ("模型原文写「right platform heel … visibly cropped」，"
                         "与实测（贴近下缘的是左脚一侧、右脚离底边更远）在指认对象上不一致"),
        "root_cause": "未确定：无法区分「九宫格细节拼图的单元格底边被误读成画幅底边」与「模型指认对象错误」",
        "why_undetermined": "本批没有保存实际发送的图片字节，只能用当前代码重建输入形态",
        "action": "不删掉裁切判定、不据此缩小人物/补背景、不宣布门禁通过；最终状态仍由离线重算的判据决定",
        "recorded_by": "offline_correction_2026-10-04",
    },
}
# 逐项门禁的判据取值（与候选级最终状态分开，不能混为一类）
RETENTION_AUDIT_VERDICT_STATES = ("pass", "needs_refine", "fail", "audit_error", "not_executed")


def retention_audit_input_evidence(out: Path, spec, *, branch, sample_id, candidate, log=print):
    """重建该次审计的**全图代理、九宫格细节拼图与提交顺序**（不修改原图，不联网）。

    必须如实标注：本批审计**没有保存实际发送的图片字节**，这里重建的是「当前代码下的同一路径产物」，
    只能作为输入形态与顺序的说明，不能冒充历史输入。
    """
    from utils.refine_quality import _detail_sheet, _proxy
    from PIL import Image
    import shutil
    folder = out / "input-evidence" / f"{branch}-{sample_id}"
    folder.mkdir(parents=True, exist_ok=True)
    first = str(_retention_path(next(row for row in spec["sources"]
                                     if row["sample_id"] == sample_id)["image"]))
    style = str(_retention_path(spec["style"]["ref_image"]))
    rows = [
        {"order": 1, "role": "full_image", "source": "first_image", "path": first,
         "sha256": _sha256_file(first), "note": "审计的 Image 1（首图，期望来源）"},
        {"order": 2, "role": "full_image", "source": "candidate", "path": candidate,
         "sha256": _sha256_file(candidate), "note": "审计的 Image 2（候选，全图代理 1536px）"},
        {"order": 3, "role": "full_image", "source": "style_reference", "path": style,
         "sha256": _sha256_file(style), "note": "审计的 Image 3（画风参考）"},
        {"order": 4, "role": "detail_sheet", "source": "candidate", "path": candidate,
         "sha256": _sha256_file(candidate),
         "note": "审计的 Image 4：候选的 3×3 九宫格细节拼图（由 refine_quality._detail_sheet 生成，"
                 "按设计只用于局部细节，不用于判断完整画幅）"},
    ]
    rebuilt = {}
    try:
        proxy_path = _proxy(candidate, f"correction-{branch}-{sample_id}")
        sheet_path = _detail_sheet(candidate)
        for key, source_path in (("candidate_full_proxy", proxy_path), ("candidate_detail_sheet", sheet_path)):
            target = folder / f"{key}{Path(source_path).suffix}"
            shutil.copyfile(source_path, target)
            rebuilt[key] = {"path": str(target), "sha256": _sha256_file(target),
                            "size": list(Image.open(target).size),
                            "origin": _file_origin(target),
                            "not_the_original_input": True}
        for path in (proxy_path, sheet_path):
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception as error:  # noqa: BLE001
        rebuilt["error"] = f"{type(error).__name__}: {error}"
    evidence = {"branch": branch, "sample_id": sample_id, "candidate": candidate,
                "candidate_sha256": _sha256_file(candidate), "submission_order": rows,
                "rebuilt_inputs": rebuilt,
                "origin": "rebuilt_by_current_code（本批没有保存实际发送字节）",
                "note": ("细节拼图只用于看局部；完整画幅判断必须看全图。"
                         "重建件不能作为历史输入的证明。")}
    write_json_atomic(str(folder / "input-evidence.json"), evidence)
    log(f"[retention] 输入证据 {branch}-{sample_id}：重建代理与细节拼图（不改原图）")
    return evidence


def retention_shoe_region_evidence(image_path, *, bottom_band=0.12, log=print):
    """鞋部区域证据：在底部条带里找**最大的深色连通域**（鞋体），给出其边界与是否碰到画幅边。

    只用深色低饱和像素 + 连通域，避免把深蓝水色、花叶算进来。不修改原图、不联网。
    """
    import numpy as np
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        band_top = int(height * (1 - bottom_band))
        grey = np.asarray(rgb).astype(np.int16)
    lum = grey.mean(axis=2)
    chroma = grey.max(axis=2) - grey.min(axis=2)
    mask = (lum < 110) & (chroma < 60)
    mask[:band_top] = False
    visited = np.zeros_like(mask, dtype=bool)
    best = None
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        count = 0
        min_y = max_y = int(start_y)
        min_x = max_x = int(start_x)
        while stack:
            y, x = stack.pop()
            count += 1
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_x, max_x = min(min_x, x), max(max_x, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] \
                            and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if best is None or count > best["pixels"]:
            best = {"pixels": count, "box": [min_x, min_y, max_x, max_y]}
    if best is None:
        return {"image": str(image_path), "size": [width, height], "found": False,
                "note": "底部条带里没有找到深色低饱和连通域"}
    min_x, min_y, max_x, max_y = best["box"]
    evidence = {"image": str(image_path), "sha256": _sha256_file(image_path), "size": [width, height],
                "found": True, "band_from_y": band_top,
                "dark_region_box": best["box"], "dark_region_pixels": best["pixels"],
                "gap_to_bottom_px": height - 1 - max_y, "gap_to_bottom_ratio": round((height - 1 - max_y) / height, 4),
                "touches_bottom_edge": max_y >= height - 1,
                "note": ("深色连通域的下边界到画幅底边的距离：>0 表示该连通域没有贴到画幅底边。"
                         "这只是「这一块深色区域」的位置证据，不等于对整个鞋部做像素级分类。")}
    log(f"[retention] 鞋部证据 {Path(image_path).name}：最大深色连通域 "
        f"{best['pixels']} px，距底边 {evidence['gap_to_bottom_px']} px")
    return evidence


def retention_corrections(out_dir=None, *, spec_path=None, src_dir=None, evidence=True, reuse_from=None,
                          log=print):
    """生成离线修正交付：只读旧证据，输出统一后的逐张/门禁/汇总与核对页。

    `out_dir` 为 None 时（只读核对路径）回退到 spec 的 `corrections_dir`；
    测试必须显式传自己的临时目录，生产由 CLI 显式传 `corrections_dir`。
    """
    from PIL import Image
    from utils.refine_quality import final_quality_decision, should_refine_quality
    spec = load_retention_spec(spec_path)
    if out_dir is None:
        out_dir = spec.get("corrections_dir")
    if not out_dir:
        raise ValueError("必须指定修正交付目录（显式 --out 或 spec.corrections_dir）")
    out = _resolve_retention_out(out_dir)
    src = _resolve_retention_out(src_dir or out_dir_ref_path(spec))
    out.mkdir(parents=True, exist_ok=True)
    reuse_scan = _retention_reuse_scan(_retention_path(reuse_from) if reuse_from else src, spec)
    rows, gate_rows = [], []
    for path in sorted(src.glob("stages/*-gates/gates.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        branch, sample_id = record.get("branch"), record.get("sample_id")
        audits = record.get("audits") or {}
        candidate = record.get("candidate")
        candidate_sha = record.get("candidate_sha256")
        verdicts, kinds = {}, {}
        for kind in RETENTION_AUDIT_KINDS:
            audit = audits.get(kind) or {}
            original = _retention_original_verdict(audit, kind, final_quality_decision, should_refine_quality)
            recomputed = _retention_recomputed_verdict(audit, kind, record, final_quality_decision,
                                                       should_refine_quality)
            if audit.get("budget_blocked"):
                state, final = "budget", "not_executed"
            elif audit.get("audit_error"):
                state, final = "audit_error", "blocked_by_audit_error"
            else:
                state, final = "evaluated", recomputed
            scan = reuse_scan.get((str(branch), str(sample_id), kind, candidate_sha)) or {}
            stored_fp = retention_record_fingerprint({**audit, "kind": kind}, spec, record)
            verdicts[kind] = {"original_model_verdict": original, "recomputed_offline": recomputed,
                              "visual_observation": (RETENTION_VISUAL_OBSERVATIONS.get((branch, sample_id, kind))
                                                     if kind == "final_review" else None),
                              "final_status": final, "record_state": state,
                              "audit_sources": (record.get("audit_sources") or {}).get(kind),
                              "candidate_sha256": candidate_sha,
                              "source_audit_file": str(path),
                              # 复用资格只对「本批复用来的」审计有意义；本批新调用（fresh）的记录同样没有指纹，
                              # 所以这里把「指纹完整性」与「当初是否复用」分开写清楚。
                              "reuse_eligibility": (scan.get("reuse_eligibility")
                                                    if (record.get("audit_sources") or {}).get(kind) == "reused"
                                                    else {"state": "not_applicable",
                                                          "note": "本批新调用，不涉及复用"}),
                              "fingerprint_completeness": {
                                  "saved_fields": {key: bool(stored_fp.get(key))
                                                   for key in RETENTION_FINGERPRINT_FIELDS},
                                  "missing": retention_fingerprint_completeness(stored_fp),
                                  "recorded_at": "本批执行时（未能保存完整指纹）",
                                  "note": "旧记录没有保存完整指纹：不补造、不宣称已验证；续跑不得据其复用。",
                              },
                              "fingerprint": stored_fp}
            kinds[kind] = final
        disputed = [kind for kind, value in kinds.items() if value == "fail"]
        status = ("blocked_by_audit_error" if "blocked_by_audit_error" in kinds.values()
                  else "blocked_by_budget" if "not_executed" in kinds.values()
                  else "available" if all(value == "pass" for value in kinds.values())
                  else "revision_required")
        gate_rows.append({"branch": branch, "sample_id": sample_id, "candidate": candidate,
                          "candidate_sha256": candidate_sha, "verdicts": verdicts,
                          "final_status": status, "failing_gates": disputed,
                          "source_gates_file": str(path),
                          "note": ("原始模型判断 / 离线重算 / 目视观察 / 最终处理状态分开保存；"
                                   "最终处理状态只由离线重算与审计状态决定，不因目视观察而放宽门禁。")})
    for row in gate_rows:
        audit_evidence = []
        if evidence:
            audit_evidence.append(retention_audit_input_evidence(
                out, spec, branch=row["branch"], sample_id=row["sample_id"],
                candidate=row["candidate"], log=log))
        review = _retention_review_for(src, row["branch"], row["sample_id"])
        aspect = None
        if row["candidate"] and Path(row["candidate"]).is_file():
            with Image.open(row["candidate"]) as image:
                aspect = {"width": image.width, "height": image.height,
                          "orientation": "landscape" if image.width > image.height else
                                         ("portrait" if image.height > image.width else "square")}
        rows.append({**row, "aspect": aspect, "colour_review": review, "input_evidence": audit_evidence,
                     "shoe_region_evidence": (retention_shoe_region_evidence(row["candidate"], log=log)
                                              if row["candidate"] and Path(row["candidate"]).is_file() else None),
                     "source_candidate": row["candidate"],
                     "note": ("配色评审依据：有对应首图与候选的全图与细节裁剪；"
                              "画风参考图只以路径文字出现，未真正随请求发送，"
                              "因此参考颜色泄漏一律记不可验证。")})
    summary = {"label": spec["id"] + "-corrections", "at": _now(), "corrections_only": True,
               "source_dir": str(src), "corrections_dir": str(out),
               "evidence_references": {"source_summary_sha256": _sha256_if_file(src / "retention-summary.json"),
                                       "source_ledger_sha256": _sha256_if_file(src / "retention-ledger.json"),
                                       "source_frozen_sha256": _sha256_if_file(src / "retention-frozen.json"),
                                       "source_gates_files": [row["source_gates_file"] for row in gate_rows]},
               "per_image": rows,
               "status_counts": {value: len([row for row in rows if row["final_status"] == value])
                                 for value in RETENTION_FINAL_STATUS},
               "no_calls": True, "no_images": True,
               "note": ("本轮只做离线核对与统一交付：没有新调用、没有新图片、没有视觉重评、没有修订生图；"
                        "旧图、旧请求/响应、旧冻结、旧账本、旧报告全部保留在 " + str(src))}
    write_json_atomic(str(out / "corrections-summary.json"), summary)
    write_json_atomic(str(out / "corrected-gates.json"),
                      {"rows": [{key: row[key] for key in
                                 ("branch", "sample_id", "candidate", "candidate_sha256", "verdicts",
                                  "final_status", "failing_gates", "source_gates_file")}
                                for row in rows]})
    write_json_atomic(str(out / "corrected-per-image.json"), {"rows": rows})
    log(f"[retention] 修正交付 → {out}（{len(rows)} 个候选，状态 {summary['status_counts']}）")
    return summary


def _retention_original_verdict(audit, kind, final_quality_decision, should_refine_quality):
    """原始模型判断（按记录里已有的字段如实还原，不改写）。"""
    if audit.get("budget_blocked"):
        return "未执行（预算阻止）"
    if audit.get("audit_error"):
        return f"审计错误：{str(audit.get('audit_error'))[:120]}"
    if kind == "identity":
        return str(audit.get("gate_action") or audit.get("mismatch"))
    if kind == "quality":
        return f"needs_refine={audit.get('needs_refine')} severity={audit.get('severity')}"
    if kind == "hands":
        return (f"needs_refine={audit.get('needs_refine')} "
                f"ownership_uncertain={len(audit.get('ownership_uncertain') or [])} "
                f"conclusion_valid={audit.get('conclusion_valid')}")
    decision = audit.get("gate_decision") or {}
    return str(decision.get("action") or decision.get("reason") or "未记录")


def _retention_recomputed_verdict(audit, kind, gates, final_quality_decision, should_refine_quality):
    """按现有同一判据离线重算（不看目视观察、不放宽）。"""
    if audit.get("budget_blocked"):
        return "not_executed"
    if audit.get("audit_error"):
        return "audit_error"
    if kind == "identity":
        return "pass" if audit.get("gate_action") == "accept" else "fail"
    if kind == "quality":
        return "needs_refine" if should_refine_quality(audit) else "pass"
    if kind == "hands":
        return "pass" if _retention_hands_clear(audit) else "fail"
    decision = gates.get("final_decision") or audit.get("gate_decision") or final_quality_decision(audit)
    return "pass" if decision.get("action") in ("accept", "accept_with_warning") else "fail"


def _observation_text(observation):
    """把人工目视观察压成一行摘要（保留原始模型判断不动）。"""
    if not observation:
        return ""
    facts = "；".join(observation.get("observed_facts") or [])
    return (f"【目视】{facts} ｜ 分歧：{observation.get('disagreement', '')} "
            f"｜ 根因：{observation.get('root_cause', '')} ｜ 处理：{observation.get('action', '')}")


def _retention_review_for(src: Path, branch, sample_id):
    """取该候选的配色评审条目，并标注证据边界（旧评审没有真正附上参考图）。"""
    for path in sorted((src / "review").glob("colour-*.json")):
        if path.name.endswith((".request.json", ".failed.json", ".invalid.json")):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("name") != f"colour-{branch}":
            continue
        entry = _retention_review_rows(record).get(sample_id)
        if not entry:
            continue
        original_leak = (entry.get("leak") or {}).get("status")
        return {"source": str(path), "entry": entry,
                "model_saw_first_image_and_candidate": True,
                "model_saw_style_reference_image": False,
                "style_reference_in_request": "path only（只有路径文字，参考图没有随请求发送）",
                "original_model_leak_claim": original_leak,
                "leak": "unverifiable",
                "leak_final": "unverifiable",
                "leak_scope": ("unverifiable（缺少参考图实际输入）：旧评审没有附上参考图，"
                               "因此它给出的 leak=none 不能当作「已证明没有参考颜色泄漏」"),
                "usable_findings": ["环境主色（对首图）", "红色点缀与腰带/两只鞋蝴蝶结落点",
                                    "受保护固有颜色", "新增显著色块", "内容变化（含画幅）"],
                "note": "参考颜色泄漏在旧证据下不可判定；其余项以对应首图与候选为证据。"}
    return None


def write_retention_corrections_page(out_dir=None, *, spec_path=None, log=print):
    """离线核对页：内嵌原有图片（首图 + 候选），逐项列出四种状态与分歧。"""
    from html import escape
    spec = load_retention_spec(spec_path)
    out = _resolve_retention_out(out_dir or spec.get("corrections_dir") or "")
    if not (out / "corrections-summary.json").is_file():
        raise FileNotFoundError(f"先跑 `retention correct` 生成修正交付：{out}")
    summary = json.loads((out / "corrections-summary.json").read_text(encoding="utf-8"))
    src = Path(summary["source_dir"])
    sources = {item["sample_id"]: item for item in spec["sources"]}

    def tag(state):
        colours = {"pass": "ok", "needs_refine": "warn", "fail": "bad", "audit_error": "bad",
                   "not_executed": "unk", "available": "ok", "revision_required": "warn",
                   "blocked_by_audit_error": "bad", "blocked_by_budget": "unk",
                   "disputed_needs_review": "warn", "conform": "ok", "partial": "warn",
                   "unverifiable": "unk", "suspected": "warn", "none": "ok",
                   "verified_match": "ok", "unverified_fingerprint": "unk", "mismatch": "bad"}
        return colours.get(str(state), "unk")

    def image(path, label):
        if not path or not Path(path).is_file():
            return f'<figure class="missing">{escape(label)}：无产物</figure>'
        return (f'<figure><img loading="lazy" src="{escape(_thumb_data_uri(path, 720, 82))}" '
                f'alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>')

    blocks = []
    by_key = {(row["branch"], row["sample_id"]): row for row in summary["per_image"]}
    for sample_id, item in sources.items():
        cards = []
        for branch in ("A", "B"):
            row = by_key.get((branch, sample_id))
            if not row:
                continue
            verdict_rows = "".join(
                f'<tr><td>{escape(kind)}</td>'
                f'<td>{escape(str(value.get("original_model_verdict")))}</td>'
                f'<td><span class="tag {tag(value.get("recomputed_offline"))}">'
                f'{escape(str(value.get("recomputed_offline")))}</span></td>'
                f'<td><span class="tag {tag(value.get("final_status"))}">{escape(str(value.get("final_status")))}</span></td>'
                f'<td>{escape(str(value.get("audit_sources")))}</td>'
                f'<td>{escape(_observation_text(value.get("visual_observation")))}</td></tr>'
                for kind, value in row["verdicts"].items())
            review = (row.get("colour_review") or {}).get("entry") or {}
            evidence = (row.get("input_evidence") or [{}])[0] if row.get("input_evidence") else {}
            rebuilt = evidence.get("rebuilt_inputs") or {}
            order_rows = "".join(
                f'<li>Image {entry.get("order")}：{escape(str(entry.get("role")))} · '
                f'{escape(str(entry.get("source")))} · {escape(str(entry.get("sha256"))[:16])} — '
                f'{escape(str(entry.get("note")))}</li>'
                for entry in (evidence.get("submission_order") or []))
            rebuilt_html = "".join(
                image(value.get("path"), f'重建输入：{key}（当前代码重建，非历史发送字节）')
                for key, value in rebuilt.items() if isinstance(value, dict) and value.get("path"))
            cards.append(
                f'<section class="card"><h3>{escape(branch)} 支 · {escape(sample_id)} '
                f'<span class="tag {tag(row["final_status"])}">{escape(row["final_status"])}</span></h3>'
                f'<div class="row">{image(row.get("source_candidate"), "候选（原有图片，未改动）")}'
                f'<div class="col">'
                f'<table class="gates"><tr><th>门禁</th><th>原始模型判断</th><th>离线重算</th>'
                f'<th>最终处理状态</th><th>审计来源</th><th>目视观察（本轮离线核对）</th></tr>{verdict_rows}</table>'
                f'<p>画幅：{escape(str((row.get("aspect") or {}).get("orientation")))}'
                f'（{(row.get("aspect") or {}).get("width")}×{(row.get("aspect") or {}).get("height")}）'
                f' ｜ 候选 sha256 {escape(str(row.get("candidate_sha256"))[:16])}</p>'
                f'<p>鞋部区域证据：最大深色连通域距画幅底边 '
                f'{escape(str((row.get("shoe_region_evidence") or {}).get("gap_to_bottom_px")))} px；'
                f'该连通域是否贴到画幅底边：'
                f'{escape(str((row.get("shoe_region_evidence") or {}).get("touches_bottom_edge")))}'
                f' <span class="muted">（只说明这一块深色区域的位置，不等于对整只鞋做像素分类）</span></p>'
                f'<p>配色：<span class="tag {tag(review.get("colour_status"))}">'
                f'{escape(str(review.get("colour_status") or "未评审"))}</span> ｜ '
                f'腰带={escape(str(review.get("accent_waistband")))} 左鞋={escape(str(review.get("accent_left_shoe_bow")))} '
                f'右鞋={escape(str(review.get("accent_right_shoe_bow")))} ｜ 保护色={escape(str(review.get("protected_colours_ok")))}</p>'
                f'<p class="muted">参考颜色泄漏：{escape(str((review.get("leak") or {}).get("status") or "-"))} — '
                f'本轮评审请求里<b>只有参考图路径文字、没有附上参考图</b>，所以泄漏结论一律记不可验证。</p>'
                f'<details><summary>该次审计提交的图片与顺序（含重建件说明）</summary><ul>{order_rows}</ul>'
                f'<p class="muted">{escape(str(evidence.get("note") or ""))}</p></details>'
                f'</div></div>'
                f'<details><summary>重建的输入形态（当前代码重建，非历史发送字节）</summary>'
                f'<div class="row">{rebuilt_html}</div></details>'
                f'</section>')
        blocks.append(f'<section class="group"><h2>{escape(sample_id)}（{escape(item["group_id"])}）</h2>'
                      f'<div class="row">{image(str(_retention_path(item["image"])), "首图（冻结输入）")}'
                      f'{image(item.get("_candidate_hint", ""), "")}</div>' + "".join(cards) + "</section>")
    header = (f'<h1>保留验证 v2 · 离线修正交付</h1>'
              f'<p class="muted">只读修正：没有新调用、没有新图片、没有视觉重评、没有修订生图。'
              f'旧证据目录：{escape(str(src))}</p>'
              '<aside><b>这一页解决什么</b>'
              '<p>① 把「原始模型判断 / 离线重算 / 最终处理状态」三者分开列出，消除文件之间的口径矛盾（不改门禁判据）。</p>'
              '<p>② 记下 B-r1「鞋被裁切」的事实分歧：模型说右脚被底边裁掉，实际产图里两只鞋与鞋底完整可见。</p>'
              '<p>③ 说明配色评审的证据边界：附了首图与候选，<b>没有附参考图</b>，所以参考颜色泄漏记不可验证。</p>'
              '</aside>'
              f'<p><a href="corrections-summary.json">corrections-summary.json</a> · '
              f'<a href="corrected-gates.json">corrected-gates.json</a> · '
              f'<a href="corrected-per-image.json">corrected-per-image.json</a> · '
              f'<a href="verification.json">verification.json</a></p>')
    css = ("body{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#25322b;background:#eef1ee;margin:0}"
           "main{max-width:1280px;margin:auto;padding:24px}"
           "section.group{background:#fff;border-radius:14px;padding:22px;margin:20px 0}"
           "section.card{border:1px solid #e0e6e1;border-radius:10px;padding:16px;margin:14px 0;background:#fbfdfb}"
           ".row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1 1 520px;min-width:320px}"
           "figure{margin:0;flex:0 0 auto;max-width:420px}"
           "figure img{width:100%;border-radius:8px;border:1px solid #d8ded9}"
           "figure.missing{padding:24px;background:#fbe9ea;color:#a12e37;border-radius:8px}"
           "figcaption{font-size:12.5px;color:#657269}"
           "table.gates{border-collapse:collapse;font-size:13px;margin:6px 0;width:100%}"
           "table.gates th,table.gates td{border:1px solid #e0e6e1;padding:2px 6px;text-align:left;vertical-align:top}"
           ".tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}"
           ".tag.ok{background:#2f8f5b}.tag.warn{background:#c98a17}.tag.bad{background:#b23b45}.tag.unk{background:#6b7785}"
           ".muted{color:#657269;font-size:13px}ul{margin:4px 0 8px 18px;padding:0}li{font-size:13px}")
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>保留验证 v2 离线修正交付</title>', f'<style>{css}</style><main>',
             header, "".join(blocks), '</main></html>']
    path = out / "corrections-review.html"
    path.write_text("".join(parts), encoding="utf-8")
    log(f"[retention] 修正核对页 → {path}")
    return path


def retention_verify_corrections(out_dir=None, *, spec_path=None, log=print):
    """自动一致性核对：把各文件的口径对齐，防止文件之间再出现矛盾。"""
    from utils.refine_quality import final_quality_decision, should_refine_quality
    spec = load_retention_spec(spec_path)
    out = _resolve_retention_out(out_dir) if out_dir else _resolve_retention_out(
        spec.get("corrections_dir") or "")
    problems = []
    gates = json.loads((out / "corrected-gates.json").read_text(encoding="utf-8"))["rows"]
    per_image = json.loads((out / "corrected-per-image.json").read_text(encoding="utf-8"))["rows"]
    summary = json.loads((out / "corrections-summary.json").read_text(encoding="utf-8"))
    by_key = {(row["branch"], row["sample_id"]): row for row in per_image}
    for row in gates:
        other = by_key.get((row["branch"], row["sample_id"]))
        if not other:
            problems.append(f"{row['branch']}-{row['sample_id']}: 修正逐张文件里缺少该候选")
            continue
        if row["final_status"] != other["final_status"]:
            problems.append(f"{row['branch']}-{row['sample_id']}: gates 与逐张的最终状态不一致")
        for kind, value in row["verdicts"].items():
            if value["recomputed_offline"] != other["verdicts"][kind]["recomputed_offline"]:
                problems.append(f"{row['branch']}-{row['sample_id']}: {kind} 的两处重算结果不一致")
            if value["candidate_sha256"] != other["candidate_sha256"]:
                problems.append(f"{row['branch']}-{row['sample_id']}: {kind} 绑定的候选 hash 不一致")
            if value["final_status"] not in RETENTION_AUDIT_VERDICT_STATES:
                problems.append(f"{row['branch']}-{row['sample_id']}: {kind} 的逐项判据取值非法")
            if value.get("record_state") == "budget" and value["final_status"] != "not_executed":
                problems.append(f"{row['branch']}-{row['sample_id']}: {kind} 预算阻止却未记 not_executed")
            if value.get("record_state") == "audit_error" and value["final_status"] != "blocked_by_audit_error":
                problems.append(f"{row['branch']}-{row['sample_id']}: {kind} 审计错误却未单独标注")
    expected_counts = {value: len([row for row in per_image if row["final_status"] == value])
                       for value in RETENTION_FINAL_STATUS}
    if expected_counts != summary["status_counts"]:
        problems.append("汇总里的状态计数与逐张文件不一致")
    for row in per_image:
        if row["final_status"] == "available" and row["failing_gates"]:
            problems.append(f"{row['branch']}-{row['sample_id']}: 标记可用但存在未通过项")
        review = row.get("colour_review") or {}
        entry = review.get("entry") or {}
        original_claim = review.get("original_model_leak_claim")
        if entry and review.get("leak") != "unverifiable":
            problems.append(f"{row['branch']}-{row['sample_id']}: 未附参考图时应把泄漏结论记为 unverifiable"
                            f"（原始模型声明 {original_claim} 必须单独保留）")
        if entry and original_claim is None:
            problems.append(f"{row['branch']}-{row['sample_id']}: 缺少原始模型的泄漏声明留档")
        if entry and "leak_scope" not in review:
            problems.append(f"{row['branch']}-{row['sample_id']}: 缺少泄漏证据边界说明")
        for kind, value in row["verdicts"].items():
            missing = retention_fingerprint_completeness(value.get("fingerprint") or {})
            if value.get("audit_sources") == "reused" and missing and \
                    value.get("reuse_eligibility", {}).get("state") != "unverified_fingerprint":
                problems.append(f"{row['branch']}-{row['sample_id']}: {kind} 复用但未标注指纹缺失状态")
    page = out / "corrections-review.html"
    if page.is_file():
        html = page.read_text(encoding="utf-8")
        if "src=\"http" in html:
            problems.append("核对页引用了外部资源，必须完全离线自包含")
        if "data:image/jpeg" not in html:
            problems.append("核对页没有内嵌任何图片")
    result = {"label": spec["id"] + "-corrections", "at": _now(), "checked_files": [
        str(out / "corrected-gates.json"), str(out / "corrected-per-image.json"),
        str(out / "corrections-summary.json")],
        "problems": problems, "status": "problems" if problems else "passed", "no_calls": True}
    write_json_atomic(str(out / "verification.json"), result)
    if problems:
        log(f"[retention] 一致性核对发现 {len(problems)} 个问题：")
        for problem in problems:
            log(f"[retention]   - {problem}")
    else:
        log("[retention] 一致性核对 passed：逐项门禁 / 逐张 / 汇总 / 页面口径一致")
    return result


def retention_recheck_preview(out_dir=None, *, spec_path=None, log=print):
    """为「参考颜色泄漏」复核准备请求预览：明确图片角色与顺序，**不调用**。"""
    spec = load_retention_spec(spec_path)
    out = _resolve_retention_out(out_dir) if out_dir else _resolve_retention_out(
        spec.get("corrections_dir") or "")
    layout = [{"order": 1, "role": "first_image", "use": "配色契约的权威来源（主色区域 + 红色点缀落点）"},
              {"order": 2, "role": "candidate", "use": "被评候选，**全图**；画幅判断只看这一张"},
              {"order": 3, "role": "style_reference", "use": "画风参考图，**本轮真正附上图片**；用于参考颜色泄漏比对"},
              {"order": 4, "role": "candidate_detached_detail", "use": "候选的局部裁剪，**明确标注不用于判断画幅，也不作为泄漏证据的唯一依据**"}]
    preview = {"label": spec["id"] + "-recheck-preview", "at": _now(), "calls_made": 0,
               "purpose": "参考颜色泄漏复核（未来需要时才执行，本轮只准备）",
               "system_prompt_file": RETENTION_COLOUR_REVIEW_PROMPT_FILE,
               "image_layout": layout,
               "items": [{"sample_id": item["sample_id"], "first_image": item["image"],
                          "first_image_sha256": item["sha256"],
                          "candidate_stage_dir": f"stages/B-{item['sample_id']}-initial-repaint",
                          "style_reference": spec["style"]["ref_image"],
                          "style_reference_sha256": spec["style"]["ref_image_sha256"]}
                         for item in spec["sources"]],
               "requirements": ["必须真正附上参考图（本次旧评审只给了路径文字）",
                                "逐张分开判定，不得跨图合并结论",
                                "颜色相近只能支持「疑似泄漏」，不能自动证明因果",
                                "细节裁剪不得用于判断完整画幅"],
               "note": "本轮只生成预览，不调用任何接口。"}
    path = out / "recheck-request-preview.json"
    write_json_atomic(str(path), preview)
    log(f"[retention] 复核请求预览（未调用）→ {path}")
    return preview


def retention_recheck_preflight(out_dir, *, spec_path=None, log=print):
    """Freeze three read-only review requests without sending or changing old evidence."""
    from PIL import Image
    from utils.style_gpt import to_data_url

    spec = load_retention_spec(spec_path)
    out = _resolve_retention_out(out_dir)
    source_dir = retention_run_dir(spec)
    if out == source_dir or out.is_relative_to(source_dir):
        raise ValueError("Use a separate recheck directory; old evidence must stay unchanged")
    cfg = _text_cfg()
    if not cfg.get("model"):
        raise ValueError("A configured review model is required")
    reference = _retention_path(spec["style"]["ref_image"])
    if _sha256_file(reference) != spec["style"]["ref_image_sha256"]:
        raise ValueError("Style reference changed; refuse to prepare a new request silently")

    def image_entry(path, role):
        path = Path(path).resolve()
        with Image.open(path) as im:
            size = list(im.size)
        return {"path": str(path), "role": role, "sha256": _sha256_file(path),
                "dimensions": size,
                "data_url_sha256": _sha256_str(to_data_url(str(path)))}

    requests, candidates = [], {}
    for branch in ("A", "B"):
        images, pairs = [], []
        for index, item in enumerate(spec["sources"], 1):
            result_path = source_dir / "stages" / f"{branch}-{item['sample_id']}-initial-repaint" / "result.json"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            candidate = Path(record["outputs"][0])
            recorded_hash = (record.get("output_sha256") or {}).get(candidate.name)
            if record.get("status") != "success" or recorded_hash != _sha256_file(candidate):
                raise ValueError(f"{branch}-{item['sample_id']}: candidate hash/status mismatch")
            source = _retention_path(item["image"])
            if _sha256_file(source) != item["sha256"]:
                raise ValueError(f"{item['sample_id']}: source hash mismatch")
            images.extend([image_entry(source, f"Q{index:02d}_source_full"),
                           image_entry(candidate, f"Q{index:02d}_candidate_full")])
            candidates[(branch, index)] = (source, candidate)
            pairs.append({"id": f"Q{index:02d}", "sample_id": item["sample_id"],
                          "source_image_number": len(images) - 1,
                          "candidate_image_number": len(images)})
        images.append(image_entry(reference, "style_reference_full"))
        payload = {"purpose": "colour_retention_with_visible_reference", "pairs": pairs,
                   "style_reference_image_number": len(images),
                   "colour_contract": retention_contract_text()}
        requests.append({"id": f"colour-{branch}", "purpose": payload["purpose"],
                         "system_prompt_file": "color-knowledge/retention-recheck-colour-system.md",
                         "user": json.dumps(payload, ensure_ascii=False), "images": images})

    source, candidate = candidates[("B", 1)]
    requests.append({"id": "boundary-B-r1", "purpose": "framing_diagnostic_not_production_gate",
                     "system_prompt_file": "color-knowledge/retention-boundary-diagnostic-system.md",
                     "user": json.dumps({"image_1": "source full frame", "image_2": "candidate full frame",
                                         "candidate_dimensions": image_entry(candidate, "candidate_full")["dimensions"]},
                                        ensure_ascii=False),
                     "images": [image_entry(source, "source_full"), image_entry(candidate, "candidate_full")]})
    for row in requests:
        row["system"] = read_prompt_file(row["system_prompt_file"])
        row["model"] = cfg["model"]
        row["endpoint"] = str(cfg["base_url"]).rstrip("/")
        row["max_completion_tokens"] = 4000
        row["system_sha256"] = _sha256_str(row["system"])
        row["user_sha256"] = _sha256_str(row["user"])
        row["request_hash"] = digest(row)
    plan = {"protocol": "retention-targeted-recheck-v1", "requests": requests,
            "budget": {"image_calls": 0, "text_calls_max": 3, "automatic_retries": 0},
            "calls_made": 0, "execution_authorized": False,
            "scope": "Read-only review; no generation, repair, publishing, or gate override"}
    plan["plan_hash"] = digest(plan)
    target = out / "recheck-preflight.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError("Frozen recheck plan changed; use a new directory")
        return old
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(target), plan)
    log(f"[retention] Three offline requests frozen at {target}; no calls made")
    return plan


def validate_targeted_recheck(value, name):
    if name.startswith("colour-"):
        rows = value.get("images") or []
        if sorted(row.get("id", "") for row in rows) != ["Q01", "Q02"]:
            raise ValueError("Expected exactly Q01 and Q02")
        for row in rows:
            for key in ("main_colour_status", "colour_status"):
                if row.get(key) not in ("conform", "partial", "fail", "unverifiable"):
                    raise ValueError("Invalid colour status")
            for key in ("accent_waistband", "accent_left_shoe_bow", "accent_right_shoe_bow"):
                if row.get(key) not in ("present", "absent", "unverifiable"):
                    raise ValueError("Missing individual accent assessment")
            if row.get("protected_colours_status") not in ("retained", "changed", "unverifiable"):
                raise ValueError("Missing protected colour assessment")
            influence = row.get("reference_influence") or {}
            if influence.get("status") not in ("none_observed", "suspected", "unverifiable"):
                raise ValueError("Missing reference comparison")
            if influence.get("causal_attribution") != "not_established":
                raise ValueError("Unjustified causal attribution")
    else:
        shoes = value.get("candidate_shoes") or []
        if sorted(row.get("screen_side", "") for row in shoes) != ["left", "right"]:
            raise ValueError("Expected two separate screen-side shoes")
        if any(row.get("status") not in ("contained", "touching_frame", "truncated", "occluded", "unverifiable") for row in shoes):
            raise ValueError("Invalid framing status")
        if value.get("scope") != "framing_diagnostic_only_not_production_gate":
            raise ValueError("Diagnostic must not override production gates")


def retention_recheck_run(out_dir, *, spec_path=None, authorized=False, log=print):
    """Send the frozen read-only reviews once, with durable reservations and raw evidence."""
    import shutil
    import requests
    from utils import send_budget
    from utils.analysis_gpt_prompt import normalize_chat_base
    from utils.style_gpt import to_data_url

    if not authorized:
        raise ValueError("Explicit authorization is required for three paid text requests")
    plan = retention_recheck_preflight(out_dir, spec_path=spec_path, log=log)
    out = _resolve_retention_out(out_dir)
    cfg = _text_cfg()
    if not cfg.get("api_key"):
        raise ValueError("Review API key is missing")
    env = {send_budget.ENV_DIR: str(out / "send-budget"),
           send_budget.ENV_RUN: plan["plan_hash"], send_budget.ENV_TEXT_LIMIT: "3",
           send_budget.ENV_TEXT_PER_STAGE: "1"}
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    records = []
    try:
        for row in plan["requests"]:
            name = row["id"]
            target = out / "vision" / f"{name}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                cached = json.loads(target.read_text(encoding="utf-8"))
                if cached.get("request_hash") != row["request_hash"]:
                    raise ValueError("Existing review does not match frozen request")
                records.append(cached)
                continue
            content = [{"type": "text", "text": row["user"]}]
            for index, entry in enumerate(row["images"], 1):
                path = Path(entry["path"])
                if _sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Frozen input changed: {path}")
                evidence = out / "input-evidence" / name / f"{index:02d}{path.suffix}"
                evidence.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, evidence)
                url = to_data_url(str(evidence))
                if _sha256_str(url) != entry["data_url_sha256"]:
                    raise ValueError("Image bytes differ from the frozen request")
                content.append({"type": "image_url", "image_url": {"url": url}})
            payload = {"model": row["model"], "max_completion_tokens": row["max_completion_tokens"],
                       "messages": [{"role": "system", "content": row["system"]},
                                    {"role": "user", "content": content}]}
            url = normalize_chat_base(row["endpoint"]) + "/chat/completions"
            write_json_atomic(str(target.with_suffix(".request.json")),
                              {"url": url, "payload": payload, "request_hash": row["request_hash"],
                               "payload_sha256": digest(payload), "images": row["images"]})
            reservation = send_budget.reserve_text_attempt(stage=name, purpose=row["purpose"])
            record = {"id": name, "request_hash": row["request_hash"], "at": _now(),
                      "scope": row["purpose"], "status": "unknown"}
            try:
                # A plain Session has zero transport retries; redirects are also disabled.
                with requests.Session() as session:
                    response = session.post(url, json=payload, timeout=REASSESS_TIMEOUT,
                                            allow_redirects=False,
                                            headers={"Authorization": "Bearer " + cfg["api_key"],
                                                     "Content-Type": "application/json"})
                target.with_suffix(".raw.txt").write_text(response.text, encoding="utf-8")
                record["http_status"] = response.status_code
                response.raise_for_status()
                if response.is_redirect:
                    raise ValueError("Redirect refused; no second request sent")
                body = response.json()
                record["usage"] = body.get("usage")
                choices = body.get("choices") or []
                message = (choices[0].get("message") or {}) if choices else {}
                raw = message.get("content") or ""
                value, note = parse_object_tolerant(raw, log=log)
                if message.get("refusal") or not value:
                    raise ValueError("Refused, empty, or non-JSON review response")
                validate_targeted_recheck(value, name)
                record.update({"status": "success", "response": value, "json_repair": note})
            except Exception as error:
                record.update({"status": "failed_or_unknown", "error": type(error).__name__,
                               "billing": "unknown; no retry"})
            write_json_atomic(str(target), record)
            send_budget.settle_text_attempt(reservation, status=record["status"],
                                            error=record.get("error", ""))
            records.append(record)
            log(f"[retention] {name}: {record['status']}; no automatic retry")
        ledger = send_budget.state(str(out / "send-budget"), "text", 3)
        summary = {"plan_hash": plan["plan_hash"], "authorized_text_limit": 3,
                   "image_calls": 0, "text_reservations": ledger["used"], "reviews": records,
                   "fees": "Usage retained when received; missing response means billing unknown. No provider invoice available",
                   "production_gates_changed": False}
        write_json_atomic(str(out / "recheck-results.json"), summary)
        from html import escape
        parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
                 '<meta name="viewport" content="width=device-width,initial-scale=1">',
                 '<title>Targeted read-only recheck</title>',
                 '<style>body{font:15px/1.6 system-ui;margin:24px;max-width:1100px}'
                 'img{width:100%;max-width:480px}figure{display:inline-block;vertical-align:top;'
                 'margin:8px;max-width:480px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
                 '<h1>定向只读复核</h1><p>不生图、不修图、不覆盖旧门禁。连接失败不等于评审不通过；缺响应时费用未知。</p>']
        for row, record in zip(plan["requests"], records):
            parts.append(f'<section><h2>{escape(row["id"])}</h2><pre>{escape(json.dumps(record, ensure_ascii=False, indent=2))}</pre>')
            for entry in row["images"]:
                parts.append(f'<figure><img src="{to_data_url(entry["path"])}"><figcaption>{escape(entry["role"])}</figcaption></figure>')
            parts.append('</section>')
        parts.append('</html>')
        (out / "recheck-review.html").write_text("\n".join(parts), encoding="utf-8")
        return summary
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# ---------------------------------------------------------------------------
# 跨主题配色执行验证（cross-theme-palette-v1，2026-10-05）
#
# 场景：**没有画风参考图**时，把「指定一套颜色 + 指定落点区域」写成结构化合同并执行。
# 只判配色与落点是否保留，不评画质或美感。
# 合同与主题在 prompts/color-knowledge/cross-theme-palette-v1.json；本模块只做校验、组装与运行。
# ---------------------------------------------------------------------------

CROSS_THEME_SPEC_PATH = "prompts/color-knowledge/cross-theme-palette-v1.json"
CROSS_THEME_LABEL = "color-palette-cross-theme-v1"
CROSS_THEME_CONTRACT_DOC = "prompts/color-knowledge/cross-theme-contract-v1.md"
CROSS_THEME_COLOUR_REVIEW_PROMPT = "color-knowledge/cross-theme-colour-review-system.md"
CROSS_THEME_ACCENT_REGIONS = ("waist_band_region", "left_shoe_bow_region", "right_shoe_bow_region")
CROSS_THEME_STATES = ("conform", "partial", "fail", "unverifiable")


def load_cross_theme_spec(path=None):
    """解析并校验跨主题 spec：合同字段齐全、区域映射可落地、保护色与授权改色不冲突。"""
    spec_path = Path(path or (BASE / CROSS_THEME_SPEC_PATH))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if str(spec.get("kind")) != "cross-theme":
        raise ValueError("跨主题 spec 的 kind 必须是 cross-theme")
    for key in ("id", "contract", "schemes", "control_group", "themes", "prompt_layout",
                "evaluation", "budget_policy"):
        if not spec.get(key):
            raise ValueError(f"跨主题 spec 缺少 {key}")
    contract = spec["contract"]
    for key in ("contract_id", "version", "authorized_recolor_regions", "protected_in_every_theme",
                "forbidden_additions", "negative", "allowed_neutrals"):
        if not contract.get(key):
            raise ValueError(f"合同缺少 {key}")
    if sorted(contract["authorized_recolor_regions"]) != sorted(CROSS_THEME_ACCENT_REGIONS):
        raise ValueError("本版合同只授权三处既有落点区域（腰带式束带 + 两只鞋的蝴蝶结）")
    if len(spec["schemes"]) != 3:
        raise ValueError("本版固定三个晋级方案")
    for scheme in spec["schemes"]:
        for key in ("id", "main", "accent", "clauses"):
            if not scheme.get(key):
                raise ValueError(f"方案 {scheme.get('id')} 缺少 {key}")
        for side in ("main", "accent"):
            block = scheme[side]
            if not (block.get("name") and block.get("hex") and block.get("role")):
                raise ValueError(f"方案 {scheme['id']} 的 {side} 必须给出 name/hex/role")
        if not str(scheme["main"]["hex"]).startswith("#") or not str(scheme["accent"]["hex"]).startswith("#"):
            raise ValueError(f"方案 {scheme['id']} 的色值必须是 #rrggbb 近似值")
    if len(spec["themes"]) != 2:
        raise ValueError("本轮固定两个新主题")
    # 区域映射必须是 {maps_to, describes} 结构：映射目标要能被机器核对
    for theme in spec["themes"]:
        for key, value in (theme.get("region_mapping") or {}).items():
            if not isinstance(value, dict) or not value.get("maps_to") or not value.get("describes"):
                raise ValueError(f"{theme['id']}：区域映射 {key} 必须是 {{maps_to, describes}} 结构")
    spec["_path"] = str(spec_path)
    if spec.get("out_dir"):
        spec["_out_dir"] = str(_retention_path(spec["out_dir"]))
    return spec


def cross_theme_contract(spec):
    """合同正文的规范化副本 + hash（hex 只是近似值，不参与验收阈值判定）。"""
    contract = spec["contract"]
    normalized = {
        "contract_id": contract["contract_id"],
        "version": contract["version"],
        "authorized_recolor_regions": sorted(contract["authorized_recolor_regions"]),
        "protected_in_every_theme": sorted(contract["protected_in_every_theme"]),
        "forbidden_additions": sorted(contract["forbidden_additions"]),
        "negative": sorted(contract["negative"]),
        "allowed_neutrals": sorted(row["name"] for row in contract["allowed_neutrals"]),
        "colors": {scheme["id"]: {"main": scheme["main"], "accent": scheme["accent"]}
                   for scheme in spec["schemes"]},
    }
    return {"contract": normalized, "contract_hash": digest(normalized),
            "doc": CROSS_THEME_CONTRACT_DOC}


def cross_theme_conflict_report(spec):
    """保护色与授权改色的冲突检查：区域不得重叠，映射必须落在已有区域上。"""
    contract = spec["contract"]
    authorized = set(contract["authorized_recolor_regions"])
    problems, checked = [], []
    for theme in spec["themes"]:
        existing = set(theme.get("existing_regions") or [])
        mapping = theme.get("region_mapping") or {}
        blocked = set(theme.get("protected_regions") or [])
        overlap = sorted(authorized & blocked)
        checked.append({"theme": theme["id"], "authorized": sorted(authorized),
                        "existing": sorted(existing),
                        "mapped": {key: (value.get("maps_to") if isinstance(value, dict) else None)
                                   for key, value in mapping.items()}})
        if overlap:
            problems.append(f"{theme['id']}：授权改色区域与保护色区域重叠 {overlap}")
        for key, value in mapping.items():
            if not isinstance(value, dict) or not value.get("maps_to") or not value.get("describes"):
                problems.append(f"{theme['id']}：区域映射 {key} 必须是 {{maps_to, describes}} 结构")
                continue
            if value["maps_to"] not in existing:
                problems.append(f"{theme['id']}：{key} 映射到的 {value['maps_to']} 不在该主题已有区域里")
        if "environment_dominant_region" not in mapping:
            problems.append(f"{theme['id']}：缺少环境主色区域映射")
        unmapped = [region for region in CROSS_THEME_ACCENT_REGIONS if region not in mapping]
        if unmapped:
            if theme.get("accent_regions_exist") is False:
                checked[-1]["unmappable"] = unmapped
            else:
                problems.append(f"{theme['id']}：点位区域没有映射 {unmapped}")
    return {"problems": problems, "checked": checked,
            "status": "conflict" if problems else "ok",
            "note": ("冲突时停下说明，不通过关闭或放宽身份/质量门禁绕过；"
                     "点缀色只允许落在既有区域，禁止为满足条款新增物件")}


def cross_theme_prompt(spec, theme, scheme=None):
    """组装一个请求的正文：subject + rendering +（配色组才有）COLOUR PLAN + 区域绑定 + 保护块。"""
    layout = spec["prompt_layout"]
    parts = [theme["subject"], theme["rendering"]]
    if scheme is not None:
        clauses = list(scheme["clauses"])
        mapping = theme["region_mapping"]
        binding = layout["region_binding_template"].format(**{
            "environment_dominant_region": mapping["environment_dominant_region"]["describes"],
            "waist_band_region": mapping["waist_band_region"]["describes"],
            "left_shoe_bow_region": mapping["left_shoe_bow_region"]["describes"],
            "right_shoe_bow_region": mapping["right_shoe_bow_region"]["describes"]})
        parts.append(layout["headers"]["color_plan"] + "\n- " + "\n- ".join(clauses + [binding]))
    parts.append(layout["headers"]["protection"] + " " + layout["protection_block"])
    return layout["separator"].join(parts)


def _cross_theme_base_prompt(prompt):
    """取「主题 + 画法」那一段：剥掉 COLOUR PLAN 与 PRESERVE 之后的正文。"""
    text = str(prompt or "")
    for marker in ("COLOUR PLAN:", "PRESERVE:"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.rstrip()


def cross_theme_requests(spec, *, out_dir=None):
    """组装 16 条请求：每主题 3 方案 × 2 次 + 无配色对照 × 2 次；全部不附参考图。"""
    out = _resolve_retention_out(out_dir) if out_dir else _retention_path(spec["_out_dir"])
    rows = []
    for theme in spec["themes"]:
        plan = [(scheme, repeat) for scheme in spec["schemes"] for repeat in (1, 2)]
        plan += [(None, repeat) for repeat in (1, 2)]
        for scheme, repeat in plan:
            scheme_id = scheme["id"] if scheme else spec["control_group"]["id"]
            request_id = f"{theme['id']}--{scheme_id}-r{repeat}"
            prompt = cross_theme_prompt(spec, theme, scheme)
            rows.append({
                "request_id": request_id, "theme": theme["id"], "scheme": scheme_id,
                "repeat": repeat, "is_control": scheme is None,
                "prompt": prompt, "prompt_sha256": _sha256_str(prompt), "prompt_chars": len(prompt),
                "reference_images": [], "image_paths": [],
                "stage_dir": str(out / "samples" / theme["id"] / f"{scheme_id}-r{repeat}"),
                "file_prefix": f"cross-{theme['id']}-{scheme_id}-r{repeat}",
                "colour_plan_present": scheme is not None,
                "contract_hash": cross_theme_contract(spec)["contract_hash"],
            })
    return rows


def cross_theme_preflight(out_dir, *, spec_path=None, log=print):
    """离线预检：合同/条款/图序/模型参数/预算/冲突检查，全部不联网。"""
    spec = load_cross_theme_spec(spec_path)
    out = _resolve_retention_out(out_dir) if out_dir else _retention_path(spec["_out_dir"])
    contract = cross_theme_contract(spec)
    conflicts = cross_theme_conflict_report(spec)
    requests = cross_theme_requests(spec, out_dir=out)

    from modules.others.api_backend import get_api_config, resolve_image_max_retries
    channel = spec["generation"]["channel"]
    # 预检与执行共用同一套开关：重试必须在**任何请求发出之前**就是 0
    retry_env = spec["generation"]["retry_env"] or {"IMAGE_MAKER_IMAGE_MAX_RETRIES": "0"}
    for key, value in retry_env.items():
        os.environ.setdefault(key, str(value))
    cfg = get_api_config(api_type=spec["generation"].get("api_type") or "aigc2d")
    retries = resolve_image_max_retries(cfg, 1)
    env_retries = str(os.environ.get("IMAGE_MAKER_IMAGE_MAX_RETRIES") or "").strip()
    from utils.gpt_image_optimize import load_config as load_repaint_config
    firmware = load_repaint_config()
    # 注意：这条链路真正发送的模型是 `generate_image_aigc2d` 的默认值（文本→图片通道），
    # 不是 repaint 固件里的模型。必须按**实际调用路径**记录，避免预检报一个不会发出的模型。
    # 用「导入时的函数签名」而不是运行时属性：测试会 monkeypatch 掉这个函数，属性反射会失败。
    from modules.others.api_backend import generate_image_aigc2d
    import inspect
    signature = inspect.signature(generate_image_aigc2d)
    model_parameter = signature.parameters.get("model")
    actual_text_to_image_model = model_parameter.default if model_parameter is not None else None
    resolved = {"model": actual_text_to_image_model,
                "api_type": spec["generation"].get("api_type") or "aigc2d",
                "resolution": cfg.get("resolution"),
                "aspect_ratio": cfg.get("aspect_ratio") or firmware.get("aspect_ratio"),
                "model_source": "generate_image_aigc2d 的默认模型（文本→图片通道实际发送值）",
                "model_unknown_reason": (None if actual_text_to_image_model
                                         else "调用函数被替换过，取不到默认模型：本轮不得据此断言模型"),
                "repaint_firmware_model": firmware.get("model"),
                "sent_model_note": ("配置里的 model 不参与这条链路；如需换模型必须显式传参，"
                                    "本轮不换")}
    problems = list(conflicts["problems"])
    if retries != 0:
        problems.append("底层图片重试没有关闭：必须设置 IMAGE_MAKER_IMAGE_MAX_RETRIES=0")
    if spec["generation"]["image_paths"]:
        problems.append("本轮不允许附参考图（image_paths 必须为空）")
    if len(requests) != spec["design"]["total_images"]:
        problems.append(f"请求数 {len(requests)} 与设计 {spec['design']['total_images']} 不一致")
    # 同主题内「内容/画法/模型/参数」必须逐字相同，只有配色条款不同
    per_theme = {}
    for row in requests:
        per_theme.setdefault(row["theme"], []).append(row)
    for theme_id, rows in per_theme.items():
        bases = {_cross_theme_base_prompt(row["prompt"]) for row in rows}
        if len(bases) != 1:
            problems.append(f"{theme_id}：同主题内的 subject/rendering 前缀不一致")
    control_bodies = [row for row in requests if row["is_control"]]
    for row in control_bodies:
        if "COLOUR PLAN:" in row["prompt"]:
            problems.append(f"{row['request_id']}：无配色对照不应出现 COLOUR PLAN 段")
    result = {"label": CROSS_THEME_LABEL, "at": _now(), "spec_path": spec["_path"],
              "out_dir": str(out), "contract": contract, "conflicts": conflicts,
              "resolved_generation": resolved, "retries_resolved": retries,
              "retries_env": env_retries,
              "budget": {"generation_max": spec["design"]["total_generation_calls"],
                         "review_max": 8, "automatic_retries": 0},
              "requests": [{key: value for key, value in row.items() if key != "prompt"}
                           for row in requests],
              "prompt_hashes": {row["request_id"]: row["prompt_sha256"] for row in requests},
              "problems": problems, "status": "problems" if problems else "passed",
              "no_network": True}
    write_json_atomic(str(out / "cross-preflight.json"), result)
    log(f"[cross] 预检 {result['status']}：{len(requests)} 条请求 / {len(problems)} 个问题 / "
        f"模型 {resolved['model']} @{resolved['resolution']} / 重试 {retries}")
    for problem in problems:
        log(f"[cross] 问题：{problem}")
    return result


def cross_theme_freeze(out_dir, preflight=None, *, log=print):
    """冻结：spec/合同/条款/主题/模型参数/逐条请求 hash。"""
    out = _resolve_retention_out(out_dir)
    preflight = preflight or json.loads((out / "cross-preflight.json").read_text(encoding="utf-8"))
    spec = load_cross_theme_spec(preflight["spec_path"])
    frozen = {"label": CROSS_THEME_LABEL, "frozen_at": _now(),
              "spec_path": spec["_path"], "spec_sha256": _sha256_file(Path(spec["_path"])),
              "contract": preflight["contract"],
              "schemes": {scheme["id"]: {"main": scheme["main"], "accent": scheme["accent"],
                                         "clauses_sha256": _sha256_str("\n".join(scheme["clauses"]))}
                          for scheme in spec["schemes"]},
              "themes": {theme["id"]: {"region_mapping": theme["region_mapping"],
                                       "subject_sha256": _sha256_str(theme["subject"]),
                                       "rendering_sha256": _sha256_str(theme["rendering"])}
                         for theme in spec["themes"]},
              "protection_block_sha256": _sha256_str(spec["prompt_layout"]["protection_block"]),
              "generation": preflight["resolved_generation"],
              "reference_images": "none",
              "request_hashes": preflight["prompt_hashes"]}
    frozen["freeze_hash"] = digest(frozen)
    write_json_atomic(str(out / "cross-frozen.json"), frozen)
    log(f"[cross] 协议已冻结 freeze_hash={frozen['freeze_hash'][:16]}（{len(frozen['request_hashes'])} 条请求）")
    return frozen


def cross_theme_generation_preflight(spec, frozen):
    """执行前核对冻结记录：模型/参数/条款/主题/请求 hash 任一变化都拒绝复用。"""
    differences = []
    if _sha256_file(Path(spec["_path"])) != frozen.get("spec_sha256"):
        differences.append("spec 内容变化")
    contract = cross_theme_contract(spec)
    if contract["contract_hash"] != (frozen.get("contract") or {}).get("contract_hash"):
        differences.append("合同内容变化")
    if _sha256_str(spec["prompt_layout"]["protection_block"]) != frozen.get("protection_block_sha256"):
        differences.append("保护块变化")
    for scheme in spec["schemes"]:
        entry = (frozen.get("schemes") or {}).get(scheme["id"]) or {}
        if _sha256_str("\n".join(scheme["clauses"])) != entry.get("clauses_sha256"):
            differences.append(f"{scheme['id']} 的条款变化")
    for theme in spec["themes"]:
        entry = (frozen.get("themes") or {}).get(theme["id"]) or {}
        if _sha256_str(theme["subject"]) != entry.get("subject_sha256"):
            differences.append(f"{theme['id']} 的主题文本变化")
    current = {row["request_id"]: row["prompt_sha256"] for row in cross_theme_requests(spec)}
    for key in sorted(set(current) | set(frozen.get("request_hashes") or {})):
        if current.get(key) != (frozen.get("request_hashes") or {}).get(key):
            differences.append(f"请求 hash 变化：{key}")
    return differences


def cross_theme_colour_injection_allowed(*, style_ref_path="", style_ref_clauses=None, request=None):
    """有有效画风参考图时**禁止**选择或注入配色。

    返回 `(allowed, reason)`；不允许时调用方必须原样使用未注入配色的请求。
    本函数只做校验，不修改请求、不调用任何模型。
    """
    path = str(style_ref_path or "").strip()
    if path and Path(path).is_file():
        return False, f"检测到有效画风参考图（{Path(path).name}）：本轮禁止选择或注入固定配色"
    if style_ref_clauses:
        return False, "本次请求带画风渲染条款（等价于带画风参考图）：禁止注入固定配色"
    if request and (request.get("extra_reference_paths") or request.get("style_ref_path")):
        return False, "请求里已经挂了画风参考图：禁止注入固定配色"
    return True, "无画风参考图，允许使用固定配色"


def cross_theme_disabled_request(original):
    """未启用配色时的合同：**原请求逐字不变**，且不额外调用任何模型。

    `original` 只读，不做任何写操作，保证返回的 prompt 与原请求字节一致。
    """
    return {"prompt": original.get("prompt", ""), "unchanged": True,
            "prompt_sha256": _sha256_str(original.get("prompt", "")),
            "extra_calls": 0,
            "note": "未启用配色：原请求逐字不变，不注入配色、不额外调用模型"}


def send_budget_env():
    """测试用：send_budget 的预算目录与数量上限两个环境变量名。"""
    from utils import send_budget
    return send_budget.ENV_DIR, send_budget.ENV_LIMIT


def cross_theme_cache_eligibility(frozen, *, spec, request_id):
    """合同/条款/主题/模型参数任一变化 → 相关缓存失效（不再复用旧产物）。"""
    differences = cross_theme_generation_preflight(spec, frozen)
    return {"eligible": not differences, "differences": differences,
            "state": "verified_match" if not differences else "invalidated",
            "request_id": request_id}


def cross_theme_snapshot(out_dir, *, spec=None, log=print):
    """任务快照：逐条请求的状态/产物/hash，可据此恢复（已成功的复用，不重发）。"""
    spec = spec or load_cross_theme_spec()
    out = _resolve_retention_out(out_dir)
    rows = []
    for request in cross_theme_requests(spec, out_dir=out):
        stage = Path(request["stage_dir"])
        record_path = stage / "result.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        rows.append({"request_id": request["request_id"], "theme": request["theme"],
                     "scheme": request["scheme"], "repeat": request["repeat"],
                     "is_control": request["is_control"], "prompt_sha256": request["prompt_sha256"],
                     "status": record.get("status", "pending"),
                     "outputs": record.get("outputs") or [],
                     "output_sha256": record.get("output_sha256") or {},
                     "slot": record.get("slot"), "error": record.get("error", "")})
    snapshot = {"label": CROSS_THEME_LABEL, "at": _now(), "out_dir": str(out), "rows": rows,
                "done": len([row for row in rows if row["status"] == "success"]),
                "total": len(rows)}
    write_json_atomic(str(out / "cross-snapshot.json"), snapshot)
    log(f"[cross] 快照：{snapshot['done']}/{snapshot['total']} 已成功")
    return snapshot


def cross_theme_plan(out_dir, *, spec_path=None, log=print):
    """生成任务计划（含每条请求的主题/方案/重复/正文 hash），供预览与冻结共用。"""
    spec = load_cross_theme_spec(spec_path)
    out = _resolve_retention_out(out_dir) if out_dir else _retention_path(spec["_out_dir"])
    plan = {"label": CROSS_THEME_LABEL, "at": _now(), "spec_path": spec["_path"],
            "contract": cross_theme_contract(spec),
            "requests": cross_theme_requests(spec, out_dir=out),
            "budget": spec["budget_policy"], "evaluation": spec["evaluation"]}
    write_json_atomic(str(out / "cross-plan.json"), plan)
    log(f"[cross] 计划 {len(plan['requests'])} 条请求 → {out / 'cross-plan.json'}")
    return plan


def validate_cross_theme_review(value, expected_ids):
    """配色评审校验：逐张覆盖一次、枚举合法、三个落点分别给出、不得混入美学评分。"""
    if not isinstance(value, dict):
        raise ValueError("配色评审必须是 JSON 对象")
    rows = value.get("images")
    if not isinstance(rows, list) or sorted(str(row.get("id")) for row in rows) != sorted(expected_ids):
        raise ValueError("配色评审必须恰好覆盖本组每张图一次")
    for row in rows:
        item_id = str(row.get("id"))
        for key in ("main_colour_status", "accent_status", "colour_status"):
            if row.get(key) not in CROSS_THEME_STATES:
                raise ValueError(f"{item_id}: {key} 非法")
        for key in CROSS_THEME_ACCENT_REGIONS:
            if row.get(key) not in ("present", "absent", "moved", "unverifiable"):
                raise ValueError(f"{item_id}: {key} 必须逐项给出")
        if str(row.get("protected_colours_ok")).lower() not in ("true", "false", "unclear"):
            raise ValueError(f"{item_id}: protected_colours_ok 非法")
        if row.get("aesthetic_score") not in (None, ""):
            raise ValueError(f"{item_id}: 不得混入美学评分")
    return value


def cross_theme_run(out_dir, *, spec_path=None, log=print, dry_run=False):
    """执行：冲突/冻结核对 → 预算前置 → 16 次首图直出（无参考图）→ 汇总。"""
    from modules.others.api_backend import generate_image_aigc2d
    from utils import send_budget
    spec = load_cross_theme_spec(spec_path)
    out = _resolve_retention_out(out_dir) if out_dir else _retention_path(spec["_out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    os.environ["IMAGE_MAKER_IMAGE_MAX_RETRIES"] = "0"
    os.environ[send_budget.ENV_DIR] = str(out / "send-budget")
    os.environ[send_budget.ENV_LIMIT] = str(spec["design"]["total_generation_calls"])
    preflight = cross_theme_preflight(out, spec_path=spec_path, log=log)
    if preflight["status"] != "passed":
        log("[cross] 预检未通过：停止，不发送任何请求")
        return {"preflight": preflight, "stopped": "preflight_problems", "stages": []}
    frozen_path = out / "cross-frozen.json"
    if frozen_path.is_file():
        differences = cross_theme_generation_preflight(spec, json.loads(frozen_path.read_text(encoding="utf-8")))
        if differences:
            raise ValueError("冻结记录与当前协议不一致，拒绝复用；请新开目录/新协议：" + "；".join(differences))
    frozen = cross_theme_freeze(out, preflight, log=log)
    if dry_run:
        return {"preflight": preflight, "frozen": frozen, "stages": [], "dry_run": True}
    stages, consecutive_failures = [], 0
    for request in cross_theme_requests(spec, out_dir=out):
        record = cross_theme_stage(out, spec, request, generate_image_aigc2d, send_budget, log=log)
        stages.append(record)
        if record["status"] == "success":
            consecutive_failures = 0
        elif record["status"] == "failed":
            consecutive_failures += 1
            if consecutive_failures >= 2:
                log("[cross] 连续两次请求失败：停止，不补抽、不重试")
                break
    summary = cross_theme_summary(out, spec=spec, log=log)
    ledger = cross_theme_ledger(out, spec=spec, log=log)
    return {"preflight": preflight, "frozen": frozen, "stages": stages,
            "summary": summary, "ledger": ledger,
            "stopped": "two_consecutive_failures" if consecutive_failures >= 2 else None}


def cross_theme_stage(out: Path, spec, request, generate_image_aigc2d, send_budget, *, log=print):
    """一次首图直出：预算前置 → 发送（无参考图、无重试）→ 结算并落盘。"""
    stage = Path(request["stage_dir"])
    stage.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(stage / "request.json"), request)
    existing = stage / "result.json"
    if existing.is_file():
        record = json.loads(existing.read_text(encoding="utf-8"))
        if record.get("status") == "success" and record.get("prompt_sha256") == request["prompt_sha256"]:
            log(f"[cross] {request['request_id']} 已成功，复用（不重发）")
            return record
    limit = int(spec["design"]["total_generation_calls"])
    before = send_budget.state(kind="image", limit=limit)
    if int(before.get("remaining") or 0) <= 0:
        record = {"stage": "first-pass", "request_id": request["request_id"], "status": "refused_by_budget",
                  "error": f"图片额度已用满（{before.get('used')}/{limit}）", "at": _now()}
        write_json_atomic(str(existing), record)
        log(f"[cross] {request['request_id']} 被预算拒绝，未发送")
        return record
    reservation = send_budget.reserve_image_attempt(
        run_id=request["request_id"], operation=f"cross-{request['request_id']}",
        model=spec["generation"].get("model") or "", resolution="",
        prompt_sha256=request["prompt_sha256"], prompt_chars=request["prompt_chars"],
        note="跨主题首图直出（无参考图）")
    error_text = ""
    try:
        outputs = generate_image_aigc2d(
            prompt=request["prompt"], image_paths=None,
            aspect_ratio=str(spec["generation"].get("aspect_ratio") or ""),
            resolution=spec["generation"].get("resolution"),
            api_type=spec["generation"].get("api_type"),
            save_sub_dir=str(stage), file_prefix=request["file_prefix"],
            face_quality_boost=bool(spec["generation"].get("face_quality_boost", False)),
            log_callback=log)
        outputs = [str(_retention_resolve(path).resolve()) for path in (outputs or [])
                   if Path(str(path)).is_file()]
    except Exception as error:  # noqa: BLE001
        outputs = []
        error_text = f"{type(error).__name__}: {error}"
    if outputs:
        for path in outputs:
            send_budget.settle_image_attempt(reservation, status="success", output_path=path,
                                            output_sha256=_sha256_file(path))
    else:
        send_budget.settle_image_attempt(reservation, status="failed", error=error_text[:300])
    record = {"stage": "first-pass", "request_id": request["request_id"], "theme": request["theme"],
              "scheme": request["scheme"], "repeat": request["repeat"], "is_control": request["is_control"],
              "status": "success" if outputs else "failed",
              "outputs": outputs, "output_sha256": {Path(p).name: _sha256_file(p) for p in outputs},
              "slot": reservation.get("slot"), "at": _now(), "error": error_text[:300],
              "prompt_sha256": request["prompt_sha256"], "prompt_chars": request["prompt_chars"],
              "reference_images": [], "colour_plan_present": request["colour_plan_present"]}
    write_json_atomic(str(existing), record)
    log(f"[cross] {request['request_id']} {record['status']}（槽位 {record['slot']}，"
        f"{len(outputs)} 张）" + (f" {error_text[:100]}" if error_text else ""))
    return record


def cross_theme_summary(out_dir, *, spec=None, log=print):
    """汇总：逐张状态 + 逐组评审结论（配色组与对照组分开体现）。"""
    spec = spec or load_cross_theme_spec()
    out = _resolve_retention_out(out_dir)
    snapshot = cross_theme_snapshot(out, spec=spec, log=lambda *a, **k: None)
    reviews = {}
    for path in sorted((out / "review").glob("group-*.json")):
        if path.name.endswith((".request.json", ".failed.json", ".invalid.json")):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        reviews[record["group_id"]] = record
    groups = {}
    for row in snapshot["rows"]:
        key = f"{row['theme']}::{row['scheme']}"
        group = groups.setdefault(key, {"theme": row["theme"], "scheme": row["scheme"],
                                        "is_control": row["is_control"], "images": []})
        group["images"].append({"repeat": row["repeat"], "status": row["status"],
                                "outputs": row["outputs"]})
    for key, group in groups.items():
        entry = reviews.get(key)
        group["review"] = entry.get("response") if entry else None
        group["review_status"] = entry.get("status") if entry else "not_reviewed"
    summary = {"label": CROSS_THEME_LABEL, "at": _now(), "spec_path": spec["_path"],
               "contract": cross_theme_contract(spec),
               "generation_total": snapshot["total"], "generation_success": snapshot["done"],
               "generation_failed": snapshot["total"] - snapshot["done"],
               "groups": groups,
               "evaluation": spec["evaluation"],
               "note": ("两次重复只能作为探索证据；无配色对照不用配色方案的判据判通过或失败；"
                        "不评画质与美感。")}
    write_json_atomic(str(out / "cross-summary.json"), summary)
    write_json_atomic(str(out / "cross-per-image.json"), {"rows": snapshot["rows"]})
    log(f"[cross] 汇总：{summary['generation_success']}/{summary['generation_total']} 成功，"
        f"{len(groups)} 组")
    return summary


def cross_theme_ledger(out_dir, *, spec=None, log=print):
    """调用账本：图片与文本分别计数，失败与未执行分开。"""
    from utils import send_budget
    spec = spec or load_cross_theme_spec()
    out = _resolve_retention_out(out_dir)
    images = send_budget.state(str(out / "send-budget"), "image",
                              int(spec["design"]["total_generation_calls"]))
    texts = send_budget.state(str(out / "send-budget"), "text", 8)
    settlements = images.get("settlements") or {}
    rows = list(images.get("slots") or [])
    ledger = {"label": CROSS_THEME_LABEL, "at": _now(), "budget_dir": str(out / "send-budget"),
              "image": {"limit": images["limit"], "reserved": len(rows),
                        "success": len([r for r in rows
                                        if (settlements.get(r.get("attempt_id")) or {}).get("status") == "success"]),
                        "failed_or_unknown": len([r for r in rows
                                                  if (settlements.get(r.get("attempt_id")) or {}).get("status")
                                                  not in ("success", None)]),
                        "attempts": [{"slot": row.get("slot"), "request": row.get("run_id"),
                                      "status": (settlements.get(row.get("attempt_id")) or {}).get("status")}
                                     for row in rows]},
              "text": {"limit": texts["limit"], "reserved": len(texts.get("slots") or [])},
              "retries": "IMAGE_MAKER_IMAGE_MAX_RETRIES=0，自动重试 0 次",
              "note": "估算与真实账单分开；服务端返回的用量按实际响应记录。"}
    write_json_atomic(str(out / "cross-ledger.json"), ledger)
    log(f"[cross] 账本：图片 {ledger['image']['success']} 成功 / {ledger['image']['failed_or_unknown']} 失败或未知"
        f" / {ledger['image']['reserved']} 次预留")
    return ledger


def write_cross_theme_gallery(out_dir, *, spec=None, log=print):
    """离线评价页：内嵌实际产图，逐组列出配色/落点/保护色/内容结论。"""
    from html import escape
    spec = spec or load_cross_theme_spec()
    out = _resolve_retention_out(out_dir)
    summary = json.loads((out / "cross-summary.json").read_text(encoding="utf-8"))
    ledger = json.loads((out / "cross-ledger.json").read_text(encoding="utf-8"))
    frozen_path = out / "cross-frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.is_file() else {}

    def tag(state):
        return {"conform": "ok", "partial": "warn", "fail": "bad", "unverifiable": "unk",
                "success": "ok", "failed": "bad", "refused_by_budget": "bad",
                "present": "ok", "absent": "bad", "moved": "warn",
                "true": "ok", "false": "bad", "unclear": "unk"}.get(str(state), "unk")

    def image(path, label):
        if not path or not Path(path).is_file():
            return f'<figure class="missing">{escape(label)}：无产物</figure>'
        return (f'<figure><img loading="lazy" src="{escape(_thumb_data_uri(path, 560, 80))}" '
                f'alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>')

    blocks = []
    for theme in spec["themes"]:
        cards = []
        for key, group in summary["groups"].items():
            if group["theme"] != theme["id"]:
                continue
            review_rows = ((group.get("review") or {}).get("images") or [])
            by_id = {str(row.get("id")): row for row in review_rows}
            shots = []
            for index, item in enumerate(group["images"], 1):
                verdict = by_id.get(f"Q{index:02d}") or {}
                label = (f"{group['scheme']} · 第 {item['repeat']} 次 · "
                         f"{'配色' if not group['is_control'] else '无配色对照'}")
                cell = [image(item["outputs"][0] if item["outputs"] else "", label)]
                if verdict:
                    cell.append(
                        f'<p>主色 <span class="tag {tag(verdict.get("main_colour_status"))}">'
                        f'{escape(str(verdict.get("main_colour_status")))}</span> ｜ 点缀 '
                        f'<span class="tag {tag(verdict.get("accent_status"))}">{escape(str(verdict.get("accent_status")))}</span>'
                        f' ｜ 总 <span class="tag {tag(verdict.get("colour_status"))}">{escape(str(verdict.get("colour_status")))}</span></p>'
                        f'<p class="muted">落点：束带={escape(str(verdict.get("waist_band_region")))} ｜ '
                        f'左鞋={escape(str(verdict.get("left_shoe_bow_region")))} ｜ '
                        f'右鞋={escape(str(verdict.get("right_shoe_bow_region")))} ｜ '
                        f'保护色={escape(str(verdict.get("protected_colours_ok")))}</p>')
                    if verdict.get("new_colour_regions"):
                        cell.append('<p class="muted">新增色块：'
                                    + escape("；".join(str(x) for x in verdict["new_colour_regions"])[:400]) + '</p>')
                    if verdict.get("content_changes"):
                        cell.append('<p class="muted">内容变化：'
                                    + escape("；".join(str(x) for x in verdict["content_changes"])[:300]) + '</p>')
                    if verdict.get("notes"):
                        cell.append(f'<p class="muted">{escape(str(verdict["notes"])[:300])}</p>')
                shots.append('<section class="shot">' + "".join(cell) + '</section>')
            cards.append(
                f'<section class="card"><h3>{escape(group["scheme"])}'
                f'{"（无配色对照）" if group["is_control"] else ""} '
                f'<span class="tag {"unk" if group["is_control"] else "ok"}">'
                f'{escape(str(group["review_status"]))}</span></h3>'
                f'<div class="row">{"".join(shots)}</div></section>')
        blocks.append(f'<section class="group"><h2>{escape(theme["label"])}'
                      f'（{escape(theme["id"])}）</h2>'
                      f'<p class="muted">{escape(theme["why_this_theme"])}</p>' + "".join(cards) + "</section>")
    header = (f'<h1>固定配色 · 跨主题执行验证（无画风参考图）</h1>'
              f'<p class="muted">协议 {escape(CROSS_THEME_LABEL)} · 模型 '
              f'{escape(str((frozen.get("generation") or {}).get("model")))} @'
              f'{escape(str((frozen.get("generation") or {}).get("resolution")))} · '
              f'合同 hash {escape(str(summary["contract"]["contract_hash"])[:16])} · '
              f'freeze {escape(str(frozen.get("freeze_hash") or "")[:16])}</p>'
              '<aside><b>判读口径</b>'
              '<p>只判环境主色、点缀色与三个既有落点、保护色、新增色块与内容变化；<b>不评画质与美感</b>。</p>'
              '<p>每组两次重复只是探索证据，不宣称普遍稳定；无配色对照不套用配色判据。</p>'
              '<p>全部请求都没有画风参考图；带参考图时禁止注入配色由校验函数保证，不是本轮结论。</p></aside>'
              f'<p>图片 {escape(str(ledger["image"]["success"]))} 成功 / '
              f'{escape(str(ledger["image"]["failed_or_unknown"]))} 失败或未知 ｜ 文本 '
              f'{escape(str(ledger["text"]["reserved"]))}/{escape(str(ledger["text"]["limit"]))} ｜ '
              f'自动重试 0</p>'
              f'<p><a href="cross-summary.json">cross-summary.json</a> · '
              f'<a href="cross-per-image.json">cross-per-image.json</a> · '
              f'<a href="cross-frozen.json">cross-frozen.json</a> · '
              f'<a href="cross-preflight.json">cross-preflight.json</a> · '
              f'<a href="cross-ledger.json">cross-ledger.json</a></p>')
    css = ("body{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#25322b;background:#eef1ee;margin:0}"
           "main{max-width:1320px;margin:auto;padding:24px}"
           "section.group{background:#fff;border-radius:14px;padding:22px;margin:20px 0}"
           "section.card{border:1px solid #e0e6e1;border-radius:10px;padding:16px;margin:14px 0;background:#fbfdfb}"
           ".row{display:flex;gap:14px;flex-wrap:wrap}"
           "section.shot{flex:1 1 420px;min-width:320px;max-width:520px}"
           "figure{margin:0}figure img{width:100%;border-radius:8px;border:1px solid #d8ded9}"
           "figure.missing{padding:30px;background:#fbe9ea;color:#a12e37;border-radius:8px}"
           "figcaption{font-size:12.5px;color:#657269}"
           ".tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}"
           ".tag.ok{background:#2f8f5b}.tag.warn{background:#c98a17}.tag.bad{background:#b23b45}.tag.unk{background:#6b7785}"
           ".muted{color:#657269;font-size:13px}")
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>跨主题固定配色验证</title>', f'<style>{css}</style><main>',
             header, "".join(blocks), '</main></html>']
    path = out / "cross-review.html"
    path.write_text("".join(parts), encoding="utf-8")
    log(f"[cross] 离线评价页 → {path}")
    return path


def cross_theme_colour_review(out_dir, *, spec=None, spec_path=None, log=print):
    """配色评审：每组（同主题同方案的两张）一次调用，逐张分开判定。"""
    from utils.analysis_gpt_prompt import call_text_model
    spec = spec or load_cross_theme_spec(spec_path)
    out = _resolve_retention_out(out_dir)
    system = read_prompt_file(CROSS_THEME_COLOUR_REVIEW_PROMPT)
    cfg = _text_cfg()
    plan = out / "review"
    plan.mkdir(parents=True, exist_ok=True)
    snapshot = cross_theme_snapshot(out, spec=spec, log=lambda *a, **k: None)
    by_group = {}
    for row in snapshot["rows"]:
        by_group.setdefault(f"{row['theme']}::{row['scheme']}", []).append(row)
    records = []
    for group_id, rows in sorted(by_group.items()):
        target = plan / f"group-{group_id.replace('::', '-')}.json"
        if target.is_file():
            log(f"[cached] 配色评审 {group_id}")
            records.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        usable = [row for row in rows if row["status"] == "success" and row["outputs"]]
        if not usable:
            log(f"[cross] {group_id}: 没有可用产物，跳过评审（不调用）")
            continue
        theme_id, scheme_id = group_id.split("::")
        theme = next(t for t in spec["themes"] if t["id"] == theme_id)
        scheme = next((s for s in spec["schemes"] if s["id"] == scheme_id), None)
        images, order = [], []
        for index, row in enumerate(sorted(usable, key=lambda r: r["repeat"]), 1):
            images.extend(_reassess_proxy(out, row["outputs"][0]))
            order.append({"id": f"Q{index:02d}", "repeat": row["repeat"],
                          "candidate_sha256": row["output_sha256"].get(
                              Path(row["outputs"][0]).name)})
        payload = {"task": "cross_theme_colour_conformance",
                   "detail_note": _reassess_detail_note(),
                   "theme": {"id": theme_id, "label": theme["label"],
                             "region_mapping": theme["region_mapping"]},
                   "scheme": ({"id": scheme["id"], "main": scheme["main"], "accent": scheme["accent"],
                               "clauses": scheme["clauses"]} if scheme else None),
                   "is_control_group": scheme is None,
                   "colour_contract": cross_theme_contract(spec)["contract"],
                   "accent_regions": list(CROSS_THEME_ACCENT_REGIONS),
                   "protected_colours": theme["protected"],
                   "statuses": list(CROSS_THEME_STATES),
                   "image_order": [{"id": row["id"]} for row in order],
                   "note": ("逐张分开判定；区域落点优先于全图色调。"
                            "若是无配色对照，只判它有没有改保护色/加物件/改内容，不要用配色条款判它")}
        user = json.dumps(payload, ensure_ascii=False)
        request_record = {"purpose": "cross_theme_colour_review", "group_id": group_id,
                          "at": _now(), "model": cfg["model"], "base_url": cfg["base_url"],
                          "system_prompt_file": CROSS_THEME_COLOUR_REVIEW_PROMPT,
                          "system": system, "user": user,
                          "images": [{"file": Path(p).name, "sha256": _sha256_file(p)} for p in images]}
        write_json_atomic(str(plan / f"group-{group_id.replace('::', '-')}.request.json"), request_record)
        raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system, user,
                              max_tokens=REASSESS_MAX_TOKENS, timeout=REASSESS_TIMEOUT,
                              image_paths=[str(p) for p in images])
        (plan / f"group-{group_id.replace('::', '-')}.raw.txt").write_text(raw, encoding="utf-8")
        value, note = parse_object_tolerant(raw, log=log)
        record = {"group_id": group_id, "at": _now(), "model": cfg["model"], "json_repair": note,
                  "mapping": {row["id"]: f"{row['repeat']}" for row in order},
                  "candidate_hashes": {row["id"]: row["candidate_sha256"] for row in order},
                  "response": value, "status": "review_record_not_gate",
                  "experiment_evidence_source": "自建首个配色资产（本项目最早版本无 assets/ 目录）"}
        try:
            if value is None:
                raise ValueError("响应不可解析")
            validate_cross_theme_review(value, [row["id"] for row in order])
            record["status"] = "reviewed"
        except ValueError as error:
            record["status"] = "unusable"
            record["validation_error"] = str(error)
            write_json_atomic(str(plan / f"group-{group_id.replace('::', '-')}.invalid.json"), record)
        write_json_atomic(str(target), record)
        records.append(record)
        log(f"[cross] 配色评审 {group_id}: {record['status']}")
    return records


def write_reassessment_gallery(out_dir, log=print):
    """可离线浏览的逐张逐条评价页：图内嵌为 data URI，不依赖网络与外部文件。"""
    from html import escape
    out = _resolve_retention_out(out_dir)
    items = load_reassess_items()
    by_item = {row["item_id"]: row for row in items}
    summary_path = out / "reassessment-summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file()
               else reassess_summary(out, items, log=log))
    preflight_path = out / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    stats = {row["item_id"]: row for row in (preflight.get("items") or [])}
    protocol_path = out / "reassessment-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    rows = summary["rows"]
    by_group = {}
    for row in rows:
        by_group.setdefault((row["suite"], row["group_id"]), []).append(row)

    def status_class(state):
        return {"conform": "ok", "partial": "warn", "diverge": "bad", "unclear": "unk"}.get(state, "unk")

    def card(row):
        source = by_item[row["item_id"]]
        stats_row = stats.get(row["item_id"]) or {}
        local = stats_row.get("local_stats") or {}
        families = "，".join(f'{f["mean_hex"]}（{f["hue_band"]}，占彩度像素 {f["weight_share_of_chromatic"]:.0%}）'
                            for f in (local.get("dominant_colour_families") or []))
        clause_html = "".join(
            f'<li><b>条款 {entry.get("index")}：</b>{escape(str(entry.get("target")))}<br>'
            f'<span class="tag {status_class(entry.get("status"))}">{escape(str(entry.get("status")))}</span> '
            f'区域：{escape("；".join(entry.get("regions") or []) or "未指明")} · '
            f'实际颜色：{escape("；".join(entry.get("observed_colours") or []) or "未记录")}<br>'
            f'证据：{escape("；".join(entry.get("evidence") or []))}</li>'
            for entry in row["clause_rows"]) or '<li class="muted">本次无配色条款（基线段，不按其他方案验收）。</li>'
        protected_html = "".join(
            f'<li>{escape(str(entry.get("name")))}：'
            f'<span class="tag {status_class("conform" if entry.get("status") == "retained" else ("bad" if entry.get("status") == "altered" else "unk"))}">'
            f'{escape(str(entry.get("status")))}</span> {escape("；".join(entry.get("evidence") or []))}</li>'
            for entry in row["protected_colours"])
        observations = row["observations"]
        lists = [("区域布局", [observations.get("region_layout") or ""]),
                 ("基调/辅助/强调", [observations.get("base_aux_accent") or ""]),
                 ("冷暖关系", [observations.get("warm_cool") or ""]),
                 ("大块非方案颜色", observations.get("large_non_plan_areas") or []),
                 ("内容观察", observations.get("content_notes") or []),
                 ("内容问题", observations.get("content_issues") or []),
                 ("画法观察", observations.get("method_notes") or []),
                 ("画法问题", observations.get("method_issues") or []),
                 ("无法确认的主张", observations.get("unsupported_claims") or [])]
        observation_html = "".join(
            f'<p><b>{escape(title)}：</b>{escape("；".join(text for text in texts if text) or "—")}</p>'
            for title, texts in lists)
        clauses_text = "\n".join(f"- {clause}" for clause in source["clauses"]) or "（本次无配色条款）"
        return (f'<section class="card"><h3>{escape(row["item_id"])} · 第 {row["cycle"]} 轮 · '
                f'{escape(row["scheme_label"])} <span class="tag {status_class(row["colour_status"])}">'
                f'{escape(row["colour_status"])}</span></h3>'
                f'<div class="row"><figure><img loading="lazy" src="{escape(_thumb_data_uri(source["image"], 900, 82))}" '
                f'alt="{escape(row["item_id"])}"></figure>'
                f'<div class="col"><details open><summary>本次实际配色条款（逐字）</summary>'
                f'<pre>{escape(clauses_text)}</pre></details>'
                f'<details><summary>实际请求全文</summary><pre>{escape(source.get("prompt_text") or "缺失")}</pre></details>'
                f'<details><summary>本地辅助统计（全局，无区域蒙版）</summary>'
                f'<pre>{escape(json.dumps(local, ensure_ascii=False, indent=1))}</pre>'
                f'<p class="muted">主要色相族：{escape(families or "—")}</p></details></div></div>'
                f'<h4>逐条符合度</h4><ul>{clause_html}</ul>'
                f'<h4>受保护颜色</h4><ul>{protected_html}</ul>'
                f'<h4>区域、分工、冷暖与内容/画法</h4>{observation_html}'
                f'<p class="muted">记录：{escape(str(row.get("record_name") or row["chunk"]))} · '
                f'证据来源：{escape(ROW_SOURCE_LABELS.get(row["evidence_source"], str(row["evidence_source"])))}'
                f'（{escape(str(row.get("evidence_ref")))}）· 内容判定：'
                f'{escape(CONTENT_VERDICT_LABELS.get(_content_failure(row)[0], _content_failure(row)[0]))}'
                + (f' · <b>重复对象存在冲突</b>：{escape(json.dumps(row["duplicate_conflicts"], ensure_ascii=False))}'
                   if row.get("duplicate_conflicts") else "")
                + f' · 原图 {escape(source["image"])} · '
                f'sha256 {escape(str(stats_row.get("image_sha256"))[:16])}</p></section>')

    group_blocks = []
    stability_text = {
        "stable_conform": "符合方案且三张一致（全部计划样本已覆盖）",
        "stable_partial": "三张判定一致，但结论是部分执行",
        "unstable": "三张判定不一致",
        "incomplete_coverage": "计划样本未全部覆盖，不能判稳定",
        "not_applicable": "本方案没有配色条款，稳定性不适用",
    }
    for group in summary["groups"]:
        subset = [row for row in rows if row["suite"] == group["suite"] and row["group_id"] == group["group_id"]]
        counts = group["colour_status_counts"]
        group_blocks.append(
            f'<section class="group"><h2>{escape(group["suite"])} / {escape(group["group_id"])} · '
            f'{escape(group["label"])}</h2>'
            f'<p><b>覆盖 {group["reviewed"]}/{group["planned"]}</b>'
            + (f' · 未评价：{escape("、".join(group["missing"]))}' if group["missing"] else "")
            + f' ｜ 配色判定：符合 {counts.get("conform", 0)} / 部分符合 {counts.get("partial", 0)} / '
            f'不符合 {counts.get("diverge", 0)} / 无法判断 {counts.get("unclear", 0)}；'
            f'条款 {group["clause_total"]} 条（符合 {group["clause_conform"]}、部分 {group["clause_partial"]}、'
            f'不符合 {group["clause_diverge"]}、无法判断 {group["clause_unclear"]}）</p>'
            f'<p>判定一致性：{"三张判定相同" if group["repeat_consistent"] else "三张判定不一致"}'
            f'（{escape(" / ".join(group["repeat_pattern"]))}）'
            f' ｜ 方案符合："{"是" if group["scheme_applied"] else "否"}"'
            f' ｜ 方案稳定性：<b>{escape(stability_text.get(group["scheme_stability"], group["scheme_stability"]))}</b></p>'
            f'<p>受保护颜色：保留 {group["protected_counts"].get("retained", 0)} / '
            f'改变 {group["protected_counts"].get("altered", 0)} / '
            f'无法判断 {group["protected_counts"].get("unclear", 0)} ｜ 内容：无明确问题 '
            f'{group["content_verdicts"]["pass_or_unverifiable"]} / 双人失败 '
            f'{group["content_verdicts"]["two_person_failure"]} / 其他失败 '
            f'{group["content_verdicts"]["content_failure"]}（配色与内容互不抵消）</p>'
            f'<p class="muted">内容问题：{escape("；".join(group["content_issues"]) or "—")}</p>'
            f'<p class="muted">画法问题：{escape("；".join(group["method_issues"]) or "—")}</p>'
            + "".join(card(row) for row in subset) + "</section>")
    css = ("body{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#25322b;background:#eef1ee;margin:0}"
           "main{max-width:1180px;margin:auto;padding:24px}"
           "section.group{background:#fff;border-radius:14px;padding:22px;margin:20px 0}"
           "section.card{border:1px solid #e0e6e1;border-radius:10px;padding:16px;margin:14px 0;background:#fbfdfb}"
           ".row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1 1 380px;min-width:320px}"
           "figure{margin:0;flex:0 0 auto}figure img{max-width:520px;width:100%;border-radius:8px;border:1px solid #d8ded9}"
           ".tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}"
           ".tag.ok{background:#2f8f5b}.tag.warn{background:#c98a17}.tag.bad{background:#b23b45}.tag.unk{background:#6b7785}"
           "pre{white-space:pre-wrap;word-break:break-word;background:#f2f5f2;padding:10px;border-radius:8px;font-size:12.5px}"
           ".muted{color:#657269;font-size:13px}ul{padding-left:20px}details{margin:6px 0}"
           "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #dde3de;padding:6px;text-align:left}")
    header = (f'<h1>配色方案符合度重评 · {len(rows)}/{summary.get("planned_items", len(items))} 张逐项评价</h1>'
              f'<p class="muted">协议 {escape(REASSESS_LABEL)} · 评价模型 '
              f'{escape(str(protocol.get("reviewer_model")))} · 协议 hash '
              f'{escape(str(protocol.get("protocol_hash"))[:16])} · 生成 {escape(str(summary.get("at")))}</p>'
              '<aside><b>判读口径</b><p>配色方案的目标是让画面呈现指定的一套色彩，'
              '不是改善画质、也不要求比基线更好看。因此这里只有“是否执行/执行到什么程度”，'
              '没有美学排名。</p>'
              '<p><b>覆盖率单列</b>：未评价的图片既不通过也不失败，`coverage.missing_items` 列出缺口；'
              '方案稳定性要求该方案计划样本全部被覆盖，“三张判定一致”与“符合方案”是两件事。</p>'
              '<p>G0/G1/T0 没有配色条款，稳定性为<b>不适用</b>，不是通过；G5/G6 只调明暗或彩度焦点，'
              '不按其他组的色相方案验收。</p>'
              '<p>内容与配色分别汇总、互不抵消：明确的双人失败单列，裁剪分辨率看不清的部分不算通过。</p>'
              '<p>本地颜色统计是全局描述量，没有区域蒙版，不用于判定区域颜色面积；'
              '同组三张图的画法一致性只能说明该组内部一致，不能证明跨方案画法未改变。</p></aside>'
              f'<p><a href="reassessment-summary.json">reassessment-summary.json</a> · '
              f'<a href="per-image-evaluations.json">per-image-evaluations.json</a> · '
              f'<a href="reassessment-protocol.json">reassessment-protocol.json</a> · '
              f'<a href="reassessment-protocol.md">reassessment-protocol.md</a> · '
              f'<a href="preflight.json">preflight.json</a> · '
              f'<a href="call-ledger.json">call-ledger.json</a> · '
              f'<a href="recovery.json">recovery.json</a> · '
              f'<a href="recovery-log.jsonl">recovery-log.jsonl</a> · '
              f'<a href="reassess-failures.jsonl">reassess-failures.jsonl</a></p>')
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>配色方案符合度重评 2026-10-04</title>', f'<style>{css}</style><main>',
             header, "".join(group_blocks), '</main></html>']
    path = out / "conformance-review.html"
    path.write_text("".join(parts), encoding="utf-8")
    log(f"[reassess] 逐张评价页 → {path}")
    return path
