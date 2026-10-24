"""Explicit, hash-bound style detail crops shared by image request assemblers."""
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from utils.prompt_loader import render_prompt_file


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detail_reference_paths(entry, reference, cache_dir=None):
    """Never guess a face location. Coordinates are relative to EXIF-oriented pixels.

    Disabled entries do no IO. Stale/invalid enabled regions fail before dispatch.
    The full reference remains separate; crops are examples, never output patches.
    """
    if not isinstance(entry, dict) or entry.get("reference_details_enabled") is not True:
        return []
    specs = entry.get("reference_detail_crops") or []
    if not isinstance(specs, list) or not 1 <= len(specs) <= 2:
        raise ValueError("局部画风参考需要1–2个已确认区域")
    import io
    source_bytes = Path(reference).read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    clean = []
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("source_sha256") != digest:
            raise ValueError("局部画风参考与原图哈希不符，请重新选区")
        role = spec.get("role")
        if role not in {"face_eyes", "hair", "material"}:
            raise ValueError("局部参考用途必须是 face_eyes / hair / material")
        box = spec.get("box")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       or not math.isfinite(v) for v in box)):
            raise ValueError("局部参考需要4个有限归一化坐标")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("局部参考区域必须位于原图内部")
        clean.append({"role": role, "box": list(box), "source_sha256": digest})
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / "cache" / "style-reference-details"
    with Image.open(io.BytesIO(source_bytes)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    results = []
    for spec in clean:
        w, h = image.size
        box = tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(spec["box"]))
        if box[2] - box[0] < 32 or box[3] - box[1] < 32:
            raise ValueError("局部画风参考过小（至少32×32原始像素）")
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        path = root / (key + ".png")
        root.mkdir(parents=True, exist_ok=True)
        # Rebuild deterministically: never trust an externally edited cached crop.
        import io
        buffer = io.BytesIO()
        image.crop(box).save(buffer, format="PNG")
        data = buffer.getvalue()
        if not path.exists() or path.read_bytes() != data:
            import os
            import tempfile
            fd, temporary = tempfile.mkstemp(dir=root, suffix=".png")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        results.append({"path": str(path), "role": spec["role"], "source_sha256": digest,
                        "sha256": hashlib.sha256(data).hexdigest(), "box": spec["box"]})
    return results


def reference_role_prompt(paths, content_count=0, primary_index=1, details=()):
    """Indices refer to the final transmitted order, not a UI-only order."""
    rows = []
    detail_by_path = {str(Path(d["path"]).resolve()): d["role"] for d in details}
    for index, path in enumerate(paths, 1):
        if index <= content_count:
            role = "content reference; use only for the content assigned by the user"
        elif index == primary_index:
            role = "full style reference; overall drawing and painting method"
        else:
            role = detail_by_path.get(str(Path(path).resolve()), "style detail")
        rows.append(f"- Image {index}: {role}.")
    return render_prompt_file("style-reference-roles.md", {"roles": "\n".join(rows)}).strip()


def style_face_quality_boost(styles, name, mode, api_type=""):
    """Reference-driven drawing follows the reference; text-only keeps old boost.

    Existing explicit profile overrides still take precedence.
    """
    from utils.styles import gemini_priority_profile_for, normalize_style_entry, ref_image_valid
    entry = normalize_style_entry((styles or {}).get(name))
    profile = gemini_priority_profile_for(styles, name, mode, api_type)
    if "face_quality_boost" in profile:
        return bool(profile["face_quality_boost"])
    if isinstance((styles or {}).get(name), dict) and "face_quality_boost" in styles[name]:
        return bool(styles[name]["face_quality_boost"])
    return mode == "off" or not ref_image_valid(entry["ref_image"])


def append_gpt_style_details(prompt, paths, entry, reference, content_count=0):
    details = detail_reference_paths(entry, reference) if reference else []
    if not details:
        return prompt, paths
    # Keep the original full style image last, as existing GPT authority text says.
    result = list(paths[:-1]) + [d["path"] for d in details] + [paths[-1]]
    if len(result) > 16:
        raise ValueError("内容图、局部参考及画风全图合计不能超过16张")
    if content_count:
        from utils.styles import build_content_style_image_roles
        prompt = prompt.replace(build_content_style_image_roles(content_count), "").strip()
    prompt += "\n\n" + reference_role_prompt(result, content_count, len(result), details)
    return prompt, result


def reference_input_manifest(paths, content_count=0, primary_index=None, encoding="gpt"):
    """Actual backend encoding hashes, distinct from the original file hashes."""
    from modules.others.api_backend import (transmitted_image_digest, _maybe_compress_image_path,
                                            _cleanup_temp_files, _guess_image_mime_type)
    if encoding not in {"gpt", "gemini"}:
        raise ValueError("Unknown reference encoding")
    primary_index = len(paths) if primary_index is None else primary_index
    rows = []
    for index, path in enumerate(paths, 1):
        if encoding == "gemini":
            row = transmitted_image_digest(path)
        else:
            temporary = []
            try:
                prepared = _maybe_compress_image_path(path, temp_files=temporary)
                data = Path(prepared).read_bytes()
                row = {"path": str(path), "mime": _guess_image_mime_type(prepared),
                       "file_sha256": file_hash(path),
                       "transmitted_sha256": hashlib.sha256(data).hexdigest(),
                       "transmitted_bytes": len(data), "reencoded": str(prepared) != str(path)}
            finally:
                _cleanup_temp_files(temporary)
        row["encoding"] = encoding
        row.update(index=index, role=("content" if index <= content_count else
                                     "style_full" if index == primary_index else "style_detail"))
        try:
            with Image.open(path) as source:
                row["oriented_size"] = list(ImageOps.exif_transpose(source).size)
        except OSError as exc:
            # A provenance sidecar must not hide an already saved output if an
            # older reference has since become unreadable.
            row["decode_error"] = str(exc)
        rows.append(row)
    return rows
