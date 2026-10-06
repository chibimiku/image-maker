"""Optional vision proposals and hash-bound, user-reviewable region files."""
import hashlib
import json
from pathlib import Path
from utils.style_face_metrics import digest, sidecar, points
from utils.style_similarity import atomic_json
from utils.prompt_loader import read_prompt_file


def validate_annotation(value, image_path):
    if value.get("version") != "style-regions/1" or value.get("image_sha256") != digest(image_path):
        raise ValueError("区域文件版本或图片 hash 不匹配")
    if value.get("pose") not in ("frontal", "three_quarter", "profile", "unknown"):
        raise ValueError("需明确面部视角")
    if value.get("face_outline"):
        points(value["face_outline"], 3)
    for polygon in value.get("hair_regions", []):
        points(polygon, 3)
    for strand in value.get("hair_strands", []):
        points(strand["points"])
        if strand.get("polarity", "dark") not in ("dark", "light"):
            raise ValueError("发丝 polarity 必须为 dark/light")
    for eye in value.get("eyes", {}).values():
        if eye.get("state") not in ("open", "closed", "occluded", "unknown"):
            raise ValueError("需明确眼睛张合/遮挡状态")
        for key in ("upper_lid", "lower_lid"):
            if eye.get(key):
                points(eye[key])
        if eye.get("iris"):
            points(eye["iris"], 3)
        for lash in eye.get("lashes", []):
            points(lash)
    for key in ("nose", "mouth"):
        if value.get(key):
            points([value[key], value[key]])
    if value.get("confirmed") and not value.get("face_outline"):
        raise ValueError("确认前需要定位目标面部轮廓")
    return value


def save_annotation(path, value):
    validate_annotation(value, path)
    target = sidecar(path)
    if target.is_file():
        previous = target.read_bytes()
        archive = target.parent / "history" / (target.stem + "-" + hashlib.sha256(previous).hexdigest() + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(previous)
    atomic_json(target, value)
    return str(target)


def propose_regions(path, config, timeout=600, force=False):
    from openai import OpenAI
    from PIL import Image, ImageOps
    from utils.image_encoding import compress_and_encode_image
    prompt = read_prompt_file("style-extraction/face-regions-v1.md")
    base_url, api_key, model = config
    if not base_url or not api_key or not model:
        raise ValueError("需要配置可看图的文本分析端点、模型和认证")
    signature = hashlib.sha256((digest(path) + hashlib.sha256(base_url.encode()).hexdigest() + model + prompt).encode()).hexdigest()
    target = sidecar(path)
    if target.is_file() and not force:
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("confirmed") or cached.get("proposal_hash") == signature:
            return validate_annotation(cached, path)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        mime, encoded = compress_and_encode_image(image, max_dim=2048, quality=95)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(model=model, max_completion_tokens=7000,
        response_format={"type": "json_object"}, messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"}}]}])
    raw = response.choices[0].message.content
    value = json.loads(raw or "{}")
    if value.get("ambiguous") or value.get("unavailable") or not value.get("face_outline"):
        raise ValueError("需要人工指定目标面部：" + str(value.get("reason", "未获得可靠面部定位")))
    value.update(version="style-regions/1", image_sha256=digest(path), confirmed=False,
                 proposal_hash=signature, source="vision-proposal", model=model)
    validate_annotation(value, path)
    save_annotation(path, value)
    return value
