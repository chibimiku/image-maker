"""Versioned measurements on explicit face/hair annotations; never guesses a crop."""
import hashlib
import json
import math
from pathlib import Path
import statistics

VERSION = "face-hair-measurements/1"
LABELS = {
    "hair_fineness": "发丝细腻程度 ↑", "hair_continuity": "仅发丝连贯性 ↑",
    "eye_brightness": "眼睛亮度 ↑", "eye_height": "上下眼睑间距 ↑",
    "eyelashes": "睫毛画法统计 ↑", "eye_width": "眼睛宽度 ↑",
    "eye_gap": "双眼间距 ↑", "eye_curvature": "上下眼睑弧度 ↑",
    "iris_ratio": "虹膜占比 ↑", "eye_highlights": "眼部高光分布 ↑",
    "lid_weight": "上下眼线粗细 ↑", "eye_tilt": "眼角倾斜 ↑",
    "face_ratios": "眼鼻口比例 ↑",
}
METRICS = tuple(LABELS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar(path):
    return Path(__file__).resolve().parents[1] / "cache/style-regions" / (digest(path) + ".json")


def annotation_snapshot(paths):
    result = {}
    for path in paths:
        filename = sidecar(path)
        result[path] = {"path": str(filename), "sha256": digest(filename) if filename.is_file() else None}
    return result


def read_annotation(path):
    filename = sidecar(path)
    if not filename.is_file():
        return None
    value = json.loads(filename.read_text(encoding="utf-8"))
    if value.get("image_sha256") != digest(path):
        raise ValueError("区域标注的图片 hash 不匹配")
    if value.get("version") != "style-regions/1":
        raise ValueError("不支持的区域标注版本")
    return value


def unavailable(reason):
    return {m: {"status": "unavailable", "value": None, "error": reason} for m in METRICS}


def points(value, minimum=2):
    import numpy as np
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < minimum or not np.isfinite(array).all():
        raise ValueError("坐标需为至少 %d 个 [x,y] 点" % minimum)
    if (array < 0).any() or (array > 1).any():
        raise ValueError("坐标必须在 EXIF 转正图的 0–1 范围")
    return array


def _mask(polygons, shape, dimensions):
    import cv2
    import numpy as np
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        vertices = points(polygon, 3) * dimensions
        cv2.fillPoly(mask, [np.round(vertices).astype(np.int32)], 1)
    return mask.astype(bool)


def _trace(gray, polyline, dimensions, unit, polarity="dark", allowed_mask=None):
    """Sample the marked stroke only, perpendicular to its local tangent."""
    import cv2
    import numpy as np
    vertices = points(polyline) * dimensions
    samples, tangents = [], []
    for a, b in zip(vertices[:-1], vertices[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        count = max(2, math.ceil(length / max(1, unit / 384)))
        tangent = (b - a) / length
        for fraction in np.linspace(0, 1, count, endpoint=False):
            samples.append(a + fraction * (b - a))
            tangents.append(tangent)
    if len(samples) < 3:
        raise ValueError("笔画路径太短")
    samples, tangents = np.asarray(samples), np.asarray(tangents)
    radius = max(2, unit * .025)
    offsets = np.linspace(-radius, radius, 21)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    positions = samples[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    profile = cv2.remap(gray.astype(np.float32), positions[:, :, 0].astype(np.float32), positions[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if allowed_mask is not None:
        sampled_mask = cv2.remap(allowed_mask.astype(np.uint8), positions[:, :, 0].astype(np.float32), positions[:, :, 1].astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if not sampled_mask.all():
            raise ValueError("发丝采样带超出头发内部区域；需校正路径或区域，未采入背景/脸部边缘")
    center = profile[:, 10]
    background = np.percentile(profile, 20 if polarity == "light" else 80, axis=1)
    contrast = center - background if polarity == "light" else background - center
    supported = contrast >= 12
    width = []
    width_profile = []
    for row, base, strength, valid in zip(profile, background, contrast, supported):
        if not valid:
            width_profile.append(None)
            continue
        threshold = base + strength / 2 if polarity == "light" else base - strength / 2
        band = row >= threshold if polarity == "light" else row <= threshold
        lo = hi = 10
        while lo > 0 and band[lo - 1]:
            lo -= 1
        while hi < 20 and band[hi + 1]:
            hi += 1
        measured_width = (hi - lo + 1) * (2 * radius / 20) / unit
        width.append(measured_width)
        width_profile.append(measured_width)
    runs, current = [], 0
    for valid in supported:
        if valid:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    end_count = max(1, len(width_profile) // 5)
    root_widths = [v for v in width_profile[:end_count] if v is not None]
    tip_widths = [v for v in width_profile[-end_count:] if v is not None]
    taper = statistics.mean(tip_widths) / statistics.mean(root_widths) if root_widths and tip_widths else None
    return {"tip_to_root_width": taper, "coverage": float(supported.mean()), "longest_run": max(runs, default=0) / len(supported),
            "gap_fraction": float((~supported).mean()), "gap_transitions": float(np.count_nonzero(supported[1:] != supported[:-1])) / len(supported),
            "width_median": float(np.median(width)) if width else None,
            "width_p10": float(np.percentile(width, 10)) if width else None,
            "width_p90": float(np.percentile(width, 90)) if width else None,
            "width_variation": float(np.std(width) / (np.mean(width) + 1e-9)) if width else None,
            "samples": samples, "length": float(np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()) / unit}


def descriptors(path, annotation):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    if annotation is None:
        return unavailable("需要面部/头发区域定位；未使用上中部构图区兜底")
    result = unavailable("未标注所需区域或路径")
    with Image.open(path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    height, width = image.shape[:2]
    dimensions = np.array([width - 1, height - 1], dtype=float)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face = points(annotation.get("face_outline"), 3) * dimensions
    eyes = annotation.get("eyes") or {}
    centers = []
    for eye in eyes.values():
        if eye.get("upper_lid"):
            centers.append((points(eye["upper_lid"]) * dimensions).mean(axis=0))
    axis = centers[1] - centers[0] if len(centers) == 2 else np.array([1., 0.])
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    if axis[0] < 0:
        axis = -axis
    vertical = np.array([-axis[1], axis[0]])
    face_width = float(np.ptp(face @ axis))
    if face_width < 64:
        return unavailable("原图脸宽不足64像素，不能可靠评价细笔画")
    def put(metric, values, extra=None):
        if values and all(math.isfinite(float(v)) for v in values.values()):
            result[metric] = {"status": "ok", "value": None, "components": {k: float(v) for k, v in values.items()}, "detail": extra or {}}
    # Only annotated strands inside annotated hair regions participate.
    hair = annotation.get("hair_regions") or []
    strands = annotation.get("hair_strands") or []
    if hair and strands:
        hair_mask = _mask(hair, gray.shape, dimensions)
        traces = []
        for strand in strands:
            try:
                trace = _trace(gray, strand["points"], dimensions, face_width, strand.get("polarity", "dark"), allowed_mask=hair_mask)
                locations = np.round(trace.pop("samples")).astype(int)
                if not hair_mask[locations[:, 1], locations[:, 0]].all():
                    raise ValueError("发丝路径超出确认的头发区域")
                traces.append(trace)
            except (ValueError, KeyError) as exc:
                result["hair_fineness"] = result["hair_continuity"] = {"status": "unavailable", "value": None, "error": str(exc)}
                traces = []
                break
        if traces:
            put("hair_continuity", {k: statistics.mean(t[k] for t in traces) for k in ("coverage", "longest_run", "gap_fraction", "gap_transitions")}, {"marked_strand_count": len(traces), "traces": traces})
            readable = [t for t in traces if t["width_median"] is not None]
            if len(readable) >= 3:
                put("hair_fineness", {k: statistics.mean(t[k] for t in readable) for k in ("width_median", "width_p10", "width_p90", "width_variation")}, {"readable_strands": len(readable), "note": "仅笔画宽度与变化，非人类审美评分"})
    eye_data = {}
    for name in ("viewer_left", "viewer_right"):
        eye = eyes.get(name) or {}
        if eye.get("state") != "open" or not eye.get("upper_lid") or not eye.get("lower_lid"):
            continue
        try:
            upper, lower = points(eye["upper_lid"]) * dimensions, points(eye["lower_lid"]) * dimensions
            upper = upper[np.argsort(upper @ axis)]
            lower = lower[np.argsort(lower @ axis)]
            chord = upper[-1] - upper[0]
            eye_width = float(np.linalg.norm(chord))
            if eye_width < 16:
                continue
            if max(np.linalg.norm(upper[0] - lower[0]), np.linalg.norm(upper[-1] - lower[-1])) > eye_width * .12:
                raise ValueError("上下眼睑的眼角端点不一致")
            unit_axis = chord / eye_width
            unit_vertical = np.array([-unit_axis[1], unit_axis[0]])
            if unit_vertical @ vertical < 0:
                unit_vertical = -unit_vertical
            grid = np.linspace(0, eye_width, 41)
            def curve(vertices):
                xy = vertices - upper[0]
                x, y = xy @ unit_axis, xy @ unit_vertical
                if (np.diff(x) < -1e-6).any():
                    raise ValueError("眼睑曲线在眼角坐标系不能回折")
                return np.interp(grid, x, y) / eye_width
            up, down = curve(upper), curve(lower)
            if ((down - up)[1:-1] <= 0).any():
                raise ValueError("上下眼睑曲线交叉或未张开")
            mask = _mask([np.vstack((upper / dimensions, lower[::-1] / dimensions)).tolist()], gray.shape, dimensions)
            values = gray[mask]
            if not len(values):
                continue
            eye_data[name] = {"width": eye_width, "upper": upper, "lower": lower, "mask": mask,
                              "height": float(np.mean(down - up)), "max_height": float(np.max(down - up)),
                              "upper_arc": float(-np.min(up)), "lower_arc": float(np.max(down)),
                              "upper_bend": float(np.mean(np.abs(np.diff(up, n=2))) * 40 ** 2),
                              "lower_bend": float(np.mean(np.abs(np.diff(down, n=2))) * 40 ** 2),
                              "brightness": float(values.mean() / 255), "brightness_p90": float(np.percentile(values, 90) / 255),
                              "highlight_ratio": float((values >= 240).mean()),
                              "tilt": float(math.atan2(chord @ vertical, chord @ axis))}
        except (ValueError, TypeError) as exc:
            for metric in ("eye_height", "eye_curvature", "eye_width", "eye_gap"):
                result[metric] = {"status": "unavailable", "value": None, "error": str(exc)}
    if len(eye_data) == 2:
        put("eye_width", {n: e["width"] / face_width for n, e in eye_data.items()})
        put("eye_height", {n + "_" + k: e[k] for n, e in eye_data.items() for k in ("height", "max_height")})
        put("eye_brightness", {n + "_" + k: e[k] for n, e in eye_data.items() for k in ("brightness", "brightness_p90")})
        put("eye_curvature", {n + "_" + k: e[k] for n, e in eye_data.items() for k in ("upper_arc", "lower_arc", "upper_bend", "lower_bend")})
        put("eye_tilt", {n: e["tilt"] for n, e in eye_data.items()})
        left, right = eye_data["viewer_left"], eye_data["viewer_right"]
        gap = float((right["upper"][0] - left["upper"][-1]) @ axis)
        if gap >= 0:
            put("eye_gap", {"gap_over_face": gap / face_width, "gap_over_eye": gap / statistics.mean((left["width"], right["width"]))})
        iris_values, highlight_values, lash_values, lid_values = {}, {}, {}, {}
        for name, data in eye_data.items():
            eye = eyes[name]
            if eye.get("iris"):
                iris_mask = _mask([eye["iris"]], gray.shape, dimensions) & data["mask"]
                iris_values[name] = float(iris_mask.sum() / data["mask"].sum())
                if iris_mask.sum():
                    highlights = ((gray >= 240) & iris_mask).astype(np.uint8)
                    count, labels, stats, centroids = cv2.connectedComponentsWithStats(highlights, connectivity=8)
                    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
                    highlight_values[name + "_area_ratio"] = float(highlights.sum() / iris_mask.sum())
                    highlight_values[name + "_count"] = count - 1
                    highlight_values[name + "_largest_ratio"] = float(max(areas, default=0) / iris_mask.sum())
            lashes = eye.get("lashes") or []
            if lashes:
                traces = [_trace(gray, lash, dimensions, data["width"]) for lash in lashes]
                readable = [t for t in traces if t["width_median"] is not None]
                if readable:
                    lash_values.update({name + "_" + k: statistics.mean(t[k] for t in readable) for k in ("width_median", "width_variation", "length")})
                    lash_values[name + "_count"] = len(lashes)
                    # Absolute orientation relative to the eye chord; rotation normalized.
                    chord = data["upper"][-1] - data["upper"][0]
                    chord /= np.linalg.norm(chord)
                    angles = []
                    for lash in lashes:
                        vertices = points(lash) * dimensions
                        vector = vertices[-1] - vertices[0]
                        vector /= max(float(np.linalg.norm(vector)), 1e-9)
                        angles.append(float(abs(chord[0] * vector[1] - chord[1] * vector[0])))
                    lash_values[name + "_direction_verticality"] = statistics.mean(angles)
                    tapers = [t["tip_to_root_width"] for t in readable if t["tip_to_root_width"] is not None]
                    if len(tapers) == len(readable):
                        lash_values[name + "_tip_to_root"] = statistics.mean(tapers)
            for key in ("upper_lid", "lower_lid"):
                trace = _trace(gray, eye[key], dimensions, data["width"])
                if trace["width_median"] is not None:
                    lid_values[name + "_" + key] = trace["width_median"]
        if len(iris_values) == 2:
            put("iris_ratio", iris_values)
        if len(highlight_values) == 6:
            put("eye_highlights", highlight_values, {"note": "只测虹膜内亮度≥240的候选高光，排除眼白；亮度阈值固定"})
        if all(any(k.startswith(n) for k in lash_values) for n in eye_data):
            put("eyelashes", lash_values, {"note": "标注睫毛的数量/宽度/长度/宽度变化；不能概括所有睫毛美术语言"})
        if len(lid_values) == 4:
            put("lid_weight", lid_values)
        if annotation.get("nose") and annotation.get("mouth"):
            eye_center = np.mean([e["upper"].mean(axis=0) for e in eye_data.values()], axis=0)
            nose = points([annotation["nose"], annotation["mouth"]]) * dimensions
            put("face_ratios", {"eyes_to_nose": float((nose[0] - eye_center) @ vertical / face_width),
                                "nose_to_mouth": float((nose[1] - nose[0]) @ vertical / face_width)})
    return result


def compare_face_features(first, second, annotations, cache=None):
    cache = cache if cache is not None else {}
    try:
        for path in (first, second):
            if path not in cache:
                cache[path] = descriptors(path, annotations.get(path))
        a, b = cache[first], cache[second]
        confirmed = all((annotations.get(path) or {}).get("confirmed") is True for path in (first, second))
        result = {}
        for metric in METRICS:
            one, two = a[metric], b[metric]
            if one["status"] != "ok" or two["status"] != "ok":
                result[metric] = {"status": "unavailable", "value": None, "detail": {"first": one, "reference": two}}
                continue
            if metric not in ("hair_fineness", "hair_continuity") and (annotations[first].get("pose") != annotations[second].get("pose") or annotations[first].get("pose") not in ("frontal", "three_quarter")):
                result[metric] = {"status": "not_comparable", "value": None, "error": "脸部视角不一致或未知"}
                continue
            if set(one["components"]) != set(two["components"]):
                result[metric] = {"status": "not_comparable", "value": None, "error": "两图有效分量不一致；不按剩余分量重新加权", "detail": {"first": one, "reference": two}}
                continue
            scores = {}
            for key, value in one["components"].items():
                other = two["components"][key]
                denominator = abs(value) + abs(other)
                scores[key] = 1.0 if denominator == 0 else 1 - abs(value - other) / denominator
            result[metric] = {"status": "ok" if confirmed else "provisional", "value": statistics.mean(scores.values()),
                              "detail": {"first": one, "reference": two, "components": scores, "formula_version": VERSION}}
        return result
    except Exception as exc:
        return unavailable(f"区域标注无效：{type(exc).__name__}: {exc}")
