"""E3 受控局部干预用的**参数化可控图样**（确定性绘制 + 解析坐标）。

协议 §7 明确：真实图难控制时使用可控绘制图样作为机制测试，并声明它不代表真实图效度；
并且「先取得正确的区域，而不是使用已知错误自动提案」。

因此这里不调用任何生成模型：脸型、眼睑曲线、虹膜、睫毛、头发区域与发丝路径都由同一组
几何参数**解析**产生，标注坐标与图像像素同源，干预只改动协议指定的那一个因素。

每个干预档位的所有参数都写进随图产出的 `*.params.json`，供人工核查实际达到的量。
其余像素由确定性绘制保证逐字节一致（除被指定的区域外不做任何修改）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

VERSION = "style-experiment-synthetic-face/1"
SIZE = 1024
SUPERSAMPLE = 4  # 4× 超采样后降采样，得到确定的抗锯齿
EYE_LID_SAMPLES = 64   # 生成曲线的采样点数（最终按 x 排序后重采样为 41 点）
LASH_SAMPLES = 12

DEFAULT_COLORS = {
    "background_top": (232, 238, 246),
    "background_bottom": (208, 218, 232),
    "face": (247, 226, 209),
    "face_line": (196, 150, 132),
    "hair": (206, 186, 158),
    "strand": (74, 58, 48),
    "sclera": (252, 252, 252),
    "lid_line": (52, 44, 62),
    "iris": (200, 200, 200),
    "iris_line": (140, 140, 140),
    "pupil": (70, 70, 70),
    "highlight": (255, 255, 255),
    "lash": (38, 32, 46),
    "brow": (120, 96, 84),
    "mouth": (176, 118, 118),
    "nose": (214, 182, 164),
}

#: 六个案例的几何参数（两主画风各三例：PU = puracotte-style-v2，SA = sakurapion-style）
CASES = [
    {"case_id": "E3-PU-01", "style_id": "puracotte", "cx": 512.0, "cy": 470.0, "face_w": 300.0, "face_h": 380.0,
     "eye_w": 108.0, "eye_gap": 46.0, "upper_arc": 0.36, "lower_arc": 0.20, "iris_fraction": 0.80,
     "lash_len": 26.0, "strand_x": (300.0, 310.0, 320.0)},
    {"case_id": "E3-PU-02", "style_id": "puracotte", "cx": 500.0, "cy": 452.0, "face_w": 286.0, "face_h": 366.0,
     "eye_w": 116.0, "eye_gap": 40.0, "upper_arc": 0.40, "lower_arc": 0.22, "iris_fraction": 0.84,
     "lash_len": 30.0, "strand_x": (286.0, 296.0, 306.0)},
    {"case_id": "E3-PU-03", "style_id": "puracotte", "cx": 522.0, "cy": 486.0, "face_w": 312.0, "face_h": 392.0,
     "eye_w": 100.0, "eye_gap": 52.0, "upper_arc": 0.33, "lower_arc": 0.18, "iris_fraction": 0.78,
     "lash_len": 24.0, "strand_x": (312.0, 322.0, 332.0)},
    {"case_id": "E3-SA-01", "style_id": "sakurapion", "cx": 508.0, "cy": 462.0, "face_w": 296.0, "face_h": 374.0,
     "eye_w": 104.0, "eye_gap": 48.0, "upper_arc": 0.44, "lower_arc": 0.26, "iris_fraction": 0.74,
     "lash_len": 32.0, "strand_x": (295.0, 305.0, 315.0)},
    {"case_id": "E3-SA-02", "style_id": "sakurapion", "cx": 516.0, "cy": 476.0, "face_w": 306.0, "face_h": 384.0,
     "eye_w": 112.0, "eye_gap": 44.0, "upper_arc": 0.38, "lower_arc": 0.24, "iris_fraction": 0.82,
     "lash_len": 28.0, "strand_x": (300.0, 310.0, 320.0)},
    {"case_id": "E3-SA-03", "style_id": "sakurapion", "cx": 504.0, "cy": 458.0, "face_w": 292.0, "face_h": 372.0,
     "eye_w": 100.0, "eye_gap": 50.0, "upper_arc": 0.42, "lower_arc": 0.22, "iris_fraction": 0.76,
     "lash_len": 34.0, "strand_x": (288.0, 298.0, 308.0)},
]

#: 协议 §7 的 8 种干预与四个等级（1.0 = 原图参数）
INTERVENTIONS = {
    "hair_fineness": {"target": "hair_fineness", "levels": [1.0, 1.15, 1.30, 1.50], "effect": "strand_width"},
    "hair_continuity": {"target": "hair_continuity", "levels": [1.0, 0.05, 0.10, 0.20], "effect": "strand_gap"},
    "eye_brightness": {"target": "eye_brightness", "levels": [1.0, 0.95, 0.90, 0.80], "effect": "iris_scale"},
    "eye_height": {"target": "eye_height", "levels": [1.0, 1.05, 1.10, 1.20], "effect": "height_scale"},
    "eye_width": {"target": "eye_width", "levels": [1.0, 1.05, 1.10, 1.20], "effect": "width_scale"},
    "eye_gap": {"target": "eye_gap", "levels": [1.0, 0.005, 0.01, 0.02], "effect": "gap_shift_face_fraction"},
    "eyelashes": {"target": "eyelashes", "levels": [1.0, 1.15, 1.30, 1.50], "effect": "lash_width"},
    "eye_curvature": {"target": "eye_curvature", "levels": [1.0, 1.10, 1.20, 1.40], "effect": "upper_arc_scale"},
}
LEVEL_NAMES = ("original", "weak", "medium", "strong")

#: 两个负对照：不改动任何被测区域
NEGATIVE_CONTROLS = {
    "NC1-hair": {"description": "头发区域外（画面右侧边缘）画一条黑线",
                 "operation": "line", "start": (960, 170), "end": (966, 880), "width": 9,
                 "color": (16, 16, 16), "expect": ["hair_fineness", "hair_continuity"]},
    "NC2-eye": {"description": "眼区外（左颊）加白点",
                "operation": "ellipse", "center": (300, 700), "axes": (22, 22),
                "color": (255, 255, 255), "expect": ["eye_brightness"]},
}


def params_for(case_id: str) -> dict:
    for case in CASES:
        if case["case_id"] == case_id:
            return dict(case)
    raise KeyError("未知案例：" + case_id)


def _eye_centers(case):
    dx = case["eye_w"] / 2.0 + case["eye_gap"] / 2.0
    return [(case["cx"] - dx, case["cy"]), (case["cx"] + dx, case["cy"])]


def _lid_points(cx, cy, width, upper_arc, lower_arc, samples=EYE_LID_SAMPLES):
    """解析眼睑曲线（上、下眼睑，外→内顺序由 x 升序保证）。"""
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    upper, lower = [], []
    for index in range(samples + 1):
        fraction = index / samples
        x = x0 + fraction * width
        depth = 1.0 - (2.0 * x - x0 - x1) ** 2 / width ** 2
        depth = max(depth, 0.0)
        upper.append((x, cy - upper_arc * width * depth))
        lower.append((x, cy + lower_arc * width * depth))
    return upper, lower


def _lash_curve(root, tip, samples=LASH_SAMPLES):
    points = []
    for index in range(samples + 1):
        fraction = index / samples
        x = root[0] + fraction * (tip[0] - root[0])
        bow = 0.22 * (tip[1] - root[1]) * math.sin(math.pi * fraction)
        y = root[1] + fraction * (tip[1] - root[1]) + bow
        points.append((x, y))
    return points


def _lash_root_above_lid(eye_cx, cy, width, upper_arc, fraction_along, stroke_half):
    """求眼睑上落在指定横向位置的点，再沿眼睑垂线向外偏移半个笔画宽。"""
    x = eye_cx - width / 2.0 + fraction_along * width
    depth = max(1.0 - (2.0 * x - (eye_cx - width / 2.0) - (eye_cx + width / 2.0)) ** 2 / width ** 2, 0.0)
    y = cy - upper_arc * width * depth
    slope = -upper_arc * width * (2.0 * (x - eye_cx)) * 2.0 / width ** 2 * width
    tangent = (1.0, slope)
    norm = math.hypot(*tangent)
    normal = (-tangent[1] / norm, tangent[0] / norm)
    if normal[1] > 0:
        normal = (-normal[0], -normal[1])
    return (x + normal[0] * stroke_half, y + normal[1] * stroke_half)


def _strand_paths(case):
    """三条细发丝路径（竖直，落在左侧发束区域内部，并留出足够采样带余量）。"""
    paths = []
    base_y = case["cy"] - case["face_h"] * 0.30
    for index, x in enumerate(case["strand_x"]):
        top = (x, base_y + index * 18.0)
        bottom = (x, base_y + 250.0 + index * 26.0)
        paths.append([top, bottom])
    return paths


def _strand_widths(case):
    return [4.0, 4.6, 5.2]


#: 发丝采样带的横向半宽（相对脸宽）与安全余量：`utils/style_face_metrics` 固定为 2.5%
STRAND_BAND_HALF = 0.025
#: 发丝描边最粗时的半宽（像素），粗化干预上限 ×1.5
STRAND_MAX_HALF_PX = 5.2 * 1.5 / 2.0
#: 区域横向余量（像素）
STRAND_MARGIN_PX = 5.0


def _hair_region(case):
    """左侧发束区域：由三条发丝路径的坐标**解析**推出，保证采样带完全落在内部。"""
    band_half_px = case["face_w"] * STRAND_BAND_HALF
    offset = band_half_px + STRAND_MAX_HALF_PX + STRAND_MARGIN_PX
    xs = [point[0] for path in _strand_paths(case) for point in path]
    ys = [point[1] for path in _strand_paths(case) for point in path]
    left_edge = min(xs) - offset
    right_edge = max(xs) + offset
    inner = case["cx"] - case["face_w"] / 2.0
    if right_edge > inner:
        raise ValueError("发束区域会压到面部，请调整 strand_x")
    top = min(ys) - offset
    bottom = max(ys) + offset
    return [[(left_edge, top), (right_edge, top), (right_edge, bottom), (left_edge, bottom)]]


def variant_parameters(case, intervention=None, level=1.0):
    """把一个干预档位翻译成绘制参数；未涉及的参数保持原值。"""
    result = {
        "strand_width_scale": 1.0,
        "strand_missing_fraction": 0.0,
        "iris_scale": 1.0,
        "height_scale": 1.0,
        "width_scale": 1.0,
        "gap_shift": 0.0,
        "lash_width_scale": 1.0,
        "upper_arc_scale": 1.0,
        "intervention": intervention,
        "level": level,
        "level_name": LEVEL_NAMES[INTERVENTIONS[intervention]["levels"].index(level)] if intervention else "original",
    }
    if not intervention:
        return result
    effect = INTERVENTIONS[intervention]["effect"]
    if effect == "strand_width":
        result["strand_width_scale"] = level
    elif effect == "strand_gap":
        result["strand_missing_fraction"] = level
    elif effect == "iris_scale":
        result["iris_scale"] = level
    elif effect == "height_scale":
        result["height_scale"] = level
    elif effect == "width_scale":
        result["width_scale"] = level
    elif effect == "gap_shift_face_fraction":
        result["gap_shift"] = level
    elif effect == "lash_width":
        result["lash_width_scale"] = level
    elif effect == "upper_arc_scale":
        result["upper_arc_scale"] = level
    else:
        raise KeyError("未知干预效果：" + effect)
    return result


def _draw(canvas, scale):
    import cv2
    import numpy as np
    return canvas, cv2, np


def render(case_id, parameters, colors=None, negative_control=None):
    """绘制一张图样并返回 (RGB 数组, 标注 JSON)。"""
    import cv2
    import numpy as np

    case = params_for(case_id)
    palette = {**DEFAULT_COLORS, **(colors or {})}
    size = SIZE * SUPERSAMPLE
    canvas = np.zeros((size, size, 3), np.uint8)
    gradient = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :, None]
    for channel in range(3):
        canvas[..., channel] = (palette["background_top"][channel] * (1 - gradient)
                                + palette["background_bottom"][channel] * gradient).astype(np.uint8)[..., 0]
    s = SUPERSAMPLE

    def poly(points, color, width=0, closed=True):
        pts = np.round(np.asarray(points, dtype=np.float64) * s).astype(np.int32).reshape(-1, 1, 2)
        if width > 0:
            cv2.polylines(canvas, [pts], closed, color, max(1, int(round(width * s))), cv2.LINE_AA)
        elif closed:
            cv2.fillPoly(canvas, [pts], color)
        else:
            cv2.polylines(canvas, [pts], closed, color, 1, cv2.LINE_AA)

    def circle(center, radius, color, filled=True, width=1):
        center = (int(round(center[0] * s)), int(round(center[1] * s)))
        cv2.circle(canvas, center, max(1, int(round(radius * s))), color, -1 if filled else max(1, int(round(width * s))), cv2.LINE_AA)

    # 1) 后发外轮廓 2) 左侧发束区域（都在面部之下，避免遮挡眼睛）3) 面部
    hair_outline = [(case["cx"] - case["face_w"] * 0.92, case["cy"] + case["face_h"] * 0.72),
                    (case["cx"] - case["face_w"] * 0.98, case["cy"] - case["face_h"] * 0.35),
                    (case["cx"] - case["face_w"] * 0.55, case["cy"] - case["face_h"] * 0.78),
                    (case["cx"], case["cy"] - case["face_h"] * 0.92),
                    (case["cx"] + case["face_w"] * 0.55, case["cy"] - case["face_h"] * 0.78),
                    (case["cx"] + case["face_w"] * 0.98, case["cy"] - case["face_h"] * 0.35),
                    (case["cx"] + case["face_w"] * 0.92, case["cy"] + case["face_h"] * 0.72),
                    (case["cx"] + case["face_w"] * 0.30, case["cy"] + case["face_h"] * 0.86),
                    (case["cx"] - case["face_w"] * 0.30, case["cy"] + case["face_h"] * 0.86)]
    poly(hair_outline, palette["hair"])

    for region in _hair_region(case):
        poly(region, palette["hair"])

    face_outline = []
    for index in range(73):
        angle = -math.pi / 2 + index * (2 * math.pi / 72)
        face_outline.append((case["cx"] + math.cos(angle) * case["face_w"] / 2.0,
                             case["cy"] + math.sin(angle) * case["face_h"] / 2.0))
    poly(face_outline, palette["face"])
    poly(face_outline, palette["face_line"], width=1.6, closed=True)

    # 发丝（可选缺失段），标注始终保留完整路径
    strands_annotation = []
    for path, base_width in zip(_strand_paths(case), _strand_widths(case)):
        width = base_width * parameters["strand_width_scale"]
        missing = parameters["strand_missing_fraction"]
        # 绘制用的可见段：整条路径，或按可见长度缺口切出的若干段（确定性）
        segments = []
        if missing <= 0:
            segments = [(start, end) for start, end in zip(path[:-1], path[1:])]
        else:
            cut = missing / 2.0
            for start, end in zip(path[:-1], path[1:]):
                middle_low = (start[0] + (end[0] - start[0]) * (0.5 - cut),
                              start[1] + (end[1] - start[1]) * (0.5 - cut))
                middle_high = (start[0] + (end[0] - start[0]) * (0.5 + cut),
                               start[1] + (end[1] - start[1]) * (0.5 + cut))
                segments.extend([(start, middle_low), (middle_high, end)])
        for start, end in segments:
            poly([start, end], palette["strand"], width=width, closed=False)
        strands_annotation.append({"points": [list(point) for point in path], "polarity": "dark",
                                   "drawn_width_px": round(width, 4),
                                   "missing_fraction": missing})

    # 眼睛
    eyes_annotation = {}
    centers = []
    for index, (eye_cx, eye_cy) in enumerate(_eye_centers(case)):
        width = case["eye_w"] * parameters["width_scale"]
        if parameters["gap_shift"]:
            direction = -1.0 if index == 0 else 1.0
            eye_cx += direction * parameters["gap_shift"] * case["face_w"]
        height = parameters["height_scale"]
        upper_arc = case["upper_arc"] * parameters["upper_arc_scale"] * height
        lower_arc = case["lower_arc"] * height
        centers.append((eye_cx, eye_cy))
        upper, lower = _lid_points(eye_cx, eye_cy, width, upper_arc, lower_arc)
        opening = [*upper, *reversed(lower)]
        poly(opening, palette["sclera"])
        iris_radius = width * 0.5 * case["iris_fraction"] * 0.5
        iris_polygon = []
        for step in range(49):
            angle = step * (2 * math.pi / 48)
            iris_polygon.append((eye_cx + math.cos(angle) * iris_radius, eye_cy + math.sin(angle) * iris_radius))
        poly(iris_polygon, tuple(int(round(channel * parameters["iris_scale"])) for channel in palette["iris"]))
        poly(iris_polygon, palette["iris_line"], width=1.4, closed=True)
        circle((eye_cx, eye_cy), iris_radius * 0.42, palette["pupil"])
        highlight_center = (eye_cx - iris_radius * 0.38, eye_cy - iris_radius * 0.42)
        circle(highlight_center, iris_radius * 0.20, palette["highlight"])
        poly(upper, palette["lid_line"], width=3.0, closed=False)
        poly(lower, palette["lid_line"], width=2.0, closed=False)

        lash_width = 2.4 * parameters["lash_width_scale"]
        lashes = []
        for along in (0.26, 0.44, 0.62):
            root = _lash_root_above_lid(eye_cx, eye_cy, width, upper_arc, along, lash_width / 2.0 + 0.6)
            length = case["lash_len"] * (1.0 + 0.15 * (along - 0.44) / 0.18)
            tip = (root[0] - length * 0.45, root[1] - length)
            points = _lash_curve(root, tip)
            poly(points, palette["lash"], width=lash_width, closed=False)
            lashes.append([list(point) for point in points])
        eyes_annotation["viewer_left" if index == 0 else "viewer_right"] = {
            "state": "open",
            "upper_lid": [list(point) for point in upper],
            "lower_lid": [list(point) for point in lower],
            "iris": [list(point) for point in iris_polygon],
            "lashes": lashes,
        }

    # 眉毛
    for eye_cx, eye_cy in centers:
        brow_y = eye_cy - case["eye_w"] * 0.62
        poly([(eye_cx - case["eye_w"] * 0.5, brow_y + 6), (eye_cx, brow_y),
              (eye_cx + case["eye_w"] * 0.5, brow_y + 4)], palette["brow"], width=6.0, closed=False)

    nose = (case["cx"], case["cy"] + case["face_h"] * 0.16)
    mouth = (case["cx"], case["cy"] + case["face_h"] * 0.30)
    poly([(nose[0] - 6, nose[1]), (nose[0] + 6, nose[1])], palette["nose"], width=2.0, closed=False)
    poly([(mouth[0] - 18, mouth[1]), (mouth[0], mouth[1] + 7), (mouth[0] + 18, mouth[1])],
         palette["mouth"], width=4.0, closed=False)

    if negative_control:
        control = NEGATIVE_CONTROLS[negative_control]
        if control["operation"] == "line":
            poly([control["start"], control["end"]], control["color"], width=control["width"], closed=False)
        elif control["operation"] == "ellipse":
            circle(control["center"], control["axes"][0], control["color"])

    # 降采样到最终尺寸
    small = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    dimensions = (SIZE - 1.0, SIZE - 1.0)

    def normalize(points):
        return [[round(point[0] / dimensions[0], 8), round(point[1] / dimensions[1], 8)] for point in points]

    annotation = {
        "version": "style-regions/1",
        "image_sha256": "PENDING",
        "confirmed": False,
        "source": "synthetic-parametric-pattern",
        "pose": "frontal",
        "target_description": f"{case_id} 参数化图样：正面单脸，双眼可见",
        "face_outline": normalize(face_outline),
        "eyes": {name: {"state": value["state"],
                        "upper_lid": normalize(value["upper_lid"]),
                        "lower_lid": normalize(value["lower_lid"]),
                        "iris": normalize(value["iris"]),
                        "lashes": [normalize(lash) for lash in value["lashes"]]}
                 for name, value in eyes_annotation.items()},
        "hair_regions": [normalize(region) for region in _hair_region(case)],
        "hair_strands": [{"points": normalize(strand["points"]), "polarity": strand["polarity"]}
                         for strand in strands_annotation],
        "nose": normalize([nose])[0],
        "mouth": normalize([mouth])[0],
        "limitations": ["可控绘制图样用于机制测试；不代表真实插画的所有画法与效度",
                        "坐标由同一组几何参数解析产生，不是模型猜测"],
    }
    audit = {
        "case_id": case_id,
        "style_id": case["style_id"],
        "variant_parameters": parameters,
        "colors": palette,
        "negative_control": negative_control,
        "strands": [{"drawn_width_px": strand["drawn_width_px"],
                     "missing_fraction": strand["missing_fraction"]} for strand in strands_annotation],
        "case_geometry": case,
        "renderer_version": VERSION,
        "timestamp": None,
    }
    return rgb, annotation, audit


def write_variant(directory, name, rgb, annotation, audit):
    """写图 + 区域标注 + 参数审计（返回三者路径）。"""
    from PIL import Image
    import hashlib
    import time
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{name}.png"
    Image.fromarray(rgb).save(image_path, format="PNG", optimize=False, compress_level=6)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    annotation = dict(annotation)
    annotation["image_sha256"] = digest
    annotation_path = directory / f"{name}.annotation.json"
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = dict(audit)
    audit["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    audit["image_sha256"] = digest
    audit_path = directory / f"{name}.params.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, annotation_path, audit_path


def variant_matrix():
    """返回 [(case_id, intervention, level, level_name, parameters)]——含原图与全部档位。"""
    rows = []
    for case in CASES:
        rows.append((case["case_id"], None, 1.0, "original",
                     variant_parameters(case, None, 1.0)))
        for intervention, spec in INTERVENTIONS.items():
            for level in spec["levels"][1:]:
                rows.append((case["case_id"], intervention, level,
                             LEVEL_NAMES[spec["levels"].index(level)],
                             variant_parameters(case, intervention, level)))
    return rows
