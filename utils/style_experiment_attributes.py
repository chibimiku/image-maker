"""画风受控实验用的**记录型**图像属性与近重复检测（只读，不产生分数）。

这里的数值**不是**画风指标，只用来完成协议要求的三件事：

1. 数据冻结时记录构图/亮暗/背景密度/配色分布，用于报告混杂与做分层选样；
2. 近重复检测（SHA-256 + dHash + 灰度平均绝对差），供人工核查后冻结 split；
3. E2 的 P/N 选择依据（只按这些冻结属性匹配，**不得读取任何指标值**）。

所有阈值与算法版本固定在本文件里，写进冻结包后不再改动。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

VERSION = "style-experiment-attributes/1"

#: 构图分类阈值（人脸可见面积占比）——先声明后使用
FRAMING_THRESHOLDS = {"face_closeup_ge": 0.05, "upper_body_ge": 0.015}
#: 亮暗分类阈值（中部区域灰度中位数）
BRIGHTNESS_THRESHOLD = 128.0
#: 背景密度分类阈值（四边环形区域与中心区域的边缘密度比）
BACKGROUND_DENSITY_THRESHOLD = 0.85
#: 近重复**候选**阈值（标记用；只有确定性规则才自动剔除，见 near_duplicate_decisions）
NEAR_DUPLICATE_HAMMING = 4
NEAR_DUPLICATE_MAD = 0.008
#: 确定性重复的自动剔除阈值：16×16 灰度归一化 MAD ≤ 该值（实测再编码副本 ≤ 0.011）
EXACT_DUPLICATE_MAD = 0.011
#: 灰度直方图桶数（近重复用的确定性指纹之一，JSON 安全）
GRAY_HISTOGRAM_BINS = 32
#: 分析用的归一化长边
ANALYSIS_LONG_SIDE = 512


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(gray) -> str:
    """64 位差值哈希（9x8 横向比较），返回 16 位十六进制字符串。"""
    import cv2
    import numpy as np
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def hamming(first: str, second: str) -> int:
    return bin(int(first, 16) ^ int(second, 16)).count("1")


def _gray_small(path, size=32):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = ANALYSIS_LONG_SIDE / max(width, height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                               Image.Resampling.BICUBIC)
        array = np.asarray(resized)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return array, gray, cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def gray_histogram(gray, bins=GRAY_HISTOGRAM_BINS):
    """确定性灰度直方图（归一化）+ 8x8 缩略灰度串（JSON 安全），用于近重复判定的第二个指纹。"""
    import cv2
    import numpy as np
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    counts, _ = np.histogram(gray, bins=bins, range=(0, 256))
    total = float(counts.sum()) or 1.0
    return (counts / total).round(6).tolist(), [int(round(value)) for value in small.flatten().tolist()]


def color_profile(rgb) -> dict:
    """主色与色相分布：只用冻结属性做配色匹配，不涉及任何画风指标。"""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV 0-179 → 0-358
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    weights = saturation * value
    histogram, _ = np.histogram(hue, bins=36, range=(0, 360), weights=weights)
    total = float(histogram.sum()) or 1.0
    share = (histogram / total).round(6).tolist()
    chromatic = weights > 0.15
    dominant_hue = None
    if chromatic.sum() > 0:
        dominant_hue = float(np.median(hue[chromatic]))
    return {
        "hue_share_36": share,
        "dominant_hue": dominant_hue,
        "mean_saturation": float(saturation.mean()),
        "mean_value": float(value.mean()),
        "chromatic_pixel_ratio": float(chromatic.mean()),
    }


def hue_distance(first, second) -> float:
    """两个主色相之间的环形距离（0–180），用于 E2 的「配色相近」分块。"""
    if first is None or second is None:
        return float("nan")
    diff = abs(float(first) - float(second)) % 360.0
    return min(diff, 360.0 - diff)


def subject_mask(gray):
    """确定性「背景 vs 主体」近似：边缘能量低阈值化 + 膨胀。

    只用固定卷积与形态学，不含学习模型；掩膜与掩膜 SHA-256 一并冻结，供人工核查。
    """
    import cv2
    import numpy as np
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    threshold = float(np.percentile(gradient, 75))
    mask = (gradient > threshold).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask, threshold


def measure(path) -> dict:
    """单张图的冻结属性（构图 / 亮暗 / 背景密度 / 配色 / 近重复指纹）。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    path = Path(path)
    with Image.open(path) as source:
        original_width, original_height = source.size
    rgb, gray, small = _gray_small(path)
    height, width = gray.shape[:2]          # 分析用（长边 512）尺寸
    histogram, thumbnail = gray_histogram(gray)

    face_ratio = 0.0
    face_box = None
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces):
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            face_ratio = float(w * h) / float(width * height)
            face_box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    except Exception as exc:  # 检测器缺失不阻塞冻结，但要如实记录
        face_box = {"error": f"{type(exc).__name__}: {exc}"}

    if face_ratio >= FRAMING_THRESHOLDS["face_closeup_ge"]:
        framing = "face_closeup"
    elif face_ratio >= FRAMING_THRESHOLDS["upper_body_ge"]:
        framing = "upper_body"
    else:
        framing = "full_body_or_scene"

    band = max(4, int(round(min(width, height) * 0.15)))
    central = gray[band:height - band, band:width - band]
    median_gray = float(np.median(central if central.size else gray))
    brightness = "bright" if median_gray >= BRIGHTNESS_THRESHOLD else "dark"

    mask, threshold = subject_mask(gray)
    total_edges = int((mask > 0).sum())
    border = np.zeros_like(mask)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    border_edges = int(((mask > 0) & (border > 0)).sum())
    background_ratio = border_edges / float(total_edges or 1)
    background_density = "busy" if background_ratio >= BACKGROUND_DENSITY_THRESHOLD else "clean"

    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "width": int(original_width),
        "height": int(original_height),
        "analysis_width": int(width),
        "analysis_height": int(height),
        "aspect": round(float(original_width) / float(original_height), 6),
        "face_area_ratio": round(face_ratio, 6),
        "face_box": face_box,
        "framing": framing,
        "median_gray_center": round(median_gray, 3),
        "brightness": brightness,
        "background_border_edge_ratio": round(background_ratio, 6),
        "background_density": background_density,
        "significant_edge_ratio": round(total_edges / float(gray.size), 6),
        "subject_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "subject_mask_edge_threshold": threshold,
        "dhash64": dhash64(gray),
        "gray_histogram_32": histogram,
        "gray_thumbnail_16x16": thumbnail,
        **color_profile(rgb),
        "attribute_version": VERSION,
        "thresholds": {
            "framing": FRAMING_THRESHOLDS,
            "brightness": BRIGHTNESS_THRESHOLD,
            "background_density": BACKGROUND_DENSITY_THRESHOLD,
            "analysis_long_side": ANALYSIS_LONG_SIDE,
        },
    }


def normalize_work_name(file_name: str) -> str:
    """把**已知的再导出命名规则**归一化，用于判定「同一作品」。

    只处理本机语料里可验证、且不会误伤不同作品的规则：

    - `prejpg-` 前缀（JPEG 预处理副本）；
    - 尾部 `_fixed` 或 `_fixed_<YYYY-MM-DD_HH-MM-SS>`（再导出标记）。

    **不**剥离裸时间戳后缀：kishida_mel / puracotte 语料里 `名字_2025-02-17_00-03-34.png`
    是不同作品，剥掉会把整批图误判成同一作品（真实踩过：240 张被剔到只剩 1 张）。
    分页后缀 `-1` / `-p0` 一律保留。
    """
    import re
    name = Path(file_name).stem.lower()
    name = re.sub(r"^prejpg-", "", name)
    name = re.sub(r"_fixed(_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})?$", "", name)
    return name


def near_duplicate_decisions(records) -> dict:
    """确定性重复判定：自动剔除「同一作品」；其余只标记待人工核查。

    自动剔除只用三条可复核规则：
    1. SHA-256 完全相同；
    2. 归一化文件名相同（再导出命名规则，见 `normalize_work_name`）；
    3. 16×16 灰度归一化 MAD ≤ `EXACT_DUPLICATE_MAD`（实测再编码集群 ≤ 0.011，
       而内容真实不同的最接近对为 0.0152，阈值落在两者之间）。
    """
    auto_excluded, review = [], []
    by_name = {}
    for row in records:
        by_name.setdefault(normalize_work_name(row.get("file_name") or Path(row["path"]).name), []).append(row)
    for name, group in sorted(by_name.items()):
        if len(group) < 2:
            continue
        keep = sorted(group, key=lambda row: row["path"].lower())[0]
        for row in group:
            if row["path"] == keep["path"]:
                continue
            auto_excluded.append({"path": row["path"], "kept": keep["path"], "rule": "normalized_filename",
                                  "normalized": name})
    for pair in near_duplicates(records, same_style_only=False):
        if pair["same_sha256"]:
            auto_excluded.append({"path": pair["second_path"], "kept": pair["first_path"], "rule": "identical_sha256"})
        elif pair["gray_mad"] <= EXACT_DUPLICATE_MAD:
            auto_excluded.append({"path": pair["second_path"], "kept": pair["first_path"],
                                  "rule": "mad_below_exact_threshold", "gray_mad": pair["gray_mad"]})
        else:
            review.append(dict(pair, review=None,
                               question="这两张是否为同一作品（裁剪/放大/改色/转载/再编码）？"))
    seen = set()
    unique = []
    for item in auto_excluded:
        if item["path"] in seen:
            continue
        seen.add(item["path"])
        unique.append(item)
    return {"auto_excluded": sorted(unique, key=lambda item: item["path"].lower()),
            "awaiting_human_review": review,
            "thresholds": {"exact_duplicate_mad": EXACT_DUPLICATE_MAD,
                           "candidate_hamming": NEAR_DUPLICATE_HAMMING,
                           "candidate_mad": NEAR_DUPLICATE_MAD}}


def near_duplicates(records, same_style_only: bool = True) -> list[dict]:
    """返回近重复**候选**对（必须人工核查；只有确定性规则才算重复）。

    模糊候选使用两个确定性指纹：dHash-64 汉明距离，以及 16×16 灰度缩略的平均绝对差
    （归一化到 0–1）。不含任何学习模型。
    """
    found = []
    items = [row for row in records if row.get("dhash64") and row.get("gray_thumbnail_16x16")]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if same_style_only and a.get("style_id") != b.get("style_id"):
                continue
            distance = hamming(a["dhash64"], b["dhash64"])
            thumbnail_a = a["gray_thumbnail_16x16"]
            thumbnail_b = b["gray_thumbnail_16x16"]
            mad = sum(abs(x - y) for x, y in zip(thumbnail_a, thumbnail_b)) / (len(thumbnail_a) * 255.0)
            if distance <= NEAR_DUPLICATE_HAMMING or mad <= NEAR_DUPLICATE_MAD or a["sha256"] == b["sha256"]:
                found.append({
                    "first": a.get("image_id") or a["path"],
                    "second": b.get("image_id") or b["path"],
                    "first_path": a["path"], "second_path": b["path"],
                    "first_split": a.get("split"), "second_split": b.get("split"),
                    "dhash_hamming": distance,
                    "gray_mad": round(mad, 6),
                    "same_sha256": a["sha256"] == b["sha256"],
                    "certainty": "exact" if a["sha256"] == b["sha256"] else "fuzzy",
                    "thresholds": {"hamming": NEAR_DUPLICATE_HAMMING, "mad": NEAR_DUPLICATE_MAD},
                    "review": None,
                })
    return found


def summary(records) -> dict:
    """按画风汇总混杂分布，供报告如实列出不平衡。"""
    import collections
    result = {}
    for record in records:
        style = record.get("style_id") or "unknown"
        bucket = result.setdefault(style, collections.Counter())
        bucket[record["framing"]] += 1
        bucket[record["brightness"]] += 1
        bucket[record["background_density"]] += 1
        bucket["split:" + str(record.get("split"))] += 1
    return {style: dict(counter) for style, counter in sorted(result.items())}


def required_megapixels(records) -> float:
    """实际参与比较的像素总量（报告里说明数据规模）。"""
    return round(sum(r["width"] * r["height"] for r in records) / 1e6, 2)


def luminance_of(path, hex_rgb):
    """辅助：把 #rrggbb 转灰度（用于可控图样的确定性配色）。"""
    value = hex_rgb.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return int(round(0.299 * r + 0.587 * g + 0.114 * b))


def bounded(value, low=0.0, high=255.0):
    return float(max(low, min(high, value)))


def safe_log(value: float) -> float:
    return math.log(max(value, 1e-12))


# --------------------------------------------------------------------------- #
# E2 确定性干扰（N0–N4）：算法、参数与产物 hash 全部冻结
# --------------------------------------------------------------------------- #

TRANSFORM_VERSION = "style-experiment-transforms/1"


def _read_rgb(path):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return np.asarray(image).copy()


def _write_png(array, path):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=False, compress_level=6)
    return path


def transform_n0(path, out_path):
    """N0：同尺寸无损 PNG 转存（编码基线）。"""
    array = _read_rgb(path)
    target = _write_png(array, out_path)
    return {"variant": "N0", "operation": "lossless PNG re-encode", "parameters": {},
            "path": str(target), "sha256": _sha256(target), "resolution": list(array.shape[1::-1])}


def transform_n1(path, out_path, hue_shift_degrees=30.0):
    """N1：HSV 色相轮转 +30°，明度通道不主动修改。"""
    import cv2
    import numpy as np
    array = _read_rgb(path)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV).astype(np.int32)
    shift = int(round(hue_shift_degrees / 2.0))  # OpenCV 色相 0–179 对应 0–358°
    original = hsv[..., 0].copy()
    hsv[..., 0] = (original + shift) % 180
    converted = np.clip(hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(converted, cv2.COLOR_HSV2RGB)
    target = _write_png(result, out_path)
    value_delta = float(np.abs(cv2.cvtColor(result, cv2.COLOR_RGB2HSV)[..., 2].astype(float)
                               - cv2.cvtColor(array, cv2.COLOR_RGB2HSV)[..., 2].astype(float)).mean())
    return {"variant": "N1", "operation": "HSV hue rotation", "parameters": {"hue_shift_degrees": hue_shift_degrees},
            "path": str(target), "sha256": _sha256(target), "resolution": list(array.shape[1::-1]),
            "recorded": {"mean_V_channel_shift": round(value_delta, 4),
                         "note": "只做确定性 8 位 HSV 往返；不做任何 CLAHE / 反锐化等旧配方"}}


def transform_n2(path, out_path, intensity_scale=0.9):
    """N2：灰度强度 ×0.9（记录截断/裁切像素比例）。"""
    import numpy as np
    array = _read_rgb(path)
    scaled = array.astype(np.float32) * intensity_scale
    clipped = float((scaled > 255).mean())
    under = float((scaled < 0).mean())
    result = np.clip(scaled, 0, 255).astype(np.uint8)
    target = _write_png(result, out_path)
    return {"variant": "N2", "operation": "RGB intensity scaling", "parameters": {"intensity_scale": intensity_scale},
            "path": str(target), "sha256": _sha256(target), "resolution": list(array.shape[1::-1]),
            "recorded": {"clipped_high_ratio": clipped, "clipped_low_ratio": under,
                         "mean_abs_delta": round(float(np.abs(result.astype(float) - array.astype(float)).mean()), 4)}}


def transform_n3(path, out_path, blur_sigma=6.0, mask_threshold_percentile=75.0, dilate=2, close=15):
    """N3：背景区域轻度高斯模糊，人物不变。

    背景 = 主体梯度的补集（`subject_mask`，同一确定性算法）。**只模糊背景像素**，
    主体像素逐字节保持原值；掩膜与覆盖率一并冻结，供人工核查。
    """
    import cv2
    import numpy as np
    array = _read_rgb(path)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    mask, threshold = subject_mask_with_percentile(gray, mask_threshold_percentile, dilate, close)
    background = mask == 0
    blurred = cv2.GaussianBlur(array, (0, 0), blur_sigma)
    result = array.copy()
    result[background] = blurred[background]
    identical = bool(np.array_equal(result[~background], array[~background]))
    target = _write_png(result, out_path)
    mask_path = Path(str(out_path)).with_suffix(".mask.png")
    _write_png((mask * 255).astype(np.uint8)[..., None].repeat(3, axis=2), mask_path)
    return {"variant": "N3", "operation": "background-only gaussian blur",
            "parameters": {"blur_sigma": blur_sigma, "mask_threshold_percentile": mask_threshold_percentile,
                           "dilate_iterations": dilate, "close_kernel": close},
            "path": str(target), "sha256": _sha256(target), "resolution": list(array.shape[1::-1]),
            "recorded": {"background_pixel_ratio": float(background.mean()),
                         "subject_pixels_byte_identical": identical,
                         "mask_path": str(mask_path), "mask_sha256": _sha256(mask_path)},
            "limitations": ["背景/主体由确定性边缘阈值近似，不是人工分割；边界与插值会影响结果，"
                            "不预设模糊后画风必然完全相同"]}


def subject_mask_with_percentile(gray, percentile=75.0, dilate=2, close=15):
    import cv2
    import numpy as np
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    threshold = float(np.percentile(gradient, percentile))
    mask = (gradient > threshold).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=int(dilate))
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return mask, threshold


def transform_n4(path, out_path, shift_fraction=0.05):
    """N4：内容整体平移画幅宽度的 5%，左侧留白用最左列边缘复制补边。"""
    import numpy as np
    array = _read_rgb(path)
    height, width = array.shape[:2]
    shift = int(round(width * shift_fraction))
    result = np.empty_like(array)
    if shift >= width:
        raise ValueError("平移量不能超过画幅宽度")
    result[:, shift:] = array[:, :width - shift]
    result[:, :shift] = array[:, shift:shift + 1]
    target = _write_png(result, out_path)
    return {"variant": "N4", "operation": "whole-frame translation with edge replication",
            "parameters": {"shift_fraction_of_width": shift_fraction, "shift_pixels": shift,
                           "padding": "replicate leftmost shifted column"},
            "path": str(target), "sha256": _sha256(target), "resolution": list(array.shape[1::-1]),
            "recorded": {"pad_columns": shift},
            "limitations": ["涉及边界与插值；像素内容位移后不预设画风完全相同"]}


TRANSFORMS = {"N0": transform_n0, "N1": transform_n1, "N2": transform_n2,
              "N3": transform_n3, "N4": transform_n4}


def build_contact_sheet(items, out_path, cell=320, columns=4):
    """把 (标签, 图片路径) 拼成一张核查用的接触表（人工确认干扰没有破坏人物结构）。"""
    import cv2
    import numpy as np
    rows = math.ceil(len(items) / columns)
    sheet = np.full((rows * (cell + 22), columns * cell, 3), 32, dtype=np.uint8)
    for index, (label, path) in enumerate(items):
        array = _read_rgb(path)
        scale = min(cell / array.shape[1], cell / array.shape[0])
        thumb = cv2.resize(array, (max(1, int(array.shape[1] * scale)), max(1, int(array.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        top, left = row * (cell + 22) + 22, column * cell
        sheet[top:top + thumb.shape[0], left:left + thumb.shape[1]] = thumb
        cv2.putText(sheet, str(label)[:28], (left + 4, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return target
