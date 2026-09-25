# -*- coding: utf-8 -*-
"""生图产物后处理流水线（GUI 勾选框与 CLI 工具共用）。

两步都是"在像素层面精修"，不改变画面内容：
1. `overlay_structure_lines` —— **结构线叠加**：抽「长结构边」→ 按局部色调整色 → 按 strength 叠回。
   纯本地、零 API 成本。实测三族平均段长 +6~22%、端点密度最多 −16%，配色/饱和度几乎不动
   （见 `docs/gpt-image-tid-style/README.md` §4.20）。
2. `local_repaint_composite` —— **局部重绘 + 羽化贴回**：按区域裁切 → 放大 → 走 Gemini 重绘（v5 固件 +
   区域强调句）→ 羽化贴回。近似"遮罩式局部重绘"（Gemini 的 generateContent 没有 mask 参数）。

区域预设（比例坐标 left,top,right,bottom）：upper / head / hair / face / skirt / full。
区域强调句用于把重绘注意力集中到该部位（头发最容易糊，所以 hair 有专门条款）。
"""
import os
import time

import cv2
import numpy as np

from utils import output_isolation

REGION_PRESETS = {
    "upper": (0.00, 0.00, 1.00, 0.62),
    "head": (0.00, 0.00, 1.00, 0.40),
    "hair": (0.02, 0.00, 0.98, 0.52),
    "face": (0.18, 0.02, 0.82, 0.34),
    "skirt": (0.05, 0.42, 0.95, 0.92),
    "subject": None,            # 整个人物：自动检测主体框（见 detect_subject_box）
    "subject_no_face": None,    # 人物不含面部：主体框（含手）+ 把脸区从贴回遮罩里排除
    "subject_keep_hands": None,  # 人物不含面部、且保留手部原像素（怕手被重绘坏时用）
    "shoes": (0.00, 0.80, 1.00, 1.00),   # 鞋子/靴子带（绑带、鞋带单独修）
    "shoes_zoom": None,     # 鞋/靴紧框（自动检测双脚位置，像素预算更高）
    "waist": (0.18, 0.26, 0.82, 0.54),   # 束腰/胸衣带（系带、扣眼单独修）
    "thigh": (0.20, 0.46, 0.95, 0.80),   # 大腿/袜带区（吊袜带、长袜花边单独修）
    "full": (0.00, 0.00, 1.00, 1.00),
}
REGION_LABELS = {
    "hair": "头发", "face": "脸部", "head": "头部", "skirt": "裙子", "upper": "上半身",
    "subject": "整个人物（全部重绘）",
    "subject_no_face": "人物·保脸（手参与重绘）",
    "subject_keep_hands": "人物·保脸保手（手保持原样）",
    "shoes": "鞋子/靴子带（绑带鞋带单独修）",
    "shoes_zoom": "鞋/靴紧框（自动定位双脚，细节最大）",
    "waist": "束腰/胸衣带（系带扣眼单独修）",
    "thigh": "大腿/袜带区（吊袜带长袜花边单独修）",
    "full": "整张",
}
# 需要从贴回遮罩里排除的区域（只有需要保留原像素时才用）
# 细节密集的区域默认用更高分辨率重绘（蕾丝/系带/扣件这类小结构在 2K 下会被糊掉）
DETAIL_REGION_RESOLUTION = {"waist": "4K", "thigh": "4K", "shoes": "4K", "shoes_zoom": "4K",
                            "face": "4K", "hair": "4K"}
# 为什么**没有**「手部自动紧框」区域（2026-09-23 实测后撤掉，别再顺手加回来）：
# 姿态模型是给人像照片训练的，在插画/厚涂上给出的手腕位置经常离谱 —— 实测一张全身插画，
# 两个「手」框分别落在**袖子和胸衣蕾丝**上，模型照着手部条款"修手"，就在蕾丝上画出了几根裸手手指
# （见 docs/gpt-image-tid-style/BEST-PIPELINE.md §二十五）。要修手请用坐标区域手动框：
#   python tools/local_repaint_composite.py --image <图> --crop <x0,y0,x1,y1> ...
# 手部画坏的根本解法在**首图阶段**：别把输入照片当内容参考图（见 §二十四）。
REGION_EXCLUSIONS = {"subject_no_face": ["face"],
                     "subject_keep_hands": ["face", "hands"]}
REGION_TOOLTIPS = {
    "hair": "只重画头发（最常用的修发丝区域）",
    "face": "只重画脸部：保留原有五官构造，只清断线",
    "head": "头部整块（含头发与脸）",
    "skirt": "裙子/裙摆褶皱",
    "upper": "上半身（衣领、肩带、腰线 + 头发）",
    "subject": "整个角色都参与重绘（脸和手也会被重画）",
    "subject_no_face": "整身重绘，但**脸部像素保持原样**（手照常参与重绘，戴手套状态会被锁定）",
    "subject_keep_hands": "整身重绘，但**脸和手都保持原样**（怕手指被重画坏、或手部细节已经很满意时用）",
    "full": "整张图（含背景）",
    "shoes": "只重画画面下缘的鞋/靴区（绑带、鞋带、扣件单独修，避免整身重绘把它们糊掉）",
    "shoes_zoom": "自动定位双脚并只重画鞋/靴紧框：像素预算最高，用来修鞋眼、鞋带、扣件这种小硬件",
    "waist": "只重画束腰/胸衣：前后系带、扣眼、腰线转折要连续可数",
    "thigh": "只重画大腿与袜带区：吊袜带、带扣、长袜花边要连续、有厚度",
}

DYNAMIC_REGIONS = {"subject", "subject_no_face", "subject_keep_hands", "shoes_zoom"}

# 「紧框」区域：box 只覆盖人物的一小块（目前只有鞋/靴紧框），贴回遮罩不能用这个 box 去算，
# 详见 `local_repaint_composite` 里 `TIGHT_REGIONS` 的说明。
TIGHT_REGIONS = {"shoes_zoom"}
# 「戴不戴手套」是角色设计状态，重绘时不能改（戴手套的手不能被画成裸手）
GLOVE_STATE_CLAUSE = (
    "\n\nHANDS AND GLOVES: keep the hands exactly in the state the source shows. If the character wears gloves "
    "(long gloves, lace gloves, fingerless gloves, mittens, wrist cuffs), the repaint must keep them worn - same "
    "material, colour, length, cut and trim, with the same coverage of the fingers - and must NOT turn a gloved "
    "hand into a bare hand, and never paint bare fingers inside a glove. If the hands are bare, keep them bare with "
    "the same number of fingers, correct finger order and readable joints. Never hide the hands, never merge them "
    "into the sleeve, hair or clothing, and never add or remove hand coverings."
)

REGION_EMPHASIS = {
    "hair": (
        "\n\nFOCUS: this crop is a HAIR-STRUCTURE repair. Treat the hair as the highest priority: rebuild every "
        "main hair lock as ONE continuous tapered strand that stays connected from root to tip; merge stray "
        "flyaway micro-strands into the nearest main lock; keep the strand count countable and the group "
        "structure readable; remove hairy fragments and disconnected wisps. Keep the source's own line weight, "
        "softness and translucency - do not switch to hard black outlines or cel shading, and do not add new "
        "accessories, flowers or ornaments to the hair."
    ),
    "subject_no_face": (
        "\n\nFOCUS: this crop is the WHOLE FIGURE (the face is excluded from the output). Keep the silhouette, hair "
        "locks, clothing contours, folds, hands and limb outlines coherent and countable; the background stays "
        "softer and less detailed than the figure." + GLOVE_STATE_CLAUSE
    ),
    "subject_keep_hands": (
        "\n\nFOCUS: this crop is the WHOLE FIGURE (the face and both hands are excluded from the output). Repair "
        "hair locks, clothing contours and folds only; leave the hands untouched." + GLOVE_STATE_CLAUSE
    ),
    "waist": (
        "\n\nFOCUS: this crop is the CORSET / BODICE band. The lacing must read as ONE countable system: "
        "the SAME cord crosses back and forth through EVENLY SPACED eyelets, so the number of visible crossings "
        "matches the number of eyelet pairs, and both cord ends tie in a visible bow or knot; draw each eyelet as a "
        "small dark opening with a rim, and let the cord pass THROUGH it instead of floating over the panel. Keep a "
        "constant cord thickness with visible tension and a small cast shadow under each crossing. The waist seam, "
        "stays and boning lines follow the body's curvature. Never turn the lacing into random hatching or parallel "
        "dashes, never merge cords into the fabric, never let a cord end disappear, and keep left/right symmetric."
        + GLOVE_STATE_CLAUSE
    ),
    "thigh": (
        "\n\nFOCUS: this crop is the THIGH / LEGWEAR band. Every garter strap must be a single continuous band "
        "that starts at a visible anchor (garter belt, suspender clip, buckle or elastic loop) and ends on the "
        "stocking edge: draw the clip or buckle as a small readable hardware shape, keep a constant strap width, and "
        "let the strap sit ON TOP of the stocking with a cast shadow. The stocking's lace top is a connected scallop "
        "pattern with clear negative space and a countable number of scallops. Skin, stocking and strap silhouettes "
        "stay separated by value, not by outlines. Never invent extra straps, never let a strap float unattached at "
        "either end, and never blur the stocking lace into the skin." + GLOVE_STATE_CLAUSE
    ),
    "shoes_zoom": (
        "\n\nFOCUS: this crop is a TIGHT close-up of ONE OR TWO shoes/boots. Because the crop is small, every "
        "hardware element must be drawn as a recognizable object: each eyelet is a small ring with a dark opening "
        "and a rim; each lace is one continuous cord that passes THROUGH the eyelets, crossing in a countable X "
        "pattern that ends in a visible bow with two loops and two tails; each buckle has a frame, a pin or hole and "
        "a cast shadow; the sole edge, welt stitching and heel block follow the shoe's perspective. Keep the leather "
        "sheen restrained and the whites from clipping. Never leave a lace end floating, never draw a buckle as a "
        "blob, never merge a strap into the shoe body, and keep left/right shoes symmetric in design." + GLOVE_STATE_CLAUSE
    ),
    "shoes": (
        "\n\nFOCUS: this crop is the FOOTWEAR band. Rebuild the shoes/boots as clean, countable structures: each "
        "lace is ONE continuous cord that crosses through evenly spaced eyelets and finishes in a visible bow or knot "
        "(the number of crossings must match the number of eyelet pairs); straps are continuous overlapping bands of "
        "constant width with a visible buckle, buckle hole or small hardware shape at the end; seams, sole edge and "
        "heel follow the shoe's perspective. Draw a small dark opening for every eyelet so the cord reads as passing "
        "through it. Never turn laces into a tangled scribble, never let a lace end vanish, never merge straps into "
        "the boot body, and keep left/right shoes consistent with each other." + GLOVE_STATE_CLAUSE
    ),
    "face": (
        "\n\nFOCUS: this crop is a FACE repair. Keep the facial construction (eyes, brows, nose, mouth, jaw) and "
        "its rendering exactly as in the source; only clean up broken contours and stray strokes around the eyes "
        "and mouth. Do not beautify, reshape, enlarge or restyle the face."
    ),
    "skirt": (
        "\n\nFOCUS: this crop is a DRESS/FOLD repair. Make the main skirt folds run continuously from the waist "
        "downward and fade out naturally at the hem; keep the ruffle and lace edges countable and connected. Keep "
        "the source's fabric texture; do not turn folds into comic ink lines or flat cel shapes."
    ),
    "upper": (
        "\n\nFOCUS: hair locks and clothing contours (collar, straps, waist line) are the priority for this crop."
    ),
    "subject": (
        "\n\nFOCUS: this crop is the WHOLE FIGURE, hands included. Keep the silhouette, hair locks, clothing "
        "contours, folds and the limb outlines coherent and countable, while leaving the background softer and less "
        "detailed than the figure. Do not add, remove or move any part of the character, and do not redesign the "
        "outfit."
        + GLOVE_STATE_CLAUSE
    ),
}
DEFAULT_STRUCTURE_STRENGTH = 0.280
DEFAULT_MIN_LEN = 120
DEFAULT_DARKEN = 0.18
DEFAULT_FEATHER = 48
DEFAULT_SCALE = 0.0      # 0/None = 自适应（见 auto_repaint_scale），别再用固定 1.5
MAX_REPAINT_EDGE = 2048
# 手部框用的姿态模型（ultralytics YOLO pose）：第一册会下载到 cache/models/
HAND_POSE_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "cache", "models", "yolo11n-pose.pt")
HAND_KEYPOINT_CONF = 0.3  # Gemini 图片通道的输入长边上限，超过会被压缩（放大再压缩=必然变糊）


# ----------------------------------------------------------------- 基础 IO

def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img, jpeg_quality=96):
    ext = os.path.splitext(path)[1].lower() or ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf.tofile(path)
    return path


# ----------------------------------------------------------------- 结构线叠加

def extract_structure_lines(edge_source, min_len=DEFAULT_MIN_LEN, low=60, high=140,
                            close_iter=2, dilate=0, thin=True):
    """提取「长结构边」掩膜：双边去噪 → Canny → 闭运算连断口 → 只保留长连通域。"""
    gray = cv2.cvtColor(edge_source, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    edges = cv2.Canny(gray, low, high)
    if close_iter:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_iter)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros_like(edges)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_len:
            keep[labels == i] = 255
    if thin:
        # 骨架化到 1px（Sol：结构线必须是发丝级细线，不能压成黑边）；ximgproc 缺失时用形态学腐蚀细化
        try:
            keep = cv2.ximgproc.thinning(keep)          # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            kernel = np.ones((3, 3), np.uint8)
            thin_mask = keep.copy()
            for _ in range(2):
                eroded = cv2.erode(thin_mask, kernel)
                opened = cv2.dilate(eroded, kernel)
                thin_mask = np.where(opened > 0, opened, thin_mask)
            keep = thin_mask
    if dilate:
        keep = cv2.dilate(keep, np.ones((2, 2), np.uint8), iterations=dilate)
    return keep


# 模型认的宽高比（与 prompts/gpt-image-optimize/config.json 的 aspect_ratio_options 一致）
MODEL_ASPECT_RATIOS = ((1, 1), (3, 2), (2, 3), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9))


def snapped_aspect_ratio(image_path, fallback: str = "auto") -> str:
    """按图片实际比例给出**最接近的官方比例标签**（如 "2:3"）。

    为什么要显式下发：`aspect_ratio=auto` 时模型理论上跟随输入图，但实测（tinkle 那次）
    给两张参考图 + 长固件时它会自作主张输出 1:1（源图 2:3 → 产物 2048x2048）。
    显式传最接近的比例可以锁住输出方向/比例，避免"宽图出竖图""竖图出方图"。
    """
    try:
        img = imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            if width > 0 and height > 0:
                ratio = width / float(height)
                best = min(MODEL_ASPECT_RATIOS, key=lambda pair: abs(pair[0] / pair[1] - ratio))
                return f"{best[0]}:{best[1]}"
    except Exception:  # noqa: BLE001
        pass
    return fallback


def build_line_anchor(image_path, out_path, min_len=DEFAULT_MIN_LEN, darken=0.75):
    """生成「线锚图」：白底 + 长结构线（§⑲ 里当第二张参考图用的那种线稿）。

    重绘时把它作为**第二张参考**一起送进去（源图 + 线锚图 = 双参考），
    模型就能同时看到"原图内容"与"结构线在哪"，线条连通性明显好于单参考。
    """
    base = imread(image_path)
    if base is None:
        raise RuntimeError(f"读不到图片: {image_path}")
    mask = extract_structure_lines(base, min_len=min_len)
    anchor_img = np.full_like(base, 255)
    if mask.max() > 0:
        tone = cv2.GaussianBlur(base, (0, 0), 2.0).astype(np.float32)
        line_col = np.clip(tone * (1.0 - darken), 0, 255)
        anchor_img[mask > 0] = line_col[mask > 0].astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imwrite(out_path, anchor_img)
    return out_path


def overlay_structure_lines(base, edge_source=None, strength=DEFAULT_STRUCTURE_STRENGTH,
                           min_len=DEFAULT_MIN_LEN, darken=DEFAULT_DARKEN, low=60, high=140,
                           thin=True, detail_dampen=0.7, line_rgb=None):
    """把结构线按局部色调整色后叠回 base；返回 (结果图, 掩膜)。

    - 细线：掩膜默认骨架化到 1px（`thin=True`）；
    - 低对比：`darken` 默认 0.18（线色≈局部色调 × 0.82，接近插画自己的浅线）；
    - `detail_dampen`：**高频细节区自动减弱**——花边/鞋带/地毯花纹这类密集边缘如果等权叠线，
      会被压成粗黑线团（实测 tinkle 那轮靴子绑带就是这样变乱的），按局部线密度把 alpha 拉低。
    """
    src = base if edge_source is None else edge_source
    if src.shape[:2] != base.shape[:2]:
        src = cv2.resize(src, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
    mask = extract_structure_lines(src, min_len=min_len, low=low, high=high, thin=thin)
    if mask.max() == 0 or strength <= 0:
        return base.copy(), mask
    tone = cv2.GaussianBlur(base, (0, 0), 2.0).astype(np.float32)
    if line_rgb is not None:
        line_col = np.zeros_like(tone)
        line_col[..., 0], line_col[..., 1], line_col[..., 2] = tuple(line_rgb)
    else:
        line_col = np.clip(tone * (1.0 - darken), 0, 255)
    weight = np.ones_like(mask, dtype=np.float32)
    if detail_dampen > 0:
        density = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 9.0)
        weight = np.clip(1.0 - float(detail_dampen) * np.clip(density * 3.0, 0.0, 1.0), 0.0, 1.0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * float(strength) * weight[..., None]
    out = base.astype(np.float32) * (1 - alpha) + line_col * alpha
    return np.clip(out, 0, 255).astype(np.uint8), mask


def reference_brightness(path):
    """参考图（通常是输入照片）的平均亮度（0~255），读不到返回 None。"""
    try:
        img = imread(path)
        if img is None:
            return None
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 0].mean())
    except Exception:  # noqa: BLE001
        return None


def white_mask(image, v_floor=200, s_ceil=90):
    """白色/高亮材质（白裙、蕾丝、皮肤高光）的粗略蒙版：明度高、饱和度低。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    m = np.clip((v - float(v_floor)) / max(1.0, 255.0 - float(v_floor)), 0, 1)
    m *= np.clip((float(s_ceil) - sat) / max(1.0, float(s_ceil)), 0, 1)
    return (m * 255).astype(np.uint8)


def local_contrast_enhance(image, amount=0.55, protect_white=True, radius=12, v_floor=200, s_ceil=90):
    """局部对比增强（unsharp on luminance）：让被高光压平的结构重新出现。

    - `amount`：增强量（0~1）；`protect_white=False` 时对全图生效；
    - 默认只对**白色/低饱和高亮**区域生效（`white_mask`），避免把已经正常的区域再拉对比；
    - 只在亮度通道做，不动色相与彩度，避免偏色。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    blur = cv2.GaussianBlur(L, (0, 0), float(radius))
    detail = (L - blur) * float(amount)
    if protect_white:
        m = white_mask(image, v_floor=v_floor, s_ceil=s_ceil).astype(np.float32) / 255.0
    else:
        m = np.ones_like(L, dtype=np.float32)
    lab[:, :, 0] = np.clip(L + detail * m, 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def skin_mask(image, h_lo=0, h_hi=25, s_lo=30, s_hi=170, v_lo=70):
    """粗略肤色蒙版（含 H 160~180 的环绕段），只用于轻微暖化，不做精确分割。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (((h >= h_lo) & (h <= h_hi)) | (h >= 160)) & (sat >= s_lo) & (sat <= s_hi) & (v >= v_lo)
    out = (m.astype(np.uint8)) * 255
    out = cv2.GaussianBlur(out, (0, 0), 6.0)
    return out


def reference_saturation(path):
    """参考图（输入照片）的平均 HSV 饱和度（0~255），读不到返回 None。"""
    try:
        img = imread(path)
        if img is None:
            return None
        return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    except Exception:  # noqa: BLE001
        return None


def line_contrast(image, min_len=60, close_iter=2):
    """量「线条像素 vs 周边底色」的平均明度差（正 = 线比底亮，负 = 线比底暗）。"""
    mask = extract_structure_lines(image, min_len=min_len, close_iter=close_iter, thin=True)
    if mask.max() == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(gray[mask == 0].mean() - gray[mask > 0].mean()) if (mask == 0).any() else 0.0


def ink_lines(image, target_sep=10.0, amount=1.0, min_len=60, close_iter=2,
              max_darken=40.0, sigma=6.0, protect_light=0.0, max_passes=4):
    """给已有线条「加墨」：只压线条像素，让线条相对**局部底色**保持可见的分离度。

    与 `overlay_structure_lines` 的区别：后者是把新抽的线**叠上去**（会改变画面），
    这里只把**已经画出来的线**压深，不改构图、不加新线，用来解决"线比底还亮 → 看着稀碎"的问题
    （实测 024958 那版线-底对比 −15.4 看起来清楚；后期版本只有 −0.5，线条就糊了）。

    **多遍迭代**：单遍压深后线条边缘跟着变暗、重新抽线时量到的分离度只回来一点点
    （实测目标 8.0 的一遍只从 −2.5 走到 1.2），于是同样跑 `--preset quality`，线条清晰度却
    因首图而异。这里每遍都按**和日志同一条口径**（`line_contrast`：非线像素均值 − 线像素均值）
    重新量，不够就再压一遍，到目标或 `max_passes` 为止；每像素累计压深不超过 `max_darken`，
    避免迭代把线压成黑粗块。
    """
    out = image
    target = float(target_sep)
    budget = np.full(out.shape[:2], float(max_darken), np.float32)   # 每像素剩余可压深额度
    for _ in range(max(1, int(max_passes))):
        current = line_contrast(out, min_len=min_len, close_iter=close_iter)
        if current >= target:
            break
        mask = extract_structure_lines(out, min_len=min_len, close_iter=close_iter, thin=True)
        if mask.max() == 0:
            break
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        # 均匀压深 delta 会让"非线 − 线"的均值大约上升 delta（线像素均值下降 delta）
        delta = np.full(L.shape, max(0.0, (target - current) * float(amount)), np.float32)
        if protect_light > 0:
            lim = 255.0 * float(protect_light)
            delta = np.where(L > lim, delta * 0.3, delta)
        need = np.where(mask > 0, np.minimum(delta, budget), 0.0)
        if need.max() <= 0.0:
            break
        budget = np.maximum(0.0, budget - need)
        lab[:, :, 0] = np.clip(L - need, 0, 255)
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def tone_calibrate(image, target_brightness=None, contrast=1.00, chroma=1.10,
                   highlight_protect=0.92, max_power=1.6, min_power=0.6,
                   target_saturation=None, sat_strength=0.7, max_chroma=1.15,
                   highlight_strength=0.90, skin_warm=0.0, sat_target_scale=1.0):
    """确定性色调校准（顺序：饱和度 → 亮度/对比/高光 → 肤色暖化）。

    - `target_brightness`：目标平均亮度（取输入照片）；幂律 `L^p`，`p = log(目标/255)/log(当前/255)`；
    - `contrast`：中对比（只在 highlight_protect 分位以下作用）。**默认 1.00（不打对比）**：
      实测 S 曲线会把细线对比压掉，段长 87→77、碎线 0.018→0.024（"线条变稀碎"的主因）；
    - `highlight_strength`：高光段压缩比例（0.75~0.80 = 比默认再压 3~5%）；
    - `target_saturation` / `sat_target_scale` / `sat_strength`：按参考图饱和度在 **HSV 的 S** 通道求解
      （可降到 0.45×；升饱和受 max_chroma 限制）；
    - `skin_warm`：只在肤色蒙版内轻微暖化（Lab a+ / b-）。
    **饱和度先做、亮度与高光最后做**：否则 HSV 往返会抬高 V，抵消高光压缩（实测白裁剪反而上升）。
    """
    out = image
    if target_saturation:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        cur_sat = float(hsv[:, :, 1].mean())
        if cur_sat > 1:
            want = (float(target_saturation) * float(sat_target_scale)) / cur_sat
            f = 1.0 + float(sat_strength) * (want - 1.0)
            f = float(min(float(max_chroma), max(0.45, f)))
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * f, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif chroma and abs(float(chroma) - 1.0) > 1e-3:
        lab0 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        for ch in (1, 2):
            lab0[:, :, ch] = np.clip(128.0 + (lab0[:, :, ch] - 128.0) * float(chroma), 0, 255)
        out = cv2.cvtColor(np.clip(lab0, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    x = np.clip(lab[:, :, 0] / 255.0, 0, 1)
    if target_brightness:
        cur = float(x.mean())
        tgt = float(target_brightness) / 255.0
        if 0.02 < cur < 0.98 and 0.02 < tgt < 0.98:
            p = float(np.log(tgt) / np.log(cur))
            p = float(min(max_power, max(min_power, p)))
            x = np.power(x, p)
    thr = float(np.percentile(x, 100 * float(highlight_protect or 1.0)))
    if contrast and abs(float(contrast) - 1.0) > 1e-3:
        lo = np.clip(0.5 + (x - 0.5) * float(contrast), 0, None)
        x = np.where(x <= thr, np.minimum(lo, thr), x)
    hi = x > thr
    if hi.any():
        x = np.where(hi, thr + (x - thr) * float(highlight_strength), x)
    lab[:, :, 0] = np.clip(x * 255.0, 0, 255)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    if skin_warm and abs(float(skin_warm)) > 1e-3:
        m = skin_mask(out).astype(np.float32) / 255.0
        lab3 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        alpha = float(skin_warm)
        lab3[:, :, 1] = np.clip(lab3[:, :, 1] + 14.0 * alpha * m, 0, 255)
        lab3[:, :, 2] = np.clip(lab3[:, :, 2] - 11.0 * alpha * m, 0, 255)
        out = cv2.cvtColor(np.clip(lab3, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def structure_overlay_file(image_path, out_path, strength=DEFAULT_STRUCTURE_STRENGTH,
                          edge_source_path=None, min_len=DEFAULT_MIN_LEN, darken=DEFAULT_DARKEN,
                          thin=True, detail_dampen=0.7, line_rgb=None):
    base = imread(image_path)
    src = imread(edge_source_path) if edge_source_path else base
    if base is None or src is None:
        raise RuntimeError(f"读不到图片: {image_path}")
    out, mask = overlay_structure_lines(base, src, strength=strength, min_len=min_len, darken=darken,
                                        thin=thin, detail_dampen=detail_dampen, line_rgb=line_rgb)
    imwrite(out_path, out)
    return {"out": out_path, "coverage": float((mask > 0).mean())}


# ----------------------------------------------------------------- 局部重绘贴回

def detect_face_box(img, expand=0.0, within=None):
    """用 Haar 检测最可能的人脸，返回 (x0,y0,x1,y1)；检不到/不可信时返回 None。

    做可信度过滤：人脸中心必须在画面上半部（竖构图人像的常识），大小合理，
    且（给了 within 时）落在该框的上半部——否则宁可回退到几何推算，
    比拿一个误检的"膝盖脸"去排除要安全得多。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.08, 5, minSize=(max(24, int(h * 0.05)), max(24, int(h * 0.05))))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cy = y + fh / 2.0
    if fh > 0.45 * h or cy > 0.55 * h:          # 太大或落在下半部 → 误检
        return None
    if within is not None:
        wx0, wy0, wx1, wy1 = within
        if not (wx0 <= x <= wx1 and wy0 <= y <= wy1) or cy > wy0 + 0.55 * (wy1 - wy0):
            return None
    px, py = int(fw * expand), int(fh * expand)
    return (max(0, x - px), max(0, y - py), min(w, x + fw + px), min(h, y + fh + py))


# GrabCut 内部用 OpenCV 全局 RNG 初始化 GMM（k-means），**不设种子时同一张图每次调用结果都不同**：
# 实测同一张 1696x2528 的产物，连调三次 `detect_subject_box` 得到 (46,663,…)、(337,351,…)、(46,422,…)
# 三个不同的人物框 —— 于是"同一条命令跑两次"会裁到不同位置，局部重绘的裁切框、主体遮罩、
# 鞋靴紧框全都在漂移，拼接缝/漏贴的位置也就没法复现、没法比较。
# 固定种子后 GrabCut 对同一输入稳定复现（验证：连调三次得到同一个框）。
GRABCUT_SEED = 20260923


def _seeded_grabcut(img, mask, rect, bgd_model, fgd_model, iterations, mode=cv2.GC_INIT_WITH_RECT):
    """带固定种子的 GrabCut（见 `GRABCUT_SEED` 说明：不设种子结果不可复现）。"""
    try:
        cv2.setRNGSeed(GRABCUT_SEED)
    except Exception:  # noqa: BLE001 - 老版本 OpenCV 没有 setRNGSeed 也不该中断
        pass
    return cv2.grabCut(img, mask, rect, bgd_model, fgd_model, int(iterations), mode)


def detect_subject_box(img, fallback=(0.06, 0.02, 0.94, 0.98)):
    """估一个人物框：GrabCut（中心矩形做前景先验）→ 最大前景连通域 bbox；失败回退比例框。

    没有 opencv-contrib（无 saliency 模块），所以用 GrabCut + 人脸锚点做兜底。
    """
    h, w = img.shape[:2]
    try:
        scale = 480.0 / max(h, w) if max(h, w) > 480 else 1.0
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale != 1.0 else img
        sh, sw = small.shape[:2]
        rect = (int(sw * 0.08), int(sh * 0.02), int(sw * 0.84), int(sh * 0.96))
        if rect[2] <= 2 or rect[3] <= 2:
            raise ValueError("rect too small")
        mask = np.zeros((sh, sw), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        _seeded_grabcut(small, mask, rect, bgd, fgd, 3)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh, area = stats[idx]
            if area > 0.03 * sw * sh:
                x0 = int(x / scale)
                y0 = int(y / scale)
                x1 = int((x + bw) / scale)
                y1 = int((y + bh) / scale)
                # 稍微外扩，避免把发梢/裙摆切掉
                pad_x, pad_y = int((x1 - x0) * 0.06), int((y1 - y0) * 0.04)
                box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y))
                if (box[2] - box[0]) > w * 0.25 and (box[3] - box[1]) > h * 0.3:
                    face = detect_face_box(img, within=box)
                    if face:      # 保证脸在框内
                        box = (min(box[0], face[0]), min(box[1], face[1]),
                               max(box[2], face[2]), max(box[3], face[3]))
                    return box
    except Exception:  # noqa: BLE001 - 检测失败就回退比例框
        pass
    return (int(fallback[0] * w), int(fallback[1] * h), int(fallback[2] * w), int(fallback[3] * h))


_HAND_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "cache", "hand_boxes.json")


def _hand_cache_load():
    import json
    try:
        with open(_HAND_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def _hand_cache_save(cache):
    import json
    try:
        os.makedirs(os.path.dirname(_HAND_CACHE_PATH), exist_ok=True)
        with open(_HAND_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass


def _hand_cache_key(image_path, subject_box):
    try:
        st = os.stat(image_path)
        meta = f"{os.path.abspath(image_path)}|{int(st.st_mtime)}|{st.st_size}|{subject_box}"
    except Exception:  # noqa: BLE001
        meta = f"{image_path}|{subject_box}"
    import hashlib
    return hashlib.md5(meta.encode("utf-8")).hexdigest()


def detect_hand_boxes(image, subject_box=None, conf=HAND_KEYPOINT_CONF, image_path=None):
    """找出画面里的手（返回 [(x0,y0,x1,y1), ...]）。

    **姿态检测跑在子进程里**（`tools/detect_hands_pose.py`）：torch/ultralytics 的 DLL
    在 GUI 主进程里加载会 access violation（实测崩过），所以主流程永不 import torch；
    子进程崩溃/超时也只是返回空手部框，不影响画图。结果按 (路径+mtime+尺寸+人物框) 缓存到
    `cache/hand_boxes.json`，同一张图只跑一次。
    子进程不可用时退回**肤色色块**（纯 OpenCV，安全）。
    """
    h, w = image.shape[:2]
    boxes = []
    # 1) 缓存（有 image_path 才能按文件缓存；否则用内容+人物框做 key）
    cache_key = None
    cache = None
    if image_path and os.path.isfile(str(image_path)):
        cache = _hand_cache_load()
        cache_key = _hand_cache_key(str(image_path), subject_box)
        if cache_key in cache:
            return [tuple(b) for b in (cache[cache_key] or [])]
    # 2) 子进程姿态检测
    try:
        import json
        import subprocess
        import sys
        import tempfile
        tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "detect_hands_pose.py")
        if image_path and os.path.isfile(str(image_path)) and os.path.isfile(tool):
            with tempfile.TemporaryDirectory(prefix="hands_") as tmp:
                src = os.path.join(tmp, "image.png")
                imwrite(src, image)
                out_json = os.path.join(tmp, "hands.json")
                cmd = [sys.executable, tool, "--image", src, "--out", out_json, "--conf", str(conf)]
                if subject_box:
                    cmd += ["--subject", ",".join(str(int(v)) for v in subject_box)]
                subprocess.run(cmd, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.isfile(out_json):
                    data = json.load(open(out_json, encoding="utf-8"))
                    if data.get("ok"):
                        boxes = [tuple(int(v) for v in b) for b in (data.get("hands") or [])]
    except Exception:  # noqa: BLE001 - 子进程失败就当没找到手
        boxes = boxes or []
    if boxes and cache is not None and cache_key:
        cache[cache_key] = [list(b) for b in boxes]
        _hand_cache_save(cache)
    if boxes:
        return boxes[:4]
    # 3) 兜底：裸手肤色色块（戴手套的手找不到，尽力而为）
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255)) | cv2.inRange(hsv, (160, 30, 60), (180, 180, 255))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.0004 * h * w or area > 0.02 * h * w:
                continue
            if y < 0.35 * h:
                continue
            if not (0.35 <= bw / max(1, bh) <= 3.0):
                continue
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    except Exception:  # noqa: BLE001
        boxes = []
    if boxes and cache is not None and cache_key:
        cache[cache_key] = [list(b) for b in boxes]
        _hand_cache_save(cache)
    return boxes[:4]


def detect_footwear_box(image, subject_box=None, bottom_ratio=0.28, pad=0.02):
    """鞋/靴的**紧框**：取主体遮罩里最靠下、面积最大的连通块（通常是双脚），再向下留一点边。

    为什么要紧框：局部重绘的输入长边上限是 2048，裁得越紧，同样的像素预算覆盖的真实细节越少、
    每像素信息越多 —— 鞋眼/鞋带/扣件这类小硬件才画得清。
    """
    h, w = image.shape[:2]
    try:
        mask = subject_mask(image, box=subject_box, pad_ratio=0.0)
    except Exception:  # noqa: BLE001
        mask = None
    y_start = int(h * (1.0 - float(bottom_ratio)))
    if mask is not None and (mask > 0).any():
        band = np.zeros_like(mask)
        band[y_start:, :] = mask[y_start:, :]
        n, labels, stats, _ = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), connectivity=8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh, area = stats[idx]
            if area > 0.0002 * h * w:
                px, py = int(w * pad), int(h * pad)
                return (int(max(0, x - px)), int(max(0, y - py)),
                        int(min(w, x + bw + px)), int(min(h, y + bh + py)))
    # 兜底：画面下缘 20%
    return (0, int(h * 0.80), w, h)


def resolve_region_box(region, width, height, image=None):
    """把区域名/比例/像素坐标解析成像素 box。

    `subject` / `subject_no_face` 需要传 image（自动检测人物框），否则回退成上半身。
    """
    text = str(region or "upper").strip().lower()
    if text in DYNAMIC_REGIONS:
        if image is None:
            vals = REGION_PRESETS["upper"]
            return (int(vals[0] * width), int(vals[1] * height), int(vals[2] * width), int(vals[3] * height))
        if text == "shoes_zoom":
            # 「鞋/靴紧框」= 主体遮罩里最靠下、面积最大的连通块（`detect_footwear_box`）。
            # 这里曾经漏接：DYNAMIC_REGIONS 里带着 shoes_zoom，却一律返回整个人物框 →
            # 鞋靴 FOCUS 条款被贴到整张全身裁切上，像素预算也摊平到全身，
            # 「鞋眼/鞋带/扣件」这些小硬件根本没得到紧框该有的放大倍数
            # （实测两个区域的 -crop.png 是逐字节相同的文件）。
            return detect_footwear_box(image)
        return detect_subject_box(image)
    if text in REGION_PRESETS:
        vals = REGION_PRESETS[text]
    elif "," in text:
        vals = tuple(float(v) for v in text.split(","))
        if len(vals) != 4:
            raise ValueError("区域需要 4 个数（比例 left,top,right,bottom 或像素 x,y,w,h）")
    else:
        raise ValueError(f"未知区域: {region}")
    if max(vals) <= 1.5:
        x0, y0, x1, y1 = vals
        return (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
    return tuple(int(v) for v in vals)


def resolve_exclusion_boxes(region, image, box=None, manual=None, image_path=None):
    """需要保留原像素的区域（「人物（不含面部与手）」会把脸和手都排除在贴回之外）。

    - face：Haar 人脸优先，动漫脸检不到时从人物框上半部居中断定；
    - hands：`detect_hand_boxes`（YOLO pose 手腕优先，戴手套也能找到）；
    - `manual`：调用方给的手动框 [(x0,y0,x1,y1), ...]，会一并加入（方便手动兜底）。
    """
    keys = REGION_EXCLUSIONS.get(str(region or "").strip().lower(), [])
    image_h, image_w = image.shape[:2]
    out = []
    for key in keys:
        if key == "face":
            face = detect_face_box(image, expand=0.12, within=box)
            if face is None:
                if box:
                    sx0, sy0, sx1, sy1 = box
                    bw, bh = sx1 - sx0, sy1 - sy0
                    fw, fh = int(bw * 0.42), int(bh * 0.20)
                    cx = (sx0 + sx1) // 2
                    top = sy0 + int(bh * 0.01)
                    face = (max(0, cx - fw // 2), max(0, top), min(image_w, cx + fw // 2),
                            min(image_h, top + fh))
                else:
                    vals = REGION_PRESETS["face"]
                    face = (int(vals[0] * image_w), int(vals[1] * image_h),
                            int(vals[2] * image_w), int(vals[3] * image_h))
            out.append(face)
        elif key == "hands":
            out.extend(detect_hand_boxes(image, subject_box=box, image_path=image_path))
    for item in (manual or []):
        if len(item) == 4:
            out.append(tuple(int(v) for v in item))
    return out


def subject_mask(image, box=None, pad_ratio=0.015, iterations=5, max_edge=640):
    """主体遮罩（0=背景 / 255=主体），用于"只贴回角色、背景保持原像素"。

    做法：`detect_subject_box` 定框（已含最大前景连通域 + 人脸锚定）→ 框内 GrabCut（多轮）→
    只保留最大连通域 → 填洞 → 按 `pad_ratio` 膨胀一点（避免把轮廓边缘切掉）。
    局部重绘第二遍（如 shoes）只有限制在主体内，才不会把地毯/家具重画一遍造成水平拼接缝。
    """
    h, w = image.shape[:2]
    if box is None:
        box = detect_subject_box(image)
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.full((h, w), 255, np.uint8)
    scale = min(1.0, max_edge / float(max(x1 - x0, y1 - y0)))
    small = cv2.resize(image[y0:y1, x0:x1], (max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))),
                       interpolation=cv2.INTER_AREA)
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (2, 2, max(3, small.shape[1] - 4), max(3, small.shape[0] - 4))
    binary = None
    try:
        _seeded_grabcut(small, mask, rect, bgd, fgd, int(iterations))
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # 只保留最大连通域（去掉误判的地毯/家具碎块）
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = np.where(labels == biggest, 255, 0).astype(np.uint8)
        # 填洞：轮廓填充后再取并集
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        binary = cv2.bitwise_or(binary, filled)
    except Exception:  # noqa: BLE001
        binary = np.full(small.shape[:2], 255, np.uint8)
    binary = cv2.resize(binary, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    pad_px = int(max(6, max(x1 - x0, y1 - y0) * float(pad_ratio)))
    binary = cv2.dilate(binary, np.ones((pad_px | 1, pad_px | 1), np.uint8), iterations=1)
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = binary
    return full


def feather_blend(base, patch, box, feather=DEFAULT_FEATHER, exclude_boxes=None, restrict_mask=None):
    """把 patch 羽化贴回 base 的 box 区域；exclude_boxes 里的像素保持原样（反向羽化）。

    `restrict_mask`（与 base 同尺寸的 0/255 图）不为空时**只在遮罩内贴回**：背景（地毯/家具/椅子腿）
    保持 base 原像素。第二遍局部重绘（如 shoes 带）如果不限制，模型会把整条带子重画一遍，
    家具与地毯纹理随之改变，图上就会出现明显的水平拼接缝与"多出来的椅子腿"。
    """
    x0, y0, x1, y1 = box
    h, w = max(1, y1 - y0), max(1, x1 - x0)
    patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LANCZOS4)
    f = max(1, min(int(feather), h // 3, w // 3))
    mask = np.zeros((h, w), np.float32)
    mask[f: h - f, f: w - f] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), f / 2.0)
    for ex in (exclude_boxes or []):
        ex0, ey0 = max(x0, int(ex[0])), max(y0, int(ex[1]))
        ex1, ey1 = min(x1, int(ex[2])), min(y1, int(ex[3]))
        if ex1 - ex0 < 2 or ey1 - ey0 < 2:
            continue
        keep = np.ones((h, w), np.float32)
        keep[ey0 - y0: ey1 - y0, ex0 - x0: ex1 - x0] = 0.0
        soft = max(2.0, min(f / 2.0, 24.0))
        keep = cv2.GaussianBlur(keep, (0, 0), soft)
        mask = mask * np.clip(keep, 0, 1)
    if restrict_mask is not None:
        rm = restrict_mask
        if rm.shape[:2] != base.shape[:2]:
            rm = cv2.resize(rm, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        zone = (rm[y0:y1, x0:x1].astype(np.float32) / 255.0)
        # 先羽化一点点，避免遮罩边缘出现硬切；再与主体遮罩相乘（背景 alpha→0）
        zone = cv2.GaussianBlur(zone, (0, 0), max(1.0, f / 4.0))
        mask = mask * np.clip(zone, 0, 1)
    mask = np.clip(mask, 0, 1)[..., None]
    roi = base[y0:y1, x0:x1].astype(np.float32)
    out = base.copy()
    out[y0:y1, x0:x1] = np.clip(roi * (1 - mask) + patch.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return out


def auto_repaint_scale(crop_w, crop_h, max_edge=MAX_REPAINT_EDGE, min_target=1536, max_scale=2.0):
    """裁切区送重绘前的缩放系数（自适应，避免"放大→接口压缩→再放大"这种必然变糊的路径）。

    规则：目标长边 = clamp(裁切长边, min_target, max_edge)；小裁切可放大到最多 `max_scale`，
    大裁切原样（或轻微下采样到上限），**永不产生超过接口上限的输入**。
    """
    long_edge = max(int(crop_w or 0), int(crop_h or 0))
    if long_edge <= 0:
        return 1.0
    target = min(max_edge, max(long_edge, int(min_target)))
    return round(min(max_scale, target / long_edge), 3)


def resolve_firmware_text(firmware):
    """把 firmware 参数解析成真正的提示词文本。

    GUI 传进来的常是**固件文件路径**（`repaint-system-conservative-v5.md`），而
    `generate_image_repaint(prompt=...)` 要的是文本——直接传路径会把文件名当提示词发出去
    （日志里 promptTokenCount 只有 20 就是这个问题），重绘就退化成"无固件的裸重绘"。
    """
    text = str(firmware or "").strip()
    if not text:
        return ""
    candidates = [text]
    if not os.path.isabs(text):
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), text))
    for cand in candidates:
        if os.path.isfile(cand) and os.path.splitext(cand)[1].lower() in (".md", ".txt"):
            try:
                with open(cand, encoding="utf-8") as f:
                    body = f.read().strip()
                if body:
                    return body
            except Exception:  # noqa: BLE001
                pass
    return text


def local_repaint_composite(image_path, out_path, region="upper", firmware=None,
                            resolution="2K", scale=None, feather=DEFAULT_FEATHER,
                            save_sub_dir="", keep_crop=False, emphasize=True,
                            patch_path="", repaint_callable=None, log_callback=None,
                            scratch_dir=None, exclude_boxes=None,
                            restrict_to_subject=False, subject_pad=0.02, qa_callback=None):
    """按区域局部重绘 + 羽化贴回。

    `repaint_callable` 可注入（测试用）：签名 (crop_path, prompt_path) -> 产物路径。
    默认走 `modules.others.api_backend.generate_image_repaint`（v5 固件 + 区域强调句）。
    """
    log = log_callback or (lambda m: None)
    base = imread(image_path)
    if base is None:
        raise RuntimeError(f"读不到图片: {image_path}")
    h, w = base.shape[:2]
    box = resolve_region_box(region, w, h, image=base)
    x0, y0, x1, y1 = box
    exclude_boxes = resolve_exclusion_boxes(region, base, box=box, manual=exclude_boxes,
                                            image_path=image_path)

    # 主体遮罩：只在人物上贴回，背景（地毯/家具）保持 base 原像素。
    # 不做限制时，第二遍局部重绘会把整条带子（含家具）重画一遍 → 图上出现水平拼接缝与"多出来的椅子腿"。
    restrict_mask = None
    if restrict_to_subject:
        try:
            # 紧框区域（box 只是人物的一小块，如鞋靴）：直接拿它跑 GrabCut，等于告诉算法
            # "框内全是前景、且主体贴着四条边"，而 GrabCut 的矩形先验假设主体不贴边 →
            # 遮罩基本为空、贴回等于没贴。这些区域改成用**整个人物框**生成遮罩。
            mask_box = None if str(region).strip().lower() in TIGHT_REGIONS else box
            restrict_mask = subject_mask(base, box=mask_box, pad_ratio=subject_pad)
            log(f"[后处理] 主体限制贴回已启用（遮罩覆盖 {float((restrict_mask > 0).mean()):.3f}）")
            # 遮罩在**本次区域**里几乎没覆盖时不要用它：整图 GrabCut 估歪（实测亮蕾丝背景上只框住
            # 下半身）会让这次重绘整块被切掉 —— 表现就是「改动像素占比 0.000」，白跑一次 API 还看不出问题。
            # 这种情况宁可整框贴回（区域本来就是人身上的一小块），也不要静默地什么都不做。
            try:
                zone_cover = float((restrict_mask[y0:y1, x0:x1] > 0).mean())
                if zone_cover < 0.02:
                    log(f"[后处理] ⚠ 主体遮罩在区域内几乎没有覆盖（{zone_cover:.3f}）→ 改为整框贴回")
                    restrict_mask = None
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            log(f"[后处理] 主体遮罩生成失败，改为整框贴回: {exc}")
            restrict_mask = None

    def _finish(patch_img, pad=(0, 0, 0, 0), crop_size=None, extra=None):
        """统一收尾：补边裁回 → 羽化贴回（含主体限制）→ QA → 落盘。"""
        pt, pb, pl, pr = pad
        if any(pad) and crop_size is not None:
            target_h, target_w = crop_size
            patch_img = cv2.resize(patch_img, (target_w, target_h),
                                   interpolation=cv2.INTER_AREA if patch_img.shape[1] > target_w
                                   else cv2.INTER_LANCZOS4)
            patch_img = patch_img[pt:max(pt + 1, target_h - pb), pl:max(pl + 1, target_w - pr)]
        merged = feather_blend(base, patch_img, box, feather=feather,
                               exclude_boxes=exclude_boxes, restrict_mask=restrict_mask)
        imwrite(out_path, merged)
        if qa_callback is not None:
            try:
                zone = merged[y0:y1, x0:x1].astype(np.float32)
                old = base[y0:y1, x0:x1].astype(np.float32)
                changed = float((np.abs(zone - old).mean(axis=2) > 12).mean())
                outside = 0.0
                if restrict_mask is not None:
                    bgm = restrict_mask[y0:y1, x0:x1] == 0
                    if bgm.sum():
                        diff = np.abs(zone - old).mean(axis=2)
                        outside = float((diff[bgm] > 12).mean())
                qa_callback({"changed_ratio": changed, "background_changed": outside,
                             "box": box, "padded": pad})
            except Exception:  # noqa: BLE001
                pass
        log(f"[后处理] 已贴回 → {out_path}"
            + (f"（保留原像素 {len(exclude_boxes)} 块）" if exclude_boxes else ""))
        result = {"out": out_path, "box": box, "exclude_boxes": exclude_boxes}
        if extra:
            result.update(extra)
        return result

    if patch_path:
        patch = imread(patch_path)
        if patch is None:
            raise RuntimeError(f"读不到局部重绘结果: {patch_path}")
        log(f"[后处理] 用已有局部重绘结果 {os.path.basename(patch_path)} 贴回 {box}"
            + (f"（排除 {len(exclude_boxes)} 块区域）" if exclude_boxes else ""))
        return _finish(patch, extra={"patch": patch_path})

    crop = base[y0:y1, x0:x1]
    if not scale or float(scale) <= 0:            # 默认自适应，避免超过接口长边上限
        scale = auto_repaint_scale(crop.shape[1], crop.shape[0])
    if scale and abs(float(scale) - 1.0) > 1e-3:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
    # 补边到最接近的官方比例（而不是把内容拉成那个比例）——回来再裁掉，避免错位/拉伸
    crop_h, crop_w = crop.shape[:2]
    src_ratio = crop_w / float(max(1, crop_h))
    pad_top = pad_bottom = pad_left = pad_right = 0
    best = min(MODEL_ASPECT_RATIOS, key=lambda pair: abs(pair[0] / pair[1] - src_ratio))
    target_ratio = best[0] / best[1]
    if abs(target_ratio - src_ratio) > 0.01:
        if target_ratio < src_ratio:            # 目标更"高" → 上下补边
            extra_px = max(0, int(round(crop_w / target_ratio)) - crop_h)
            pad_top, pad_bottom = extra_px // 2, extra_px - extra_px // 2
        else:                                    # 目标更"宽" → 左右补边
            extra_px = max(0, int(round(crop_h * target_ratio)) - crop_w)
            pad_left, pad_right = extra_px // 2, extra_px - extra_px // 2
        if pad_top or pad_bottom:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE)
        if pad_left or pad_right:
            crop = cv2.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE)
    scratch = scratch_dir or os.path.dirname(os.path.abspath(out_path))
    os.makedirs(scratch, exist_ok=True)
    crop_path = os.path.join(scratch, os.path.splitext(os.path.basename(out_path))[0] + "-crop.png")
    imwrite(crop_path, crop)
    log(f"[后处理] 局部重绘 区域={REGION_LABELS.get(str(region).lower(), region)} box={box} "
        f"裁切 {crop.shape[1]}x{crop.shape[0]}（缩放 {scale}x，接口长边上限 {MAX_REPAINT_EDGE}，"
        f"补边 {pad_top}/{pad_bottom}/{pad_left}/{pad_right}）")

    if repaint_callable is not None:
        patch_path = repaint_callable(crop_path, resolve_firmware_text(firmware))
    else:
        from modules.others.api_backend import generate_image_repaint
        firmware_text = resolve_firmware_text(firmware)
        if emphasize:
            firmware_text += REGION_EMPHASIS.get(str(region).lower(), "")
        prompt = firmware_text or None
        if not save_sub_dir:
            save_sub_dir = os.path.join("pipeline-steps", os.path.basename(os.path.abspath(scratch)))
        saved = generate_image_repaint(
            source_paths=[crop_path], resolution=resolution, prompt=prompt,
            aspect_ratio=snapped_aspect_ratio(crop_path),
            use_detail_suffix=False,
            save_sub_dir=save_sub_dir, file_prefix="local",
        )
        patch_path = saved[0] if saved else ""
    if not patch_path or not os.path.isfile(patch_path):
        raise RuntimeError("局部重绘没有返回图片")
    patch = imread(patch_path)
    if keep_crop:
        log(f"[后处理] 裁切图保留: {crop_path}  局部重绘结果: {patch_path}")
    return _finish(patch, pad=(pad_top, pad_bottom, pad_left, pad_right),
                   crop_size=(crop_h, crop_w), extra={"patch": patch_path, "crop": crop_path})


# ----------------------------------------------------------------- 流水线

def default_pipeline():
    return {
        # `dual_reference` 是**旧键**：只在 `reference_mode` 缺失时兜底（True → 线锚图）。
        # 新调用方请显式传 `reference_mode`；GUI 默认 none，style 仅供显式实验。
        "repaint": {"enabled": False, "dual_reference": True},
        "contrast": {"enabled": False, "amount": 0.55, "protect_white": True, "radius": 12},
        "ink": {"enabled": False, "target_sep": 10.0, "amount": 1.0, "max_darken": 40.0,
                "sigma": 6.0, "protect_light": 0.0},
        "tone": {"enabled": False, "reference_path": "", "target_brightness": None,
                 "target_saturation": None, "contrast": 1.00, "chroma": 1.10,
                 "highlight_protect": 0.92, "highlight_strength": 0.90,
                 "skin_warm": 0.0, "sat_target_scale": 1.0},
        "structure": {"enabled": False, "strength": DEFAULT_STRUCTURE_STRENGTH,
                      "min_len": DEFAULT_MIN_LEN, "darken": DEFAULT_DARKEN,
                      "thin": True, "detail_dampen": 0.7, "line_rgb": None},
        "local": {"enabled": False, "region": "hair", "scale": DEFAULT_SCALE,
                  "feather": DEFAULT_FEATHER, "resolution": "2K"},
    }


STEP_LABELS = {"repaint": "重绘提线", "structure": "结构线叠加", "contrast": "白色材质局部对比",
               "ink": "线条加墨",
               "local": "局部重绘+羽化贴回",
               "tone": "色调校准"}
STEP_TAGS = {"repaint": "rp", "structure": "sline", "contrast": "contrast", "local": "local",
             "tone": "tone", "ink": "ink"}


def _manifest_path(work_dir):
    return os.path.join(work_dir, "pipeline-manifest.json")


def load_manifest(work_dir):
    """读流水线清单（记录每一步的输入/产物/状态，用于断点重试）。"""
    import json
    try:
        with open(_manifest_path(work_dir), encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def save_manifest(work_dir, manifest):
    import json
    os.makedirs(work_dir, exist_ok=True)
    with open(_manifest_path(work_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def pipeline_failures(work_dir):
    """返回上次运行里失败的步骤（供 GUI 提示「可重试」）。"""
    manifest = load_manifest(work_dir)
    out = []
    for item in manifest.get("items") or []:
        for step in item.get("steps") or []:
            if step.get("status") == "failed":
                out.append({"source": item.get("source"), "step": step.get("key"),
                            "error": step.get("error")})
    return out


def final_product_name(stem, run_id, steps_cfg, ext=".png"):
    """最终产物的文件名：<来源名>-<run-id>-final-<工序串><ext>。

    带 `-final-` 标记是为了让人一眼看出这是最终图，而不是中间的 sline 图；
    工序串（rp/sline35/local-hair）说明这条产物经过了哪些处理，细节看 manifest。
    """
    tags = []
    for key in ("repaint", "structure", "contrast", "local", "tone", "ink"):
        cfg = (steps_cfg or {}).get(key) or {}
        if not cfg.get("enabled"):
            continue
        if key == "structure":
            tags.append(f"sline{int(float(cfg.get('strength', DEFAULT_STRUCTURE_STRENGTH)) * 100):02d}")
        elif key == "local":
            # 多区域时把区域列表都写进名字：只写第一个区域的话，跑 4 个区域和跑 1 个区域
            # 在文件名上完全看不出区别（见 §二十三 的排查过程）。
            regions = [str(r) for r in (cfg.get("regions") or []) if str(r).strip()]
            if len(regions) > 1:
                tags.append("local-" + ",".join(regions))
            else:
                tags.append(f"local-{cfg.get('region') or (regions[0] if regions else 'hair')}")
        else:
            tags.append(STEP_TAGS.get(key, key))
    tag = "+".join(tags) or "raw"
    return f"{stem}-{run_id}-final-{tag}{ext}"


def new_run_id() -> str:
    """每次流水线运行一个短 id：HHMMSS-xxxxxx。多线程并行时用它区分中间目录与产物名。"""
    import uuid
    return time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]


def date_output_dir(date_str=None):
    """最终产物的落盘目录：data/<YYYYMMDD>/（发布用，不再塞子目录）。"""
    import datetime as _dt
    return os.path.join("data", date_str or _dt.datetime.now().strftime("%Y%m%d"))


def final_output_path(effective_dir, base_stem, run_id, steps_cfg, source_dir=None):
    """最终产物路径（含测试产出隔离）。

    `effective_dir` 是**已经改写过的**落盘目录（`run_pipeline` 里会把 `data/<日期>` 换成
    `data/test-result/<日期>`），所以文件名前缀要按**原始目录** `source_dir` 来判定 ——
    `resolve_output_target` 只在目标是 `data/<日期>/` 时加 `test-` 前缀。
    """
    name = final_product_name(base_stem, run_id, steps_cfg)
    _, filename = output_isolation.resolve_output_target(source_dir or effective_dir, name)
    os.makedirs(effective_dir, exist_ok=True)
    return os.path.join(effective_dir, filename)


STYLE_REF_ROLE_IN_REPAINT = (
    "\n\n" + "CHARACTER-INTRINSIC FEATURES ARE NOT PART OF THE STYLE: keep the subject's OWN hair colour, eye colour, skin tone, body proportions and outfit colours exactly as described by the content; the style reference must only change the rendering language (palette temperature of the environment, brushwork, line character, edge treatment, texture), never the subject's intrinsic colouring or design." +
    "\n\nREFERENCE ROLES: Image 1 is the SOURCE and the only truth for content, identity, pose and composition. "
    "Any further image is a STYLE REFERENCE that supplies rendering language only (palette, line character, "
    "brushwork, edge treatment, material handling). Never copy the style reference's character, features, outfit, "
    "pose, props, background or composition, and never let it override the source's content."
    "\nThe style reference is NEVER a source of content: nothing in the output may come from it except the way "
    "paint is applied. It is not a second character, not a second outfit, not a second scene."
)

STYLE_REF_ROLE_NEUTRAL_IN_REPAINT = (
    "\n\nNEUTRAL STYLE REFERENCE MODE: Image 1 is the SOURCE and controls every concrete colour, value relationship, "
    "character feature, object, garment, accessory, pose and composition. Image 2 may influence only abstract "
    "brush motion, stroke taper, edge softness, paint layering and non-semantic surface finish. Do not read or "
    "transfer Image 2's palette, hue distribution, lighting colour, saturation, eye construction, hair design, "
    "clothing, motifs or objects. If a rendering method cannot be separated from Image 2's content or colours, "
    "do not transfer it."
)

# 完整画风图模式下，允许迁移“怎么画五官/头发”的抽象语法，同时把人物身份和具体设计留在源图。
# v5 固件默认会像素级锁眼睛；没有这段具体例外时，重绘只能迁移配色与材质，脸和头发仍像 GPT 首图。
STYLE_REF_FACE_HAIR_GRAMMAR = (
    "\n\nSPECIFIC STYLE-SPACE EXCEPTION FOR FACE AND HAIR: transfer HOW the reference artist abstracts and "
    "renders a face and hair without transferring WHO the reference character is. You may translate the source "
    "eyes into the reference's visual grammar (upper-lid angle and weight, lash grouping, iris-to-pupil ratio, "
    "highlight design and degree of simplification), the source face into the reference's abstraction grammar "
    "(cheek/jaw softness, nose and mouth economy), and the source hair into the reference's lock grouping, strand "
    "density, contour rhythm and highlight-shape grammar. This specific exception overrides earlier instructions "
    "to preserve the source eye drawing pixel-for-pixel. It does NOT permit copying the reference person's face, "
    "feature placement, expression, eye or hair colour, hairstyle, bangs, hair length, accessories, age or identity. "
    "Keep the source character's recognisable identity, expression, gaze, intrinsic colours and concrete hair design; "
    "render those same facts with the reference artist's facial and hair drawing language."
)

# 「身份/内容锁」：**每一档编辑范围都会追加**（放在最后，权重最高）。
# 为什么必须有（§三十一 补记）：`person_noface` 那档原本允许"可以改身体/衣服/头发"，
# 结果 tid 那组把**画风参考图的角色**（粉色头发 + 水手服 + 金鱼/水波）整套搬了进来，只留下源图的姿势与场景。
# 因此范围句只能放宽"渲染质量"，绝不能放宽"这个人是谁、穿什么、画面里有什么"。
IDENTITY_LOCK_CLAUSE = (
    "\n\nIDENTITY AND CONTENT LOCK (applies to every instruction above): the SOURCE image remains the only truth "
    "for WHO and WHAT is in the picture. Keep the same character design, the same hair colour, length and hairstyle, "
    "the same eye colour, the same skin tone, the same outfit (same cut, materials, colours and trims), the same "
    "accessories, the same props, the same background and the same framing. Rendering quality may improve; the "
    "design may not change. Never add anything that is not already in the source and never import anything from the "
    "style reference: not its character, face, hair colour or length, not its eyes, clothing, uniform or accessories, "
    "not its props, animals, plants, water, weather, motifs or background. When a detail could plausibly come from "
    "either image, it must come from the source. If you cannot keep the source's design while improving the "
    "rendering, keep the source's design and change nothing there."
)


# 「重绘编辑范围」（`repaint.scope`）：**不裁切、不贴回**，只用提示词要求模型保留不该动的部分。
# 为什么要有这个（§三十一）：裁切→重绘→贴回那条路（局部重绘）会被模型"重新构图"搞成错位块，
# 实测 5 画风里报废 2 张；改成「整张新图 + 范围要求」后几何天然对齐，不可能拼错边界。
# ⚠️ 范围句只能放宽「渲染质量」，**不能放宽身份/设计/画面内容** —— 每次都会在最后追加 IDENTITY_LOCK_CLAUSE。
REPAINT_SCOPE_CLAUSES = {
    "full": "",
    "person_only":
        "\n\nEDIT SCOPE (this pass): work on the CHARACTER only. Keep the background, furniture, props, floor, "
        "walls, curtains and the lighting pattern exactly as they are in the source: same shapes, same colours, "
        "same values, same placement. Do not repaint, restyle, move, add or remove any background element.",
    "person_noface":
        "\n\nEDIT SCOPE (this pass): improve how the character's body, clothing, hair mass, hands, legwear and "
        "footwear are RENDERED - linework, edge cleanliness, material readability, small malformed details - while "
        "keeping their design exactly as the source has it. Preserve the FACE pixel-faithfully: same eye shape and "
        "size, same lash pattern, same iris colour and internal detail, same brows, nose, mouth, blush, face shading "
        "and position. Do not restyle, beautify, enlarge or redraw the face, and do not treat this as licence to "
        "redesign the character.",
    "details":
        "\n\nEDIT SCOPE (this pass): concentrate on the small worn details - gloved hands and fingers, ribbons and bows, "
        "corset lacing and buckle hardware, garter straps and stocking lace, shoe straps, laces and charms, necklace and "
        "other jewellery. Make each of them read as a continuous, countable, physically anchored structure. "
        "Leave the face, the hair mass, the dress silhouette, the pose and the background otherwise untouched.",
    "lines_only":
        "\n\nEDIT SCOPE (this pass): repair line structure only. Connect contours that are clearly broken and merge stray "
        "fragments so the main curves read as continuous strokes. Do not change colours, shading, materials, shapes, "
        "positions, framing or the amount of detail anywhere in the image.",
}
REPAINT_SCOPE_LABELS = {
    "full": "整图重绘（不额外限制，身份锁仍然生效）",
    "person_only": "只重绘人物（背景保持原样）",
    "person_noface": "人物可精修（脸保持原样，身份/设计不变）",
    "details": "只修细节（手/丝带/系带/袜带/鞋带/项链）",
    "lines_only": "只连通线条（不改色不改内容）",
}


def run_pipeline(paths, steps, firmware=None, out_suffix="-pp", log_callback=None,
                 work_dir=None, resume=True, final_dir=None, style_ref_path=None, style_clauses=None):
    """对一批产物依次跑勾选的后处理步骤；返回最终产物路径列表。

    - `work_dir`：中间产物 + `pipeline-manifest.json` 的目录（默认 <源目录>/pipeline-steps，
      中间文件不再跟成图混在一起）。
    - `resume=True`：清单里已成功的步骤且产物仍在 → 直接复用，只跑失败/未跑的步骤（智能重启）。
    - `final_dir`：**最后一个启用工序**的落盘目录，默认 `data/<YYYYMMDD>/`（发布目录）；
      中间步骤的产物仍写在 `work_dir`（pipeline-steps）。
    - `style_ref_path` / `style_clauses`：画风参考图与画风条款。重绘时 `cfg["reference_mode"]` 决定第二张参考是
      **线锚图（line_anchor）**、**画风图（style）**还是**两者（both）**；**没给 `reference_mode` 时按旧键
      `cfg["dual_reference"]` 兜底（True → line_anchor）** —— 新调用方请显式传 `reference_mode`（GUI 默认 none）；
      给画风图时会追加
      `STYLE_REF_ROLE_IN_REPAINT` + `style_clauses`（Sol 第 3 轮给的英文条款）。
    """
    steps = steps or {}
    log = log_callback or (lambda m: None)
    sources = [p for p in (paths or []) if p and os.path.isfile(p)]
    if not sources:
        return []
    final_dir = final_dir or date_output_dir()
    raw_final_dir = final_dir          # 隔离改写前的原始落盘目录（决定文件名前缀）
    run_id = new_run_id()
    # 测试产出隔离：pytest 会话里（`IMAGE_MAKER_TEST_OUTPUT=1`）默认落盘目录改道到
    # `data/test-result/<日期>/` 且产物名加 `test-` 前缀。**不这么做的话**，用例里
    # `run_pipeline(...)` 不传 `final_dir` 就会往真实日期目录扔一堆
    # `img-233500-a35670-final-sline50.png` 这种废文件（用户 2026-09-23 反馈的正是这个）。
    if output_isolation.test_output_enabled():
        isolated_dir, _ = output_isolation.resolve_output_target(final_dir, "x")
        if isolated_dir != final_dir:
            log(f"[流水线] 测试产出隔离：{final_dir} → {isolated_dir}")
            final_dir = isolated_dir
            if work_dir:
                work_dir, _ = output_isolation.resolve_output_target(work_dir, "x")
    # 中间产物：每次运行独立子目录（多线程并行也不会互相覆盖）
    work_dir = work_dir or os.path.join(final_dir, "pipeline-steps", run_id)
    os.makedirs(work_dir, exist_ok=True)

    manifest = load_manifest(work_dir) if resume else {}
    items = [it for it in (manifest.get("items") or []) if it.get("source") in sources]
    manifest["items"] = items
    by_source = {it.get("source"): it for it in items}

    enabled_order = [k for k in ("repaint", "structure", "contrast", "local", "tone", "ink")
                     if (steps.get(k) or {}).get("enabled")]

    last_key = enabled_order[-1] if enabled_order else ""

    final_paths = []
    for path in sources:
        item = by_source.get(path)
        if item is None:
            item = {"source": path, "steps": [], "final": ""}
            items.append(item)
        done = {s.get("key"): s for s in (item.get("steps") or []) if s.get("status") == "succeeded"}
        base_stem = os.path.splitext(os.path.basename(path))[0]     # 原始来源名：最终产物用它
        current = path
        for key in ("repaint", "structure", "contrast", "local", "tone", "ink"):
            cfg = steps.get(key) or {}
            if not cfg.get("enabled"):
                continue
            prev = done.get(key)
            if resume and prev and prev.get("out") and os.path.isfile(prev["out"]):
                log(f"[工序] {STEP_LABELS[key]}：上次已成功，跳过（复用 {os.path.basename(prev['out'])}）")
                current = prev["out"]
                continue
            log(f"[工序] 开始：{STEP_LABELS[key]}")
            try:
                os.makedirs(work_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(current))[0]
                if key == "repaint":
                    from modules.others.api_backend import generate_image_repaint
                    fw_text = resolve_firmware_text(firmware)
                    # 双参考（源图 + 线锚图）：§⑲ 实测线条连通性最好的配方，默认开
                    extra_refs = []
                    ref_mode = str(cfg.get("reference_mode") or "").strip().lower()
                    if not ref_mode:
                        ref_mode = "line_anchor" if cfg.get("dual_reference", True) else "none"
                    if ref_mode in ("style", "style_neutral", "both") and style_ref_path and os.path.isfile(str(style_ref_path)):
                        extra_refs.append(str(style_ref_path))
                    if ref_mode in ("line_anchor", "both"):
                        from utils.post_process import build_line_anchor as _bla  # noqa: PLC0415
                        anchor_path = os.path.join(work_dir, f"{os.path.splitext(os.path.basename(current))[0]}-lineanchor.png")
                        try:
                            _bla(current, anchor_path)
                            extra_refs.append(anchor_path)
                        except Exception as exc:  # noqa: BLE001
                            log(f"[工序] 线锚图生成失败，跳过: {exc}")
                    if extra_refs:
                        log(f"[工序] 重绘参考：源图 + {len(extra_refs)} 张（{ref_mode}）")
                    # 最后一步 → 直接落 data/<日期>/；否则落 pipeline-steps/（中间产物）
                    if key == last_key:
                        os.makedirs(final_dir, exist_ok=True)
                        sub_dir = os.path.abspath(final_dir)
                        prefix = f"{base_stem}-{run_id}-final-rp"
                    else:
                        # 后端把 save_sub_dir 拼在 data/<日期>/ 下；直接给约定路径，
                        # 不用 os.path.relpath（跨盘符会 ValueError）
                        sub_dir = os.path.abspath(work_dir)
                        prefix = "repaint"
                    repaint_ratio = snapped_aspect_ratio(current)
                    repaint_prompt = fw_text
                    if ref_mode in ("style", "style_neutral", "both") and style_ref_path and os.path.isfile(str(style_ref_path)):
                        repaint_prompt = (repaint_prompt or "") + (
                            STYLE_REF_ROLE_NEUTRAL_IN_REPAINT if ref_mode == "style_neutral"
                            else STYLE_REF_ROLE_IN_REPAINT)
                        if ref_mode in ("style", "both"):
                            repaint_prompt += STYLE_REF_FACE_HAIR_GRAMMAR
                        clauses = [str(c).strip() for c in (style_clauses or []) if str(c).strip()]
                        if clauses:
                            repaint_prompt += "\n\nSTYLE LANGUAGE (from the reference image):\n- " + "\n- ".join(clauses)
                    # 编辑范围：**不裁切、不贴回**，只用提示词要求模型保留不该动的部分（见 §三十一）
                    # 范围句只能放宽「渲染质量」；身份/设计/内容由 IDENTITY_LOCK_CLAUSE 兜底（放在最后 = 权重最高）
                    scope_key = str(cfg.get("scope") or "").strip().lower()
                    scope_clause = REPAINT_SCOPE_CLAUSES.get(scope_key, "")
                    if scope_clause:
                        repaint_prompt = (repaint_prompt or "") + scope_clause
                        log(f"[工序] 重绘编辑范围：{REPAINT_SCOPE_LABELS.get(scope_key, scope_key)}")
                    if scope_key in REPAINT_SCOPE_CLAUSES:
                        repaint_prompt = (repaint_prompt or "") + IDENTITY_LOCK_CLAUSE
                    saved = generate_image_repaint(
                        source_paths=[current], resolution=str(cfg.get("resolution") or "2K"),
                        prompt=repaint_prompt or None, use_detail_suffix=False,
                        aspect_ratio=repaint_ratio,
                        extra_reference_paths=extra_refs,
                        save_sub_dir=sub_dir, file_prefix=prefix)
                    log(f"[工序] 重绘输出比例锁定为 {repaint_ratio}（按源图比例）")
                    out = saved[0] if saved else ""
                    if not out:
                        raise RuntimeError("重绘没有返回图片")
                elif key == "contrast":
                    is_final = key == last_key
                    target_dir = final_dir if is_final else work_dir
                    os.makedirs(target_dir, exist_ok=True)
                    img = imread(current)
                    out_img = local_contrast_enhance(
                        img, amount=float(cfg.get("amount", 0.55)),
                        protect_white=bool(cfg.get("protect_white", True)),
                        radius=int(cfg.get("radius", 12)))
                    out = final_output_path(target_dir, base_stem, run_id, steps, raw_final_dir) if is_final \
                        else os.path.join(target_dir, f"{stem}-contrast.png")
                    imwrite(out, out_img)
                    log(f"[工序] 白色材质局部对比增强 amount={cfg.get('amount', 0.55)} "
                        f"protect_white={cfg.get('protect_white', True)}")
                elif key == "ink":
                    is_final = key == last_key
                    target_dir = final_dir if is_final else work_dir
                    os.makedirs(target_dir, exist_ok=True)
                    img = imread(current)
                    before = line_contrast(img)
                    out_img = ink_lines(img, target_sep=float(cfg.get("target_sep", 10.0)),
                                        amount=float(cfg.get("amount", 1.0)),
                                        max_darken=float(cfg.get("max_darken", 40.0)),
                                        sigma=float(cfg.get("sigma", 6.0)),
                                        protect_light=float(cfg.get("protect_light", 0.0)))
                    out = final_output_path(target_dir, base_stem, run_id, steps, raw_final_dir) if is_final \
                        else os.path.join(target_dir, f"{stem}-ink.png")
                    imwrite(out, out_img)
                    log(f"[工序] 线条加墨 线-底对比 {before:.1f} → {line_contrast(out_img):.1f} "
                        f"（目标分离度 {cfg.get('target_sep', 10.0)}）")
                elif key == "tone":
                    is_final = key == last_key
                    target_dir = final_dir if is_final else work_dir
                    os.makedirs(target_dir, exist_ok=True)
                    img = imread(current)
                    ref_path = str(cfg.get("reference_path") or "")
                    target = cfg.get("target_brightness") or reference_brightness(ref_path)
                    target_sat = cfg.get("target_saturation") or reference_saturation(ref_path)
                    out_img = tone_calibrate(img, target_brightness=target, target_saturation=target_sat,
                                             contrast=float(cfg.get("contrast", 1.00)),
                                             chroma=float(cfg.get("chroma", 1.10)),
                                             sat_target_scale=float(cfg.get("sat_target_scale", 1.0)),
                                             highlight_protect=float(cfg.get("highlight_protect", 0.92)),
                                             highlight_strength=float(cfg.get("highlight_strength", 0.90)),
                                             skin_warm=float(cfg.get("skin_warm", 0.0)))
                    if is_final:
                        out = final_output_path(target_dir, base_stem, run_id, steps, raw_final_dir)
                    else:
                        out = os.path.join(target_dir, f"{stem}-tone.png")
                    imwrite(out, out_img)
                    log(f"[工序] 色调校准 目标亮度 {target if target else '未指定'} / 对比 {cfg.get('contrast', 1.08)} / "
                        f"色度 {cfg.get('chroma', 1.10)}")
                elif key == "structure":
                    is_final = key == last_key
                    target_dir = final_dir if is_final else work_dir
                    os.makedirs(target_dir, exist_ok=True)
                    if is_final:
                        out = final_output_path(target_dir, base_stem, run_id, steps, raw_final_dir)
                    else:
                        out = os.path.join(target_dir, stem
                                       + f"-sline{int(float(cfg.get('strength', DEFAULT_STRUCTURE_STRENGTH)) * 100):02d}.png")
                    info = structure_overlay_file(
                        current, out, strength=float(cfg.get("strength", DEFAULT_STRUCTURE_STRENGTH)),
                        min_len=int(cfg.get("min_len", DEFAULT_MIN_LEN)),
                        darken=float(cfg.get("darken", DEFAULT_DARKEN)),
                        thin=bool(cfg.get("thin", True)),
                        detail_dampen=float(cfg.get("detail_dampen", 0.7)),
                        line_rgb=cfg.get("line_rgb"))
                    log(f"[工序] 结构线叠加 覆盖率 {info['coverage']:.4f}")
                else:
                    regions = [str(r) for r in (cfg.get("regions") or [cfg.get("region") or "hair"])
                               if str(r).strip()]
                    if not regions:
                        regions = ["hair"]
                    # 名字要如实反映"这一版贴回了哪些区域"：以前只取 regions[0]，于是含 4 个区域
                    # 结果的中间文件叫 -local-subject_no_face.png（最后一块还会把它覆盖掉），
                    # 最终产物名也只写第一个区域，光看文件名分不清实际跑了哪几个区域。
                    region_tag = "local-" + (",".join(regions) if len(regions) > 1 else regions[0])
                    is_final = key == last_key
                    target_dir = final_dir if is_final else work_dir
                    os.makedirs(target_dir, exist_ok=True)
                    if is_final:
                        out = final_output_path(target_dir, base_stem, run_id, steps, raw_final_dir)
                    else:
                        out = os.path.join(target_dir, stem + f"-{region_tag}.png")
                    # 多区域：按顺序逐块重绘贴回（例如先整身、再单独修靴子绑带）
                    local_out = out
                    for _idx, _region in enumerate(regions):
                        _tmp_out = local_out if _idx == len(regions) - 1 else os.path.join(
                            work_dir, f"{os.path.splitext(os.path.basename(current))[0]}-local-{_region}.png")
                        _region_key = str(_region)
                        # 默认所有局部区域都限制在主体遮罩内贴回（否则 waist/thigh/shoes 这类带宽区域
                        # 会把背景一起重画，产生矩形拼接带）；只有整张图区域例外，或用配置显式覆盖。
                        if cfg.get("restrict_to_subject") is False:
                            _restrict = False
                        elif cfg.get("restrict_to_subject") is True:
                            _restrict = True
                        else:
                            _restrict = _region_key != "full"
                        _res = str(cfg.get("resolution") or "2K")
                        if _res.upper() == "2K" and cfg.get("detail_boost", True):
                            _res = DETAIL_REGION_RESOLUTION.get(_region_key, _res)
                        local_repaint_composite(
                            current, _tmp_out, region=_region_key, firmware=firmware,
                            resolution=_res,
                            restrict_to_subject=_restrict,
                            qa_callback=lambda info, r=_region_key: log(
                                f"[QA] 区域 {r}：改动像素占比 {info['changed_ratio']:.3f}"
                                f"（补边 {info['padded']}）"),
                            scratch_dir=work_dir,
                            scale=float(cfg.get("scale") or 0) or None,
                            feather=int(cfg.get("feather", DEFAULT_FEATHER)),
                            log_callback=log)
                item["steps"] = [s for s in (item.get("steps") or []) if s.get("key") != key]
                item["steps"].append({"key": key, "status": "succeeded", "out": out})
                item["final"] = out
                current = out
                save_manifest(work_dir, manifest)
            except Exception as exc:  # noqa: BLE001 - 记进清单，便于从这个节点重试
                item["steps"] = [s for s in (item.get("steps") or []) if s.get("key") != key]
                item["steps"].append({"key": key, "status": "failed", "out": "",
                                      "error": f"{type(exc).__name__}: {exc}"})
                # 失败也要交出一个可用的最终产物（把最后成功的状态写成 final，名字带 partial 便于识别），
                # 否则用户拿到的是中间文件、也不知道该从哪一步重试。
                try:
                    os.makedirs(final_dir, exist_ok=True)
                    partial = os.path.join(
                        final_dir,
                        final_product_name(base_stem, run_id, steps).replace(
                            "-final-", f"-final-partial{'+'.join(STEP_TAGS.get(k, k) for k in enabled_order[:enabled_order.index(key)]) or '0'}-").replace("-partial-", "-partial-"))
                    img = imread(current)
                    if img is not None:
                        imwrite(partial, img)
                        item["final"] = partial
                        log(f"⚠️ 已把最后成功的状态写成最终产物（partial）：{os.path.basename(partial)}")
                except Exception as copy_exc:  # noqa: BLE001
                    log(f"⚠️ 生成 partial 最终产物失败：{copy_exc}")
                item["final"] = item.get("final") or current
                save_manifest(work_dir, manifest)
                log(f"❌ 工序失败（可从这一步重试）：{STEP_LABELS[key]} - {type(exc).__name__}: {exc}")
                break
        final_paths.append(current)
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_manifest(work_dir, manifest)
    return final_paths
