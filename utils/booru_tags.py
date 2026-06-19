import re

# 面部质量黑名单：这些标签如果出现在 prompt 中，会导致生成图片面部模糊/打码/遮挡
_FACIAL_DEGRADING_TAGS = {
    "mosaic", "mosaic_censoring", "censored", "censoring",
    "blurry", "blurred", "blurry_face", "blurred_face",
    "blurry_foreground", "blurry_background",
    "pixelated", "pixelation", "pixel_art",
    "lowres", "low_resolution", "jpeg_artifacts",
    "bad_anatomy", "bad_face", "bad_hands",
    "distorted", "deformed", "disfigured",
    "ugly", "poorly_drawn", "sketch",
    "unfinished", "rough", "draft",
    "censored_artwork", "bar_censor", "mosaic_censor",
    "decensored", "uncensored",
    "missing_face", "faceless", "hidden_face",
    "mask", "face_mask", "covered_face",
    "mouth_cover", "eye_cover",
}

# 面部质量黑名单关键词子串匹配（更宽松的过滤）
_FACIAL_DEGRADING_SUBSTRINGS = [
    "mosaic", "censor", "blur", "pixelat",
    "low_res", "lowres", "distort", "deform",
    "bad_face", "missing_face", "faceless",
    "hidden_face", "covered_face",
]


def filter_facial_degrading_tags(tag_list):
    """过滤掉可能导致面部模糊/打码/遮挡的标签。返回过滤后的新列表。"""
    if not tag_list:
        return tag_list
    filtered = []
    for tag in tag_list:
        tag_lower = str(tag).strip().lower().replace(" ", "_").replace("-", "_")
        # 精确匹配黑名单
        if tag_lower in _FACIAL_DEGRADING_TAGS:
            continue
        # 子串模糊匹配
        blocked = False
        for substr in _FACIAL_DEGRADING_SUBSTRINGS:
            if substr in tag_lower:
                blocked = True
                break
        if blocked:
            continue
        filtered.append(tag)
    return filtered


def filter_facial_degrading_from_text(text: str) -> str:
    """从文本描述中移除面部模糊/打码/遮挡相关短语。返回清理后的文本。"""
    if not text or not isinstance(text, str):
        return text
    # 移除常见的面部模糊/打码描述短语
    patterns = [
        r'\b(blurred|blurry|pixelated?|mosaic(ed)?|censored)\s+(face|facial|features?)\b',
        r'\b(face|facial|features?)\s+(is|are|appears?|looks?)\s+(blurred|blurry|pixelated?|mosaic(ed)?|censored)\b',
        r'\b(face|facial)\s+(censorship|mosaic|blur|blurring|pixelation)\b',
        r'\b(censored|mosaic(ed)?)\s+(artwork|image|picture|photo)\b',
        r'\bwith\s+(a\s+)?(blurred|blurry|mosaic(ed)?|pixelated?|censored)\s+(face|facial)\b',
        r'\b(due|because)\s+to\s+(censorship|mosaic|blurring|pixelation)\b[^.]*\.?',
        r'\bthe\s+(face|facial\s+area)\s+is\s+(obscured|hidden|covered)\s+(by|with)\s+(a\s+)?(mosaic|blur|pixelation|censor)\b',
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    # 清理多余空格和标点
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(r'\s,', ',', result)
    result = re.sub(r'\s\.', '.', result)
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'\.\s*\.', '.', result)
    return result.strip()


def normalize_booru_tags(booru_tags, limit=30, output_style="underscore"):
    if isinstance(booru_tags, str):
        raw_items = [booru_tags]
    elif isinstance(booru_tags, list):
        raw_items = booru_tags
    else:
        raw_items = []
    normalized_tags = []
    seen = set()
    for item in raw_items:
        parts = str(item).split(",")
        for part in parts:
            tag = str(part).strip().lower()
            if not tag:
                continue
            tag = re.sub(r"[_\s]+", " ", tag).strip()
            if output_style == "space":
                tag = re.sub(r"\s+", " ", tag).strip()
            else:
                tag = tag.replace(" ", "_")
                tag = re.sub(r"_+", "_", tag).strip("_")
            if not tag or tag in seen:
                continue
            seen.add(tag)
            normalized_tags.append(tag)
            if len(normalized_tags) >= limit:
                return normalized_tags
    return normalized_tags
