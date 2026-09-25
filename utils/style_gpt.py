# -*- coding: utf-8 -*-
"""gpt-image 系列的「短版画风说明」（`prompt_gpt`）：转换规则、校验与取用。

背景与实测依据见 `docs/gpt-image-tid-style/`：
- gpt-image 通道把画风说明与主体拼成**一条** prompt，长文字会抢走参考图的话语权；
- 实测最有用的形态是**结构化字段版**（8 个字段、约 250 字符），比「保留什么/换掉什么」式长段落更贴参考图
  （HSV 0.644 vs 0.287），也比压缩版说明书更稳。
- `prompt_gpt` 只描述**画法**，绝不允许出现主体/角色/姿势/场景词，否则会把参考图角色的发色瞳色带进产物。

取用优先级（生图链路统一走 `resolve_style_prompt`）：
1. 显式参数 `prompt_text`
2. gpt-image 通道（api_type 归一化后含 gpt）且存在 `prompt_gpt` → 用它
3. gpt-image 通道但缺 `prompt_gpt` → 退回 `prompt_compressed`（压缩版说明书，比全量好）
4. 其它通道 → 全量 `prompt`

画风条目还可带 `repaint_clauses`（渲染语言条款）：手写优先，缺失时由 `resolve_style_clauses`
按 `prompt_gpt` 字段**确定性派生**（见本文件 §渲染语言条款），保证任何画风在首图与重绘两条链路上
都拿到同等的"线条/配色/不要画成什么"约束。

本模块不依赖 PyQt，供 GUI / CLI / 转换工具共用。
"""
import os
import re

from utils.prompt_loader import read_prompt_file
from utils.styles import normalize_style_entry

SYSTEM_PROMPT_FILE = "style-gpt-convert-system.md"
USER_TEMPLATE_FILE = "style-gpt-convert-user.md"

# 8 个字段就是转换规则的核心：只描述画法，不描述内容
FIELD_KEYS = (
    "Palette",
    "Lighting",
    "Brushwork",
    "Edges",
    "Texture",
    "Composition density",
    "Detail level",
    "Avoid",
)

PROMPT_GPT_PREFIX = (
    "Create a new image in the visual style described below.\n\n"
)
PROMPT_GPT_SUFFIX = (
    "\n\nThis style description controls the rendering only. "
    "The scene below controls all content.\n{subject}"
)

MIN_CHARS = 150
MAX_CHARS = 680
MAX_FIELD_CHARS = 160

# 主体/角色/构图泄漏词：一旦出现在画风说明里，说明模型把参考图主体写进去了。
# 注意别把渲染层常用词误判：uniform(均匀光)、skin(材质)、face(可能指"表面")、lights(高光) 都不列入；
# hair/eyes 用带边界的正则单独判（避免 lights 命中 eyes）。
SUBJECT_BAN_WORDS = (
    "girl", "boy", "woman", "man", "character", "hairstyle", "eyelash", "eyelashes",
    "pose", "outfit", "costume", "dress", "skirt", "braid", "ponytail",
    "smile", "blush", "portrait", " she ", " her ", " he ", " his ",
    "少女", "女孩", "角色", "人物", "发色", "瞳色", "表情", "姿势", "服装", "连衣裙",
)
SUBJECT_BAN_PATTERNS = (
    r"\bhair\b",
    r"\beyes?\b",
    r"\biris(es)?\b",
    r"\beye\s+colou?r\b",
    r"\bhair\s+colou?r\b",
)
FIELD_LINE_RE = re.compile(rf"^\s*({'|'.join(re.escape(k) for k in FIELD_KEYS)})\s*[:：]\s*(.+?)\s*$")
ANY_FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /-]{2,24})\s*[:：]\s*(.+?)\s*$")


def is_gpt_image_api_type(api_type) -> bool:
    """api_type / 站点名是否是 gpt-image 通道（大小写、`-`/`_` 无关）。"""
    text = str(api_type or "").strip().lower().replace("-", "_").replace(".", "_")
    return "gpt" in text


# --------------------------------------------------------------- 渲染语言条款（repaint_clauses）
#
# 背景：画风条目上有个可选字段 `repaint_clauses`（一组英文"渲染语言"条款），链路里用在两处：
#   1. 首图生成：`build_gpt_image_request(extra_clauses=…)` → 追加 `RENDERING LANGUAGE (follow exactly):` 段；
#   2. 重绘：`run_pipeline(style_clauses=…)` → 追加 `STYLE LANGUAGE (from the reference image):` 段。
# 它最初是 **tinkle 那一条画风单独手写**的（2026-09-23，见 BEST-PIPELINE §十三），
# 于是其余画风一条都没有 —— 首图/重绘都少了"线条怎么画、颜色锚点在哪、不要画成什么"的约束。
# 这里给没有手写条款的画风**按它自己的 `prompt_gpt` 字段确定性派生**一份（不新增任何内容，
# 只把 8 个字段值改写成祈使句），保证任何画风在两条链路上都拿到同等的渲染语言约束。
CLAUSE_TEMPLATES = {
    "Palette": ("Build the palette from {v}; keep those anchors present in shadows and midtones so the "
                "image never turns pale, grey or washed out."),
    "Lighting": "Light the image with {v}, keeping clear highlight and shadow separation.",
    "Brushwork": "Render surfaces with {v}.",
    "Edges": ("Render contours with {v}; taper every stroke toward its endpoint, keep primary silhouette "
              "lines heavier than interior detail, and never use uniform mechanical line weight."),
    "Texture": "Render material texture as {v}.",
    "Composition density": "Keep the composition density at {v}.",
    "Detail level": "Render detail as {v}.",
    "Avoid": "Avoid {v}.",
}
# 与字段无关、但所有画风都适用的两条（对应 tinkle 手写条款里最起作用的那两条）
UNIVERSAL_CLAUSES = (
    "Preserve real chroma in midtones and shadows; do not desaturate the whole image towards white.",
    "Keep important contours crisp while atmospheric or distant elements use softer edges; "
    "no global haze, bloom or washed-out lighting.",
)
CLAUSE_MAX_CLAUSES = 12
CLAUSE_MAX_CHARS = 1600


def _sentence_value(text: str) -> str:
    """把字段值放进句子中段时收敛首字母大小写（别把 CMYK / PBR 这类缩写改坏）。"""
    value = str(text or "").strip()
    if len(value) >= 2 and value[0].isupper() and value[1].islower():
        return value[0].lower() + value[1:]
    return value


def derive_style_clauses(prompt_gpt: str) -> list:
    """按 `prompt_gpt` 的 8 个字段派生渲染语言条款（确定性、不调模型）。"""
    fields = parse_fields(prompt_gpt)
    clauses = []
    for key in FIELD_KEYS:
        value = _sentence_value(str(fields.get(key) or "").strip().rstrip("."))
        template = CLAUSE_TEMPLATES.get(key)
        if not value or not template:
            continue
        clause = template.format(v=value)
        clauses.append(clause)
        if key == "Palette":          # 颜色锚点之后紧跟两条通用条款（保住彩度/避免全局发灰）
            clauses.extend(UNIVERSAL_CLAUSES)
    # 去重 + 长度护栏（保证不会因为条款把提示词顶到硬上限）
    seen, out = set(), []
    for clause in clauses:
        text = " ".join(str(clause).split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    while len(" ".join(out)) > CLAUSE_MAX_CHARS and len(out) > 2:
        out.pop()
    return out[:CLAUSE_MAX_CLAUSES]


def resolve_style_clauses(entry) -> tuple:
    """取画风的渲染语言条款：手写的 `repaint_clauses` 优先，缺了就按 `prompt_gpt` 派生。

    返回 `(clauses, source)`，`source` ∈ `entry`（手写）/ `derived`（派生）/ `none`（没有画风说明）。
    """
    if not isinstance(entry, dict):
        return [], "none"
    written = [str(c).strip() for c in (entry.get("repaint_clauses") or []) if str(c).strip()]
    if written:
        return written, "entry"
    body = str(entry.get("prompt_gpt") or entry.get("prompt_short") or entry.get("gpt_prompt") or "").strip()
    if not body:
        return [], "none"
    derived = derive_style_clauses(body)
    return (derived, "derived") if derived else ([], "none")


def resolve_neutral_repaint_clauses(entry) -> tuple:
    """只描述画法，不读取参考图的具体配色、构图或角色内容。"""
    if not isinstance(entry, dict):
        return [], "none"
    body = str(entry.get("prompt_gpt") or entry.get("prompt_short") or entry.get("gpt_prompt") or "").strip()
    if not body:
        return [], "none"
    fields = parse_fields(body)
    clauses = [
        "Use the reference only for abstract rendering mechanics. Ignore its exact palette, hue distribution, "
        "colour temperature, saturation pattern and value placement; preserve the source image's own local colours.",
        "Do not import any recognisable motif, object, facial construction, eye design, hairstyle, garment detail, "
        "accessory or composition from the reference.",
    ]
    templates = {
        "Brushwork": "Match only the brushwork mechanics: {v}.",
        "Edges": "Match only the edge and stroke behaviour: {v}.",
        "Texture": "Match only the non-semantic surface treatment: {v}.",
        "Avoid": "Avoid these rendering defects where they do not conflict with the source: {v}.",
    }
    for key in ("Brushwork", "Edges", "Texture", "Avoid"):
        value = _sentence_value(str(fields.get(key) or "").strip().rstrip("."))
        if value:
            clauses.append(templates[key].format(v=value))
    clauses.extend((
        "Preserve the source's hair, eye, skin, clothing, accessory and background colours exactly; style transfer "
        "must not recolour any identity-bearing feature.",
        "Improve continuity only on clearly intended structural contours; keep secondary painterly strokes soft, "
        "irregular and locally coloured.",
    ))
    return clauses, "neutral-derived"


def style_prompt_gpt(styles, name) -> str:
    """取样式的 gpt-image 短版说明（`prompt_gpt`），可能为空。"""
    entry = (styles or {}).get(name)
    if isinstance(entry, dict):
        value = entry.get("prompt_gpt") or entry.get("prompt_short") or entry.get("gpt_prompt") or ""
        return str(value or "")
    return ""


def resolve_style_prompt(styles, name, api_type="", fallback=None):
    """按通道选画风说明：gpt-image 通道优先用 `prompt_gpt`，其次压缩版，其它通道用全量。

    `fallback` 为全量未命中时的兜底文本（默认空串）。
    """
    entry = normalize_style_entry((styles or {}).get(name))
    full = entry["prompt"]
    compressed = entry["prompt_compressed"]
    gpt = style_prompt_gpt(styles, name)
    if is_gpt_image_api_type(api_type):
        return gpt or compressed or full
    return full or fallback or compressed


def style_ref_image_for_api(styles, name, api_type=""):
    """取该画风要用的参考图（gpt-image 与其它通道目前同源，留出扩展点）。"""
    return normalize_style_entry((styles or {}).get(name))["ref_image"]


def compose_prompt_gpt(body: str, subject: str) -> str:
    """把字段版画风说明与主体拼成最终 gpt-image 提示词（单条）。"""
    core = (body or "").strip()
    return (PROMPT_GPT_PREFIX + core + PROMPT_GPT_SUFFIX.format(subject=str(subject or "").strip())).strip()


def parse_fields(text: str) -> dict:
    """把字段版文本解析成 {字段名: 值}；未知字段忽略，保持大小写不敏感。"""
    fields = {}
    lowered = {k.lower(): k for k in FIELD_KEYS}
    for raw in str(text or "").splitlines():
        m = ANY_FIELD_RE.match(raw)
        if not m:
            continue
        key, value = m.group(1).strip().lower(), m.group(2).strip()
        if key in lowered and value:
            fields[lowered[key]] = value
    return fields


def validate_prompt_gpt(text: str):
    """校验字段版画风说明。返回 (ok: bool, errors: list[str])。"""
    errors = []
    body = str(text or "").strip()
    if not body:
        return False, ["内容为空"]
    fields = parse_fields(body)
    missing = [k for k in FIELD_KEYS if k not in fields]
    if missing:
        errors.append("缺字段: " + ", ".join(missing))
    for key, value in fields.items():
        if len(value) > MAX_FIELD_CHARS:
            errors.append(f"{key} 过长({len(value)}>{MAX_FIELD_CHARS})")
    total = len(body)
    if total < MIN_CHARS:
        errors.append(f"太短({total}<{MIN_CHARS})")
    if total > MAX_CHARS:
        errors.append(f"太长({total}>{MAX_CHARS})")
    low = body.lower()
    banned = [w for w in SUBJECT_BAN_WORDS if w in low]
    for pattern in SUBJECT_BAN_PATTERNS:
        for m in re.finditer(pattern, low):
            banned.append(m.group(0).strip())
    if banned:
        errors.append("疑似写了主体/角色词: " + ", ".join(sorted(set(banned))))
    # 只允许 8 个字段行 + 可选空行
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not FIELD_LINE_RE.match(line):
            errors.append(f"非字段行: {line[:40]}")
    return (not errors), errors


def format_prompt_gpt(fields: dict) -> str:
    """按固定顺序把字段渲染成规范文本。"""
    return "\n".join(f"{k}: {str(fields.get(k, '')).strip()}" for k in FIELD_KEYS)


def repair_prompt_gpt(text: str) -> str:
    """把模型的自由输出尽量规整成 8 字段格式：能解析出字段就重排；否则原样返回。"""
    fields = parse_fields(text)
    if len(fields) >= max(4, len(FIELD_KEYS) - 2):
        return format_prompt_gpt(fields)
    return str(text or "").strip()


def compact_fields(fields: dict, max_value_chars: int = 70) -> dict:
    """字段过长时做保守截断（逗号/空格边界收口），避免整条被长度校验拒收。"""
    out = {}
    for key, value in (fields or {}).items():
        text = str(value or "").strip().rstrip(".")
        if len(text) <= max_value_chars:
            out[key] = text
            continue
        cut = text[:max_value_chars]
        if "," in cut:
            cut = cut.rsplit(",", 1)[0]
        elif " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        out[key] = cut.strip().rstrip(",")
    return out


def enforce_total_length(text: str, limit: int = MAX_CHARS) -> str:
    """按行边界把整段压到 limit 以内；单行仍超长时再在逗号处收口。"""
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    if len("".join(lines)) <= limit:
        return "\n".join(lines)
    budget = max(40, (limit - 20) // max(1, len(lines)))
    trimmed = []
    for ln in lines:
        if ":" in ln:
            key, value = ln.split(":", 1)
            key = key.strip()
            value = value.strip()
            if len(value) > budget:
                value = value[:budget]
                if "," in value:
                    value = value.rsplit(",", 1)[0]
                elif " " in value:
                    value = value.rsplit(" ", 1)[0]
                value = value.strip().rstrip(",")
            trimmed.append(f"{key}: {value}")
        else:
            trimmed.append(ln[:budget])
    return "\n".join(trimmed)


def repair_prompt_gpt_lenient(text: str) -> str:
    """比 repair_prompt_gpt 更积极：解析字段 → 压缩 → 强制收口到长度上限。"""
    fields = parse_fields(text)
    if not fields:
        return enforce_total_length(str(text or "").strip())
    body = format_prompt_gpt(compact_fields(fields))
    if len(body) > MAX_CHARS:
        body = enforce_total_length(body)
    return body


def build_conversion_prompts(style_name: str, style_text: str, has_image: bool = False,
                             image_stats_text: str = ""):
    """返回 (system_prompt, user_prompt)，供文本模型做风格→字段版转换。

    三种「看得到画风」的强度：
    - `image_stats_text` 非空：提示词里带上参考图的**客观统计**（亮度/饱和/留白/色相分布/主色/边缘密度），
      由 `tools/style_ref_stats.py` 算出，不依赖模型视觉能力，最省时且最稳（推荐）；
    - `has_image=True`：把参考图作为多模态输入一起发（慢，且部分模型不支持）；
    - 两者都空：只按长说明书文字转换。
    """
    try:
        system = read_prompt_file(SYSTEM_PROMPT_FILE).strip()
    except FileNotFoundError:
        system = _FALLBACK_SYSTEM
    try:
        user_tpl = read_prompt_file(USER_TEMPLATE_FILE)
    except FileNotFoundError:
        user_tpl = _FALLBACK_USER
    if image_stats_text:
        hint = (
            "The measured statistics of the reference image are given below. They are authoritative for "
            "colour, brightness, saturation, empty-space ratio and edge density — the Palette / Lighting / "
            "Composition density / Edges / Detail level values must agree with them. The text spec is "
            "secondary: use it for material and technique rules that the numbers cannot express.\n\n"
            + str(image_stats_text).strip()
        )
    elif has_image:
        hint = "Describe what the attached image actually looks like; treat the text spec as secondary."
    else:
        hint = "No image is attached: rely on the text spec only."
    user = (
        user_tpl.replace("{style_name}", str(style_name or ""))
        .replace("{image_hint}", hint)
        .replace("{style_text}", str(style_text or ""))
    )
    return system, user


def to_data_url(path: str) -> str:
    """把本地图片读成 data URL（供文本模型多模态输入）。"""
    import base64
    import mimetypes

    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{payload}"


_FALLBACK_SYSTEM = (
    "You convert a long anime art-style specification into a compact style spec for OpenAI image models "
    "(gpt-image). Output ONLY the 8 fields below, in this order, one per line, no extra text:\n"
    + "\n".join(f"{k}: <3-14 words>" for k in FIELD_KEYS)
    + "\nRules: describe rendering only; never mention subjects, characters, faces, hair, eyes, poses, "
      "outfits or scenes; 'Avoid' lists rendering-level anti-patterns; keep the whole thing under 300 characters."
)
_FALLBACK_USER = "Style name: {style_name}\n\nFull style specification:\n{style_text}"
