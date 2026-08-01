# -*- coding: utf-8 -*-
"""画风预设（config-styles.json）统一样式解析与「艺术风格参考图」模式组装。

config-styles.json 条目兼容两种格式：
- 旧格式：{"style_name": "指令文本字符串"}
- 新格式：{"style_name": {"prompt": "指令文本", "ref_image": "参考图路径(可选)"}}

风格参考图模式：
- MODE_OFF:        关闭（不使用参考图）
- MODE_HEAD:       头部插入（样式指令 + 风格参考指令块，参考图作为附件）
- MODE_PRIORITY:   参考优先（样式指令压缩为精简版，参考图主导画风）
- MODE_INTERLEAVE: 图文交错（风格参考指令紧跟参考图之后）
"""
import os

from utils.prompt_loader import render_prompt_file

STYLE_REF_PROMPT_FILE = "style-ref-image.md"
STYLE_REF_FALLBACK_PROMPT = (
    "You are now in Art Style Reference Mode. The attached image(s) are STYLE REFERENCES ONLY: "
    "extract only the artistic painting style (line art, coloring and shading technique, lighting, "
    "color palette, rendering conventions, detail/finish level). NEVER copy the subject, character, "
    "identity, hairstyle, outfit, pose, scene, composition or any text from the reference image. "
    "The subject and composition must come entirely from the text prompt, rendered in the extracted style."
)

# 参考优先模式的头部声明（参考图为主，文字为辅）
REF_PRIORITY_PREAMBLE = (
    "## REFERENCE-PRIORITY MODE (参考优先模式)\n"
    "The attached reference image is the PRIMARY and authoritative source of the art style. "
    "Replicate its rendering, coloring, lighting, line work and finish as closely as possible.\n"
    "The condensed style spec below is SECONDARY and lists only essential structural constraints. "
    "Where the spec and the reference image disagree, ALWAYS follow the reference image."
)

MODE_OFF = "off"
MODE_HEAD = "head"
MODE_PRIORITY = "priority"
MODE_INTERLEAVE = "interleave"

STYLE_REF_MODES = [
    (MODE_OFF, "关闭"),
    (MODE_HEAD, "头部插入"),
    (MODE_PRIORITY, "参考优先"),
    (MODE_INTERLEAVE, "图文交错"),
]


def normalize_style_entry(entry):
    """把任意样式条目归一化为 {"prompt": str, "ref_image": str, "prompt_compressed": str}，兼容新旧格式。"""
    if isinstance(entry, str):
        return {"prompt": entry, "ref_image": "", "prompt_compressed": ""}
    if isinstance(entry, dict):
        prompt = entry.get("prompt") or entry.get("text") or entry.get("instructions") or ""
        ref = entry.get("ref_image") or entry.get("ref_image_path") or entry.get("image") or ""
        compressed = entry.get("prompt_compressed") or entry.get("compressed") or ""
        return {"prompt": str(prompt), "ref_image": str(ref or ""), "prompt_compressed": str(compressed or "")}
    return {"prompt": "", "ref_image": "", "prompt_compressed": ""}


def style_prompt(styles, name):
    """取样式指令文本（兼容新旧格式）。"""
    return normalize_style_entry((styles or {}).get(name))["prompt"]


def style_prompt_compressed(styles, name):
    """取样式压缩版指令（参考优先模式用；可能为空）。"""
    return normalize_style_entry((styles or {}).get(name))["prompt_compressed"]


def style_ref_image(styles, name):
    """取样式参考图路径（兼容新旧格式）。"""
    return normalize_style_entry((styles or {}).get(name))["ref_image"]


def build_style_entry(prompt, ref_image="", prompt_compressed=""):
    """构造新格式样式条目：{"prompt": ..., "ref_image": ..., "prompt_compressed": ...}（空字段省略）。"""
    entry = {"prompt": str(prompt or "")}
    if ref_image:
        entry["ref_image"] = str(ref_image)
    if prompt_compressed:
        entry["prompt_compressed"] = str(prompt_compressed)
    return entry


def ref_image_valid(path):
    """参考图是否可用：路径非空且文件存在。"""
    return bool(path) and os.path.isfile(str(path))


def compress_style_text(text, head_chars=900, tail_chars=800):
    """把长样式指令压缩为精简版（保留头部总纲 + 尾部约束/结构规则），用于参考优先模式。"""
    text = str(text or "")
    if len(text) <= head_chars + tail_chars:
        return text
    head = text[:head_chars].rsplit("\n", 1)[0]
    tail = text[-tail_chars:].lstrip("\n")
    return head + "\n...(其余样式细则省略，以参考图为准)...\n" + tail


def build_style_ref_instruction(image_count=1):
    """读取「艺术风格参考图」指令模板；模板缺失时使用内置兜底指令。"""
    try:
        return render_prompt_file(STYLE_REF_PROMPT_FILE, {"image_count": str(int(image_count))}).strip()
    except Exception:
        return STYLE_REF_FALLBACK_PROMPT


def assemble_style_instructions(mode, style_text, has_ref_image, prompt_compressed=""):
    """根据模式组装头部指令与图后指令。

    返回 (head_instructions, post_instructions)：
    - head_instructions: 放到请求文本头部
    - post_instructions: 放到所有参考图之后（图文交错用）
    参考图无效或模式为关闭时，只返回样式指令原文。
    参考优先模式优先使用配置里已固化的 prompt_compressed，缺失时回退到本地启发式压缩。
    """
    style_text = str(style_text or "")
    if mode == MODE_OFF or not has_ref_image:
        return style_text, ""
    ref_block = build_style_ref_instruction()
    if mode == MODE_HEAD:
        head = f"{ref_block}\n\n{style_text}" if style_text else ref_block
        return head, ""
    if mode == MODE_PRIORITY:
        compressed = str(prompt_compressed or "").strip() or compress_style_text(style_text)
        head = f"{REF_PRIORITY_PREAMBLE}\n\n{ref_block}\n\n{compressed}"
        return head, ""
    if mode == MODE_INTERLEAVE:
        return style_text, ref_block
    return style_text, ""


def build_ref_gen_params(styles, style_name, mode):
    """统一组装「风格参考图」生图参数。

    返回 (head_instructions, post_instructions, ref_image_paths)：
    - head_instructions: 放请求文本头部（样式指令 + 参考指令块 / 参考优先压缩版）
    - post_instructions: 放所有参考图之后（图文交错用）
    - ref_image_paths:   需要作为内联附件传入的参考图路径列表
    参考图无效（未配置或文件不存在）或模式为关闭时，返回 (样式原文, "", [])。
    """
    entry = normalize_style_entry((styles or {}).get(style_name))
    has_ref = ref_image_valid(entry["ref_image"])
    if not has_ref:
        mode = MODE_OFF
    head, post = assemble_style_instructions(mode, entry["prompt"], has_ref, entry["prompt_compressed"])
    ref_paths = [entry["ref_image"]] if (has_ref and mode != MODE_OFF) else []
    return head, post, ref_paths


def save_styles_file(path, styles):
    """写回 config-styles.json。"""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(styles, f, ensure_ascii=False, indent=4)
