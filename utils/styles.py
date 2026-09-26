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
    """把任意样式条目归一化为
    {"prompt": str, "ref_image": str, "prompt_compressed": str, "prompt_gpt": str,
     "enabled": bool, "motif_clauses": list[str], "motif_enabled": bool}，兼容新旧格式。
    旧条目缺少 `enabled` 时视为启用；装饰母题缺省为关闭。

    `prompt_gpt` 是给 gpt-image 通道用的「短版字段式画风说明」（见 utils/style_gpt.py 与
    docs/gpt-image-tid-style/）：gpt-image 会把画风说明与主体拼成一条 prompt，长说明书会
    抢走参考图的话语权，所以该通道优先用这一份。
    """
    if isinstance(entry, str):
        return {"prompt": entry, "ref_image": "", "prompt_compressed": "", "prompt_gpt": "",
                "enabled": True, "motif_clauses": [], "motif_enabled": False}
    if isinstance(entry, dict):
        prompt = entry.get("prompt") or entry.get("text") or entry.get("instructions") or ""
        ref = entry.get("ref_image") or entry.get("ref_image_path") or entry.get("image") or ""
        compressed = entry.get("prompt_compressed") or entry.get("compressed") or ""
        gpt_prompt = entry.get("prompt_gpt") or entry.get("prompt_short") or entry.get("gpt_prompt") or ""
        return {
            "prompt": str(prompt),
            "ref_image": str(ref or ""),
            "prompt_compressed": str(compressed or ""),
            "prompt_gpt": str(gpt_prompt or ""),
            "enabled": entry.get("enabled", True) is not False,
            "motif_clauses": [str(v).strip() for v in (entry.get("motif_clauses") or [])
                              if str(v).strip()][:4],
            "motif_enabled": entry.get("motif_enabled", False) is True,
        }
    return {"prompt": "", "ref_image": "", "prompt_compressed": "", "prompt_gpt": "",
            "enabled": True, "motif_clauses": [], "motif_enabled": False}


def style_motif_prompt(styles, name) -> str:
    """返回只用于首次生成的低权重装饰母题；重绘链路不应使用。"""
    entry = normalize_style_entry((styles or {}).get(name))
    if not entry["motif_enabled"] or not entry["motif_clauses"]:
        return ""
    return motif_prompt_from_clauses(entry["motif_clauses"])


def motif_prompt_from_clauses(clauses) -> str:
    """把母题词汇包装成低权重、可省略且不得改写内容的生成条款。"""
    clean = [str(v).strip() for v in (clauses or []) if str(v).strip()][:4]
    if not clean:
        return ""
    motifs = "; ".join(clean)
    return (
        "OPTIONAL STYLE MOTIFS: When compatible with the requested scene, add only a few small, "
        "subordinate background or edge accents from this vocabulary: " + motifs + ". "
        "Omit them when they compete with the subject; never alter identity, outfit, pose, props, "
        "setting or composition to accommodate them."
    )


def style_enabled(styles, name) -> bool:
    """画风是否对生成/测试列表可见；缺字段兼容为启用。"""
    return bool(normalize_style_entry((styles or {}).get(name))["enabled"])


def enabled_style_names(styles) -> list[str]:
    """按配置原顺序返回启用的画风名。管理界面仍应展示全部条目。"""
    return [str(name) for name in (styles or {}) if style_enabled(styles, name)]


def style_prompt(styles, name):
    """取样式指令文本（兼容新旧格式）。"""
    base = normalize_style_entry((styles or {}).get(name))["prompt"]
    motif = style_motif_prompt(styles, name)
    return "\n\n".join(v for v in (base, motif) if v)


def style_prompt_compressed(styles, name):
    """取样式压缩版指令（参考优先模式用；可能为空）。"""
    base = normalize_style_entry((styles or {}).get(name))["prompt_compressed"]
    motif = style_motif_prompt(styles, name)
    return "\n\n".join(v for v in (base, motif) if v)


def style_prompt_gpt(styles, name):
    """取 gpt-image 通道用的短版字段式说明（可能为空）。"""
    return normalize_style_entry((styles or {}).get(name))["prompt_gpt"]


def style_ref_image(styles, name):
    """取样式参考图路径（兼容新旧格式）。"""
    return normalize_style_entry((styles or {}).get(name))["ref_image"]


def build_style_entry(prompt, ref_image="", prompt_compressed="", prompt_gpt="", enabled=True,
                      motif_clauses=None, motif_enabled=False):
    """构造新格式样式条目（空字段省略）。"""
    entry = {"prompt": str(prompt or ""), "enabled": bool(enabled)}
    if ref_image:
        entry["ref_image"] = str(ref_image)
    if prompt_compressed:
        entry["prompt_compressed"] = str(prompt_compressed)
    if prompt_gpt:
        entry["prompt_gpt"] = str(prompt_gpt)
    motifs = [str(v).strip() for v in (motif_clauses or []) if str(v).strip()][:4]
    if motifs:
        entry["motif_clauses"] = motifs
        entry["motif_enabled"] = bool(motif_enabled)
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


# 「画风参考图 + 用户内容图」同时存在时的角色分工说明（gpt-image 通道会把所有图当"要编辑的原图"，
# 不写清楚它会把画风参考图的角色/服装/构图一起搬进来——见 docs/gpt-image-tid-style/README.md §4.18/§4.19）
STYLE_REF_ROLE_INSTRUCTION = (
    "IMAGE ROLES (read carefully):\n"
    "- The FIRST {content_count} image(s) are CONTENT REFERENCES: they define the subject, character, pose, "
    "outfit design, props, scene and composition.\n"
    "- The LAST image is the ART-STYLE REFERENCE ONLY: use its palette, colour temperature, lighting, "
    "brushwork, edge treatment, line character, texture and overall rendering grammar.\n"
    "Do NOT copy the style reference's character, face, hairstyle, hair colour, eye colour, outfit, pose, "
    "props, background or composition. Do NOT copy the content reference's photographic rendering. "
    "If the two disagree about content, the content references win; the style reference never overrides "
    "content, identity, facial drawing or composition."
)


def compose_style_prompt(style_text, user_prompt, style_ref_attached=False, content_image_count=0):
    """把「画风指令」与「用户提示词」拼成最终提示词，并在同时挂了两类图时补上角色分工说明。

    - 只挂画风参考图（没有内容图）：画风指令 + 用户提示词 + 一句"场景/构图只能来自文字"声明
      （否则强画风参考图会把它的场景/道具一起带进来——实测 waterink-style 的参考图是室内、
       文本写的是海边，产物被拉进室内）。
    - 同时挂内容图 + 画风参考图：补 STYLE_REF_ROLE_INSTRUCTION，明确"前 N 张是内容、最后一张是画风"。
    """
    style = str(style_text or "").strip()
    user = str(user_prompt or "").strip()
    parts = []
    if style_ref_attached and int(content_image_count) > 0:
        parts.append(STYLE_REF_ROLE_INSTRUCTION.replace("{content_count}", str(int(content_image_count))))
    if style:
        parts.append(style)
    if user:
        parts.append(user)
    if style_ref_attached and int(content_image_count) <= 0 and user:
        parts.append(STYLE_REF_SCENE_FROM_TEXT)
    return "\n\n".join(parts)


# 「只挂画风参考图」时必须声明场景来自文字（否则参考图的场景/道具会被带进来）
STYLE_REF_SCENE_FROM_TEXT = (
    "The scene, setting, props, character design and composition must come ENTIRELY from the text above. "
    "The reference image is a style sample only: do not copy its room, furniture, props, background layout, "
    "camera angle or subject; borrow only its palette, lighting, brushwork and rendering grammar."
)


def ordered_reference_images(user_images, style_ref_path="", style_ref_attached=False):
    """参考图提交顺序：内容图在前，画风参考图**放最后**（与 STYLE_REF_ROLE_INSTRUCTION 的措辞一致）。"""
    images = [str(p) for p in (user_images or []) if p]
    if style_ref_attached and style_ref_path:
        images.append(str(style_ref_path))
    return images


# 「服饰部件参考图 + 画风参考图」同时存在时的角色分工（服饰素材采集 → 穿着这套衣服的角色）
GARMENT_REF_ROLE_INSTRUCTION = (
    "IMAGE ROLES (read carefully):\n"
    "- The FIRST {part_count} image(s) are GARMENT REFERENCES: product shots of clothing pieces. Render EXACTLY "
    "these designs on the character - the same garment type, silhouette, colour, print/pattern, lace, ribbon, "
    "trim and hardware. Do not invent a different outfit, do not swap colours, and do not drop a piece.\n"
    "- The LAST image is the ART-STYLE REFERENCE ONLY: use its palette, colour temperature, lighting, "
    "brushwork, edge treatment, line character, texture and overall rendering grammar.\n"
    "Do NOT copy the style reference's character, face, hairstyle, hair colour, eye colour, outfit, pose, props, "
    "background or composition. Do not render the garment references as flat product photos or as a collage; "
    "put the pieces ON the character as worn clothing."
)


def compose_garment_prompt(part_paths, style_text, character_hint="", style_ref_path="", extra=""):
    """服饰部件 + 画风参考图的提示词组装（供服饰素材采集/服装生图使用）。

    返回 (prompt, image_paths)：内容部件在前、画风参考图最后。
    """
    parts = [str(p) for p in (part_paths or []) if p]
    style = str(style_text or "").strip()
    hint = str(character_hint or "").strip()
    blocks = []
    if parts:
        blocks.append(GARMENT_REF_ROLE_INSTRUCTION.replace("{part_count}", str(len(parts))))
    if style:
        blocks.append(style)
    subject = ("Design one character wearing exactly these collected garments: "
               + (hint or "a full-body standing pose, three-quarter view, clean readable silhouette, "
                          "the outfit fully visible"))
    blocks.append(subject)
    if extra:
        blocks.append(str(extra).strip())
    prompt = "\n\n".join(b for b in blocks if b)
    return prompt, ordered_reference_images(parts, style_ref_path, style_ref_attached=bool(style_ref_path))


def assemble_style_instructions(mode, style_text, has_ref_image, prompt_compressed="", prompt_gpt=""):
    """根据模式组装头部指令与图后指令。

    返回 (head_instructions, post_instructions)：
    - head_instructions: 放到请求文本头部
    - post_instructions: 放到所有参考图之后（图文交错用）
    参考图无效或模式为关闭时，只返回样式指令原文（gpt-image 通道优先用 prompt_gpt 短版）。
    参考优先模式优先使用配置里已固化的 prompt_compressed，缺失时回退到本地启发式压缩。
    """
    style_text = str(style_text or "")
    gpt_text = str(prompt_gpt or "").strip()
    if mode == MODE_OFF or not has_ref_image:
        return (gpt_text or style_text), ""
    ref_block = build_style_ref_instruction()
    if mode == MODE_HEAD:
        head = f"{ref_block}\n\n{style_text}" if style_text else ref_block
        return head, ""
    if mode == MODE_PRIORITY:
        compressed = gpt_text or str(prompt_compressed or "").strip() or compress_style_text(style_text)
        head = f"{REF_PRIORITY_PREAMBLE}\n\n{ref_block}\n\n{compressed}"
        return head, ""
    if mode == MODE_INTERLEAVE:
        return (gpt_text or style_text), ref_block
    return (gpt_text or style_text), ""


def build_ref_gen_params(styles, style_name, mode, api_type=""):
    """统一组装「风格参考图」生图参数。

    返回 (head_instructions, post_instructions, ref_image_paths)：
    - head_instructions: 放请求文本头部（样式指令 + 参考指令块 / 参考优先压缩版）
    - post_instructions: 放所有参考图之后（图文交错用）
    - ref_image_paths:   需要作为内联附件传入的参考图路径列表
    参考图无效（未配置或文件不存在）或模式为关闭时，返回 (样式原文, "", [])。
    `api_type` 是 gpt-image 通道时优先使用入口里的 `prompt_gpt` 短版（见 utils/style_gpt.py）。
    """
    entry = normalize_style_entry((styles or {}).get(style_name))
    prompt_gpt = entry["prompt_gpt"] if _is_gpt_image_api(api_type) else ""
    has_ref = ref_image_valid(entry["ref_image"])
    if not has_ref:
        mode = MODE_OFF
    head, post = assemble_style_instructions(
        mode, entry["prompt"], has_ref, entry["prompt_compressed"], prompt_gpt
    )
    ref_paths = [entry["ref_image"]] if (has_ref and mode != MODE_OFF) else []
    return head, post, ref_paths


def _is_gpt_image_api(api_type) -> bool:
    """api_type / 站点名是否属于 gpt-image 通道（避免 utils 层互相 import）。"""
    text = str(api_type or "").strip().lower().replace("-", "_").replace(".", "_")
    return "gpt" in text


def save_styles_file(path, styles):
    """写回 config-styles.json（读旧 → 只改需要的键由调用方负责）。"""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(styles, f, ensure_ascii=False, indent=4)
