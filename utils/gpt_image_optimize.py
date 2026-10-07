"""gpt-image 产物优化（Gemini 重绘提线）运行时：prompt 装载、拼接与落盘参数。

理论、实验数据与指标口径见 `docs/gpt-image-optimize/README.md`。
Prompt 与配置都在 `prompts/gpt-image-optimize/`，本模块不写死任何长文本。

被三处复用（单一事实来源，避免 GUI / CLI / 后端各写一份）：
- GUI：`modules/image_generation/gpt_image2_tab.py`
- CLI：`tools/gpt_image2_gen.py --repaint`
- 后端：`modules/others/api_backend.generate_image_repaint`
"""
import json
import os
import re

from utils.prompt_loader import PROMPTS_DIR, read_prompt_file, render_prompt_file

CONFIG_RELATIVE_PATH = "gpt-image-optimize/config.json"
PROMPT_DIR_RELATIVE = "gpt-image-optimize"

# 宽高比：auto = 不传该字段，让模型按输入图比例输出（官方默认行为，换任意源图都不会变形）
ASPECT_RATIO_AUTO = "auto"
# 官方（Gemini 3 图片模型）支持的比例清单
ASPECT_RATIO_OPTIONS = (
    ASPECT_RATIO_AUTO, "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
)
_ASPECT_AUTO_ALIASES = {"", "auto", "自动", "跟随源图", "keep", "same", "original", "none", "null"}

# 与 `prompts/gpt-image-optimize/config.json` 同义的兜底值（配置文件缺失时仍可跑）
DEFAULTS = {
    "enabled": False,
    "site": "new.aigc2d",
    "api_type": "aigc2d",
    "model": "gemini-3-pro-image-preview",
    "resolution": "2K",
    # auto = 不传 aspectRatio，由模型按输入图比例输出（官方默认行为）
    "aspect_ratio": ASPECT_RATIO_AUTO,
    "repeat": 1,
    "save_sub_dir": "gpt_image_repaint",
    "file_prefix": "repaint",
    "system_prompt": f"{PROMPT_DIR_RELATIVE}/repaint-system-conservative-v5.md",
    "detail_suffix": f"{PROMPT_DIR_RELATIVE}/repaint-detail-suffix.md",
    "use_detail_suffix": False,
    "model_options": ["gemini-3-pro-image-preview", "gemini-3-pro-image"],
    "resolution_options": ["1K", "2K", "4K"],
    "aspect_ratio_options": list(ASPECT_RATIO_OPTIONS),
}

_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]*)\n(.*?)```", re.S)


def _as_bool(value, fallback: bool = False) -> bool:
    """稳健布尔解析：字符串 "false"/"0"/"no" 也算 False（JSON 与手填配置都吃）。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return fallback


def load_config() -> dict:
    """读取重绘配置；文件不存在或损坏时回退 DEFAULTS（不抛异常，方便离线/测试）。"""
    config = dict(DEFAULTS)
    path = os.path.join(PROMPTS_DIR, CONFIG_RELATIVE_PATH.replace("/", os.sep))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            config.update({k: v for k, v in data.items() if not str(k).startswith("_")})
    except Exception:  # noqa: BLE001 - 配置坏了不该让 Tab 起不来
        pass
    return config


def save_config(config: dict) -> str:
    """把配置写回 `prompts/gpt-image-optimize/config.json`（保留 `_comment` 等说明字段）。"""
    path = os.path.join(PROMPTS_DIR, CONFIG_RELATIVE_PATH.replace("/", os.sep))
    existing = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            existing = raw
    except Exception:  # noqa: BLE001
        existing = {}
    merged = dict(existing)
    merged.update(config or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    return path


def extract_prompt_text(text: str) -> str:
    """从 md 里取 prompt 正文：有 ``` 代码块就只取块内内容，否则整篇。"""
    blocks = _CODE_BLOCK_RE.findall(text or "")
    if blocks:
        return "\n".join(block.strip() for block in blocks if block.strip()).strip()
    return str(text or "").strip()


def read_prompt_relative(relative_path: str) -> str:
    """读 prompts/ 下的 prompt 正文（自动剥掉 md 代码围栏与说明文字）。"""
    return extract_prompt_text(read_prompt_file(relative_path))


def build_repaint_prompt(config: dict = None) -> str:
    """组装最终重绘 prompt：主 prompt + 细节强化后缀（可关）。

    后缀用 `\\n\\n` 连接，与实验中验证过的追加方式一致（旧版直接拼
    `, detailed face, ...` 也能跑，但语义上是独立段落更稳）。
    """
    conf = dict(DEFAULTS)
    conf.update(config or {})
    # 与实验口径一致：主 prompt + 细节后缀两个文件都完整装载，用 `\n\n` 拼接
    prompt = read_prompt_relative(str(conf.get("system_prompt")))
    if not _as_bool(conf.get("use_detail_suffix", True), True):
        return prompt
    suffix_path = str(conf.get("detail_suffix") or "").strip()
    if not suffix_path:
        return prompt
    try:
        suffix = read_prompt_relative(suffix_path)
    except FileNotFoundError:
        return prompt
    return f"{prompt}\n\n{suffix}" if suffix else prompt


def build_prompt_via_template(relative_path: str, replacements=None) -> str:
    """按 `{key}` 占位替换装载 prompt（保留给需要带变量的场景）。"""
    return extract_prompt_text(render_prompt_file(relative_path, replacements or {}))


# ---------------------------------------------------------------------------
# 「材质表面」定向过一遍（2026-10-06 用户反馈：生成图皮革太平涂、没有参考图那种光泽）
# ---------------------------------------------------------------------------
#
# 设计约束（用户 2026-10-06 指出）：**这套词汇和固件必须跨任务通用**。
# 别的地方要处理的可能是皮夹克、皮手套、皮鞋、漆皮包、丝袜、金属扣、羽毛……
# 所以三条硬规则：
#   1. 词条只描述**材质外观与表面响应**，不认物件、不写"靴子/裙摆/台阶"这类本次任务的具体名词；
#   2. 目标（targets）默认按**材质外观**指认，并且明说"目标之外一处都不许动"；
#   3. 允许调用方传 `--material-spec/--targets` 自定义（`MATERIAL_SPEC_KEYS`），
#      新增材质只需往 `MATERIALS` 加一条，不改固件、不改工具。
#
# 条款里出现的部位词只用**通用几何词**（边缘、形体、折痕、缝线、内侧面），任何物件都能套。
MATERIAL_DIRECTIVE_RELATIVE = f"{PROMPT_DIR_RELATIVE}/material-surface-base.md"
MATERIALS = {
    "patent_leather": {
        "label": "漆皮皮革（高光皮革）",
        "material": "high-gloss patent leather",
        "clauses": [
            "Treat **every** surface this material covers as one continuous glossy skin: the sheen must not cover part "
            "of a form and leave the rest matte. Each panel of it carries the same material behaviour.",
            "Run a long, narrow, bright specular band along the length of each form, whose shape follows the surface "
            "curvature and ends in a firm edge — never a broad soft gradient, never a round blob.",
            "Add an env-reflected mid band: along the lower or inner side of each volume, a second, dimmer reflection "
            "of the surroundings that reduces the local saturation and carries colour with it.",
            "Steepen the value falloff between the specular band, the reflected band and the shadow, so the surface "
            "reads as hard and polished rather than matte fill.",
            "Keep a fine micro-grain and shallow surface creases and stitching lines over the sheen, with tight dark "
            "contact shadow in the seams, so it reads as animal hide or coated sheet rather than plastic film.",
            "Deepen only the local shadow inside folds and creases, and let the creases pinch the highlight where they "
            "cross it; keep the base colour hue unchanged.",
        ],
    },
    "matte_leather": {
        "label": "哑光皮革",
        "material": "supple matte leather",
        "clauses": [
            "Replace any flat fill with grain: a fine, slightly irregular pebbled micro-texture that is denser in "
            "shadow and smoothed out on the crests facing the light.",
            "Keep the highlight broad, low-contrast and satin-dry across the whole surface — never a mirror band, "
            "never a plastic sheen.",
            "Let shallow creases and stretch marks catch light only on their crests, with a dark seam line and soft "
            "occlusion where panels meet.",
        ],
    },
    "satin_leather": {
        "label": "缎面皮革（柔光皮革）",
        "material": "soft satin-finish leather",
        "clauses": [
            "Give the surface a broad, soft sheen that follows the form: a long, gently graded highlight instead of a "
            "hard specular line, still clearly brighter than the surrounding fill.",
            "Keep a wide mid-tone band of reflected colour between highlight and shadow so the surface reads as supple "
            "and slightly waxy rather than flat.",
            "Add fine leather grain and shallow crease lines that catch the light along their crests only.",
        ],
    },
    "fur": {
        "label": "绒毛（毛绒）",
        "material": "dense plush fur",
        "clauses": [
            "Break the silhouette edge into many short, soft, tapered tufts of slightly varying length; never leave a "
            "clean vector edge.",
            "Add directional strand clusters with a soft terminator, a lighter lit side and a deeper core shadow under "
            "the mass, so the volume has thickness.",
            "Keep a few longer flyaway tufts catching the light; no wet or glossy look anywhere on the fur.",
        ],
    },
    "lace": {
        "label": "蕾丝",
        "material": "fine bobbin lace",
        "clauses": [
            "Resolve the pattern into a legible openwork ground with a denser scalloped border, not a solid fill.",
            "Show the layer underneath through the open ground, and cast a small soft shadow from the lace onto it.",
            "Keep threads thin and continuous with tiny tension variations; no plastic or embroidered-patch look.",
        ],
    },
    "fabric_satin": {
        "label": "缎面布料",
        "material": "lustrous satin fabric",
        "clauses": [
            "Make folds read through long, smooth highlight tongues along the fold crests and dark, tucked shadow at "
            "the fold roots.",
            "Keep a silky anisotropic sheen that changes with fold direction; no leather grain, no hard specular line.",
        ],
    },
    "sheer": {
        "label": "半透明薄纱／丝袜",
        "material": "sheer translucent fabric",
        "clauses": [
            "Where the fabric overlaps skin, body or another layer, let the underlying shape show through slightly "
            "darker and less saturated instead of hiding it.",
            "Keep the weave fine and even, denser and more opaque where the surface turns away from the light or "
            "gathers in a fold.",
            "No specular highlight and no hard rim: translucency is a value and saturation change, not gloss.",
        ],
    },
    "metal": {
        "label": "金属",
        "material": "polished metal",
        "clauses": [
            "Reflect the surroundings as a compressed, high-contrast gradient rather than a solid colour patch; the "
            "reflection changes sharply across curvature.",
            "Split highlights into a small intense core with a chromatically shifted fringe, and keep edges crisp "
            "against neighbouring materials.",
            "Keep the base metal hue unchanged; do not add rust, engraving or ornament that is not already drawn.",
        ],
    },
    "wood_varnish": {
        "label": "上漆木面",
        "material": "varnished wood",
        "clauses": [
            "Keep the grain direction visible and add a restrained clear-coat sheen with a few soft long reflections "
            "running along the grain, not across it.",
            "Deepen the contact shadow where surfaces meet; no mirror-like gloss.",
        ],
    },
}

# 每个材质的习惯目标：**只按材质外观指认，不认物件**，别处任务（夹克、手套、箱子、沙发…）同样成立。
DEFAULT_MATERIAL_TARGETS = {
    "patent_leather": "every surface in the image that is already drawn as a glossy coated material — whatever object "
                      "it belongs to. Leave every other material exactly as it is.",
    "matte_leather": "every surface in the image that is already drawn as leather or a similar thick hide — whatever "
                     "object it belongs to. Leave every other material exactly as it is.",
    "satin_leather": "every surface in the image that is already drawn as leather with a soft sheen — whatever object "
                     "it belongs to. Leave every other material exactly as it is.",
    "fur": "every surface in the image that is already drawn as fur, plush or a soft pile — whatever object it belongs "
           "to. Leave every other material exactly as it is.",
    "lace": "every surface in the image that is already drawn as lace or openwork trim — whatever object it belongs "
            "to. Leave every other material exactly as it is.",
    "fabric_satin": "every surface in the image that is already drawn as a smooth lustrous cloth — whatever object it "
                    "belongs to. Leave every other material exactly as it is.",
    "sheer": "every surface in the image that is already drawn as sheer or translucent cloth — whatever object it "
             "belongs to. Leave every other material exactly as it is.",
    "metal": "every surface in the image that is already drawn as metal — whatever object it belongs to. Leave every "
             "other material exactly as it is.",
    "wood_varnish": "every surface in the image that is already drawn as finished or varnished wood — whatever object "
                    "it belongs to. Leave every other material exactly as it is.",
}

# 自定义材质规格允许的键（`--material-spec k=v;...`），值全是自然语言，不写死任何物件名
MATERIAL_SPEC_KEYS = ("label", "material", "clauses")


def material_names() -> list:
    return sorted(MATERIALS)


def describe_material(key: str) -> str:
    """`key — 中文名`；未知键返回空串（调用方据此报错）。"""
    info = MATERIALS.get(str(key or "").strip())
    return f"{key} — {info['label']}" if info else ""


def normalize_material_spec(spec=None) -> dict:
    """把 `--material-spec`（dict 或 `k=v; k=v` 字符串）规范化成一条材质词条。

    允许的键只有 `MATERIAL_SPEC_KEYS`：`label`（中文名）、`material`（英文材质表述）、
    `clauses`（渲染条款，字符串用 `|` 或换行分隔，也可直接给列表）。
    **值全是自然语言**，所以别处任务（皮夹克/手套/箱子/沙发…）不需要改代码就能加材质。
    """
    if not spec:
        raise ValueError("自定义材质规格为空：至少给 material=<英文材质表述>")
    if isinstance(spec, str):
        data = {}
        for chunk in str(spec).replace("\n", ";").split(";"):
            if not chunk.strip():
                continue
            key, _, value = chunk.partition("=")
            key = key.strip().lower()
            if key not in MATERIAL_SPEC_KEYS:
                raise ValueError(f"自定义材质规格不认识的键：{key}；可用 {', '.join(MATERIAL_SPEC_KEYS)}")
            data[key] = value.strip()
    elif isinstance(spec, dict):
        unknown = [k for k in spec if str(k).lower() not in MATERIAL_SPEC_KEYS]
        if unknown:
            raise ValueError("自定义材质规格不认识的键：" + ", ".join(map(str, unknown)))
        data = {str(k).lower(): v for k, v in spec.items()}
    else:
        raise ValueError("自定义材质规格只接受 dict 或 'k=v; k=v' 字符串")
    material_text = str(data.get("material") or "").strip()
    if not material_text:
        raise ValueError("自定义材质规格必须给 material=<英文材质表述>")
    clauses = data.get("clauses") or []
    if isinstance(clauses, str):
        clauses = [part.strip() for part in clauses.replace("\n", "|").split("|")]
    clauses = [str(c).strip() for c in clauses if str(c).strip()]
    return {
        "label": str(data.get("label") or material_text).strip(),
        "material": material_text,
        "clauses": clauses,
        "custom": True,
    }


def resolve_material_info(material=None, spec=None) -> dict:
    """取一条材质词条：内置键优先，否则用自定义规格；都不给/都不认识则抛 `ValueError`。"""
    key = str(material or "").strip()
    if key:
        info = MATERIALS.get(key)
        if info is None:
            raise ValueError("未知材质：" + key + "；可用：" + ", ".join(material_names())
                             + "；或改用 --material-spec 自定义")
        if spec:
            raise ValueError("--material 与 --material-spec 只能给一个")
        return {"key": key, "label": info["label"], "material": info["material"],
                "clauses": list(info["clauses"]), "custom": False}
    if spec:
        entry = normalize_material_spec(spec)
        entry["key"] = "custom"
        return entry
    raise ValueError("必须给 --material 或 --material-spec")


def build_material_prompt(material: str = "", targets: str = "", spec=None) -> str:
    """组装「材质表面」提示词：**专用材质固件** + 该材质的渲染条款。

    为什么不叠在保守重绘固件（v5）后面：v5 的第 5 条与 FORBIDDEN 明确禁止"额外高光、
    高光扩散、增强对比" —— 而这几条正是皮革光泽要做的事。两层叠在一起会互相打架，
    实测（2026-10-06 用户反馈那张靴子）出来的高光几乎没变化。
    所以改用 `material-surface-base.md`：它保留同样的几何/身份/配色锁，但**不禁止**
    具名表面上的高光与材质结构。

    - `material` 取 `MATERIALS` 的键，或改用 `spec`（自定义材质规格）；
    - `targets` 缺省按**材质外观**指认（`DEFAULT_MATERIAL_TARGETS`），
      明说"目标之外一处都不动" —— 不写死任何物件名，换任务同样成立。
    """
    info = resolve_material_info(material, spec)
    default_targets = DEFAULT_MATERIAL_TARGETS.get(info.get("key") or "",
                                                   "every surface in the image that is already drawn as this "
                                                   "material — whatever object it belongs to. Leave every other "
                                                   "material exactly as it is.")
    clauses = list(info.get("clauses") or [])
    return build_prompt_via_template(MATERIAL_DIRECTIVE_RELATIVE, {
        "targets": str(targets or default_targets).strip(),
        "material": info["material"],
        "material_clauses": "\n".join(f"{i}. {c}" for i, c in enumerate(clauses, 1))
                            or "1. Keep the surface reading faithful to the material named above.",
    }).strip()


def resolve_aspect_ratio(config: dict = None, fallback: str = ASPECT_RATIO_AUTO) -> str:
    """解析重绘输出宽高比返回值：`"auto"` 表示"不传字段、跟随输入图"。

    为什么要 auto：官方明确 "By default, the model matches the output image size to
    that of your input image"，写死一个比例会让非该比例的源图被拉伸或裁切。
    """
    conf = dict(DEFAULTS)
    conf.update(config or {})
    text = str(conf.get("aspect_ratio") or "").strip()
    if text.lower() in _ASPECT_AUTO_ALIASES:
        return ASPECT_RATIO_AUTO
    return text or str(fallback or ASPECT_RATIO_AUTO)


def is_auto_aspect_ratio(value) -> bool:
    return str(value or "").strip().lower() in _ASPECT_AUTO_ALIASES


def image_aspect_ratio_label(path: str) -> str:
    """读本地图片的像素宽高比，返回 `"3:2"` 这样的可读标签（读不到时返回 `""`）。"""
    try:
        from PIL import Image
        with Image.open(path) as img:
            width, height = img.size
        if not width or not height:
            return ""
        from math import gcd
        divisor = gcd(int(width), int(height))
        return f"{int(width) // divisor}:{int(height) // divisor}"
    except Exception:  # noqa: BLE001 - 读不到比例不影响重绘
        return ""


def source_model_aspect_ratio(path: str, fallback="auto") -> str:
    """源图比例映射到模型支持的标签，自动重绘也显式下发。"""
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        labels = [label for label in ASPECT_RATIO_OPTIONS if ":" in label]
        return min(labels, key=lambda label: abs(float(label.split(":")[0]) / float(label.split(":")[1]) - width / height))
    except Exception:
        return fallback


def validate_repaint_aspect(path, aspect_ratio):
    """服务器仍返回错比例时保留文件并报错，禁止继续把方图当横图加工。"""
    if is_auto_aspect_ratio(aspect_ratio):
        return
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        a, b = str(aspect_ratio).split(":")
        target = float(a) / float(b)
    except (OSError, ValueError, TypeError, ZeroDivisionError):
        return
    if abs(width / height / target - 1) > 0.03:
        raise RuntimeError(f"重绘输出比例不符：请求 {aspect_ratio}，实际 {width}×{height}。已保留产物 {path}，停止后续工序。")


def plan_output(source_path: str, config: dict = None) -> dict:
    """给出一次重绘的落盘参数：源文件名 -> 输出前缀 / 子目录。"""
    conf = dict(DEFAULTS)
    conf.update(config or {})
    stem = os.path.splitext(os.path.basename(str(source_path or "")))[0]
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._") or "src"
    prefix_base = str(conf.get("file_prefix") or DEFAULTS["file_prefix"]).strip() or "repaint"
    return {
        "save_sub_dir": str(conf.get("save_sub_dir") or DEFAULTS["save_sub_dir"]),
        "file_prefix": f"{prefix_base}_{stem}",
    }


def resolve_repaint_call(config: dict = None, *, source_path: str = "", model=None, resolution=None,
                         aspect_ratio=None, repeat=None, prompt=None, prompt_suffix=None,
                         use_detail_suffix=None, save_sub_dir=None, file_prefix=None,
                         api_type=None) -> dict:
    """解析**这一次重绘调用**的最终参数（单一事实来源）。

    第四轮 P0.3：受控实验要求「预检算出来的请求」就是「实际发出去的请求」，所以
    模型 / 分辨率 / 比例 / prompt 与后缀 / 重复次数 / 落盘位置都在这里解析一次，
    `generate_image_repaint` 与预检入口共用同一份结果。
    """
    conf = dict(DEFAULTS)
    conf.update(config or {})
    resolved_prompt_from_config = prompt is None
    if prompt is None:
        conf_for_prompt = dict(conf)
        if use_detail_suffix is not None:
            conf_for_prompt["use_detail_suffix"] = use_detail_suffix
        resolved_prompt = build_repaint_prompt(conf_for_prompt)
    else:
        resolved_prompt = str(prompt or "")
    if prompt_suffix is not None:
        resolved_suffix = str(prompt_suffix or "")
    elif use_detail_suffix is False:
        resolved_suffix = ""
    else:
        suffix_relative = str(conf.get("detail_suffix") or "").strip()
        try:
            resolved_suffix = read_prompt_relative(suffix_relative) if suffix_relative else ""
        except Exception:  # noqa: BLE001 - 后缀缺失不影响主 prompt
            resolved_suffix = ""
    try:
        resolved_repeat = max(1, int(repeat if repeat is not None else conf.get("repeat") or 1))
    except (TypeError, ValueError):
        resolved_repeat = 1
    plan = plan_output(source_path, conf) if source_path else {
        "save_sub_dir": str(conf.get("save_sub_dir") or DEFAULTS["save_sub_dir"]),
        "file_prefix": str(conf.get("file_prefix") or DEFAULTS["file_prefix"])}
    resolved_aspect = resolve_aspect_ratio({"aspect_ratio": aspect_ratio} if aspect_ratio else conf, ASPECT_RATIO_AUTO)
    if is_auto_aspect_ratio(resolved_aspect) and source_path:
        resolved_aspect = source_model_aspect_ratio(source_path)
    return {
        "model": str(model or conf.get("model") or DEFAULTS["model"]),
        "resolution": str(resolution or conf.get("resolution") or DEFAULTS["resolution"]),
        "aspect_ratio": resolved_aspect,
        "repeat": resolved_repeat,
        "prompt": resolved_prompt,
        "prompt_from_config": resolved_prompt_from_config,
        "prompt_suffix": resolved_suffix,
        "detail_suffix_applied": bool(resolved_suffix),
        "use_detail_suffix": bool(use_detail_suffix),
        "api_type": str(api_type or conf.get("api_type") or DEFAULTS["api_type"]),
        "save_sub_dir": str(save_sub_dir or plan["save_sub_dir"]),
        "file_prefix": str(file_prefix or plan["file_prefix"]),
        "config_source": str(PROMPT_DIR_RELATIVE + "/config.json"),
        "detail_suffix_source": str(conf.get("detail_suffix") or ""),
        "system_prompt_source": str(conf.get("system_prompt") or ""),
    }
