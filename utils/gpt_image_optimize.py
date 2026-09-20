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
    "system_prompt": f"{PROMPT_DIR_RELATIVE}/repaint-system.md",
    "detail_suffix": f"{PROMPT_DIR_RELATIVE}/repaint-detail-suffix.md",
    "use_detail_suffix": True,
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
