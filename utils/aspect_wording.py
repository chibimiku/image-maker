"""内容锚里的「画幅措辞」校正（只服务分析链路）。

问题（2026-10-01 定位）：分析链路的正文会混进一个和实际图片不符的画幅声明 ——
Step 1 的 vision 把横向图写成 "a balanced vertical tableau"，Step 2 的精修又按
`prompts/refine-desc.md` 的指示把推断出来的 "2:3" 写进正文，于是 gpt-image 通道下发的
内容锚开头就是 "Vertical 2:3 illustration"。实测这段文本能压过请求里的 size：
全项目 5 个「请求 1536x1024」的首图里，唯二出成竖图的两次，正文都写着竖幅措辞
（634dfc91 的 "Vertical 2:3 illustration"、1805bc93 的 "in a portrait 2:3 composition"），
而写着 "16:9 landscape illustration scene" 或没有画幅措辞的三次都按 size 出了横图。

所以这里提供两个用途：

1. 下发首图前按**本次实际生效的尺寸**改写正文里的画幅声明（用户手动选的尺寸优先）；
2. 落盘 `gpt_image_prompt` 时按**源图实测比例**改写，别让假画幅留在产物里继续误导下游。

画风参考图的朝向永远不参与 —— 它只提供画法，不提供画幅。
"""
import re

# gpt-image-2 的三档尺寸 → (朝向词, 比例词)
SIZE_LABELS = {
    "1024x1536": ("Vertical", "2:3"),
    "1536x1024": ("Horizontal", "3:2"),
    "1024x1024": ("Square", "1:1"),
}

_PORTRAIT_RATIOS = ("2:3", "3:4", "3:5", "5:8", "9:16", "1:2", "2:5")
_LANDSCAPE_RATIOS = ("3:2", "4:3", "5:3", "5:2", "8:5", "16:9", "2:1")
_SQUARE_RATIOS = ("1:1",)

# 只在正文开头这一小段里动手：画幅声明永远在最前面，越往后越容易误伤普通形容词
DEFAULT_WINDOW = 400

_ORIENTATION = r"(?:vertical|horizontal|portrait|landscape|square)"
_ASPECT_NOUN = (r"(?:illustration|scene|composition|frame|format|canvas|crop|tableau"
                r"|portrait|image|panel|view)")
_RATIO = r"\d{1,2}:\d{1,2}"


def aspect_label(size: str = "", ratio: str = "") -> tuple:
    """把尺寸或比例归一成 ``(朝向词, 比例)``；看不懂就返回 ``("", "")``。

    只认「事实」（实测尺寸 / 实测比例），不认描述文本里写的画幅 —— 后者正是要被替换的。
    """
    candidate = str(size or "").strip().lower().replace(" ", "")
    if candidate in SIZE_LABELS:
        return SIZE_LABELS[candidate]

    text = str(ratio or "").strip()
    if not text and ":" in candidate:
        text = candidate          # 调用方把比例写在 size 参数里（如 "16:9"）也认
    if not text:
        return "", ""
    if "x" in text.lower() and ":" not in text:
        width_text, _, height_text = text.lower().partition("x")
        try:
            width_val, height_val = float(width_text), float(height_text)
        except ValueError:
            return "", ""
        if width_val <= 0 or height_val <= 0:
            return "", ""
        if abs(width_val - height_val) <= 0.02 * max(width_val, height_val):
            return "Square", "1:1"
        return ("Landscape", "3:2") if width_val > height_val else ("Portrait", "2:3")

    if ":" not in text:
        return "", ""
    ratio_text = text
    if ratio_text in _SQUARE_RATIOS:
        return "Square", ratio_text
    if ratio_text in _PORTRAIT_RATIOS:
        return "Vertical", ratio_text
    if ratio_text in _LANDSCAPE_RATIOS:
        return "Horizontal", ratio_text
    width_text, _, height_text = ratio_text.partition(":")
    try:
        width_val, height_val = float(width_text), float(height_text)
    except ValueError:
        return "", ""
    if width_val <= 0 or height_val <= 0:
        return "", ""
    if abs(width_val - height_val) <= 0.02 * max(width_val, height_val):
        return "Square", ratio_text
    return ("Horizontal" if width_val > height_val else "Vertical"), ratio_text


def align_aspect_wording(text: str, size: str = "", ratio: str = "",
                         window: int = DEFAULT_WINDOW) -> tuple:
    """把正文开头写死的画幅声明改成与本次生效尺寸一致。

    返回 ``(新文本, 改动列表)``；看不懂尺寸/比例、或正文里本来就没有画幅声明时原样返回。
    只在开头 ``window`` 个字符内替换，并且要求「朝向词 / 比例词」与「画幅名词」同时出现，
    所以 "long vertical ribbons"、"the sleeve descends" 这类普通描述不会被误改。
    """
    source = str(text or "")
    orientation, ratio_text = aspect_label(size=size, ratio=ratio)
    if not orientation or not source.strip():
        return source, []

    head, tail = source[:max(1, int(window))], source[max(1, int(window)):]
    changes = []

    def _label(word: str, match) -> str:
        """句首大写、句中用小写，替换完读起来仍像人写的。"""
        return word if match.start() == 0 else word.lower()

    def _apply(pattern: str, build) -> None:
        nonlocal head

        def repl(match):
            replacement = build(match)
            if replacement and replacement != match.group(0):
                changes.append(f"{match.group(0)} → {replacement}")
            return replacement

        head = re.sub(pattern, repl, head)

    # "Vertical 2:3" / "portrait 2:3"
    _apply(rf"(?i)\b{_ORIENTATION}\s+{_RATIO}\b",
           lambda m: f"{_label(orientation, m)} {ratio_text}")
    # "16:9 landscape" → 调换顺序，读起来才顺
    _apply(rf"(?i)\b{_RATIO}\s+({_ORIENTATION})\b",
           lambda m: f"{_label(orientation, m)} {ratio_text}")
    # "2:3 frame" / "2:3 composition"
    _apply(rf"(?i)\b{_RATIO}\s+({_ASPECT_NOUN})\b",
           lambda m: f"{ratio_text} {m.group(1)}")
    # "vertical illustration" / "close vertical portrait"
    _apply(rf"(?i)\b{_ORIENTATION}\s+({_ASPECT_NOUN})\b",
           lambda m: f"{_label(orientation, m)} {m.group(1)}")

    return head + tail, changes
