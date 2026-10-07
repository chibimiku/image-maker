"""图片「真实格式」识别与内容指纹：扩展名不可信时唯一的事实来源。

## 背景

`new.aigc2d.com` 的 Gemini 网关**自己决定返回 PNG 还是 JPEG**：同一次请求参数、
同样的像素尺寸（如 1696x2528），逐日甚至逐小时横跳。实测每天的 PNG 占比：

    09-20 98% → 09-25 37% → 09-28 9% → 10-02 0% → 10-03 71% → 10-06 16% → 10-07 48%

客户端无从干预（有无参考图、画幅、分辨率都不构成区分）。

而发布名过去把后缀**写死成 `.png`**（`single_analyzer.first_image_publish_target`），
`promote_candidate` 又用 `shutil.copy2` 原样复制字节，于是
`<hash>_<时间>-first-<token>.png` 里装的可能是 JPEG：同目录下两张文件字节完全相同、
大小分毫不差，看起来像"同一个文件出现了两次"（用户 2026-10-07 反馈）。

## 规则

1. `sniff_format` 只认 magic bytes，**不看扩展名**；
2. `is_misnamed` 判断"扩展名在撒谎"（`.jpeg` 与 `.jpg` 视为同一格式，不算撒谎）；
3. `file_sha256` 提供去重要用的内容指纹；
4. `true_filename` 把文件名换成与字节相符的后缀。

本模块只读文件、不写不删；删除决策在 `utils.image_dedup`。
"""

from __future__ import annotations

import hashlib
import os

# magic bytes → 格式名。顺序无关，全部前缀匹配。
_MAGIC_PREFIXES = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)

# 需要跳到偏移 8 才能看清的容器格式
_RIFF_WEBP = (b"RIFF", 8, b"WEBP")

# 偏移 4 起的 ftyp 品牌
_FTYP_BRANDS = {
    "avif": "avif",
    "avis": "avif",
    "heic": "heic",
    "heix": "heic",
    "hevc": "heic",
    "mif1": "heic",
    "msf1": "heic",
}

# 规范后缀（写文件时用）
EXTENSION_BY_FORMAT = {
    "png": ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
    "gif": ".gif",
    "bmp": ".bmp",
    "tiff": ".tif",
    "avif": ".avif",
    "heic": ".heic",
}

# 后缀 → 声称的格式（`.jpg` 与 `.jpeg` 等价）
FORMAT_BY_EXTENSION = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".jpe": "jpeg",
    ".webp": "webp",
    ".gif": "gif",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".avif": "avif",
    ".heic": "heic",
}

# 视为「图片」的后缀（`image_dedup` 扫描范围）
IMAGE_EXTENSIONS = tuple(sorted(FORMAT_BY_EXTENSION))

_SNIFF_BYTES = 32


def sniff_format_bytes(data: bytes) -> str:
    """只看 magic bytes 判断格式，认不出返回空串。"""
    if not data:
        return ""
    for magic, fmt in _MAGIC_PREFIXES:
        if data.startswith(magic):
            return fmt
    head, offset, tail = _RIFF_WEBP
    if data.startswith(head) and data[offset:offset + len(tail)] == tail:
        return "webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return _FTYP_BRANDS.get(data[8:12].decode("latin-1", "replace").lower(), "")
    return ""


def sniff_format(path: str) -> str:
    """读文件头判断真实格式，认不出返回空串。文件不存在也返回空串。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return ""
    return sniff_format_bytes(head)


def declared_format(path: str) -> str:
    """扩展名声称的格式，非图片后缀返回空串。"""
    return FORMAT_BY_EXTENSION.get(os.path.splitext(str(path))[1].lower(), "")


def canonical_extension(fmt: str) -> str:
    """格式名 → 规范后缀（`jpeg` → `.jpg`）。认不出返回空串。"""
    return EXTENSION_BY_FORMAT.get(str(fmt or "").lower(), "")


def true_extension(path: str) -> str:
    """该文件**真实**应对应的后缀；认不出字节时回退到原后缀。"""
    return canonical_extension(sniff_format(path)) or os.path.splitext(str(path))[1]


def is_misnamed(path: str) -> bool:
    """扩展名与真实字节不符（如 `.png` 里装着 JPEG）。

    非图片后缀、或字节认不出时**返回 False**：判不了就不动它。
    """
    claimed = declared_format(path)
    if not claimed:
        return False
    actual = sniff_format(path)
    if not actual:
        return False
    return claimed != actual


def true_name(path: str) -> str:
    """把路径换成与字节相符的后缀（同目录、只改后缀）；认不出字节时原样返回。

    后缀**没有撒谎**时原样返回，所以 `.jpeg` 不会被改写成 `.jpg`。
    """
    actual = sniff_format(path)
    if not actual:
        return str(path)
    if declared_format(path) == actual:
        return str(path)
    return os.path.splitext(str(path))[0] + canonical_extension(actual)


def name_with_extension_of(target: str, source: str) -> str:
    """把 `target` 的后缀换成 `source` 真实格式对应的后缀。

    给「目标文件还没写出来」的场合用（正要复制的发布副本）：此时 `target` 不存在，
    探不了它自己的字节，而副本的字节将**等于** `source`，所以格式以 `source` 为准。
    """
    actual = sniff_format(source)
    if not actual:
        return str(target)
    if declared_format(target) == actual:
        return str(target)
    return os.path.splitext(str(target))[0] + canonical_extension(actual)


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """文件内容 SHA-256（十六进制小写）。读不到返回空串。"""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def unique_path(path: str) -> str:
    """重名容错：目标已存在就依次试 `<stem>-1.ext`、`<stem>-2.ext`…

    只用来给**内容不同**的落盘找位置（内容相同的重复由 `file_sha256` 认出来复用）；
    与原路径不冲突时原样返回。
    """
    path = str(path)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{stem}-{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def describe(path: str) -> dict:
    """一个文件的身份摘要，供日志与去重清单使用。"""
    actual = sniff_format(path)
    claimed = declared_format(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return {
        "path": path,
        "name": os.path.basename(str(path)),
        "size": size,
        "declared": claimed,
        "actual": actual,
        "misnamed": bool(claimed and actual and claimed != actual),
        "sha256": file_sha256(path),
    }
