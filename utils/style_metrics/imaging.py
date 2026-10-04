"""画风指标的图像加载与固定预处理。

所有指标都从这里取张量，保证「同一份输入、同一套预处理」在 CPU / CUDA / NPU 上一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LPIPS_INPUT_SIZE,
    VGG_INPUT_SIZE,
    VGG_RESIZE_MODE,
)

_PIL_RESAMPLE = getattr(Image, "Resampling", Image)


def load_rgb(path: str | Path) -> Image.Image:
    """读图并统一成 RGB（丢弃 alpha，EXIF 方向已由 PIL 处理）。"""
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_exact(im: Image.Image, size: int, mode: str = VGG_RESIZE_MODE) -> Image.Image:
    """直接缩放到 size x size（不保持长宽比）。

    固定方形输入是为了让两张不同长宽比的图仍能逐层比较 Gram / 通道统计量；
    代价是引入缩放形变，这一点在文档里明示，不允许声称与「原尺寸比较」等价。
    """
    resample = {
        "bicubic": _PIL_RESAMPLE.BICUBIC,
        "bilinear": _PIL_RESAMPLE.BILINEAR,
        "lanczos": _PIL_RESAMPLE.LANCZOS,
        "nearest": _PIL_RESAMPLE.NEAREST,
    }[mode]
    return im.resize((size, size), resample)


def to_tensor01(im: Image.Image) -> torch.Tensor:
    """PIL -> float32 CHW，取值 [0, 1]。"""
    arr = np.asarray(im, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """按 ImageNet mean/std 归一化（VGG-19 是 ImageNet 预训练权重，必须配套）。

    接受 CHW 或 NCHW；广播形状按维度决定，避免把 CHW 悄悄变成 (1,C,H,W)。
    """
    view = (1, 3, 1, 1) if x.dim() == 4 else (3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(view)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(view)
    return (x - mean) / std


def to_imagenet_vgg(images: Sequence[Image.Image], size: int = VGG_INPUT_SIZE) -> torch.Tensor:
    """VGG-19 指标输入：NCHW float32，512x512，ImageNet 归一化。"""
    tensors = [imagenet_normalize(to_tensor01(resize_exact(im, size))) for im in images]
    return torch.stack(tensors, dim=0)


def to_imagenet_vgg_numpy(images: Sequence[Image.Image], size: int = VGG_INPUT_SIZE) -> np.ndarray:
    """同 :func:`to_imagenet_vgg`，但返回 numpy（喂给 OpenVINO）。"""
    return to_imagenet_vgg(images, size).numpy().astype(np.float32)


def to_lpips(images: Sequence[Image.Image], size: int = LPIPS_INPUT_SIZE) -> torch.Tensor:
    """LPIPS 官方输入：NCHW float32，[-1, 1]。"""
    tensors = [to_tensor01(resize_exact(im, size)) * 2.0 - 1.0 for im in images]
    return torch.stack(tensors, dim=0)


def resize_short_edge_then_center_crop(
    im: Image.Image, size: int, mode: str = "bicubic"
) -> Image.Image:
    """torchvision ``Resize(size)`` + ``CenterCrop(size)`` 的等价实现（官方 CSD 预处理）。

    ``Resize(224)`` 传整数时是把**短边**缩到 224 并保持长宽比，再中心裁剪成 224x224。
    这与 :func:`resize_exact` 的直接拉伸**不是**同一件事，不能互换。
    """
    resample = {
        "bicubic": _PIL_RESAMPLE.BICUBIC,
        "bilinear": _PIL_RESAMPLE.BILINEAR,
        "lanczos": _PIL_RESAMPLE.LANCZOS,
    }[mode]
    w, h = im.size
    scale = size / min(w, h)
    new = (max(size, int(round(w * scale))), max(size, int(round(h * scale))))
    resized = im.resize(new, resample)
    left = (new[0] - size) // 2
    top = (new[1] - size) // 2
    return resized.crop((left, top, left + size, top + size))


def to_csd(images: Sequence[Image.Image], size: int, mean: Sequence[float], std: Sequence[float]):
    """CSD 输入：官方 ``transforms_branch0``（短边 Resize + CenterCrop + CLIP 归一化）。"""
    tensors = []
    for im in images:
        x = to_tensor01(resize_short_edge_then_center_crop(im, size))
        m = torch.tensor(list(mean), dtype=x.dtype).view(3, 1, 1)
        s = torch.tensor(list(std), dtype=x.dtype).view(3, 1, 1)
        tensors.append((x - m) / s)
    return torch.stack(tensors, dim=0)


def make_test_images(seed: int = 20261004) -> dict[str, Image.Image]:
    """生成确定性的临时测试图（不读取用户素材、不启动生图任务）。

    返回结构：
      ``base``        —— 基准图
      ``base_png``    —— base 无损 PNG 往返（同图，距离应为 ~0）
      ``variant_soft``—— 暖色调 + 轻微模糊（与 base 差异小）
      ``variant_hard``—— 冷色调 + 强纹理 + 不同构图（与 base 差异大）
    """
    rng = np.random.default_rng(seed)
    h = w = 384
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def _base_array() -> np.ndarray:
        r = 0.55 + 0.30 * np.sin(xx / 47.0) * np.cos(yy / 61.0)
        g = 0.50 + 0.28 * np.sin((xx + yy) / 53.0)
        b = 0.60 + 0.25 * np.cos((xx - yy) / 71.0)
        disk = ((xx - w * 0.35) ** 2 + (yy - h * 0.40) ** 2) < (w * 0.18) ** 2
        r = np.where(disk, 0.95, r)
        g = np.where(disk, 0.55, g)
        b = np.where(disk, 0.35, b)
        arr = np.stack([r, g, b], axis=-1)
        arr += rng.normal(0.0, 0.010, arr.shape).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    base = Image.fromarray((_base_array() * 255).astype(np.uint8), "RGB")

    soft = np.asarray(base, dtype=np.float32) / 255.0
    soft = soft * np.array([1.10, 1.02, 0.86], dtype=np.float32)
    # 真·二维模糊（5x5 高斯核），保证「不同图但差异较小」这一档确实与同图拉开距离。
    g1d = np.array([0.06, 0.24, 0.40, 0.24, 0.06], dtype=np.float32)
    kernel = np.outer(g1d, g1d)
    padded = np.pad(soft, ((2, 2), (2, 2), (0, 0)), mode="edge")
    blurred = np.zeros_like(soft)
    for dy in range(5):
        for dx in range(5):
            blurred += kernel[dy, dx] * padded[dy : dy + h, dx : dx + w, :]
    soft = np.clip(blurred, 0.0, 1.0)
    variant_soft = Image.fromarray((soft * 255).astype(np.uint8), "RGB")

    hard = np.asarray(base, dtype=np.float32) / 255.0
    hard = hard * np.array([0.72, 0.92, 1.25], dtype=np.float32)
    stripe = (np.sin((xx * 0.9 + yy * 0.4) / 2.2) > 0).astype(np.float32)
    hard = hard * (0.55 + 0.45 * stripe[:, :, None])
    hard = np.clip(np.roll(hard, shift=(37, -23), axis=(0, 1)), 0.0, 1.0)
    variant_hard = Image.fromarray((hard * 255).astype(np.uint8), "RGB")

    import io

    buf = io.BytesIO()
    base.save(buf, format="PNG")
    buf.seek(0)
    base_png = Image.open(buf).convert("RGB")

    return {
        "base": base,
        "base_png": base_png,
        "variant_soft": variant_soft,
        "variant_hard": variant_hard,
    }
