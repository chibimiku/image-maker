"""Torchvision 预训练 VGG-19 多层特征编码器（Gram / AdaIN 共用）。

- 结构：``torchvision.models.vgg19`` 的 ``features``，ImageNet 预训练权重。
- 权重：本地 ``models/style-metrics/vgg19/vgg19-dcbb9e9d.pth``（SHA-256 见 config.WEIGHTS），
  与 ``torchvision.models.VGG19_Weights.IMAGENET1K_V1`` 同一份文件。
- 输出层：``relu1_1`` / ``relu2_1`` / ``relu3_1`` / ``relu4_1`` / ``relu5_1``，
  即 ``features`` 下标 1 / 6 / 11 / 20 / 29，与 Gatys 与 AdaIN 论文一致。
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torchvision

from .config import VGG19_LAYERS, WEIGHTS


class VGG19Encoder(nn.Module):
    """返回指定 relu 层特征图的 VGG-19 编码器（无分类头）。"""

    def __init__(self, layers: Iterable[str] | None = None, weights_path: str | None = None):
        super().__init__()
        self.layer_names = tuple(layers or VGG19_LAYERS.keys())
        unknown = [n for n in self.layer_names if n not in VGG19_LAYERS]
        if unknown:
            raise ValueError(f"未知的 VGG19 层: {unknown}")
        self._indices = {n: VGG19_LAYERS[n] for n in self.layer_names}
        self._max_index = max(self._indices.values())

        self.features = self._load_features(weights_path)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _load_features(weights_path: str | None) -> nn.Sequential:
        from .inventory import require_weight

        model = torchvision.models.vgg19(weights=None)
        # 缺失时抛 WeightsMissing（带官方 URL / 字节数 / SHA-256 / 修复命令），
        # 不要让它变成一句干巴巴的 FileNotFoundError。
        path = str(weights_path) if weights_path else str(require_weight("vgg19"))
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError as exc:
            from .inventory import WeightsMissing

            raise WeightsMissing("vgg19", f"读取失败（{exc}）", path=path) from exc
        # torchvision 的 checkpoint 就是纯 state_dict；兼容被包一层的情况。
        if isinstance(state, dict) and "state_dict" in state and "features.0.weight" not in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("classifier.")]
        unexpected = [k for k in unexpected if not k.startswith("classifier.")]
        if missing or unexpected:
            raise RuntimeError(
                f"VGG19 权重不匹配: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return model.features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        wanted = {v: k for k, v in self._indices.items()}
        h = x
        for idx, layer in enumerate(self.features):
            h = layer(h)
            if idx in wanted:
                out[wanted[idx]] = h
            if idx >= self._max_index:
                break
        return out

    def to_precision(self, precision: str) -> "VGG19Encoder":
        """精度开关。FP32/FP16 均为独立版本，FP16 不等价于原始 FP32 指标。"""
        if precision == "fp16" and next(self.parameters()).device.type != "cpu":
            self.features.half()
        elif precision == "fp32":
            self.features.float()
        else:
            raise ValueError(f"不支持的精度: {precision}")
        return self


_CACHE: dict[str, VGG19Encoder] = {}


def get_encoder(device: torch.device | str, precision: str = "fp32") -> VGG19Encoder:
    """按 (device, precision) 复用编码器实例，避免反复读 574MB 权重。"""
    key = f"{torch.device(device)}|{precision}"
    enc = _CACHE.get(key)
    if enc is None:
        enc = VGG19Encoder()
        enc = enc.to(device)
        if precision == "fp16":
            if torch.device(device).type == "cpu":
                raise ValueError("CPU 上不支持 fp16 VGG19 特征（未被验证）。")
            enc.to_precision("fp16")
        enc.eval()
        _CACHE[key] = enc
    return enc


def clear_cache() -> None:
    _CACHE.clear()
