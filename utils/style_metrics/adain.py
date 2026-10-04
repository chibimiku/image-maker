"""AdaIN 特征统计距离（Huang & Belongie, arXiv:1703.06868）。

AdaIN 只把风格当成「深度特征的通道均值与标准差」。本项目**不做风格迁移解码器**，
只用同一套 VGG-19 特征把两张图的统计量拿出来比较：:

    mu_l   = mean_{h,w} F_l          # R^{C_l}
    sigma_l= std_{h,w}  F_l          # R^{C_l}，无偏=False（torch 默认）
    d_l    = ||mu_a - mu_b||_2 + ||sigma_a - sigma_b||_2

汇总标量 ``adain_distance`` = ``sum_l d_l``（config.VGG19_ADAIN_LAYERS）。

逐层同时报告 ``mean_l2`` / ``std_l2`` / ``mean_cos_distance`` / ``std_cos_distance``，
便于区分是「整体色调偏移」还是「对比/质感变化」。

注意：这是深度特征统计量的距离，**不是**原图 RGB 均值差，两者不可互换引用。
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

from .config import VGG19_ADAIN_LAYERS

_EPS = 1e-12


def feature_stats(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NCHW -> (mu, sigma)，均为 N x C。"""
    if feat.dim() != 4:
        raise ValueError(f"需要 NCHW 特征，收到 {tuple(feat.shape)}")
    x = feat.float()
    mu = x.mean(dim=(2, 3))
    sigma = x.std(dim=(2, 3), unbiased=False)
    return mu, sigma


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a + _EPS, b + _EPS, dim=1)


def adain_distance(
    feats_a: Mapping[str, torch.Tensor],
    feats_b: Mapping[str, torch.Tensor],
    layers: Iterable[str] | None = None,
) -> dict:
    """多层 AdaIN 统计距离。返回 ``{"distance": float, "layers": {...}}``。"""
    layers = tuple(layers or VGG19_ADAIN_LAYERS)
    per_layer: dict[str, dict] = {}
    total = torch.zeros((), dtype=torch.float64)
    for name in layers:
        if name not in feats_a or name not in feats_b:
            raise KeyError(f"特征里缺少层 {name}")
        fa, fb = feats_a[name], feats_b[name]
        if fa.shape != fb.shape:
            raise ValueError(f"层 {name} 特征形状不一致: {tuple(fa.shape)} vs {tuple(fb.shape)}")
        mua, sga = feature_stats(fa)
        mub, sgb = feature_stats(fb)
        mean_l2 = torch.linalg.vector_norm(mua.double() - mub.double(), dim=1)
        std_l2 = torch.linalg.vector_norm(sga.double() - sgb.double(), dim=1)
        total = total + (mean_l2 + std_l2).mean().cpu()
        per_layer[name] = {
            "channels": int(fa.shape[1]),
            "mean_l2": float(mean_l2.mean().cpu()),
            "std_l2": float(std_l2.mean().cpu()),
            "mean_cos_distance": float(_cos_dist(mua, mub).mean().cpu()),
            "std_cos_distance": float(_cos_dist(sga, sgb).mean().cpu()),
            "contribution": float((mean_l2 + std_l2).mean().cpu()),
        }
    return {"distance": float(total), "layers": per_layer}
