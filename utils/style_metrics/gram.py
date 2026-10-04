"""Gram 风格距离（Gatys 等, arXiv:1508.06576）。

定义（本项目的固定口径，改动即视为改口径）：

对第 l 层特征 ``F_l``（形状 ``C_l x H_l x W_l``）：:

    G_l = (F_l @ F_l^T) / (C_l * H_l * W_l)          # 归一化 Gram，Gatys 式(4) 的 N=C*H*W
    d_l = ||G_l^a - G_l^b||_F^2 / (4 * C_l^2)        # Gatys 式(11) 的单层项

汇总标量 ``gram_distance`` = ``sum_l d_l``（对 config.VGG19_GRAM_LAYERS 五层求和）。

同时逐层报告两个无量纲量，便于人工判断：
- ``frobenius``        : ``||G_a - G_b||_F``
- ``relative``         : ``||G_a - G_b||_F / (||G_a||_F + ||G_b||_F + eps)``
- ``cosine_distance``  : ``1 - cos(vec(G_a), vec(G_b))``，取值 ``[0, 2]``

同图自比时三项全为 0（数学恒等），可用于验证实现与数值稳定性。
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

from .config import VGG19_GRAM_LAYERS

_EPS = 1e-12


def gram_matrix(feat: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """``feat``: NCHW -> NG 的 Gram 矩阵（每张图一个）。"""
    if feat.dim() != 4:
        raise ValueError(f"Gram 需要 NCHW 特征，收到 {tuple(feat.shape)}")
    n, c, h, w = feat.shape
    flat = feat.reshape(n, c, h * w)
    g = torch.bmm(flat, flat.transpose(1, 2))
    if normalize:
        g = g / (c * h * w)
    return g


def _layer_terms(ga: torch.Tensor, gb: torch.Tensor) -> dict[str, float]:
    diff = ga - gb
    fro = torch.linalg.vector_norm(diff, dim=(1, 2))
    na = torch.linalg.vector_norm(ga, dim=(1, 2))
    nb = torch.linalg.vector_norm(gb, dim=(1, 2))
    rel = fro / (na + nb + _EPS)
    cos = torch.nn.functional.cosine_similarity(
        ga.flatten(1) + _EPS, gb.flatten(1) + _EPS, dim=1
    )
    return {
        "frobenius": float(fro.mean()),
        "relative": float(rel.mean()),
        "cosine_distance": float((1.0 - cos).mean()),
    }


def gram_distance(
    feats_a: Mapping[str, torch.Tensor],
    feats_b: Mapping[str, torch.Tensor],
    layers: Iterable[str] | None = None,
    normalize: bool = True,
) -> dict:
    """多层 Gram 风格距离。返回 ``{"distance": float, "layers": {...}}``。"""
    layers = tuple(layers or VGG19_GRAM_LAYERS)
    per_layer: dict[str, dict] = {}
    total = torch.zeros((), dtype=torch.float64)
    for name in layers:
        if name not in feats_a or name not in feats_b:
            raise KeyError(f"特征里缺少层 {name}")
        fa, fb = feats_a[name], feats_b[name]
        if fa.shape != fb.shape:
            raise ValueError(f"层 {name} 特征形状不一致: {tuple(fa.shape)} vs {tuple(fb.shape)}")
        c = fa.shape[1]
        ga = gram_matrix(fa.float(), normalize=normalize)
        gb = gram_matrix(fb.float(), normalize=normalize)
        diff = (ga - gb).double()
        contribution = torch.linalg.vector_norm(diff, dim=(1, 2)) ** 2 / (4.0 * c * c)
        total = total + contribution.mean().cpu()
        per_layer[name] = {
            "channels": int(c),
            "contribution": float(contribution.mean().cpu()),
            **_layer_terms(ga.double(), gb.double()),
        }
    # 尺度无关的伴随指标：Gatys 标量受 (H*W)^2 与 C^2 影响，数量级很小（1e-7~1e-3），
    # 单看它不直观，所以同时给出可比较的余弦距离。
    cos_vals = [v["cosine_distance"] for v in per_layer.values()]
    rel_vals = [v["relative"] for v in per_layer.values()]
    summary = {
        "mean_layer_cosine_distance": float(sum(cos_vals) / len(cos_vals)) if cos_vals else None,
        "max_layer_relative": max(rel_vals) if rel_vals else None,
        "scale_note": (
            "gram_distance 是 Gatys 式(11) 单层项对五层求和，量级随层分辨率平方衰减；"
            "判读相对差异请用 mean_layer_cosine_distance / max_layer_relative。"
        ),
    }
    return {"distance": float(total), "layers": per_layer, "summary": summary}
