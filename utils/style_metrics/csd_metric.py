"""CSD 画风描述子余弦相似度（Somepalli 等, arXiv:2404.01292）。

官方来源与结构（逐条对照 ``github.com/learn2phoenix/CSD``）：

- ``CSD/model.py`` 的 ``CSD_CLIP(name='vit_large', content_proj_head='default')``：
  ``backbone = clip.load("ViT-L/14").visual``（OpenAI 官方 CLIP），随后 ``backbone.proj = None``。
- ``CSD/utils.py::extract_features(..., eval_embed='head')`` 取的是 **style 头**：
  ``style_output = normalize(feature @ last_layer_style, dim=1, p=2)``。
- ``main_sim.py`` 里 CSD 的 ``preprocess = transforms_branch0``（``CSD/loss_utils.py``）。
- 检索口径 ``search.py --method IP`` ⇒ 描述子已 L2 归一化，**相似度 = 内积 = 余弦相似度**。

**这里绝不用普通 CLIP / OpenCLIP / ResNet 冒充 CSD**：本模块必须同时具备
(a) OpenAI CLIP ViT-L/14 的 visual 骨架、(b) CSD 官方训练出来的 ``last_layer_style``。

与仓库 HEAD 版 ``model.py`` 的一处**实测差异**（必须记录，不能糊过去）：

``model.py`` 写的是 ``self.last_layer_style = copy.deepcopy(self.backbone.proj)``
（即一个 ``nn.Linear(1024, 768)``），但官方权重里 ``last_layer_style`` 存的是**裸张量、
形状 ``(1024, 768)``**（(in, out)），而且键里**根本没有 ``backbone.proj``**。
实测 ``torch.Tensor @ nn.Linear`` 会直接 ``TypeError``（``nn.Module`` 没有 ``__rmatmul__``），
所以 HEAD 版 ``model.py`` 与官方 checkpoint 并不自洽；若照抄并用 ``strict=False`` 加载，
尺寸不匹配会被静默跳过、style 头退化成 CLIP 原始 ``proj`` —— 那就成了「拿普通 CLIP 冒充 CSD」。
因此这里按 **checkpoint 的真实形状**建 ``nn.Parameter(1024, 768)``，并以 ``strict=True``
加载；加载不干净就直接报错，绝不带着未加载的头部继续算。

官方权重来自 Hugging Face ``tomg-group-umd/CSD-ViT-L``：

- 字节数 2438228893，SHA-256 ``40e92fad…``（与 HF LFS 元数据一致，已本地复核）。
- 许可：模型卡写 **CC-BY-4.0**；代码仓库 LICENSE 是 **MIT**，而 ``CSD/model.py`` 的
  ``PyTorchModelHubMixin(license="mit")`` 又写 MIT —— 权重与代码口径冲突，如实并记。
- 上游 README 顶部仍挂 DISCLAIMER：上传权重与论文数字有出入（issue #14 至今 open）。
  本模块只保证「按官方权重与官方预处理复现描述子」，**不声称复现论文指标**。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from .config import MODEL_ROOT, WEIGHTS, torch_home
from .runner import MetricOutcome

CSD_DIR = MODEL_ROOT / "csd"
DEFAULT_CSD_WEIGHTS = WEIGHTS["csd"]["path"]
CLIP_DOWNLOAD_ROOT = WEIGHTS["clip_vit_l14"]["path"].parent

#: 官方 transforms_branch0（CSD/loss_utils.py，逐字对齐）
CSD_SIZE = 224
CSD_MEAN = (0.48145466, 0.4578275, 0.40821073)
CSD_STD = (0.26862954, 0.26130258, 0.27577711)

#: 官方权重（HF LFS 元数据 + 本地实测一致）；这里从 config.WEIGHTS 单点取值，
#: 避免同一个哈希散落两处、改一处忘一处。
CSD_WEIGHTS_BYTES = WEIGHTS["csd"]["bytes"]
CSD_WEIGHTS_SHA256 = WEIGHTS["csd"]["sha256"]
CSD_WEIGHTS_URL = WEIGHTS["csd"]["source"]
CLIP_WEIGHTS_BYTES = WEIGHTS["clip_vit_l14"]["bytes"]

#: OpenAI CLIP ViT-L/14 骨架（文件名里的 sha256 就是官方校验值，已本地复核一致）
CLIP_VIT_L14_SHA256 = WEIGHTS["clip_vit_l14"]["sha256"]
CLIP_VIT_L14_URL = WEIGHTS["clip_vit_l14"]["source"]

EMBED_DIM = 1024
STYLE_DIM = 768


class CSDUnavailable(RuntimeError):
    """官方权重/依赖缺失；由调用方转成 status="unavailable"，不静默换模型。"""


# --------------------------------------------------------------------------- #
# 官方架构复刻
# --------------------------------------------------------------------------- #


def _convert_weights_float(model: nn.Module) -> None:
    """官方 ``CSD/utils.py::convert_weights_float``：把 CLIP 的 fp16 权重拉回 fp32。"""

    def _convert(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()
        for name in ("text_projection", "proj"):
            attr = getattr(l, name, None)
            if attr is not None:
                attr.data = attr.data.float()

    model.apply(_convert)


class CSDCLIP(nn.Module):
    """``CSD/model.py::CSD_CLIP`` 的推理复刻（按 checkpoint 真实参数形状）。"""

    def __init__(self, clip_visual: nn.Module | None = None, load_clip: bool = True):
        super().__init__()
        if clip_visual is None:
            if not load_clip:
                raise ValueError("必须提供 CLIP visual 骨架")
            import clip as openai_clip

            from .inventory import require_weight

            # 骨架缺失时给同样的可操作报错；CLIP 官方下载器只会在导入 torch 后崩，
            # 这里先做存在性检查，报错信息才带得上来源 URL 与修复命令。
            require_weight("clip_vit_l14")
            clip_model, _ = openai_clip.load(
                "ViT-L/14", device="cpu", download_root=str(CLIP_DOWNLOAD_ROOT)
            )
            clip_visual = clip_model.visual
        self.backbone = clip_visual
        _convert_weights_float(self.backbone)
        # 官方 CSD/model.py：去掉 CLIP 自带的对比投影（官方权重里也没有 backbone.proj）
        self.backbone.proj = None
        self.embedding_dim = EMBED_DIM
        # 关键：官方 checkpoint 里这是裸 (1024, 768)（in, out）矩阵，不是 nn.Linear。
        self.last_layer_style = nn.Parameter(torch.zeros(EMBED_DIM, STYLE_DIM))
        self.last_layer_content = nn.Parameter(torch.zeros(EMBED_DIM, STYLE_DIM))

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)  # (N, 1024)，proj 已置空

    def style_descriptor(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        feature = self.features(images)
        style = feature @ self.last_layer_style
        if normalize:
            style = nn.functional.normalize(style, dim=1, p=2)
        return style

    def forward(self, images: torch.Tensor):
        feature = self.features(images)
        style = nn.functional.normalize(feature @ self.last_layer_style, dim=1, p=2)
        content = nn.functional.normalize(feature @ self.last_layer_content, dim=1, p=2)
        return feature, content, style


# --------------------------------------------------------------------------- #
# 权重加载
# --------------------------------------------------------------------------- #


def load_csd_state_dict(path: str | Path | None = None) -> tuple[dict, dict]:
    """读取官方 checkpoint：``{'model_state_dict': ...}``，去掉 ``module.`` 前缀。

    缺失时抛 :class:`~utils.style_metrics.inventory.WeightsMissing`（带来源与修复命令），
    **不会**退化成任何其它模型。
    """
    from .inventory import require_weight

    resolved = Path(path) if path else require_weight("csd")
    if not resolved.exists():
        from .inventory import WeightsMissing

        raise WeightsMissing("csd", "文件不存在", path=resolved)
    path = resolved
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
        meta["format"] = "training checkpoint (model_state_dict)"
        meta["checkpoint_top_keys"] = sorted(blob.keys())
        meta["iter"] = blob.get("iter")
        args = blob.get("args")
        if args is not None:
            meta["train_args"] = {
                k: str(getattr(args, k)) for k in ("arch", "epochs", "batch_size") if hasattr(args, k)
            }
    elif isinstance(blob, dict):
        state = blob
        meta["format"] = "raw state_dict"
    else:
        raise CSDUnavailable(f"无法识别的 CSD 权重格式: {type(blob)}")

    cleaned = {}
    prefixes = set()
    for k, v in state.items():
        if k.startswith("module."):
            prefixes.add("module.")
            k = k.replace("module.", "", 1)
        cleaned[k] = v
    meta["stripped_prefixes"] = sorted(prefixes)
    meta["num_tensors"] = len(cleaned)
    return cleaned, meta


# --------------------------------------------------------------------------- #
# 指标封装
# --------------------------------------------------------------------------- #


class CSDMetric:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: torch.device | str = "cpu",
        precision: str = "fp32",
        verify_weights: bool = False,
    ):
        os.environ["TORCH_HOME"] = str(torch_home())
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_CSD_WEIGHTS
        if not self.weights_path.exists():
            from .inventory import WeightsMissing

            raise WeightsMissing("csd", "文件不存在", path=self.weights_path)
        self.device = torch.device(device)
        self.precision = precision
        self.weights_ok: bool | None = None
        if verify_weights:
            from .inventory import sha256_file

            digest = sha256_file(self.weights_path)
            self.weights_ok = digest == CSD_WEIGHTS_SHA256
            if not self.weights_ok:
                raise CSDUnavailable(
                    f"CSD 权重 SHA-256 不符：{digest} != {CSD_WEIGHTS_SHA256}"
                )

        state, meta = load_csd_state_dict(self.weights_path)
        self.checkpoint_meta = meta
        self.model = CSDCLIP()
        # strict=True：缺键/多键/尺寸不符一律报错，避免静默退化成普通 CLIP。
        msg = self.model.load_state_dict(state, strict=True)
        self.load_msg = {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}
        if msg.missing_keys or msg.unexpected_keys:  # pragma: no cover - strict=True 时不会走到
            raise CSDUnavailable(f"CSD 权重未干净加载: {self.load_msg}")
        self.model = self.model.to(self.device).eval()
        if precision == "fp16" and self.device.type != "cpu":
            self.model.half()

    # ---------------------------------------------------------------- #
    def descriptor(self, images: Sequence) -> torch.Tensor:
        from .imaging import to_csd

        batch = to_csd(images, CSD_SIZE, CSD_MEAN, CSD_STD).to(self.device)
        if self.precision == "fp16" and self.device.type != "cpu":
            batch = batch.half()
        with torch.no_grad():
            desc = self.model.style_descriptor(batch)
        return desc.float().cpu()

    def cosine(self, img_a, img_b) -> tuple[float, torch.Tensor, torch.Tensor]:
        da = self.descriptor([img_a])
        db = self.descriptor([img_b])
        # 官方描述子已 L2 归一化，内积即余弦（search.py --method IP）；这里再归一化一次防 fp16 漂移。
        da = nn.functional.normalize(da, dim=1)
        db = nn.functional.normalize(db, dim=1)
        return float(torch.sum(da * db, dim=1).mean()), da, db

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "learn2phoenix/CSD（CSD/model.py::CSD_CLIP 复刻，按 checkpoint 真实形状）",
            "backbone": "OpenAI CLIP ViT-L/14 visual（官方 clip.load，proj 置空）",
            "style_head": "CSD 官方 last_layer_style，裸参数 (1024,768)，L2 归一化",
            "weights": str(self.weights_path),
            "weights_bytes": self.weights_path.stat().st_size if self.weights_path.exists() else None,
            "weights_sha256_expected": CSD_WEIGHTS_SHA256,
            "weights_source": CSD_WEIGHTS_URL,
            "clip_backbone_sha256": CLIP_VIT_L14_SHA256,
            "checkpoint": self.checkpoint_meta,
            "load_state_dict": self.load_msg,
            "descriptor_dim": STYLE_DIM,
            "similarity": "cosine / inner product（官方 search.py --method IP）",
            "preprocessing": {
                "resize": f"Resize({CSD_SIZE}, BICUBIC) 短边 + CenterCrop({CSD_SIZE})",
                "mean": list(CSD_MEAN),
                "std": list(CSD_STD),
                "note": "CLIP 归一化，不是 ImageNet 归一化",
            },
            "license_note": "代码 MIT / 模型卡 CC-BY-4.0（上游口径冲突，两者并列）",
            "upstream_caveat": (
                "官方 README 顶部 DISCLAIMER：上传权重与论文报告数字有出入（issue #14 仍 open）。"
                "本实现只声称复现官方描述子，不声称复现论文指标。"
            ),
        }


# --------------------------------------------------------------------------- #
# runner 接口
# --------------------------------------------------------------------------- #


class NPUCSDMetric:
    """CSD 的 NPU 执行包装：官方 style 描述子编码器 → ONNX → OpenVINO NPU。

    分工如实记录：**描述子编码器在 NPU，余弦相似度在 CPU**（只是 768 维点积）。
    精度是 FP16（NPU 硬件限制），属独立版本，与 CPU fp32 不等价。
    """

    def __init__(self, input_size: int = CSD_SIZE, onnx_path: str | Path | None = None):
        from .config import ONNX_CACHE
        from .openvino_backend import NPUEncoder, export_csd_onnx

        self.input_size = input_size
        path = Path(onnx_path) if onnx_path else ONNX_CACHE / f"csd_vit_l_{input_size}.onnx"
        self.onnx_path = export_csd_onnx(path, input_size=input_size)
        self.encoder = NPUEncoder(self.onnx_path)
        self.encoder.compile()

    def descriptor(self, images: Sequence) -> torch.Tensor:
        import numpy as np

        from .imaging import to_csd

        arr = to_csd(images, self.input_size, CSD_MEAN, CSD_STD).numpy().astype("float32")
        res = self.encoder.infer(arr)
        desc = torch.from_numpy(np.asarray(list(res.values())[0], dtype=np.float32))
        return nn.functional.normalize(desc, dim=1)

    def cosine(self, img_a, img_b) -> tuple[float, torch.Tensor, torch.Tensor]:
        da = self.descriptor([img_a])
        db = self.descriptor([img_b])
        return float(torch.sum(da * db, dim=1).mean()), da, db

    def describe(self) -> dict:
        return {
            "implementation": "learn2phoenix/CSD style 描述子编码器 → ONNX → OpenVINO",
            "execution": self.encoder.describe(),
            "precision": "fp16（NPU 硬件限制；独立版本，不等价于 CPU fp32）",
            "split": "描述子编码器在 NPU，余弦相似度在 CPU",
        }


_CACHE: dict[str, "CSDMetric"] = {}


def get_metric(
    weights_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    precision: str = "fp32",
    verify_weights: bool = False,
) -> "CSDMetric":
    """复用 CSD 实例（权重 2.44 GB，每次重读会把耗时测成磁盘 IO）。"""
    key = f"{weights_path or DEFAULT_CSD_WEIGHTS}|{torch.device(device)}|{precision}"
    metric = _CACHE.get(key)
    if metric is None:
        metric = CSDMetric(
            weights_path=weights_path,
            device=device,
            precision=precision,
            verify_weights=verify_weights,
        )
        _CACHE[key] = metric
    return metric


def clear_cache() -> None:
    _CACHE.clear()


def compare(img_a, img_b, backend) -> MetricOutcome:
    from .devices import torch_device

    if backend.actual == "npu":
        metric = NPUCSDMetric()
        value, da, db = metric.cosine(img_a, img_b)
        return MetricOutcome(
            "csd",
            "ok",
            value,
            {
                **metric.describe(),
                "descriptor_dim": STYLE_DIM,
                "descriptor_a_norm": float(torch.linalg.vector_norm(da)),
                "descriptor_b_norm": float(torch.linalg.vector_norm(db)),
                "weights_sha256_expected": CSD_WEIGHTS_SHA256,
            },
        )
    device = torch_device(backend) or torch.device("cpu")
    metric = get_metric(device=device, precision=backend.precision)
    value, da, db = metric.cosine(img_a, img_b)
    return MetricOutcome(
        "csd",
        "ok",
        value,
        {
            **metric.describe(),
            "descriptor_a_norm": float(torch.linalg.vector_norm(da)),
            "descriptor_b_norm": float(torch.linalg.vector_norm(db)),
        },
    )


def cli_report(weights_path: str | Path | None = None) -> dict:
    """只报告 CSD 权重与依赖状态，不做前向（供 CLI 的 --inventory 用）。"""
    path = Path(weights_path or DEFAULT_CSD_WEIGHTS)
    out: dict = {
        "weights": str(path),
        "exists": path.exists(),
        "expected_bytes": CSD_WEIGHTS_BYTES,
        "expected_sha256": CSD_WEIGHTS_SHA256,
        "source_url": CSD_WEIGHTS_URL,
        "clip_backbone_url": CLIP_VIT_L14_URL,
        "clip_backbone_expected_sha256": CLIP_VIT_L14_SHA256,
    }
    if path.exists():
        out["bytes"] = path.stat().st_size
        out["size_matches"] = out["bytes"] == CSD_WEIGHTS_BYTES
    try:
        import clip

        out["openai_clip_import"] = "ok"
        out["openai_clip_module"] = clip.__file__
        out["has_load"] = hasattr(clip, "load")
        out["has_tokenize"] = hasattr(clip, "tokenize")
    except Exception as exc:
        out["openai_clip_import"] = f"FAIL {type(exc).__name__}: {exc}"
    return out
