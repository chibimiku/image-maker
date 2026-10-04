"""LPIPS 感知距离（Zhang 等, CVPR 2018）——作者官方实现 + 官方校准权重。

严格约束（不得违反）：

- 只用 ``richzhang/PerceptualSimilarity`` 的官方实现（PyPI ``lpips`` 包，版本 0.1.4）
  与作者发布的 **v0.1 校准权重** ``lpips/weights/v0.1/alex.pth``；
- AlexNet 骨架必须是 ImageNet 预训练（``AlexNet_Weights.IMAGENET1K_V1``），
  本地文件 ``models/style-metrics/lpips/trunk/alexnet-owt-7be5be79.pth``，SHA-256 已核对；
- **禁止**用随机权重、VGG 特征 L2、或任意「未校准特征距离」冒充 LPIPS。

输入契约：RGB，缩放到 256x256，取值 ``[-1, 1]``（官方 ``normalize=False`` 分支）。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from .config import LPIPS_INPUT_SIZE, LPIPS_NET, LPIPS_VERSION, WEIGHTS, torch_home
from .imaging import to_lpips

_NET_KEY = {"alex": "lpips_alex", "vgg": "lpips_vgg"}


def _prepare_env() -> None:
    """把 torchvision 的下载根固定到工作区，避免回落到用户主目录。"""
    os.environ["TORCH_HOME"] = str(torch_home())
    # 官方 lpips 0.1.4 用 tv.alexnet(pretrained=True) 这个 0.13 起就废弃的 kwarg。
    # 行为等价于 weights=AlexNet_Weights.IMAGENET1K_V1（官方语义，未改变权重选择），
    # 只是会打印 UserWarning，这里压掉以免污染 CLI 输出；不改动 lpips 的实现。
    warnings.filterwarnings("ignore", message=".*pretrained.*is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")


class LPIPSMetric:
    """官方 LPIPS 模块的薄封装（CPU / CUDA 张量路径）。"""

    def __init__(
        self,
        net: str = LPIPS_NET,
        version: str = LPIPS_VERSION,
        weights_path: str | Path | None = None,
        device: torch.device | str = "cpu",
        precision: str = "fp32",
    ):
        _prepare_env()
        import lpips  # 官方包；import 失败说明环境缺依赖

        key = _NET_KEY.get(net)
        if key is None:
            raise ValueError(f"未登记的 LPIPS 骨干: {net}")
        self.net = net
        self.version = version
        # 缺失时抛 WeightsMissing（带官方 URL / 字节数 / SHA-256 / 修复命令）
        from .inventory import require_weight

        self.weights_path = Path(weights_path) if weights_path else require_weight(key)
        if not self.weights_path.exists():
            from .inventory import WeightsMissing

            raise WeightsMissing(key, "文件不存在", path=self.weights_path)
        # LPIPS 是「校准权重 + ImageNet 骨架」两件套，缺骨架同样要早报
        require_weight("lpips_alex_trunk")

        self.device = torch.device(device)
        self.precision = precision
        self.model = lpips.LPIPS(
            pretrained=True,
            net=net,
            version=version,
            lpips=True,
            spatial=False,
            pnet_rand=False,  # 必须是 ImageNet 预训练骨架，不许随机
            pnet_tune=False,
            model_path=str(self.weights_path),
            eval_mode=True,
            verbose=False,
        ).to(self.device)
        if precision == "fp16":
            if self.device.type == "cpu":
                raise ValueError("CPU 上不使用 fp16 LPIPS（未验证，不允许宣称等价）。")
            self.model.half()
        elif precision != "fp32":
            raise ValueError(f"不支持的精度: {precision}")
        self.model.eval()

    # ------------------------------------------------------------------ #
    def forward_tensors(self, a: torch.Tensor, b: torch.Tensor, ret_per_layer: bool = False):
        a = a.to(self.device)
        b = b.to(self.device)
        if self.precision == "fp16":
            a, b = a.half(), b.half()
        with torch.no_grad():
            out = self.model(a, b, retPerLayer=ret_per_layer)
        return out

    def distance_images(self, img_a: Image.Image, img_b: Image.Image, size: int = LPIPS_INPUT_SIZE):
        """返回 ``(distance, per_layer)``；per_layer 为五层标量列表（官方口径）。"""
        pa = to_lpips([img_a], size)
        pb = to_lpips([img_b], size)
        val, res = self.forward_tensors(pa, pb, ret_per_layer=True)
        per_layer = [float(r.mean().float().cpu()) for r in res]
        return float(val.mean().float().cpu()), per_layer

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "richzhang/PerceptualSimilarity (PyPI lpips)",
            "calibrated_weights": str(self.weights_path),
            "backbone": f"torchvision {self.net} (ImageNet IMAGENET1K_V1)",
            "version": self.version,
            "input_size": LPIPS_INPUT_SIZE,
        }


def lpips_distance_pairs(
    pairs: Sequence[tuple[Image.Image, Image.Image]],
    device: torch.device | str = "cpu",
    precision: str = "fp32",
) -> list[float]:
    """批量（逐对）计算 LPIPS，保留官方逐图前景口径（不做跨图平均）。"""
    metric = get_metric(device=device, precision=precision)
    return [metric.distance_images(a, b)[0] for a, b in pairs]


_CACHE: dict[str, "LPIPSMetric"] = {}


def get_metric(
    net: str = LPIPS_NET,
    device: torch.device | str = "cpu",
    precision: str = "fp32",
    weights_path: str | Path | None = None,
) -> "LPIPSMetric":
    """复用 LPIPS 实例。

    不缓存的话每次调用都要重新读校准权重 + 244MB 的 AlexNet 骨架，
    测出来的「耗时」其实是磁盘 IO 而不是前向计算。
    """
    key = f"{net}|{torch.device(device)}|{precision}|{weights_path}"
    metric = _CACHE.get(key)
    if metric is None:
        metric = LPIPSMetric(
            net=net, device=device, precision=precision, weights_path=weights_path
        )
        _CACHE[key] = metric
    return metric


def clear_cache() -> None:
    _CACHE.clear()


class NPULPIPSMetric:
    """LPIPS 的 NPU 执行包装：官方模块 → ONNX → OpenVINO NPU。

    - 权重仍是作者官方校准权重；**没有**换成随机权重或别的距离。
    - NPU 只支持 FP16，因此结果是**独立精度版本**，与 CPU fp32 不等价（差异见验证 JSON）。
    - 预处理与 torch 路径完全一致（256x256、[-1,1]）。
    """

    def __init__(self, input_size: int = LPIPS_INPUT_SIZE, onnx_path: str | Path | None = None):
        from .config import ONNX_CACHE
        from .openvino_backend import NPUEncoder, export_lpips_onnx

        self.input_size = input_size
        path = Path(onnx_path) if onnx_path else ONNX_CACHE / f"lpips_alex_{input_size}.onnx"
        self.onnx_path = export_lpips_onnx(path, input_size=input_size)
        self.encoder = NPUEncoder(self.onnx_path)
        self.encoder.compile()
        self.encoder.input_name = "a"  # infer() 只喂单输入；LPIPS 需要两个，下面单独处理

    def distance_images(self, img_a: Image.Image, img_b: Image.Image) -> tuple[float, list[float]]:
        import numpy as np

        pa = to_lpips([img_a], self.input_size).numpy().astype("float32")
        pb = to_lpips([img_b], self.input_size).numpy().astype("float32")
        compiled = self.encoder.compiled
        if self.encoder._request is None:  # noqa: SLF001
            self.encoder._request = compiled.create_infer_request()  # noqa: SLF001
        result = self.encoder._request.infer(  # noqa: SLF001
            {"a": np.ascontiguousarray(pa), "b": np.ascontiguousarray(pb)}
        )
        dist = float(np.asarray(result["dist"]).reshape(-1)[0])
        per_layer = [float(np.asarray(result[f"layer{i}"]).reshape(-1)[0]) for i in range(5)]
        return dist, per_layer

    def describe(self) -> dict:
        return {
            "implementation": "richzhang/PerceptualSimilarity 官方模块 → ONNX → OpenVINO",
            "execution": self.encoder.describe(),
            "precision": "fp16（NPU 硬件限制；独立版本，不等价于 CPU fp32）",
        }
