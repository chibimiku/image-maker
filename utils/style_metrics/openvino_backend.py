"""NPU 执行路径：把 PyTorch 特征编码器导出 ONNX，交给 OpenVINO 编译到 Intel NPU。

本机 NPU 是 **Intel(R) AI Boost**（OpenVINO 的 ``NPU`` 设备，驱动版本见
``devices.hardware_report``）。它不是 Qualcomm HTP，也不走 DirectML ——
所以这里用 OpenVINO 的 NPU 插件，而不是 ONNX Runtime QNN/DML。

关键纪律：

1. 「OpenVINO 列出了 NPU 设备」「ONNX 导出成功」「模型编译成功」都不等于 NPU 推理成功；
   只有真实 infer 出有限数值、并且运行时的 ``EXECUTION_DEVICES`` 指向 NPU 才算。
2. NPU 只支持 **FP16/INT8**。因此 NPU 结果是与 CPU FP32 **不等价**的独立版本，
   所有输出里都必须带 ``precision=fp16`` 标记。
3. 特征编码器在 NPU 上跑、Gram/AdaIN 距离在 CPU 上算——这种分工必须如实记录，
   不能含糊成「整个指标跑在 NPU 上」。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import MODEL_ROOT, ONNX_CACHE, VGG19_LAYERS
from .devices import _as_list

# --------------------------------------------------------------------------- #
# ONNX 导出
# --------------------------------------------------------------------------- #


def export_vgg19_onnx(
    out_path: str | Path,
    layers: Sequence[str] | None = None,
    input_size: int = 512,
    opset: int = 17,
    force: bool = False,
) -> Path:
    """把 VGG-19 多层特征编码器导出成 ONNX（固定 batch=1、固定输入尺寸）。

    只导出**特征编码器**（relu 中间层），不含分类 logits。
    """
    import torch

    from .vgg_encoder import VGG19Encoder

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = tuple(layers or VGG19_LAYERS.keys())
    enc = VGG19Encoder(layers=names).eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        torch.onnx.export(
            enc,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=list(names),
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    return out_path


def export_lpips_onnx(
    out_path: str | Path,
    input_size: int = 256,
    net: str = "alex",
    version: str = "0.1",
    opset: int = 17,
    force: bool = False,
) -> Path:
    """把**作者官方 LPIPS 模块**整体导出成 ONNX（含 ScalingLayer 与校准的 lin 层）。

    导出的仍是官方实现与官方校准权重，只是换个执行后端；不是重新实现一个
    「长得像 LPIPS」的距离。输入是两张图（a/b），输出是五个逐层标量 + 总和。
    """
    import torch

    from .lpips_metric import LPIPSMetric

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric = LPIPSMetric(net=net, version=version, device="cpu", precision="fp32")
    model = metric.model.eval()

    class _Wrap(torch.nn.Module):
        """把 (dist, per_layer) 摊平成多个输出，便于 ONNX / NPU 取用。"""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, a, b):
            dist, per_layer = self.inner(a, b, retPerLayer=True)
            flat = [p.reshape(1) for p in per_layer]
            return (dist.reshape(1), *flat)

    wrap = _Wrap(model).eval()
    dummy_a = torch.zeros(1, 3, input_size, input_size)
    dummy_b = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        torch.onnx.export(
            wrap,
            (dummy_a, dummy_b),
            str(out_path),
            input_names=["a", "b"],
            output_names=["dist", "layer0", "layer1", "layer2", "layer3", "layer4"],
            opset_version=opset,
            dynamo=False,
        )
    return out_path


def export_csd_onnx(
    out_path: str | Path,
    input_size: int = 224,
    weights_path: str | Path | None = None,
    opset: int = 17,
    force: bool = False,
) -> Path:
    """把 CSD 的 **style 描述子编码器**（ViT-L/14 visual + 官方 style 头）导出成 ONNX。

    只导出描述子通路（输出 768 维），不导出 content 头、不导出任何分类 logits。
    """
    import torch

    from .csd_metric import CSDMetric

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric = CSDMetric(weights_path=weights_path, device="cpu", precision="fp32")
    inner = metric.model.eval()

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.style_descriptor(x)

    wrap = _Wrap(inner).eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        torch.onnx.export(
            wrap,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["descriptor"],
            opset_version=opset,
            dynamo=False,
        )
    return out_path


# --------------------------------------------------------------------------- #
# 编译 / 推理
# --------------------------------------------------------------------------- #


@dataclass
class NPUEncoder:
    """OpenVINO 编译后的特征编码器，带可核验的执行证据。"""

    onnx_path: Path
    device: str = "NPU"
    execution_devices: list = field(default_factory=list)
    compile_seconds: float = 0.0
    input_name: str = "input"
    output_names: list = field(default_factory=list)
    device_name: str = ""
    property_evidence: dict = field(default_factory=dict)
    _compiled: Any = None
    _request: Any = None

    def compile(self, extra_props: dict | None = None) -> "NPUEncoder":
        import openvino as ov

        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if extra_props:
            cfg.update(extra_props)
        t0 = time.perf_counter()
        self._compiled = core.compile_model(str(self.onnx_path), self.device, cfg)
        self.compile_seconds = time.perf_counter() - t0
        try:
            self.execution_devices = _as_list(self._compiled.get_property("EXECUTION_DEVICES"))
        except Exception:
            self.execution_devices = []
        try:
            self.device_name = str(core.get_property(self.device, "FULL_DEVICE_NAME"))
        except Exception:
            self.device_name = ""
        for key in ("NPU_DRIVER_VERSION", "DEVICE_TYPE", "OPTIMIZATION_CAPABILITIES", "NPU_COMPILER_VERSION"):
            try:
                self.property_evidence[key] = str(core.get_property(self.device, key))
            except Exception:
                pass
        for key in ("INFERENCE_PRECISION_HINT", "PERFORMANCE_HINT", "EXECUTION_DEVICES"):
            try:
                self.property_evidence["compiled:" + key] = str(self._compiled.get_property(key))
            except Exception:
                pass
        self._request = self._compiled.create_infer_request()
        return self

    @property
    def compiled(self):
        if self._compiled is None:
            self.compile()
        return self._compiled

    def infer(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """返回 ``{层名: 特征数组}``（NCHW float32，已去掉 fp16 的数值包装）。"""
        compiled = self.compiled
        if self._request is None:
            self._request = compiled.create_infer_request()
        result = self._request.infer({self.input_name: np.ascontiguousarray(x, dtype=np.float32)})
        compiled_outputs = compiled.outputs
        out: dict[str, np.ndarray] = {}
        if not self.output_names:
            self.output_names = [o.get_any_name() for o in compiled_outputs]
        for name in self.output_names:
            out[name] = np.asarray(result[name])
        return out

    def timing(self, x: np.ndarray, warmup: int = 2, repeat: int = 5) -> dict:
        """固定输入下的耗时（区分编译耗时与预热后推理耗时），报告中位数。"""
        for _ in range(warmup):
            self.infer(x)
        samples = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            self.infer(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        return {
            "compile_seconds": self.compile_seconds,
            "warmup_runs": warmup,
            "repeat": repeat,
            "inference_ms_median": samples[len(samples) // 2],
            "inference_ms_min": samples[0],
            "inference_ms_max": samples[-1],
            "execution_devices": self.execution_devices,
        }

    def describe(self) -> dict:
        return {
            "backend": "openvino-npu",
            "device": self.device,
            "device_name": self.device_name,
            "onnx": str(self.onnx_path),
            "onnx_bytes": self.onnx_path.stat().st_size if self.onnx_path.exists() else None,
            "execution_devices": self.execution_devices,
            "compile_seconds": self.compile_seconds,
            "property_evidence": self.property_evidence,
        }


def write_sidecar(encoder: NPUEncoder, extra: dict | None = None) -> Path:
    """把执行证据写一份 JSON 存档，方便人工复核（不改动指标输出）。"""
    path = MODEL_ROOT / "onnx" / "last_npu_run.json"
    payload = encoder.describe()
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path
