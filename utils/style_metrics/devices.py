"""设备与后端选择：``auto`` / ``cuda`` / ``cpu`` / ``npu``。

纪律：

- **显式选择 NPU 时，不可用就报错并说明原因**，绝不静默回落到 CPU 再声称 NPU 成功。
- ``auto`` 只挑选**经运行期探针验证过**的后端，并报告最终选择与理由。
- 「可用设备列表里出现了 NPU」「ONNX 导出成功」「模型编译成功」都**不算** NPU 推理成功；
  只有真实跑完一次前向并拿到有限数值才算。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODEL_ROOT

PROBE_CACHE = MODEL_ROOT / "onnx" / "_npu_probe.json"


def _as_list(value) -> list:
    """OpenVINO 的某些属性既可能是 list 也可能是裸字符串（"NPU"）。

    直接 ``list("NPU")`` 会拆成 ``['N','P','U']`` —— 这是真实踩过的坑。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


# --------------------------------------------------------------------------- #
# 探针
# --------------------------------------------------------------------------- #


def cuda_available() -> dict[str, Any]:
    """CUDA 探针：不只看 is_available，还真的做一次矩阵乘与同步。"""
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            info["reason"] = "torch.cuda.is_available() == False"
            return info
        dev = torch.device("cuda:0")
        a = torch.randn(256, 256, device=dev)
        b = torch.randn(256, 256, device=dev)
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        info.update(
            available=True,
            device_name=torch.cuda.get_device_name(0),
            capability=f"{props.major}.{props.minor}",
            total_memory_bytes=int(props.total_memory),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            probe_value=float(c),
        )
    except Exception as exc:  # pragma: no cover - 依赖真实硬件
        info["reason"] = f"{type(exc).__name__}: {exc}"
    return info


def openvino_available() -> dict[str, Any]:
    """OpenVINO 运行时探针（列设备 ≠ 可用，仍需 compile+infer）。"""
    info: dict[str, Any] = {"available": False, "devices": []}
    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        info["available"] = True
        info["version"] = ov.__version__
        names = {}
        for d in devices:
            try:
                names[d] = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception as exc:  # pragma: no cover
                names[d] = f"<{type(exc).__name__}>"
        info["devices"] = devices
        info["device_names"] = names
        if "NPU" in devices:
            for key in ("FULL_DEVICE_NAME", "NPU_DRIVER_VERSION", "DEVICE_TYPE", "OPTIMIZATION_CAPABILITIES"):
                try:
                    info[f"npu_{key}"] = str(core.get_property("NPU", key))
                except Exception:
                    pass
    except Exception as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
    return info


def _build_probe_model():
    """内存里搭一个最小的 conv+relu，用于「真的跑一次 NPU」的探针。

    用 OpenVINO 原生 opset 构图，避免为探针落 ONNX 文件。
    """
    import openvino as ov

    opset = ov.opset13
    w = np.random.default_rng(7).standard_normal((8, 3, 3, 3)).astype(np.float32)
    x = opset.parameter([1, 3, 32, 32], np.float32, name="x")
    wc = opset.constant(w, np.float32, name="w")
    y = opset.relu(
        opset.convolution(
            x, wc, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1]
        )
    )
    return ov.Model([y], [x], "npu_probe")


def npu_probe(use_cache: bool = True, device: str = "NPU") -> dict[str, Any]:
    """在 NPU 上真实编译并跑一次前向，只有拿到有限输出才算通过。"""
    info: dict[str, Any] = {"available": False, "backend": "openvino", "device": device}
    if use_cache and PROBE_CACHE.exists():
        try:
            cached = json.loads(PROBE_CACHE.read_text(encoding="utf-8"))
            if cached.get("ov_version") == openvino_available().get("version"):
                return cached
        except Exception:
            pass
    try:
        import openvino as ov

        core = ov.Core()
        if device not in core.available_devices:
            info["reason"] = f"OpenVINO available_devices 里没有 {device}: {core.available_devices}"
            return info
        model = _build_probe_model()
        compiled = core.compile_model(model, device)
        try:
            info["execution_devices"] = _as_list(compiled.get_property("EXECUTION_DEVICES"))
        except Exception:
            info["execution_devices"] = []
        req = compiled.create_infer_request()
        out = list(req.infer({"x": np.zeros((1, 3, 32, 32), np.float32)}).values())[0]
        info["available"] = bool(np.isfinite(out).all())
        info["probe_output_shape"] = list(out.shape)
        info["probe_output_sum"] = float(out.sum())
        info["ov_version"] = ov.__version__
        info["device_name"] = str(core.get_property(device, "FULL_DEVICE_NAME"))
        if not info["available"]:
            info["reason"] = "NPU 前向输出含 NaN/Inf"
    except Exception as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return info
    try:
        PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PROBE_CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #


@dataclass
class Backend:
    """一次运行的实际后端决策。"""

    requested: str
    actual: str
    backend: str
    device_name: str = ""
    precision: str = "fp32"
    fallback: bool = False
    note: str = ""
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_npu(self) -> bool:
        return self.actual == "npu"


class DeviceUnavailableError(RuntimeError):
    """显式请求的设备不可用；调用方必须如实上报，不能静默换设备。"""


def resolve_device(
    requested: str,
    precision: str = "fp32",
    allow_npu: bool = True,
    probe_npu: bool = True,
) -> Backend:
    """把 CLI 的 ``--device`` 解析成真实后端。

    ``auto``：按 ``cuda -> npu -> cpu`` 顺序取第一个**运行期验证通过**的后端。
    """
    requested = (requested or "auto").strip().lower()
    if requested not in ("auto", "cuda", "cpu", "npu"):
        raise ValueError(f"未知设备: {requested!r}（可选 auto/cuda/cpu/npu）")

    if requested == "cpu":
        return Backend(
            requested="cpu",
            actual="cpu",
            backend="torch-cpu",
            device_name=_cpu_name(),
            precision="fp32",
            evidence={"note": "CPU 强制 fp32（不宣称 CPU fp16 已等价验证）"},
        )

    if requested == "cuda":
        info = cuda_available()
        if not info.get("available"):
            raise DeviceUnavailableError(f"请求 cuda 但不可用: {info.get('reason')}")
        return Backend(
            requested="cuda",
            actual="cuda",
            backend="torch-cuda",
            device_name=info.get("device_name", ""),
            precision=precision,
            evidence=info,
        )

    if requested == "npu":
        if not allow_npu:
            raise DeviceUnavailableError("调用方禁用了 NPU 路径")
        info = npu_probe() if probe_npu else {"available": False, "reason": "未探测"}
        if not info.get("available"):
            raise DeviceUnavailableError(
                "请求 npu 但不可用: " + str(info.get("reason") or info)
            )
        return Backend(
            requested="npu",
            actual="npu",
            backend="openvino-npu",
            device_name=info.get("device_name", ""),
            precision="fp16" if precision == "fp32" else precision,
            note="Intel NPU 只支持 FP16/INT8，特征按 fp16 计算（独立版本，不等价于 fp32）",
            evidence=info,
        )

    # auto
    cuda = cuda_available()
    if cuda.get("available"):
        return Backend(
            requested="auto",
            actual="cuda",
            backend="torch-cuda",
            device_name=cuda.get("device_name", ""),
            precision=precision,
            note="auto 选择 cuda（运行期探针通过）",
            evidence=cuda,
        )
    if allow_npu:
        npu = npu_probe() if probe_npu else {"available": False, "reason": "未探测"}
        if npu.get("available"):
            return Backend(
                requested="auto",
                actual="npu",
                backend="openvino-npu",
                device_name=npu.get("device_name", ""),
                precision="fp16",
                note="auto 选择 npu（cuda 不可用，NPU 运行期探针通过）",
                evidence=npu,
            )
    return Backend(
        requested="auto",
        actual="cpu",
        backend="torch-cpu",
        device_name=_cpu_name(),
        precision="fp32",
        note="auto 回落到 cpu（cuda/npu 均未通过运行期探针）",
        evidence={"cuda": cuda},
    )


def _cpu_name() -> str:
    try:
        import platform

        return f"{platform.machine()} / {platform.processor() or 'unknown'}"
    except Exception:  # pragma: no cover
        return os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def torch_device(backend: Backend):
    import torch

    if backend.actual == "cuda":
        return torch.device("cuda:0")
    if backend.actual == "cpu":
        return torch.device("cpu")
    return None  # npu 不走 torch 张量路径的骨干


def hardware_report() -> dict[str, Any]:
    """汇总 CPU / GPU / NPU 的真实型号、驱动与运行时（不做推测）。"""
    report: dict[str, Any] = {}
    try:
        import platform
        import torch

        report["cpu"] = {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }
    except Exception as exc:
        report["cpu"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["cuda"] = cuda_available()
    report["openvino"] = openvino_available()
    report["npu"] = npu_probe()

    # 厂商驱动 / 系统层面的证据（只读，不修改）
    for cmd, key in (
        ("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", "nvidia_smi"),
    ):
        try:
            import subprocess

            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
            report[key] = (out.stdout or out.stderr).strip()
        except Exception as exc:
            report[key] = f"<{type(exc).__name__}: {exc}>"
    return report
