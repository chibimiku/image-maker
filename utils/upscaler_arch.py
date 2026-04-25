import os
from typing import Any

from utils.upscaler_real_cugan import (
    REAL_CUGAN_ARCH_KEY,
    REAL_CUGAN_ARCH_LABEL,
    REAL_CUGAN_DIR_NAME,
    normalize_real_cugan_arch,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED_MODEL_EXTS = (".pt", ".pth", ".safetensors")
ONNX_MODEL_EXTS = (".onnx",)

UPSCALER_ARCH_CHOICES = [
    ("ESRGAN(DAT)", "esrgan"),
    ("SRFormer-Light", "srformer-light"),
    ("OmniSR", "omnisr"),
    (REAL_CUGAN_ARCH_LABEL, REAL_CUGAN_ARCH_KEY),
]

_UPSCALER_ARCH_DIR_MAP = {
    "esrgan": "ESRGAN",
    "srformer-light": "SRFormer-Light",
    "omnisr": "OmniSR",
    REAL_CUGAN_ARCH_KEY: REAL_CUGAN_DIR_NAME,
}


def normalize_upscaler_arch(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"esrgan", "dat", "rrdb"}:
        return "esrgan"
    if text in {"srformer", "srformer-light", "srformer_light", "srformerlight"}:
        return "srformer-light"
    if text in {"omnisr", "omni_sr", "omni-sr"}:
        return "omnisr"
    real_cugan = normalize_real_cugan_arch(text)
    if real_cugan:
        return real_cugan
    return "esrgan"


def get_upscaler_arch_label(arch: Any) -> str:
    normalized = normalize_upscaler_arch(arch)
    for label, value in UPSCALER_ARCH_CHOICES:
        if value == normalized:
            return label
    return "ESRGAN(DAT)"


def _legacy_esrgan_model_dirs() -> list[str]:
    return [
        os.path.join(BASE_DIR, "data", "models", "ESRGAN"),
        os.path.join(BASE_DIR, "models", "ESRGAN"),
    ]


def get_upscaler_model_dirs(arch: Any, model_dir: str | None = None) -> list[str]:
    if model_dir:
        return [model_dir]
    normalized = normalize_upscaler_arch(arch)
    if normalized == "esrgan":
        return _legacy_esrgan_model_dirs()
    dir_name = _UPSCALER_ARCH_DIR_MAP.get(normalized, "ESRGAN")
    return [os.path.join(BASE_DIR, "models", "upscaler", dir_name)]


def get_upscaler_model_dir_hint(arch: Any) -> str:
    return " 或 ".join(get_upscaler_model_dirs(arch))


def list_upscaler_models(
    arch: Any,
    model_dir: str | None = None,
    allowed_exts: tuple[str, ...] | None = None,
) -> list[str]:
    exts = tuple(x.lower() for x in (allowed_exts or SUPPORTED_MODEL_EXTS))
    merged: dict[str, str] = {}
    for target_dir in get_upscaler_model_dirs(arch, model_dir=model_dir):
        if not os.path.isdir(target_dir):
            continue
        for name in os.listdir(target_dir):
            lower = name.lower()
            if lower.endswith(exts):
                merged[lower] = name
    return sorted(merged.values(), key=lambda x: x.lower())
