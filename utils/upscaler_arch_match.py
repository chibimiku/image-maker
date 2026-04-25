from typing import Any

from utils.upscaler_arch import normalize_upscaler_arch
from utils.upscaler_real_cugan import normalize_loaded_real_cugan_arch_name


def normalize_loaded_arch_name(arch_name: Any) -> str:
    text = str(arch_name or "").strip().lower()
    real_cugan = normalize_loaded_real_cugan_arch_name(text)
    if real_cugan:
        return real_cugan
    if "omnisr" in text:
        return "omnisr"
    if "srformer" in text:
        return "srformer-light"
    if "esrgan" in text or "dat" in text or "rrdb" in text:
        return "esrgan"
    return ""


def is_arch_compatible(selected_arch: Any, loaded_arch_name: Any) -> bool:
    expected = normalize_upscaler_arch(selected_arch)
    loaded = normalize_loaded_arch_name(loaded_arch_name)
    return bool(loaded) and expected == loaded
