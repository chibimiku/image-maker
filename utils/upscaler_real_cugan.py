from typing import Any


REAL_CUGAN_ARCH_KEY = "real-cugan"
REAL_CUGAN_ARCH_LABEL = "Real-CUGAN"
REAL_CUGAN_DIR_NAME = "Real-CUGAN"
REAL_CUGAN_ARCH_ALIASES = {
    "real-cugan",
    "real_cugan",
    "realcugan",
    "cugan",
    "realcugan-upcunet",
}


def normalize_real_cugan_arch(value: Any) -> str:
    text = str(value or "").strip().lower()
    return REAL_CUGAN_ARCH_KEY if text in REAL_CUGAN_ARCH_ALIASES else ""


def normalize_loaded_real_cugan_arch_name(arch_name: Any) -> str:
    text = str(arch_name or "").strip().lower()
    if "real-cugan" in text or "realcugan" in text:
        return REAL_CUGAN_ARCH_KEY
    if "upcunet" in text or "up_cunet" in text:
        return REAL_CUGAN_ARCH_KEY
    return ""
