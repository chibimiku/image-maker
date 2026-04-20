import os
import importlib
import json
import re
import threading
import time
from contextlib import nullcontext
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from utils.upscaler import ExtrasUpscalePipeline, UpscalerHandle
from utils.upscaler_arch import (
    ONNX_MODEL_EXTS,
    SUPPORTED_MODEL_EXTS,
    get_upscaler_model_dirs,
    list_upscaler_models,
    normalize_upscaler_arch,
)
from utils.upscaler_arch_match import is_arch_compatible


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESRGAN_MODEL_DIR = os.path.join(BASE_DIR, "data", "models", "ESRGAN")

DEFAULT_UPSCALE_OPTIONS = {
    "enabled": False,
    "model_name": "",
    "model_arch": "esrgan",
    "upscale_mode": 0,
    "upscale_by": 2.0,
    "max_side_length": 0,
    "upscale_to_width": 1024,
    "upscale_to_height": 1024,
    "upscale_crop": False,
    "upscaler_2_name": "",
    "upscaler_2_visibility": 0.0,
    "cache_size": 4,
    "webp_target_mb": 10.0,
    "inference_device": "cpu",
    "downsample_method": "lanczos",
    "process_dump_enabled": True,
}

DOWNSAMPLE_METHODS = {
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
    "bicubic": Image.Resampling.BICUBIC,
    "mitchell_wand": None,
}


def normalize_model_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return ""
    return text


def _model_dirs(model_dir: str | None = None) -> list[str]:
    return get_upscaler_model_dirs("esrgan", model_dir=model_dir)


def list_esrgan_models(model_dir: str | None = None) -> list[str]:
    return list_upscaler_models("esrgan", model_dir=model_dir)


def list_upscaler_models_by_arch(
    model_arch: str,
    model_dir: str | None = None,
    allowed_exts: tuple[str, ...] | None = None,
) -> list[str]:
    return list_upscaler_models(model_arch, model_dir=model_dir, allowed_exts=allowed_exts)


def normalize_upscale_options(raw_options: dict | None) -> dict:
    options = dict(DEFAULT_UPSCALE_OPTIONS)
    if isinstance(raw_options, dict):
        options.update(raw_options)
    options["enabled"] = bool(options.get("enabled", False))
    options["model_name"] = normalize_model_name(options.get("model_name", ""))
    options["model_arch"] = normalize_upscaler_arch(options.get("model_arch", "esrgan"))
    options["upscale_mode"] = 1 if int(options.get("upscale_mode", 0)) == 1 else 0
    options["upscale_by"] = max(1.0, float(options.get("upscale_by", 2.0)))
    options["max_side_length"] = max(0, int(options.get("max_side_length", 0)))
    options["upscale_to_width"] = max(1, int(options.get("upscale_to_width", 1024)))
    options["upscale_to_height"] = max(1, int(options.get("upscale_to_height", 1024)))
    options["upscale_crop"] = bool(options.get("upscale_crop", False))
    options["upscaler_2_name"] = normalize_model_name(options.get("upscaler_2_name", ""))
    visibility = float(options.get("upscaler_2_visibility", 0.0))
    options["upscaler_2_visibility"] = min(1.0, max(0.0, visibility))
    options["cache_size"] = max(1, int(options.get("cache_size", 4)))
    options["webp_target_mb"] = max(0.1, float(options.get("webp_target_mb", 10.0)))
    device = str(options.get("inference_device", "cpu") or "cpu").strip().lower()
    options["inference_device"] = device if device in {"cpu", "auto", "npu"} else "cpu"
    method = str(options.get("downsample_method", "lanczos") or "lanczos").strip().lower()
    options["downsample_method"] = method if method in DOWNSAMPLE_METHODS else "lanczos"
    options["process_dump_enabled"] = bool(options.get("process_dump_enabled", True))
    return options


def _allowed_model_exts_for_device(inference_device: str) -> tuple[str, ...]:
    return ONNX_MODEL_EXTS if str(inference_device).strip().lower() == "npu" else SUPPORTED_MODEL_EXTS


def _resize_with_downsample_method(
    image: Image.Image,
    target_w: int,
    target_h: int,
    method: str,
    step_logger: Callable[[str], None] | None = None,
) -> Image.Image:
    resolved_method = str(method or "lanczos").strip().lower()
    if resolved_method == "mitchell_wand":
        try:
            wand_image_module = importlib.import_module("wand.image")
            np = importlib.import_module("numpy")
            wand_cls = getattr(wand_image_module, "Image", None)
            if wand_cls is None:
                raise RuntimeError("wand.image.Image 不可用")
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with wand_cls.from_array(arr) as wi:
                wi.resize(target_w, target_h, filter="mitchell")
                blob = wi.make_blob(format="png")
            with Image.open(BytesIO(blob)) as out_img:
                return out_img.convert("RGB")
        except Exception as e:
            if callable(step_logger):
                step_logger(f"Mitchell(Wand) 下采样不可用，已回退 Lanczos: {e}")
            return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    pil_method = DOWNSAMPLE_METHODS.get(resolved_method, Image.Resampling.LANCZOS)
    return image.resize((target_w, target_h), pil_method)


def _next_available_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    idx = 1
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _next_available_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    idx = 1
    while True:
        candidate = f"{base_dir}_{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _build_process_dump_dir(image_path: str) -> str:
    root, _ = os.path.splitext(image_path)
    return _next_available_dir(f"{root}-process")


def _normalize_stage_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "stage"


def _build_fixed_png_path(image_path: str) -> str:
    root, _ = os.path.splitext(image_path)
    return _next_available_path(f"{root}-fixed.png")


def _build_webp_path(fixed_png_path: str) -> str:
    root, _ = os.path.splitext(fixed_png_path)
    return _next_available_path(f"{root}.webp")


def _is_jpg(path: str) -> bool:
    lower = str(path or "").lower()
    return lower.endswith(".jpg") or lower.endswith(".jpeg")


def _extract_scale_from_name(name: str, fallback: float = 4.0) -> float:
    text = str(name or "").lower()
    patterns = [
        r"(\d+(?:\.\d+)?)x",
        r"x(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return max(1.0, float(m.group(1)))
            except Exception:
                pass
    return max(1.0, float(fallback))


def _run_with_heartbeat(
    *,
    task_label: str,
    heartbeat_interval_sec: float,
    step_logger: Callable[[str], None] | None,
    fn: Callable[[], Any],
) -> Any:
    if not callable(step_logger):
        return fn()

    done = threading.Event()
    started_at = time.perf_counter()

    def _heartbeat():
        tick = 0
        while not done.wait(max(0.5, float(heartbeat_interval_sec))):
            tick += 1
            elapsed = time.perf_counter() - started_at
            step_logger(
                f"{task_label}进行中: 心跳={tick}, 已耗时={elapsed:.2f}s "
                "(CPU 推理阶段可能长时间无细粒度内部进度)"
            )

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        done.set()
        thread.join(timeout=0.2)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _cleanup_cuda_memory(torch_module: Any) -> None:
    cuda_obj = getattr(torch_module, "cuda", None)
    if cuda_obj is None:
        return
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
            try:
                torch_module.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _safe_int_round(value: float) -> int:
    return int(round(float(value)))


def _build_upscaler_fn_with_spandrel(
    model_name: str,
    model_path: str,
    inference_device: str = "cpu",
    model_arch: str = "esrgan",
) -> tuple[Callable[[Image.Image, float, Callable[[str], None] | None], Image.Image], float, str, str]:
    try:
        torch = importlib.import_module("torch")
    except Exception as e:
        raise RuntimeError(
            "未安装 torch，无法执行 DAT 推理。请先安装 torch（CPU 或 CUDA 版本）。"
        ) from e

    try:
        spandrel = importlib.import_module("spandrel")
    except Exception as e:
        raise RuntimeError(
            "未安装 spandrel，无法加载 upscaler 模型。请先安装 spandrel。"
        ) from e

    has_extra_arches = False
    try:
        importlib.import_module("spandrel_extra_arches")
        has_extra_arches = True
    except Exception:
        has_extra_arches = False

    loader_cls = getattr(spandrel, "ModelLoader", None)
    if loader_cls is None:
        raise RuntimeError("spandrel 版本不兼容：找不到 ModelLoader。")

    loader = loader_cls()
    descriptor = None
    last_error = None
    last_method = ""
    for method_name in ("load_from_file", "load_from_path", "load"):
        fn = getattr(loader, method_name, None)
        if not callable(fn):
            continue
        try:
            descriptor = fn(model_path)
            break
        except Exception as e:
            last_method = method_name
            last_error = e
    if descriptor is None:
        err_text = str(last_error).strip()
        err_detail = (
            err_text
            if err_text
            else f"{type(last_error).__name__}: {repr(last_error)}"
        )
        hint = ""
        normalized_arch = normalize_upscaler_arch(model_arch)
        if normalized_arch in {"srformer-light", "omnisr", "real-cugan"} and not has_extra_arches:
            hint = (
                " 检测到当前缺少 `spandrel_extra_arches`，"
                "SRFormer-Light/OmniSR/Real-CUGAN 需要该扩展来注册架构。"
                "请先安装：`pip install spandrel-extra-arches`。"
            )
        raise RuntimeError(
            f"spandrel 无法加载模型: {model_name}. "
            f"最后方法={last_method or 'unknown'}，错误={err_detail}.{hint}"
        )

    arch_obj = getattr(descriptor, "architecture", None) or getattr(descriptor, "arch", None)
    arch_name = str(arch_obj or "").strip() or descriptor.__class__.__name__

    descriptor_scale = getattr(descriptor, "scale", None)
    if descriptor_scale is None:
        descriptor_scale = getattr(descriptor, "upscale", None)
    if descriptor_scale is None:
        descriptor_scale = getattr(descriptor, "upscale_factor", None)
    try:
        native_scale = max(1.0, float(descriptor_scale))
    except Exception:
        native_scale = _extract_scale_from_name(model_name, fallback=4.0)

    model = getattr(descriptor, "model", None)
    if model is None:
        raise RuntimeError("spandrel descriptor 不包含可执行的 model 对象。")

    use_auto = str(inference_device or "cpu").lower() == "auto"
    device = "cuda" if use_auto and bool(getattr(torch, "cuda", None)) and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    def _run_model(
        x: Any,
        *,
        amp_enabled: bool,
    ) -> Any:
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (device == "cuda" and amp_enabled)
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                y = model(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
        return y

    def _to_pil(y: Any) -> Image.Image:
        y = y.clamp(0.0, 1.0)
        y = (y * 255.0).round().to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(y, mode="RGB")

    def _upscale_tiled(
        x: Any,
        tile_size: int,
        tile_pad: int = 32,
        amp_enabled: bool = True,
    ) -> Image.Image:
        _, _, h, w = x.shape
        stride = max(16, int(tile_size) - int(tile_pad) * 2)
        out_u8 = None
        scale_h = None
        scale_w = None

        ys = list(range(0, h, stride))
        xs = list(range(0, w, stride))
        for y_start in ys:
            for x_start in xs:
                y0 = max(0, y_start - tile_pad)
                y1 = min(h, y_start + tile_size + tile_pad)
                x0 = max(0, x_start - tile_pad)
                x1 = min(w, x_start + tile_size + tile_pad)

                tile = x[:, :, y0:y1, x0:x1].to(device)
                tile_out = _run_model(tile, amp_enabled=amp_enabled)
                tile_out = tile_out[0]
                cur_scale_h = tile_out.shape[1] / max(1, (y1 - y0))
                cur_scale_w = tile_out.shape[2] / max(1, (x1 - x0))
                if out_u8 is None:
                    scale_h = cur_scale_h
                    scale_w = cur_scale_w
                    out_h = _safe_int_round(h * scale_h)
                    out_w = _safe_int_round(w * scale_w)
                    out_u8 = torch.empty((3, out_h, out_w), dtype=torch.uint8, device="cpu")

                valid_y0 = y_start
                valid_y1 = min(y_start + tile_size, h)
                valid_x0 = x_start
                valid_x1 = min(x_start + tile_size, w)

                in_valid_y0 = valid_y0 - y0
                in_valid_y1 = valid_y1 - y0
                in_valid_x0 = valid_x0 - x0
                in_valid_x1 = valid_x1 - x0

                out_y0 = _safe_int_round(valid_y0 * scale_h)
                out_y1 = _safe_int_round(valid_y1 * scale_h)
                out_x0 = _safe_int_round(valid_x0 * scale_w)
                out_x1 = _safe_int_round(valid_x1 * scale_w)

                tile_out_y0 = _safe_int_round(in_valid_y0 * scale_h)
                tile_out_y1 = _safe_int_round(in_valid_y1 * scale_h)
                tile_out_x0 = _safe_int_round(in_valid_x0 * scale_w)
                tile_out_x1 = _safe_int_round(in_valid_x1 * scale_w)

                tile_u8 = (
                    tile_out.clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                )
                out_u8[:, out_y0:out_y1, out_x0:out_x1] = tile_u8[
                    :,
                    tile_out_y0:tile_out_y1,
                    tile_out_x0:tile_out_x1,
                ]
                del tile_u8
                del tile_out
                del tile
                if device == "cuda":
                    _cleanup_cuda_memory(torch)

        if out_u8 is None:
            raise RuntimeError("分块推理失败：未生成有效输出。")

        arr = out_u8.permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")

    def upscale(
        image: Image.Image,
        scale: float,
        step_logger: Callable[[str], None] | None = None,
    ) -> Image.Image:
        src = image.convert("RGB")
        np_img = importlib.import_module("numpy").asarray(src).astype("float32") / 255.0
        x = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(device)
        if device != "cuda":
            try:
                y = _run_model(x, amp_enabled=False)
                return _to_pil(y)
            except (MemoryError, RuntimeError) as cpu_err:
                if callable(step_logger):
                    step_logger(
                        f"CPU 单次推理内存不足({type(cpu_err).__name__})，切换到分块推理。"
                    )
                import gc; gc.collect()
                last_cpu_err = cpu_err
                for tile_size in (768, 640, 512, 384, 256):
                    try:
                        if callable(step_logger):
                            step_logger(f"CPU 分块重试: tile={tile_size}, overlap=32")
                        out = _upscale_tiled(
                            x=x,
                            tile_size=tile_size,
                            tile_pad=32,
                            amp_enabled=False,
                        )
                        if callable(step_logger):
                            step_logger(f"CPU 分块推理成功: tile={tile_size}")
                        return out
                    except (MemoryError, RuntimeError) as tile_e:
                        last_cpu_err = tile_e
                        if callable(step_logger):
                            step_logger(
                                f"CPU 分块推理仍失败: tile={tile_size}，继续尝试更小 tile。"
                            )
                        import gc; gc.collect()
                        continue
                raise RuntimeError(
                    "CPU 内存不足：已自动尝试分块推理（tile=768..256）仍失败。"
                    "请降低输入尺寸或使用更小的模型。"
                ) from last_cpu_err

        try:
            # 阶段 1: CUDA fp32 整图推理（精度最高，显存消耗最大）
            try:
                y = _run_model(x, amp_enabled=False)
                return _to_pil(y)
            except RuntimeError as e:
                if not _is_cuda_oom_error(e):
                    raise
                if callable(step_logger):
                    step_logger(
                        "CUDA fp32 整图推理 OOM，尝试 fp16 整图推理。"
                    )
                _cleanup_cuda_memory(torch)

            # 阶段 2: CUDA fp16 整图推理（省显存，但 DAT 等注意力模型可能精度不足）
            try:
                y = _run_model(x, amp_enabled=True)
                return _to_pil(y)
            except RuntimeError as e:
                if not _is_cuda_oom_error(e):
                    raise
                if callable(step_logger):
                    step_logger(
                        "CUDA fp16 整图推理仍 OOM，切换到分块推理。"
                    )
                _cleanup_cuda_memory(torch)
                x = x.cpu()  # 释放 GPU 上的完整输入，tiling 时按切片搬运

            # 阶段 3: 分块推理（先 fp32 再 fp16），逐级缩小 tile
            last_oom: RuntimeError | None = None
            for amp in (False, True):
                mode_label = "fp16" if amp else "fp32"
                for tile_size in (1024, 768, 640, 512, 384, 256):
                    try:
                        if callable(step_logger):
                            step_logger(
                                f"分块重试: tile={tile_size}, overlap=32, 模式={mode_label}"
                            )
                        out = _upscale_tiled(
                            x=x,
                            tile_size=tile_size,
                            tile_pad=32,
                            amp_enabled=amp,
                        )
                        if callable(step_logger):
                            step_logger(f"分块推理成功: tile={tile_size}, 模式={mode_label}")
                        return out
                    except RuntimeError as tile_e:
                        if _is_cuda_oom_error(tile_e):
                            last_oom = tile_e
                            if callable(step_logger):
                                step_logger(
                                    f"分块推理 OOM: tile={tile_size}, 模式={mode_label}，继续尝试。"
                                )
                            _cleanup_cuda_memory(torch)
                            continue
                        raise
                # fp32 全部 tile 失败后，尝试 fp16 更小 tile 前先清理
                _cleanup_cuda_memory(torch)
            raise RuntimeError(
                "CUDA 显存不足：已自动尝试整图(fp32/fp16)+分块推理(tile=1024..256, fp32/fp16)仍失败。"
                "请降低放大倍率/输入尺寸，或切换到 CPU。"
            ) from last_oom
        finally:
            try:
                del x
            except Exception:
                pass
            _cleanup_cuda_memory(torch)

    return upscale, native_scale, arch_name, device


def _build_upscaler_fn_with_onnx(
    model_name: str,
    model_path: str,
    model_arch: str = "esrgan",
) -> tuple[Callable[[Image.Image, float, Callable[[str], None] | None], Image.Image], float, str, str]:
    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as e:
        raise RuntimeError(
            "NPU 模式需要 onnxruntime。请先安装：`pip install onnxruntime`。"
        ) from e
    np = importlib.import_module("numpy")

    available = list(getattr(ort, "get_available_providers", lambda: [])() or [])
    preferred = [
        "QNNExecutionProvider",
        "CANNExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in preferred if p in available]
    if not providers:
        if available:
            providers = [available[0]]
        else:
            raise RuntimeError("onnxruntime 未检测到可用的 ExecutionProvider。")

    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = ", ".join(session.get_providers() or providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise RuntimeError("ONNX 模型缺少输入或输出定义。")

    input_tensor = inputs[0]
    input_name = input_tensor.name
    output_name = outputs[0].name
    input_shape = list(getattr(input_tensor, "shape", []) or [])

    def _read_fixed_hw() -> tuple[int | None, int | None]:
        if len(input_shape) != 4:
            return None, None
        h = input_shape[2]
        w = input_shape[3]
        h = int(h) if isinstance(h, int) and h > 0 else None
        w = int(w) if isinstance(w, int) and w > 0 else None
        return h, w

    model_h, model_w = _read_fixed_hw()
    native_scale = _extract_scale_from_name(model_name, fallback=2.0)
    arch_name = f"onnx-{normalize_upscaler_arch(model_arch)}"

    def _to_onnx_input(image: Image.Image, step_logger: Callable[[str], None] | None) -> Any:
        src = image.convert("RGB")
        if model_h and model_w and (src.height != model_h or src.width != model_w):
            if callable(step_logger):
                step_logger(
                    f"ONNX 输入尺寸固定为 {model_w}x{model_h}，"
                    f"当前输入 {src.width}x{src.height}，先做对齐缩放。"
                )
            src = src.resize((model_w, model_h), Image.Resampling.BICUBIC)
        arr = np.asarray(src).astype("float32") / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f"ONNX 输入预处理失败，期望 RGB HWC，实际 shape={arr.shape}")
        x = np.transpose(arr, (2, 0, 1))[None, ...]
        return x

    def _to_pil(y: Any) -> Image.Image:
        arr = np.asarray(y)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise RuntimeError(f"ONNX 输出维度不受支持: shape={arr.shape}")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        arr = np.clip(arr, 0.0, 1.0) if arr.dtype != np.uint8 else arr
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).round().astype("uint8")
        return Image.fromarray(arr, mode="RGB")

    def upscale(
        image: Image.Image,
        scale: float,
        step_logger: Callable[[str], None] | None = None,
    ) -> Image.Image:
        x = _to_onnx_input(image, step_logger)
        y = session.run([output_name], {input_name: x})[0]
        out = _to_pil(y)
        if callable(step_logger):
            step_logger(f"ONNX 推理完成: provider={active_provider}, 输出={out.width}x{out.height}")
        return out

    return upscale, native_scale, arch_name, active_provider


class LocalESRGANProvider:
    """本地 upscaler 推理 Provider（基于 spandrel + torch）。"""

    _CACHE_LOCK = threading.RLock()
    _GLOBAL_RUNTIME_CACHE: dict[
        str,
        tuple[
            Callable[[Image.Image, float, Callable[[str], None] | None], Image.Image],
            float,
            str,
            str,
        ],
    ] = {}

    def __init__(
        self,
        model_dir: str | None = None,
        model_arch: str = "esrgan",
        inference_device: str = "cpu",
        downsample_method: str = "lanczos",
        step_logger: Callable[[str], None] | None = None,
        process_stage_callback: Callable[[str, Image.Image, dict], None] | None = None,
    ):
        self.model_arch = normalize_upscaler_arch(model_arch)
        model_dirs = get_upscaler_model_dirs(self.model_arch, model_dir=model_dir)
        self.model_dir = model_dirs[0] if model_dirs else (model_dir or DEFAULT_ESRGAN_MODEL_DIR)
        dev = str(inference_device or "cpu").strip().lower()
        self.inference_device = dev if dev in {"cpu", "auto", "npu"} else "cpu"
        method = str(downsample_method or "lanczos").strip().lower()
        self.downsample_method = method if method in DOWNSAMPLE_METHODS else "lanczos"
        self.step_logger = step_logger
        self.process_stage_callback = process_stage_callback
        model_exts = _allowed_model_exts_for_device(self.inference_device)
        self._model_path_map: dict[str, str] = {}
        for target_dir in model_dirs:
            if not os.path.isdir(target_dir):
                continue
            for name in os.listdir(target_dir):
                if name.lower().endswith(model_exts):
                    self._model_path_map[name] = os.path.join(target_dir, name)
        self._runtime_cache = LocalESRGANProvider._GLOBAL_RUNTIME_CACHE

    def _resolve(self, name: str | None) -> UpscalerHandle | None:
        model_name = normalize_model_name(name)
        if not model_name:
            return None
        model_path = self._model_path_map.get(model_name)
        if not model_path:
            raise AssertionError(f"could not find upscaler model: {model_name}")

        with LocalESRGANProvider._CACHE_LOCK:
            cache_key = f"{self.inference_device}::{self.model_arch}::{model_name}"
            runtime = self._runtime_cache.get(cache_key)
            if runtime is None:
                if self.inference_device == "npu":
                    runtime = _build_upscaler_fn_with_onnx(
                        model_name=model_name,
                        model_path=model_path,
                        model_arch=self.model_arch,
                    )
                else:
                    runtime = _build_upscaler_fn_with_spandrel(
                        model_name=model_name,
                        model_path=model_path,
                        inference_device=self.inference_device,
                        model_arch=self.model_arch,
                    )
                self._runtime_cache[cache_key] = runtime
        upscale_fn, native_scale, arch_name, device = runtime
        if self.inference_device != "npu" and not is_arch_compatible(self.model_arch, arch_name):
            raise AssertionError(
                f"模型架构不匹配: 你当前选择的是 `{self.model_arch}`，"
                f"但模型 `{model_name}` 实际识别为 `{arch_name}`。"
                "请切换正确的模型类型，或把模型放到对应目录后再刷新。"
            )

        def upscale(image: Image.Image, scale: float) -> Image.Image:
            src = image.convert("RGB")
            target_w = max(1, int(round(src.width * float(scale))))
            target_h = max(1, int(round(src.height * float(scale))))
            input_pixels_mp = (src.width * src.height) / 1_000_000.0
            target_pixels_mp = (target_w * target_h) / 1_000_000.0
            if callable(self.step_logger):
                self.step_logger(
                    f"模型推理开始: model={model_name}, arch={arch_name}, device={device}, "
                    f"native_scale=x{native_scale:g}, 输入={src.width}x{src.height}, 目标={target_w}x{target_h}"
                )
                self.step_logger(
                    f"推理规模评估: 输入={input_pixels_mp:.2f}MP, 目标={target_pixels_mp:.2f}MP"
                )
                if str(device).lower() == "cpu" and target_pixels_mp >= 30.0:
                    self.step_logger(
                        "提示: 当前为 CPU 大图推理，耗时可能较长（数十秒到数分钟），期间会持续输出心跳日志。"
                    )
            t0 = time.perf_counter()
            out = _run_with_heartbeat(
                task_label=f"模型推理({model_name})",
                heartbeat_interval_sec=10.0,
                step_logger=self.step_logger,
                fn=lambda: upscale_fn(src, scale, self.step_logger),
            )
            if callable(self.process_stage_callback):
                self.process_stage_callback(
                    "provider_raw",
                    out,
                    {"model_name": model_name, "arch": arch_name, "device": device},
                )
            infer_cost = time.perf_counter() - t0
            if callable(self.step_logger):
                self.step_logger(
                    f"模型推理完成: 输出={out.width}x{out.height}, 耗时={infer_cost:.2f}s"
                )
            if out.size != (target_w, target_h):
                t1 = time.perf_counter()
                out = _resize_with_downsample_method(
                    image=out,
                    target_w=target_w,
                    target_h=target_h,
                    method=self.downsample_method,
                    step_logger=self.step_logger,
                )
                resize_cost = time.perf_counter() - t1
                if callable(self.step_logger):
                    self.step_logger(
                        f"下采样对齐完成: method={self.downsample_method}, 输出={out.width}x{out.height}, "
                        f"耗时={resize_cost:.2f}s"
                    )
                if callable(self.process_stage_callback):
                    self.process_stage_callback(
                        "provider_align",
                        out,
                        {
                            "model_name": model_name,
                            "method": self.downsample_method,
                            "target_size": f"{target_w}x{target_h}",
                        },
                    )
            return out

        return UpscalerHandle(name=f"{model_name} [{arch_name} x{native_scale:g} on {device}]", upscale=upscale)

    def resolve_primary(self, name: str | None) -> UpscalerHandle | None:
        return self._resolve(name)

    def resolve_secondary(self, name: str | None) -> UpscalerHandle | None:
        return self._resolve(name)

    @classmethod
    def loaded_models_status(cls) -> dict[str, dict[str, Any]]:
        with cls._CACHE_LOCK:
            result: dict[str, dict[str, Any]] = {}
            for key, runtime in cls._GLOBAL_RUNTIME_CACHE.items():
                if "::" in key:
                    parts = key.split("::", 2)
                    if len(parts) == 3:
                        device_mode, selected_arch, model_name = parts
                    else:
                        device_mode = parts[0]
                        selected_arch = "esrgan"
                        model_name = parts[1]
                else:
                    device_mode, selected_arch, model_name = "cpu", "esrgan", key
                result[key] = {
                    "model_name": model_name,
                    "mode": device_mode,
                    "selected_arch": selected_arch,
                    "native_scale": runtime[1],
                    "arch": runtime[2],
                    "device": runtime[3],
                }
            return result

    @classmethod
    def release_loaded_models(cls) -> dict[str, Any]:
        with cls._CACHE_LOCK:
            released = len(cls._GLOBAL_RUNTIME_CACHE)
            cls._GLOBAL_RUNTIME_CACHE.clear()
        cuda_cache_cleared = False
        cuda_error = ""
        try:
            torch = importlib.import_module("torch")
            cuda_obj = getattr(torch, "cuda", None)
            if cuda_obj is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_cache_cleared = True
        except Exception as e:
            cuda_error = str(e)
        return {
            "released_models": released,
            "cuda_cache_cleared": cuda_cache_cleared,
            "cuda_error": cuda_error,
        }


def get_loaded_upscale_models_status() -> dict[str, dict[str, Any]]:
    return LocalESRGANProvider.loaded_models_status()


def release_loaded_upscale_models() -> dict[str, Any]:
    return LocalESRGANProvider.release_loaded_models()


@dataclass
class UpscaleResult:
    source_path: str
    skipped: bool
    fixed_png_path: str = ""
    fixed_png_size_bytes: int = 0
    webp_path: str = ""
    webp_size_bytes: int = 0
    process_dir: str = ""
    trace_json_path: str = ""
    error: str = ""


def upscale_image_to_fixed_png(image_path: str, options: dict) -> UpscaleResult:
    from webp_compressor import compress_image_to_webp_best_quality

    try:
        opts = normalize_upscale_options(options)
        opts["model_name"] = normalize_model_name(opts.get("model_name", ""))
        opts["upscaler_2_name"] = normalize_model_name(opts.get("upscaler_2_name", ""))
        if not _is_jpg(image_path):
            return UpscaleResult(source_path=image_path, skipped=True)
        if not opts.get("model_name"):
            return UpscaleResult(source_path=image_path, skipped=False, error="未选择放大模型")

        def _noop(_text: str):
            return None

        step_logger = opts.get("_step_logger")
        if not callable(step_logger):
            step_logger = _noop
        process_dump_enabled = bool(opts.get("process_dump_enabled", True))
        process_dir = ""
        trace_json_path = ""
        trace_rows: list[dict[str, Any]] = []
        stage_index = 0

        def save_stage_image(stage_name: str, image: Image.Image, meta: dict | None = None):
            nonlocal stage_index, process_dir
            if not process_dump_enabled:
                return
            if not process_dir:
                process_dir = _build_process_dump_dir(image_path)
                os.makedirs(process_dir, exist_ok=True)
                step_logger(f"过程文件目录: {process_dir}")
            stage_index += 1
            normalized = _normalize_stage_name(stage_name)
            filename = f"{stage_index:02d}-{normalized}.png"
            out_path = os.path.join(process_dir, filename)
            image.convert("RGB").save(out_path, format="PNG")
            row = {
                "index": stage_index,
                "stage": stage_name,
                "file": out_path,
                "width": int(image.width),
                "height": int(image.height),
            }
            if isinstance(meta, dict) and meta:
                row["meta"] = dict(meta)
            trace_rows.append(row)

        total_steps = 8
        step_index = 0

        def emit_step(text: str):
            nonlocal step_index
            step_index += 1
            step_logger(f"[步骤 {step_index}/{total_steps}] {text}")

        t_total = time.perf_counter()
        emit_step("参数校验完成")
        provider = LocalESRGANProvider(
            model_arch=str(opts.get("model_arch", "esrgan")),
            inference_device=str(opts.get("inference_device", "cpu")),
            downsample_method=str(opts.get("downsample_method", "lanczos")),
            step_logger=step_logger,
            process_stage_callback=save_stage_image if process_dump_enabled else None,
        )
        pipeline = ExtrasUpscalePipeline(
            provider=provider,
            cache_size_getter=lambda: int(opts["cache_size"]),
        )
        emit_step(
            f"推理管线初始化完成: device_mode={opts.get('inference_device')}, "
            f"downsample={opts.get('downsample_method')}"
        )
        info: dict[str, Any] = {}
        emit_step("开始读取图片")
        with Image.open(image_path) as img:
            src = img.convert("RGB")
            emit_step(f"图片读取完成: 输入尺寸={src.width}x{src.height}")
            save_stage_image("input", src)
            t_up = time.perf_counter()
            out = pipeline.upscale_image(
                image=src,
                info=info,
                upscale_mode=int(opts["upscale_mode"]),
                upscale_by=float(opts["upscale_by"]),
                max_side_length=int(opts["max_side_length"]),
                upscale_to_width=int(opts["upscale_to_width"]),
                upscale_to_height=int(opts["upscale_to_height"]),
                upscale_crop=bool(opts["upscale_crop"]),
                upscaler_1_name=str(opts["model_name"]),
                upscaler_2_name=str(opts.get("upscaler_2_name") or None),
                upscaler_2_visibility=float(opts["upscaler_2_visibility"]),
                stage_callback=save_stage_image if process_dump_enabled else None,
            )
            emit_step(f"放大阶段完成: 输出尺寸={out.width}x{out.height}, 耗时={time.perf_counter() - t_up:.2f}s")
            save_stage_image("pipeline_output", out)
            fixed_png_path = _build_fixed_png_path(image_path)
            t_save = time.perf_counter()
            out.save(fixed_png_path, format="PNG")
            emit_step(f"保存 PNG 完成: {os.path.basename(fixed_png_path)}, 耗时={time.perf_counter() - t_save:.2f}s")
        fixed_size = os.path.getsize(fixed_png_path)

        result = UpscaleResult(
            source_path=image_path,
            skipped=False,
            fixed_png_path=fixed_png_path,
            fixed_png_size_bytes=fixed_size,
            process_dir=process_dir,
        )

        target_mb = float(opts["webp_target_mb"])
        target_bytes = int(target_mb * 1024 * 1024)
        emit_step(
            f"体积检查完成: PNG={fixed_size / 1024 / 1024:.2f}MB, 目标WebP={target_mb:.2f}MB"
        )
        if fixed_size > target_bytes:
            webp_path = _build_webp_path(fixed_png_path)
            t_webp = time.perf_counter()
            ok, final_path, final_size, _ = compress_image_to_webp_best_quality(
                input_path=fixed_png_path,
                target_mb=target_mb,
                output_path=webp_path,
            )
            if ok:
                result.webp_path = final_path
                result.webp_size_bytes = final_size
            emit_step(
                f"WebP 压缩完成: {'成功' if ok else '失败'}, "
                f"耗时={time.perf_counter() - t_webp:.2f}s"
            )
        else:
            emit_step("WebP 压缩跳过: PNG 已低于目标体积")
        if process_dump_enabled and process_dir:
            trace_json_path = os.path.join(process_dir, "trace.json")
            with open(trace_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_path": image_path,
                        "fixed_png_path": fixed_png_path,
                        "webp_path": result.webp_path,
                        "target_mode": int(opts.get("upscale_mode", 0)),
                        "target_scale": float(opts.get("upscale_by", 2.0)),
                        "target_size": {
                            "width": int(opts.get("upscale_to_width", 1024)),
                            "height": int(opts.get("upscale_to_height", 1024)),
                            "crop": bool(opts.get("upscale_crop", False)),
                        },
                        "model": {
                            "name": str(opts.get("model_name", "")),
                            "arch": str(opts.get("model_arch", "esrgan")),
                            "secondary_name": str(opts.get("upscaler_2_name", "")),
                            "secondary_visibility": float(opts.get("upscaler_2_visibility", 0.0)),
                        },
                        "steps": trace_rows,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            result.trace_json_path = trace_json_path
            step_logger(f"过程追踪文件已保存: {trace_json_path}")
        emit_step(f"整图处理完成: 总耗时={time.perf_counter() - t_total:.2f}s")
        return result
    except Exception as e:
        return UpscaleResult(source_path=image_path, skipped=False, error=str(e))


class JpgAutoUpscaleThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, image_paths: list[str], options: dict, task_name: str = "JPG 后处理"):
        super().__init__()
        self.image_paths = list(image_paths or [])
        self.options = normalize_upscale_options(options)
        self.task_name = task_name
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def run(self):
        results = []
        total = len(self.image_paths)
        self.log_signal.emit(
            f"{self.task_name} 配置: model={self.options.get('model_name')!r}, "
            f"model_arch={self.options.get('model_arch')}, "
            f"scale={self.options.get('upscale_by')}, device_mode={self.options.get('inference_device')}, "
            f"downsample={self.options.get('downsample_method')}"
        )
        for idx, image_path in enumerate(self.image_paths, start=1):
            if self._cancel_requested or self.isInterruptionRequested():
                self.log_signal.emit(f"{self.task_name}已取消。")
                break
            self.log_signal.emit(f"{self.task_name}: ({idx}/{total}) 处理 {os.path.basename(image_path)}")
            image_started_at = time.perf_counter()

            def _step_logger(step_text: str):
                elapsed = time.perf_counter() - image_started_at
                self.log_signal.emit(
                    f"{self.task_name}: ({idx}/{total}) {os.path.basename(image_path)} "
                    f"{step_text} | 已耗时={elapsed:.2f}s"
                )

            options = dict(self.options)
            options["_step_logger"] = _step_logger
            result = upscale_image_to_fixed_png(image_path=image_path, options=options)
            results.append(result)
            if result.skipped:
                self.log_signal.emit(f"跳过非 JPG 文件: {os.path.basename(image_path)}")
                self.progress_signal.emit(idx, total)
                continue
            if result.error:
                self.log_signal.emit(f"后处理失败: {os.path.basename(image_path)} -> {result.error}")
                self.progress_signal.emit(idx, total)
                continue
            self.log_signal.emit(f"已输出放大 PNG: {result.fixed_png_path}")
            if result.webp_path:
                self.log_signal.emit(f"已输出定体积 WebP: {result.webp_path}")
            if result.process_dir:
                self.log_signal.emit(f"已输出过程文件目录: {result.process_dir}")
            if result.trace_json_path:
                self.log_signal.emit(f"已输出过程追踪: {result.trace_json_path}")
            self.progress_signal.emit(idx, total)
        packed = [
            {
                "source_path": item.source_path,
                "skipped": item.skipped,
                "fixed_png_path": item.fixed_png_path,
                "fixed_png_size_bytes": item.fixed_png_size_bytes,
                "webp_path": item.webp_path,
                "webp_size_bytes": item.webp_size_bytes,
                "process_dir": item.process_dir,
                "trace_json_path": item.trace_json_path,
                "error": item.error,
            }
            for item in results
        ]
        self.finish_signal.emit(packed)
