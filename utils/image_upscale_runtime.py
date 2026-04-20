import os
import importlib
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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESRGAN_MODEL_DIR = os.path.join(BASE_DIR, "data", "models", "ESRGAN")
LEGACY_ESRGAN_MODEL_DIR = os.path.join(BASE_DIR, "models", "ESRGAN")
SUPPORTED_MODEL_EXTS = (".pt", ".pth", ".safetensors")

DEFAULT_UPSCALE_OPTIONS = {
    "enabled": False,
    "model_name": "",
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
    if model_dir:
        return [model_dir]
    return [DEFAULT_ESRGAN_MODEL_DIR, LEGACY_ESRGAN_MODEL_DIR]


def list_esrgan_models(model_dir: str | None = None) -> list[str]:
    merged: dict[str, str] = {}
    for target_dir in _model_dirs(model_dir):
        if not os.path.isdir(target_dir):
            continue
        for name in os.listdir(target_dir):
            lower = name.lower()
            if lower.endswith(SUPPORTED_MODEL_EXTS):
                merged[lower] = name
    return sorted(merged.values(), key=lambda x: x.lower())


def normalize_upscale_options(raw_options: dict | None) -> dict:
    options = dict(DEFAULT_UPSCALE_OPTIONS)
    if isinstance(raw_options, dict):
        options.update(raw_options)
    options["enabled"] = bool(options.get("enabled", False))
    options["model_name"] = normalize_model_name(options.get("model_name", ""))
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
    options["inference_device"] = device if device in {"cpu", "auto"} else "cpu"
    method = str(options.get("downsample_method", "lanczos") or "lanczos").strip().lower()
    options["downsample_method"] = method if method in DOWNSAMPLE_METHODS else "lanczos"
    return options


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


def _detect_model_type(model_name: str, model_path: str) -> tuple[str, str]:
    filename = os.path.basename(model_name or model_path or "")
    lower_name = filename.lower()
    if "dat" in lower_name:
        return "dat", f"文件名命中 DAT 关键词: {filename}"

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".safetensors":
        try:
            safetensors = importlib.import_module("safetensors")
            safe_open = getattr(safetensors, "safe_open", None)
            if callable(safe_open):
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    meta = f.metadata() or {}
                    text = " ".join([str(k) + " " + str(v) for k, v in meta.items()]).lower()
                    if "dat" in text:
                        return "dat", "safetensors metadata 命中 DAT 关键词"
                    if text.strip():
                        return "non-dat", "safetensors metadata 未发现 DAT 关键词"
        except Exception:
            pass

    return "non-dat", "文件名与可读元数据均未识别到 DAT 特征"


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
    model_name: str, model_path: str, inference_device: str = "cpu"
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
            "未安装 spandrel，无法加载 DAT/ESRGAN 模型。请先安装 spandrel。"
        ) from e

    loader_cls = getattr(spandrel, "ModelLoader", None)
    if loader_cls is None:
        raise RuntimeError("spandrel 版本不兼容：找不到 ModelLoader。")

    loader = loader_cls()
    descriptor = None
    last_error = None
    for method_name in ("load_from_file", "load_from_path", "load"):
        fn = getattr(loader, method_name, None)
        if not callable(fn):
            continue
        try:
            descriptor = fn(model_path)
            break
        except Exception as e:
            last_error = e
    if descriptor is None:
        raise RuntimeError(
            f"spandrel 无法加载模型: {model_name}. "
            f"最后错误: {last_error}"
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

                tile = x[:, :, y0:y1, x0:x1]
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
            y = _run_model(x, amp_enabled=False)
            return _to_pil(y)

        try:
            try:
                y = _run_model(x, amp_enabled=True)
                return _to_pil(y)
            except RuntimeError as e:
                if not _is_cuda_oom_error(e):
                    raise
                if callable(step_logger):
                    step_logger(
                        "检测到 CUDA OOM，开始自动显存清理并切换分块推理重试。"
                    )
                _cleanup_cuda_memory(torch)
                last_oom = e
                for tile_size in (1024, 768, 640, 512, 384, 256):
                    try:
                        if callable(step_logger):
                            step_logger(
                                f"分块重试: tile={tile_size}, overlap=32, 模式=fp16"
                            )
                        out = _upscale_tiled(
                            x=x,
                            tile_size=tile_size,
                            tile_pad=32,
                            amp_enabled=True,
                        )
                        if callable(step_logger):
                            step_logger(f"分块推理成功: tile={tile_size}")
                        return out
                    except RuntimeError as tile_e:
                        if _is_cuda_oom_error(tile_e):
                            last_oom = tile_e
                            if callable(step_logger):
                                step_logger(
                                    f"分块推理仍 OOM: tile={tile_size}，继续尝试更小 tile。"
                                )
                            _cleanup_cuda_memory(torch)
                            continue
                        raise
                raise RuntimeError(
                    "CUDA 显存不足：已自动尝试分块推理（tile=1024..256）仍失败。"
                    "请降低放大倍率/输入尺寸，或切换到 CPU。"
                ) from last_oom
        finally:
            try:
                del x
            except Exception:
                pass
            _cleanup_cuda_memory(torch)

    return upscale, native_scale, arch_name, device


class LocalESRGANProvider:
    """
    本地 DAT 推理 Provider（基于 spandrel + torch）。
    仅允许 DAT 类型模型；非 DAT 会给出明确提示。
    """

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
        inference_device: str = "cpu",
        downsample_method: str = "lanczos",
        step_logger: Callable[[str], None] | None = None,
    ):
        self.model_dir = model_dir or DEFAULT_ESRGAN_MODEL_DIR
        dev = str(inference_device or "cpu").strip().lower()
        self.inference_device = dev if dev in {"cpu", "auto"} else "cpu"
        method = str(downsample_method or "lanczos").strip().lower()
        self.downsample_method = method if method in DOWNSAMPLE_METHODS else "lanczos"
        self.step_logger = step_logger
        self._model_path_map: dict[str, str] = {}
        for target_dir in _model_dirs(model_dir):
            if not os.path.isdir(target_dir):
                continue
            for name in os.listdir(target_dir):
                if name.lower().endswith(SUPPORTED_MODEL_EXTS):
                    self._model_path_map[name] = os.path.join(target_dir, name)
        self._runtime_cache = LocalESRGANProvider._GLOBAL_RUNTIME_CACHE

    def _resolve(self, name: str | None) -> UpscalerHandle | None:
        model_name = normalize_model_name(name)
        if not model_name:
            return None
        model_path = self._model_path_map.get(model_name)
        if not model_path:
            raise AssertionError(f"could not find upscaler model: {model_name}")

        model_type, reason = _detect_model_type(model_name, model_path)
        if model_type != "dat":
            raise AssertionError(
                "当前仅支持 DAT 模型推理。"
                f"模型 `{model_name}` 识别为非 DAT（原因: {reason}）。"
                "请更换 DAT 模型（如文件名包含 DAT）。"
            )

        with LocalESRGANProvider._CACHE_LOCK:
            cache_key = f"{self.inference_device}::{model_name}"
            runtime = self._runtime_cache.get(cache_key)
            if runtime is None:
                runtime = _build_upscaler_fn_with_spandrel(
                    model_name=model_name,
                    model_path=model_path,
                    inference_device=self.inference_device,
                )
                self._runtime_cache[cache_key] = runtime
        upscale_fn, native_scale, arch_name, device = runtime

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
                    device_mode, model_name = key.split("::", 1)
                else:
                    device_mode, model_name = "cpu", key
                result[key] = {
                    "model_name": model_name,
                    "mode": device_mode,
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
        total_steps = 8
        step_index = 0

        def emit_step(text: str):
            nonlocal step_index
            step_index += 1
            step_logger(f"[步骤 {step_index}/{total_steps}] {text}")

        t_total = time.perf_counter()
        emit_step("参数校验完成")
        provider = LocalESRGANProvider(
            inference_device=str(opts.get("inference_device", "cpu")),
            downsample_method=str(opts.get("downsample_method", "lanczos")),
            step_logger=step_logger,
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
            )
            emit_step(f"放大阶段完成: 输出尺寸={out.width}x{out.height}, 耗时={time.perf_counter() - t_up:.2f}s")
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
            self.progress_signal.emit(idx, total)
        packed = [
            {
                "source_path": item.source_path,
                "skipped": item.skipped,
                "fixed_png_path": item.fixed_png_path,
                "fixed_png_size_bytes": item.fixed_png_size_bytes,
                "webp_path": item.webp_path,
                "webp_size_bytes": item.webp_size_bytes,
                "error": item.error,
            }
            for item in results
        ]
        self.finish_signal.emit(packed)
