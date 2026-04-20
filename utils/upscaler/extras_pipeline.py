from typing import Callable

from PIL import Image

from .cache import ImageCache
from .core import crop_to_target, make_cache_key, resolve_upscale_request
from .types import UpscaleRequest, UpscalerProvider


class ExtrasUpscalePipeline:
    def __init__(self, provider: UpscalerProvider, cache_size_getter: Callable[[], int]):
        self.provider = provider
        self.cache_size_getter = cache_size_getter
        self.cache = ImageCache()

    def compute_target_size(
        self,
        image: Image.Image,
        *,
        upscale_mode: int,
        upscale_by: float,
        max_side_length: int,
        upscale_to_width: int,
        upscale_to_height: int,
    ) -> tuple[int, int]:
        request = UpscaleRequest(
            mode=upscale_mode,
            by=upscale_by,
            max_side_length=max_side_length,
            to_width=upscale_to_width,
            to_height=upscale_to_height,
            crop=False,
        )
        resolved = resolve_upscale_request(image, request, info={})
        return int(image.width * resolved.by), int(image.height * resolved.by)

    def upscale_image(
        self,
        image: Image.Image,
        info: dict,
        *,
        upscale_mode: int,
        upscale_by: float,
        max_side_length: int,
        upscale_to_width: int,
        upscale_to_height: int,
        upscale_crop: bool,
        upscaler_1_name: str | None,
        upscaler_2_name: str | None,
        upscaler_2_visibility: float,
        stage_callback: Callable[[str, Image.Image, dict], None] | None = None,
    ) -> Image.Image:
        upscaler1 = self.provider.resolve_primary(upscaler_1_name)
        if upscaler1 is None:
            return image

        request = UpscaleRequest(
            mode=upscale_mode,
            by=upscale_by,
            max_side_length=max_side_length,
            to_width=upscale_to_width,
            to_height=upscale_to_height,
            crop=upscale_crop,
        )

        upscaled_image = self._upscale(
            image,
            info,
            upscaler1,
            request,
            stage_name="primary",
            stage_callback=stage_callback,
        )
        info["Postprocess upscaler"] = upscaler1.name

        upscaler2 = self.provider.resolve_secondary(upscaler_2_name)
        if upscaler2 is not None and upscaler_2_visibility > 0:
            second_upscale = self._upscale(
                image,
                info,
                upscaler2,
                request,
                stage_name="secondary",
                stage_callback=stage_callback,
            )
            if upscaled_image.mode != second_upscale.mode:
                second_upscale = second_upscale.convert(upscaled_image.mode)
            upscaled_image = Image.blend(upscaled_image, second_upscale, upscaler_2_visibility)
            info["Postprocess upscaler 2"] = upscaler2.name
            if callable(stage_callback):
                stage_callback(
                    "blend",
                    upscaled_image,
                    {"upscaler_2_visibility": float(upscaler_2_visibility)},
                )

        return upscaled_image

    def _upscale(
        self,
        image: Image.Image,
        info: dict,
        upscaler,
        request: UpscaleRequest,
        stage_name: str,
        stage_callback: Callable[[str, Image.Image, dict], None] | None = None,
    ) -> Image.Image:
        resolved = resolve_upscale_request(image, request, info)
        cache_key = make_cache_key(image, upscaler.name, resolved)
        cached_image = self.cache.get(cache_key)

        if cached_image is not None:
            result = cached_image
            if callable(stage_callback):
                stage_callback(
                    f"{stage_name}_result",
                    result,
                    {"source": "cache", "upscaler": upscaler.name},
                )
        else:
            result = upscaler.upscale(image, resolved.by)
            self.cache.put(cache_key, result, max_items=max(self.cache_size_getter(), 1))
            if callable(stage_callback):
                stage_callback(
                    f"{stage_name}_result",
                    result,
                    {"source": "inference", "upscaler": upscaler.name},
                )

        if resolved.mode == 1 and resolved.crop:
            result = crop_to_target(result, resolved.to_width, resolved.to_height)
            info["Postprocess crop to"] = f"{result.width}x{result.height}"
            if callable(stage_callback):
                stage_callback(
                    f"{stage_name}_crop",
                    result,
                    {"target_size": f"{resolved.to_width}x{resolved.to_height}"},
                )

        return result

    def clear_cache(self):
        self.cache.clear()
