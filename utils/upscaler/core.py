import numpy as np
from PIL import Image

from .types import ResolvedUpscaleRequest, UpscaleRequest


def limit_size_by_one_dimension(w: int, h: int, limit: int) -> tuple[int, int]:
    if h > w and h > limit:
        w = limit * w // h
        h = limit
    elif w > limit:
        h = limit * h // w
        w = limit

    return (int(w), int(h))


def resolve_upscale_request(image: Image.Image, request: UpscaleRequest, info: dict) -> ResolvedUpscaleRequest:
    upscale_mode = request.mode
    upscale_by = request.by
    upscale_to_width = request.to_width
    upscale_to_height = request.to_height
    upscale_crop = request.crop

    if upscale_mode == 1:
        upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
        info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
    else:
        info["Postprocess upscale by"] = upscale_by
        if request.max_side_length != 0 and max(*image.size) * upscale_by > request.max_side_length:
            upscale_mode = 1
            upscale_crop = False
            upscale_to_width, upscale_to_height = limit_size_by_one_dimension(image.width * upscale_by, image.height * upscale_by, request.max_side_length)
            upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
            info["Max side length"] = request.max_side_length

    return ResolvedUpscaleRequest(
        mode=upscale_mode,
        by=upscale_by,
        to_width=upscale_to_width,
        to_height=upscale_to_height,
        crop=upscale_crop,
    )


def make_cache_key(image: Image.Image, upscaler_name: str, request: ResolvedUpscaleRequest) -> tuple:
    return (
        hash(np.array(image.getdata()).tobytes()),
        upscaler_name,
        request.mode,
        request.by,
        request.to_width,
        request.to_height,
        request.crop,
    )


def crop_to_target(image: Image.Image, width: int, height: int) -> Image.Image:
    cropped = Image.new("RGB", (width, height))
    cropped.paste(image, box=(width // 2 - image.width // 2, height // 2 - image.height // 2))
    return cropped
