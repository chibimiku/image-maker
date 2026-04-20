from PIL import Image

from modules import shared

from .types import UpscalerHandle


class WebUIUpscalerProvider:
    def _resolve(self, name: str | None, *, allow_none: bool, exclude_none: bool) -> UpscalerHandle | None:
        normalized_name = None if name == "None" else name

        if normalized_name is None:
            return None if allow_none else self._not_found(name)

        upscaler = next(
            iter(
                [
                    x
                    for x in shared.sd_upscalers
                    if x.name == normalized_name and (not exclude_none or x.name != "None")
                ]
            ),
            None,
        )

        if upscaler is None:
            return self._not_found(name)

        def upscale(image: Image.Image, scale: float) -> Image.Image:
            return upscaler.scaler.upscale(image, scale, upscaler.data_path)

        return UpscalerHandle(name=upscaler.name, upscale=upscale)

    def resolve_primary(self, name: str | None) -> UpscalerHandle | None:
        return self._resolve(name, allow_none=True, exclude_none=False)

    def resolve_secondary(self, name: str | None) -> UpscalerHandle | None:
        return self._resolve(name, allow_none=True, exclude_none=True)

    @staticmethod
    def _not_found(name: str | None):
        raise AssertionError(f"could not find upscaler named {name}")
