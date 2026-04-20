from dataclasses import dataclass
from typing import Callable, Protocol

from PIL import Image


@dataclass(frozen=True)
class UpscaleRequest:
    mode: int
    by: float
    max_side_length: int
    to_width: int
    to_height: int
    crop: bool


@dataclass(frozen=True)
class ResolvedUpscaleRequest:
    mode: int
    by: float
    to_width: int
    to_height: int
    crop: bool


@dataclass(frozen=True)
class UpscalerHandle:
    name: str
    upscale: Callable[[Image.Image, float], Image.Image]


class UpscalerProvider(Protocol):
    def resolve_primary(self, name: str | None) -> UpscalerHandle | None:
        """Resolve a primary upscaler by name."""

    def resolve_secondary(self, name: str | None) -> UpscalerHandle | None:
        """Resolve a secondary upscaler by name."""
