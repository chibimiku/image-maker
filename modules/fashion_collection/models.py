from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PART_DRESS = "dress"
PART_SHOES = "shoes"
PART_SOCKS = "socks"

SUPPORTED_PARTS = (PART_DRESS, PART_SHOES, PART_SOCKS)


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass
class CatalogItem:
    source_site: str
    item_id: str
    item_url: str
    title: str
    category_slug: str = ""
    category_label: str = ""
    brand: str = ""
    item_number: str = ""
    thumbnail_url: str = ""
    image_urls: list[str] = field(default_factory=list)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def normalized_title(self) -> str:
        return _clean_text(self.title)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollectedAsset:
    part: str
    item: CatalogItem
    image_url: str
    local_path: str
    source_search_url: str
    prompt_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "image_url": self.image_url,
            "local_path": self.local_path,
            "source_search_url": self.source_search_url,
            "prompt_hint": self.prompt_hint,
            "item": self.item.to_dict(),
        }


@dataclass
class CollectionBundle:
    site_name: str
    brand_slug: str
    search_url: str
    output_dir: str
    assets: list[CollectedAsset] = field(default_factory=list)
    missing_parts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_name": self.site_name,
            "brand_slug": self.brand_slug,
            "search_url": self.search_url,
            "output_dir": self.output_dir,
            "missing_parts": list(self.missing_parts),
            "assets": [asset.to_dict() for asset in self.assets],
        }
