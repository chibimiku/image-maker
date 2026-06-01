from __future__ import annotations

from dataclasses import dataclass, field

from .models import CatalogItem


@dataclass
class SearchRequest:
    brand_slug: str
    max_pages: int = 1
    per_part_limit: int = 8
    preferred_parts: list[str] = field(default_factory=list)
    site_key: str = "lolibrary"


class FashionCatalogSiteAdapter:
    site_name = ""

    def build_search_url(self, brand_slug: str) -> str:
        raise NotImplementedError()

    def search_items(self, request: SearchRequest) -> list[CatalogItem]:
        raise NotImplementedError()
