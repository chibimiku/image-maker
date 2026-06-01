from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import requests

from .lolibrary_adapter import LolibraryAdapter
from .wear_adapter import WearAdapter
from .models import (
    CollectionBundle,
    CollectedAsset,
    PART_DRESS,
    PART_SHOES,
    PART_SOCKS,
    SUPPORTED_PARTS,
)
from .site_base import SearchRequest


PART_CATEGORY_KEYWORDS = {
    PART_DRESS: {"jsk", "op", "jumperskirt", "one piece", "onepiece", "dress", "sets"},
    PART_SHOES: {"shoes", "shoe", "heels", "boots", "pumps"},
    PART_SOCKS: {"socks", "sock", "otks", "utks", "tights", "leggings"},
}

PART_PROMPT_HINTS = {
    PART_DRESS: "用于参考连衣裙/主服装的版型、花纹、材质与配色。",
    PART_SHOES: "用于参考鞋子的鞋型、鞋跟、装饰、材质与颜色。",
    PART_SOCKS: "用于参考袜子的长度、花边、图案、透明度与颜色。",
}


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name or "").strip(), flags=re.U)
    return name.strip("._") or "item"


def infer_part_from_item(category_slug: str, title: str, tags: list[str]) -> str:
    haystack = " ".join([category_slug or "", title or "", " ".join(tags or [])]).lower()
    for part, keywords in PART_CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return part
    return ""


def select_primary_image(image_urls: list[str], fallback_url: str = "") -> str:
    for url in image_urls or []:
        if "/images/" in str(url):
            return str(url)
    if image_urls:
        return str(image_urls[0])
    return str(fallback_url or "")


class FashionCollectionService:
    def __init__(self, adapter: LolibraryAdapter | None = None, timeout: int = 20):
        self.lolibrary_adapter = adapter or LolibraryAdapter(timeout=timeout)
        self.wear_adapter = WearAdapter(timeout=timeout)
        self.adapter = self.lolibrary_adapter
        self.session = self.adapter.session
        self.timeout = timeout

    def collect_bundle(
        self,
        site_key: str,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
    ) -> CollectionBundle:
        if str(site_key or "").strip().lower() == "wear":
            return self.collect_wear_bundle(
                brand_slug=brand_slug,
                output_dir=output_dir,
                max_pages=max_pages,
                preferred_parts=preferred_parts,
            )
        return self.collect_lolibrary_bundle(
            brand_slug=brand_slug,
            output_dir=output_dir,
            max_pages=max_pages,
            preferred_parts=preferred_parts,
        )

    def collect_lolibrary_bundle(
        self,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
    ) -> CollectionBundle:
        self.adapter = self.lolibrary_adapter
        self.session = self.adapter.session
        preferred_parts = [part for part in (preferred_parts or list(SUPPORTED_PARTS)) if part in SUPPORTED_PARTS]
        os.makedirs(output_dir, exist_ok=True)
        request = SearchRequest(brand_slug=brand_slug, max_pages=max_pages, preferred_parts=preferred_parts, site_key="lolibrary")
        search_url = self.adapter.build_search_url(brand_slug)
        search_items = self.adapter.search_items(request)

        selected_assets: list[CollectedAsset] = []
        taken_parts: set[str] = set()

        for item in search_items:
            detail = self.adapter.fetch_item_detail(item.item_url)
            part = infer_part_from_item(detail.category_slug, detail.title, detail.tags)
            if part not in preferred_parts or part in taken_parts:
                continue

            image_url = select_primary_image(detail.image_urls, detail.thumbnail_url or item.thumbnail_url)
            if not image_url:
                continue

            local_path = self.download_image(image_url, os.path.join(output_dir, part), f"{detail.item_id}_{part}")
            selected_assets.append(
                CollectedAsset(
                    part=part,
                    item=detail,
                    image_url=image_url,
                    local_path=local_path,
                    source_search_url=search_url,
                    prompt_hint=PART_PROMPT_HINTS.get(part, ""),
                )
            )
            taken_parts.add(part)
            if len(taken_parts) >= len(preferred_parts):
                break

        missing_parts = [part for part in preferred_parts if part not in taken_parts]
        return CollectionBundle(
            site_name=self.adapter.site_name,
            brand_slug=brand_slug,
            search_url=search_url,
            output_dir=output_dir,
            assets=selected_assets,
            missing_parts=missing_parts,
        )

    def collect_wear_bundle(
        self,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
    ) -> CollectionBundle:
        self.adapter = self.wear_adapter
        self.session = self.adapter.session
        preferred_parts = [part for part in (preferred_parts or list(SUPPORTED_PARTS)) if part in SUPPORTED_PARTS]
        os.makedirs(output_dir, exist_ok=True)

        selected_assets: list[CollectedAsset] = []
        taken_parts: set[str] = set()

        for part in preferred_parts:
            search_url = self.wear_adapter.get_part_search_url(part)
            coordinates = self.wear_adapter.search_coordinates(part=part, max_pages=max_pages)
            selected_item = None
            for coordinate in coordinates:
                items = self.wear_adapter.fetch_coordinate_items(coordinate["coordinate_url"])
                for item in items:
                    inferred = infer_part_from_item(item.category_slug, item.title, item.tags)
                    if inferred != part:
                        continue
                    if not self.wear_adapter.matches_brand(item, brand_slug):
                        continue
                    selected_item = (item, coordinate)
                    break
                if selected_item:
                    break

            if not selected_item:
                continue

            item, coordinate = selected_item
            image_url = select_primary_image(item.image_urls, item.thumbnail_url or coordinate.get("thumbnail_url", ""))
            if not image_url:
                continue
            local_path = self.download_image(image_url, os.path.join(output_dir, part), f"{item.item_id}_{part}")
            selected_assets.append(
                CollectedAsset(
                    part=part,
                    item=item,
                    image_url=image_url,
                    local_path=local_path,
                    source_search_url=search_url,
                    prompt_hint=PART_PROMPT_HINTS.get(part, ""),
                )
            )
            taken_parts.add(part)

        missing_parts = [part for part in preferred_parts if part not in taken_parts]
        return CollectionBundle(
            site_name=self.wear_adapter.site_name,
            brand_slug=brand_slug,
            search_url=self.wear_adapter.build_search_url(brand_slug),
            output_dir=output_dir,
            assets=selected_assets,
            missing_parts=missing_parts,
        )

    def download_image(self, image_url: str, target_dir: str, filename_stem: str) -> str:
        os.makedirs(target_dir, exist_ok=True)
        response = self.session.get(image_url, timeout=self.timeout)
        response.raise_for_status()

        ext = self._guess_extension(image_url, response.headers.get("content-type", ""))
        local_path = os.path.join(target_dir, f"{sanitize_filename(filename_stem)}{ext}")
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path

    @staticmethod
    def _guess_extension(image_url: str, content_type: str) -> str:
        if "png" in (content_type or "").lower():
            return ".png"
        if "webp" in (content_type or "").lower():
            return ".webp"
        path = urlparse(image_url).path.lower()
        _, ext = os.path.splitext(path)
        if ext in {".jpg", ".jpeg", ".png", ".webp"}:
            return ext
        return ".jpg"
