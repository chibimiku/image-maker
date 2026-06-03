from __future__ import annotations

import json
import re
from urllib.parse import urljoin

import requests

from .models import CatalogItem, PART_BAG, PART_DRESS, PART_HAIR_ACCESSORY, PART_SHOES, PART_SOCKS
from .networking import request_with_proxy_fallback


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

PART_SEARCH_URLS = {
    PART_DRESS: "https://wear.jp/women-category/onepiece/dress/",
    PART_SHOES: "https://wear.jp/women-category/shoes/sandal/",
    PART_SOCKS: "https://wear.jp/women-category/leg-wear/socks/",
    PART_HAIR_ACCESSORY: "https://wear.jp/women-category/hair-accessory/",
    PART_BAG: "https://wear.jp/women-category/bag/handbag/",
}


def _extract_next_data(html_text: str) -> dict:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html_text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except Exception:
        return {}


def _find_content_tiles(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("content_tiles"), list):
            return payload["content_tiles"]
        for value in payload.values():
            found = _find_content_tiles(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_content_tiles(value)
            if found:
                return found
    return []


def _normalize_brand_key(brand_data: dict | None) -> str:
    brand_data = brand_data or {}
    return str(brand_data.get("key") or brand_data.get("name") or "").strip().lower()


class WearAdapter:
    site_name = "WEAR"

    def __init__(self, session: requests.Session | None = None, timeout: int = 20):
        self.session = session or requests.Session()
        self.timeout = max(5, int(timeout))
        self.session.headers.update(DEFAULT_HEADERS)

    def build_search_url(self, brand_slug: str = "") -> str:
        return PART_SEARCH_URLS[PART_DRESS]

    def get_part_search_url(self, part: str) -> str:
        return PART_SEARCH_URLS.get(part, "https://wear.jp/women-coordinate/")

    def _net(self):
        """Get proxy and log callbacks for internal use."""
        proxy = getattr(self, 'proxy_url', None)
        log = getattr(self, 'log_callback', None)
        return proxy, log

    def search_coordinates(self, part: str, max_pages: int = 1) -> list[dict]:
        results: list[dict] = []
        seen_urls: set[str] = set()
        base_url = self.get_part_search_url(part)
        max_page = max(1, max_pages)
        proxy, log = self._net()
        for page in range(1, max_page + 1):
            if log:
                log(f"[WEAR] 抓取第 {page}/{max_page} 页: {part}")
            page_url = base_url if page == 1 else f"{base_url}?pageno={page}"
            response = request_with_proxy_fallback(self.session, "GET", page_url, timeout=self.timeout, proxy_url=proxy, log_callback=log)
            response.raise_for_status()
            page_items = self.parse_search_html(response.text, part=part, base_url="https://wear.jp", seen_urls=seen_urls)
            if log:
                log(f"[WEAR] 第 {page} 页解析到 {len(page_items)} 个穿搭 (累计 {len(results) + len(page_items)})")
            results.extend(page_items)
        return results

    def fetch_coordinate_items(self, coordinate_url: str) -> list[CatalogItem]:
        proxy, log = self._net()
        response = request_with_proxy_fallback(self.session, "GET", coordinate_url, timeout=self.timeout, proxy_url=proxy, log_callback=log)
        response.raise_for_status()
        return self.parse_coordinate_html(response.text, coordinate_url=coordinate_url)

    @classmethod
    def parse_search_html(cls, html_text: str, part: str, base_url: str = "https://wear.jp", seen_urls: set[str] | None = None) -> list[dict]:
        seen_urls = seen_urls or set()
        data = _extract_next_data(html_text)
        tiles = _find_content_tiles(data)
        results = []
        for tile in tiles:
            coordinate_tile = (tile or {}).get("coordinate_tile") or {}
            coordinate = coordinate_tile.get("coordinate") or {}
            url = str(coordinate.get("url") or "").strip()
            if not url:
                continue
            full_url = urljoin(base_url, url)
            if full_url in seen_urls:
                continue
            image = coordinate.get("image") or {}
            member = coordinate.get("member") or {}
            results.append(
                {
                    "part": part,
                    "coordinate_url": full_url,
                    "title": str(coordinate.get("title") or "").strip(),
                    "thumbnail_url": str(image.get("url_500") or image.get("url") or "").strip(),
                    "member_name": str(member.get("name") or "").strip(),
                }
            )
            seen_urls.add(full_url)
        return results

    @classmethod
    def parse_coordinate_html(cls, html_text: str, coordinate_url: str) -> list[CatalogItem]:
        data = _extract_next_data(html_text)
        page_props = ((data.get("props") or {}).get("pageProps") or {})
        coordinate = page_props.get("coordinate") or {}
        coordinate_id = str(coordinate.get("id") or coordinate_url.rstrip("/").split("/")[-1])
        coordinate_tags = [str(tag.get("name") or "").strip() for tag in (coordinate.get("tags") or []) if str(tag.get("name") or "").strip()]

        items: list[CatalogItem] = []
        for index, raw in enumerate(page_props.get("coordinateItems") or []):
            item = raw.get("item") or {}
            item_brand = item.get("brand") or {}
            item_category = item.get("item_category") or {}
            item_child_category = item.get("item_child_category") or {}
            item_image = raw.get("image") or {}
            item_id = str(item.get("id") or raw.get("id") or f"{coordinate_id}-{index}")
            image_urls = [value for value in [item_image.get("url_500"), item_image.get("url")] if value]

            items.append(
                CatalogItem(
                    source_site="wear",
                    item_id=item_id,
                    item_url=coordinate_url,
                    title=str(raw.get("name") or item.get("name") or "").strip() or item_id,
                    category_slug=str(item_child_category.get("keyword") or item_category.get("keyword") or "").strip(),
                    category_label=str(item_child_category.get("name") or item_category.get("name") or "").strip(),
                    brand=str(item_brand.get("name") or "").strip(),
                    thumbnail_url=str(item_image.get("url_215") or item_image.get("url_500") or item_image.get("url") or "").strip(),
                    image_urls=image_urls,
                    tags=coordinate_tags,
                    notes=str(item.get("search_condition_label") or "").strip(),
                )
            )
        return items

    @staticmethod
    def matches_brand(item: CatalogItem, brand_slug: str) -> bool:
        expected = str(brand_slug or "").strip().lower()
        if not expected:
            return True
        haystacks = [str(item.brand or "").strip().lower(), str(item.notes or "").strip().lower()]
        return any(expected and expected in text for text in haystacks)
