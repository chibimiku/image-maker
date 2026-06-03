from __future__ import annotations

import html
import re
from urllib.parse import urljoin

import requests

from .models import CatalogItem
from .networking import request_with_proxy_fallback
from .site_base import FashionCatalogSiteAdapter, SearchRequest


SEARCH_URL_TEMPLATE = "https://lolibrary.org/search?brands[]={brand_slug}&page={page}"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return " ".join(html.unescape(text).split()).strip()


def _extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.I | re.S)
    return match.group(1).strip() if match else ""


def _extract_all(pattern: str, text: str) -> list[str]:
    results = []
    for match in re.findall(pattern, text, re.I | re.S):
        value = match[0] if isinstance(match, tuple) else match
        value = html.unescape(str(value or "").strip())
        if value and value not in results:
            results.append(value)
    return results


class LolibraryAdapter(FashionCatalogSiteAdapter):
    site_name = "Lolibrary"

    def __init__(self, session: requests.Session | None = None, timeout: int = 20):
        self.session = session or requests.Session()
        self.timeout = max(5, int(timeout))
        self.session.headers.update(DEFAULT_HEADERS)

    def build_search_url(self, brand_slug: str, categories: list[str] | None = None) -> str:
        url = SEARCH_URL_TEMPLATE.format(brand_slug=brand_slug, page=1)
        for cat in (categories or []):
            url = url + f"&categories[]={cat}&categories_matcher=OR"
        return url

    def _net(self):
        """Get proxy and log callbacks for internal use."""
        proxy = getattr(self, 'proxy_url', None)
        log = getattr(self, 'log_callback', None)
        return proxy, log

    def search_items(self, request: SearchRequest) -> list[CatalogItem]:
        results: list[CatalogItem] = []
        seen_ids: set[str] = set()
        categories = request.categories if hasattr(request, 'categories') else []
        max_page = max(1, request.max_pages)
        proxy, log = self._net()
        for page in range(1, max_page + 1):
            if log:
                log(f"[Lolibrary] 抓取第 {page}/{max_page} 页...")
            url = SEARCH_URL_TEMPLATE.format(brand_slug=request.brand_slug, page=page)
            for cat in categories:
                url = url + f"&categories[]={cat}&categories_matcher=OR"
            response = request_with_proxy_fallback(self.session, "GET", url, timeout=self.timeout, proxy_url=proxy, log_callback=log)
            response.raise_for_status()
            page_items = self.parse_search_html(response.text, base_url="https://lolibrary.org", seen_ids=seen_ids)
            if log:
                log(f"[Lolibrary] 第 {page} 页解析到 {len(page_items)} 件商品 (累计 {len(results) + len(page_items)})")
            results.extend(page_items)
        return results

    def fetch_item_detail(self, item_url: str) -> CatalogItem:
        proxy, log = self._net()
        response = request_with_proxy_fallback(self.session, "GET", item_url, timeout=self.timeout, proxy_url=proxy, log_callback=log)
        response.raise_for_status()
        return self.parse_item_html(response.text, item_url=item_url)

    @classmethod
    def parse_search_html(cls, html_text: str, base_url: str = "https://lolibrary.org", seen_ids: set[str] | None = None) -> list[CatalogItem]:
        seen_ids = seen_ids or set()
        items: list[CatalogItem] = []
        item_link_pattern = re.compile(
            r'<a\s+href="(?P<href>https://lolibrary\.org/items/[^"]+|/items/[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.I | re.S,
        )
        for match in item_link_pattern.finditer(html_text):
            raw_href = match.group("href")
            item_href = urljoin(base_url, raw_href)
            item_id = item_href.rstrip("/").split("/")[-1]
            if item_id in seen_ids:
                continue
            title = _strip_tags(match.group("title"))
            context = html_text[match.start() : match.start() + 1200]
            thumbnail_url = html.unescape(_extract_first(r'<img src="([^"]+)"', context))
            brand = _strip_tags(
                _extract_first(r'<a href="https://lolibrary\.org/brands/[^"]+"[^>]*title="([^"]+)"', context)
            )
            category_label = _strip_tags(
                _extract_first(r'<a href="https://lolibrary\.org/categories/[^"]+"[^>]*title="([^"]+)"', context)
            )
            category_href = _extract_first(r'<a href="https://lolibrary\.org/categories/([^"]+)"', context)
            items.append(
                CatalogItem(
                    source_site="lolibrary",
                    item_id=item_id,
                    item_url=item_href,
                    title=title or item_id,
                    category_slug=(category_href or "").split("?")[0].strip("/"),
                    category_label=category_label,
                    brand=brand,
                    thumbnail_url=thumbnail_url,
                )
            )
            seen_ids.add(item_id)
        return items

    @classmethod
    def parse_item_html(cls, html_text: str, item_url: str) -> CatalogItem:
        item_id = item_url.rstrip("/").split("/")[-1]
        title = _strip_tags(_extract_first(r"<h1[^>]*>\s*(.*?)\s*</h1>", html_text))
        brand = _strip_tags(_extract_first(r'#### Brand.*?<a [^>]*>\s*(.*?)\s*</a>', html_text))
        category_label = _strip_tags(_extract_first(r'#### Category.*?<a [^>]*>\s*(.*?)\s*</a>', html_text))
        category_href = _extract_first(r'#### Category.*?<a href="https://lolibrary\.org/categories/([^"]+)"', html_text)
        image_urls = _extract_all(r'src="(https://lolibrary\.global\.ssl\.fastly\.net/images/[^"?]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"', html_text)

        og_image = html.unescape(_extract_first(r'property="og:image"\s+content="([^"]+)"', html_text))
        if og_image and og_image not in image_urls:
            image_urls.insert(0, og_image)

        notes = _strip_tags(_extract_first(r"#### Notes(.*?)(?:####|$)", html_text))
        tags = [_strip_tags(tag) for tag in _extract_all(r'#### Tags(.*?)(?:####|$)', html_text)]

        # The markdown fetch may flatten tags; keep a simple fallback from raw text.
        tag_values = []
        tags_block = _extract_first(r"#### Tags(.*?)(?:####|$)", html_text)
        if tags_block:
            tag_values = [value for value in _extract_all(r"\[\s*(.*?)\s*\]\(", tags_block) if value]

        return CatalogItem(
            source_site="lolibrary",
            item_id=item_id,
            item_url=item_url,
            title=title or item_id,
            category_slug=(category_href or "").split("?")[0].strip("/"),
            category_label=category_label,
            brand=brand,
            image_urls=image_urls,
            notes=notes,
            tags=tag_values or tags,
        )
