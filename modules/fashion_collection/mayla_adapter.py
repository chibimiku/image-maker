from __future__ import annotations

import logging
import os
import re
import tempfile
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeout

from .models import CatalogItem, PART_BAG, PART_DRESS, PART_HAIR_ACCESSORY, PART_SHOES, PART_SOCKS

logger = logging.getLogger(__name__)

CLASSIFICATION_BASE = "https://mayla.jp/main/category/classification/"

# Maps our fashion parts to mayla classification kn codes.
# Each part maps to a list of kn codes that may contain relevant items.
PART_TO_KN_CODES: dict[str, list[str]] = {
    PART_SHOES: ["b0", "b1", "b2", "b3", "b4", "b8", "bC", "b5"],
    PART_SOCKS: ["bY"],
    PART_DRESS: ["bD", "bE", "bF", "bG", "bL"],
    PART_HAIR_ACCESSORY: ["bM", "b6", "bN", "bP", "bi", "bK", "bR", "bB"],
    PART_BAG: ["b9", "bT", "bU"],
}

# Human-readable labels for kn codes (for logging)
KN_LABELS: dict[str, str] = {
    "b0": "パンプス", "b1": "サンダル", "b2": "ブーツ", "b3": "ブーティ",
    "b4": "ローファー", "b5": "レインシューズ", "b8": "スニーカー", "bC": "オックスフォード",
    "bD": "ワンピース", "bE": "トップス", "bF": "ボトムス", "bG": "アウター", "bL": "ルームウェア",
    "bM": "イヤオブジェ", "b6": "ヘアオブジェ", "bN": "ネックオブジェ",
    "bP": "リングオブジェ", "bi": "リストオブジェ", "bK": "マスクリーフ",
    "bR": "チャーム", "bB": "帽子",
    "bY": "靴下",
    "b9": "バッグ", "bT": "財布", "bU": "ポーチ",
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

MAX_RETRIES = 2


class MaylaAdapter:
    """Fetches product data from mayla.jp using Playwright browser automation.

    The mayla.jp site uses JavaScript rendering and cannot be scraped with
    plain HTTP requests.  This adapter drives a headless Chromium browser
    under the hood.
    """

    site_name = "MAYLA"

    def __init__(self, timeout: int = 30, headless: bool = True):
        self.timeout = max(10, int(timeout))
        self.headless = headless
        self._browser: Browser | None = None
        self._playwright = None

    # ------------------------------------------------------------------
    # Public API used by FashionCollectionService
    # ------------------------------------------------------------------

    def build_search_url(self, brand_slug: str = "") -> str:
        return CLASSIFICATION_BASE + "?kn=b0"

    def get_part_search_url(self, part: str) -> str:
        kn_codes = PART_TO_KN_CODES.get(part, ["b0"])
        return CLASSIFICATION_BASE + "?kn=" + kn_codes[0]

    def search_products(self, part: str, max_pages: int = 1) -> list[dict]:
        """Search classification pages and return raw product dicts.

        Each dict: {'product_url', 'title', 'thumbnail_url', 'part'}
        """
        kn_codes = PART_TO_KN_CODES.get(part, ["b0"])
        results: list[dict] = []
        seen_urls: set[str] = set()

        for kn in kn_codes:
            url = CLASSIFICATION_BASE + "?kn=" + kn
            label = KN_LABELS.get(kn, kn)
            logger.info("MAYLA search %s category: %s (%s)", part, label, url)

            page = self._new_page()
            try:
                self._goto(page, url)
                items = self._parse_classification_page(page, part, seen_urls)
                results.extend(items)
                logger.info("MAYLA %s: found %d items in %s", part, len(items), label)
            except Exception as exc:
                logger.warning("MAYLA search %s (%s) failed: %s", part, kn, exc)
            finally:
                page.close()

            if max_pages > 1:
                # Some categories may have pagination via pageno parameter
                for p in range(2, max_pages + 1):
                    page_url = f"{url}&pageno={p}" if "?" in url else f"{url}?pageno={p}"
                    page = self._new_page()
                    try:
                        self._goto(page, page_url)
                        items = self._parse_classification_page(page, part, seen_urls)
                        if not items:
                            page.close()
                            break
                        results.extend(items)
                    except Exception:
                        pass
                    finally:
                        page.close()

        return results

    def fetch_product_detail(self, product_url: str) -> CatalogItem:
        """Navigate to a product page and extract its detail info."""
        page = self._new_page()
        try:
            self._goto(page, product_url)
            return self._parse_product_page(page, product_url)
        finally:
            page.close()

    @staticmethod
    def matches_brand(item: CatalogItem, brand_slug: str) -> bool:
        expected = str(brand_slug or "").strip().lower()
        if not expected:
            return True
        haystacks = [str(item.brand or "").strip().lower(), str(item.notes or "").strip().lower()]
        return any(expected and expected in text for text in haystacks)

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    def _ensure_browser(self):
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)

    def _new_page(self) -> Page:
        self._ensure_browser()
        context = self._browser.new_context(
            user_agent=DEFAULT_HEADERS["User-Agent"],
            locale="ja-JP",
        )
        return context.new_page()

    def close(self):
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Page helpers
    # ------------------------------------------------------------------

    def _goto(self, page: Page, url: str):
        for attempt in range(MAX_RETRIES + 1):
            try:
                page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                return
            except PlaywrightTimeout:
                if attempt >= MAX_RETRIES:
                    raise
                logger.debug("MAYLA retry goto %s (attempt %d)", url, attempt + 1)

    # ------------------------------------------------------------------
    # Classification page parser
    # ------------------------------------------------------------------

    def _parse_classification_page(self, page: Page, part: str, seen_urls: set[str]) -> list[dict]:
        """Extract product cards from a classification listing page."""
        results: list[dict] = []

        links = page.eval_on_selector_all(
            "a",
            """els => els.map(el => ({
                href: el.href,
                text: el.textContent.trim()
            }))""",
        )

        # Build a thumbnail map: product_code -> image_src
        thumb_map: dict[str, str] = {}
        imgs = page.eval_on_selector_all(
            "img",
            "els => els.map(el => ({src: el.src}))",
        )
        for img in imgs:
            src = (img.get("src") or "").strip()
            if not src or "nostock" in src or "btn_" in src or "tracking" in src.lower():
                continue
            m = re.search(r"/([a-z]{2,4}\d+)", src)
            if m:
                code = m.group(1)
                if code not in thumb_map:
                    thumb_map[code] = src

        for link in links:
            href = (link.get("href") or "").strip()
            if "/SHOP/" not in href:
                continue
            if href in seen_urls:
                continue
            seen_urls.add(href)

            text = (link.get("text") or "").strip()
            if not text:
                continue

            # Look up thumbnail by product code
            code_match = re.search(r"/SHOP/([^.]+)\.html", href)
            thumbnail_url = ""
            if code_match:
                thumbnail_url = thumb_map.get(code_match.group(1), "")

            results.append({
                "part": part,
                "product_url": href,
                "title": text,
                "thumbnail_url": thumbnail_url,
            })

        return results

    # ------------------------------------------------------------------
    # Product detail page parser
    # ------------------------------------------------------------------

    def _parse_product_page(self, page: Page, product_url: str) -> CatalogItem:
        """Extract CatalogItem from a product detail page."""
        code_match = re.search(r"/SHOP/([^.]+)\.html", product_url)
        product_code = code_match.group(1) if code_match else product_url.rstrip("/").split("/")[-1]

        # Title
        title = page.title() or ""
        # Strip site suffix
        title = re.sub(r"\s*-?\s*mayla\s*classic\s*$", "", title, flags=re.I).strip() or product_code

        # Collect all image URLs
        image_urls: list[str] = []
        imgs = page.eval_on_selector_all(
            "img",
            "els => els.map(el => ({src: el.src, className: el.className}))",
        )
        for img in imgs:
            src = (img.get("src") or "").strip()
            cls = (img.get("className") or "").lower()
            if not src or "tracking" in src.lower():
                continue
            # Prefer product images
            if "nostock" in src or "btn_" in src or "icon" in cls:
                continue
            image_urls.append(src)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for u in image_urls:
            if u not in seen:
                deduped.append(u)
                seen.add(u)

        # Main thumbnail — prefer the llimg (large) version from shopserve
        thumbnail_url = ""
        for u in deduped:
            if "llimg" in u or "mainImg" in u.lower():
                thumbnail_url = u
                break
        if not thumbnail_url and deduped:
            thumbnail_url = deduped[0]

        # Brand is always "MAYLA"
        brand = "MAYLA"

        # Extract notes / description from meta
        notes = ""
        metas = page.eval_on_selector_all(
            'meta[name="description"]',
            "els => els.map(el => el.getAttribute('content'))",
        )
        if metas and metas[0]:
            notes = str(metas[0]).strip()

        # Extract keywords as tags
        tags: list[str] = []
        kw_metas = page.eval_on_selector_all(
            'meta[name="keywords"]',
            "els => els.map(el => el.getAttribute('content'))",
        )
        if kw_metas and kw_metas[0]:
            raw_kw = str(kw_metas[0])
            tags = [t.strip() for t in raw_kw.replace("，", ",").split(",") if t.strip()]
            # Remove boilerplate tags
            tags = [t for t in tags if t not in ("ショッピング", "mayla classic", "シューズ", "可愛い", "リボン", "体温が二度上がる")]

        return CatalogItem(
            source_site="mayla",
            item_id=product_code,
            item_url=product_url,
            title=title,
            brand=brand,
            thumbnail_url=thumbnail_url,
            image_urls=deduped,
            notes=notes,
            tags=tags,
        )
