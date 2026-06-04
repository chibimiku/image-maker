"""从 Lolibrary 搜索页爬取品牌 slug 列表，缓存到本地文件。"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time

import requests

logger = logging.getLogger(__name__)

LOLIBRARY_BASE = "https://lolibrary.org"
CACHE_FILENAME = "brand_cache.json"
SCRAPE_PAGES = 60  # 爬取多少页搜索结果来收集品牌


def _get_cache_path(project_root: str | None = None) -> str:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, CACHE_FILENAME)


def scrape_brands(pages: int = SCRAPE_PAGES, session: requests.Session | None = None) -> dict[str, str]:
    """从 Lolibrary 全网搜索页抓取品牌列表，返回 {slug: display_name} 字典。"""
    session = session or requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    brands: dict[str, str] = {}
    brand_re = re.compile(r'href="https://lolibrary\.org/brands/([^"/]+)"[^>]*title="([^"]+)"', re.I)

    for page in range(1, pages + 1):
        url = f"{LOLIBRARY_BASE}/search?sort=added_new&page={page}"
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except Exception:
            logger.warning("品牌爬取: 第 %d 页请求失败，跳过", page)
            continue

        found = dict(brand_re.findall(resp.text))
        for slug, name in found.items():
            slug = slug.strip()
            name = name.strip()
            if slug and name and slug not in brands:
                brands[slug] = name

        logger.info("品牌爬取: 第 %d/%d 页, 累计 %d 个品牌", page, pages, len(brands))
        if page < pages:
            time.sleep(0.4)  # 请求间隔

    logger.info("品牌爬取完成: 共 %d 个品牌", len(brands))
    return brands


def refresh_cache(project_root: str | None = None, pages: int = SCRAPE_PAGES) -> dict[str, str]:
    """刷新品牌缓存到本地 JSON 文件，返回品牌字典。"""
    cache_path = _get_cache_path(project_root)
    brands = scrape_brands(pages=pages)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(brands, f, ensure_ascii=False, indent=2)
    logger.info("品牌缓存已保存: %s (%d 个)", cache_path, len(brands))
    return brands


def load_cache(project_root: str | None = None) -> dict[str, str] | None:
    """读取本地品牌缓存，不存在返回 None。"""
    cache_path = _get_cache_path(project_root)
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return data
    except Exception:
        logger.warning("品牌缓存读取失败", exc_info=True)
    return None


def get_or_refresh_cache(project_root: str | None = None) -> dict[str, str]:
    """获取品牌缓存，不存在则自动刷新。"""
    brands = load_cache(project_root)
    if not brands:
        logger.info("品牌缓存不存在，开始自动刷新...")
        brands = refresh_cache(project_root)
    return brands


def pick_random_brands(count: int, project_root: str | None = None) -> list[str]:
    """从缓存中随机选取 count 个 brand_slug。"""
    brands = get_or_refresh_cache(project_root)
    slugs = list(brands.keys())
    if not slugs:
        return []
    if len(slugs) <= count:
        return slugs[:]
    return random.sample(slugs, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = refresh_cache()
    print(f"共爬取 {len(result)} 个品牌")
    for s, n in sorted(result.items())[:20]:
        print(f"  {s}: {n}")
    if len(result) > 20:
        print(f"  ... 共 {len(result)} 个")
