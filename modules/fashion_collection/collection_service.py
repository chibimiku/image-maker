from __future__ import annotations

import json as _json_module
import logging
import os
import random
import re
from typing import Callable
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

# Project root for prompt file loading
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from .lolibrary_adapter import LolibraryAdapter
from .mayla_adapter import MaylaAdapter
from .wear_adapter import WearAdapter
from .models import (
    CollectionBundle,
    CollectedAsset,
    PART_BAG,
    PART_DRESS,
    PART_HAIR_ACCESSORY,
    PART_SHOES,
    PART_SOCKS,
    SUPPORTED_PARTS,
)
from .site_base import SearchRequest
from .networking import request_with_proxy_fallback
from .theme_profiles import ThemeProfile, get_theme_profile


PART_CATEGORY_KEYWORDS = {
    PART_DRESS: {"jsk", "op", "jumperskirt", "one piece", "onepiece", "dress", "sets"},
    PART_SHOES: {"shoes", "shoe", "heels", "boots", "pumps"},
    PART_SOCKS: {"socks", "sock", "otks", "utks", "tights", "leggings"},
    PART_HAIR_ACCESSORY: {"hair accessory", "hair-accessory", "hair band", "head band", "barrette", "hair clip", "hairpin", "scrunchies", "headpiece"},
    PART_BAG: {"bag", "handbag", "shoulder bag", "tote bag", "backpack", "basket bag", "clutch"},
}

PART_PROMPT_HINTS = {
    PART_DRESS: "用于参考连衣裙/主服装的版型、花纹、材质与配色。",
    PART_SHOES: "用于参考鞋子的鞋型、鞋跟、装饰、材质与颜色。",
    PART_SOCKS: "用于参考袜子的长度、花边、图案、透明度与颜色。",
    PART_HAIR_ACCESSORY: "用于参考发饰的佩戴位置、蝴蝶结/花朵/发箍等结构、材质与颜色。",
    PART_BAG: "用于参考包袋或手持物的造型、材质、金属件、提手与配色。",
}

# Maps fashion parts to Lolibrary category slugs for targeted category search.
# Only includes categories confirmed to exist on Lolibrary. Parts without a
# mapped category fall back to brand-only search (which works well for dresses
# but may miss niche items like accessories).
PART_LOLIBRARY_CATEGORIES: dict[str, list[str]] = {
    PART_DRESS: [],
    PART_SHOES: ["shoes"],
    PART_SOCKS: ["socks"],
    PART_HAIR_ACCESSORY: [],
    PART_BAG: [],
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
    candidates = build_image_download_candidates(image_urls, fallback_url)
    return candidates[0] if candidates else ""


def _image_download_priority(url: str) -> tuple[int, int, int]:
    parsed = urlparse(str(url or ""))
    query = parse_qs(parsed.query)
    width = 99999
    try:
        width = int((query.get("width") or ["99999"])[0])
    except Exception:
        width = 99999
    has_query = 0 if parsed.query else 1
    is_fastly_image = 0 if "/images/" in parsed.path else 1
    return (is_fastly_image, has_query, width)


def build_image_download_candidates(image_urls: list[str], fallback_url: str = "") -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for url in sorted([str(u) for u in (image_urls or []) if str(u or "").strip()], key=_image_download_priority):
        if url not in seen:
            ordered.append(url)
            seen.add(url)
    fallback = str(fallback_url or "").strip()
    if fallback and fallback not in seen:
        ordered.append(fallback)
    return ordered


def build_item_haystack(item) -> str:
    return " ".join(
        [
            str(getattr(item, "title", "") or ""),
            str(getattr(item, "brand", "") or ""),
            str(getattr(item, "category_slug", "") or ""),
            str(getattr(item, "category_label", "") or ""),
            str(getattr(item, "notes", "") or ""),
            " ".join(getattr(item, "tags", []) or []),
        ]
    ).lower()


def score_item_for_theme(item, part: str, theme_profile: ThemeProfile | None) -> int:
    if not theme_profile:
        return 0

    haystack = build_item_haystack(item)
    score = 0

    for keyword in theme_profile.style_tags:
        if keyword.lower() in haystack:
            score += 8
    for keyword in theme_profile.color_tags:
        if keyword.lower() in haystack:
            score += 6
    for keyword in theme_profile.element_tags:
        if keyword.lower() in haystack:
            score += 6
    for keyword in theme_profile.negative_tags:
        if keyword.lower() in haystack:
            score -= 8
    if part == PART_DRESS:
        for keyword in theme_profile.dress_tags:
            if keyword.lower() in haystack:
                score += 8
        for brand in theme_profile.dress_brands:
            if brand.lower() in haystack:
                score += 12
    elif part == PART_SHOES:
        for keyword in theme_profile.shoes_tags:
            if keyword.lower() in haystack:
                score += 8
    elif part == PART_SOCKS:
        for keyword in theme_profile.socks_tags:
            if keyword.lower() in haystack:
                score += 8
    elif part == PART_HAIR_ACCESSORY:
        for keyword in getattr(theme_profile, "hair_accessory_tags", ()):
            if keyword.lower() in haystack:
                score += 8
    elif part == PART_BAG:
        for keyword in getattr(theme_profile, "bag_tags", ()):
            if keyword.lower() in haystack:
                score += 8
    return score


# ---------------------------------------------------------------------------
# Color coordination system
# ---------------------------------------------------------------------------

# Color keywords mapped to their canonical color group.
# Format: (keyword_regex, canonical_group)
# Groups: pink, red, blue, purple, yellow, green, mint, brown, black, white, neutral
_COLOR_PATTERNS: list[tuple[str, str]] = [
    # pink
    (r"\bpink\b", "pink"), (r"\bピンク\b", "pink"),
    (r"\bsalmon\b", "pink"), (r"\brose\b", "pink"), (r"\brose\b", "pink"),
    (r"\bsax\b", "blue"), (r"\bサックス\b", "blue"),
    (r"\blavender\b", "purple"), (r"\bラベンダー\b", "purple"),
    (r"\blilac\b", "purple"),
    (r"\bmint\b", "mint"), (r"\bミント\b", "mint"),
    # red
    (r"\bred\b", "red"), (r"\b赤\b", "red"), (r"\bレッド\b", "red"),
    (r"\bwine\b", "red"), (r"\bワイン\b", "red"),
    (r"\bbordeaux\b", "red"), (r"\bボルドー\b", "red"),
    # blue
    (r"\bblue\b", "blue"), (r"\b青\b", "blue"), (r"\bブルー\b", "blue"),
    (r"\bnavy\b", "blue"), (r"\bネイビー\b", "blue"),
    # purple
    (r"\bpurple\b", "purple"), (r"\b紫\b", "purple"), (r"\bパープル\b", "purple"),
    # yellow
    (r"\byellow\b", "yellow"), (r"\b黄\b", "yellow"), (r"\bイエロー\b", "yellow"),
    (r"\bgold\b", "yellow"), (r"\bゴールド\b", "yellow"),
    # green
    (r"\bgreen\b", "green"), (r"\b緑\b", "green"), (r"\bグリーン\b", "green"),
    # brown
    (r"\bbrown\b", "brown"), (r"\b茶\b", "brown"), (r"\bブラウン\b", "brown"),
    # black
    (r"\bblack\b", "black"), (r"\b黒\b", "black"), (r"\bブラック\b", "black"),
    # white / neutral
    (r"\bwhite\b", "white"), (r"\b白\b", "white"), (r"\bホワイト\b", "white"),
    (r"\bcream\b", "neutral"), (r"\bクリーム\b", "neutral"),
    (r"\bivory\b", "neutral"), (r"\bアイボリー\b", "neutral"),
    (r"\bbeige\b", "neutral"), (r"\bベージュ\b", "neutral"),
    (r"\boff.?white\b", "neutral"),
    (r"\bsilver\b", "neutral"), (r"\bシルバー\b", "neutral"),
    (r"\bgray\b", "neutral"), (r"\bgrey\b", "neutral"),
    (r"\bグレー\b", "neutral"), (r"\bジェイ\b", "neutral"),
    (r"\bcoral\b", "pink"),
    (r"\borange\b", "red"), (r"\bオレンジ\b", "red"),
    (r"\bchocolate\b", "brown"),
    (r"\bpearl\b", "white"),  # pearls = white/neutral aesthetic
]

# Which groups are considered compatible with each other.
# "neutral" and "white" are universal — compatible with everything.
# Items within the same group are compatible.
_COLOR_COMPAT_MATRIX: dict[str, set[str]] = {
    "pink":    {"pink", "white", "neutral", "red", "purple", "mint"},
    "red":     {"red", "pink", "white", "neutral", "black", "yellow"},
    "blue":    {"blue", "white", "neutral", "purple", "mint", "black"},
    "purple":  {"purple", "pink", "blue", "white", "neutral", "black"},
    "yellow":  {"yellow", "red", "white", "neutral", "brown", "green", "black"},
    "green":   {"green", "yellow", "white", "neutral", "brown", "mint"},
    "mint":    {"mint", "green", "blue", "pink", "white", "neutral"},
    "brown":   {"brown", "yellow", "green", "white", "neutral", "black"},
    "black":   {"black", "red", "white", "neutral", "blue", "purple", "brown", "yellow"},
    "white":   {"white", "pink", "red", "blue", "purple", "yellow", "green", "mint", "brown", "black", "neutral"},
    "neutral": {"neutral", "pink", "red", "blue", "purple", "yellow", "green", "mint", "brown", "black", "white"},
}


def _extract_colors(item) -> frozenset[str]:
    """Extract canonical color groups from an item's title, tags, and notes."""
    title = str(getattr(item, "title", "") or "")
    tags = " ".join(getattr(item, "tags", []) or [])
    notes = str(getattr(item, "notes", "") or "")
    cat_label = str(getattr(item, "category_label", "") or "")
    text = " ".join([title, tags, notes, cat_label]).lower()

    groups: set[str] = set()
    matched_details: list[str] = []
    for pattern, group in _COLOR_PATTERNS:
        m = re.search(pattern, text)
        if m:
            groups.add(group)
            matched_details.append(f"{pattern}={group}")

    logger.info(
        "颜色提取: item=%s, 扫描文本长度=%d, 命中=%s, 命中详情=[%s]",
        title[:80], len(text), sorted(groups), ", ".join(matched_details) if matched_details else "无",
    )
    return frozenset(groups)


def _colors_compatible(dress_colors: frozenset[str], accessory_colors: frozenset[str]) -> bool:
    """Check if accessory colors are compatible with the dress's color palette.

    Returns True if:
    - Either set is empty (no color data → can't judge, assume compatible)
    - The accessory has only neutral/white colors
    - At least one accessory color group is compatible with at least one dress color group
    """
    logger.info(
        "颜色协调判断: 裙子色组=%s, 配件色组=%s",
        sorted(dress_colors), sorted(accessory_colors),
    )
    if not dress_colors or not accessory_colors:
        logger.info("颜色协调判断: 有一方无颜色数据，默认兼容通过")
        return True
    # If accessory is all neutral/white, it's always compatible
    if accessory_colors.issubset({"white", "neutral"}):
        logger.info("颜色协调判断: 配件全为白/中性色，自动兼容通过")
        return True
    for a_color in accessory_colors:
        compat = _COLOR_COMPAT_MATRIX.get(a_color, set())
        intersection = compat.intersection(dress_colors)
        logger.info(
            "颜色协调判断: 配件色=%s, 兼容组=%s, 与裙子交集=%s",
            a_color, sorted(compat), sorted(intersection),
        )
        if intersection:
            logger.info("颜色协调判断: 配件色 %s 与裙子匹配，兼容通过", a_color)
            return True
    logger.info("颜色协调判断: 无兼容色组，判定为不兼容")
    return False


def _load_color_analysis_prompts() -> tuple[str, str]:
    """Load color analysis prompt templates from the prompts/ directory."""
    system_path = os.path.join(_PROJECT_ROOT, "prompts", "fashion-color-analysis-system.md")
    user_path = os.path.join(_PROJECT_ROOT, "prompts", "fashion-color-analysis-user.md")
    logger.info("颜色/LLM: 加载 prompt 模板, system=%s, user=%s", system_path, user_path)
    system = ""
    user = ""
    if os.path.isfile(system_path):
        with open(system_path, "r", encoding="utf-8") as f:
            system = f.read()
        logger.info("颜色/LLM: system prompt 加载成功, 长度=%d", len(system))
    else:
        logger.warning("颜色/LLM: system prompt 文件不存在: %s", system_path)
    if os.path.isfile(user_path):
        with open(user_path, "r", encoding="utf-8") as f:
            user = f.read()
        logger.info("颜色/LLM: user prompt 加载成功, 长度=%d", len(user))
    else:
        logger.warning("颜色/LLM: user prompt 文件不存在: %s", user_path)
    return system, user


def _analyze_dress_colors_via_llm(dress_item, log_callback=None) -> dict | None:
    """Use LLM to analyze the dress's color profile for coordination.

    Returns a dict with keys: primary_color, secondary_colors, compatible_accessory_colors,
    avoid_accessory_colors, etc.  Returns None on failure.
    """

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    dress_title = str(getattr(dress_item, "title", "") or "")
    dress_brand = str(getattr(dress_item, "brand", "") or "")
    dress_notes = str(getattr(dress_item, "notes", "") or "")
    dress_tags = ", ".join(getattr(dress_item, "tags", []) or [])
    logger.info(
        "颜色/LLM: 开始分析连衣裙色调, title=%s, brand=%s, notes_len=%d, tags=%s",
        dress_title[:100], dress_brand, len(dress_notes), dress_tags[:200],
    )

    # Load prompt templates
    system_prompt, user_template = _load_color_analysis_prompts()
    if not system_prompt or not user_template:
        logger.warning("颜色/LLM: Prompt 模板缺失 (system=%s, user=%s)，回退到规则匹配", bool(system_prompt), bool(user_template))
        _log("[颜色/LLM] Prompt 模板缺失，回退到规则匹配")
        return None

    # Build user prompt
    user_prompt = user_template
    user_prompt = user_prompt.replace("{{dress_title}}", dress_title)
    user_prompt = user_prompt.replace("{{dress_brand}}", dress_brand)
    user_prompt = user_prompt.replace("{{dress_notes}}", dress_notes or "无")
    user_prompt = user_prompt.replace("{{dress_tags}}", dress_tags or "无")
    logger.info("颜色/LLM: user prompt 构建完成, 长度=%d, 前200字=%s", len(user_prompt), user_prompt[:200])

    # Get API config (same endpoint as image generation / analysis)
    try:
        from modules.others.api_backend import get_api_config, fetch_llm_json

        cfg = get_api_config(api_type="aigc2d")
        api_base = str(cfg.get("base_url", "") or "").strip()
        api_key = cfg.get("api_key", "")
        logger.info("颜色/LLM: API 配置, base_url=%s, has_key=%s", api_base[:80] if api_base else "(空)", bool(api_key))
        if not api_key:
            logger.warning("颜色/LLM: api_key 缺失，回退到规则匹配")
            _log("[颜色/LLM] api_key 缺失，回退到规则匹配")
            return None
        if "/v1beta/models/" in api_base:
            api_base = api_base.split("/v1beta/models/")[0] + "/v1"
            logger.info("颜色/LLM: 修正 api_base -> %s", api_base[:80])

        # Use a fast chat model for this text-only analysis
        model = "gemini-2.5-flash"
        logger.info("颜色/LLM: 模型=%s, temperature=0.3", model)

        _log(f"[颜色/LLM] 正在请求 LLM 分析 {dress_title[:60]} 的色调...")
        raw = fetch_llm_json(
            base_url=api_base,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_content=user_prompt,
            temperature=0.3,
        )
        logger.info("颜色/LLM: fetch_llm_json 返回, raw_type=%s, raw_len=%d, 前300字=%s",
                     type(raw).__name__, len(str(raw)), str(raw)[:300])

        result = _json_module.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(result, dict):
            logger.warning("颜色/LLM: LLM 返回非 dict (type=%s)，回退到规则匹配", type(result).__name__)
            _log("[颜色/LLM] LLM 返回格式异常，回退到规则匹配")
            return None

        logger.info("颜色/LLM: 解析结果 keys=%s", list(result.keys()))
        primary = str(result.get("primary_color", "") or "").strip()
        if primary:
            secondary = [str(c) for c in (result.get("secondary_colors") or []) if str(c or "").strip()]
            compatible = [str(c) for c in (result.get("compatible_accessory_colors") or []) if str(c or "").strip()]
            avoid = [str(c) for c in (result.get("avoid_accessory_colors") or []) if str(c or "").strip()]
            reasoning = str(result.get("reasoning", "") or "").strip()
            logger.info(
                "颜色/LLM: 主色=%s, 辅色=%s, 推荐配件色=%s, 避免配件色=%s, 理由=%s",
                primary, secondary, compatible, avoid, reasoning[:120] if reasoning else "(空)",
            )
            _log(f"[颜色/LLM] 主色={primary}, 辅色={secondary}, 推荐={compatible}, 避免={avoid}")
            if reasoning:
                _log(f"[颜色/LLM] 理由: {reasoning}")
            return {
                "primary_color": primary.lower(),
                "secondary_colors": [c.lower() for c in secondary],
                "compatible_colors": [c.lower() for c in compatible],
                "avoid_colors": [c.lower() for c in avoid],
                "reasoning": reasoning,
            }
        else:
            logger.warning("颜色/LLM: primary_color 为空，回退到规则匹配。完整返回=%s", str(result)[:500])
            _log("[颜色/LLM] LLM 未返回有效主色，回退到规则匹配")
            return None

    except Exception as exc:
        logger.error("颜色/LLM: 请求异常 (%s: %s)，回退到规则匹配", type(exc).__name__, exc, exc_info=True)
        _log(f"[颜色/LLM] LLM 请求异常 ({exc})，回退到规则匹配")
        return None


def _enrich_item_colors(item, adapter, log_callback=None) -> frozenset[str]:
    """Fetch item detail to get richer color data, then extract colors.

    Falls back to basic extraction from search data if detail fetch fails.
    """
    colors = _extract_colors(item)
    item_title = str(getattr(item, "title", "") or "")
    # Already got good color data from search info
    if len(colors) >= 1:
        logger.info("颜色富化: 搜索页已提取到颜色=%s, item=%s, 无需请求详情页", sorted(colors), item_title[:80])
        return colors
    # Try detail page for richer data
    logger.info("颜色富化: 搜索页无颜色数据, item=%s, 尝试请求详情页 %s", item_title[:80], getattr(item, "item_url", "?"))
    try:
        detail = adapter.fetch_item_detail(item.item_url)
        detail_colors = _extract_colors(detail)
        logger.info("颜色富化: 详情页颜色提取结果=%s", sorted(detail_colors))
        return detail_colors
    except Exception as exc:
        logger.warning("颜色富化: 详情页请求失败 (%s: %s), 回退", type(exc).__name__, exc)
        return colors


# ---------------------------------------------------------------------------


class FashionCollectionService:
    def __init__(self, adapter: LolibraryAdapter | None = None, timeout: int = 8, proxy_url: str | None = None):
        self.lolibrary_adapter = adapter or LolibraryAdapter(timeout=timeout)
        self.wear_adapter = WearAdapter(timeout=timeout)
        self.mayla_adapter = MaylaAdapter(timeout=timeout * 4)
        self.adapter = self.lolibrary_adapter
        self.session = self.adapter.session
        self.timeout = timeout
        self.proxy_url = proxy_url
        self.max_theme_candidates_per_part = 8
        self.enable_color_match = False

    def set_proxy_url(self, proxy_url: str | None) -> None:
        self.proxy_url = str(proxy_url).strip() if proxy_url else None

    def collect_bundle(
        self,
        site_key: str,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
        theme: str = "",
        log_callback: Callable[[str], None] | None = None,
    ) -> CollectionBundle:
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        normalized_site = str(site_key or "").strip().lower()
        _log(f"[Fashion] 开始采集，站点={normalized_site}，品牌={brand_slug or '不限'}，部位={preferred_parts}，主题={theme}，扫描页数={max_pages}")
        if normalized_site == "hybrid":
            return self.collect_hybrid_bundle(
                brand_slug=brand_slug,
                output_dir=output_dir,
                max_pages=max_pages,
                preferred_parts=preferred_parts,
                theme=theme,
                log_callback=log_callback,
            )
        if normalized_site == "wear":
            return self.collect_wear_bundle(
                brand_slug=brand_slug,
                output_dir=output_dir,
                max_pages=max_pages,
                preferred_parts=preferred_parts,
                theme=theme,
                log_callback=log_callback,
            )
        if normalized_site == "mayla":
            return self.collect_mayla_bundle(
                brand_slug=brand_slug,
                output_dir=output_dir,
                max_pages=max_pages,
                preferred_parts=preferred_parts,
                theme=theme,
                log_callback=log_callback,
            )
        return self.collect_lolibrary_bundle(
            brand_slug=brand_slug,
            output_dir=output_dir,
            max_pages=max_pages,
            preferred_parts=preferred_parts,
            theme=theme,
            log_callback=log_callback,
        )

    def collect_lolibrary_bundle(
        self,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
        theme: str = "",
        log_callback: Callable[[str], None] | None = None,
    ) -> CollectionBundle:
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        self.adapter = self.lolibrary_adapter
        self.adapter.log_callback = log_callback
        self.adapter.proxy_url = self.proxy_url
        self.session = self.adapter.session
        preferred_parts = [part for part in (preferred_parts or list(SUPPORTED_PARTS)) if part in SUPPORTED_PARTS]
        os.makedirs(output_dir, exist_ok=True)
        theme_profile = get_theme_profile(theme)

        selected_assets: list[CollectedAsset] = []
        taken_parts: set[str] = set()
        search_url = self.adapter.build_search_url(brand_slug)
        _log(f"[Lolibrary] 开始采集 {len(preferred_parts)} 个部位: {', '.join(preferred_parts)}")

        # Color coordination state
        dress_colors: frozenset[str] | None = None
        dress_llm_colors: dict[str, object] | None = None  # LLM-enhanced result
        color_match_enabled = bool(self.enable_color_match and PART_DRESS in preferred_parts)
        if color_match_enabled:
            logger.info("Lolibrary颜色协调: 已启用, preferred_parts=%s", preferred_parts)
            _log("[Lolibrary/颜色] 颜色协调已启用，将先选取连衣裙，后续配件颜色需与裙子适配")

        for part in preferred_parts:
            _log(f"[Lolibrary/{part}] 开始搜索...")
            categories = PART_LOLIBRARY_CATEGORIES.get(part, [])
            # Try category-filtered search first; if no results, fall back to
            # brand-only search (no category filter).
            attempts: list[list[str]] = [categories] if categories else [[]]
            if categories:
                attempts.append([])
            for req_cats in attempts:
                cat_label = f"分类={req_cats[0]}" if req_cats else "品牌全量搜索"
                _log(f"[Lolibrary/{part}] 尝试搜索方式: {cat_label}")
                request = SearchRequest(
                    brand_slug=brand_slug,
                    max_pages=max_pages,
                    preferred_parts=[part],
                    site_key="lolibrary",
                    categories=req_cats,
                )
                search_items = self.adapter.search_items(request)
                _log(f"[Lolibrary/{part}] 搜索返回 {len(search_items)} 件商品")
                if not search_items:
                    _log(f"[Lolibrary/{part}] 无结果，尝试其他搜索方式...")
                    continue

                candidates: list[tuple[int, CollectedAsset, list[str]]] = []
                for item in search_items:
                    inferred = infer_part_from_item(item.category_slug, item.title, item.tags)
                    if inferred and inferred != part:
                        continue
                    image_candidates = build_image_download_candidates([item.thumbnail_url], item.thumbnail_url)
                    if not image_candidates:
                        continue
                    part_candidate = CollectedAsset(
                        part=part,
                        item=item,
                        image_url=image_candidates[0],
                        local_path="",
                        source_search_url=search_url,
                        prompt_hint=PART_PROMPT_HINTS.get(part, ""),
                    )
                    score = score_item_for_theme(item, part, theme_profile)
                    candidates.append((score, part_candidate, image_candidates))

                if not candidates:
                    _log(f"[Lolibrary/{part}] 主题匹配后无候选项，尝试其他搜索方式...")
                    continue

                candidates.sort(key=lambda t: t[0], reverse=True)
                _log(f"[Lolibrary/{part}] 主题匹配 {len(candidates)} 个候选项 (最高分={candidates[0][0]})，随机打乱后逐个尝试下载...")
                random.shuffle(candidates)

                for idx, (score, part_candidate, img_candidates) in enumerate(candidates):
                    _log(f"[Lolibrary/{part}] [{idx+1}/{len(candidates)}] 尝试: {part_candidate.item.title} (评分={score})")

                    # --- color coordination check for accessories ---
                    if color_match_enabled and (dress_colors or dress_llm_colors) and part != PART_DRESS:
                        item_colors = _extract_colors(part_candidate.item)
                        ok = True
                        logger.info(
                            "Lolibrary颜色检查: part=%s, item=%s, 配件色=%s, 裙子色(规则)=%s, 裙子色(LLM)=%s",
                            part, part_candidate.item.title[:80], sorted(item_colors),
                            sorted(dress_colors) if dress_colors else [],
                            dress_llm_colors.get("primary_color") if dress_llm_colors else "无",
                        )
                        # Phase 1: LLM-enhanced check (if available)
                        if dress_llm_colors and isinstance(dress_llm_colors, dict):
                            avoid = dress_llm_colors.get("avoid_colors") or []
                            compat = dress_llm_colors.get("compatible_colors") or []
                            for ic in item_colors:
                                if any(a in ic or ic in a for a in avoid):
                                    ok = False
                                    logger.info("Lolibrary颜色检查: LLM避免色命中, 配件色=%s, 避免列表=%s", ic, avoid)
                                    _log(f"[Lolibrary/{part}] 颜色不协调(LLM): 避免色={avoid}, 配件色={sorted(item_colors) if item_colors else '未知'}，跳过")
                                    break
                            if ok and compat and not any(any(c in ic or ic in c for c in compat) for ic in item_colors):
                                # None of the accessory colors match any compatible color — still ok if rules pass
                                logger.info("Lolibrary颜色检查: LLM推荐色未命中但不过滤, 配件色=%s, 推荐列表=%s", sorted(item_colors), compat)
                        # Phase 2: regex matrix check (fallback / double-check)
                        if ok and dress_colors and not _colors_compatible(dress_colors, item_colors):
                            ok = False
                            logger.info("Lolibrary颜色检查: 规则矩阵判定不兼容 -> 跳过")
                            _log(f"[Lolibrary/{part}] 颜色不协调: 裙子色调={sorted(dress_colors)}, 配件色调={sorted(item_colors) if item_colors else '未知'}，跳过")
                        if not ok:
                            continue
                        if item_colors:
                            primary = dress_llm_colors.get("primary_color", "") if dress_llm_colors else ""
                            logger.info("Lolibrary颜色检查: 通过, 裙子=%s, 配件=%s", primary or sorted(dress_colors or []), sorted(item_colors))
                            _log(f"[Lolibrary/{part}] 颜色协调通过: 裙子={primary or sorted(dress_colors or [])}, 配件={sorted(item_colors)}")

                    for image_url in img_candidates:
                        try:
                            local_path = self.download_image(image_url, os.path.join(output_dir, part), f"{part_candidate.item.item_id}_{part}")
                            _log(f"[Lolibrary/{part}] 缩略图下载成功: {os.path.basename(local_path)}")
                        except requests.RequestException as e:
                            _log(f"[Lolibrary/{part}] 缩略图下载失败: {e}")
                            continue
                        selected_assets.append(
                            CollectedAsset(
                                part=part,
                                item=part_candidate.item,
                                image_url=image_url,
                                local_path=local_path,
                                source_search_url=part_candidate.source_search_url,
                                prompt_hint=part_candidate.prompt_hint,
                            )
                        )
                        taken_parts.add(part)
                        if color_match_enabled and part == PART_DRESS:
                            logger.info("Lolibrary连衣裙采集成功, 开始提取颜色。item=%s", part_candidate.item.title[:80])
                            dress_colors = _enrich_item_colors(part_candidate.item, self.adapter, log_callback)
                            _log(f"[Lolibrary/颜色] 连衣裙色调(规则): {sorted(dress_colors) if dress_colors else '未识别'}")
                            dress_llm_colors = _analyze_dress_colors_via_llm(part_candidate.item, log_callback=log_callback)
                            if dress_llm_colors:
                                _log(f"[Lolibrary/颜色] LLM 主色={dress_llm_colors.get('primary_color')}, 推荐配件色={dress_llm_colors.get('compatible_colors')}")
                            else:
                                logger.info("Lolibrary连衣裙: LLM分析失败, 仅使用规则匹配")
                        break
                    if part in taken_parts:
                        _log(f"[Lolibrary/{part}] 采集完成!")
                        break
                    # If thumbnail download failed, try the detail page
                    _log(f"[Lolibrary/{part}] 缩略图均失败，尝试打开详情页: {part_candidate.item.item_url}")
                    try:
                        detail = self.adapter.fetch_item_detail(part_candidate.item.item_url)
                        _log(f"[Lolibrary/{part}] 详情页获取成功，找到 {len(detail.image_urls)} 张图片")
                    except requests.RequestException as e:
                        _log(f"[Lolibrary/{part}] 详情页获取失败: {e}")
                        continue
                    detail_imgs = build_image_download_candidates(detail.image_urls, detail.thumbnail_url or part_candidate.item.thumbnail_url)
                    for dimg_url in detail_imgs:
                        try:
                            local_path = self.download_image(dimg_url, os.path.join(output_dir, part), f"{detail.item_id}_{part}")
                            _log(f"[Lolibrary/{part}] 详情图下载成功: {os.path.basename(local_path)}")
                        except requests.RequestException as e:
                            _log(f"[Lolibrary/{part}] 详情图下载失败: {e}")
                            continue
                        selected_assets.append(
                            CollectedAsset(
                                part=part,
                                item=detail,
                                image_url=dimg_url,
                                local_path=local_path,
                                source_search_url=part_candidate.source_search_url,
                                prompt_hint=part_candidate.prompt_hint,
                            )
                        )
                        taken_parts.add(part)
                        if color_match_enabled and part == PART_DRESS:
                            logger.info("Lolibrary连衣裙(详情页)采集成功, 开始提取颜色。item=%s", detail.title[:80])
                            dress_colors = _enrich_item_colors(detail, self.adapter, log_callback)
                            _log(f"[Lolibrary/颜色] 连衣裙色调(详情页/规则): {sorted(dress_colors) if dress_colors else '未识别'}")
                            dress_llm_colors = _analyze_dress_colors_via_llm(detail, log_callback=log_callback)
                            if dress_llm_colors:
                                _log(f"[Lolibrary/颜色] LLM 主色={dress_llm_colors.get('primary_color')}, 推荐配件色={dress_llm_colors.get('compatible_colors')}")
                            else:
                                logger.info("Lolibrary连衣裙(详情页): LLM分析失败, 仅使用规则匹配")
                        break
                    if part in taken_parts:
                        _log(f"[Lolibrary/{part}] 详情页采集完成!")
                        break
                if part in taken_parts:
                    break
            if part not in taken_parts:
                _log(f"[Lolibrary/{part}] 所有搜索方式和候选项均失败，该部位缺失")

        missing_parts = [part for part in preferred_parts if part not in taken_parts]
        _log(f"[Lolibrary] 采集结束: 成功={list(taken_parts)}, 缺失={missing_parts}")
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
        theme: str = "",
        log_callback: Callable[[str], None] | None = None,
    ) -> CollectionBundle:
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        self.adapter = self.wear_adapter
        self.adapter.log_callback = log_callback
        self.adapter.proxy_url = self.proxy_url
        self.session = self.adapter.session
        preferred_parts = [part for part in (preferred_parts or list(SUPPORTED_PARTS)) if part in SUPPORTED_PARTS]
        os.makedirs(output_dir, exist_ok=True)
        theme_profile = get_theme_profile(theme)

        selected_assets: list[CollectedAsset] = []
        taken_parts: set[str] = set()
        _log(f"[WEAR] 开始采集 {len(preferred_parts)} 个部位: {', '.join(preferred_parts)}")

        # Color coordination state
        wear_dress_colors: frozenset[str] | None = None
        wear_dress_llm_colors: dict[str, object] | None = None
        wear_color_match_enabled = bool(self.enable_color_match and PART_DRESS in preferred_parts)
        if wear_color_match_enabled:
            logger.info("WEAR颜色协调: 已启用, preferred_parts=%s", preferred_parts)
            _log("[WEAR/颜色] 颜色协调已启用")

        for part in preferred_parts:
            _log(f"[WEAR/{part}] 开始搜索...")
            search_url = self.wear_adapter.get_part_search_url(part)
            coordinates = self.wear_adapter.search_coordinates(part=part, max_pages=max_pages)
            _log(f"[WEAR/{part}] 搜索返回 {len(coordinates)} 个穿搭坐标")
            if not coordinates:
                _log(f"[WEAR/{part}] 无穿搭坐标，跳过")
                continue

            candidates: list[tuple[int, object, dict]] = []
            for idx, coordinate in enumerate(coordinates):
                _log(f"[WEAR/{part}] [{idx+1}/{len(coordinates)}] 解析穿搭: {coordinate.get('title', coordinate.get('coordinate_url'))[:60]}")
                items = self.wear_adapter.fetch_coordinate_items(coordinate["coordinate_url"])
                for item in items:
                    inferred = infer_part_from_item(item.category_slug, item.title, item.tags)
                    if inferred != part:
                        continue
                    if not self.wear_adapter.matches_brand(item, brand_slug):
                        continue
                    score = score_item_for_theme(item, part, theme_profile)
                    candidates.append((score, item, coordinate))
                _log(f"[WEAR/{part}] 该穿搭中找到 {len(items)} 个单品，匹配 {len([c for c in candidates if c[2] is coordinate])} 个")

            if not candidates:
                _log(f"[WEAR/{part}] 无候选项，跳过")
                continue

            candidates.sort(key=lambda value: value[0], reverse=True)
            _log(f"[WEAR/{part}] 主题匹配 {len(candidates)} 个候选项 (最高分={candidates[0][0]})")
            random.shuffle(candidates)

            for idx, (score, item, coordinate) in enumerate(candidates):
                _log(f"[WEAR/{part}] [{idx+1}/{len(candidates)}] 尝试: {item.title} (评分={score})")

                # --- color coordination check for accessories ---
                if wear_color_match_enabled and (wear_dress_colors or wear_dress_llm_colors) and part != PART_DRESS:
                    item_colors = _extract_colors(item)
                    logger.info(
                        "WEAR颜色检查: part=%s, item=%s, 配件色=%s, 裙子色(规则)=%s, 裙子色(LLM)=%s",
                        part, item.title[:80], sorted(item_colors),
                        sorted(wear_dress_colors) if wear_dress_colors else [],
                        wear_dress_llm_colors.get("primary_color") if wear_dress_llm_colors else "无",
                    )
                    ok = True
                    if wear_dress_llm_colors and isinstance(wear_dress_llm_colors, dict):
                        avoid = wear_dress_llm_colors.get("avoid_colors") or []
                        for ic in item_colors:
                            if any(a in ic or ic in a for a in avoid):
                                ok = False
                                logger.info("WEAR颜色检查: LLM避免色命中, 配件色=%s, 避免列表=%s", ic, avoid)
                                _log(f"[WEAR/{part}] 颜色不协调(LLM): 避免色={avoid}, 配件色={sorted(item_colors) if item_colors else '未知'}，跳过")
                                break
                    if ok and wear_dress_colors and not _colors_compatible(wear_dress_colors, item_colors):
                        ok = False
                        logger.info("WEAR颜色检查: 规则矩阵判定不兼容 -> 跳过")
                        _log(f"[WEAR/{part}] 颜色不协调: 裙子色调={sorted(wear_dress_colors)}, 配件色调={sorted(item_colors) if item_colors else '未知'}，跳过")
                    if not ok:
                        continue
                    if item_colors:
                        primary = wear_dress_llm_colors.get("primary_color", "") if wear_dress_llm_colors else ""
                        logger.info("WEAR颜色检查: 通过, 裙子=%s, 配件=%s", primary or sorted(wear_dress_colors or []), sorted(item_colors))
                        _log(f"[WEAR/{part}] 颜色协调通过: 裙子={primary or sorted(wear_dress_colors or [])}, 配件={sorted(item_colors)}")

                image_url = select_primary_image(item.image_urls, item.thumbnail_url or coordinate.get("thumbnail_url", ""))
                if not image_url:
                    _log(f"[WEAR/{part}] 无可用图片，跳过")
                    continue
                try:
                    local_path = self.download_image(image_url, os.path.join(output_dir, part), f"{item.item_id}_{part}")
                    _log(f"[WEAR/{part}] 下载成功: {os.path.basename(local_path)}")
                except requests.RequestException as e:
                    _log(f"[WEAR/{part}] 下载失败: {e}")
                    continue
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
                if wear_color_match_enabled and part == PART_DRESS:
                    logger.info("WEAR连衣裙采集成功, 开始提取颜色。item=%s", item.title[:80])
                    wear_dress_colors = _extract_colors(item)
                    _log(f"[WEAR/颜色] 连衣裙色调(规则): {sorted(wear_dress_colors) if wear_dress_colors else '未识别'}")
                    wear_dress_llm_colors = _analyze_dress_colors_via_llm(item, log_callback=log_callback)
                    if wear_dress_llm_colors:
                        _log(f"[WEAR/颜色] LLM 主色={wear_dress_llm_colors.get('primary_color')}, 推荐配件色={wear_dress_llm_colors.get('compatible_colors')}")
                    else:
                        logger.info("WEAR连衣裙: LLM分析失败, 仅使用规则匹配")
                _log(f"[WEAR/{part}] 采集完成!")
                break
            if part not in taken_parts:
                _log(f"[WEAR/{part}] 所有候选项下载失败，该部位缺失")

        missing_parts = [part for part in preferred_parts if part not in taken_parts]
        _log(f"[WEAR] 采集结束: 成功={list(taken_parts)}, 缺失={missing_parts}")
        return CollectionBundle(
            site_name=self.wear_adapter.site_name,
            brand_slug=brand_slug,
            search_url=self.wear_adapter.build_search_url(brand_slug),
            output_dir=output_dir,
            assets=selected_assets,
            missing_parts=missing_parts,
        )

    def collect_mayla_bundle(
        self,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
        theme: str = "",
        log_callback: Callable[[str], None] | None = None,
    ) -> CollectionBundle:
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        preferred_parts = [part for part in (preferred_parts or list(SUPPORTED_PARTS)) if part in SUPPORTED_PARTS]
        os.makedirs(output_dir, exist_ok=True)
        theme_profile = get_theme_profile(theme)

        selected_assets: list[CollectedAsset] = []
        taken_parts: set[str] = set()
        _log(f"[MAYLA] 开始采集 {len(preferred_parts)} 个部位: {', '.join(preferred_parts)}")

        for part in preferred_parts:
            _log(f"[MAYLA/{part}] 启动浏览器搜索...")
            search_url = self.mayla_adapter.get_part_search_url(part)
            products = self.mayla_adapter.search_products(part=part, max_pages=max_pages)
            _log(f"[MAYLA/{part}] 搜索返回 {len(products)} 个商品")
            if not products:
                _log(f"[MAYLA/{part}] 无商品，跳过")
                continue

            candidates: list[tuple[int, object, dict]] = []
            for idx, product in enumerate(products):
                product_url = product.get("product_url", "")
                if not product_url:
                    continue
                _log(f"[MAYLA/{part}] [{idx+1}/{len(products)}] 打开详情页: {product.get('title', product_url)[:60]}")
                try:
                    item = self.mayla_adapter.fetch_product_detail(product_url)
                except Exception as e:
                    _log(f"[MAYLA/{part}] 详情页获取失败: {e}")
                    continue
                if not self.mayla_adapter.matches_brand(item, brand_slug):
                    continue
                score = score_item_for_theme(item, part, theme_profile)
                candidates.append((score, item, product))

            if not candidates:
                _log(f"[MAYLA/{part}] 无候选项，跳过")
                continue

            candidates.sort(key=lambda value: value[0], reverse=True)
            _log(f"[MAYLA/{part}] 主题匹配 {len(candidates)} 个候选项 (最高分={candidates[0][0]})")
            random.shuffle(candidates)

            for idx, (score, item, product) in enumerate(candidates):
                _log(f"[MAYLA/{part}] [{idx+1}/{len(candidates)}] 尝试: {item.title} (评分={score})")
                image_url = select_primary_image(item.image_urls, item.thumbnail_url or product.get("thumbnail_url", ""))
                if not image_url:
                    _log(f"[MAYLA/{part}] 无可用图片，跳过")
                    continue
                try:
                    local_path = self.download_image(image_url, os.path.join(output_dir, part), f"{item.item_id}_{part}")
                    _log(f"[MAYLA/{part}] 下载成功: {os.path.basename(local_path)}")
                except Exception as e:
                    _log(f"[MAYLA/{part}] 下载失败: {e}")
                    continue
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
                _log(f"[MAYLA/{part}] 采集完成!")
                break
            if part not in taken_parts:
                _log(f"[MAYLA/{part}] 所有候选项下载失败，该部位缺失")

        missing_parts = [part for part in preferred_parts if part not in taken_parts]
        _log(f"[MAYLA] 采集结束: 成功={list(taken_parts)}, 缺失={missing_parts}")
        return CollectionBundle(
            site_name=self.mayla_adapter.site_name,
            brand_slug=brand_slug,
            search_url=self.mayla_adapter.build_search_url(brand_slug),
            output_dir=output_dir,
            assets=selected_assets,
            missing_parts=missing_parts,
        )

    def collect_hybrid_bundle(
        self,
        brand_slug: str,
        output_dir: str,
        max_pages: int = 1,
        preferred_parts: list[str] | None = None,
        theme: str = "",
        log_callback: Callable[[str], None] | None = None,
    ) -> CollectionBundle:
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        preferred_parts = [part for part in (preferred_parts or list(SUPPORTED_PARTS)) if part in SUPPORTED_PARTS]
        os.makedirs(output_dir, exist_ok=True)
        theme_profile = get_theme_profile(theme)
        preferred_site_map = dict((theme_profile.preferred_site_map if theme_profile else {}) or {})

        dress_parts: list[str] = []
        wear_parts: list[str] = []
        for part in preferred_parts:
            default_site = "lolibrary" if part == PART_DRESS else "wear"
            site = preferred_site_map.get(part, default_site)
            if site == "lolibrary":
                dress_parts.append(part)
            else:
                wear_parts.append(part)

        _log(f"[Hybrid] 路由: Lolibrary={dress_parts}, WEAR={wear_parts}")

        bundles: list[CollectionBundle] = []
        if dress_parts:
            lolibrary_brand = brand_slug
            if not lolibrary_brand and theme_profile and theme_profile.dress_brands:
                lolibrary_brand = "angelic-pretty"
            _log(f"[Hybrid] 开始 Lolibrary 采集 {dress_parts}，品牌={lolibrary_brand}")
            bundles.append(
                self.collect_lolibrary_bundle(
                    brand_slug=lolibrary_brand,
                    output_dir=os.path.join(output_dir, "lolibrary"),
                    max_pages=max_pages,
                    preferred_parts=dress_parts,
                    theme=theme,
                    log_callback=log_callback,
                )
            )
        if wear_parts:
            _log(f"[Hybrid] 开始 WEAR 采集 {wear_parts}")
            bundles.append(
                self.collect_wear_bundle(
                    brand_slug=brand_slug,
                    output_dir=os.path.join(output_dir, "wear"),
                    max_pages=max_pages,
                    preferred_parts=wear_parts,
                    theme=theme,
                    log_callback=log_callback,
                )
            )
        merged_assets_preview: list[CollectedAsset] = []
        found_parts_preview: set[str] = set()
        for bundle in bundles:
            merged_assets_preview.extend(bundle.assets)
            found_parts_preview.update(asset.part for asset in bundle.assets)
        missing_after_primary = [part for part in preferred_parts if part not in found_parts_preview]
        _log(f"[Hybrid] 首轮采集后缺失: {missing_after_primary}")
        fallback_wear_parts = [part for part in missing_after_primary if part not in wear_parts]
        if fallback_wear_parts:
            _log(f"[Hybrid] 启动 WEAR 兜底采集 {fallback_wear_parts}")
            bundles.append(
                self.collect_wear_bundle(
                    brand_slug=brand_slug,
                    output_dir=os.path.join(output_dir, "wear-fallback"),
                    max_pages=max_pages,
                    preferred_parts=fallback_wear_parts,
                    theme=theme,
                    log_callback=log_callback,
                )
            )

        merged_assets: list[CollectedAsset] = []
        found_parts: set[str] = set()
        search_urls: list[str] = []
        site_names: list[str] = []
        for bundle in bundles:
            merged_assets.extend(bundle.assets)
            found_parts.update(asset.part for asset in bundle.assets)
            if bundle.search_url:
                search_urls.append(bundle.search_url)
            if bundle.site_name:
                site_names.append(bundle.site_name)
        missing_parts = [part for part in preferred_parts if part not in found_parts]
        _log(f"[Hybrid] 采集结束: 成功={list(found_parts)}, 缺失={missing_parts}")
        return CollectionBundle(
            site_name=" + ".join(site_names) or "Hybrid",
            brand_slug=brand_slug,
            search_url=" | ".join(search_urls),
            output_dir=output_dir,
            assets=merged_assets,
            missing_parts=missing_parts,
        )

    def download_image(self, image_url: str, target_dir: str, filename_stem: str) -> str:
        os.makedirs(target_dir, exist_ok=True)
        response = request_with_proxy_fallback(
            self.session, "GET", image_url, timeout=self.timeout, proxy_url=self.proxy_url
        )
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
