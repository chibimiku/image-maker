from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThemeProfile:
    key: str
    title: str
    aliases: tuple[str, ...] = ()
    dress_brands: tuple[str, ...] = ()
    style_tags: tuple[str, ...] = ()
    color_tags: tuple[str, ...] = ()
    element_tags: tuple[str, ...] = ()
    dress_tags: tuple[str, ...] = ()
    shoes_tags: tuple[str, ...] = ()
    socks_tags: tuple[str, ...] = ()
    hair_accessory_tags: tuple[str, ...] = ()
    bag_tags: tuple[str, ...] = ()
    negative_tags: tuple[str, ...] = ()
    default_style_names: tuple[str, ...] = ()
    default_prompt: str = ""
    default_instructions: str = ""
    preferred_site_map: dict[str, str] = field(default_factory=dict)


SWEET_LOLITA_PROFILE = ThemeProfile(
    key="sweet-lolita",
    title="甜美洛丽塔",
    aliases=("甜美洛丽塔", "sweet-lolita", "sweet lolita", "甜lolita"),
    dress_brands=(
        "angelic pretty",
        "baby, the stars shine bright",
        "baby the stars shine bright",
        "metamorphose",
        "innocent world",
    ),
    style_tags=("sweet", "lolita", "romantic", "girly", "cute"),
    color_tags=("pink", "white", "ivory", "cream", "sax", "light blue", "rose"),
    element_tags=("lace", "frill", "ribbon", "bow", "floral", "heart", "tiered"),
    dress_tags=("jsk", "op", "jumperskirt", "one piece", "onepiece", "dress", "a-line"),
    shoes_tags=("tea party", "round toe", "platform", "strap", "ribbon", "bow"),
    socks_tags=("lace socks", "frill socks", "over knee", "knee socks", "otk", "crew socks"),
    hair_accessory_tags=("head bow", "headband", "hair clip", "barrette", "ribbon", "lace", "beret"),
    bag_tags=("heart bag", "handbag", "mini bag", "basket bag", "shoulder bag", "ribbon", "frill"),
    negative_tags=("black gothic", "punk", "sport", "sneaker", "street", "men"),
    default_style_names=("shiratamaco-style", "puracotte-style"),
    default_prompt="请生成一位甜美洛丽塔风格的少女，全身像，站姿自然，服装华丽精致，画面明亮梦幻，突出裙装、鞋子、袜子、发饰与包袋的配套感。",
    default_instructions="请严格参考输入的服饰图片完成一位甜美洛丽塔少女的穿搭组合，强调蕾丝、荷叶边、蝴蝶结与柔和配色，输出精致日系少女插画风格，人物完整，全身清晰，服装细节丰富，配饰与包袋需要与主裙协调。",
    preferred_site_map={"dress": "lolibrary", "shoes": "lolibrary", "socks": "lolibrary", "hair_accessory": "lolibrary", "bag": "lolibrary"},
)


THEME_PROFILES = {
    SWEET_LOLITA_PROFILE.key: SWEET_LOLITA_PROFILE,
}


def normalize_theme_key(theme: str) -> str:
    value = str(theme or "").strip().lower()
    if not value:
        return ""
    for profile in THEME_PROFILES.values():
        candidates = {profile.key.lower(), profile.title.lower(), *(alias.lower() for alias in profile.aliases)}
        if value in candidates:
            return profile.key
    return value


def get_theme_profile(theme: str) -> ThemeProfile | None:
    key = normalize_theme_key(theme)
    if not key:
        return None
    return THEME_PROFILES.get(key)
