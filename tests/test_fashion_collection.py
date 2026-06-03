import json
from pathlib import Path
from unittest.mock import Mock

from modules.fashion_collection.collection_service import build_image_download_candidates, score_item_for_theme
from modules.fashion_collection.generation_plan import (
    build_reference_prompt,
    build_scene_and_character_description,
    resolve_style_bundle,
)
from modules.fashion_collection.lolibrary_adapter import LolibraryAdapter
from modules.fashion_collection.make_pic_bridge import build_make_pic_state, export_make_pic_state
from modules.fashion_collection.models import CatalogItem, CollectionBundle, CollectedAsset
from modules.fashion_collection.networking import request_with_proxy_fallback
from modules.fashion_collection.theme_profiles import get_theme_profile
from modules.fashion_collection.wear_adapter import PART_SEARCH_URLS, WearAdapter
import requests


def test_lolibrary_parse_search_html_extracts_cards():
    html = """
    <div class="card">
      <div class="card-body text-center">
        <p class="mb-0"><a href="https://lolibrary.org/items/ap-sample-jsk">Sample JSK</a></p>
        <p class="text-muted itemnum mb-0">123-ABC</p>
        <div><img src="https://lolibrary.global.ssl.fastly.net/images/sample.jpeg?width=300&amp;height=300"></div>
      </div>
      <ul class="list-group list-group-flush">
        <li><a href="https://lolibrary.org/brands/angelic-pretty" title="Angelic Pretty">Angelic Pretty</a></li>
        <li><a href="https://lolibrary.org/categories/jsk" title="JSK">JSK</a></li>
      </ul>
    </div>
    """
    items = LolibraryAdapter.parse_search_html(html)
    assert len(items) == 1
    item = items[0]
    assert item.item_id == "ap-sample-jsk"
    assert item.title == "Sample JSK"
    assert item.category_slug == "jsk"
    assert item.brand == "Angelic Pretty"


def test_build_make_pic_state_maps_parts_to_slots(tmp_path: Path):
    dress_item = CatalogItem(
        source_site="lolibrary",
        item_id="dress-1",
        item_url="https://lolibrary.org/items/dress-1",
        title="Dress One",
    )
    shoe_item = CatalogItem(
        source_site="lolibrary",
        item_id="shoe-1",
        item_url="https://lolibrary.org/items/shoe-1",
        title="Shoes One",
    )
    bundle = CollectionBundle(
        site_name="Lolibrary",
        brand_slug="angelic-pretty",
        search_url="https://lolibrary.org/search?brands[]=angelic-pretty",
        output_dir=str(tmp_path),
        assets=[
            CollectedAsset("dress", dress_item, "https://img/dress.jpg", str(tmp_path / "dress.jpg"), "https://search", "dress prompt"),
            CollectedAsset("shoes", shoe_item, "https://img/shoes.jpg", str(tmp_path / "shoes.jpg"), "https://search", "shoe prompt"),
            CollectedAsset(
                "hair_accessory",
                CatalogItem(source_site="wear", item_id="hair-1", item_url="https://wear.jp/item/3", title="Hair Bow"),
                "https://img/hair.jpg",
                str(tmp_path / "hair.jpg"),
                "https://search",
                "hair prompt",
            ),
            CollectedAsset(
                "bag",
                CatalogItem(source_site="wear", item_id="bag-1", item_url="https://wear.jp/item/4", title="Heart Bag"),
                "https://img/bag.jpg",
                str(tmp_path / "bag.jpg"),
                "https://search",
                "bag prompt",
            ),
        ],
    )
    state = build_make_pic_state(bundle, instructions="style", extra_prompt="extra", aspect_ratio="3:4")
    assert state["instruction"] == "style"
    assert state["extra_prompt"] == "extra"
    assert state["slots"]["衣服1"]["filepath"].endswith("dress.jpg")
    assert state["slots"]["鞋子"]["prompt"] == "shoe prompt"
    assert state["slots"]["发饰"]["prompt"] == "hair prompt"
    assert state["slots"]["手持物"]["prompt"] == "bag prompt"


def test_export_make_pic_state_writes_json(tmp_path: Path):
    item = CatalogItem(
        source_site="lolibrary",
        item_id="sock-1",
        item_url="https://lolibrary.org/items/sock-1",
        title="Socks One",
    )
    bundle = CollectionBundle(
        site_name="Lolibrary",
        brand_slug="angelic-pretty",
        search_url="https://lolibrary.org/search?brands[]=angelic-pretty",
        output_dir=str(tmp_path),
        assets=[
            CollectedAsset("socks", item, "https://img/socks.jpg", str(tmp_path / "socks.jpg"), "https://search", "sock prompt"),
        ],
    )
    state_path = export_make_pic_state(bundle, str(tmp_path / "cache" / "last_state.json"))
    payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
    assert payload["slots"]["袜子"]["prompt"] == "sock prompt"


def test_wear_parse_search_html_extracts_coordinate_tiles():
    html = """
    <script id="__NEXT_DATA__" type="application/json">{
      "props": {
        "pageProps": {
          "fallback": {
            "sample": {
              "content_tiles": [
                {
                  "content_tile_type": "coordinate_tile",
                  "coordinate_tile": {
                    "coordinate": {
                      "url": "/alice/123456/",
                      "title": "Dress coordinate",
                      "image": {
                        "url": "https://images.wear2.jp/coordinate/test.jpg",
                        "url_500": "https://images.wear2.jp/coordinate/test_500.jpg"
                      },
                      "member": {
                        "name": "Alice"
                      }
                    }
                  }
                }
              ]
            }
          }
        }
      }
    }</script>
    """
    results = WearAdapter.parse_search_html(html, part="dress")
    assert len(results) == 1
    assert results[0]["coordinate_url"] == "https://wear.jp/alice/123456/"
    assert results[0]["thumbnail_url"].endswith("test_500.jpg")


def test_wear_parse_coordinate_html_extracts_items():
    html = """
    <script id="__NEXT_DATA__" type="application/json">{
      "props": {
        "pageProps": {
          "coordinate": {
            "id": 123456,
            "tags": [{"name": "spring"}]
          },
          "coordinateItems": [
            {
              "id": 99,
              "name": "Sample Shoes",
              "image": {
                "url": "https://c.imgz.jp/test_shoes.jpg",
                "url_500": "https://c.imgz.jp/test_shoes_500.jpg",
                "url_215": "https://c.imgz.jp/test_shoes_215.jpg"
              },
              "item": {
                "id": 88,
                "name": "Sample Shoes",
                "search_condition_label": "BrandX / シューズ / サンダル",
                "brand": {"name": "BrandX", "key": "brandx"},
                "item_category": {"keyword": "shoes", "name": "シューズ"},
                "item_child_category": {"keyword": "sandal", "name": "サンダル"}
              }
            }
          ]
        }
      }
    }</script>
    """
    items = WearAdapter.parse_coordinate_html(html, "https://wear.jp/alice/123456/")
    assert len(items) == 1
    item = items[0]
    assert item.category_slug == "sandal"
    assert item.brand == "BrandX"
    assert item.image_urls[0].endswith("test_shoes_500.jpg")


def test_theme_profile_alias_lookup_for_sweet_lolita():
    profile = get_theme_profile("甜美洛丽塔")
    assert profile is not None
    assert profile.key == "sweet-lolita"


def test_score_item_for_theme_prefers_sweet_lolita_keywords():
    profile = get_theme_profile("甜美洛丽塔")
    themed_item = CatalogItem(
        source_site="lolibrary",
        item_id="1",
        item_url="https://lolibrary.org/items/1",
        title="Angelic Pretty Pink Ribbon Lace JSK",
        brand="Angelic Pretty",
        category_slug="jsk",
        notes="sweet lolita dress",
        tags=["lace", "ribbon", "pink"],
    )
    plain_item = CatalogItem(
        source_site="lolibrary",
        item_id="2",
        item_url="https://lolibrary.org/items/2",
        title="Black Sport Sneaker Dress",
        brand="Other",
        category_slug="dress",
        notes="street style",
        tags=["sport"],
    )
    assert score_item_for_theme(themed_item, "dress", profile) > score_item_for_theme(plain_item, "dress", profile)


def test_resolve_style_bundle_uses_theme_default_styles():
    profile = get_theme_profile("甜美洛丽塔")
    names, text = resolve_style_bundle(
        "",
        {"shiratamaco-style": "soft pastel style", "puracotte-style": "lace detail style"},
        theme_profile=profile,
    )
    assert names == ["shiratamaco-style", "puracotte-style"]
    assert "soft pastel style" in text
    assert "lace detail style" in text


def test_build_reference_prompt_includes_brand_and_parts(tmp_path: Path):
    item = CatalogItem(
        source_site="wear",
        item_id="dress-1",
        item_url="https://wear.jp/item/1",
        title="Sample Dress",
        brand="BrandX",
    )
    bundle = CollectionBundle(
        site_name="WEAR",
        brand_slug="",
        search_url="https://wear.jp/women-category/onepiece/dress/",
        output_dir=str(tmp_path),
        assets=[CollectedAsset("dress", item, "https://img/dress.jpg", str(tmp_path / "dress.jpg"), "https://search", "dress prompt")],
    )
    prompt = build_reference_prompt("甜美洛丽塔少女", bundle, scene_text="场景设定：花园午后。", character_text="主角描述：一位少女。")
    assert "Sample Dress" in prompt
    assert "BrandX" in prompt
    assert "dress prompt" in prompt
    assert "场景设定" in prompt
    assert "主角描述" in prompt


def test_hybrid_collection_splits_parts_by_theme(monkeypatch, tmp_path: Path):
    from modules.fashion_collection.collection_service import FashionCollectionService

    service = FashionCollectionService()
    calls = {"lolibrary": None, "wear": None}

    def fake_lolibrary_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme):
        calls["lolibrary"] = {
            "brand_slug": brand_slug,
            "preferred_parts": list(preferred_parts),
            "theme": theme,
        }
        item = CatalogItem("lolibrary", "dress-1", "https://lolibrary.org/items/1", "Sweet JSK", brand="Angelic Pretty")
        asset = CollectedAsset("dress", item, "https://img/dress.jpg", str(tmp_path / "dress.jpg"), "https://search", "dress prompt")
        return CollectionBundle("Lolibrary", brand_slug, "https://lolibrary.org/search?brands[]=angelic-pretty", output_dir, [asset], [])

    def fake_wear_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme):
        calls["wear"] = {
            "brand_slug": brand_slug,
            "preferred_parts": list(preferred_parts),
            "theme": theme,
        }
        shoe = CatalogItem("wear", "shoe-1", "https://wear.jp/item/1", "Tea Party Shoes", brand="BrandX")
        sock = CatalogItem("wear", "sock-1", "https://wear.jp/item/2", "Lace Socks", brand="BrandY")
        return CollectionBundle(
            "WEAR",
            brand_slug,
            "https://wear.jp/women-category/",
            output_dir,
            [
                CollectedAsset("shoes", shoe, "https://img/shoes.jpg", str(tmp_path / "shoes.jpg"), "https://search", "shoe prompt"),
                CollectedAsset("socks", sock, "https://img/socks.jpg", str(tmp_path / "socks.jpg"), "https://search", "sock prompt"),
            ],
            [],
        )

    monkeypatch.setattr(service, "collect_lolibrary_bundle", fake_lolibrary_bundle)
    monkeypatch.setattr(service, "collect_wear_bundle", fake_wear_bundle)

    bundle = service.collect_bundle(
        site_key="hybrid",
        brand_slug="",
        output_dir=str(tmp_path / "hybrid"),
        max_pages=1,
        preferred_parts=["dress", "shoes", "socks"],
        theme="甜美洛丽塔",
    )
    assert calls["lolibrary"]["preferred_parts"] == ["dress"]
    assert calls["lolibrary"]["brand_slug"] == "angelic-pretty"
    assert calls["wear"]["preferred_parts"] == ["shoes", "socks"]
    assert len(bundle.assets) == 3


def test_hybrid_collection_falls_back_to_wear_for_missing_dress(monkeypatch, tmp_path: Path):
    from modules.fashion_collection.collection_service import FashionCollectionService

    service = FashionCollectionService()
    wear_calls = []

    def fake_lolibrary_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme):
        return CollectionBundle("Lolibrary", brand_slug, "https://lolibrary.org/search?brands[]=angelic-pretty", output_dir, [], ["dress"])

    def fake_wear_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme):
        wear_calls.append(list(preferred_parts))
        assets = []
        if "dress" in preferred_parts:
            dress = CatalogItem("wear", "dress-1", "https://wear.jp/item/1", "Fallback Dress", brand="WearBrand")
            assets.append(CollectedAsset("dress", dress, "https://img/dress.jpg", str(tmp_path / "dress.jpg"), "https://search", "dress prompt"))
        return CollectionBundle("WEAR", brand_slug, "https://wear.jp/women-category/", output_dir, assets, [])

    monkeypatch.setattr(service, "collect_lolibrary_bundle", fake_lolibrary_bundle)
    monkeypatch.setattr(service, "collect_wear_bundle", fake_wear_bundle)

    bundle = service.collect_bundle(
        site_key="hybrid",
        brand_slug="",
        output_dir=str(tmp_path / "hybrid"),
        max_pages=1,
        preferred_parts=["dress"],
        theme="甜美洛丽塔",
    )
    assert wear_calls == [["dress"]]
    assert bundle.missing_parts == []
    assert bundle.assets[0].part == "dress"


def test_build_image_download_candidates_prefers_thumbnail_queries():
    urls = [
        "https://lolibrary.global.ssl.fastly.net/images/full.jpeg",
        "https://lolibrary.global.ssl.fastly.net/images/thumb500.jpeg?width=500&height=500&fit=bounds",
        "https://lolibrary.global.ssl.fastly.net/images/thumb300.jpeg?width=300&height=300&fit=bounds",
    ]
    ordered = build_image_download_candidates(urls)
    assert ordered[0].endswith("width=300&height=300&fit=bounds")
    assert ordered[1].endswith("width=500&height=500&fit=bounds")
    assert ordered[-1].endswith("full.jpeg")


def test_request_with_proxy_fallback_retries_direct_when_proxy_fails():
    session = Mock()
    proxy_error = requests.exceptions.ProxyError("proxy down")
    direct_response = object()
    session.request.side_effect = [proxy_error, direct_response]

    response = request_with_proxy_fallback(session, "GET", "https://example.com", timeout=8)

    assert response is direct_response
    assert session.request.call_count == 2
    first_call = session.request.call_args_list[0]
    second_call = session.request.call_args_list[1]
    assert first_call.kwargs["proxies"]["http"] == "http://127.0.0.1:7897"
    assert "proxies" not in second_call.kwargs


def test_wear_part_search_urls_cover_five_categories():
    assert PART_SEARCH_URLS["dress"].endswith("/women-category/onepiece/dress/")
    assert PART_SEARCH_URLS["shoes"].endswith("/women-category/shoes/sandal/")
    assert PART_SEARCH_URLS["socks"].endswith("/women-category/leg-wear/socks/")
    assert PART_SEARCH_URLS["hair_accessory"].endswith("/women-category/hair-accessory/")
    assert PART_SEARCH_URLS["bag"].endswith("/women-category/bag/handbag/")


def test_build_scene_and_character_description_supports_two_characters(tmp_path: Path):
    profile = get_theme_profile("甜美洛丽塔")
    bundle = CollectionBundle(
        site_name="Hybrid",
        brand_slug="",
        search_url="https://wear.jp/",
        output_dir=str(tmp_path),
        assets=[
            CollectedAsset(
                "dress",
                CatalogItem(source_site="lolibrary", item_id="1", item_url="https://lolibrary.org/items/1", title="Sweet JSK"),
                "https://img/dress.jpg",
                str(tmp_path / "dress.jpg"),
                "https://search",
                "dress prompt",
            ),
            CollectedAsset(
                "bag",
                CatalogItem(source_site="wear", item_id="2", item_url="https://wear.jp/item/2", title="Heart Bag"),
                "https://img/bag.jpg",
                str(tmp_path / "bag.jpg"),
                "https://search",
                "bag prompt",
            ),
        ],
    )
    scene_text, character_text = build_scene_and_character_description(bundle, profile, character_count=2)
    assert "场景设定" in scene_text
    assert "两位" in character_text
    assert "包袋" in scene_text
