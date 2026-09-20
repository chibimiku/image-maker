import json
from pathlib import Path
from unittest.mock import Mock

import modules.fashion_collection.generation_plan as generation_plan
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
from modules.others import api_backend
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


def _install_fake_hybrid_collectors(monkeypatch, tmp_path: Path):
    """装一套假的 Lolibrary/WEAR 采集器，按调用顺序记录参数，返回 (service, calls)。"""
    from modules.fashion_collection.collection_service import FashionCollectionService

    service = FashionCollectionService()
    calls: dict[str, list] = {"lolibrary": [], "wear": []}

    def fake_lolibrary_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme, log_callback=None):
        calls["lolibrary"].append(
            {
                "brand_slug": brand_slug,
                "preferred_parts": list(preferred_parts),
                "theme": theme,
                "log_callback": log_callback,
            }
        )
        item = CatalogItem("lolibrary", "dress-1", "https://lolibrary.org/items/1", "Sweet JSK", brand="Angelic Pretty")
        asset = CollectedAsset("dress", item, "https://img/dress.jpg", str(tmp_path / "dress.jpg"), "https://search", "dress prompt")
        return CollectionBundle("Lolibrary", brand_slug, "https://lolibrary.org/search?brands[]=angelic-pretty", output_dir, [asset], [])

    def fake_wear_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme, log_callback=None):
        calls["wear"].append(
            {
                "brand_slug": brand_slug,
                "preferred_parts": list(preferred_parts),
                "theme": theme,
                "log_callback": log_callback,
            }
        )
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
    return service, calls


def test_hybrid_collection_follows_theme_profile_site_map(monkeypatch, tmp_path: Path):
    """甜美洛丽塔画像把全部部位都指向 Lolibrary：不该再调 WEAR，品牌取画像默认。"""
    service, calls = _install_fake_hybrid_collectors(monkeypatch, tmp_path)
    logs = []

    bundle = service.collect_bundle(
        site_key="hybrid",
        brand_slug="",
        output_dir=str(tmp_path / "hybrid"),
        max_pages=1,
        preferred_parts=["dress", "shoes", "socks"],
        theme="甜美洛丽塔",
        log_callback=logs.append,
    )

    assert [call["preferred_parts"] for call in calls["lolibrary"]] == [["dress", "shoes", "socks"]]
    assert calls["lolibrary"][0]["brand_slug"] == "angelic-pretty"
    # 假 Lolibrary 只回了一件连衣裙，缺失的鞋袜由 WEAR 兜底补齐
    assert [call["preferred_parts"] for call in calls["wear"]] == [["shoes", "socks"]]
    assert calls["wear"][0]["brand_slug"] == ""
    assert any("[Hybrid] 路由" in line for line in logs)
    # 日志回调要透传给子采集（传下去的是能写进 logs 的回调，不是 None）
    calls["lolibrary"][0]["log_callback"]("[Hybrid] 测试探针")
    assert logs[-1] == "[Hybrid] 测试探针"
    assert len(bundle.assets) == 3
    assert bundle.missing_parts == []


def test_hybrid_collection_splits_parts_without_theme_profile(monkeypatch, tmp_path: Path):
    """没有命中画像时按默认规则拆分：dress → Lolibrary，其余 → WEAR。"""
    service, calls = _install_fake_hybrid_collectors(monkeypatch, tmp_path)
    logs = []

    bundle = service.collect_bundle(
        site_key="hybrid",
        brand_slug="",
        output_dir=str(tmp_path / "hybrid"),
        max_pages=1,
        preferred_parts=["dress", "shoes", "socks"],
        theme="不存在的主题",
        log_callback=logs.append,
    )

    assert [call["preferred_parts"] for call in calls["lolibrary"]] == [["dress"]]
    assert calls["lolibrary"][0]["brand_slug"] == ""                 # 无画像就不臆造品牌
    assert [call["preferred_parts"] for call in calls["wear"]] == [["shoes", "socks"]]
    calls["wear"][0]["log_callback"]("[Hybrid] 测试探针")
    assert logs[-1] == "[Hybrid] 测试探针"
    assert len(bundle.assets) == 3
    assert bundle.missing_parts == []


def test_hybrid_collection_falls_back_to_wear_for_missing_dress(monkeypatch, tmp_path: Path):
    from modules.fashion_collection.collection_service import FashionCollectionService

    service = FashionCollectionService()
    wear_calls = []

    def fake_lolibrary_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme, log_callback=None):
        return CollectionBundle("Lolibrary", brand_slug, "https://lolibrary.org/search?brands[]=angelic-pretty", output_dir, [], ["dress"])

    def fake_wear_bundle(*, brand_slug, output_dir, max_pages, preferred_parts, theme, log_callback=None):
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
    direct_response = Mock()
    session.request.side_effect = [proxy_error, direct_response]
    proxy_url = "http://proxy.example:7897"
    logs = []

    response = request_with_proxy_fallback(
        session, "GET", "https://example.com", timeout=8, proxy_url=proxy_url, log_callback=logs.append
    )

    assert response is direct_response
    assert session.request.call_count == 2
    first_call = session.request.call_args_list[0]
    second_call = session.request.call_args_list[1]
    assert first_call.kwargs["proxies"] == {"http": proxy_url, "https": proxy_url}
    assert "proxies" not in second_call.kwargs
    assert direct_response.raise_for_status.call_count == 1
    assert any("回退直连" in line for line in logs)


def test_request_with_proxy_fallback_skips_proxy_when_not_configured():
    session = Mock()
    direct_response = Mock()
    session.request.return_value = direct_response

    response = request_with_proxy_fallback(session, "GET", "https://example.com", timeout=8, proxy_url="")

    assert response is direct_response
    assert session.request.call_count == 1
    assert "proxies" not in session.request.call_args_list[0].kwargs


def test_wear_part_search_urls_cover_five_categories():
    assert PART_SEARCH_URLS["dress"].endswith("/women-category/onepiece/dress/")
    assert PART_SEARCH_URLS["shoes"].endswith("/women-category/shoes/sandal/")
    assert PART_SEARCH_URLS["socks"].endswith("/women-category/leg-wear/socks/")
    assert PART_SEARCH_URLS["hair_accessory"].endswith("/women-category/hair-accessory/")
    assert PART_SEARCH_URLS["bag"].endswith("/women-category/bag/handbag/")


def _two_character_bundle(tmp_path: Path) -> CollectionBundle:
    return CollectionBundle(
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


def test_build_scene_and_character_description_falls_back_to_static_pool(monkeypatch, tmp_path: Path):
    """LLM 不可用时走静态随机池：双人/单人落在对应模板池，场景带上穿搭部位。"""
    profile = get_theme_profile("甜美洛丽塔")
    bundle = _two_character_bundle(tmp_path)
    # 单测不联网：LLM 分支返回 None，落到确定性兜底池；随机选择固定取第一条
    monkeypatch.setattr(generation_plan, "_llm_generate_scene_character", lambda **_kwargs: None)
    monkeypatch.setattr(generation_plan.random, "choice", lambda seq: seq[0])

    scene_text, character_text = build_scene_and_character_description(bundle, profile, character_count=2)
    single_scene_text, single_character_text = build_scene_and_character_description(bundle, profile, character_count=1)

    assert "场景设定" in scene_text
    assert "两位" in character_text                 # 命中 characters_2 池
    assert "包袋" in scene_text                     # part_summary 带上了包袋
    assert "一位" in single_character_text          # 单人命中的是 characters_1 池
    assert character_text != single_character_text


def test_build_scene_and_character_description_uses_llm_result(monkeypatch, tmp_path: Path):
    """LLM 可用时直接用它的结果，且把双人要求写进 system prompt。"""
    profile = get_theme_profile("甜美洛丽塔")
    bundle = _two_character_bundle(tmp_path)
    captured = {}
    monkeypatch.setattr(
        api_backend,
        "get_api_config",
        lambda **_kwargs: {"base_url": "https://example.invalid/v1", "api_key": "test-key"},
    )

    def fake_fetch_llm_json(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {"scene": "场景设定：LLM 生成的包袋陈列场景。", "character": "主角描述：两位少女同框互动。"},
            ensure_ascii=False,
        )

    monkeypatch.setattr(api_backend, "fetch_llm_json", fake_fetch_llm_json)

    scene_text, character_text = build_scene_and_character_description(bundle, profile, character_count=2)

    assert scene_text == "场景设定：LLM 生成的包袋陈列场景。"
    assert character_text == "主角描述：两位少女同框互动。"
    assert captured["base_url"] == "https://example.invalid/v1"
    assert captured["api_key"] == "test-key"
    assert "双人" in captured["system_prompt"]
    assert "连衣裙" in captured["user_content"]
