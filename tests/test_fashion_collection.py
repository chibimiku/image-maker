import json
from pathlib import Path

from modules.fashion_collection.lolibrary_adapter import LolibraryAdapter
from modules.fashion_collection.make_pic_bridge import build_make_pic_state, export_make_pic_state
from modules.fashion_collection.models import CatalogItem, CollectionBundle, CollectedAsset
from modules.fashion_collection.wear_adapter import WearAdapter


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
        ],
    )
    state = build_make_pic_state(bundle, instructions="style", extra_prompt="extra", aspect_ratio="3:4")
    assert state["instruction"] == "style"
    assert state["extra_prompt"] == "extra"
    assert state["slots"]["衣服1"]["filepath"].endswith("dress.jpg")
    assert state["slots"]["鞋子"]["prompt"] == "shoe prompt"


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
