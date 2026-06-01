import json

from utils.web_probe_cli import (
    download_urls,
    extract_links,
    extract_next_data,
    extract_regex_matches,
    guess_cookie_file,
    load_cookies,
    load_cookies_from_netscape,
    parse_header_items,
    parse_cookie_string,
    query_json_path,
)


def test_parse_header_items_builds_dict():
    headers = parse_header_items(["Accept: application/json", "X-Test: 123"])
    assert headers == {"Accept": "application/json", "X-Test": "123"}


def test_extract_next_data_and_query_path():
    payload = {
        "props": {
            "pageProps": {
                "coordinateItems": [
                    {"name": "dress"},
                    {"name": "shoes"},
                ]
            }
        }
    }
    html = (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(payload, ensure_ascii=False)
        + "</script></body></html>"
    )
    data = extract_next_data(html)
    assert query_json_path(data, "props.pageProps.coordinateItems[1].name") == "shoes"


def test_extract_regex_matches_supports_group_unique_and_limit():
    text = "href='a' href='b' href='a'"
    matches = extract_regex_matches(text, r"href='([^']+)'", group=1, unique=True, limit=2)
    assert matches == ["a", "b"]


def test_extract_links_supports_absolute_and_filter():
    html = """
    <a href="/items/1">one</a>
    <img src="https://cdn.example.com/a.jpg">
    <a href="/items/2">two</a>
    """
    matches = extract_links(
        html,
        base_url="https://example.com/search",
        attr="both",
        contains="items",
        absolute=True,
    )
    assert matches == ["https://example.com/items/1", "https://example.com/items/2"]


def test_parse_cookie_string_builds_dict():
    cookies = parse_cookie_string("a=1; b=hello; csrftoken=xyz")
    assert cookies == {"a": "1", "b": "hello", "csrftoken": "xyz"}


def test_load_cookies_from_netscape_text():
    text = """
    # Netscape HTTP Cookie File
    .example.com\tTRUE\t/\tFALSE\t0\tsessionid\tabc123
    .example.com\tTRUE\t/\tFALSE\t0\tcsrftoken\txyz
    """
    cookies = load_cookies_from_netscape(text)
    assert cookies == {"sessionid": "abc123", "csrftoken": "xyz"}


def test_load_cookies_supports_json_file(tmp_path):
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text(
        json.dumps(
            [
                {"name": "sessionid", "value": "abc"},
                {"name": "csrftoken", "value": "xyz"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cookies = load_cookies(cookie_file=str(cookie_file))
    assert cookies == {"sessionid": "abc", "csrftoken": "xyz"}


def test_guess_cookie_file_matches_host_and_root_domain(tmp_path):
    cookie_dir = tmp_path / "cookies"
    cookie_dir.mkdir()
    root_file = cookie_dir / "wear.jp.json"
    root_file.write_text("{}", encoding="utf-8")
    matched = guess_cookie_file("https://www.wear.jp/abc", str(cookie_dir))
    assert matched == str(root_file)


def test_download_urls_saves_files(monkeypatch, tmp_path):
    class DummyResponse:
        def __init__(self, content, headers=None):
            self.content = content
            self.headers = headers or {"content-type": "image/png"}

        def raise_for_status(self):
            return None

    class DummySession:
        def get(self, url, timeout=None, headers=None, cookies=None):
            return DummyResponse(b"PNGDATA", {"content-type": "image/png"})

    import utils.web_probe_cli as mod

    monkeypatch.setattr(mod.requests, "Session", lambda: DummySession())
    saved = download_urls(["https://example.com/a"], str(tmp_path))
    assert len(saved) == 1
    assert saved[0].endswith(".png")
