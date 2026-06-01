from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def fetch_text(
    target: str,
    timeout: int = 20,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    encoding: str = "",
    from_file: bool = False,
) -> str:
    if from_file:
        with open(target, "r", encoding=encoding or "utf-8") as f:
            return f.read()

    session = requests.Session()
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)
    response = session.get(
        target,
        timeout=max(1, int(timeout)),
        headers=merged_headers,
        cookies=cookies or None,
    )
    response.raise_for_status()
    if encoding:
        response.encoding = encoding
    return response.text


def download_urls(
    urls: list[str],
    output_dir: str,
    timeout: int = 20,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> list[str]:
    session = requests.Session()
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: list[str] = []
    for index, url in enumerate(urls):
        response = session.get(
            url,
            timeout=max(1, int(timeout)),
            headers=merged_headers,
            cookies=cookies or None,
        )
        response.raise_for_status()
        path = urlparse(url).path
        filename = os.path.basename(path) or f"download_{index + 1}"
        if "." not in filename:
            content_type = str(response.headers.get("content-type") or "").lower()
            if "png" in content_type:
                filename += ".png"
            elif "webp" in content_type:
                filename += ".webp"
            elif "jpeg" in content_type or "jpg" in content_type:
                filename += ".jpg"
        file_path = os.path.join(output_dir, filename)
        stem, ext = os.path.splitext(file_path)
        suffix = 1
        while os.path.exists(file_path):
            file_path = f"{stem}_{suffix}{ext}"
            suffix += 1
        with open(file_path, "wb") as f:
            f.write(response.content)
        saved_paths.append(file_path)
    return saved_paths


def parse_header_items(header_items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in header_items or []:
        if ":" not in str(item):
            raise ValueError(f"Header 格式不合法: {item}")
        key, value = str(item).split(":", 1)
        result[key.strip()] = value.strip()
    return result


def parse_cookie_string(cookie_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in str(cookie_text or "").split(";"):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        result[key] = value.strip()
    return result


def load_cookies_from_json(data: Any) -> dict[str, str]:
    if isinstance(data, dict):
        return {str(k).strip(): str(v) for k, v in data.items() if str(k).strip()}
    if isinstance(data, list):
        result: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            value = str(item.get("value") or "")
            if name:
                result[name] = value
        return result
    raise ValueError("JSON cookies 仅支持对象或数组格式")


def load_cookies_from_netscape(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in str(text or "").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) >= 7:
            name = parts[5].strip()
            value = parts[6].strip()
            if name:
                result[name] = value
    return result


def load_cookies(cookie_text: str = "", cookie_file: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if cookie_file:
        with open(cookie_file, "r", encoding="utf-8") as f:
            content = f.read()
        stripped = content.strip()
        loaded = {}
        if stripped.startswith("{") or stripped.startswith("["):
            loaded = load_cookies_from_json(json.loads(stripped))
        else:
            loaded = load_cookies_from_netscape(content)
            if not loaded:
                loaded = parse_cookie_string(content)
        result.update(loaded)
    if cookie_text:
        result.update(parse_cookie_string(cookie_text))
    return result


def _hostname_candidates(target: str) -> list[str]:
    host = (urlparse(target).hostname or "").lower().strip()
    if not host:
        return []
    parts = host.split(".")
    candidates = [host]
    if host.startswith("www.") and len(host) > 4:
        candidates.append(host[4:])
    if len(parts) >= 2:
        root = ".".join(parts[-2:])
        if root not in candidates:
            candidates.append(root)
    return candidates


def guess_cookie_file(target: str, cookie_dir: str) -> str:
    if not cookie_dir:
        return ""
    base_dir = os.path.abspath(cookie_dir)
    names = []
    for host in _hostname_candidates(target):
        names.extend(
            [
                f"{host}.json",
                f"{host}.txt",
                f"{host}.cookies",
                f"{host}.cookies.txt",
                host,
            ]
        )
    for name in names:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    return ""


def extract_next_data(text: str, script_id: str = "__NEXT_DATA__") -> Any:
    pattern = rf'<script[^>]+id="{re.escape(script_id)}"[^>]*>(.*?)</script>'
    match = re.search(pattern, text, re.I | re.S)
    if not match:
        raise ValueError(f"未找到 script id={script_id} 的 JSON 数据")
    return json.loads(match.group(1))


def query_json_path(data: Any, path: str) -> Any:
    if not path.strip():
        return data

    current = data
    token_pattern = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")
    for raw_token in path.split("."):
        token = raw_token.strip()
        if not token:
            continue
        matches = token_pattern.findall(token)
        if not matches:
            raise KeyError(f"无法解析路径片段: {token}")
        for key_part, index_part in matches:
            if key_part:
                if not isinstance(current, dict):
                    raise KeyError(f"当前节点不是对象，无法读取键: {key_part}")
                current = current[key_part]
            else:
                if not isinstance(current, list):
                    raise KeyError(f"当前节点不是数组，无法读取索引: {index_part}")
                current = current[int(index_part)]
    return current


def extract_regex_matches(
    text: str,
    pattern: str,
    group: int = 0,
    limit: int = 0,
    unique: bool = False,
    ignore_case: bool = False,
    multiline: bool = False,
) -> list[str]:
    flags = 0
    if ignore_case:
        flags |= re.I
    if multiline:
        flags |= re.S
    compiled = re.compile(pattern, flags)
    results: list[str] = []
    seen: set[str] = set()

    for match in compiled.finditer(text):
        value = match.group(int(group))
        if unique:
            if value in seen:
                continue
            seen.add(value)
        results.append(value)
        if limit > 0 and len(results) >= limit:
            break
    return results


def extract_links(
    text: str,
    base_url: str = "",
    attr: str = "both",
    contains: str = "",
    limit: int = 0,
    absolute: bool = False,
) -> list[str]:
    attr_pattern = {"href": "href", "src": "src"}.get(attr, "href|src")
    pattern = re.compile(rf"(?:{attr_pattern})=[\"']([^\"']+)[\"']", re.I)
    results: list[str] = []
    seen: set[str] = set()
    needle = contains.strip().lower()
    for value in pattern.findall(text):
        link = urljoin(base_url, value) if absolute and base_url else value
        if needle and needle not in link.lower():
            continue
        if link in seen:
            continue
        seen.add(link)
        results.append(link)
        if limit > 0 and len(results) >= limit:
            break
    return results


def write_output(value: Any, output_path: str = "", as_json: bool = False):
    if isinstance(value, (dict, list)) or as_json:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用网页抓取/提取 CLI")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("target", help="URL 或本地文件路径")
    common.add_argument("--from-file", action="store_true", help="将 target 视为本地文件路径")
    common.add_argument("--timeout", type=int, default=20, help="请求超时秒数")
    common.add_argument("--encoding", type=str, default="", help="强制响应编码，如 utf-8")
    common.add_argument("--header", action="append", default=[], help="附加请求头，格式 Key:Value")
    common.add_argument("--cookie", type=str, default="", help="直接传 Cookie 字符串，如 a=1; b=2")
    common.add_argument("--cookie-file", type=str, default="", help="Cookie 文件路径，支持 JSON / Netscape / 纯 cookie 字符串")
    common.add_argument("--cookie-dir-auto", type=str, default="", help="按目标域名自动寻找 Cookie 文件的目录")
    common.add_argument("--out", type=str, default="", help="输出文件路径")

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_fetch = subparsers.add_parser("fetch", parents=[common], help="抓取原始 HTML/文本")
    parser_fetch.add_argument("--print-chars", type=int, default=0, help="只打印前 N 个字符")

    parser_next = subparsers.add_parser("next-data", parents=[common], help="提取 Next.js __NEXT_DATA__ JSON")
    parser_next.add_argument("--script-id", type=str, default="__NEXT_DATA__", help="script 节点 id")
    parser_next.add_argument("--query", type=str, default="", help="JSON 路径，如 props.pageProps.coordinateItems[0]")

    parser_regex = subparsers.add_parser("regex", parents=[common], help="运行正则提取")
    parser_regex.add_argument("pattern", help="正则表达式")
    parser_regex.add_argument("--group", type=int, default=0, help="返回第几个捕获组")
    parser_regex.add_argument("--limit", type=int, default=0, help="最多返回多少条")
    parser_regex.add_argument("--unique", action="store_true", help="去重")
    parser_regex.add_argument("-i", "--ignore-case", action="store_true", help="忽略大小写")
    parser_regex.add_argument("--multiline", action="store_true", help="启用跨行匹配")

    parser_links = subparsers.add_parser("links", parents=[common], help="提取 href/src 链接")
    parser_links.add_argument("--attr", choices=["href", "src", "both"], default="both", help="提取哪种属性")
    parser_links.add_argument("--contains", type=str, default="", help="仅保留包含指定子串的链接")
    parser_links.add_argument("--limit", type=int, default=0, help="最多返回多少条")
    parser_links.add_argument("--absolute", action="store_true", help="将相对链接转换为绝对链接")

    parser_download = subparsers.add_parser("download", parents=[common], help="提取链接并直接下载")
    parser_download.add_argument("--attr", choices=["href", "src", "both"], default="src", help="从哪种属性提取链接")
    parser_download.add_argument("--contains", type=str, default="", help="仅下载包含指定子串的链接")
    parser_download.add_argument("--limit", type=int, default=0, help="最多下载多少条")
    parser_download.add_argument("--absolute", action="store_true", help="将相对链接转换为绝对链接")
    parser_download.add_argument("--download-dir", type=str, required=True, help="下载目录")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    headers = parse_header_items(args.header)
    cookie_file = args.cookie_file
    if not cookie_file and getattr(args, "cookie_dir_auto", "") and not bool(args.from_file):
        cookie_file = guess_cookie_file(args.target, args.cookie_dir_auto)
    cookies = load_cookies(args.cookie, cookie_file)
    text = fetch_text(
        args.target,
        timeout=args.timeout,
        headers=headers,
        cookies=cookies,
        encoding=args.encoding,
        from_file=bool(args.from_file),
    )

    if args.command == "fetch":
        output_text = text[: max(0, int(args.print_chars))] if args.print_chars > 0 else text
        write_output(output_text, output_path=args.out, as_json=False)
        return 0

    if args.command == "next-data":
        data = extract_next_data(text, script_id=args.script_id)
        value = query_json_path(data, args.query) if args.query else data
        write_output(value, output_path=args.out, as_json=True)
        return 0

    if args.command == "regex":
        matches = extract_regex_matches(
            text,
            args.pattern,
            group=args.group,
            limit=args.limit,
            unique=bool(args.unique),
            ignore_case=bool(args.ignore_case),
            multiline=bool(args.multiline),
        )
        write_output(matches, output_path=args.out, as_json=True)
        return 0

    if args.command == "links":
        base_url = "" if args.from_file else args.target
        matches = extract_links(
            text,
            base_url=base_url,
            attr=args.attr,
            contains=args.contains,
            limit=args.limit,
            absolute=bool(args.absolute),
        )
        write_output(matches, output_path=args.out, as_json=True)
        return 0

    if args.command == "download":
        base_url = "" if args.from_file else args.target
        matches = extract_links(
            text,
            base_url=base_url,
            attr=args.attr,
            contains=args.contains,
            limit=args.limit,
            absolute=bool(args.absolute),
        )
        saved_paths = download_urls(
            matches,
            output_dir=args.download_dir,
            timeout=args.timeout,
            headers=headers,
            cookies=cookies,
        )
        write_output(saved_paths, output_path=args.out, as_json=True)
        return 0

    parser.error(f"未知命令: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
