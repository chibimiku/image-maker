from __future__ import annotations

from typing import Callable

import requests

DEFAULT_HTTP_PROXY = "http://127.0.0.1:7897"
DEFAULT_HTTPS_PROXY = "http://127.0.0.1:7897"


def get_default_proxy_settings(proxy_url: str | None = None) -> dict[str, str] | None:
    """Return proxies dict or None if proxy is not configured/enabled."""
    if not proxy_url:
        return None
    url = str(proxy_url).strip()
    if not url:
        return None
    return {"http": url, "https": url}


def request_with_proxy_fallback(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int | float,
    proxy_url: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    **kwargs,
) -> requests.Response:
    """Request with optional proxy-first fallback-to-direct strategy.

    If proxy_url is provided, try via proxy first; on failure, retry direct.
    Logs proxy usage via log_callback when available.
    """
    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    request_kwargs = dict(kwargs)
    proxy_settings = get_default_proxy_settings(proxy_url)

    if proxy_settings:
        proxy_kwargs = dict(request_kwargs)
        proxy_kwargs["proxies"] = proxy_settings
        _log(f"[网络] 通过代理 {proxy_url} 请求: {method} {url}")
        try:
            resp = session.request(method=method, url=url, timeout=timeout, **proxy_kwargs)
            resp.raise_for_status()
            _log(f"[网络] 代理请求成功: {resp.status_code}")
            return resp
        except requests.RequestException as e:
            _log(f"[网络] 代理请求失败 ({e})，回退直连...")

    _log(f"[网络] 直连请求: {method} {url}")
    resp = session.request(method=method, url=url, timeout=timeout, **request_kwargs)
    resp.raise_for_status()
    _log(f"[网络] 直连请求成功: {resp.status_code}")
    return resp
