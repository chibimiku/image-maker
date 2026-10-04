"""权重抓取工具（本机联网走 Python requests）。

只做「按 URL 下载 + 记录字节数与 SHA-256」，不修改任何系统状态、不动驱动。
下载落到工作区（``cache/style-metrics-download`` 或 ``models/style-metrics``）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import requests

from .inventory import sha256_file

CHUNK = 1 << 20


def download(
    url: str,
    dest: str | Path,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    timeout: int = 120,
    log=print,
) -> dict:
    """流式下载到 ``dest``（先写 ``.part`` 再改名），返回来源/字节数/SHA-256。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    info: dict = {"url": url, "path": str(dest)}

    if dest.exists():
        size = dest.stat().st_size
        if expected_bytes and size == expected_bytes:
            digest = sha256_file(dest)
            if not expected_sha256 or digest == expected_sha256:
                info.update(status="cached", bytes=size, sha256=digest)
                log(f"[cached] {dest.name} {size} bytes")
                return info

    part = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        h = hashlib.sha256()
        written = 0
        with open(part, "wb") as fh:
            for block in resp.iter_content(CHUNK):
                if not block:
                    continue
                fh.write(block)
                h.update(block)
                written += len(block)
                if total and written % (128 * CHUNK) < CHUNK:
                    log(f"  {dest.name}: {written / 1e6:.0f}/{total / 1e6:.0f} MB")
    digest = h.hexdigest()
    if expected_bytes is not None and written != expected_bytes:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} 字节数不符：{written} != {expected_bytes}")
    if expected_sha256 and digest != expected_sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} SHA-256 不符：{digest} != {expected_sha256}")
    shutil.move(str(part), str(dest))
    info.update(
        status="downloaded",
        bytes=written,
        sha256=digest,
        seconds=round(time.time() - t0, 1),
    )
    log(f"[downloaded] {dest.name} {written} bytes sha256={digest}")
    return info


def main(argv: list[str] | None = None) -> int:
    """``python -m utils.style_metrics.fetch <url> <dest> [--sha256 X]``"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("用法: python -m utils.style_metrics.fetch <url> <dest> [--sha256 <hex>]")
        return 2
    url, dest = argv[0], argv[1]
    sha = None
    if "--sha256" in argv:
        sha = argv[argv.index("--sha256") + 1]
    info = download(url, dest, expected_sha256=sha)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
