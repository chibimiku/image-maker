# -*- coding: utf-8 -*-
"""极简 .env 装载：把仓库根目录的 `.env` 读进 os.environ。

## 为什么需要它

只靠 `setx` 设的用户环境变量有个坑：**任何在 setx 之前启动的进程（包括常驻的
DSH / 编辑器 / 已开着的终端）内存里的环境块是旧的**，它们 spawn 出来的子进程
同样拿不到新变量。于是「密钥只放环境变量」在那种父子链里会直接失效。

`.env` 文件由本模块在导入时读取，因此**和父进程环境无关**，怎么启动都能生效；
同时它被 `.gitignore` 忽略，不会进仓库。

## 文件格式（仓库根目录 `.env`）

    # 注释行
    IMAGE_MAKER_AIGC2D_API_KEY=sk-xxxx
    export IMAGE_MAKER_AUTODL_API_KEY="hOVi5Z..."     # export / 引号 / 行尾注释都支持

## 规则

- 默认**不覆盖**已经存在的环境变量（真实环境变量优先于 .env，和常见约定一致）
- 找不到文件就静默跳过；任何解析异常都只记日志，绝不让程序起不来
- `.env` 不入库：新增密钥只写这里 + 变量名写文档
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

ENV_FILENAME = ".env"
_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_QUOTED_RE = re.compile(r"""^(?P<q>['"])(?P<val>.*?)(?P=q)""")

_loaded = False
_last_report: dict = {}


def default_env_path() -> str:
    """仓库根目录的 .env（本文件位于 <repo>/utils/ 下）。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ENV_FILENAME)


def parse_env_text(text: str) -> dict:
    """把 .env 文本解析成 dict（不落盘、不改环境），供测试直接用。"""
    values: dict = {}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        # 引号值：取引号内内容，引号之后的（行尾注释等）丢掉
        quoted = _QUOTED_RE.match(value)
        if quoted:
            value = quoted.group("val")
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key] = value
    return values


def load_env_file(path: str = None, override: bool = False) -> dict:
    """装载 .env。返回 {变量名: 'set'|'kept'}，便于上层日志/自检。"""
    target = path or default_env_path()
    report: dict = {}
    if not os.path.exists(target):
        return report
    try:
        with open(target, "r", encoding="utf-8") as f:
            values = parse_env_text(f.read())
    except OSError as exc:
        logger.warning("读取 .env 失败（已忽略）: %s", exc)
        return report

    for key, value in values.items():
        if not value:
            continue
        if os.environ.get(key) and not override:
            report[key] = "kept"      # 已有真实环境变量，保留它
            continue
        os.environ[key] = value
        report[key] = "set"
    return report


def ensure_env_loaded(path: str = None) -> dict:
    """进程内只真正装载一次（幂等），供库入口调用。"""
    global _loaded, _last_report
    if _loaded and path is None:
        return _last_report
    report = load_env_file(path)
    if path is None:
        _loaded = True
        _last_report = report
        if report:
            logger.info("已从 .env 装载环境变量: %s", ", ".join(sorted(report)))
    return report
