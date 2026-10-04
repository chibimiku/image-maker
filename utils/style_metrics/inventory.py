"""权重清单与 SHA-256 校验。

报告里出现的每一个权重文件都必须能追到官方来源 URL、实际字节数与 SHA-256；
校验失败要显式报错，不允许「大概是这份」含糊过去。

**容错约定**：权重缺失或哈希不符时抛 :class:`WeightsMissing`，它自带
「路径 + 官方 URL + 字节数 + SHA-256 + 可直接复制的下载命令」，
调用方（runner / CLI）据此报 ``status="unavailable"`` 并继续跑其它指标，
**不静默跳过、不换别的模型顶替、不把缺失当 0 分**。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable

from .config import WEIGHTS


class WeightsMissing(FileNotFoundError):
    """权重文件缺失或校验失败；消息本体即「怎么修」的说明。"""

    def __init__(self, key: str, reason: str = "文件不存在", path: str | Path | None = None):
        self.key = key
        entry = WEIGHTS.get(key, {})
        self.weight_path = Path(path or entry.get("path", f"<未登记:{key}>"))
        self.source_url = entry.get("source")
        self.expected_sha256 = entry.get("sha256")
        self.expected_bytes = entry.get("bytes")
        super().__init__(self.message(reason))

    def message(self, reason: str) -> str:
        lines = [
            f"缺少画风指标权重 [{self.key}]：{reason}",
            f"  期望路径 : {self.weight_path}",
            f"  官方来源 : {self.source_url or '<未登记>'}",
            f"  字节数   : {self.expected_bytes if self.expected_bytes is not None else '<未登记>'}",
            f"  SHA-256  : {self.expected_sha256 or '<未登记>'}",
            f"  修复命令 : {self.remedy()}",
        ]
        return "\n".join(lines)

    def remedy(self) -> str:
        """给一条可直接粘贴执行的下载命令（沿用项目「Python 自己出网」的约定）。"""
        py = sys.executable or "python"
        return (
            f'"{py}" -m utils.style_metrics.fetch "{self.source_url}" '
            f'"{self.weight_path}" --sha256 {self.expected_sha256}'
        )

    def as_dict(self) -> dict:
        return {
            "reason": "weights_missing",
            "key": self.key,
            "expected_path": str(self.weight_path),
            "source_url": self.source_url,
            "expected_bytes": self.expected_bytes,
            "expected_sha256": self.expected_sha256,
            "remedy": self.remedy(),
        }


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def require_weight(key: str, verify: bool = False) -> Path:
    """取权重路径；缺失直接抛 :class:`WeightsMissing`（带完整修复信息）。

    ``verify=True`` 时额外核对字节数与 SHA-256（大文件很慢，只在显式要求时开）。
    """
    entry = WEIGHTS.get(key)
    if entry is None:
        raise WeightsMissing(key, "未在 config.WEIGHTS 登记")
    path = Path(entry["path"])
    if not path.exists():
        raise WeightsMissing(key, "文件不存在")
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise WeightsMissing(
            key, f"字节数不符（实际 {path.stat().st_size}，期望 {expected_bytes}）"
        )
    if verify:
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise WeightsMissing(key, f"SHA-256 不符（实际 {digest}）")
    return path


def describe_weight(key: str, verify: bool = False) -> dict:
    entry = WEIGHTS[key]
    path = Path(entry["path"])
    info = {
        "key": key,
        "path": str(path),
        "exists": path.exists(),
        "expected_sha256": entry["sha256"],
        "expected_bytes": entry["bytes"],
        "source_url": entry["source"],
        "license": entry["license"],
        "origin": entry["origin"],
    }
    if path.exists():
        info["bytes"] = path.stat().st_size
        info["size_matches"] = info["bytes"] == entry["bytes"]
        if verify:
            info["sha256"] = sha256_file(path)
            info["sha256_matches"] = info["sha256"] == entry["sha256"]
    else:
        info["size_matches"] = False
        if verify:
            info["sha256"] = None
            info["sha256_matches"] = False
        # 缺失时直接附上修复命令，方便界面/日志原样展示
        info["remedy"] = WeightsMissing(key).remedy()
    return info


def list_inventory(keys: Iterable[str] | None = None, verify: bool = False) -> dict:
    keys = list(keys or WEIGHTS.keys())
    return {k: describe_weight(k, verify=verify) for k in keys}


#: 每个指标依赖哪些权重（缺失检查与报错文案共用一份映射）
METRIC_WEIGHTS: dict[str, tuple[str, ...]] = {
    "gram": ("vgg19",),
    "adain": ("vgg19",),
    "lpips": ("lpips_alex", "lpips_alex_trunk"),
    "csd": ("csd", "clip_vit_l14"),
}


def missing_for_metric(metric: str) -> list[WeightsMissing]:
    """返回该指标缺哪些权重（空列表 = 齐了）。只做存在性与字节数检查，不算哈希。"""
    problems: list[WeightsMissing] = []
    for key in METRIC_WEIGHTS.get(metric, ()):
        try:
            require_weight(key)
        except WeightsMissing as exc:
            problems.append(exc)
    return problems


def preflight(metrics: Iterable[str]) -> dict:
    """跑前预检：返回 ``{metric: [缺失项字典]}``，全部齐了就是空 dict。"""
    report: dict[str, list[dict]] = {}
    for metric in metrics:
        problems = missing_for_metric(metric)
        if problems:
            report[metric] = [p.as_dict() for p in problems]
    return report


def format_preflight(report: dict) -> str:
    """把预检结果拼成可读的多行说明。"""
    lines = []
    for metric, problems in report.items():
        for p in problems:
            lines.append(
                f"  [{metric}] 缺 {p['key']}：{p['expected_path']}\n"
                f"        来源 {p['source_url']}\n"
                f"        修复 {p['remedy']}"
            )
    return "\n".join(lines)


def assert_weights(keys: Iterable[str]) -> None:
    """缺失或哈希不符直接抛错（用于验证入口的前置检查）。"""
    problems = []
    for key in keys:
        try:
            require_weight(key, verify=True)
        except WeightsMissing as exc:
            problems.append(str(exc))
    if problems:
        raise WeightsMissing("<多个>", "权重校验失败:\n  " + "\n  ".join(problems))
