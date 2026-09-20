"""文本/视觉分析 LLM 调用的失败重试。

对面服务不稳定时的常见失败（429 限流、5xx、超时、连接断开）不该让整个分析任务报废，
这里统一封装：可配置次数与间隔，等待期间可被取消，并把每次重试写进日志。

配置写在 conf/config.json（设置 →「文本分析 API」页可改）：

- ``text_retry_enabled``          默认 True，关掉则一次失败就直接报错
- ``text_retry_times``            默认 5，首次之外的重试次数
- ``text_retry_interval_seconds`` 默认 300（5 分钟），两次尝试之间的等待

只有「对面服务的问题」才重试：429 / 5xx / 超时 / 连接失败。
401/403/404、参数错误、审核拦截这类重试也没用的错误会立刻抛出。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "conf", "config.json")

DEFAULT_RETRY_ENABLED = True
DEFAULT_RETRY_TIMES = 5
DEFAULT_RETRY_INTERVAL_SECONDS = 300

# 等待期间每隔多久在日志里报一次剩余时间
RETRY_REPORT_INTERVAL_SECONDS = 60.0
# 取消检查的粒度：等待被切成这么小的片，取消能及时生效
RETRY_WAIT_SLICE_SECONDS = 1.0

_RATE_LIMIT_HINTS = ("rate limit", "too many requests", "请求频率", "限流", "quota")
_TIMEOUT_HINTS = ("timed out", "timeout", "read timed out", "connection reset", "connection aborted")
_RETRYABLE_EXCEPTION_NAMES = {
    # openai SDK 里这两类没有 HTTP 状态码可判，但都属于典型可恢复的传输层故障
    "APITimeoutError",
    "APIConnectionError",
}


@dataclass
class RetrySettings:
    """一次重试策略。times 表示首次之外还能重试几次。"""

    enabled: bool = DEFAULT_RETRY_ENABLED
    times: int = DEFAULT_RETRY_TIMES
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS

    def normalized(self) -> "RetrySettings":
        try:
            times = int(self.times)
        except Exception:
            times = DEFAULT_RETRY_TIMES
        try:
            interval = float(self.interval_seconds)
        except Exception:
            interval = DEFAULT_RETRY_INTERVAL_SECONDS
        return RetrySettings(
            enabled=bool(self.enabled),
            times=max(0, min(times, 100)),
            interval_seconds=max(0.0, interval),
        )


def load_retry_settings(config_path: Optional[str] = None) -> RetrySettings:
    """读 conf/config.json 里的重试配置；文件缺失/损坏时回落默认值（不抛异常）。"""
    path = config_path or CONFIG_PATH
    data: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}
    return RetrySettings(
        enabled=bool(data.get("text_retry_enabled", DEFAULT_RETRY_ENABLED)),
        times=data.get("text_retry_times", DEFAULT_RETRY_TIMES),
        interval_seconds=data.get("text_retry_interval_seconds", DEFAULT_RETRY_INTERVAL_SECONDS),
    ).normalized()


def _status_code_of(exc: Exception):
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except Exception:
        return None


def classify_retryable_error(exc: Exception):
    """判断异常是否值得重试，返回 (是否可重试, 人话原因)。"""
    status = _status_code_of(exc)
    text = str(exc or "")
    lowered = text.lower()
    name = type(exc).__name__

    if status == 429 or re.search(r"\b429\b", text) or any(hint in lowered for hint in _RATE_LIMIT_HINTS):
        return True, "429 限流（对面服务请求频率超限）"
    if status is not None and 500 <= status < 600:
        return True, f"{status} 服务端错误"
    if status is not None and 400 <= status < 500:
        # 明确是客户端/配置类错误：重试无用，立刻失败
        return False, f"{status} 请求被拒绝（配置、权限或参数问题，重试无意义）"
    if name in ("APITimeoutError", "APIConnectionError") or isinstance(exc, (TimeoutError, ConnectionError)):
        return True, "请求超时或连接中断"
    if any(hint in lowered for hint in _TIMEOUT_HINTS):
        return True, "请求超时或连接中断"
    if name in _RETRYABLE_EXCEPTION_NAMES:
        return True, f"对面服务异常（{name}）"
    return False, f"不可重试的错误（{name}）"


def format_wait(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    if total < 60:
        return f"{total} 秒"
    minutes, secs = divmod(total, 60)
    return f"{minutes} 分 {secs} 秒" if secs else f"{minutes} 分钟"


def wait_with_cancel(
    seconds: float,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    sleep_func: Callable[[float], None] = time.sleep,
    report: Optional[Callable[[float], None]] = None,
    report_interval: float = RETRY_REPORT_INTERVAL_SECONDS,
    slice_seconds: float = RETRY_WAIT_SLICE_SECONDS,
) -> bool:
    """可取消的等待；返回 False 表示等待期间被取消。"""
    remaining = max(0.0, float(seconds))
    elapsed = 0.0
    next_report = float(report_interval) if report_interval else 0.0
    slice_seconds = max(0.01, float(slice_seconds))
    while remaining > 0:
        if cancel_check and cancel_check():
            return False
        step = min(slice_seconds, remaining)
        sleep_func(step)
        remaining -= step
        elapsed += step
        if report and next_report and elapsed >= next_report:
            next_report += float(report_interval)
            if remaining > 0:
                report(remaining)
    return True


def call_with_retry(
    call: Callable[[], Any],
    *,
    settings: Optional[RetrySettings] = None,
    step_label: str = "LLM 请求",
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> Any:
    """执行一次 LLM 调用，遇到可重试错误按配置等待后重试。

    - 全部尝试失败时抛出最后一次的原始异常（上层现有的错误处理不用改）。
    - 等待期间被取消时同样抛出原始异常，交由上层按「已取消」处理。
    """
    effective = (settings or load_retry_settings()).normalized()

    def _log(message: str) -> None:
        if log_callback:
            try:
                log_callback(message)
            except Exception:
                pass

    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            retryable, reason = classify_retryable_error(exc)
            if not effective.enabled or not retryable or attempt >= effective.times:
                if retryable and effective.enabled and effective.times > 0:
                    _log(f"{step_label} 已重试 {effective.times} 次仍失败：{reason}")
                elif retryable and not effective.enabled:
                    _log(f"{step_label} 命中可重试错误（{reason}），但「失败重试」已关闭，直接报错")
                raise

            attempt += 1
            wait_seconds = effective.interval_seconds
            _log(
                f"⏳ {step_label} 失败（{reason}），{format_wait(wait_seconds)}后进行"
                f"第 {attempt}/{effective.times} 次重试"
            )
            if wait_seconds <= 0:
                continue

            def _report(remaining: float, _attempt=attempt) -> None:
                _log(
                    f"⏳ 等待重试（第 {_attempt}/{effective.times} 次）：剩余 {format_wait(remaining)}"
                )

            if not wait_with_cancel(
                wait_seconds,
                cancel_check=cancel_check,
                sleep_func=sleep_func,
                report=_report,
            ):
                _log(f"{step_label} 等待重试期间被取消，放弃重试。")
                raise
