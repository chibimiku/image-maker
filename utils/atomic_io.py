"""容错落盘：Windows 上 `os.replace` 偶发 WinError 5 时重试。

背景（2026-09-28 实测）：本机 D 盘 `data/` 下连续 60 次 `tmp -> target` 替换会失败 1~2 次
（`%TEMP%` 下 0/60），错误是 `[WinError 5] 拒绝访问`，属于目标文件被扫描/观察进程短暂占用。
断点与 CLI 结果日志都用 tmp+replace 原子写，**一次瞬时失败就会让已经付过费的整条工序链标红退出**
（实测：GPT 链跑到「逐角色人体结构审计与修订」时因 `generation-checkpoint.json.tmp -> .json`
失败而中断，图其实已经出了）。所以这里统一封装成「重试 + 最后兜底直写」。

用法:
    from utils.atomic_io import write_json_atomic
    write_json_atomic(path, data, indent=2, ensure_ascii=False)
"""
from __future__ import annotations

import json
import os
import time

RETRY_DELAYS = (0.05, 0.15, 0.3, 0.6, 1.0)


def replace_with_retry(src: str, dst: str, delays=RETRY_DELAYS) -> bool:
    """把 src 替换到 dst；瞬时占用就退避重试。返回是否用了原子路径（False=兜底直写）。"""
    last_error = None
    for attempt, delay in enumerate((0.0,) + tuple(delays)):
        if delay:
            time.sleep(delay)
        try:
            os.replace(src, dst)
            return True
        except OSError as exc:                     # WinError 5 / 32：被短暂占用
            last_error = exc
    # 原子替换一直失败：至少把内容写进目标文件，别让工序因为落盘失败而报废
    try:
        with open(src, "rb") as stream_in, open(dst, "wb") as stream_out:
            stream_out.write(stream_in.read())
        try:
            os.remove(src)
        except OSError:
            pass
        return False
    except OSError:
        raise last_error


def write_json_atomic(path: str, data, delays=RETRY_DELAYS, **dump_kwargs) -> bool:
    """写 JSON 到 path（先写 .tmp 再替换，失败自动重试）。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    options = {"ensure_ascii": False, "indent": 2}
    options.update(dump_kwargs)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(data, stream, **options)
    return replace_with_retry(temp, path, delays)
