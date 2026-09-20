"""自动化测试产出隔离（把测试产物挡在真实产出目录之外）。

## 背景

GUI 里的分析 / 批量分析流程在「不保存到原图同目录」时会落到
`data/<YYYYMMDD>/`。测试用例直接调 `StudyAnalyzerWidget.on_process_finished`
之类的入口，于是假数据（`hash_0` / `task_00` / `conc_task_00` / `test_hash`）
就跟着真实产出一起写进了日期目录：

    20260920-012607-hash_0-title_0.json
    20260920-012607-hash_0-title_0-prompts.txt
    20260920-012607-hash_0-title_0-original-prompts.txt

这些文件没有对应的图片，混在真实产出里既难辨认又会干扰投稿 Server 的匹配。

## 规则

1. 进程设置了环境变量 `IMAGE_MAKER_TEST_OUTPUT=1` 时，任何落在
   `data/<日期>/...` 的产出会被重定向到 `data/test-result/<日期>/...`，
   文件名统一加 `test-` 前缀（前缀可用 `IMAGE_MAKER_TEST_PREFIX` 覆盖）。
   未设置或为 `0/off/no/false` 时行为与改动前完全一致。
2. `tests/conftest.py` 在测试会话里自动打开 1，并在会话结束时兜底把漏网写进
   日期目录的历史产物挪到 `data/test-result/`。

## 命令行

    python -m utils.output_isolation            # 预演：只列会被搬走的文件
    python -m utils.output_isolation --apply    # 真的搬
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterable

TEST_OUTPUT_ENV = "IMAGE_MAKER_TEST_OUTPUT"
TEST_PREFIX_ENV = "IMAGE_MAKER_TEST_PREFIX"
DEFAULT_TEST_PREFIX = "test-"
TEST_RESULT_DIR = "test-result"

_FALSEY = {"", "0", "off", "no", "false", "none"}

DATE_DIR_RE = re.compile(r"^\d{8}$")

# 测试用例里造的假 key（见 tests/test_single_analyzer_thread_safety.py）
_TEST_KEY_PATTERN = (
    r"hash_\d+-title_\d+"
    r"|task_\d+-title_\d+"
    r"|conc_task_\d+-conc_title_\d+"
    r"|test_hash-test_title"
)

LEGACY_TEST_ARTIFACT_RE = re.compile(
    r"^\d{8}-\d{6}-(?:" + _TEST_KEY_PATTERN + r")"
    r"(?:-prompts\.txt|-original-prompts\.txt|\.json)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 环境开关
# ---------------------------------------------------------------------------

def test_output_enabled() -> bool:
    """当前进程是否处于「测试产出隔离」模式。"""
    return os.environ.get(TEST_OUTPUT_ENV, "").strip().lower() not in _FALSEY


def test_output_prefix() -> str:
    """测试产物文件名前缀，默认 `test-`。"""
    prefix = os.environ.get(TEST_PREFIX_ENV, "").strip()
    return prefix or DEFAULT_TEST_PREFIX


def is_test_context() -> bool:
    """是否应由 pytest 自动接管（未显式开启但确实在 pytest 里跑）。"""
    return test_output_enabled() or bool(os.environ.get("PYTEST_CURRENT_TEST"))


# ---------------------------------------------------------------------------
# 保存路径改写
# ---------------------------------------------------------------------------

def _split_relative(save_dir: str) -> list[str]:
    return os.path.normpath(str(save_dir)).replace("\\", "/").split("/")


def is_datestamped_output_dir(save_dir: str) -> bool:
    """`data/<YYYYMMDD>[...]` 形式的真实产出目录（相对路径）。"""
    parts = _split_relative(save_dir)
    if len(parts) < 2 or parts[0] != "data":
        return False
    if parts[1] == TEST_RESULT_DIR:
        return False
    return bool(DATE_DIR_RE.fullmatch(parts[1]))


def resolve_output_target(save_dir: str, base_filename: str) -> tuple[str, str]:
    """按需把 `(save_dir, base_filename)` 重定向到 test-result 并加前缀。

    只在开启测试模式、且目标是 `data/<日期>/...` 时改写；其余情况（`tmp_path`、
    保存到原图同目录等）原样返回，避免影响既有测试断言。
    """
    if not test_output_enabled():
        return save_dir, base_filename

    parts = _split_relative(save_dir)
    if not is_datestamped_output_dir(save_dir):
        return save_dir, base_filename

    redirected = os.path.join("data", TEST_RESULT_DIR, *parts[1:])
    prefix = test_output_prefix()
    if base_filename.startswith(prefix):
        return redirected, base_filename
    return redirected, f"{prefix}{base_filename}"


# ---------------------------------------------------------------------------
# 历史产物搬迁
# ---------------------------------------------------------------------------

def is_legacy_test_artifact(filename: str) -> bool:
    """文件名是否是历史上的测试产物（`…-hash_0-title_0.json` 一族）。"""
    return bool(LEGACY_TEST_ARTIFACT_RE.fullmatch(os.path.basename(filename)))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    idx = 1
    while True:
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _iter_date_dir_candidates(date_dir: Path) -> Iterable[Path]:
    """日期目录下第一层与第二层的文件（覆盖 `batch-result` 这类子目录）。"""
    for entry in sorted(date_dir.iterdir()):
        if entry.is_file():
            yield entry
        elif entry.is_dir() and not entry.name.startswith("."):
            for sub in sorted(entry.iterdir()):
                if sub.is_file():
                    yield sub


def find_legacy_test_artifacts(data_root: str | os.PathLike | None = None) -> list[Path]:
    """找出日期目录里残留的测试产物（不移动）。"""
    root = Path(data_root) if data_root is not None else repo_root() / "data"
    if not root.is_dir():
        return []
    found: list[Path] = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or not DATE_DIR_RE.fullmatch(date_dir.name):
            continue
        for path in _iter_date_dir_candidates(date_dir):
            if is_legacy_test_artifact(path.name):
                found.append(path)
    return found


def relocate_legacy_test_artifacts(
    data_root: str | os.PathLike | None = None,
    apply: bool = False,
    log=None,
) -> list[tuple[Path, Path]]:
    """把日期目录里的测试产物搬到 `data/test-result/<日期>/`。

    返回 `[(原路径, 目标路径), ...]`；`apply=False` 时只做预演。
    """
    root = Path(data_root) if data_root is not None else repo_root() / "data"
    dest_root = root / TEST_RESULT_DIR
    moves: list[tuple[Path, Path]] = []

    for source in find_legacy_test_artifacts(root):
        relative_parts = source.relative_to(root).parts
        dest = dest_root.joinpath(*relative_parts)
        dest = _unique_path(dest) if dest.exists() else dest
        if apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
        moves.append((source, dest))
        if log is not None:
            log(f"{'MOVED' if apply else 'DRY  '} {source} -> {dest}")

    return moves


def _main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    apply = "--apply" in args
    moves = relocate_legacy_test_artifacts(apply=apply, log=print)
    if not moves:
        print("没有需要搬迁的测试产物。")
        return 0
    verb = "已搬迁" if apply else "待搬迁（加 --apply 才会真的搬）"
    print(f"\n{verb} {len(moves)} 个文件 -> {TEST_RESULT_DIR}/<日期>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
