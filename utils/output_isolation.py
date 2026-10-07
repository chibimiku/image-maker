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

1. 进程处于**测试上下文**时（显式设了 `IMAGE_MAKER_TEST_OUTPUT=1`，或者进程里挂着 pytest 的
   `PYTEST_CURRENT_TEST`），任何落在 `data/<日期>/...` 的产出会被重定向到
   `data/test-result/<日期>/...`，文件名统一加 `test-` 前缀（前缀可用 `IMAGE_MAKER_TEST_PREFIX` 覆盖）。
   显式写 `IMAGE_MAKER_TEST_OUTPUT=0/off/no/false` 时行为与改动前完全一致（可用来关掉）。
2. `tests/conftest.py` 在测试会话里自动打开 1，并在会话结束时兜底把漏网写进
   日期目录的历史产物挪到 `data/test-result/`。

## 2026-10-07 事故：开关在用例体内被清掉，隔离整体失效

现象：`data/20261007/` 根目录堆了一串测试假数据

    20261007-112423-abc123-白发少女.json
    20261007-112423-abc123-白发少女-prompts.txt
    20261007-112423-abc123-白发少女-original-prompts.txt

根因链（已复现）：`tests/conftest.py` 的 **会话级** `isolate_test_output_artifacts`
把 `IMAGE_MAKER_TEST_OUTPUT=1` 写进 `os.environ` 没问题；但**函数级 autouse** 的
`_isolate_image_env` 为了屏蔽开发机 .env，会把所有 `IMAGE_MAKER_*` 变量
`monkeypatch.delenv` 掉 —— 里面就包括这个开关。于是**每个用例体内** `test_output_enabled()`
读到的是「关」（探针实测：`IMAGE_MAKER_TEST_OUTPUT = None`、`test_output_enabled() = False`），
`resolve_output_target()` 原样返回，测试产物全部直落真实日期目录；只有
`hash_0 / task_00 / test_hash` 这些**名字在名单里**的遗留产物被会话结束时的兜底搬迁救回来，
`abc123-白发少女` 这种名单外的就留下了。

两道修复（缺一不可）：

- `tests/conftest.py::_isolate_image_env` 不再清 `IMAGE_MAKER_TEST_OUTPUT` /
  `IMAGE_MAKER_TEST_PREFIX`（根因）；
- `resolve_output_target()` 改成看 `is_test_context()`：**没有显式开关**时，只要进程里挂着
  pytest 的 `PYTEST_CURRENT_TEST` 就自动隔离（兜底，任何 fixture 再乱清也不会漏）。
  显式写 `0/off/no/false` 仍然优先，可用来关掉。
- `is_legacy_test_artifact()` 增补 `abc123` 假 hash 一族，让兜底搬迁也能收走这一类。

## 命令行

    python -m utils.output_isolation            # 预演：只列会被搬走的文件
    python -m utils.output_isolation --apply    # 真的搬
"""

from __future__ import annotations

import json
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

_ANALYSIS_ARTIFACT_SUFFIX = (
    r"(?:-prompts\.txt|-original-prompts\.txt|-gpt-image-prompts\.txt"
    r"|-gpt-image-short-prompts\.txt|\.json)$"
)

LEGACY_TEST_ARTIFACT_RE = re.compile(
    r"^\d{8}-\d{6}-(?:" + _TEST_KEY_PATTERN + r")" + _ANALYSIS_ARTIFACT_SUFFIX,
    re.IGNORECASE,
)

# `abc123` 是 tests/test_analysis_channel.py / test_first_image_audit_state.py 等用例造的
# 假任务 hash（真实产物一律是 8 位十六进制 md5 前缀），所以按它整段认领不会误伤真实产物：
#     20261007-112423-abc123-白发少女.json
#     20261007-112423-abc123-白发少女-original-prompts.txt
_FAKE_TASK_HASHES = ("abc123",)
FAKE_HASH_ARTIFACT_RE = re.compile(
    r"^\d{8}-\d{6}-(?:" + "|".join(_FAKE_TASK_HASHES) + r")-[^\s/]*?"
    + _ANALYSIS_ARTIFACT_SUFFIX,
    re.IGNORECASE,
)

# 只被测试写过的日期目录子目录（生产链路用 `gpt_image_repaint` / `gpt-image-2`）：
# 出现在 `data/<日期>/` 里就是测试残留，整个子目录算测试产物。
TEST_ONLY_SUBDIRS = ("tmp_repaint",)

# 测试用例里造的图片名（`utils/post_process.run_pipeline` 的产物名 = 源图 stem + run-id + 工序串）。
# 真实产出是从素材名派生的（日期-时间-标题…），这些词只出现在测试里。
_TEST_IMAGE_STEMS = (
    "img", "img2", "img3", "in", "in2", "in3", "out", "src", "src2", "base", "base2",
    "patch", "sub", "sub2", "shot", "shot1", "shot2", "shot3", "multi", "tight",
    "maskless", "det", "shoe", "hand", "manual", "keep", "maskless2",
)
_TEST_STEM_ALT = "|".join(sorted(_TEST_IMAGE_STEMS, key=len, reverse=True))
# 中间产物后缀（sline50 / local-hair / tone / ink / crop / lineanchor / pp）与最终产物（-final-）
_PIPELINE_JUNK_SUFFIX = (
    r"-final-[^/]*\.(?:png|jpg|jpeg|webp)"
    r"|-sline\d+\.(?:png|jpg|jpeg|webp)"
    r"|-local-[^/]*\.(?:png|jpg|jpeg|webp)"
    r"|-(?:tone|ink|pp|lineanchor|crop)\.(?:png|jpg|jpeg|webp)"
)
PIPELINE_TEST_ARTIFACT_RE = re.compile(
    r"^(?:" + _TEST_STEM_ALT + r")(?:-\d{6}-[0-9a-f]{6})?(?:" + _PIPELINE_JUNK_SUFFIX + r")$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 环境开关
# ---------------------------------------------------------------------------

def _explicit_switch() -> bool | None:
    """环境里**显式**写的开关值；变量没出现时返回 None（交给 pytest 自动判定）。"""
    if TEST_OUTPUT_ENV not in os.environ:
        return None
    return os.environ[TEST_OUTPUT_ENV].strip().lower() not in _FALSEY


def test_output_enabled() -> bool:
    """当前进程是否处于「测试产出隔离」模式（只看显式环境变量）。"""
    return _explicit_switch() is True


def test_output_prefix() -> str:
    """测试产物文件名前缀，默认 `test-`。"""
    prefix = os.environ.get(TEST_PREFIX_ENV, "").strip()
    return prefix or DEFAULT_TEST_PREFIX


def is_test_context() -> bool:
    """是否应由 pytest / 隔离开关接管落盘路径。

    - 显式设了 `IMAGE_MAKER_TEST_OUTPUT`（含显式 `0/off/…`）→ 以它为准；
    - 没设但进程里挂着 `PYTEST_CURRENT_TEST`（pytest 每个用例都会设，子进程也会继承）
      → 判定为测试上下文，照常隔离。

    这条兜底是 2026-10-07 事故的修复之一：`tests/conftest.py::_isolate_image_env`
    曾经把 `IMAGE_MAKER_*` 一律 delenv，把会话级的开关在**用例体内**清成「关」，
    于是隔离整体失效、假数据直落 `data/<日期>/`（详见模块 docstring）。
    """
    switch = _explicit_switch()
    if switch is not None:
        return switch
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


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

    只在**测试上下文**（显式开关或 pytest 会话，见 `is_test_context`）且目标是
    `data/<日期>/...` 时改写；其余情况（`tmp_path`、保存到原图同目录、绝对路径）原样返回，
    避免影响既有测试断言。
    """
    if not is_test_context():
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
    """文件名是否是历史上的测试产物。

    - `…-hash_0-title_0.json` 一族（分析落盘）；
    - `…-abc123-<标题>.json` 一族（用例造的假任务 hash，真实 hash 是 8 位十六进制）；
    - 流水线产物：`img-233500-a35670-final-sline50.png` 这种「测试临时图名 + run-id + 工序串」
      （`run_pipeline` 不传 final_dir 时默认落到 `data/<日期>/`，pytest 里就留下了一堆废文件）。
    """
    name = os.path.basename(filename)
    return bool(LEGACY_TEST_ARTIFACT_RE.fullmatch(name)
                or FAKE_HASH_ARTIFACT_RE.fullmatch(name)
                or PIPELINE_TEST_ARTIFACT_RE.fullmatch(name))


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


def _mentions_pytest_temp(path_text: str) -> bool:
    """路径是不是 pytest 的临时目录（`…/pytest-of-<用户>/pytest-<N>/…`）。"""
    return "/pytest-of-" in str(path_text or "").replace("\\", "/").lower()


def _pipeline_run_dir_is_test_artifact(run_dir: Path) -> bool:
    """`pipeline-steps/<run>/` 是不是测试跑出来的：清单里**所有**源图都在 pytest 临时目录。

    实例（2026-10-07 的 `data/20261007/pipeline-steps/112424-eba369`）：

        "source": "C:\\\\Users\\\\…\\\\Temp\\\\dsh-ECEPOl\\\\pytest-of-tester\\\\pytest-4\\\\…\\\\img.png"

    真实运行的源图是 `data/<日期>/…` 里的成图，所以这个判据不会误伤真实产物。
    """
    manifest = run_dir / "pipeline-manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    sources = [str(item.get("source") or "") for item in (payload.get("items") or [])
               if isinstance(item, dict)]
    sources = [text for text in sources if text]
    return bool(sources) and all(_mentions_pytest_temp(text) for text in sources)


def _iter_date_dir_candidates(date_dir: Path) -> Iterable[Path]:
    """日期目录里**可判定的**测试产物（文件按名字，目录按内容证据）。

    - 第一层与第二层的文件（覆盖 `batch-result` 这类子目录）按文件名认领；
    - `tmp_repaint` 这种**只被测试用到**的子目录整个认领（生产链路用的是
      `gpt_image_repaint` / `gpt-image-2`，见 `tools/` 与 `api_backend`）；
    - `pipeline-steps/<run>/` 由 `pipeline-manifest.json` 的源图路径判定。
    """
    for entry in sorted(date_dir.iterdir()):
        if entry.is_file():
            if is_legacy_test_artifact(entry.name):
                yield entry
        elif entry.is_dir() and not entry.name.startswith("."):
            if entry.name in TEST_ONLY_SUBDIRS:
                yield entry
                continue
            for sub in sorted(entry.iterdir()):
                if sub.is_file():
                    if is_legacy_test_artifact(sub.name):
                        yield sub
                elif sub.is_dir() and _pipeline_run_dir_is_test_artifact(sub):
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
        found.extend(_iter_date_dir_candidates(date_dir))
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
