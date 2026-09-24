"""全局测试守卫。

1. 任何用例都不许把改动留在用户真实的 conf/*.json 里。
2. 任何用例都不许把测试产出写进 data/<日期>/ 真实产出目录。

## 守卫一：conf/*.json

真实事故（2026-09-12）：`app.py` 的 `AppWindow` 构造与画风同步会调用
`save_text_config()`，它直接写 `conf/config.json` 的 `last_used_style`。
smoke 测试里的 `sync_selected_style("测试共享风格")` 会把一个并不存在的画风写进去，
于是用户下次启动 app 时读到无效画风、回落成「默认(无附加)」——
看起来就像「记住上次选择的画风预设的功能丢了」。
`conf/config.json`（style_ref_mode 等）、`conf/config-sd.json`、
`conf/config-collector.json` 同样会被用例改写。

这里的做法：每个用例结束后比对 conf/*.json 的字节内容，被改了就直接还原，
并在终端列出还原了哪些文件。测试想验证落盘行为时请用 tmp_path 自备配置文件。

## 守卫二：data/<日期>/ 真实产出目录

真实事故（2026-09-12 起）：`test_single_analyzer_thread_safety.py` 之类的用例直接调
`SingleAnalyzerWidget.on_process_finished`，而该入口在「不保存到原图同目录」时会把结果
写进 `data/<YYYYMMDD>/`，于是假 key（`hash_0` / `task_00` / `conc_task_00` / `test_hash`）
跟着真实产出一起堆进日期目录：

    20260920-012607-hash_0-title_0.json
    20260920-012607-hash_0-title_0-prompts.txt
    20260920-012607-hash_0-title_0-original-prompts.txt

这些文件没有对应图片，只会干扰人工辨认和投稿 Server 的匹配。

这里的做法：整场测试打开 `utils.output_isolation` 的隔离开关，所有落在
`data/<日期>/` 的产出都会被改道到 `data/test-result/<日期>/` 并加 `test-` 前缀；
会话结束时再做一次兜底扫描，把漏网写进日期目录的测试产物搬过去。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def _snapshot_conf_files():
    """记录 conf/*.json 的内容指纹（mtime_ns, size, bytes）。"""
    snapshot = {}
    if not CONF_DIR.is_dir():
        return snapshot
    for path in sorted(CONF_DIR.glob("*.json")):
        try:
            stat = path.stat()
            snapshot[path] = (stat.st_mtime_ns, stat.st_size, path.read_bytes())
        except OSError:
            continue
    return snapshot


@pytest.fixture(autouse=True)
def protect_user_conf_files():
    """用例跑完后把 conf/*.json 还原成用例开始前的样子。"""
    snapshot = _snapshot_conf_files()
    yield

    restored = []
    for path, (mtime_ns, size, original) in snapshot.items():
        try:
            stat = path.stat()
            if stat.st_mtime_ns == mtime_ns and stat.st_size == size:
                continue
            if path.read_bytes() == original:
                continue
            path.write_bytes(original)
            restored.append(path.name)
        except OSError:
            # 文件被删/被占用：尽力还原，不因守卫本身让用例失败
            try:
                path.write_bytes(original)
                restored.append(path.name)
            except OSError:
                pass
    if restored:
        print(f"\n[tests/conftest.py] 已还原被用例改写的用户配置: {', '.join(sorted(restored))}")

    created = sorted(
        path.name for path in CONF_DIR.glob("*.json") if path not in snapshot
    )
    if created:
        print(f"[tests/conftest.py] 提示：用例在 conf/ 下新建了文件（未自动删除）: {', '.join(created)}")


@pytest.fixture(scope="session", autouse=True)
def isolate_test_output_artifacts():
    """整场测试打开产出隔离：data/<日期>/ -> data/test-result/<日期>/ + `test-` 前缀。

    `utils.output_isolation.resolve_output_target` 会读这个环境变量，
    因而分析 / 批量分析 / 无头分析链路都不会再往真实日期目录写东西。
    """
    from utils.output_isolation import TEST_OUTPUT_ENV, relocate_legacy_test_artifacts

    previous = os.environ.get(TEST_OUTPUT_ENV)
    os.environ[TEST_OUTPUT_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TEST_OUTPUT_ENV, None)
        else:
            os.environ[TEST_OUTPUT_ENV] = previous

        # 兜底：万一有用例绕过 resolve_output_target 直接写日期目录，也搬走
        moved = relocate_legacy_test_artifacts(REPO_ROOT / "data", apply=True)
        if moved:
            print(
                f"\n[tests/conftest.py] 兜底搬迁 {len(moved)} 个测试产出 -> "
                f"data/test-result/<日期>/（含 {moved[0][0].name} 等）"
            )

@pytest.fixture(autouse=True)
def _isolate_image_env(request, monkeypatch):
    """清掉来自 .env 的图片节点/密钥环境变量，保证用例与开发机配置无关。

    `api_backend` 在导入时会执行 `ensure_env_loaded()`，把仓库 `.env` 里的
    `IMAGE_MAKER_*` 写进 os.environ；测试里若断言"配置文件的 key 生效"或"缺 key 报错"，
    就会被开发机的真实密钥干扰（也会受 IMAGE_MAKER_NODES / IMAGE_MAKER_CURRENT_API 影响）。
    用例需要哪个变量，自己在用例里 setenv 即可。

    例外：`tests/test_env_loader.py` 专门验证「导入 api_backend 时把 .env 装进环境」，
    清掉环境后它必然失败 —— 这个模块本身就是对真实 .env 的断言，故跳过清洗。
    """
    if getattr(getattr(request, "module", None), "__name__", "") == "test_env_loader":
        yield
        return
    for name in list(os.environ):
        if name.startswith("IMAGE_MAKER_"):
            monkeypatch.delenv(name, raising=False)
    yield
