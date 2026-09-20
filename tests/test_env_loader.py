"""`.env` 装载（utils/env_loader.py）的回归用例。

存在意义：只靠 setx 设的用户环境变量有坑——**在 setx 之前启动的进程（常驻的
DSH / 编辑器 / 老终端）内存里的环境块是旧的**，它们 spawn 出来的子进程也拿不到。
`.env` 由库入口读取，与父进程环境无关，所以「不管谁怎么拉起进程都能拿到密钥」。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.env_loader import (  # noqa: E402
    default_env_path,
    ensure_env_loaded,
    load_env_file,
    parse_env_text,
)


def test_parse_env_text_handles_quotes_export_and_comments():
    text = """
# 注释行
IMAGE_MAKER_A_API_KEY=sk-plain
export IMAGE_MAKER_B_API_KEY="sk-double"
IMAGE_MAKER_C_KEY='sk-single'   # 行尾注释
IMAGE_MAKER_D_KEY=sk-tail   # 也要能剥
BAD LINE
EMPTY_VALUE=
"""
    assert parse_env_text(text) == {
        "IMAGE_MAKER_A_API_KEY": "sk-plain",
        "IMAGE_MAKER_B_API_KEY": "sk-double",
        "IMAGE_MAKER_C_KEY": "sk-single",
        "IMAGE_MAKER_D_KEY": "sk-tail",
        "EMPTY_VALUE": "",
    }


def test_load_env_file_does_not_override_real_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("IMAGE_MAKER_KEEP=from-dotenv\nIMAGE_MAKER_NEW=added\n", encoding="utf-8")
    monkeypatch.setenv("IMAGE_MAKER_KEEP", "from-real-env")

    report = load_env_file(str(env_file), override=False)

    assert os.environ["IMAGE_MAKER_KEEP"] == "from-real-env"   # 真实环境变量优先
    assert os.environ["IMAGE_MAKER_NEW"] == "added"
    assert report["IMAGE_MAKER_KEEP"] == "kept"
    assert report["IMAGE_MAKER_NEW"] == "set"


def test_load_env_file_override_true(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("IMAGE_MAKER_SWAP=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("IMAGE_MAKER_SWAP", "from-real-env")
    load_env_file(str(env_file), override=True)
    assert os.environ["IMAGE_MAKER_SWAP"] == "from-dotenv"


def test_load_env_file_missing_is_silent(tmp_path):
    assert load_env_file(str(tmp_path / "nope.env")) == {}
    assert ensure_env_loaded(str(tmp_path / "nope.env")) == {}


def test_default_env_path_points_at_repo_root():
    path = Path(default_env_path())
    assert path.name == ".env"
    assert path.parent == REPO_ROOT


def test_env_file_is_gitignored():
    """`.env` 绝不能进仓库 —— 直接读 .gitignore 断言规则，不依赖 git 子进程
    （本机沙箱下 subprocess 开管道会被拒）。"""
    gitignore = REPO_ROOT / ".gitignore"
    rules = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    assert ".env" in rules, ".gitignore 必须忽略 .env"
    assert ".env.*" in rules, ".gitignore 应忽略 .env 的各种变体（如 .env.local）"
    assert "!.env.example" in rules, "模板 .env.example 必须能入库"

    untracked = REPO_ROOT / ".git" / "info" / "exclude"
    if untracked.exists():
        # 兜底：确认没有被误加进索引（用 DEVNULL，无管道）
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        assert tracked.returncode != 0, ".env 竟然已被 git 跟踪，必须 git rm --cached"


def test_api_backend_loads_env_file_on_import():
    """导入 api_backend 后，.env 里的变量应当在进程环境里（不依赖父进程）。"""
    from modules.others import api_backend  # noqa: F401

    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        pytest.skip("本机没有 .env")
    parsed = parse_env_text(env_file.read_text(encoding="utf-8"))
    for name in parsed:
        assert name in os.environ, f"{name} 未从 .env 装载进环境"
