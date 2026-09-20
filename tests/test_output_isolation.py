"""测试产出隔离：`utils/output_isolation.py` 的回归用例。

背景：分析入口在不「保存到原图同目录」时会把结果落到 `data/<YYYYMMDD>/`，
测试用例直接调 `on_process_finished` 就会把假数据（`hash_0` / `task_00` /
`conc_task_00` / `test_hash`）写进真实产出目录。这组用例锁住三件事：

  1. 关掉开关时行为与改动前完全一致（不能影响 tmp_path / 原图同目录等既有断言）；
  2. 打开开关后 `data/<日期>/` 的产出改道到 `data/test-result/<日期>/` 并加 `test-` 前缀；
  3. 历史遗留产物的识别只命中测试假 key，绝不误伤真实产出文件名。
"""

from __future__ import annotations

import os

import pytest

from utils.output_isolation import (
    TEST_OUTPUT_ENV,
    TEST_RESULT_DIR,
    is_legacy_test_artifact,
    relocate_legacy_test_artifacts,
    resolve_output_target,
)


@pytest.fixture
def test_mode(monkeypatch):
    monkeypatch.setenv(TEST_OUTPUT_ENV, "1")


# ---------------------------------------------------------------------------
# 1. 关掉开关：行为不变
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "0", "off", "no", "false"])
def test_disabled_leaves_paths_untouched(monkeypatch, value):
    monkeypatch.setenv(TEST_OUTPUT_ENV, value)
    save_dir, base = resolve_output_target("data/20260920", "20260920-012607-hash_0-title_0")
    assert os.path.normpath(save_dir) == os.path.normpath("data/20260920")
    assert base == "20260920-012607-hash_0-title_0"


# ---------------------------------------------------------------------------
# 2. 打开开关：改道 + 加前缀
# ---------------------------------------------------------------------------

def test_enabled_redirects_date_dir_and_prefixes(test_mode):
    save_dir, base = resolve_output_target("data/20260920", "20260920-012607-hash_0-title_0")
    assert os.path.normpath(save_dir) == os.path.normpath(f"data/{TEST_RESULT_DIR}/20260920")
    assert base == "test-20260920-012607-hash_0-title_0"


def test_enabled_keeps_subdir_under_test_result(test_mode):
    save_dir, _ = resolve_output_target("data/20260920/batch-result", "x-title")
    assert os.path.normpath(save_dir) == os.path.normpath(
        f"data/{TEST_RESULT_DIR}/20260920/batch-result"
    )


def test_enabled_does_not_touch_tmp_paths(test_mode, tmp_path):
    """tmp_path 或「保存到原图同目录」的目标本来就已经隔离，不该被改写。"""
    save_dir, base = resolve_output_target(str(tmp_path), "20260920-000000-abc-title")
    assert save_dir == str(tmp_path)
    assert base == "20260920-000000-abc-title"


def test_enabled_is_idempotent_inside_test_result(test_mode):
    """已经在 test-result 里的路径不能变成 test-result/test-result 或双重前缀。"""
    save_dir, base = resolve_output_target(
        f"data/{TEST_RESULT_DIR}/20260920", "test-20260920-000000-hash_0-title_0"
    )
    assert os.path.normpath(save_dir) == os.path.normpath(f"data/{TEST_RESULT_DIR}/20260920")
    assert base == "test-20260920-000000-hash_0-title_0"


# ---------------------------------------------------------------------------
# 3. 历史产物识别：只命中测试假 key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "20260920-012607-hash_0-title_0.json",
    "20260920-012607-hash_0-title_0-prompts.txt",
    "20260920-012607-hash_0-title_0-original-prompts.txt",
    "20260912-013914-task_04-title_4.json",
    "20260912-014049-conc_task_02-conc_title_2-original-prompts.txt",
    "20260920-004338-test_hash-test_title.json",
])
def test_legacy_pattern_matches_test_artifacts(name):
    assert is_legacy_test_artifact(name)


@pytest.mark.parametrize("name", [
    "20260920-011847-0fb30137-藤影水鏡の花嫁.json",
    "20260920-011847-0fb30137-藤影水鏡の花嫁-prompts.txt",
    "0fb30137_011941-9bc36e.png",
    "test-20260920-012607-hash_0-title_0.json",   # 已加前缀，不属于「遗留」
    "20260920-012607-hash_0-title_0.md",
    "hash_0-title_0.json",
])
def test_legacy_pattern_ignores_real_output(name):
    assert not is_legacy_test_artifact(name)


# ---------------------------------------------------------------------------
# 4. 搬迁：日期目录 -> test-result/<日期>/
# ---------------------------------------------------------------------------

def test_relocate_moves_only_test_artifacts(tmp_path):
    data_root = tmp_path / "data"
    date_dir = data_root / "20260920"
    date_dir.mkdir(parents=True)
    (date_dir / "20260920-012607-hash_0-title_0.json").write_text("{}", encoding="utf-8")
    (date_dir / "20260920-012607-hash_0-title_0-prompts.txt").write_text("x", encoding="utf-8")
    real_json = date_dir / "20260920-011847-0fb30137-藤影水鏡の花嫁.json"
    real_json.write_text("{}", encoding="utf-8")
    real_png = date_dir / "0fb30137_011941-9bc36e.png"
    real_png.write_bytes(b"png")

    moves = relocate_legacy_test_artifacts(data_root, apply=True)

    assert len(moves) == 2
    dest_dir = data_root / TEST_RESULT_DIR / "20260920"
    assert (dest_dir / "20260920-012607-hash_0-title_0.json").is_file()
    assert (dest_dir / "20260920-012607-hash_0-title_0-prompts.txt").is_file()
    # 真实产出原地不动
    assert real_json.is_file()
    assert real_png.is_file()


def test_relocate_dry_run_changes_nothing(tmp_path):
    data_root = tmp_path / "data"
    date_dir = data_root / "20260920"
    date_dir.mkdir(parents=True)
    artifact = date_dir / "20260920-012607-hash_0-title_0.json"
    artifact.write_text("{}", encoding="utf-8")

    moves = relocate_legacy_test_artifacts(data_root, apply=False)

    assert len(moves) == 1
    assert artifact.is_file()
    assert not (data_root / TEST_RESULT_DIR).exists()


def test_relocate_avoids_overwriting_existing_file(tmp_path):
    data_root = tmp_path / "data"
    date_dir = data_root / "20260920"
    dest_dir = data_root / TEST_RESULT_DIR / "20260920"
    date_dir.mkdir(parents=True)
    dest_dir.mkdir(parents=True)
    (date_dir / "20260920-012607-hash_0-title_0.json").write_text("new", encoding="utf-8")
    (dest_dir / "20260920-012607-hash_0-title_0.json").write_text("old", encoding="utf-8")

    relocate_legacy_test_artifacts(data_root, apply=True)

    assert (dest_dir / "20260920-012607-hash_0-title_0.json").read_text(encoding="utf-8") == "old"
    assert (dest_dir / "20260920-012607-hash_0-title_0_1.json").read_text(encoding="utf-8") == "new"
