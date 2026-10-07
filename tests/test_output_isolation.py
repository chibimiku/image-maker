"""测试产出隔离：`utils/output_isolation.py` 的回归用例。

背景：分析入口在不「保存到原图同目录」时会把结果落到 `data/<YYYYMMDD>/`，
测试用例直接调 `on_process_finished` 就会把假数据（`hash_0` / `task_00` /
`conc_task_00` / `test_hash` / `abc123`）写进真实产出目录。这组用例锁住四件事：

  1. 显式关掉开关时行为与改动前完全一致（不能影响 tmp_path / 原图同目录等既有断言）；
  2. 打开开关后 `data/<日期>/` 的产出改道到 `data/test-result/<日期>/` 并加 `test-` 前缀；
  3. **没有开关、只是在 pytest 里跑** 也必须隔离 —— 2026-10-07 事故就是
     `tests/conftest.py::_isolate_image_env` 把 `IMAGE_MAKER_TEST_OUTPUT` 一起清掉，
     用例体内开关读到「关」，`20261007-112423-abc123-白发少女.json` 直接落进真实日期目录；
  4. 历史遗留产物的识别只命中测试假 key，绝不误伤真实产出文件名。
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from utils.output_isolation import (
    TEST_OUTPUT_ENV,
    TEST_PREFIX_ENV,
    TEST_RESULT_DIR,
    is_legacy_test_artifact,
    is_test_context,
    relocate_legacy_test_artifacts,
    resolve_output_target,
)


@pytest.fixture
def test_mode(monkeypatch):
    monkeypatch.setenv(TEST_OUTPUT_ENV, "1")


TODAY = datetime.now().strftime("%Y%m%d")


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
# 2b. 2026-10-07 事故回归：开关被清掉时，pytest 会话本身必须兜住
# ---------------------------------------------------------------------------

def test_missing_switches_still_isolate_inside_pytest(monkeypatch, request):
    """用例体内**没有**任何隔离环境变量时也必须改道（事故的正面用例）。

    复现路径：`tests/conftest.py::_isolate_image_env` 曾经按 `IMAGE_MAKER_` 前缀把
    会话级开关一起 `delenv`，于是 `test_output_enabled()` 在用例体内恒为 False，
    `resolve_output_target()` 原样返回，产物直落 `data/<日期>/`。
    """
    assert os.environ.get("PYTEST_CURRENT_TEST"), "必须在 pytest 用例体内运行这条断言"
    monkeypatch.delenv(TEST_OUTPUT_ENV, raising=False)
    monkeypatch.delenv(TEST_PREFIX_ENV, raising=False)

    save_dir, base = resolve_output_target(f"data/{TODAY}", f"{TODAY}-112423-abc123-白发少女")
    assert os.path.normpath(save_dir) == os.path.normpath(f"data/{TEST_RESULT_DIR}/{TODAY}")
    assert base == f"test-{TODAY}-112423-abc123-白发少女"


def test_explicit_off_still_wins_over_pytest_autodetect(monkeypatch):
    """显式写 0/off 仍然优先，可以关掉（不能因为「在 pytest 里」就强行隔离）。"""
    assert os.environ.get("PYTEST_CURRENT_TEST")
    monkeypatch.setenv(TEST_OUTPUT_ENV, "0")
    assert is_test_context() is False
    save_dir, base = resolve_output_target(f"data/{TODAY}", "x-title")
    assert os.path.normpath(save_dir) == os.path.normpath(f"data/{TODAY}")
    assert base == "x-title"


def test_session_fixture_switch_survives_the_env_sanitiser():
    """会话级开关不能被子进程守卫吃掉（conftest 两处 fixture 的顺序契约）。"""
    assert os.environ.get(TEST_OUTPUT_ENV) == "1"
    assert is_test_context() is True


def test_api_backend_relative_subdir_is_isolated():
    """用例常传相对 `save_sub_dir`（tmp_repaint / gpt-image-2-test / probe），也要隔离。"""
    from modules.others import api_backend

    dir_path = api_backend.resolve_save_dir("tmp_repaint", TODAY)
    assert os.path.normpath(dir_path) == os.path.normpath(
        f"data/{TEST_RESULT_DIR}/{TODAY}/tmp_repaint"
    )
    # 根目录（不传子目录）同样改道，失败目录也走同一条路
    assert os.path.normpath(api_backend.resolve_save_dir(None, TODAY)) == os.path.normpath(
        f"data/{TEST_RESULT_DIR}/{TODAY}"
    )
    assert os.path.normpath(api_backend.resolve_save_dir("tmp_repaint", TODAY, fail_today=TODAY)) \
        == os.path.normpath(f"data/{TEST_RESULT_DIR}/{TODAY}/tmp_repaint")


def test_api_backend_absolute_subdir_is_left_alone():
    """绝对路径（受控实验的独立输出目录）不受隔离改写影响。"""
    from modules.others import api_backend

    assert api_backend.resolve_save_dir(r"D:\out\run-1", "20260920") == r"D:\out\run-1"


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
    # 2026-10-07 事故实盘文件名（用例造的假任务 hash：abc123）
    "20261007-112423-abc123-白发少女.json",
    "20261007-112423-abc123-白发少女-prompts.txt",
    "20261007-112423-abc123-白发少女-original-prompts.txt",
    "20261007-112423-abc123-白发少女-gpt-image-prompts.txt",
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
    # 真实 hash 是 8 位十六进制，6 位的 abc123 不可能是真实产物；反过来，
    # 形如 8 位十六进制的名字**不能**因为「像测试」就被认领
    "20261007-112423-0fb30137-白发少女.json",
    "20261007-112423-abc1234-白发少女.json",
    "20261007-112423-abc123-白发少女.png",
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


def test_relocate_sweeps_fake_hash_artifacts(tmp_path):
    """2026-10-07 事故里的 `abc123` 一族也要能被兜底搬迁收走，真实产物不许动。"""
    data_root = tmp_path / "data"
    date_dir = data_root / "20261007"
    date_dir.mkdir(parents=True)
    leaked = [date_dir / f"20261007-112423-abc123-白发少女{suffix}"
              for suffix in (".json", "-prompts.txt", "-original-prompts.txt")]
    for path in leaked:
        path.write_text("{}", encoding="utf-8")
    real = date_dir / "20261007-112423-0fb30137-白发少女.json"
    real.write_text("{}", encoding="utf-8")

    moves = relocate_legacy_test_artifacts(data_root, apply=True)

    assert len(moves) == 3
    dest_dir = data_root / TEST_RESULT_DIR / "20261007"
    assert all((dest_dir / path.name).is_file() for path in leaked)
    assert real.is_file()


# ---------------------------------------------------------------------------
# 5. 证据型认领：只被测试用过的子目录 / 清单里源图全是 pytest 临时文件
# ---------------------------------------------------------------------------

def test_relocate_sweeps_test_only_subdir(tmp_path):
    """`tmp_repaint/`（用例的 `save_sub_dir`）整个搬走。"""
    data_root = tmp_path / "data"
    date_dir = data_root / "20261007"
    junk = date_dir / "tmp_repaint"
    junk.mkdir(parents=True)
    (junk / "aigc2d_server_raw_112000_abcdef.txt").write_text("{}", encoding="utf-8")
    (junk / "aigc2d_server_response_112000_abcdef.json").write_text("{}", encoding="utf-8")
    keep = date_dir / "gpt_image_repaint"          # 生产链路用的目录，一个字节都不许动
    keep.mkdir()
    (keep / "repaint_1.jpg").write_bytes(b"jpg")

    moves = relocate_legacy_test_artifacts(data_root, apply=True)

    assert [source.name for source, _ in moves] == ["tmp_repaint"]
    assert not junk.exists()
    assert (data_root / TEST_RESULT_DIR / "20261007" / "tmp_repaint").is_dir()
    assert (keep / "repaint_1.jpg").is_file()


def test_relocate_sweeps_pipeline_run_whose_sources_are_pytest_temp(tmp_path):
    """`pipeline-steps/<run>/` 的清单源图全在 pytest 临时目录 → 测试残留。"""
    import json as _json

    data_root = tmp_path / "data"
    date_dir = data_root / "20261007"
    steps = date_dir / "pipeline-steps"
    test_run = steps / "112424-eba369"
    real_run = steps / "014936-a6391d"
    test_run.mkdir(parents=True)
    real_run.mkdir(parents=True)
    (test_run / "pipeline-manifest.json").write_text(_json.dumps({
        "items": [{"source": r"C:\Users\tester\AppData\Local\Temp\pytest-of-tester"
                             r"\pytest-4\test_x0\img.png", "steps": [], "final": ""}],
    }), encoding="utf-8")
    (test_run / "img-112424-eba369-tone.png").write_bytes(b"png")
    (real_run / "pipeline-manifest.json").write_text(_json.dumps({
        "items": [{"source": "data/20261007/0fb30137_011941-9bc36e.jpg", "steps": [], "final": ""}],
    }), encoding="utf-8")
    (real_run / "real-tone.png").write_bytes(b"png")

    moves = relocate_legacy_test_artifacts(data_root, apply=True)

    assert [source.name for source, _ in moves] == ["112424-eba369"]
    dest = data_root / TEST_RESULT_DIR / "20261007" / "pipeline-steps" / "112424-eba369"
    assert (dest / "img-112424-eba369-tone.png").is_file()
    assert (real_run / "pipeline-manifest.json").is_file()
    assert (real_run / "real-tone.png").is_file()


def test_pipeline_run_without_manifest_is_left_alone(tmp_path):
    """没有清单（读不出证据）就跑不了判据，宁可不搬。"""
    data_root = tmp_path / "data"
    run_dir = data_root / "20261007" / "pipeline-steps" / "mystery"
    run_dir.mkdir(parents=True)
    (run_dir / "something-tone.png").write_bytes(b"png")

    assert relocate_legacy_test_artifacts(data_root, apply=True) == []
    assert (run_dir / "something-tone.png").is_file()
