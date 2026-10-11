"""`publish_server` 的 JSON→交付图 解析规则（2026-10-09 修复）回归。

覆盖历史上真实出现过的命名形态：
  · 生成流程的 JSON（key = 生成任务 hash）+ 输入源图在别处（旧规则会配错图）
  · 同 hash 下留着的打回候选 `-aN_…` / 改过名的 `rejected-…`
  · 旧产物把画风名放在最前面（`ajicoma-<hash>-…-final.jpg`）
  · 「保存到原图同目录」流程（源图就是投稿图）
  · 台账 / 队列两种权威来源
不联网、不写真库（每个用例把 DB_PATH 指到 tmp）。
"""
import json
import os
import sqlite3

import pytest

import publish_server as ps


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "DB_PATH", str(tmp_path / "queue.db"))
    ps.init_db()
    return tmp_path / "queue.db"


def _img(path, payload=b"image"):
    path.write_bytes(payload)
    return path


def _json(path, **fields):
    path.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")
    return path


# ── 核心 bug：生成流程的 source_image_path 是输入源图，不是交付图 ─────────────

def test_generation_json_resolves_to_deliverable_not_input_source(tmp_path):
    deliverable = _img(tmp_path / "35616ebb_233237-first-c1d0e4.jpg")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    source = _img(outside / "db8f99d1-1a25-4b70-a429-48edffca9362.png")
    meta = _json(tmp_path / "20261008-233237-35616ebb-月明かりのエルフたち.json",
                 source_image_path=str(source), japanese_title="月明かりのエルフたち")

    assert ps.find_image_for_json(str(meta)) == str(deliverable)


def test_external_source_is_never_returned_when_nothing_matches(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    source = _img(outside / "00032-390414688.png")
    meta = _json(tmp_path / "20260524-011630-c44f0228-玻璃苑の薔薇舞姫.json",
                 source_image_path=str(source))

    # 旧行为会把这张输入源图交出去；现在必须返回 None（宁可让用户重拖图片）
    assert ps.find_image_for_json(str(meta)) is None


# ── 打回候选 `-aN` / `rejected-` ────────────────────────────────────────────

def test_rejected_candidate_is_skipped_when_publish_name_exists(tmp_path):
    _img(tmp_path / "1529469f_232937-first-4c3e9b-a1_233047-0d58e5.jpg")
    deliverable = _img(tmp_path / "1529469f_232937-first-4c3e9b.jpg")
    meta = _json(tmp_path / "20261008-232937-1529469f-駅の光で微笑む人.json",
                 source_image_path="")

    assert ps.find_image_for_json(str(meta)) == str(deliverable)


def test_rejected_only_candidate_is_still_delivered(tmp_path):
    # 首图审计耗尽时就是这样交付的（20261006/07 的 19 个历史样本）
    only = _img(tmp_path / "465715da_234225-first-25b4fb-a2_234548-ee7538.jpg")
    meta = _json(tmp_path / "20261006-234225-465715da-青星の舞姫.json")

    assert ps.find_image_for_json(str(meta)) == str(only)


def test_renamed_rejected_candidate_is_deprioritized(tmp_path):
    _img(tmp_path / "rejected-eddcc1aa_011346-first-f1954f-a1_011346-aaaaaa.jpg")
    deliverable = _img(tmp_path / "eddcc1aa_011346-first-f1954f.jpg")
    meta = _json(tmp_path / "20261007-011312-eddcc1aa-深紅の玉座.json")

    assert ps.find_image_for_json(str(meta)) == str(deliverable)


# ── 台账 / 队列两种权威来源 ─────────────────────────────────────────────────

def test_audit_ledger_selected_file_wins_when_two_deliverables(tmp_path):
    first = _img(tmp_path / "f15f7dcd_011545-first-6d2dc5.jpg")
    _img(tmp_path / "f15f7dcd_011545-first-6d2dc5-1.jpg")
    (tmp_path / "f15f7dcd_011545-first-6d2dc5-audit-first-image-audit.json").write_text(
        json.dumps({"selected_file": str(first), "selected_round": 2}), encoding="utf-8")
    meta = _json(tmp_path / "20261007-011545-f15f7dcd-蒼衣の舞姫.json")

    assert ps.find_image_for_json(str(meta)) == str(first)


def test_ledger_pointing_at_a_different_key_is_ignored(tmp_path):
    odd = _img(tmp_path / "aaaa1111_011545-first-6d2dc5.jpg")
    deliverable = _img(tmp_path / "f15f7dcd_011545-first-6d2dc5.jpg")
    (tmp_path / "f15f7dcd_011545-first-6d2dc5-audit-first-image-audit.json").write_text(
        json.dumps({"selected_file": str(odd)}), encoding="utf-8")
    meta = _json(tmp_path / "20261007-011545-f15f7dcd-蒼衣の舞姫.json")

    assert ps.find_image_for_json(str(meta)) == str(deliverable)


def test_queued_pairing_is_reused_for_already_enqueued_json(tmp_path):
    a = _img(tmp_path / "8c5eca6c_002121-first-dfca6d-a1_002227-c54279.jpg")
    b = _img(tmp_path / "8c5eca6c_002121-first-dfca6d-a2_002304-261126.jpg")
    meta = _json(tmp_path / "20261007-002121-8c5eca6c-薔薇の夜会.json")
    conn = sqlite3.connect(ps.DB_PATH)
    conn.execute("INSERT INTO publish_queue (uuid, image_path, image_md5, json_path) VALUES (?,?,?,?)",
                 ["row-1", str(b), "md5", str(meta)])
    conn.commit()
    conn.close()

    assert ps.find_image_for_json(str(meta)) == str(b)
    assert ps.find_image_for_json(str(meta)) != str(a)


# ── 旧命名 / 旧流程兼容 ─────────────────────────────────────────────────────

def test_legacy_style_prefix_filename_is_matched_by_embedded_hash(tmp_path):
    legacy = _img(tmp_path / "ajicoma-960cddea-quality-refine_105842-1e7c96-final.jpg")
    meta = _json(tmp_path / "20260926-105642-960cddea-金飾舞姫.json",
                 source_image_path="elsewhere.jpg")

    assert ps.find_image_for_json(str(meta)) == str(legacy)


def test_save_to_source_flow_keeps_using_the_analyzed_image(tmp_path):
    # 「保存到原图同目录（不生成图片）」：JSON 名里的 key 就是源图名的第一段
    source = _img(tmp_path / "70803b3b_013508-aaaaaa.jpg")
    meta = _json(tmp_path / "20260603-013508-70803b3b-薔薇園薄光の姫姿.json",
                 source_image_path=str(source))

    assert ps.find_image_for_json(str(meta)) == str(source)


def test_source_in_another_dir_is_used_when_its_name_carries_the_key(tmp_path):
    other = tmp_path / "moved"
    other.mkdir()
    moved = _img(other / "35616ebb_233237-first-c1d0e4.jpg")
    meta = _json(tmp_path / "20261008-233237-35616ebb-月明かりのエルフたち.json",
                 source_image_path=str(moved))

    assert ps.find_image_for_json(str(meta)) == str(moved)


def test_gemini_legacy_key_prefers_the_exact_source(tmp_path):
    exact = _img(tmp_path / "Gemini_Generated_Image_9rk5jl9rk5jl9rk5.jpg")
    _img(tmp_path / "Gemini_Generated_Image_other.jpg")
    meta = _json(tmp_path / "20261004-015352-Gemini-桂冠の姫.json", source_image_path=str(exact))

    assert ps.find_image_for_json(str(meta)) == str(exact)


def test_two_runs_without_ledger_or_queue_are_ambiguous(tmp_path):
    _img(tmp_path / "4c93618b_181330-2ddd5d.jpg")
    _img(tmp_path / "4c93618b_205508-306127.jpg")
    meta = _json(tmp_path / "20260606-181302-4c93618b-白綺羅幻燈の侍女.json")

    assert ps.find_image_for_json(str(meta)) is None


# ── 入队入口：报错要准，绝不静默配错 ───────────────────────────────────────

def test_add_json_to_queue_enqueues_the_deliverable(tmp_path):
    deliverable = _img(tmp_path / "aa0e510f_012009-first-6a325a.jpg")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    source = _img(outside / "looks-the-same.png")
    meta = _json(tmp_path / "20261008-012009-aa0e510f-窓辺に座る少女.json",
                 source_image_path=str(source))

    result = ps.add_json_to_queue(str(meta))

    assert result["ok"], result
    conn = sqlite3.connect(ps.DB_PATH)
    row = conn.execute("SELECT image_path, json_path, pixiv_status FROM publish_queue").fetchone()
    conn.close()
    assert row[0] == str(deliverable)
    assert row[1] == str(meta)
    assert row[2] == "pending"


def test_add_json_to_queue_reports_ambiguous_instead_of_guessing(tmp_path):
    _img(tmp_path / "4c93618b_181330-2ddd5d.jpg")
    _img(tmp_path / "4c93618b_205508-306127.jpg")
    meta = _json(tmp_path / "20260606-181302-4c93618b-白綺羅幻燈の侍女.json")

    result = ps.add_json_to_queue(str(meta))

    assert not result["ok"]
    assert result["error"] == "ambiguous_image"
    conn = sqlite3.connect(ps.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM publish_queue").fetchone()[0] == 0
    conn.close()


def test_add_json_to_queue_refuses_a_mismatched_pairing(tmp_path, monkeypatch):
    """最后一道闸：万一定位函数给出与 JSON 对不上的图，也绝不入队。"""
    stray = _img(tmp_path / "ffffffff_011312-first-c1dc39.jpg")
    meta = _json(tmp_path / "20261008-011312-edcc86e9-深紅の玉座.json")
    monkeypatch.setattr(ps, "find_image_for_json", lambda _p: str(stray))

    result = ps.add_json_to_queue(str(meta))

    assert not result["ok"]
    assert result["error"] == "mismatched_pairing"
    conn = sqlite3.connect(ps.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM publish_queue").fetchone()[0] == 0
    conn.close()


def test_queued_pairing_may_keep_a_non_key_image(tmp_path, monkeypatch):
    """队列里既有的历史配对（老命名）不该被新规则判成错配。"""
    legacy = _img(tmp_path / "Gemini_Generated_Image_9rk5jl9rk5jl9rk5.jpg")
    meta = _json(tmp_path / "20261004-015352-Gemini-桂冠の姫.json")
    conn = sqlite3.connect(ps.DB_PATH)
    conn.execute("INSERT INTO publish_queue (uuid, image_path, image_md5, json_path) VALUES (?,?,?,?)",
                 ["row-legacy", str(legacy), "md5", str(meta)])
    conn.commit()
    conn.close()

    result = ps.add_json_to_queue(str(meta))

    assert not result["ok"]              # 已在队列里
    assert "已在队列中" in result["message"]


# ── 反向查找（图片 → JSON）不许被这次改动带坏 ───────────────────────────────

def test_deliverable_image_still_finds_its_json(tmp_path):
    deliverable = _img(tmp_path / "35616ebb_233237-first-c1d0e4.jpg")
    _img(tmp_path / "35616ebb_233237-first-c1d0e4-a2_233445-91d0dc.jpg")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    source = _img(outside / "db8f99d1.png")
    meta = _json(tmp_path / "20261008-233237-35616ebb-月明かりのエルフたち.json",
                 source_image_path=str(source))

    assert ps.find_metadata_json(str(deliverable)) == str(meta)
