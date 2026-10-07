# -*- coding: utf-8 -*-
"""`tools/repair_image_paths.py` 的离线回归：把记录里指向已删文件的路径改指到在盘的那一份。

覆盖两层：
1) 审计台账 `selected_file` —— 用台账自己的 `selected_round` 自证，不依赖外部清单；
2) 发布队列 `publish_queue.image_path` —— 用去重清单的「已删 → 幸存」映射。

全部离线：只用临时目录，不碰真实 data / DB。
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools import repair_image_paths as repair


def _jpeg(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 64)
    return str(path)


def _write_ledger(directory, prefix, payload):
    path = Path(directory) / f"{prefix}-first-image-audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _make_db(path, rows):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE publish_queue (uuid TEXT PRIMARY KEY, image_path TEXT, image_md5 TEXT, "
        "json_path TEXT, metadata TEXT, pixiv_status TEXT, chichipui_status TEXT, "
        "created_at TEXT, updated_at TEXT)")
    for index, image_path in enumerate(rows):
        connection.execute(
            "INSERT INTO publish_queue (uuid, image_path, created_at, updated_at) "
            "VALUES (?, ?, '', '')", [f"uuid-{index:04d}", image_path])
    connection.commit()
    connection.close()
    return str(path)


class LedgerRepairTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _case(self, attempts=1, selected_round=1):
        """造一份台账：候选是 -aN.jpg，selected_file 指着已被删掉的假 .png。"""
        stem = "8d0cb278_020028-first-ed899a"
        candidates = [
            {"round": n, "file": _jpeg(self.tmp / f"{stem}-a{n}_0201{n:02d}-aaaaaa.jpg")}
            for n in range(1, attempts + 1)
        ]
        payload = {
            "max_attempts": attempts,
            "stopped_reason": "passed",
            "selected_file": str(self.tmp / f"{stem}.png"),     # 已删，不存在
            "selected_round": selected_round,
            "candidates": candidates,
        }
        ledger = _write_ledger(self.tmp, f"{stem}-audit", payload)
        return stem, ledger, candidates

    def test_selected_file_is_repointed_to_the_selected_round(self):
        _stem, ledger, candidates = self._case(attempts=2, selected_round=2)
        result = repair.repair_ledgers([str(self.tmp)], {}, apply=True)
        self.assertEqual(len(result["changed"]), 1)
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(payload["selected_file"], candidates[1]["file"])
        self.assertTrue(os.path.isfile(payload["selected_file"]))
        # 别的字段一个字都不许动
        self.assertEqual(payload["selected_round"], 2)
        self.assertEqual([row["round"] for row in payload["candidates"]], [1, 2])

    def test_dry_run_writes_nothing(self):
        _stem, ledger, _c = self._case()
        before = ledger.read_bytes()
        result = repair.repair_ledgers([str(self.tmp)], {}, apply=False)
        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(ledger.read_bytes(), before)

    def test_ledger_that_already_points_at_a_real_file_is_left_alone(self):
        stem = "bac7b4cc_012940-first-ef4a58"
        alive = _jpeg(self.tmp / f"{stem}.jpg")
        ledger = _write_ledger(self.tmp, f"{stem}-audit", {
            "selected_file": alive, "selected_round": 1,
            "candidates": [{"round": 1, "file": alive}],
        })
        before = ledger.read_bytes()
        result = repair.repair_ledgers([str(self.tmp)], {}, apply=True)
        self.assertEqual(result["changed"], [])
        self.assertEqual(len(result["intact"]), 1)
        self.assertEqual(ledger.read_bytes(), before)

    def test_ledger_is_skipped_when_no_target_file_exists(self):
        stem = "deadbeef_000000-first-aaaaaa"
        ledger = _write_ledger(self.tmp, f"{stem}-audit", {
            "selected_file": str(self.tmp / f"{stem}.png"),
            "selected_round": 1,
            "candidates": [{"round": 1, "file": str(self.tmp / "gone.jpg")}],
        })
        before = ledger.read_bytes()
        result = repair.repair_ledgers([str(self.tmp)], {}, apply=True)
        self.assertEqual(result["changed"], [])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(ledger.read_bytes(), before)

    def test_report_mapping_is_used_when_the_ledger_cannot_self_prove(self):
        stem = "feedface_000000-first-bbbbbb"
        survivor = _jpeg(self.tmp / f"{stem}-a1_010101-cccccc.jpg")
        old = str(self.tmp / f"{stem}.png")
        ledger = _write_ledger(self.tmp, f"{stem}-audit", {
            "selected_file": old, "selected_round": 1,
            "candidates": [{"round": 1, "file": str(self.tmp / "gone.jpg")}],
        })
        result = repair.repair_ledgers([str(self.tmp)], {repair._norm(old): survivor}, apply=True)
        self.assertEqual(len(result["changed"]), 1)
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(payload["selected_file"], survivor)


class PublishDbRepairTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rows_pointing_at_deleted_files_are_repointed(self):
        survivor = _jpeg(self.tmp / "data" / "x-a1_010101-aaaaaa.jpg")
        old = str(self.tmp / "data" / "x.png")          # 已删
        alive = _jpeg(self.tmp / "data" / "keep.png")   # 本来就对
        db = _make_db(self.tmp / "queue.db", [old, alive])
        result = repair.repair_publish_db(db, {repair._norm(old): survivor}, apply=True)
        self.assertEqual(len(result["changed"]), 1)
        self.assertEqual(len(result["intact"]), 1)
        connection = sqlite3.connect(db)
        stored = dict(connection.execute("SELECT uuid, image_path FROM publish_queue").fetchall())
        connection.close()
        self.assertEqual(stored["uuid-0000"], survivor)
        self.assertEqual(stored["uuid-0001"], alive)

    def test_dry_run_leaves_the_database_untouched(self):
        survivor = _jpeg(self.tmp / "data" / "x-a1_010101-aaaaaa.jpg")
        old = str(self.tmp / "data" / "x.png")
        db = _make_db(self.tmp / "queue.db", [old])
        result = repair.repair_publish_db(db, {repair._norm(old): survivor}, apply=False)
        self.assertEqual(len(result["changed"]), 1)
        connection = sqlite3.connect(db)
        stored = connection.execute("SELECT image_path FROM publish_queue").fetchone()[0]
        connection.close()
        self.assertEqual(stored, old)

    def test_unmapped_missing_row_is_reported_not_changed(self):
        db = _make_db(self.tmp / "queue.db", [str(self.tmp / "nowhere.png")])
        result = repair.repair_publish_db(db, {}, apply=True)
        self.assertEqual(result["changed"], [])
        self.assertEqual(len(result["skipped"]), 1)

    def test_missing_database_is_not_an_error(self):
        result = repair.repair_publish_db(str(self.tmp / "nope.db"), {}, apply=True)
        self.assertFalse(result["present"])
        self.assertEqual(result["changed"], [])

    def test_survivor_from_ledger_picks_the_selected_round(self):
        payload = {"selected_round": 2, "candidates": [
            {"round": 1, "file": "one.jpg"}, {"round": 2, "file": "two.jpg"}]}
        self.assertEqual(repair.survivor_from_ledger(payload), "two.jpg")
        self.assertEqual(repair.survivor_from_ledger({"selected_round": 9, "candidates": []}), "")


if __name__ == "__main__":
    unittest.main()
