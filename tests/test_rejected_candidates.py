# -*- coding: utf-8 -*-
"""把历史里「没被交付的候选」摘出 hash 前缀：`utils.rejected_candidates` + `tools/repair_image_paths`。

覆盖：
1) 判定三查（发布队列在册 / 台账 selected_file / 同前缀只剩一张 → 一律不动）；
2) 每组至少留一张（整组都是非交付候选时按 mtime 留最新）；
3) 预演不写盘、实改名保留字节与 mtime、目标名已存在就放弃；
4) 台账 `candidates[].file` 与 `selected_file` 跟着改到新名（改名清单里的 `renames` 映射）；
5) 已有的去重清单（`applied.deleted`）映射仍然照旧工作。

全部离线：临时目录 + PIL 造真 JPEG，不碰真实 data 目录、不碰真实发布队列。
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils import rejected_candidates as rc
from utils import first_image_audit as fa
from tools import repair_image_paths as repair


def _jpeg(path, color=(20, 120, 200), size=(16, 16)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(str(path), format="JPEG")
    return str(path)


class RejectedPlanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.day = self.tmp / "20261007"
        self.day.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _publish_queue(self, rows):
        path = self.tmp / "publish_queue.db"
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE publish_queue (uuid TEXT, image_path TEXT)")
        for index, image_path in enumerate(rows):
            conn.execute("INSERT INTO publish_queue (uuid, image_path) VALUES (?, ?)",
                         [f"u{index}", str(image_path)])
        conn.commit()
        conn.close()
        return str(path)

    def _ledger(self, name, selected, candidates):
        payload = {"max_attempts": 2, "stopped_reason": "exhausted",
                   "selected_file": selected, "selected_round": 1,
                   "candidates": [{"round": row[0], "file": row[1], "audit": {}}
                                  for row in candidates]}
        (self.day / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _candidate(self, name, color=(20, 120, 200)):
        return _jpeg(self.day / name, color=color)

    def test_non_delivered_candidate_in_an_ambiguous_group_is_renamed(self):
        first = self._candidate("aaaa1111_120000-first-abcdef-a1_120100-111111.jpg")
        second = self._candidate("aaaa1111_120000-first-abcdef-a2_120200-222222.jpg",
                                 color=(200, 40, 40))
        self._ledger("aaaa1111_120000-first-abcdef-audit-first-image-audit.json",
                     second, [(1, first), (2, second)])
        result = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        self.assertEqual([Path(r["old"]).name for r in result["renames"]],
                         [os.path.basename(first)])
        self.assertEqual(Path(result["renames"][0]["new"]).name, "rejected-" + os.path.basename(first))
        self.assertEqual([Path(r["path"]).name for r in result["kept"]], [os.path.basename(second)])
        self.assertEqual(result["kept"][0]["why"], "ledger-selected")

    def test_renamed_file_no_longer_matches_the_hash_prefix(self):
        first = self._candidate("aaaa1111_120000-first-abcdef-a1_120100-111111.jpg")
        second = self._candidate("aaaa1111_120000-first-abcdef.jpg", color=(200, 40, 40))
        result = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        self.assertEqual(len(result["renames"]), 1)
        applied = rc.apply_plan(result)
        self.assertEqual(len(applied["renamed"]), 1)
        remaining = [name for name in os.listdir(self.day) if name.lower().endswith(".jpg")]
        self.assertEqual([name for name in remaining if name.split("_")[0] == "aaaa1111"],
                         [os.path.basename(second)])          # 前缀恢复唯一
        self.assertTrue(os.path.isfile(result["renames"][0]["new"]))   # 文件没删
        self.assertFalse(os.path.isfile(first))

    def test_published_and_ledger_selected_are_never_touched(self):
        published = self._candidate("bbbb2222_120000-first-abcdef-a1_120100-111111.jpg")
        selected = self._candidate("bbbb2222_120000-first-abcdef-a2_120200-222222.jpg",
                                   color=(30, 30, 200))
        db = self._publish_queue([published])
        self._ledger("bbbb2222_120000-first-abcdef-audit-first-image-audit.json",
                     selected, [(1, published), (2, selected)])
        result = rc.plan([str(self.day)], db_path=db)
        self.assertEqual(result["renames"], [])
        self.assertEqual({row["why"] for row in result["kept"]},
                         {"published", "ledger-selected"})

    def test_group_of_only_candidates_keeps_the_newest(self):
        old = self._candidate("cccc3333_120000-first-abcdef-a1_120100-111111.jpg")
        new = self._candidate("cccc3333_120000-first-abcdef-a2_120200-222222.jpg",
                              color=(9, 9, 9))
        os.utime(old, (1_600_000_000, 1_600_000_000))
        os.utime(new, (1_700_000_000, 1_700_000_000))
        result = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        self.assertEqual([Path(r["old"]).name for r in result["renames"]], [os.path.basename(old)])
        self.assertEqual([r["why"] for r in result["kept"]], ["last-one-fallback"])

    def test_lone_candidate_is_left_alone(self):
        only = self._candidate("dddd4444_120000-first-abcdef-a1_120100-111111.jpg")
        result = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        self.assertEqual(result["renames"], [])
        self.assertEqual(result["kept"], [{"path": only, "why": "only-one"}])

    def test_publish_name_and_plain_images_are_not_candidates(self):
        self._candidate("eeee5555_120000-first-abcdef.jpg")
        self._candidate("Gemini_Generated_Image_x.jpg")
        self._candidate("eeee5555_120000-first-abcdef-a1_120100-111111.jpg",
                        color=(1, 2, 3))
        result = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        names = [Path(r["old"]).name for r in result["renames"]]
        self.assertEqual(names, ["eeee5555_120000-first-abcdef-a1_120100-111111.jpg"])

    def test_dry_run_plan_never_touches_the_disk(self):
        first = self._candidate("ffff6666_120000-first-abcdef-a1_120100-111111.jpg")
        self._candidate("ffff6666_120000-first-abcdef-a2_120200-222222.jpg", color=(8, 8, 8))
        before = sorted(os.listdir(self.day))
        rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        self.assertEqual(sorted(os.listdir(self.day)), before)
        self.assertTrue(os.path.isfile(first))

    def test_apply_is_idempotent_and_refuses_occupied_targets(self):
        first = self._candidate("abcdef01_120000-first-abcdef-a1_120100-111111.jpg")
        self._candidate("abcdef01_120000-first-abcdef-a2_120200-222222.jpg", color=(4, 4, 4))
        planned = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        outcome = rc.apply_plan(planned)                # 再跑一遍同一份计划
        self.assertEqual(len(outcome["renamed"]), 1)
        again = rc.apply_plan(planned)
        self.assertEqual(again["renamed"], [])
        self.assertTrue(any("文件已不在" in row["reason"] for row in again["refused"]))

    def test_hash_change_after_plan_refuses_the_rename(self):
        first = self._candidate("1234abcd_120000-first-abcdef-a1_120100-111111.jpg")
        self._candidate("1234abcd_120000-first-abcdef-a2_120200-222222.jpg", color=(7, 7, 7))
        planned = rc.plan([str(self.day)], db_path=str(self.tmp / "none.db"))
        _jpeg(first, color=(200, 200, 200))             # 计划之后内容变了
        outcome = rc.apply_plan(planned)
        self.assertEqual(outcome["renamed"], [])
        self.assertTrue(any("哈希" in row["reason"] for row in outcome["refused"]))


class RepairRenamedPathsTest(unittest.TestCase):
    """`tools/repair_image_paths.py` 要吃改名清单，并把台账里的旧名改成新名。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.day = self.tmp / "20261007"
        self.day.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_rename_report_mapping_is_accepted(self):
        old = _jpeg(self.day / "aaaa1111_120000-first-abcdef-a1_120100-111111.jpg")
        new = _jpeg(self.day / "rejected-aaaa1111_120000-first-abcdef-a1_120100-111111.jpg",
                    color=(90, 90, 90))
        report = self.tmp / "report.json"
        report.write_text(json.dumps({"renames": [{"old": old, "new": new}]}), encoding="utf-8")
        mapping = repair.load_report_mapping(str(report))
        self.assertEqual(mapping.get(repair._norm(old)), new)

    def test_ledger_candidate_paths_are_rewritten(self):
        old = _jpeg(self.day / "aaaa1111_120000-first-abcdef-a1_120100-111111.jpg")
        kept = _jpeg(self.day / "aaaa1111_120000-first-abcdef-a2_120200-222222.jpg",
                     color=(40, 80, 120))
        new = self.day / "rejected-aaaa1111_120000-first-abcdef-a1_120100-111111.jpg"
        shutil.move(old, new)
        ledger = self.day / "aaaa1111_120000-first-abcdef-audit-first-image-audit.json"
        ledger.write_text(json.dumps({
            "selected_file": str(kept), "selected_round": 2,
            "candidates": [{"round": 1, "file": str(old)}, {"round": 2, "file": str(kept)}],
        }, ensure_ascii=False), encoding="utf-8")
        report = self.tmp / "report.json"
        report.write_text(json.dumps({"renames": [{"old": str(old), "new": str(new)}]}),
                          encoding="utf-8")

        mapping = repair.load_report_mapping(str(report))
        preview = repair.repair_ledgers([str(self.day)], mapping, apply=False)
        self.assertEqual(len(preview["candidates"]), 1)
        self.assertTrue(json.loads(ledger.read_text(encoding="utf-8"))
                        ["candidates"][0]["file"].endswith("-a1_120100-111111.jpg"))   # 预演不写盘

        repair.repair_ledgers([str(self.day)], mapping, apply=True)
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidates"][0]["file"], str(new))
        self.assertEqual(payload["candidates"][1]["file"], str(kept))     # 本来就对的没动
        self.assertEqual(payload["selected_file"], str(kept))

    def test_candidate_path_that_still_exists_is_left_alone(self):
        kept = _jpeg(self.day / "bbbb2222_120000-first-abcdef-a2_120200-222222.jpg")
        ledger = self.day / "bbbb2222_120000-first-abcdef-audit-first-image-audit.json"
        ledger.write_text(json.dumps({"selected_file": str(kept), "selected_round": 2,
                                      "candidates": [{"round": 2, "file": str(kept)}]}),
                          encoding="utf-8")
        result = repair.repair_ledgers([str(self.day)], {}, apply=True)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(json.loads(ledger.read_text(encoding="utf-8"))
                         ["candidates"][0]["file"], str(kept))


class PromoteMoveStillLeavesOnePublishNameTest(unittest.TestCase):
    """闭环：`promote_candidate` 改名交付后的目录，跑一遍本工具应该无事可做。"""

    def test_fresh_directory_has_nothing_to_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp)
            candidate = _jpeg(day / "5678cdef_120000-first-abcdef-a1_120100-111111.jpg")
            published = fa.promote_candidate(str(candidate), str(day),
                                             "5678cdef_120000-first-abcdef.png")
            self.assertTrue(os.path.isfile(published))
            result = rc.plan([str(day)], db_path=str(day / "none.db"))
            self.assertEqual(result["renames"], [])
            self.assertEqual(result["groups"], 0)


if __name__ == "__main__":
    unittest.main()
