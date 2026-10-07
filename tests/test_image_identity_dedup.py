# -*- coding: utf-8 -*-
"""「扩展名撒谎」识别 + SHA-256 去重的离线回归。

覆盖三层：
1) `utils.image_identity`：magic bytes 嗅探、假后缀判定、后缀纠正、重名让位；
2) `utils.first_image_audit.promote_candidate`：发布名后缀跟真实字节走、重名不覆盖；
3) `utils.image_dedup`：分组规则、幸存者选择、预演/实删、删前三重复核。

全部离线，用 PIL 造真 PNG / 真 JPEG，不碰网络、不碰真实 data 目录。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils import first_image_audit as fa
from utils import image_dedup as dedup
from utils import image_identity as ident


def _png(path, color=(200, 30, 60), size=(32, 32)):
    Image.new("RGB", size, color).save(str(path), format="PNG")
    return str(path)


def _jpeg(path, color=(20, 120, 200), size=(32, 32)):
    Image.new("RGB", size, color).save(str(path), format="JPEG")
    return str(path)


class IdentityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_bytes_tell_the_real_format(self):
        self.assertEqual(ident.sniff_format(_png(self.tmp / "a.png")), "png")
        self.assertEqual(ident.sniff_format(_jpeg(self.tmp / "a.jpg")), "jpeg")

    def test_misnamed_png_that_is_really_jpeg(self):
        path = self.tmp / "fake.png"
        _jpeg(path)
        self.assertTrue(ident.is_misnamed(str(path)))
        self.assertEqual(ident.declared_format(str(path)), "png")
        self.assertEqual(ident.sniff_format(str(path)), "jpeg")

    def test_real_png_and_real_jpeg_are_not_misnamed(self):
        self.assertFalse(ident.is_misnamed(_png(self.tmp / "ok.png")))
        self.assertFalse(ident.is_misnamed(_jpeg(self.tmp / "ok.jpg")))
        # .jpeg 与 .jpg 是同一格式，不算撒谎
        path = self.tmp / "ok.jpeg"
        _jpeg(path)
        self.assertFalse(ident.is_misnamed(str(path)))

    def test_unknown_extension_is_never_judged(self):
        path = self.tmp / "mystery.bin"
        _jpeg(path)
        self.assertFalse(ident.is_misnamed(str(path)))
        self.assertFalse(ident.declared_format(str(path)))

    def test_true_name_only_fixes_lying_extensions(self):
        fake = self.tmp / "fake.png"
        _jpeg(fake)
        self.assertEqual(os.path.basename(ident.true_name(str(fake))), "fake.jpg")
        real = self.tmp / "real.png"
        _png(real)
        self.assertEqual(ident.true_name(str(real)), str(real))
        dotted = self.tmp / "kept.jpeg"
        _jpeg(dotted)
        self.assertEqual(ident.true_name(str(dotted)), str(dotted))

    def test_unique_path_steps_aside(self):
        first = self.tmp / "a.png"
        _png(first)
        second = ident.unique_path(str(first))
        self.assertEqual(os.path.basename(second), "a-1.png")
        _png(second)
        self.assertEqual(os.path.basename(ident.unique_path(str(first))), "a-2.png")
        self.assertEqual(ident.unique_path(str(self.tmp / "free.png")),
                         str(self.tmp / "free.png"))


class PromoteCandidateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "publish"
        self.logs = []

    def tearDown(self):
        self._tmp.cleanup()

    def _log(self, message):
        self.logs.append(str(message))

    def test_publish_extension_follows_real_bytes(self):
        """发布名写死 .png，真实字节是 JPEG → 落盘必须是 .jpg。"""
        src = _jpeg(self.tmp / "cand.jpg")
        result = fa.promote_candidate(src, str(self.out), "8d0cb278_020028-first-ed899a.png",
                                      log_callback=self._log)
        self.assertTrue(result.endswith(".jpg"), result)
        self.assertTrue(os.path.isfile(result))
        self.assertEqual(ident.sniff_format(result), "jpeg")
        self.assertEqual(os.path.basename(result), "8d0cb278_020028-first-ed899a.jpg")
        self.assertTrue(any("magic bytes" in line for line in self.logs))

    def test_real_png_publish_name_is_left_alone(self):
        src = _png(self.tmp / "cand.png")
        result = fa.promote_candidate(src, str(self.out), "h_000000-first-abcdef.png")
        self.assertEqual(os.path.basename(result), "h_000000-first-abcdef.png")

    def test_identical_target_is_reused_not_duplicated(self):
        src = _jpeg(self.tmp / "cand.jpg")
        first = fa.promote_candidate(src, str(self.out), "h_000000-first-aaaaaa.png",
                                     log_callback=self._log)
        second = fa.promote_candidate(src, str(self.out), "h_000000-first-aaaaaa.png",
                                      log_callback=self._log)
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.out.iterdir())), 1)
        self.assertTrue(any("内容相同" in line for line in self.logs))

    def test_collision_with_other_content_keeps_both(self):
        a = _jpeg(self.tmp / "a.jpg", color=(255, 0, 0))
        b = _jpeg(self.tmp / "b.jpg", color=(0, 0, 255))
        first = fa.promote_candidate(a, str(self.out), "h_000000-first-aaaaaa.png")
        second = fa.promote_candidate(b, str(self.out), "h_000000-first-aaaaaa.png",
                                      log_callback=self._log)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.isfile(first))
        self.assertTrue(os.path.isfile(second))
        self.assertEqual(os.path.basename(second), "h_000000-first-aaaaaa-1.jpg")
        self.assertTrue(any("两份都保留" in line for line in self.logs))

    def test_missing_candidate_still_returns_empty(self):
        self.assertEqual(fa.promote_candidate(str(self.tmp / "nope.jpg"),
                                              str(self.out), "x.png"), "")

    def test_generate_with_audit_publishes_a_jpeg_under_jpg(self):
        """端到端：候选是 JPEG、发布名写死 `.png` → 交付路径必须落成 `.jpg`。"""
        self.out.mkdir(parents=True, exist_ok=True)
        candidate = self.out / "h_000000-first-abcdef-a1_010101-aaaaaa.jpg"
        _jpeg(candidate)

        outcome = fa.generate_with_audit(
            lambda prompt, round_index: [str(candidate)],
            prompt="P", max_attempts=1, audit=None,
            candidate_dir=str(self.out), publish_dir=str(self.out),
            publish_name="h_000000-first-abcdef.png",
            record_prefix="h_000000-first-abcdef",
            log_callback=self._log)
        self.assertEqual(os.path.basename(outcome["published"]),
                         "h_000000-first-abcdef.jpg")
        self.assertTrue(os.path.isfile(outcome["published"]))
        self.assertEqual(ident.sniff_format(outcome["published"]), "jpeg")
        # 候选原件保留，审计台账仍指着它
        self.assertTrue(candidate.is_file())


class DedupPlanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _pair(self):
        """造一对：真后缀的候选 + 同字节的假 PNG 发布名。"""
        truthful = self.tmp / "8d0cb278_020028-first-ed899a-a1_020112-1ffc69.jpg"
        _jpeg(truthful)
        fake = self.tmp / "8d0cb278_020028-first-ed899a.png"
        shutil.copy2(truthful, fake)
        return truthful, fake

    def test_fake_png_with_truthful_twin_is_removable(self):
        truthful, fake = self._pair()
        result = dedup.plan([str(self.tmp)])
        names = [os.path.basename(row["path"]) for row in result["removable"]]
        self.assertEqual(names, [fake.name])
        self.assertEqual(result["removable"][0]["kept"], str(truthful))
        self.assertEqual(result["removable"][0]["duplicate_kind"], "misnamed")

    def test_misnamed_without_truthful_twin_is_never_deleted(self):
        a = self.tmp / "one.png"
        _jpeg(a)
        b = self.tmp / "two.png"
        shutil.copy2(a, b)
        result = dedup.plan([str(self.tmp)])
        self.assertEqual(result["removable"], [])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(result["skipped"][0]["reason"], "no_truthful_twin")

    def test_truthful_duplicates_are_info_only_by_default(self):
        a = _jpeg(self.tmp / "x-a1_020112-1ffc69.jpg")
        b = self.tmp / "x-first-ed899a.jpg"
        shutil.copy2(a, b)
        result = dedup.plan([str(self.tmp)])
        self.assertEqual(result["removable"], [])
        self.assertEqual(len(result["info"]), 1)
        self.assertEqual(result["info"][0]["kept"], str(b))   # 幸存者优先发布名

    def test_mode_all_also_collapses_truthful_duplicates(self):
        a = _jpeg(self.tmp / "x-a1_020112-1ffc69.jpg")
        b = self.tmp / "x-first-ed899a.jpg"
        shutil.copy2(a, b)
        result = dedup.plan([str(self.tmp)], mode=dedup.MODE_ALL)
        names = [os.path.basename(row["path"]) for row in result["removable"]]
        self.assertEqual(names, [os.path.basename(a)])
        self.assertEqual(result["removable"][0]["kept"], str(b))

    def test_directories_are_never_merged(self):
        """同字节文件分散在两个目录时，各算一组，绝不跨目录合并成"重复"。"""
        (self.tmp / "d1").mkdir()
        (self.tmp / "d2").mkdir()
        a = _jpeg(self.tmp / "d1" / "same.jpg")
        shutil.copy2(a, self.tmp / "d2" / "same.jpg")
        result = dedup.plan([str(self.tmp)], recursive=True)
        self.assertEqual(result["removable"], [])
        self.assertEqual(result["groups"], [])          # 没有哪个目录内部凑够两张
        self.assertEqual(len(dedup.scan([str(self.tmp)], recursive=True)), 2)

        # 只在 d1 内部凑一对 → 只动 d1，d2 那张不受影响
        shutil.copy2(a, self.tmp / "d1" / "same-first-ed899a.png")
        result = dedup.plan([str(self.tmp)], recursive=True)
        self.assertEqual(len(result["removable"]), 1)
        self.assertTrue(result["removable"][0]["path"].startswith(str(self.tmp / "d1")))

    def test_bad_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            dedup.plan([str(self.tmp)], mode="whatever")


class DedupApplyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _pair(self):
        truthful = self.tmp / "x-a1_020112-1ffc69.jpg"
        _jpeg(truthful)
        fake = self.tmp / "x-first-ed899a.png"
        shutil.copy2(truthful, fake)
        return truthful, fake

    def test_apply_deletes_only_the_fake_one(self):
        truthful, fake = self._pair()
        result = dedup.apply_plan(dedup.plan([str(self.tmp)]))
        self.assertEqual([os.path.basename(r["path"]) for r in result["deleted"]], [fake.name])
        self.assertFalse(fake.exists())
        self.assertTrue(truthful.exists())
        self.assertGreater(result["freed_bytes"], 0)

    def test_apply_refuses_when_survivor_is_gone(self):
        truthful, fake = self._pair()
        planned = dedup.plan([str(self.tmp)])
        os.remove(truthful)
        result = dedup.apply_plan(planned)
        self.assertEqual(result["deleted"], [])
        self.assertEqual(len(result["refused"]), 1)
        self.assertTrue(fake.exists())          # 没有幸存者就一张都不删

    def test_apply_refuses_when_hash_changed(self):
        truthful, fake = self._pair()
        planned = dedup.plan([str(self.tmp)])
        _jpeg(fake, color=(1, 2, 3))            # 计划之后内容变了
        result = dedup.apply_plan(planned)
        self.assertEqual(result["deleted"], [])
        self.assertTrue(any("哈希" in row["reason"] for row in result["refused"]))
        self.assertTrue(fake.exists())

    def test_apply_refuses_when_file_vanished(self):
        truthful, fake = self._pair()
        planned = dedup.plan([str(self.tmp)])
        os.remove(fake)
        result = dedup.apply_plan(planned)
        self.assertEqual(result["deleted"], [])
        self.assertEqual(result["refused"][0]["reason"], "文件已不在")


if __name__ == "__main__":
    unittest.main()
