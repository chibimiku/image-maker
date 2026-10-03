# -*- coding: utf-8 -*-
"""第四轮 P0.4 回归：付费动作的**发送前**原子预算（全程假 HTTP，零付费）。

覆盖 PROTOCOL §P0.4 的每一条：硬上限 8、第 9 次发送为 0、未知尝试计数、重启复用账本、
相同 run 拒绝、底层自动重试被关闭、"不能确认一次执行最多一次发送就不准启动"。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from PIL import Image

import utils.send_budget as send_budget


def _png(path, color="white"):
    Image.new("RGB", (32, 32), color).save(path)
    return str(path)


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self._env = patch.dict(os.environ, {"IMAGE_MAKER_SEND_BUDGET_DIR": self.dir,
                                            "IMAGE_MAKER_SEND_BUDGET_LIMIT": "8"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_ninth_attempt_is_refused(self):
        for index in range(8):
            send_budget.reserve_image_attempt(run_id="run-%d" % index, operation="repaint")
        self.assertEqual(send_budget.state(self.dir, "image", 8)["used"], 8)
        with self.assertRaises(send_budget.SendBudgetExhausted):
            send_budget.reserve_image_attempt(run_id="run-9", operation="repaint")
        self.assertEqual(send_budget.state(self.dir, "image", 8)["used"], 8)  # 没有第 9 个槽位
        self.assertEqual(send_budget.state(self.dir, "image", 8)["remaining"], 0)

    def test_budget_is_persisted_and_survives_restart(self):
        """账本是文件系统本身：换一个「进程」（重新读盘）必须看到同样的已用额度。"""
        first = send_budget.reserve_image_attempt(run_id="run-a", operation="repaint")
        send_budget.settle_image_attempt(first, status="success", output_path="/tmp/x.png")
        state = send_budget.state(self.dir, "image", 8)
        self.assertEqual(state["used"], 1)
        self.assertEqual(state["dispatched_run_ids"], ["run-a"])

    def test_same_run_id_is_refused_on_second_dispatch(self):
        send_budget.reserve_image_attempt(run_id="run-a", operation="repaint")
        with self.assertRaises(send_budget.SendBudgetExhausted) as info:
            send_budget.reserve_image_attempt(run_id="run-a", operation="repaint")
        self.assertIn("已经派发过", str(info.exception))

    def test_unknown_and_success_settlements_are_recorded_separately(self):
        good = send_budget.reserve_image_attempt(run_id="run-good", operation="repaint")
        send_budget.settle_image_attempt(good, status="success", output_path="/tmp/a.png",
                                         output_sha256="a" * 64)
        bad = send_budget.reserve_image_attempt(run_id="run-unknown", operation="repaint")
        send_budget.settle_image_attempt(bad, status="unknown", error="read timeout")
        rows = [json.loads(line) for line in
                (Path(self.dir) / "images.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        settled = [row for row in rows if row["event"] == "settled"]
        self.assertEqual({row["status"] for row in settled}, {"success", "unknown"})
        self.assertEqual([row["billed"] for row in settled if row["status"] == "unknown"], ["unknown"])
        # 作废不重置额度：两次尝试都还在账上
        self.assertEqual(send_budget.state(self.dir, "image", 8)["used"], 2)

    def test_reuse_only_for_successful_products(self):
        with tempfile.TemporaryDirectory() as folder:
            product = _png(Path(folder) / "out.png")
            good = send_budget.reserve_image_attempt(run_id="run-good", operation="repaint")
            send_budget.settle_image_attempt(good, status="success", output_path=product)
            failed = send_budget.reserve_image_attempt(run_id="run-failed", operation="repaint")
            send_budget.settle_image_attempt(failed, status="failed")
            self.assertEqual(send_budget.can_reuse_success(self.dir, "run-good"), product)
            self.assertEqual(send_budget.can_reuse_success(self.dir, "run-failed"), "")

    def test_text_budget_per_stage_and_global(self):
        with patch.dict(os.environ, {"IMAGE_MAKER_TEXT_BUDGET_LIMIT": "64",
                                     "IMAGE_MAKER_TEXT_BUDGET_PER_STAGE": "2"}):
            send_budget.reserve_text_attempt(stage="run-a/quality-audit")
            send_budget.reserve_text_attempt(stage="run-a/quality-audit")
            with self.assertRaises(send_budget.SendBudgetExhausted):
                send_budget.reserve_text_attempt(stage="run-a/quality-audit")
            send_budget.reserve_text_attempt(stage="run-b/quality-audit")  # 别的 run 仍可用


class ImageRetryTests(unittest.TestCase):
    """底层自动重试必须能被关掉，而且要能在真正发送点被验证。"""

    def test_retry_override_forces_a_single_http_post(self):
        from modules.others import api_backend
        with tempfile.TemporaryDirectory() as folder:
            image = _png(Path(folder) / "base.png")
            config = {"base_url": "https://example.invalid/v1beta/models/", "api_key": "k",
                      "timeout": 5, "max_retries": 1}
            calls = []

            def boom(*args, **kwargs):
                calls.append(args)
                raise requests.exceptions.ConnectionError("offline")

            with patch.object(api_backend, "get_api_config", lambda **kwargs: config), \
                    patch.object(api_backend.requests, "post", boom), \
                    patch.dict(os.environ, {"IMAGE_MAKER_IMAGE_MAX_RETRIES": "0"}):
                api_backend.generate_image_aigc2d(prompt="x", image_paths=[image], model="m",
                                                  resolution="2K", api_type="aigc2d")
            self.assertEqual(len(calls), 1)

            calls.clear()
            env = {k: v for k, v in os.environ.items() if k != "IMAGE_MAKER_IMAGE_MAX_RETRIES"}
            with patch.object(api_backend, "get_api_config", lambda **kwargs: config), \
                    patch.object(api_backend.requests, "post", boom), \
                    patch.dict(os.environ, env, clear=True):
                api_backend.generate_image_aigc2d(prompt="x", image_paths=[image], model="m",
                                                  resolution="2K", api_type="aigc2d")
            self.assertEqual(len(calls), 2)   # 未覆盖时会重发一次 —— 所以受控实验必须覆盖

    def test_dispatch_refuses_to_send_when_retries_are_not_zero(self):
        from modules.others import api_backend
        with tempfile.TemporaryDirectory() as folder:
            image = _png(Path(folder) / "base.png")
            out = Path(folder) / "out"
            config = {"base_url": "https://example.invalid/v1beta/models/", "api_key": "k",
                      "timeout": 5, "max_retries": 1}
            calls = []

            def boom(*args, **kwargs):
                calls.append(args)
                raise AssertionError("预算模式不允许真的发出去")

            env = {k: v for k, v in os.environ.items() if k != "IMAGE_MAKER_IMAGE_MAX_RETRIES"}
            env["IMAGE_MAKER_SEND_BUDGET_DIR"] = str(Path(folder) / "budget")
            env["IMAGE_MAKER_SEND_BUDGET_LIMIT"] = "8"
            env["IMAGE_MAKER_SEND_BUDGET_RUN_ID"] = "run-guard"
            with patch.object(api_backend, "get_api_config", lambda **kwargs: config), \
                    patch.object(api_backend.requests, "post", boom), \
                    patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError) as info:
                    api_backend.generate_image_repaint([image], prompt="p", use_detail_suffix=False,
                                                       resolution="2K", save_sub_dir=str(out),
                                                       file_prefix="guard")
            self.assertIn("max_retries", str(info.exception))
            self.assertEqual(calls, [])
            ledger = (Path(folder) / "budget" / "images.jsonl").read_text(encoding="utf-8")
            self.assertIn('"status": "refused"', ledger)

    def test_dispatch_settles_success_and_records_transmit_digest(self):
        from modules.others import api_backend
        with tempfile.TemporaryDirectory() as folder:
            image = _png(Path(folder) / "base.png", "white")
            product = _png(Path(folder) / "fake-out.png", "black")
            out = Path(folder) / "out"
            env = {k: v for k, v in os.environ.items() if k != "IMAGE_MAKER_IMAGE_MAX_RETRIES"}
            env.update({"IMAGE_MAKER_SEND_BUDGET_DIR": str(Path(folder) / "budget"),
                        "IMAGE_MAKER_SEND_BUDGET_LIMIT": "8",
                        "IMAGE_MAKER_SEND_BUDGET_RUN_ID": "run-ok",
                        "IMAGE_MAKER_IMAGE_MAX_RETRIES": "0"})
            with patch.object(api_backend, "generate_image_aigc2d",
                              lambda **kwargs: [product]), \
                    patch.dict(os.environ, env, clear=True):
                saved = api_backend.generate_image_repaint([image], prompt="p", use_detail_suffix=False,
                                                           resolution="2K", save_sub_dir=str(out),
                                                           file_prefix="ok")
            self.assertEqual(saved, [product])
            rows = [json.loads(line) for line in
                    (Path(folder) / "budget" / "images.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            slot_file = next((Path(folder) / "budget" / "slots").glob("*.json"))
            reserved = json.loads(slot_file.read_text(encoding="utf-8"))
            settled = [row for row in rows if row["event"] == "settled"][0]
            self.assertEqual(reserved["reference_images"][0]["path"], os.path.abspath(image))
            self.assertTrue(reserved["reference_images"][0]["transmitted_sha256"])
            self.assertEqual(settled["status"], "success")
            self.assertEqual(settled["output_path"], product)

    def test_transmit_digest_differs_from_file_digest(self):
        """文件哈希与传输字节哈希是两件事（后端会重编码），不能互相冒充。"""
        from modules.others.api_backend import transmitted_image_digest, _file_digest
        with tempfile.TemporaryDirectory() as folder:
            image = _png(Path(folder) / "base.png")
            facts = transmitted_image_digest(image)
            self.assertNotEqual(facts["file_sha256"], facts["transmitted_sha256"])
            self.assertEqual(facts["file_sha256"], _file_digest(image))
            self.assertGreater(facts["transmitted_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
