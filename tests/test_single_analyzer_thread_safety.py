import json
import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as single_analyzer_module
from modules.image_analysis.single_analyzer import SingleAnalyzerWidget


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def text_config():
    return ("https://example.invalid/v1", "test-key", "test-model")


def image_config():
    return ("https://example.invalid/v1", "image-key", "image-model", "openai")


def styles_config():
    return {"默认风格": "masterpiece", "维多利亚": "victorian dress"}


def ar_policy():
    return {
        "default_aspect_ratio": "1:1",
        "override_first": "不覆盖(沿用原逻辑)",
        "override_second": "3:4",
    }


class MockThread:
    def __init__(self, thread_no, task_hash, task_id):
        self.meta_thread_no = thread_no
        self.meta_task_hash = task_hash
        self.meta_task_id = task_id
        self.last_status = "success"


def test_on_process_finished_thread_safety_no_race_condition(qapp, monkeypatch, tmp_path):
    """测试 on_process_finished 方法在多线程场景下不会出现竞态条件
    
    模拟5个分析任务快速连续完成的场景，验证每个任务的 prompt_bundle 使用的是自己的提示词数据。
    
    这个测试验证了修复：在构建 prompt_bundle 时，直接从 result_json 和 safe_task_hash 获取数据，
    完全不依赖实例变量，避免多个任务同时完成时，实例变量被后续任务覆盖导致使用错误的提示词。
    """
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    widget.auto_gen_ref_cb.setChecked(True)

    results = []

    def mock_trigger_image_generation(prompt_type, is_auto=False, prompt_bundle=None, **kwargs):
        results.append({
            "prompt_type": prompt_type,
            "prompt_bundle": prompt_bundle,
        })
        return True

    monkeypatch.setattr(widget, "trigger_image_generation", mock_trigger_image_generation)

    num_tasks = 5
    task_data = []
    for i in range(num_tasks):
        task_hash = f"task_{i:02d}"
        result_json = {
            "english_description": f"refined_prompt_{i}",
            "original_english_description": f"original_prompt_{i}",
            "aspect_ratio": "1:1",
            "japanese_title": f"title_{i}",
        }
        thread = MockThread(i, task_hash, f"id_{i}")
        task_data.append((thread, result_json))

    for thread, result_json in task_data:
        widget.on_process_finished(thread, result_json)

    assert len(results) == num_tasks, f"Expected {num_tasks} results, got {len(results)}"

    for i, result in enumerate(results):
        prompt_bundle = result["prompt_bundle"]
        expected_task_hash = f"task_{i:02d}"
        expected_refined = f"refined_prompt_{i}"
        
        assert prompt_bundle["task_hash"] == expected_task_hash, \
            f"Task {i}: Expected task_hash '{expected_task_hash}', got '{prompt_bundle['task_hash']}'"
        
        assert prompt_bundle["refined_prompt"] == expected_refined, \
            f"Task {i}: Expected refined_prompt '{expected_refined}', got '{prompt_bundle['refined_prompt']}'"

    widget.close()


def test_on_process_finished_thread_safety_concurrent(qapp, monkeypatch, tmp_path):
    """测试 on_process_finished 在并发场景下的数据隔离
    
    使用线程池模拟多个分析任务几乎同时完成的场景，验证每个任务的 prompt_bundle 数据是隔离的。
    
    这是一个更严格的测试，确保即使在并发环境下，每个任务的数据也不会被其他任务污染。
    """
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    widget.auto_gen_ref_cb.setChecked(True)

    results = []
    results_lock = threading.Lock()
    task_completed = threading.Barrier(5)

    def mock_trigger_image_generation(prompt_type, is_auto=False, prompt_bundle=None, **kwargs):
        with results_lock:
            results.append({
                "prompt_type": prompt_type,
                "prompt_bundle": prompt_bundle,
            })
        task_completed.wait()
        return True

    monkeypatch.setattr(widget, "trigger_image_generation", mock_trigger_image_generation)

    num_tasks = 5
    task_data = []
    for i in range(num_tasks):
        task_hash = f"conc_task_{i:02d}"
        result_json = {
            "english_description": f"conc_refined_{i}",
            "original_english_description": f"conc_original_{i}",
            "aspect_ratio": "1:1",
            "japanese_title": f"conc_title_{i}",
        }
        thread = MockThread(i, task_hash, f"conc_id_{i}")
        task_data.append((thread, result_json))

    def run_task(thread, result_json):
        widget.on_process_finished(thread, result_json)

    threads = []
    for thread, result_json in task_data:
        t = threading.Thread(target=run_task, args=(thread, result_json))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert len(results) == num_tasks, f"Expected {num_tasks} results, got {len(results)}"

    task_hash_map = {}
    for result in results:
        bundle = result["prompt_bundle"]
        task_hash = bundle["task_hash"]
        task_hash_map[task_hash] = bundle

    for i in range(num_tasks):
        expected_task_hash = f"conc_task_{i:02d}"
        assert expected_task_hash in task_hash_map, f"Missing task_hash: {expected_task_hash}"
        
        bundle = task_hash_map[expected_task_hash]
        assert bundle["refined_prompt"] == f"conc_refined_{i}", \
            f"Task {i}: Expected refined_prompt 'conc_refined_{i}', got '{bundle['refined_prompt']}'"
        assert bundle["original_prompt"] == f"conc_original_{i}", \
            f"Task {i}: Expected original_prompt 'conc_original_{i}', got '{bundle['original_prompt']}'"

    widget.close()


def test_on_process_finished_prompt_bundle_isolation(qapp, monkeypatch, tmp_path):
    """测试每个任务的 prompt_bundle 数据是隔离的
    
    验证即使多个任务快速连续完成，每个任务的 prompt_bundle 也只包含自己的数据，
    不会混入其他任务的数据。
    """
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    widget.auto_gen_ref_cb.setChecked(True)

    results = []

    def mock_trigger_image_generation(prompt_type, is_auto=False, prompt_bundle=None, **kwargs):
        results.append({
            "prompt_type": prompt_type,
            "prompt_bundle": prompt_bundle,
        })
        return True

    monkeypatch.setattr(widget, "trigger_image_generation", mock_trigger_image_generation)

    num_tasks = 3
    for i in range(num_tasks):
        task_hash = f"hash_{i}"
        result_json = {
            "english_description": f"desc_{i}",
            "original_english_description": f"orig_{i}",
            "aspect_ratio": f"{i+1}:{i+2}",
            "japanese_title": f"title_{i}",
        }
        thread = MockThread(i, task_hash, f"id_{i}")
        widget.on_process_finished(thread, result_json)

    assert len(results) == num_tasks

    for i, result in enumerate(results):
        bundle = result["prompt_bundle"]
        assert bundle["task_hash"] == f"hash_{i}"
        assert bundle["refined_prompt"] == f"desc_{i}"
        assert bundle["original_prompt"] == f"orig_{i}"
        assert bundle["aspect_ratio"] == f"{i+1}:{i+2}"

    widget.close()


def test_on_process_finished_analysis_json_path_in_bundle(qapp, monkeypatch, tmp_path):
    """测试 analysis_json_path 被正确包含在 prompt_bundle 中
    
    验证修复：将 analysis_json_path 包含在 prompt_bundle 中，避免使用被覆盖的实例变量。
    """
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    widget.auto_gen_ref_cb.setChecked(True)

    results = []

    def mock_trigger_image_generation(prompt_type, is_auto=False, prompt_bundle=None, **kwargs):
        results.append(prompt_bundle)
        return True

    monkeypatch.setattr(widget, "trigger_image_generation", mock_trigger_image_generation)

    result_json = {
        "english_description": "test_refined",
        "original_english_description": "test_original",
        "aspect_ratio": "1:1",
        "japanese_title": "test_title",
    }
    thread = MockThread(1, "test_hash", "test_id")

    widget.on_process_finished(thread, result_json)

    assert len(results) == 1
    bundle = results[0]
    
    assert "analysis_json_path" in bundle, "analysis_json_path should be in prompt_bundle"
    assert bundle["analysis_json_path"] != "", "analysis_json_path should not be empty"
    assert "test_hash" in bundle["analysis_json_path"], "analysis_json_path should contain task hash"

    widget.close()