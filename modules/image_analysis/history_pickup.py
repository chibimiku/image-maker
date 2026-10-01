"""Browse GPT generation artifacts and branch a checkpoint from a chosen image."""

import copy
import datetime
import json
import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton, QRadioButton, QVBoxLayout)

from utils.generation_checkpoint import GenerationCheckpoint, STAGE_LABELS


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

PROMPT_TYPE_LABELS = {"original": "原始提示词", "refined": "优化提示词"}
# 优化提示词是自动生图的主用法，摆在前面也是默认勾选。
PROMPT_TYPE_ORDER = ("refined", "original")


def discover_generation_checkpoints(data_root, task_hash=""):
    """Find saved GPT attempts; a selected task only sees its own checkpoints."""
    root = Path(data_root)
    found = []
    selected = str(task_hash or "").strip().lower()
    if not root.is_dir():
        return found
    for date_dir in root.iterdir():
        if not date_dir.is_dir() or not (len(date_dir.name) == 8 and date_dir.name.isdigit()):
            continue
        work_root = date_dir / "analysis-gpt-image"
        if not work_root.is_dir():
            continue
        for task_dir in work_root.iterdir():
            if not task_dir.is_dir():
                continue
            if selected and not task_dir.name.lower().startswith(selected + "-"):
                continue
            for path in task_dir.glob("generation-checkpoint*.json"):
                if path.is_file():
                    found.append(str(path.resolve()))
    found.sort(key=lambda path: (os.path.getmtime(path), path), reverse=True)
    return found


def _key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def list_process_artifacts(checkpoint_path):
    """Include audits and prompts as well as images, ordered by file time."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    with open(checkpoint_path, encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    root = Path(checkpoint_path).parent
    labels = {}
    stages = checkpoint.get("stages") or {}
    for stage, entry in stages.items():
        for output in entry.get("outputs") or []:
            labels[_key(output)] = (stage, STAGE_LABELS.get(stage, stage))
    for operation, output in (checkpoint.get("operations") or {}).items():
        if not isinstance(output, list):
            continue
        stage = operation.split("-", 1)[0]
        for path in output:
            if isinstance(path, str):
                labels[_key(path)] = (stage, STAGE_LABELS.get(stage, stage) + " / " + operation)
    for manifest in root.glob("pipeline-steps*/pipeline-manifest.json"):
        try:
            with manifest.open(encoding="utf-8") as stream:
                data = json.load(stream)
            for item in data.get("items") or []:
                for step in item.get("steps") or []:
                    if step.get("out"):
                        labels[_key(step["out"])] = ("pipeline", "流水线 / " + str(step.get("key") or "工序"))
        except (OSError, ValueError, TypeError):
            continue
    artifacts = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        key = _key(path)
        stage, label = labels.get(key, ("", "过程文件"))
        name = path.name.lower()
        if not stage and (name.startswith("final-quality-") or name.startswith("final-repair-")):
            stage, label = "final_review", STAGE_LABELS["final_review"]
        if not stage:
            for candidate in reversed(list(STAGE_LABELS)):
                if candidate.replace("_", "-") in name:
                    stage, label = candidate, STAGE_LABELS[candidate]
                    break
        if not stage and ("replay" in name or "server_response" in name
                          or name.endswith(".request.json")):
            stage, label = "first", "GPT 首图 / 请求与响应"
        if not stage and "pipeline-manifest" in name:
            stage, label = "pipeline", "Gemini 重绘与本地工序 / 清单"
        if not stage and "generation-checkpoint" in name:
            label = "工序断点"
        if not stage and path.suffix.lower() in IMAGE_SUFFIXES:
            if "final-rescue" in name or "final-repair" in name:
                stage, label = "final_review", STAGE_LABELS["final_review"] + " / 候选图"
            else:
                stage, label = "pipeline", "过程图（工序未标记）"
        artifacts.append({"path": str(path), "stage": stage, "label": label,
                          "image": path.suffix.lower() in IMAGE_SUFFIXES,
                          "mtime": path.stat().st_mtime})
    return sorted(artifacts, key=lambda item: (item["mtime"], item["path"]))


def branch_from_artifact(checkpoint_path, image_path, stage):
    """Reuse the chosen pixels as one completed stage, then rerun downstream gates."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    image_path = os.path.abspath(image_path)
    process_dir = os.path.dirname(checkpoint_path)
    if os.path.commonpath((process_dir, image_path)) != process_dir or not os.path.isfile(image_path):
        raise ValueError("只能从该断点目录中选择仍存在的过程图")
    with open(checkpoint_path, encoding="utf-8") as stream:
        original = json.load(stream)
    keys = list(STAGE_LABELS)
    if stage not in keys or stage == "publish":
        raise ValueError("所选文件没有可续跑的工序")
    if stage == "final_review":
        # A failed final audit never certifies its candidate. Recheck it.
        next_stage = "final_review"
        previous = [key for key in keys[:keys.index(stage)]
                    if original.get("stages", {}).get(key, {}).get("status") == "success"]
        stage = previous[-1] if previous else "first"
    else:
        next_stage = keys[keys.index(stage) + 1]
    stamp = datetime.datetime.now().strftime("%H%M%S%f")
    target = os.path.join(process_dir, f"generation-checkpoint-pickup-{stamp}.json")
    branch = GenerationCheckpoint(target, copy.deepcopy(original.get("snapshot") or {}))
    branch.data = copy.deepcopy(original)
    branch.reset_from(next_stage)
    if next_stage == "pipeline":
        # The post-process runner has its own manifest cache. A new branch must
        # not silently pick up the old repaint instead of executing it again.
        branch.data["pipeline_work_dir"] = os.path.join(process_dir, "pipeline-steps-pickup-" + stamp)
    branch.data["stages"].setdefault(stage, {}).update(status="success", outputs=[image_path])
    branch.data.update(status="error", current_stage=next_stage, last_outputs=[image_path],
                       pickup_source=image_path, pickup_parent=checkpoint_path)
    branch.save()
    return target, next_stage


class HistoryPickupDialog(QDialog):
    def __init__(self, checkpoint_path, parent=None):
        super().__init__(parent)
        self.checkpoint_paths = [os.path.abspath(path) for path in
                                 (checkpoint_path if isinstance(checkpoint_path, list) else [checkpoint_path])]
        self.checkpoint_path = self.checkpoint_paths[0]
        self.artifacts = []
        self.action = ""
        self.selected_artifact = None
        self.setWindowTitle("拾取历史")
        self.resize(1000, 690)
        layout = QVBoxLayout(self)
        self.attempts = QComboBox()
        for path in self.checkpoint_paths:
            self.attempts.addItem(f"{Path(path).parent.parent.parent.name} / "
                                  f"{Path(path).parent.name} / {Path(path).name}")
        layout.addWidget(self.attempts)
        self.header = QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)
        row = QHBoxLayout()
        layout.addLayout(row, 1)
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(490)
        row.addWidget(self.file_list, 1)
        preview_layout = QVBoxLayout()
        row.addLayout(preview_layout, 1)
        self.preview = QLabel("选择文件预览")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(400, 450)
        self.preview.setStyleSheet("background: #202020; color: white;")
        preview_layout.addWidget(self.preview, 1)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        preview_layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.resume_button = QPushButton("从选中图的下一工序继续")
        self.publish_button = QPushButton("将选中图作为最终图发布")
        # 断点里没有可用的图（或用户根本不想要这些图）时的退路：拿源图重新分析一次
        self.rerun_button = QPushButton("没有可用的过程图？重新分析")
        close_button = QPushButton("关闭")
        buttons.addWidget(self.resume_button)
        buttons.addWidget(self.publish_button)
        buttons.addWidget(self.rerun_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        self.resume_button.clicked.connect(lambda: self._choose("resume"))
        self.publish_button.clicked.connect(lambda: self._choose("publish"))
        self.rerun_button.clicked.connect(lambda: self._choose("rerun-analysis"))
        close_button.clicked.connect(self.reject)
        self.file_list.currentRowChanged.connect(self._update_selection)
        self.attempts.currentIndexChanged.connect(self._load_attempt)
        self._load_attempt(0)

    def _load_attempt(self, index):
        if index < 0:
            return
        self.checkpoint_path = self.checkpoint_paths[index]
        self.artifacts = list_process_artifacts(self.checkpoint_path)
        with open(self.checkpoint_path, encoding="utf-8") as stream:
            checkpoint = json.load(stream)
        self.header.setText("按文件时间排序；选择过程图后可从下一工序续跑，或人工选为发布最终图。\n"
                            f"当前断点：{checkpoint.get('status', '未知')} / "
                            f"{STAGE_LABELS.get(checkpoint.get('current_stage'), checkpoint.get('current_stage', '未知'))}。"
                            "人工发布保留原审计失败记录。")
        self.file_list.clear()
        for artifact in self.artifacts:
            timestamp = datetime.datetime.fromtimestamp(artifact["mtime"]).strftime("%H:%M:%S")
            relative = os.path.relpath(artifact["path"], os.path.dirname(self.checkpoint_path))
            self.file_list.addItem(QListWidgetItem(f"{timestamp}  {artifact['label']}\n{relative}"))
        latest = (checkpoint.get("last_outputs") or [""])[-1]
        latest_rows = [index for index, artifact in enumerate(self.artifacts)
                       if _key(artifact["path"]) == _key(latest)]
        image_rows = [index for index, artifact in enumerate(self.artifacts) if artifact["image"]]
        if image_rows:
            self.file_list.setCurrentRow(latest_rows[-1] if latest_rows else image_rows[-1])
        else:
            self._update_selection(-1)

    def _update_selection(self, row):
        artifact = self.artifacts[row] if 0 <= row < len(self.artifacts) else None
        self.selected_artifact = artifact
        can_image = bool(artifact and artifact["image"])
        self.resume_button.setEnabled(can_image and artifact["stage"] in STAGE_LABELS
                                      and artifact["stage"] != "publish")
        self.publish_button.setEnabled(can_image)
        if not artifact:
            self.preview.setText("没有过程文件")
            self.detail.clear()
            return
        self.detail.setText(f"{artifact['label']}\n{artifact['path']}")
        if can_image:
            pixmap = QPixmap(artifact["path"])
            if not pixmap.isNull():
                self.preview.setPixmap(pixmap.scaled(self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                return
        self.preview.setPixmap(QPixmap())
        self.preview.setText("文本 / 审计文件；仅图片可续跑或发布")

    def _choose(self, action):
        # 「重新分析」不需要选图：分析产物丢了的时候，断点里的过程图救不了场
        if action == "rerun-analysis":
            self.action = action
            self.accept()
            return
        if self.selected_artifact and self.selected_artifact["image"]:
            self.action = action
            self.accept()


# ==================== Gemini 通道的「按相同参数重跑」 ====================
# Gemini 生图没有 GPT 工序断点可续跑，重跑＝按记录里保存的参数**新建**一条队列任务。


def available_prompt_types(record, result_json=None):
    """这条记录里还能重跑的提示词类型（有文本才算），顺序固定：优化 → 原始。"""
    record = record or {}
    data = result_json if isinstance(result_json, dict) else record.get("result_json")
    data = data if isinstance(data, dict) else {}
    texts = {
        "original": record.get("original_prompt") or data.get("original_english_description"),
        "refined": record.get("refined_prompt") or data.get("english_description"),
    }
    return [key for key in PROMPT_TYPE_ORDER if str(texts.get(key) or "").strip()]


def default_rerun_targets(record, available):
    """默认重跑哪几种提示词：沿用该任务上次实际生图用过的，其次优化提示词。"""
    available = [key for key in PROMPT_TYPE_ORDER if key in list(available or [])]
    if not available:
        return []
    params = (record or {}).get("generation_params")
    used = ([key for key in (params or {}).get("prompt_types") or [] if key in available]
            if isinstance(params, dict) else [])
    if used:
        return used
    return [key for key in PROMPT_TYPE_ORDER if key in available][:1]


class AnalysisRerunDialog(QDialog):
    """没有可拾取的过程图时：用这条记录的源图**重新分析**一次。

    分析产物丢了（只有断点、或者分析根本没跑完）时，用户手里还有源图 —— 不该被一句
    「没有可用的分析提示词」堵死。这里给他两条路：只重新分析，或分析完直接生图。

    生图通道由记录本身决定（`channel`）：gpt-image 记录沿用 gpt 通道，其余走 Gemini。
    """

    def __init__(self, summary_lines, channel="gemini", parent=None):
        super().__init__(parent)
        self.channel = "gpt-image" if str(channel).strip() == "gpt-image" else "gemini"
        self.mode = "analyze"
        self.gen_targets = []
        self.setWindowTitle("重新分析")
        self.resize(620, 400)
        layout = QVBoxLayout(self)
        tip = QLabel("这条记录没有可直接拾取的过程图（没有断点，或断点里没有可续跑/发布的图），"
                     "也没有可用的分析提示词。\n"
                     "源图还在，可以按相同画风 / 画幅**重新分析**一次；原记录与原图保留不动。")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        info = QLabel("\n".join(str(line) for line in (summary_lines or [])))
        info.setWordWrap(True)
        info.setStyleSheet("background: #202020; color: #e6e6e6; padding: 8px;")
        layout.addWidget(info)
        self.analyze_only = QRadioButton("只重新分析（生成分析产物与提示词）")
        self.analyze_and_generate = QRadioButton("重新分析后自动生图")
        self.analyze_only.setChecked(True)
        layout.addWidget(self.analyze_only)
        layout.addWidget(self.analyze_and_generate)
        prompt_row = QHBoxLayout()
        prompt_row.addSpacing(24)
        prompt_row.addWidget(QLabel("生图用："))
        self.prompt_boxes = {}
        for key in PROMPT_TYPE_ORDER:
            box = QCheckBox(PROMPT_TYPE_LABELS[key])
            box.setChecked(key == "refined")
            box.setEnabled(False)
            self.prompt_boxes[key] = box
            prompt_row.addWidget(box)
        prompt_row.addStretch(1)
        layout.addLayout(prompt_row)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #8a8a8a;")
        layout.addWidget(self.hint)
        layout.addStretch(1)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.start_button = QPushButton("开始重新分析")
        cancel_button = QPushButton("取消")
        buttons.addWidget(self.start_button)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        self.analyze_only.toggled.connect(self._on_mode_changed)
        self.start_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        self._on_mode_changed()

    def _on_mode_changed(self, *_args):
        generate = self.analyze_and_generate.isChecked()
        self.mode = "generate" if generate else "analyze"
        for box in self.prompt_boxes.values():
            box.setEnabled(generate)
        self.hint.setText(
            "重新分析完成后按当前画风继续生图，沿用原任务号（发布器仍能关联投稿 JSON）。"
            if generate else
            "只跑分析链，产出与手动点「开始分析」一致。"
        )

    def selected_targets(self):
        return [key for key, box in self.prompt_boxes.items() if box.isChecked()]

    def accept(self):
        """要求生图却一种提示词都没勾 → 当作没确认，避免建出一条空任务。"""
        if self.mode == "generate" and not self.selected_targets():
            return
        self.gen_targets = self.selected_targets() if self.mode == "generate" else []
        super().accept()


class GeminiRerunDialog(QDialog):
    """确认「按相同参数新建一条 Gemini 生图任务」，并挑选要重跑的提示词。"""

    def __init__(self, summary_lines, available, defaults=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("重跑 Gemini 生图")
        self.resize(560, 340)
        layout = QVBoxLayout(self)
        tip = QLabel("这条队列任务走的是 Gemini 生图通道，没有自动保存的 GPT 工序断点。\n"
                     "重跑会按下面的参数新建一条队列任务；原任务与原图保留不动。")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        info = QLabel("\n".join(str(line) for line in (summary_lines or [])))
        info.setWordWrap(True)
        info.setStyleSheet("background: #202020; color: #e6e6e6; padding: 8px;")
        layout.addWidget(info)
        layout.addWidget(QLabel("要重跑的提示词："))
        chosen = list(defaults or [])
        self.prompt_boxes = {}
        for key in PROMPT_TYPE_ORDER:
            if key not in list(available or []):
                continue
            box = QCheckBox(PROMPT_TYPE_LABELS[key])
            box.setChecked(key in chosen)
            self.prompt_boxes[key] = box
            layout.addWidget(box)
        layout.addStretch(1)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.start_button = QPushButton("新建队列任务并生图")
        cancel_button = QPushButton("取消")
        buttons.addWidget(self.start_button)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        self.start_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def selected_targets(self):
        return [key for key, box in self.prompt_boxes.items() if box.isChecked()]

    def accept(self):
        """一种提示词都没勾就当作没确认，避免建出一条空任务。"""
        if not self.selected_targets():
            return
        super().accept()
