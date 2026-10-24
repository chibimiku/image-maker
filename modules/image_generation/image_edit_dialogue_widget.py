"""Qt frontend for conversational batch edits; all API work stays in a QThread."""
import copy
import html
import os
import uuid
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTextEdit, QPlainTextEdit, QComboBox, QListWidget, QListWidgetItem,
                             QFileDialog, QSplitter, QSizePolicy, QSpinBox)

from modules.others.api_backend import get_api_config
from modules.image_generation import image_edit_dialogue as runtime


class DialogueWorker(QThread):
    event = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, session, path, text_config, image_config, user, ids,
                 retry_id=None, preferred_versions=None, parent=None):
        super().__init__(parent)
        self.runner = runtime.DialogueRunner(session, path, text_config, image_config,
                                             self.event.emit, self.isInterruptionRequested)
        self.user, self.ids, self.retry_id = user, ids, retry_id
        self.preferred_versions = preferred_versions

    def run(self):
        try:
            self.runner.run(self.user, self.ids, self.retry_id, self.preferred_versions)
        except Exception as exc:
            self.failed.emit(str(exc))


class DialogueEditPanel(QWidget):
    busy_changed = pyqtSignal(bool)

    def __init__(self, owner, session_path=None):
        super().__init__(owner)
        self.owner = owner
        self.worker = None
        self.busy = False
        self.path = session_path or runtime.default_session_path()
        self.session = runtime.new_session()
        self.load_error = ""
        self.active_path = self.path
        try:
            self.session = runtime.load_session(self.path)
        except Exception as exc:
            # Preserve corrupt/unsupported files; the next turn gets a new path.
            self.load_error = f"历史记录未加载：{exc}；原文件保留。"
            self.path = str(Path(self.path).with_name(f"recovered-{uuid.uuid4().hex[:8]}.json"))
            self.active_path = self.path
        self._build_ui()
        self.render_session()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        config = get_api_config(api_type="aigc2d")
        if self.session["turns"]:
            previous = self.session["turns"][-1].get("models", {})
            config = {**config, **{target: previous[source] for source, target in
                      (("edit", "model"), ("resolution", "resolution"), ("timeout", "timeout")) if previous.get(source)}}
        self.model_combo.addItem(str(config.get("model") or ""))
        self.model_combo.setToolTip("使用 aigc2d 节点的 Gemini 模型；可手输已开通的 Gemini 图片模型名")
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1K", "2K", "4K"])
        self.resolution_combo.setCurrentText(str(config.get("resolution") or "2K"))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 900)
        self.timeout_spin.setValue(int(config.get("timeout") or 360))
        self.timeout_spin.setSuffix("s")
        self.timeout_spin.setToolTip("单张 Gemini 请求超时；Pro 可按需要增加到 360 秒")
        self.open_btn = QPushButton("打开对话")
        self.new_btn = QPushButton("新对话")
        self.open_btn.clicked.connect(self.open_session)
        self.new_btn.clicked.connect(self.new_conversation)
        row.addWidget(QLabel("Gemini"))
        row.addWidget(self.model_combo, 1)
        row.addWidget(self.resolution_combo)
        row.addWidget(self.timeout_spin)
        row.addWidget(self.open_btn)
        row.addWidget(self.new_btn)
        layout.addLayout(row)
        self.scope_label = QLabel()
        self.scope_label.setWordWrap(True)
        layout.addWidget(self.scope_label)
        split = QSplitter(Qt.Orientation.Vertical)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(90)
        split.addWidget(self.transcript)
        results = QWidget()
        results_layout = QHBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self.version_list = QListWidget()
        self.version_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.version_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.version_list.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.version_list.currentItemChanged.connect(self.show_version)
        self.version_list.itemDoubleClicked.connect(self.open_version)
        self.preview = QLabel("选择版本查看预览；双击打开图片")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(100, 80)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        results_layout.addWidget(self.version_list, 1)
        results_layout.addWidget(self.preview, 1)
        split.addWidget(results)
        split.setSizes([240, 150])
        layout.addWidget(split, 1)
        input_row = QHBoxLayout()
        self.base_combo = QComboBox()
        self.base_combo.addItems(["按对话选择输入（默认最新）", "从原图编辑", "从右侧选中版本编辑"])
        self.base_combo.setToolTip("选择具体版本会将本轮限制为右侧选中的一张图")
        self.retry_btn = QPushButton("重试本轮失败项")
        self.retry_btn.clicked.connect(self.retry_failed)
        input_row.addWidget(self.base_combo, 1)
        input_row.addWidget(self.retry_btn)
        layout.addLayout(input_row)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("例如：选穿黑裙的两张，让成年角色更有魅力，保留人物和画风。\n后续可说：第二张保留原来的裙子，只调整表情。")
        self.input.setMinimumHeight(70)
        self.input.setMaximumHeight(130)
        layout.addWidget(self.input)
        actions = QHBoxLayout()
        self.send_btn = QPushButton("发送并编辑")
        self.send_btn.setToolTip("理解要求并选图 → 逐张看图制定指令 → Gemini 编辑 → 看图复核。每轮最多 24 张；失败不自动重试。")
        self.stop_btn = QPushButton("停止后续任务")
        self.send_btn.clicked.connect(self.send)
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        actions.addWidget(self.send_btn, 1)
        actions.addWidget(self.stop_btn)
        layout.addLayout(actions)
        self.owner.image_list.itemSelectionChanged.connect(self.refresh_scope)
        self.owner.image_list.itemChanged.connect(self.refresh_scope)
        self.owner.image_list.model().rowsInserted.connect(self.refresh_scope)
        self.owner.image_list.model().rowsRemoved.connect(self.refresh_scope)
        self.refresh_scope()

    def selected_paths(self):
        checked, highlighted, all_paths = [], [], []
        for index in range(self.owner.image_list.count()):
            item = self.owner.image_list.item(index)
            path = item.data(Qt.ItemDataRole.UserRole)
            all_paths.append(path)
            if item.checkState() == Qt.CheckState.Checked:
                checked.append(path)
            if item.isSelected():
                highlighted.append(path)
        return checked or highlighted or all_paths

    def refresh_scope(self, *args):
        count = len(self.selected_paths())
        self.scope_label.setText(f"本轮范围：{count} 张（勾选优先，其次高亮；均未选择时为列表全部）。可在对话中继续筛选。")

    def _configs(self):
        url, key, model = self.owner.config_getter_func()
        text = {"base_url": url, "api_key": key, "model": model}
        image = get_api_config(api_type="aigc2d")
        image.update(model=self.model_combo.currentText().strip(), resolution=self.resolution_combo.currentText(),
                     timeout=self.timeout_spin.value())
        return text, image

    def send(self):
        if self.busy or self.owner.processing:
            return
        user = self.input.toPlainText().strip()
        if not user:
            self.notice("请输入编辑要求。")
            return
        try:
            paths = self.selected_paths()
            if not paths or len(paths) > runtime.MAX_IMAGES:
                raise ValueError(f"请添加图片，并将本轮范围限定在 1–{runtime.MAX_IMAGES} 张。")
            all_paths = [self.owner.image_list.item(i).data(Qt.ItemDataRole.UserRole)
                         for i in range(self.owner.image_list.count())]
            all_ids = runtime.register_images(self.session, all_paths)
            for index, image_id in enumerate(all_ids):
                runtime.image_by_id(self.session, image_id)["display_index"] = index + 1
            ids = runtime.register_images(self.session, paths)
            preferred = {}
            if self.base_combo.currentIndex() == 1:
                preferred = {image_id: 0 for image_id in ids}
            elif self.base_combo.currentIndex() == 2:
                selected = self.version_list.currentItem()
                if not selected:
                    raise ValueError("请先在版本列表选择一项。")
                image_id, version = selected.data(Qt.ItemDataRole.UserRole)
                if image_id not in ids:
                    raise ValueError("选中版本不在左侧本轮图片范围内。")
                ids, preferred = [image_id], {image_id: version}
            self._start(user, ids, preferred=preferred)
        except Exception as exc:
            self.notice(str(exc))

    def _start(self, user, ids, retry_id=None, preferred=None):
        text, image = self._configs()
        if not all(text.values()):
            raise ValueError("请先配置支持看图的文字分析 API。")
        if not image.get("api_key") or not image["model"].lower().startswith("gemini"):
            raise ValueError("请先配置 aigc2d 密钥，并在此处输入 Gemini 图片模型名。")
        runtime.save_session(self.session, self.path)
        self.busy = True
        self.set_controls_busy(True)
        self.busy_changed.emit(True)
        self.worker = DialogueWorker(self.session, self.path, text, image, user, ids,
                                     retry_id, preferred, self)
        self.worker.event.connect(self.handle_event)
        self.worker.failed.connect(self.notice)
        self.worker.finished.connect(self.finished)
        self.worker.start()
        if retry_id is None:
            self.input.clear()

    def retry_failed(self):
        if self.busy or self.owner.processing or not self.session["turns"]:
            return
        turn = self.session["turns"][-1]
        ids = [task["image_id"] for task in turn.get("tasks", [])
               if task["status"] in {"error", "interrupted", "cancelled"}]
        try:
            if not ids:
                raise ValueError("本轮没有可重试的失败项。分析选图失败时请重新发送要求。")
            self._start(turn["user"], ids, retry_id=turn["id"])
        except Exception as exc:
            self.notice(str(exc))

    def stop(self):
        if self.worker:
            self.worker.requestInterruption()
            self.stop_btn.setEnabled(False)
            self.notice("已请求停止。当前已发送请求会等待返回并保存，之后不再处理其他图片。")

    def set_controls_busy(self, busy):
        for control in (self.input, self.send_btn, self.new_btn, self.open_btn, self.model_combo,
                        self.resolution_combo, self.timeout_spin, self.base_combo):
            control.setEnabled(not busy)
        self.stop_btn.setEnabled(self.busy)
        self.retry_btn.setEnabled(not busy and bool(self.session["turns"]) and any(
            task["status"] in {"error", "interrupted", "cancelled"}
            for task in self.session["turns"][-1].get("tasks", [])))

    def handle_event(self, event):
        if event["type"] == "session":
            self.session = event["session"]
            self.render_session()

    def notice(self, text):
        self.transcript.append(f"<p>{html.escape(text)}</p>")

    def finished(self):
        worker, self.worker = self.worker, None
        self.busy = False
        self.set_controls_busy(False)
        self.busy_changed.emit(False)
        if worker:
            worker.deleteLater()

    def render_session(self):
        paragraphs = []
        labels = {"queued": "等待", "planning": "看图分析中", "generating": "Gemini 编辑中",
                  "success": "已返回图片", "error": "失败", "skip": "跳过", "clarify": "需澄清",
                  "cancelled": "已停止", "interrupted": "已中断"}
        for turn in self.session["turns"]:
            paragraphs.append(f"<p><b>你：</b>{html.escape(turn['user']).replace(chr(10), '<br>')}</p>")
            paragraphs.append(f"<p><b>助手：</b>{html.escape(turn.get('message', ''))}</p>")
            for task in turn.get("tasks", []):
                description = task.get("error") or task.get("summary") or task["instruction"]
                if task.get("review_status") == "reviewing":
                    description += "（正在看图复核）"
                elif task.get("review_status") in {"interrupted", "cancelled", "pending"}:
                    description += "（图片已保存，复核未完成）"
                paragraphs.append(f"<p>#{task['image_id']} · v{task['version']} · "
                                  f"{labels.get(task['status'], task['status'])}：{html.escape(description).replace(chr(10), '<br>')}</p>")
        self.transcript.setHtml("".join(paragraphs))
        by_path = {os.path.normcase(row["original"]): row for row in self.session["images"]}
        for index in range(self.owner.image_list.count()):
            item = self.owner.image_list.item(index)
            path = item.data(Qt.ItemDataRole.UserRole)
            row = by_path.get(os.path.normcase(os.path.abspath(path)))
            if row:
                item.setText(f"#{row['id']} · {os.path.basename(path)}")
                item.setToolTip(os.path.abspath(path))
            else:
                item.setText(os.path.basename(path))
        if self.load_error:
            self.notice(self.load_error)
        self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())
        previous = self.version_list.currentItem()
        previous_data = previous.data(Qt.ItemDataRole.UserRole) if previous else None
        self.version_list.clear()
        for row in self.session["images"]:
            for index, version in enumerate(row["versions"]):
                review = version.get("review", {})
                marker = {"pass": "复核通过", "fail": "需查看", "uncertain": "复核不确定"}.get(review.get("verdict"), "")
                item = QListWidgetItem(f"#{row['id']} · {'原图' if index == 0 else 'v' + str(index)} · {marker} · {os.path.basename(row['original'])}")
                item.setData(Qt.ItemDataRole.UserRole, (row["id"], index))
                item.setToolTip(version["path"] + "\n" + review.get("summary", "") + "\n" + version.get("prompt", ""))
                self.version_list.addItem(item)
                if (row["id"], index) == previous_data:
                    self.version_list.setCurrentItem(item)
        if not self.busy:
            self.set_controls_busy(False)

    def show_version(self, item, *args):
        if not item:
            self.preview.clear()
            return
        image_id, index = item.data(Qt.ItemDataRole.UserRole)
        version = runtime.image_by_id(self.session, image_id)["versions"][index]
        pixmap = QPixmap(version["path"])
        self.preview.setPixmap(pixmap.scaled(max(100, self.preview.width()), max(80, self.preview.height()),
                                            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.preview.setToolTip(version["path"])

    def open_version(self, item):
        image_id, index = item.data(Qt.ItemDataRole.UserRole)
        QDesktopServices.openUrl(QUrl.fromLocalFile(runtime.image_by_id(self.session, image_id)["versions"][index]["path"]))

    def open_session(self):
        if self.busy or self.owner.processing:
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开图片编辑对话", str(Path(self.path).parent), "JSON (*.json)")
        if not path:
            return
        try:
            session = runtime.load_session(path)
            self.archive_current()
            self.session, self.path = session, self.active_path
            runtime.save_session(self.session, self.path)
            self.load_error = ""
            self.base_combo.setCurrentIndex(0)
            self.owner._add_images_from_paths([row["original"] for row in session["images"]])
            self.render_session()
        except Exception as exc:
            self.notice(f"打开失败：{exc}")

    def new_conversation(self):
        if self.busy or self.owner.processing:
            return
        self.archive_current()
        self.session = runtime.new_session()
        self.path = self.active_path
        runtime.save_session(self.session, self.path)
        self.load_error = ""
        self.base_combo.setCurrentIndex(0)
        self.render_session()

    def archive_current(self):
        if self.session["turns"] or self.session["images"]:
            archive = str(Path(self.path).with_name(f"conversation-{self.session['id'][:12]}.json"))
            runtime.save_session(self.session, archive)

    def request_shutdown(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return False
        return True
