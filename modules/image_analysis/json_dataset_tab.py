import os
import json
import shutil
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QTextEdit, QFileDialog, QMessageBox, QLabel, QListWidget, QListWidgetItem, QAbstractItemView
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from utils.task_runtime import append_log_line, set_task_status


class JsonBatchCancelledError(Exception):
    pass


class JsonBatchWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str, str)

    def __init__(self, json_paths, output_directory, forced_tags_text, blacklist_tags_text):
        super().__init__()
        self.json_paths = json_paths
        self.output_directory = output_directory
        self.forced_tags_text = forced_tags_text
        self.blacklist_tags_text = blacklist_tags_text
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def _check_cancel(self):
        if self._cancel_requested or self.isInterruptionRequested():
            raise JsonBatchCancelledError()

    def run(self):
        success_count = 0
        fail_count = 0
        try:
            forced_tags = self._parse_tags_input(self.forced_tags_text)
            blacklist_tags = {tag.lower() for tag in self._parse_tags_input(self.blacklist_tags_text)}

            os.makedirs(self.output_directory, exist_ok=True)
            dataset_directories = {
                "long": os.path.join(self.output_directory, "long"),
                "short": os.path.join(self.output_directory, "short"),
                "booru": os.path.join(self.output_directory, "booru"),
                "mixed": os.path.join(self.output_directory, "mixed"),
            }
            for directory in dataset_directories.values():
                os.makedirs(directory, exist_ok=True)

            for idx, json_path in enumerate(self.json_paths, start=1):
                self._check_cancel()
                try:
                    self.log_signal.emit(f"[{idx}/{len(self.json_paths)}] 开始处理: {json_path}")
                    self._check_cancel()
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    source_image_path = str(data.get("source_image_path", "")).strip()
                    if not source_image_path or not os.path.isfile(source_image_path):
                        raise FileNotFoundError(f"source_image_path 不存在: {source_image_path}")

                    original_english_description = str(data.get("original_english_description", "")).strip()
                    short_description = str(data.get("short_description", "")).strip()
                    booru_tags_raw = data.get("booru-tags", [])

                    booru_tags = self._normalize_booru_tags(booru_tags_raw)
                    booru_tags = [tag for tag in booru_tags if tag.lower() not in blacklist_tags]
                    booru_tags = self._merge_unique_tags(forced_tags, booru_tags)
                    booru_text = ", ".join(booru_tags)
                    forced_tag_keys = {tag.lower() for tag in forced_tags}
                    booru_tags_without_forced = [tag for tag in booru_tags if tag.lower() not in forced_tag_keys]
                    mixed_parts = [
                        ", ".join(forced_tags),
                        short_description,
                        ", ".join(booru_tags_without_forced),
                    ]
                    mixed_text = ", ".join(part for part in mixed_parts if part)

                    src_filename = os.path.basename(source_image_path)
                    src_stem, src_ext = os.path.splitext(src_filename)
                    unique_stem = self._allocate_unique_stem(src_stem, src_ext, dataset_directories)

                    grouped_contents = {
                        "long": original_english_description,
                        "short": short_description,
                        "booru": booru_text,
                        "mixed": mixed_text,
                    }

                    for key, text in grouped_contents.items():
                        self._check_cancel()
                        image_dst = os.path.join(dataset_directories[key], f"{unique_stem}{src_ext}")
                        txt_dst = os.path.join(dataset_directories[key], f"{unique_stem}.txt")
                        shutil.copy2(source_image_path, image_dst)
                        with open(txt_dst, "w", encoding="utf-8") as wf:
                            wf.write(text)

                    success_count += 1
                    self.log_signal.emit(f"处理成功: {json_path}")
                except JsonBatchCancelledError:
                    raise
                except Exception as e:
                    fail_count += 1
                    self.log_signal.emit(f"处理失败: {json_path} | {e}")

            summary = f"处理完成，总数 {len(self.json_paths)}，成功 {success_count}，失败 {fail_count}"
            self.finished_signal.emit("success", summary)
        except JsonBatchCancelledError:
            summary = f"已取消，本次共处理 {len(self.json_paths)} 个文件，已完成 {success_count}，失败 {fail_count}"
            self.log_signal.emit("🛑 已收到取消请求，正在结束 JSON 数据集导出...")
            self.finished_signal.emit("cancelled", summary)
        except Exception as e:
            self.finished_signal.emit("error", f"处理失败: {e}")

    def _allocate_unique_stem(self, stem, ext, dataset_directories):
        index = 0
        while True:
            candidate_stem = stem if index == 0 else f"{stem}_{index}"
            conflict = False
            for directory in dataset_directories.values():
                image_candidate = os.path.join(directory, f"{candidate_stem}{ext}")
                txt_candidate = os.path.join(directory, f"{candidate_stem}.txt")
                if os.path.exists(image_candidate) or os.path.exists(txt_candidate):
                    conflict = True
                    break
            if not conflict:
                return candidate_stem
            index += 1

    def _normalize_booru_tags(self, booru_tags_raw):
        if isinstance(booru_tags_raw, list):
            return [str(tag).strip() for tag in booru_tags_raw if str(tag).strip()]
        if isinstance(booru_tags_raw, str):
            return self._parse_tags_input(booru_tags_raw)
        return []

    def _merge_unique_tags(self, base_tags, append_tags):
        merged = []
        seen = set()
        for tag in list(base_tags) + list(append_tags):
            cleaned = str(tag).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
        return merged

    def _parse_tags_input(self, text):
        raw_text = str(text or "").replace("，", ",")
        parts = [part.strip() for part in raw_text.split(",")]
        return [part for part in parts if part]


class JsonFileDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_json_files(paths)
        event.acceptProposedAction()

    def add_json_files(self, file_paths):
        existing = {self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())}
        added_count = 0
        for path in file_paths:
            norm_path = os.path.normpath(str(path))
            if not norm_path:
                continue
            if os.path.isfile(norm_path) and norm_path.lower().endswith(".json") and norm_path not in existing:
                item = QListWidgetItem(os.path.basename(norm_path))
                item.setToolTip(norm_path)
                item.setData(Qt.ItemDataRole.UserRole, norm_path)
                self.addItem(item)
                existing.add(norm_path)
                added_count += 1
        return added_count

    def get_all_paths(self):
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]


class JsonDatasetWidget(QWidget):
    quick_split_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_completed_output_dir = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        self.output_dir_input = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_input)
        self.select_output_btn = QPushButton("选择")
        self.select_output_btn.clicked.connect(self.choose_output_directory)
        output_row.addWidget(self.select_output_btn)
        form.addRow("输出目录:", output_row)

        self.forced_tags_input = QLineEdit()
        self.forced_tags_input.setPlaceholderText("以半角逗号分隔，如: masterpiece, best quality")
        form.addRow("强制添加标签:", self.forced_tags_input)

        self.blacklist_tags_input = QLineEdit()
        self.blacklist_tags_input.setPlaceholderText("以半角逗号分隔，如: lowres, bad anatomy")
        form.addRow("标签黑名单:", self.blacklist_tags_input)

        layout.addLayout(form)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("JSON 文件列表（支持拖拽多个文件）:"))
        header_layout.addStretch()
        self.add_files_btn = QPushButton("添加文件")
        self.add_files_btn.clicked.connect(self.browse_json_files)
        self.remove_selected_btn = QPushButton("移除选中")
        self.remove_selected_btn.clicked.connect(self.remove_selected_files)
        self.clear_files_btn = QPushButton("清空列表")
        self.clear_files_btn.clicked.connect(self.clear_files)
        header_layout.addWidget(self.add_files_btn)
        header_layout.addWidget(self.remove_selected_btn)
        header_layout.addWidget(self.clear_files_btn)
        layout.addLayout(header_layout)

        self.file_list = JsonFileDropListWidget()
        self.file_list.setMinimumHeight(180)
        layout.addWidget(self.file_list)

        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("取消任务")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.open_output_btn = QPushButton("导出完成后打开输出目录")
        self.open_output_btn.clicked.connect(self.open_output_directory)
        self.quick_split_btn = QPushButton("导出完成后用于图片分类切分")
        self.quick_split_btn.clicked.connect(self.quick_jump_to_pic_cate)
        self.quick_split_btn.setEnabled(False)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.open_output_btn)
        action_layout.addWidget(self.quick_split_btn)
        action_layout.addWidget(self.clear_log_btn)
        layout.addLayout(action_layout)

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    def choose_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.output_dir_input.setText(directory)

    def browse_json_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择 JSON 文件", "", "JSON Files (*.json)")
        added_count = self.file_list.add_json_files(files)
        if added_count:
            self.log_msg(f"已添加 {added_count} 个 JSON 文件")

    def remove_selected_files(self):
        selected_items = self.file_list.selectedItems()
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def clear_files(self):
        self.file_list.clear()

    def log_msg(self, text):
        append_log_line(self.log_text, text)

    def set_task_state(self, state, detail=""):
        set_task_status(self.status_label, state, detail)

    def set_running_state(self, running, cancelling=False):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running and (not cancelling))
        self.open_output_btn.setEnabled(not running)
        self.add_files_btn.setEnabled(not running)
        self.remove_selected_btn.setEnabled(not running)
        self.clear_files_btn.setEnabled(not running)
        self.select_output_btn.setEnabled(not running)
        self.output_dir_input.setEnabled(not running)
        self.forced_tags_input.setEnabled(not running)
        self.blacklist_tags_input.setEnabled(not running)
        self.quick_split_btn.setEnabled((not running) and bool(self.last_completed_output_dir))

    def open_output_directory(self):
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "提示", "输出目录不存在")
            return
        os.startfile(output_dir)

    def start_processing(self):
        json_paths = self.file_list.get_all_paths()
        output_dir = self.output_dir_input.text().strip()

        if not json_paths:
            QMessageBox.warning(self, "提示", "请先添加至少一个 JSON 文件")
            return
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return

        os.makedirs(output_dir, exist_ok=True)
        self.last_completed_output_dir = ""
        self.quick_split_btn.setEnabled(False)
        self.set_running_state(True)
        self.set_task_state("running", f"准备导出 {len(json_paths)} 个 JSON")
        self.log_msg(f"开始处理，共 {len(json_paths)} 个文件")

        self.worker = JsonBatchWorkerThread(
            json_paths=json_paths,
            output_directory=output_dir,
            forced_tags_text=self.forced_tags_input.text().strip(),
            blacklist_tags_text=self.blacklist_tags_input.text().strip(),
        )
        self.worker.log_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(self.on_processing_finished)
        self.worker.start()

    def cancel_processing(self):
        if self.worker is None:
            self.log_msg("当前没有正在运行的 JSON 数据集导出任务。")
            return
        self.worker.request_cancel()
        self.set_running_state(True, cancelling=True)
        self.set_task_state("cancelling", "等待当前文件处理完成")
        self.log_msg("已请求取消 JSON 数据集导出，等待当前文件处理完成...")

    def on_processing_finished(self, status, message):
        worker = self.worker
        self.worker = None
        self.set_running_state(False)
        if worker is not None:
            worker.deleteLater()
        self.log_msg(message)
        if status == "success":
            self.set_task_state("success", message)
            output_dir = self.output_dir_input.text().strip()
            if output_dir and os.path.isdir(output_dir):
                self.last_completed_output_dir = output_dir
                self.quick_split_btn.setEnabled(True)
            QMessageBox.information(self, "完成", message)
        elif status == "cancelled":
            self.set_task_state("cancelled", message)
        else:
            self.set_task_state("error", message)
            QMessageBox.warning(self, "失败", message)

    def prefill_for_batch(self, json_paths, output_dir):
        valid_paths = [path for path in (json_paths or []) if os.path.isfile(path) and str(path).lower().endswith(".json")]
        self.output_dir_input.setText(output_dir or "")
        self.file_list.clear()
        added_count = self.file_list.add_json_files(valid_paths)
        self.last_completed_output_dir = ""
        self.quick_split_btn.setEnabled(False)
        self.set_task_state("idle", f"已载入 {added_count} 个 JSON")
        self.log_msg(f"已接收批量分析结果，共 {added_count} 个 JSON，输出目录已设置")

    def quick_jump_to_pic_cate(self):
        if not self.last_completed_output_dir:
            QMessageBox.warning(self, "提示", "请先完成 JSON数据集导出")
            return
        self.quick_split_requested.emit(self.last_completed_output_dir)
