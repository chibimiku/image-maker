import os
from contextlib import redirect_stdout, redirect_stderr
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLineEdit, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QMessageBox, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from utils.pic_cate import do_main as pic_cate_do_main, PicCateCancelledError
from utils.task_runtime import append_log_line, set_task_status


class SignalBridge:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text: str):
        if not text:
            return 0
        for row in str(text).splitlines():
            if row:
                self.signal.emit(row)
        return len(text)

    def flush(self):
        return


class PicCateWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str, str)

    def __init__(self, source_directory, target_directory, trimmed_directory, train_name):
        super().__init__()
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.trimmed_directory = trimmed_directory
        self.train_name = train_name
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def _should_cancel(self):
        return self._cancel_requested or self.isInterruptionRequested()

    def run(self):
        try:
            os.makedirs(self.target_directory, exist_ok=True)
            os.makedirs(self.trimmed_directory, exist_ok=True)
            cpu_count = os.cpu_count() or 1
            worker_count = max(1, cpu_count // 2)
            bridge = SignalBridge(self.log_signal)
            with redirect_stdout(bridge), redirect_stderr(bridge):
                self.log_signal.emit(f"检测到 CPU 核心数: {cpu_count}，切分线程数: {worker_count}")
                pic_cate_do_main(
                    self.source_directory,
                    self.target_directory,
                    self.trimmed_directory,
                    self.train_name,
                    worker_count=worker_count,
                    should_cancel=self._should_cancel
                )
            if self._should_cancel():
                self.finished_signal.emit("cancelled", "图片分类切分已取消")
            else:
                self.finished_signal.emit("success", "处理完成")
        except PicCateCancelledError:
            self.log_signal.emit("🛑 已收到取消请求，正在结束图片分类切分...")
            self.finished_signal.emit("cancelled", "图片分类切分已取消")
        except Exception as e:
            self.log_signal.emit(f"处理失败: {e}")
            self.finished_signal.emit("error", f"处理失败: {e}")


class PicCateWidget(QWidget):
    def __init__(self, save_values_callback):
        super().__init__()
        self.save_values_callback = save_values_callback
        self.worker = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        self.source_input = QLineEdit()
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_input)
        self.source_btn = QPushButton("选择")
        self.source_btn.clicked.connect(lambda: self.pick_directory(self.source_input, "选择原图目录"))
        source_row.addWidget(self.source_btn)
        form.addRow("原图目录:", source_row)

        self.target_input = QLineEdit()
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_input)
        self.target_btn = QPushButton("选择")
        self.target_btn.clicked.connect(lambda: self.pick_directory(self.target_input, "选择分类复制目录"))
        target_row.addWidget(self.target_btn)
        form.addRow("分类复制目录:", target_row)

        self.trimmed_input = QLineEdit()
        trimmed_row = QHBoxLayout()
        trimmed_row.addWidget(self.trimmed_input)
        self.trimmed_btn = QPushButton("选择")
        self.trimmed_btn.clicked.connect(lambda: self.pick_directory(self.trimmed_input, "选择裁剪训练输出目录"))
        trimmed_row.addWidget(self.trimmed_btn)
        form.addRow("裁剪训练输出目录:", trimmed_row)

        self.train_name_input = QLineEdit()
        form.addRow("训练集名称:", self.train_name_input)

        layout.addLayout(form)

        actions = QHBoxLayout()
        self.start_btn = QPushButton("开始切分")
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("取消任务")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        actions.addWidget(self.start_btn)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.clear_log_btn)
        layout.addLayout(actions)

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.setLayout(layout)

        self.source_input.textChanged.connect(self.on_values_changed)
        self.target_input.textChanged.connect(self.on_values_changed)
        self.trimmed_input.textChanged.connect(self.on_values_changed)
        self.train_name_input.textChanged.connect(self.on_values_changed)

    def pick_directory(self, line_edit, title):
        directory = QFileDialog.getExistingDirectory(self, title)
        if directory:
            line_edit.setText(directory)

    def get_values(self):
        return {
            "source_directory": self.source_input.text().strip(),
            "target_directory": self.target_input.text().strip(),
            "trimmed_directory": self.trimmed_input.text().strip(),
            "train_name": self.train_name_input.text().strip()
        }

    def set_values(self, values):
        values = values or {}
        for widget in [self.source_input, self.target_input, self.trimmed_input, self.train_name_input]:
            widget.blockSignals(True)
        self.source_input.setText(values.get("source_directory", ""))
        self.target_input.setText(values.get("target_directory", ""))
        self.trimmed_input.setText(values.get("trimmed_directory", ""))
        self.train_name_input.setText(values.get("train_name", ""))
        for widget in [self.source_input, self.target_input, self.trimmed_input, self.train_name_input]:
            widget.blockSignals(False)

    def on_values_changed(self):
        if callable(self.save_values_callback):
            self.save_values_callback(self.get_values())

    def log_msg(self, text):
        append_log_line(self.log_text, text)

    def set_task_state(self, state, detail=""):
        set_task_status(self.status_label, state, detail)

    def set_running_state(self, running, cancelling=False):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running and (not cancelling))
        self.source_input.setEnabled(not running)
        self.target_input.setEnabled(not running)
        self.trimmed_input.setEnabled(not running)
        self.train_name_input.setEnabled(not running)
        self.source_btn.setEnabled(not running)
        self.target_btn.setEnabled(not running)
        self.trimmed_btn.setEnabled(not running)

    def start_processing(self):
        values = self.get_values()
        if not values["source_directory"]:
            QMessageBox.warning(self, "提示", "请先填写原图目录")
            return
        if not os.path.isdir(values["source_directory"]):
            QMessageBox.warning(self, "提示", "原图目录不存在")
            return
        if not values["target_directory"]:
            QMessageBox.warning(self, "提示", "请先填写分类复制目录")
            return
        if not values["trimmed_directory"]:
            QMessageBox.warning(self, "提示", "请先填写裁剪训练输出目录")
            return
        if not values["train_name"]:
            QMessageBox.warning(self, "提示", "请先填写训练集名称")
            return

        self.on_values_changed()
        self.log_msg("开始处理...")
        self.set_running_state(True)
        self.set_task_state("running", "图片分类切分进行中")

        self.worker = PicCateWorkerThread(
            values["source_directory"],
            values["target_directory"],
            values["trimmed_directory"],
            values["train_name"]
        )
        self.worker.log_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(self.on_processing_finished)
        self.worker.start()

    def cancel_processing(self):
        if self.worker is None:
            self.log_msg("当前没有正在运行的图片分类切分任务。")
            return
        self.worker.request_cancel()
        self.set_running_state(True, cancelling=True)
        self.set_task_state("cancelling", "等待当前图片处理结束")
        self.log_msg("已请求取消图片分类切分，等待当前图片处理结束...")

    def on_processing_finished(self, status, message):
        worker = self.worker
        self.worker = None
        self.set_running_state(False)
        if worker is not None:
            worker.deleteLater()
        self.log_msg(message)
        if status == "success":
            self.set_task_state("success", message)
            QMessageBox.information(self, "完成", message)
        elif status == "cancelled":
            self.set_task_state("cancelled", message)
        else:
            self.set_task_state("error", message)
            QMessageBox.warning(self, "失败", message)
