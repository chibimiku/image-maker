import os
import datetime
from contextlib import redirect_stdout, redirect_stderr
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLineEdit, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from utils.pic_cate import do_main as pic_cate_do_main


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
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, source_directory, target_directory, trimmed_directory, train_name):
        super().__init__()
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.trimmed_directory = trimmed_directory
        self.train_name = train_name

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
                    worker_count=worker_count
                )
            self.finished_signal.emit(True, "处理完成")
        except Exception as e:
            self.log_signal.emit(f"处理失败: {e}")
            self.finished_signal.emit(False, f"处理失败: {e}")


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
        source_btn = QPushButton("选择")
        source_btn.clicked.connect(lambda: self.pick_directory(self.source_input, "选择原图目录"))
        source_row.addWidget(source_btn)
        form.addRow("原图目录:", source_row)

        self.target_input = QLineEdit()
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_input)
        target_btn = QPushButton("选择")
        target_btn.clicked.connect(lambda: self.pick_directory(self.target_input, "选择分类复制目录"))
        target_row.addWidget(target_btn)
        form.addRow("分类复制目录:", target_row)

        self.trimmed_input = QLineEdit()
        trimmed_row = QHBoxLayout()
        trimmed_row.addWidget(self.trimmed_input)
        trimmed_btn = QPushButton("选择")
        trimmed_btn.clicked.connect(lambda: self.pick_directory(self.trimmed_input, "选择裁剪训练输出目录"))
        trimmed_row.addWidget(trimmed_btn)
        form.addRow("裁剪训练输出目录:", trimmed_row)

        self.train_name_input = QLineEdit()
        form.addRow("训练集名称:", self.train_name_input)

        layout.addLayout(form)

        actions = QHBoxLayout()
        self.start_btn = QPushButton("开始切分")
        self.start_btn.clicked.connect(self.start_processing)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        actions.addWidget(self.start_btn)
        actions.addWidget(self.clear_log_btn)
        layout.addLayout(actions)

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
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {text}")
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def set_running_state(self, running):
        self.start_btn.setEnabled(not running)
        self.source_input.setEnabled(not running)
        self.target_input.setEnabled(not running)
        self.trimmed_input.setEnabled(not running)
        self.train_name_input.setEnabled(not running)

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

        self.worker = PicCateWorkerThread(
            values["source_directory"],
            values["target_directory"],
            values["trimmed_directory"],
            values["train_name"]
        )
        self.worker.log_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, success, message):
        self.set_running_state(False)
        self.log_msg(message)
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)
