import os
import io
import base64
from PIL import Image
from openai import OpenAI
from utils.task_runtime import append_log_line, set_task_status
from utils.prompt_loader import read_prompt_file, find_missing_prompt_files

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTextEdit, QListWidget, QListWidgetItem, QFileDialog, 
                             QMessageBox, QLabel, QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap

def compress_and_encode_image(image_source, max_dim=2048):
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source 

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_width, original_height = img.size

        if max(original_width, original_height) > max_dim:
            scaling_factor = max_dim / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=100)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return "image/jpeg", base64_string

    except Exception as e:
        print(f"处理图片时发生错误: {e}")
        return None, None

STYLE_ANALYZE_PROMPT_FILE = "style-analyzer-system.md"


def get_style_analyzer_missing_prompt_files():
    return find_missing_prompt_files([STYLE_ANALYZE_PROMPT_FILE])


class StyleAnalysisCancelledError(Exception):
    pass


class MultiImageWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(str, str)

    def __init__(self, image_paths, api_key, base_url, model_name):
        super().__init__()
        self.image_paths = image_paths
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def _check_cancel(self):
        if self._cancel_requested or self.isInterruptionRequested():
            raise StyleAnalysisCancelledError()

    def run(self):
        try:
            self._check_cancel()
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            self.log_signal.emit(f"初始化 API 客户端失败: {e}")
            self.finish_signal.emit("error", "")
            return

        self.log_signal.emit(f"🚀 开始处理 {len(self.image_paths)} 张图片，使用模型 [{self.model_name}]...")
        content_list = [{"type": "text", "text": read_prompt_file(STYLE_ANALYZE_PROMPT_FILE).strip()}]

        try:
            for index, path in enumerate(self.image_paths, start=1):
                self._check_cancel()
                self.log_signal.emit(f"[{index}/{len(self.image_paths)}] 正在压缩并编码图片: {os.path.basename(path)}")
                mime_type, base64_image = compress_and_encode_image(path)
                if base64_image:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                else:
                    self.log_signal.emit(f"跳过无法处理的图片: {os.path.basename(path)}")

            self._check_cancel()
            if len(content_list) <= 1:
                self.log_signal.emit("❌ 没有可用于分析的图片数据。")
                self.finish_signal.emit("error", "")
                return

            self.log_signal.emit("✅ 图片处理完成，正在向大模型发送分析请求，请耐心等待...")
            response = client.chat.completions.create(
                model=self.model_name, 
                messages=[{"role": "user", "content": content_list}],
                temperature=0.7,
                max_completion_tokens=4096
            )
            self._check_cancel()
            result_text = response.choices[0].message.content.strip()
            self.log_signal.emit("🎉 分析完成！")
            self.finish_signal.emit("success", result_text)
        except StyleAnalysisCancelledError:
            self.log_signal.emit("🛑 已取消共同画风提取任务。")
            self.finish_signal.emit("cancelled", "")
        except Exception as e:
            self.log_signal.emit(f"❌ API 请求发生错误: {e}")
            self.finish_signal.emit("error", "")

class ImageDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(100, 100))
        self.setResizeMode(QListWidget.Adjust)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self.add_image_item(file_path)

    def truncate_text(self, text, max_len=16):
        """截断过长的文件名，保留后缀名"""
        if len(text) <= max_len:
            return text
        name, ext = os.path.splitext(text)
        keep_len = max_len - len(ext) - 3
        if keep_len > 0:
            return name[:keep_len] + "..." + ext
        return text[:max_len] + "..."

    def add_image_item(self, file_path):
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == file_path:
                return
                
        item = QListWidgetItem(self)
        item.setData(Qt.UserRole, file_path) 
        
        filename = os.path.basename(file_path)
        item.setText(self.truncate_text(filename)) # 使用截断后的文件名
        item.setToolTip(filename) # 悬浮显示完整文件名
        
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            item.setIcon(QIcon(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self.addItem(item)


class StyleAnalyzerWidget(QWidget):
    def __init__(self, config_getter_func):
        super().__init__()
        self.get_config = config_getter_func
        self.thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        hint_label = QLabel("拖拽多张同画风图片至下方区域进行综合分析：")
        top_layout.addWidget(hint_label)
        
        self.add_btn = QPushButton("添加图片")
        self.add_btn.clicked.connect(self.browse_images)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_images)
        
        top_layout.addStretch()
        top_layout.addWidget(self.add_btn)
        top_layout.addWidget(self.clear_btn)
        layout.addLayout(top_layout)

        self.image_list = ImageDropListWidget()
        self.image_list.setMinimumHeight(150)
        layout.addWidget(self.image_list)

        action_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("✨ 开始提取共同画风 ✨")
        self.analyze_btn.setFixedHeight(40)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.cancel_btn = QPushButton("取消任务")
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setFixedHeight(40)
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.clear_log_btn)
        layout.addLayout(action_layout)

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("这里会显示任务日志...")
        self.log_text.setMaximumHeight(140)
        layout.addWidget(self.log_text)

        self.result_edit = QTextEdit()
        self.result_edit.setPlaceholderText("分析完成后的艺术风格记录文本将显示在这里，您可以直接复制和编辑...")
        layout.addWidget(self.result_edit)
        self.setLayout(layout)

    def browse_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        for f in files:
            self.image_list.add_image_item(f)

    def clear_images(self):
        self.image_list.clear()

    def log_msg(self, text):
        append_log_line(self.log_text, text)

    def set_task_state(self, state, detail=""):
        set_task_status(self.status_label, state, detail)

    def set_running_state(self, running, cancelling=False):
        self.analyze_btn.setEnabled(not running)
        self.add_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running and (not cancelling))

    def start_analysis(self):
        image_paths = [self.image_list.item(i).data(Qt.UserRole) for i in range(self.image_list.count())]
        missing_prompt_files = get_style_analyzer_missing_prompt_files()
        if missing_prompt_files:
            missing_text = "\n".join(missing_prompt_files)
            QMessageBox.warning(self, "缺少 Prompt 文件", f"以下 Prompt 文件不存在，请补齐后再执行：\n{missing_text}")
            self.log_msg(f"❌ 缺少 Prompt 文件，已中止共同画风提取：\n{missing_text}")
            return
        
        if len(image_paths) < 2:
            QMessageBox.information(self, "提示", "建议提供至少两张图片以提取共同画风。")
            if len(image_paths) == 0: return

        base_url, api_key, model_name = self.get_config()
        
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "请确保在【全局配置】中配置了文本分析的 API Key 和模型！")
            return

        self.set_running_state(True)
        self.set_task_state("running", f"准备分析 {len(image_paths)} 张图片")
        self.log_msg(f"开始共同画风提取，共 {len(image_paths)} 张图片。")
        self.result_edit.clear()
        
        self.thread = MultiImageWorkerThread(image_paths, api_key, base_url, model_name)
        self.thread.log_signal.connect(self.log_msg)
        self.thread.finish_signal.connect(self.on_analysis_finished)
        self.thread.start()

    def cancel_analysis(self):
        if self.thread is None:
            self.log_msg("当前没有正在运行的共同画风提取任务。")
            return
        self.thread.request_cancel()
        self.set_running_state(True, cancelling=True)
        self.set_task_state("cancelling", "等待当前步骤结束")
        self.log_msg("已请求取消共同画风提取任务，等待当前步骤结束...")

    def on_analysis_finished(self, status, text):
        thread = self.thread
        self.thread = None
        self.set_running_state(False)
        if thread is not None:
            thread.deleteLater()

        if status == "success" and text:
            self.result_edit.setPlainText(text)
            self.set_task_state("success", "共同画风提取完成")
            self.log_msg("共同画风提取完成。")
        else:
            if status == "cancelled":
                self.set_task_state("cancelled", "任务已取消")
            else:
                self.set_task_state("error", "请查看任务日志")
                QMessageBox.warning(self, "错误", "分析失败或未返回内容，请查看任务日志。")
