import os
import io
import base64
from PIL import Image
from openai import OpenAI

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

STYLE_ANALYZE_PROMPT = """
You are an expert art style analyzer and prompt engineer. 
Analyze the provided images and identify their common artistic characteristics.
Generate a highly detailed, approximately 500-word English text documenting this specific art style.

The output must be formatted as direct instructions for a generative AI model, starting with a persona setup (e.g., 'You are a generative model specialized in creating images in...').

Include extremely detailed descriptions of the following elements based ONLY on the commonalities found in the provided images:
1. Overall artistic style, atmosphere, and visual impact.
2. Facial features, expressions, proportions, and rendering style.
3. Eyes (shape, multi-layered iris details, specific highlight patterns like crystalline or moist looks, upper and lower eyelashes).
4. Hair (texture, strands, lighting, coloring, specular highlights, gradients).
5. Clothing, shoes, and socks (textures, materials, lace, leather, reflections, specific recurring details).
6. Line art style (e.g., delicate, bold, colored lines), coloring techniques, and lighting/shadow rendering (e.g., soft glows, sharp contrasts).

Output ONLY the raw English text block, ready to be copied and pasted. 
Do NOT use Markdown formatting (do not wrap in ```text or ```). 
Do NOT include any conversational filler, introductory, or concluding remarks. Just output the prompt text itself.
"""

class MultiImageWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(str)

    def __init__(self, image_paths, api_key, base_url, model_name):
        super().__init__()
        self.image_paths = image_paths
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            self.log_signal.emit(f"初始化 API 客户端失败: {e}")
            self.finish_signal.emit("")
            return

        self.log_signal.emit(f"🚀 开始处理 {len(self.image_paths)} 张图片，使用模型 [{self.model_name}]...")
        
        content_list = [{"type": "text", "text": STYLE_ANALYZE_PROMPT}]
        
        for path in self.image_paths:
            self.log_signal.emit(f"正在压缩并编码图片: {os.path.basename(path)}")
            mime_type, base64_image = compress_and_encode_image(path)
            if base64_image:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                })

        self.log_signal.emit("✅ 图片处理完成，正在向大模型发送分析请求，请耐心等待...")

        try:
            response = client.chat.completions.create(
                model=self.model_name, 
                messages=[{"role": "user", "content": content_list}],
                temperature=0.7,
                max_completion_tokens=4096
            )
            result_text = response.choices[0].message.content.strip()
            self.log_signal.emit("🎉 分析完成！")
            self.finish_signal.emit(result_text)
        except Exception as e:
            self.log_signal.emit(f"❌ API 请求发生错误: {e}")
            self.finish_signal.emit("")

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

        self.analyze_btn = QPushButton("✨ 开始提取共同画风 ✨")
        self.analyze_btn.setFixedHeight(40)
        self.analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_btn)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

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

    def update_status(self, text):
        self.status_label.setText(text)

    def start_analysis(self):
        image_paths = [self.image_list.item(i).data(Qt.UserRole) for i in range(self.image_list.count())]
        
        if len(image_paths) < 2:
            QMessageBox.information(self, "提示", "建议提供至少两张图片以提取共同画风。")
            if len(image_paths) == 0: return

        base_url, api_key, model_name = self.get_config()
        
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "请确保在【全局配置】中配置了文本分析的 API Key 和模型！")
            return

        self.analyze_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.result_edit.clear()
        
        self.thread = MultiImageWorkerThread(image_paths, api_key, base_url, model_name)
        self.thread.log_signal.connect(self.update_status)
        self.thread.finish_signal.connect(self.on_analysis_finished)
        self.thread.start()

    def on_analysis_finished(self, text):
        self.analyze_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        
        if text:
            self.result_edit.setPlainText(text)
        else:
            QMessageBox.warning(self, "错误", "分析失败或未返回内容，请查看控制台日志。")