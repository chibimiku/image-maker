import sys
import json
import os
import time
import traceback
import threading
import shutil
import logging
import base64
import re
import requests
from io import BytesIO
from PIL import Image

# 必须安装这两个库: pip install google-genai openai
from google import genai
from google.genai import types
import openai

# 找到这一堆导入，把 QSizePolicy 加进去
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QTabWidget, 
                             QScrollArea, QComboBox, QLineEdit, QCheckBox, 
                             QFileDialog, QInputDialog, QMessageBox, QGridLayout,
                             QDialog, QFormLayout, QListWidget, QListWidgetItem, QAbstractItemView,
                             QSizePolicy) # <--- 新增导入这个
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon

# ================== 初始化目录与日志 ==================
os.makedirs("log", exist_ok=True)
os.makedirs("cache", exist_ok=True)
STATE_FILE = os.path.join("cache", "last_state.json")
CONFIG_PATH = "config-image.json"

log_filename = os.path.join("log", f"run_{time.strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8')]
)

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        if text.strip():
            logging.info(text.strip())
        self.textWritten.emit(str(text))
    def flush(self):
        pass

def compress_image_if_needed(image_path, max_dim=2048, max_size_mb=8):
    img = Image.open(image_path)
    w, h = img.size
    if w > max_dim or h > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    if img.mode != 'RGB' and img.mode != 'RGBA':
        img = img.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    size_mb = buffer.tell() / (1024 * 1024)
    if size_mb > max_size_mb:
        buffer = BytesIO()
        img.convert('RGB').save(buffer, format="JPEG", quality=85)
        img = Image.open(buffer)
    return img

# ================== 弹窗组件 ==================

class APIConfigDialog(QDialog):
    """API 配置与模型获取的可视化界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑 API 配置")
        self.resize(450, 300)
        self.initUI()
        self.load_config()

    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.api_type_combo = QComboBox()
        self.api_type_combo.addItems(["openai", "gemini", "openai_proxy"])
        form_layout.addRow("API 类型:", self.api_type_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        form_layout.addRow("API Key:", self.api_key_input)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("留空则使用官方默认地址")
        form_layout.addRow("Base URL:", self.base_url_input)

        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setSizePolicy(self.model_combo.sizePolicy().Expanding, self.model_combo.sizePolicy().Fixed)
        self.fetch_btn = QPushButton("获取列表")
        self.fetch_btn.clicked.connect(self.fetch_models)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.fetch_btn)
        form_layout.addRow("模型名称:", model_layout)

        layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_config)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.api_type_combo.setCurrentText(data.get("api_type", "openai"))
                self.api_key_input.setText(data.get("api_key", ""))
                self.base_url_input.setText(data.get("base_url", ""))
                current_model = data.get("model", "dall-e-3")
                self.model_combo.addItem(current_model)
                self.model_combo.setCurrentText(current_model)
            except Exception as e:
                print(f"加载配置失败: {e}")

    def fetch_models(self):
        api_type = self.api_type_combo.currentText().strip()
        api_key = self.api_key_input.text().strip()
        base_url = self.base_url_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入 API Key")
            return

        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("获取中...")
        QApplication.processEvents()

        try:
            model_names = []
            if api_type == "openai":
                client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)
                models = client.models.list()
                model_names = [m.id for m in models.data]
            else:
                client_options = {"api_key": api_key}
                if base_url:
                    client_options["http_options"] = {"base_url": base_url}
                client = genai.Client(**client_options)
                models = client.models.list()
                model_names = [m.name for m in models]
            
            if not model_names:
                if api_type == "openai":
                    model_names = ["dall-e-3", "dall-e-2"]
                else:
                    model_names = ["gemini-3.1-flash-image", "gemini-2.5-pro"]

            self.model_combo.clear()
            self.model_combo.addItems(model_names)
            
            # 尝试优先选中带 image 或 dall 的模型
            for name in model_names:
                if "image" in name.lower() or "dall" in name.lower():
                    self.model_combo.setCurrentText(name)
                    break
                    
            QMessageBox.information(self, "成功", f"获取 {api_type} 模型列表成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取失败:\n{e}")
        finally:
            self.fetch_btn.setEnabled(True)
            self.fetch_btn.setText("获取列表")

    def save_config(self):
        data = {
            "api_type": self.api_type_combo.currentText().strip(),
            "api_key": self.api_key_input.text().strip(),
            "base_url": self.base_url_input.text().strip(),
            "model": self.model_combo.currentText().strip()
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败:\n{e}")

class HistoryManagerDialog(QDialog):
    # 此处保持不变，用于管理缓存的历史记录
    def __init__(self, category_name, history_dir, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.history_dir = history_dir
        self.selected_path = None
        self.setWindowTitle(f"历史记录管理 - {category_name}")
        self.resize(600, 400)
        self.initUI()
        self.load_images()

    def initUI(self):
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清空所有记录")
        clear_btn.setStyleSheet("color: red;")
        clear_btn.clicked.connect(self.clear_all)
        delete_btn = QPushButton("删除选中图片")
        delete_btn.clicked.connect(self.delete_selected)
        select_btn = QPushButton("选择并调用")
        select_btn.setStyleSheet("font-weight: bold;")
        select_btn.clicked.connect(self.select_and_close)

        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(select_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_images(self):
        self.list_widget.clear()
        if not os.path.exists(self.history_dir): return
        for filename in os.listdir(self.history_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                path = os.path.join(self.history_dir, filename)
                item = QListWidgetItem(QIcon(path), filename)
                item.setData(Qt.UserRole, path)
                self.list_widget.addItem(item)

    def on_item_double_clicked(self, item):
        self.selected_path = item.data(Qt.UserRole)
        self.accept()
    def select_and_close(self):
        item = self.list_widget.currentItem()
        if item:
            self.selected_path = item.data(Qt.UserRole)
            self.accept()
        else:
            QMessageBox.warning(self, "提示", "请先选中一张图片。")
    def delete_selected(self):
        item = self.list_widget.currentItem()
        if item:
            try:
                os.remove(item.data(Qt.UserRole))
                self.list_widget.takeItem(self.list_widget.row(item))
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除失败: {e}")
    def clear_all(self):
        if QMessageBox.question(self, '确认', '确定清空所有图片吗？', QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            for filename in os.listdir(self.history_dir):
                os.remove(os.path.join(self.history_dir, filename))
            self.list_widget.clear()

# ================== UI 核心组件 ==================

class DropImageLabel(QLabel):
    imageChanged = pyqtSignal(str)
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px dashed #aaa; background-color: #2d2d2d; color: #aaa; font-size: 11px;")
        
        # 【修改点 1】使用 setFixedSize 彻底锁死物理尺寸，不给它任何撑大布局的机会
        self.setFixedSize(80, 80)
        
        self.setAcceptDrops(True)
        # 因为我们自己手动缩放了，这里关掉避免二次拉伸模糊
        self.setScaledContents(False) 
        self.current_image_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border: 2px dashed #4CAF50; background-color: #e8f5e9; color: #4CAF50;")
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self._reset_style()
        
    def dropEvent(self, event):
        self._reset_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self.set_image(file_path)
                
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp)")
            if file_path:
                self.set_image(file_path)
                
    def set_image(self, file_path):
        if not file_path or not os.path.exists(file_path):
            self.clear_image()
            return
        self.current_image_path = file_path
        
        # 【修改点 2】加载图片后，直接在内存中把它压缩成 80x80 的缩略图，再交给 UI 渲染
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        
        self.imageChanged.emit(file_path)
        
    def clear_image(self):
        self.current_image_path = None
        self.clear()
        self.setText("点击/拖拽\n添加")
        self._reset_style()
        
    def _reset_style(self):
        self.setStyleSheet("border: 1px solid #555;" if self.current_image_path else "border: 1px dashed #aaa; background-color: #2d2d2d; color: #aaa; font-size: 11px;")

class ImageSlotWidget(QWidget):
    def __init__(self, category_name):
        super().__init__()
        self.category_name = category_name
        self.history_dir = os.path.join("cache", "history", category_name)
        os.makedirs(self.history_dir, exist_ok=True)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addWidget(QLabel(f"<b style='font-size:12px;'>{category_name}</b>"))
        title_layout.addStretch()
        
        history_btn = QPushButton("历史")
        history_btn.setFixedSize(36, 20)
        history_btn.setStyleSheet("font-size: 10px; padding: 0px;")
        history_btn.clicked.connect(self.open_history_manager)
        
        clear_btn = QPushButton("x")
        clear_btn.setFixedSize(16, 20)
        clear_btn.setStyleSheet("font-size: 10px; padding: 0px; color: #aaa;")
        clear_btn.clicked.connect(self.clear)
        
        title_layout.addWidget(history_btn)
        title_layout.addWidget(clear_btn)
        layout.addLayout(title_layout)

        self.image_label = DropImageLabel(f"点击/拖拽")
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("附加描述")
        self.prompt_input.setStyleSheet("font-size: 11px; padding: 2px;")
        layout.addWidget(self.prompt_input)

        self.save_cb = QCheckBox("存入历史")
        self.save_cb.setChecked(True)
        self.save_cb.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.save_cb)

        self.setLayout(layout)
        self.setFixedWidth(100)
        self.setStyleSheet("QWidget { background-color: #f5f5f5; border-radius: 3px; }")

    def open_history_manager(self):
        dialog = HistoryManagerDialog(self.category_name, self.history_dir, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_path:
            self.image_label.set_image(dialog.selected_path)
            self.save_cb.setChecked(False)

    def clear(self):
        self.image_label.clear_image()
        self.prompt_input.clear()
        self.save_cb.setChecked(True)

    def get_data(self):
        path = self.image_label.current_image_path
        if not path: return None
        if self.save_cb.isChecked():
            dest = os.path.join(self.history_dir, os.path.basename(path))
            if path != dest:
                try: shutil.copy2(path, dest)
                except Exception as e: print(f"保存缓存失败: {e}")
        return {
            "category": self.category_name,
            "filename": os.path.basename(path),
            "filepath": path,
            "prompt": self.prompt_input.text()
        }
        
    def load_state(self, state_dict):
        if "filepath" in state_dict and os.path.exists(state_dict["filepath"]):
            self.image_label.set_image(state_dict["filepath"])
        if "prompt" in state_dict: self.prompt_input.setText(state_dict["prompt"])
        if "save_cb" in state_dict: self.save_cb.setChecked(state_dict["save_cb"])

# ================== 标签页实现 ==================

class JsonGeneratorTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.json_file_path = None
        self.initUI()

    def initUI(self):
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        self.drop_area = QLabel("请将 JSON 文件拖拽到此区域", self)
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setStyleSheet("border: 2px dashed #aaa; padding: 30px; font-size: 14px; color: #aaa;")
        layout.addWidget(self.drop_area)

        self.run_btn = QPushButton("执行生成")
        self.run_btn.setEnabled(False) 
        self.run_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        self.run_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.run_btn)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].toLocalFile().lower().endswith('.json'):
            self.json_file_path = urls[0].toLocalFile()
            self.drop_area.setText(f"已加载: {os.path.basename(self.json_file_path)}")
            self.run_btn.setEnabled(True)

    def start_generation(self):
        if not self.json_file_path: return
        self.run_btn.setEnabled(False)
        self.drop_area.setText("正在生成中...")
        thread = threading.Thread(target=self.run_script, args=(self.json_file_path,))
        thread.daemon = True
        thread.start()

    def run_script(self, json_path):
        try: self.main_window.process_image_basic(json_path)
        except Exception as e: traceback.print_exc()
        finally:
            self.run_btn.setEnabled(True)
            self.drop_area.setText(f"已加载: {os.path.basename(self.json_file_path)}")

class CyberNikiTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.slots = {}
        self.instructions_file = os.path.join("cache", "instructions.json")
        self.initUI()
        self.load_instructions_history()

    def initUI(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        
        # 左侧
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_content = QWidget()                     # 1. 显式创建 QWidget 并引用
        left_grid = QGridLayout(left_content)        # 2. 绑定 Layout
        self._build_grid(left_grid, ["角色", "动作", "衣服1", "衣服2", "衣服3"])
        left_scroll.setWidget(left_content)          # 3. 将 Widget 放入 ScrollArea
        left_scroll.setFixedWidth(240)

        # ================= 中间预览 =================
        center_layout = QVBoxLayout()
        self.preview_label = QLabel("产出结果预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff;")
        self.preview_label.setMinimumSize(250, 350)
        
        # 强制忽略图片的原始尺寸撑大效应
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) 
        self.preview_label.setScaledContents(True)
        center_layout.addWidget(self.preview_label, 1)
        
        self.generate_btn = QPushButton("✨ 开始生成 ✨")
        self.generate_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.generate_btn.clicked.connect(self.start_generation)
        center_layout.addWidget(self.generate_btn)
        
        # 【注意】这里删除了原先多余的 top_layout.addLayout(center_layout, 1)

        # ================= 右侧 =================
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right_grid = QGridLayout(right_content)
        self._build_grid(right_grid, ["背景", "鞋子", "袜子", "手套", "发饰", "手持物"])
        right_scroll.setWidget(right_content)
        right_scroll.setFixedWidth(240)

        # ================= 唯一且正确的组装区 =================
        top_layout.addWidget(left_scroll)         # 1. 放入左边栏
        top_layout.addLayout(center_layout, 1)    # 2. 放入中间预览（由于比例设为1，它会自动撑满剩余空间）
        top_layout.addWidget(right_scroll)        # 3. 放入右边栏

        main_layout.addLayout(top_layout, 2)

        # 底部配置
        config_group = QWidget()
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(0, 5, 0, 5)
        
        inst_layout = QHBoxLayout()
        inst_layout.addWidget(QLabel("Instructions:"))
        
        # 1. 核心修复：使用 QTextEdit 替代单行的 QComboBox
        self.inst_input = QTextEdit()
        self.inst_input.setAcceptRichText(False) # 纯文本模式
        self.inst_input.setMaximumHeight(60)     # 限制最大高度为 60 像素
        self.inst_input.setPlaceholderText("在此输入或粘贴长指令，支持自动换行...")
        inst_layout.addWidget(self.inst_input, 1)

        # 2. 将历史记录选择和保存按钮放在右侧垂直布局中
        inst_btn_layout = QVBoxLayout()
        self.history_combo = QComboBox()
        self.history_combo.addItem("📜 选择历史指令...")

        self.history_combo.setMaximumWidth(180)
        self.history_combo.view().setTextElideMode(Qt.ElideRight)

        self.history_combo.currentIndexChanged.connect(self.on_history_selected)
        inst_btn_layout.addWidget(self.history_combo)
        
        save_inst_btn = QPushButton("保存指令")
        save_inst_btn.clicked.connect(self.save_current_instruction)
        inst_btn_layout.addWidget(save_inst_btn)
        
        inst_layout.addLayout(inst_btn_layout)
        config_layout.addLayout(inst_layout)

        self.extra_prompt_input = QTextEdit()
        self.extra_prompt_input.setMaximumHeight(50)
        self.extra_prompt_input.setPlaceholderText("额外补充 Prompts...")
        config_layout.addWidget(self.extra_prompt_input)

        bottom_config_layout = QHBoxLayout()
        bottom_config_layout.addWidget(QLabel("输出比例:"))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["2:3", "3:4", "1:1", "16:9", "9:16"])
        bottom_config_layout.addWidget(self.ratio_combo)
        bottom_config_layout.addStretch()
        
        edit_config_btn = QPushButton("⚙️ 编辑 API 配置")
        edit_config_btn.clicked.connect(self.main_window.open_api_config)
        bottom_config_layout.addWidget(edit_config_btn)
        config_layout.addLayout(bottom_config_layout)

        main_layout.addWidget(config_group, 0)
        self.setLayout(main_layout)

    def on_history_selected(self, index):
        if index > 0:  # 排除第一个提示语选项 "📜 选择历史指令..."
            self.inst_input.setPlainText(self.history_combo.currentText())

    def _build_grid(self, grid_layout, categories):
        row, col = 0, 0
        for cat in categories:
            slot = ImageSlotWidget(cat)
            self.slots[cat] = slot
            grid_layout.addWidget(slot, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1
        grid_layout.setRowStretch(row + 1, 1)

    def load_instructions_history(self):
        if os.path.exists(self.instructions_file):
            try:
                with open(self.instructions_file, 'r', encoding='utf-8') as f:
                    self.history_combo.addItems(json.load(f))
            except: pass

    def save_current_instruction(self):
        current_text = self.inst_input.toPlainText().strip()
        if not current_text: return
        # 收集下拉框中已有的记录（跳过第0项提示语）
        items = [self.history_combo.itemText(i) for i in range(1, self.history_combo.count())]
        if current_text not in items:
            self.history_combo.addItem(current_text)
            items.append(current_text)
            with open(self.instructions_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

    def update_preview_image(self, image_path):
        self.preview_label.setPixmap(QPixmap(image_path))

    def save_current_state(self):
        state = {
            "instruction": self.inst_input.toPlainText(),
            "extra_prompt": self.extra_prompt_input.toPlainText(),
            "aspect_ratio": self.ratio_combo.currentText(),
            "slots": {cat: {"filepath": slot.image_label.current_image_path or "", 
                            "prompt": slot.prompt_input.text(), 
                            "save_cb": slot.save_cb.isChecked()} 
                      for cat, slot in self.slots.items()}
        }
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load_last_state(self):
        if not os.path.exists(STATE_FILE): return
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.inst_input.setPlainText(state.get("instruction", ""))
            self.extra_prompt_input.setText(state.get("extra_prompt", ""))
            self.ratio_combo.setCurrentText(state.get("aspect_ratio", "2:3"))
            for cat, slot_data in state.get("slots", {}).items():
                if cat in self.slots: self.slots[cat].load_state(slot_data)
        except: pass

    def start_generation(self):
        self.save_current_state()
        active_items = [slot.get_data() for slot in self.slots.values() if slot.get_data()]
                
        # 第一处修改判定条件：
        if not active_items and not self.inst_input.toPlainText().strip() and not self.extra_prompt_input.toPlainText().strip():
            print("请至少添加一张参考图或输入提示词！")
            return
            
        # 第二处修改获取文本：
        instruction = self.inst_input.toPlainText()
        extra_prompt = self.extra_prompt_input.toPlainText()
        aspect_ratio = self.ratio_combo.currentText()

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ 生成中...")
        print("=" * 40)
        
        threading.Thread(target=self.run_generation_task, args=(active_items, instruction, extra_prompt, aspect_ratio), daemon=True).start()

    def run_generation_task(self, active_items, instruction, extra_prompt, aspect_ratio):
        try:
            output_path = self.main_window.process_image_cyber_niki(active_items, instruction, extra_prompt, aspect_ratio)
            if output_path: self.update_preview_image(output_path)
        except Exception as e:
            traceback.print_exc()
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("✨ 开始生成 ✨")
            print("=" * 40)

# ================== 主程序窗口 ==================

class ImageGeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ensure_config_exists()
        self.cyber_tab.load_last_state()

    def initUI(self):
        self.setWindowTitle('多模态图片生成控制台 (OpenAI/Gemini)')
        self.resize(900, 650)  # 调小基准尺寸，高 DPI 缩放后正好适合屏幕
        
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.cyber_tab = CyberNikiTab(self)
        self.json_tab = JsonGeneratorTab(self)
        
        self.tabs.addTab(self.cyber_tab, "👗 赛博暖暖")
        self.tabs.addTab(self.json_tab, "📄 JSON 批量生成")
        main_layout.addWidget(self.tabs, 3)

        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("执行日志 (自动存入 log/ 目录):"))
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 11px;")
        log_layout.addWidget(self.log_output)
        
        main_layout.addLayout(log_layout, 1)
        self.setLayout(main_layout)

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)

    def _image_to_base64_data_uri(self, filepath):
        """辅助方法：将图片压缩并转换为带前缀的 Base64 字符串"""
        img_pil = compress_image_if_needed(filepath)
        buffer = BytesIO()
        img_pil.convert('RGB').save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def proxy_generate_method_A(self, api_key, base_url, model_name, text_prompt, active_items, filepath_base):
        """
        方案 A：伪装成视觉对话 (Chat Completions)
        将多张图片作为 message 数组中的 image_url 传给大模型。
        """
        url = base_url.rstrip('/') + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 组装混合内容 (文本 + 多张图片)
        content_list = [{"type": "text", "text": text_prompt}]
        for item in active_items:
            data_uri = self._image_to_base64_data_uri(item['filepath'])
            content_list.append({
                "type": "image_url",
                "image_url": {"url": data_uri}
            })
            
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content_list}],
            "stream": False
        }

        print("\n" + "="*15 + " [方案A] 发送 /v1/chat/completions 原始请求 " + "="*15)
        print(f"请求地址: {url}")
        # 为了控制台不被图刷屏，打印时截断 base64
        debug_payload = json.loads(json.dumps(payload))
        for msg in debug_payload["messages"][0]["content"]:
            if msg["type"] == "image_url":
                msg["image_url"]["url"] = msg["image_url"]["url"][:40] + "...[Base64 Truncated]..."
        print(json.dumps(debug_payload, indent=2, ensure_ascii=False))
        print("="*60 + "\n")

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        # 解析返回结果
        result_text = response.json()["choices"][0]["message"]["content"]
        print(f"模型返回文本: {result_text}")
        
        # 尝试从返回的 Markdown 中提取图片 URL 或 Base64 (格式如 ![img](url) )
        match = re.search(r'!\[.*?\]\((.*?)\)', result_text)
        if match:
            img_data_str = match.group(1)
            output_file = filepath_base + ".png"
            if img_data_str.startswith("data:image"):
                # 处理返回的 Base64
                b64_data = img_data_str.split("base64,")[-1]
                with open(output_file, "wb") as f:
                    f.write(base64.b64decode(b64_data))
            elif img_data_str.startswith("http"):
                # 处理返回的 URL，额外下载一次
                img_res = requests.get(img_data_str, timeout=60)
                with open(output_file, "wb") as f:
                    f.write(img_res.content)
            return output_file
        else:
            print("❌ 无法从聊天返回中解析出图片数据！")
            return None

    def proxy_generate_method_B(self, api_key, base_url, model_name, text_prompt, active_items, req_size, filepath_base):
        """
        方案 B：强行拼接 Prompt (Images Generations)
        请求标准的生图接口，但把图片的 Base64 数据以特殊标签强行塞进 prompt 字符串中。
        """
        url = base_url.rstrip('/') + "/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 把图片编码并追加到提示词末尾
        combined_prompt = text_prompt
        for item in active_items:
            data_uri = self._image_to_base64_data_uri(item['filepath'])
            # 这里的标签格式 "<image>...</image>" 是比较通用的魔改写法
            combined_prompt += f"\n[参考图-{item['category']}]: <image>{data_uri}</image>"
            
        payload = {
            "model": model_name,
            "prompt": combined_prompt,
            "n": 1,
            "size": req_size,
            "response_format": "b64_json"
        }

        print("\n" + "="*15 + " [方案B] 发送 /v1/images/generations 原始请求 " + "="*15)
        print(f"请求地址: {url}")
        # 同样截断打印日志
        debug_payload = json.loads(json.dumps(payload))
        debug_prompt = debug_payload["prompt"]
        debug_payload["prompt"] = re.sub(r'<image>.*?</image>', '<image>[Base64 Truncated]</image>', debug_prompt)
        print(json.dumps(debug_payload, indent=2, ensure_ascii=False))
        print("="*60 + "\n")

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        res_json = response.json()
        output_file = filepath_base + ".png"
        
        if "b64_json" in res_json["data"][0]:
            img_data = base64.b64decode(res_json["data"][0]["b64_json"])
            with open(output_file, "wb") as f:
                f.write(img_data)
        elif "url" in res_json["data"][0]:
            img_res = requests.get(res_json["data"][0]["url"], timeout=60)
            with open(output_file, "wb") as f:
                f.write(img_res.content)
                
        return output_file

    def normalOutputWritten(self, text):
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def ensure_config_exists(self):
        if not os.path.exists(CONFIG_PATH):
            default_config = {
                "api_type": "openai",
                "api_key": "",
                "base_url": "",
                "model": "dall-e-3"
            }
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)

    def open_api_config(self):
        APIConfigDialog(self).exec_()

    def get_api_client(self):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as cf:
                config = json.load(cf)
                
            api_type = config.get("api_type", "openai")
            api_key = config.get("api_key")
            base_url = config.get("base_url")
            model_name = config.get("model", "dall-e-3" if api_type == "openai" else "gemini-3.1-flash-image")
            
            if not api_key:
                print("错误: 缺失 api_key 参数！请点击[编辑 API 配置]添加。")
                return None, None, None

            if api_type == "openai":
                client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)
            else:
                client_options = {"api_key": api_key}
                if base_url: client_options["http_options"] = {"base_url": base_url}
                client = genai.Client(**client_options)
                
            return client, model_name, api_type
        except Exception as e:
            print(f"读取配置失败: {e}")
            return None, None, None

    # --- 赛博暖暖生成逻辑 ---
    def process_image_cyber_niki(self, active_items, instruction, extra_prompt, aspect_ratio):
        SAVE_DIR = os.path.join("output", "ootd")
        os.makedirs(SAVE_DIR, exist_ok=True)

        client, model_name, api_type = self.get_api_client()
        if not client: return None

        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        text_prompt = "【生成任务约束】\n"
        text_prompt += "1. 严格参考提供的图片元素或文本描述特征。画面要求纯净无水印。\n"
        text_prompt += "2. 图片整体生成风格必须是非照片的艺术风格，具体请参考下方的 Instructions。\n\n"
        text_prompt += "【参考元素清单】\n"
        for item in active_items:
            text_prompt += f"- 分类: {item['category']}, 文件名: {item['filename']}, 附加描述: {item['prompt']}\n"
        
        if instruction.strip(): text_prompt += f"\n【风格 Instructions】\n{instruction}\n"
        if extra_prompt.strip(): text_prompt += f"\n【用户补充 Prompts】\n{extra_prompt}\n"

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath_base = os.path.join(SAVE_DIR, f"Niki-{timestamp}")

        try:
            # ================= 新增：第三方代理套壳模式 =================
            if api_type == "openai_proxy":
                # 重新读取配置，获取原始的 key 和 base_url
                with open(CONFIG_PATH, 'r', encoding='utf-8') as cf:
                    config = json.load(cf)
                raw_api_key = config.get("api_key")
                raw_base_url = config.get("base_url", "https://api.openai.com")
                
                size_mapping = {
                    "1:1": "1024x1024", "2:3": "1024x1792", "3:4": "1024x1792", 
                    "9:16": "1024x1792", "16:9": "1792x1024"
                }
                req_size = size_mapping.get(aspect_ratio, "1024x1024")

                # ------------------------------------------------------------------
                # 【切换测试区】目前开启的是方案 B。如果报错，请注释方案 B，打开方案 A。
                # ------------------------------------------------------------------
                
                # --- 调用方案 A ---
                # output_file = self.proxy_generate_method_A(raw_api_key, raw_base_url, model_name, text_prompt, active_items, filepath_base)
                
                # --- 调用方案 B ---
                output_file = self.proxy_generate_method_B(raw_api_key, raw_base_url, model_name, text_prompt, active_items, req_size, filepath_base)

                if output_file:
                    print(f"✅ 代理生图成功: {output_file}")
                return output_file
            if api_type == "openai":
                size_mapping = {
                    "1:1": "1024x1024",
                    "2:3": "1024x1792", "3:4": "1024x1792", "9:16": "1024x1792",
                    "16:9": "1792x1024"
                }
                req_size = size_mapping.get(aspect_ratio, "1024x1024")
                req_prompt = text_prompt[:4000]

                # 👇 --- 新增：打印 OpenAI 原始请求日志 --- 👇
                raw_request_log = {
                    "model": model_name,
                    "prompt": req_prompt,
                    "n": 1,
                    "size": req_size,
                    "response_format": "b64_json"
                }
                print("\n" + "="*15 + " 发送给 OpenAI 的原始请求参数 " + "="*15)
                print(json.dumps(raw_request_log, indent=2, ensure_ascii=False))
                print("="*60 + "\n")

                response = client.images.generate(
                    model=model_name,
                    prompt=req_prompt,
                    n=1,
                    size=req_size,
                    response_format="b64_json"
                )
                
                img_data = base64.b64decode(response.data[0].b64_json)
                output_file = filepath_base + ".png"
                with open(output_file, "wb") as f:
                    f.write(img_data)
                print(f"✅ 图片生成成功: {output_file}")
                return output_file
                
            else:
                # Gemini 模式
                contents = [text_prompt]
                debug_contents_log = [{"text": text_prompt}] # 专门用于打印日志，因为图片对象无法 JSON 序列化
                
                for item in active_items:
                    img_pil = compress_image_if_needed(item['filepath'])
                    text_part = f"[{item['category']} 参考图: {item['filename']}]:"
                    
                    contents.append(text_part)
                    contents.append(img_pil)
                    
                    # 记录日志用
                    debug_contents_log.append({"text": text_part})
                    debug_contents_log.append({"image_data": f"<PIL.Image Object: {item['filename']}, size={img_pil.size}, mode={img_pil.mode}>"})
                
                # 👇 --- 新增：打印 Gemini 原始请求日志 --- 👇
                raw_request_log = {
                    "model": model_name,
                    "contents": debug_contents_log,
                    "config": {
                        "image_size": "2K",
                        "aspect_ratio": aspect_ratio,
                        "output_mime_type": "image/png"
                    }
                }
                print("\n" + "="*15 + " 发送给 Gemini 的原始请求参数 " + "="*15)
                print(json.dumps(raw_request_log, indent=2, ensure_ascii=False))
                print("="*60 + "\n")
                
                result = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(image_size='2K', aspect_ratio=aspect_ratio, output_mime_type="image/png")
                    )
                )
                output_file = None
                for part in result.parts:
                    if part.inline_data is not None:
                        image = part.as_image()
                        output_file = filepath_base + (".png" if str(image.mime_type) == 'image/png' else ".jpg")
                        image.save(output_file)
                        print(f"✅ 图片生成成功: {output_file}")
                return output_file

        except Exception as e:
            print("\n❌ API 调用发生异常！请检查上方的原始请求参数。")
            traceback.print_exc()
        return None

    # --- 基础版的生成逻辑 ---
    def process_image_basic(self, JSON_FILE_PATH):
        SAVE_DIR = "data"
        client, model_name, api_type = self.get_api_client()
        if not client: return

        base_path = os.path.splitext(JSON_FILE_PATH)[0]
        prompt_file_path = f"{base_path}-prompts.txt"
        if not os.path.exists(prompt_file_path):
            print(f"找不到Prompt文件: {prompt_file_path}")
            return

        with open(prompt_file_path, 'r', encoding='utf-8') as pf: prompt = pf.read().strip()
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f: data = json.load(f)

        aspect_ratio = data.get("aspect_ratio", "2:3") 
        title = data.get("chinese_title", data.get("japanese_title", "untitled"))
        
        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        print(f"正在使用模型 [{model_name}] 发送生成请求...")
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(SAVE_DIR, f"{safe_title}-{timestamp}")

        try:
            if api_type == "openai_proxy":
                print("\n" + "="*15 + " [代理模式] 发送自定义 HTTP 请求 " + "="*15)
                # 很多代理商兼容 OpenAI 的 size 格式，如果他们要求 Gemini 的格式，可以把 "size" 换成 "aspect_ratio": aspect_ratio
                size_mapping = {"1:1": "1024x1024", "2:3": "1024x1792", "3:4": "1024x1792", "16:9": "1792x1024", "9:16": "1024x1792"}
                req_size = size_mapping.get(aspect_ratio, "1024x1024")
                
                # 重新读取配置拿 key 和 url，因为刚才的 get_api_client 没处理原生 token
                with open(CONFIG_PATH, 'r', encoding='utf-8') as cf:
                    config = json.load(cf)
                
                proxy_url = config.get("base_url", "").rstrip('/') + "/v1/images/generations"
                headers = {
                    "Authorization": f"Bearer {config.get('api_key')}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "prompt": prompt[:4000],
                    "n": 1,
                    "size": req_size, 
                    "response_format": "b64_json"
                }
                
                print(f"请求地址: {proxy_url}")
                print("请求头: ", {k: v for k, v in headers.items() if k != "Authorization"})
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                print("="*60 + "\n")

                response = requests.post(proxy_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status() # 如果返回 4xx/5xx 会抛出异常
                
                res_json = response.json()
                img_data = base64.b64decode(res_json["data"][0]["b64_json"])
                s_filename = filepath + ".png"
                with open(s_filename, "wb") as f:
                    f.write(img_data)
                print(f"✅ 图片生成成功: {s_filename}")

            if api_type == "openai":
                size_mapping = {"1:1": "1024x1024", "2:3": "1024x1792", "3:4": "1024x1792", "16:9": "1792x1024", "9:16": "1024x1792"}
                req_size = size_mapping.get(aspect_ratio, "1024x1024")
                req_prompt = prompt[:4000]

                # 👇 --- 新增：打印 OpenAI 原始请求日志 --- 👇
                raw_request_log = {
                    "model": model_name,
                    "prompt": req_prompt,
                    "n": 1,
                    "size": req_size,
                    "response_format": "b64_json"
                }
                print("\n" + "="*15 + " [基础版] 发送给 OpenAI 的原始请求参数 " + "="*15)
                print(json.dumps(raw_request_log, indent=2, ensure_ascii=False))
                print("="*60 + "\n")

                response = client.images.generate(
                    model=model_name,
                    prompt=req_prompt,
                    n=1,
                    size=req_size,
                    response_format="b64_json"
                )
                img_data = base64.b64decode(response.data[0].b64_json)
                s_filename = filepath + ".png"
                with open(s_filename, "wb") as f: f.write(img_data)
                print(f"✅ 图片生成成功: {s_filename}")
            else:
                # 👇 --- 新增：打印 Gemini 原始请求日志 --- 👇
                raw_request_log = {
                    "model": model_name,
                    "contents": [{"text": prompt}],
                    "config": {
                        "image_size": "2K",
                        "aspect_ratio": aspect_ratio
                    }
                }
                print("\n" + "="*15 + " [基础版] 发送给 Gemini 的原始请求参数 " + "="*15)
                print(json.dumps(raw_request_log, indent=2, ensure_ascii=False))
                print("="*60 + "\n")

                result = client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(image_config=types.ImageConfig(image_size='2K', aspect_ratio=aspect_ratio))
                )
                for part in result.parts:
                    if part.inline_data is not None:
                        image = part.as_image()
                        s_filename = filepath + (".png" if str(image.mime_type) == 'image/png' else ".jpg")
                        image.save(s_filename)
                        print(f"✅ 图片生成成功: {s_filename}")
        except Exception as e:
            print("\n❌ API 调用发生异常！请检查上方的原始请求参数。")
            traceback.print_exc()

    def closeEvent(self, event):
        self.cyber_tab.save_current_state()
        super().closeEvent(event)

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    ImageGeneratorGUI().show()
    sys.exit(app.exec_())