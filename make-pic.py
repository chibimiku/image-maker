import sys
import json
import os
import time
import traceback
import threading
import logging
from io import BytesIO

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QTabWidget, 
                             QScrollArea, QComboBox, QLineEdit, QCheckBox, 
                             QMessageBox, QGridLayout, QFileDialog,
                             QDialog, QFormLayout, QListWidget, QListWidgetItem, QAbstractItemView,
                             QSizePolicy, QInputDialog)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QImage

from PIL import Image
# 导入我们独立出去的 API 请求模块
from api_backend import generate_image_whatai, generate_image_aigc2d, get_api_config, load_config

# ================== 初始化目录与日志 ==================
os.makedirs("log", exist_ok=True)
os.makedirs("cache", exist_ok=True)
STATE_FILE = os.path.join("cache", "last_state.json")
CONFIG_PATH = "config-image.json"

# 【新增】正规的 PyQt 日志处理器
class GUILogHandler(logging.Handler, QObject):
    textWritten = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.textWritten.emit(msg + '\n')

log_filename = os.path.join("log", f"run_{time.strftime('%Y%m%d')}.log")

# 创建 GUI 日志处理器实例
gui_log_handler = GUILogHandler()
gui_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 配置全局 logging，同时输出到文件和 GUI
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, encoding='utf-8'),
                        gui_log_handler
                    ])

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        if text.strip():
            logging.info(text.strip())
        self.textWritten.emit(str(text))
    def flush(self): pass

def compress_image_if_needed(image_path, max_dim=2048, max_size_mb=8):
    img = Image.open(image_path)
    
    # 统一转为 RGBA 或 RGB，保证完美兼容 PNG
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGBA')
        
    w, h = img.size
    if w > max_dim or h > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    size_mb = buffer.tell() / (1024 * 1024)
    
    # PNG 是无损格式，遇到体积超标的情况，通过按比例缩小尺寸来限制大小
    while size_mb > max_size_mb and img.width > 256 and img.height > 256:
        new_w = int(img.width * 0.8)
        new_h = int(img.height * 0.8)
        img.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        size_mb = buffer.tell() / (1024 * 1024)
        
    return img

# ================== 弹窗组件 ==================

class APIConfigDialog(QDialog):
    # API 配置与模型获取的可视化界面
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑 API 配置")
        self.resize(450, 300)
        self.initUI()
        self.load_config()

    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        form_layout.addRow("API Key:", self.api_key_input)
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("API Base URL")
        form_layout.addRow("Base URL:", self.base_url_input)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        form_layout.addRow("模型名称:", self.model_combo)
        layout.addLayout(form_layout)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存"); save_btn.clicked.connect(self.save_config)
        cancel_btn = QPushButton("取消"); cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn); btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            data = load_config(CONFIG_PATH)
            current_api = data.get("current_api", "whatup")
            api_config = data.get("apis", {}).get(current_api, {})
            self.api_key_input.setText(api_config.get("api_key", ""))
            self.base_url_input.setText(api_config.get("base_url", ""))
            current_model = api_config.get("model", "gemini-3.1-flash-image-preview-2k")
            self.model_combo.addItem(current_model)
            self.model_combo.setCurrentText(current_model)

    def save_config(self):
        data = load_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}
        current_api = data.get("current_api", "whatup")
        api_config = data.get("apis", {})
        api_config[current_api] = {
            "api_key": self.api_key_input.text().strip(),
            "base_url": self.base_url_input.text().strip(),
            "model": self.model_combo.currentText().strip()
        }
        data.update({
            "current_api": current_api,
            "apis": api_config
        })
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
# 请在文件顶部的 PyQt5.QtWidgets 导入列表中补充 QFileDialog
from PyQt5.QtWidgets import QFileDialog

class DropImageLabel(QLabel):
    imageChanged = pyqtSignal(str)
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px dashed #aaa; background-color: transparent; color: #888; font-size: 10px;")
        self.setFixedSize(80, 80)
        
        self.setAcceptDrops(True)
        self.setScaledContents(False) 
        self.current_image_path = None

    # 【修复 Bug 4】补充 dragEnterEvent 以允许接收拖拽动作
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self.set_image(file_path)

    # 【修复 Bug 2】增加点击事件，弹出文件选择框
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp)"
            )
            if file_path:
                self.set_image(file_path)
        # 将事件向上传递，确保父组件也能借此获取焦点
        super().mousePressEvent(event)
                
    def set_image(self, file_path):
        if not file_path or not os.path.exists(file_path): return
        
        # --- 新增拦截逻辑：所有导入的图片统一转换/压缩并另存为纯净 PNG ---
        try:
            os.makedirs("cache/temp", exist_ok=True)
            safe_filename = f"import_{int(time.time() * 1000)}.png"
            tmp_path = os.path.join("cache", "temp", safe_filename)
            
            # 调用压缩与转换逻辑，返回 PIL Image 对象
            img = compress_image_if_needed(file_path)
            img.save(tmp_path, format="PNG")
            
            file_path = tmp_path  # 后续逻辑全部使用这个新生成的 PNG 路径
        except Exception as e:
            logging.error(f"图片转换PNG失败: {e}")
            return
        # ----------------------------------------------------------------

        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        self.setStyleSheet("border: none; background-color: transparent;")
        self.imageChanged.emit(file_path)
        
    def clear_image(self):
        self.current_image_path = None
        self.clear()
        self.setText("点击/粘贴\n添加图片")
        self.setStyleSheet("border: 1px dashed #aaa; background-color: transparent; color: #888; font-size: 10px;")


class ImageSlotWidget(QWidget):
    # 【修改点 1】支持焦点获取与点击高亮机制
    def __init__(self, category_name):
        super().__init__()
        self.category_name = category_name
        self.history_dir = os.path.join("cache", "history", category_name)
        os.makedirs(self.history_dir, exist_ok=True)
        
        self.setFocusPolicy(Qt.ClickFocus)
        self.setObjectName("SlotWidget")

        # 【关键修复】：允许自定义 QWidget 应用背景和边框样式
        self.setAttribute(Qt.WA_StyledBackground, True)
        # 默认加上 2px 透明边框，防止高亮时组件大小变化导致 UI 抖动
        self.setStyleSheet("#SlotWidget { background-color: #f5f5f5; border-radius: 4px; border: 2px solid transparent; }")

        self.setStyleSheet("#SlotWidget { background-color: #f5f5f5; border-radius: 4px; }")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel(f"<b style='font-size:11px;'>{category_name}</b>")
        # 【修复 Bug 1】让鼠标点击穿透文本标签，使整个 Slot 获取焦点以支持 Ctrl+V
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        # 【修复 Bug 3】增加历史记录按钮
        history_btn = QPushButton("📜")
        history_btn.setFixedSize(16, 16)
        history_btn.setStyleSheet("font-size: 11px; padding: 0px; border: none; background: transparent;")
        history_btn.clicked.connect(self.open_history_dialog)
        title_layout.addWidget(history_btn)
        
        clear_btn = QPushButton("×")
        clear_btn.setFixedSize(16, 16)
        clear_btn.setStyleSheet("font-size: 11px; padding: 0px; color: #888; border: none;")
        clear_btn.clicked.connect(self.clear)
        title_layout.addWidget(clear_btn)
        
        layout.addLayout(title_layout)

        # ... (保留原有的 image_label, prompt_input, save_cb 初始化代码) ...
        self.image_label = DropImageLabel(f"点击/粘贴\n添加图片")
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("附加描述")
        self.prompt_input.setStyleSheet("font-size: 10px; padding: 2px; border: 1px solid #ddd; border-radius: 2px;")
        layout.addWidget(self.prompt_input)

        self.save_cb = QCheckBox("存入历史")
        self.save_cb.setChecked(True)
        self.save_cb.setStyleSheet("font-size: 9px;")
        layout.addWidget(self.save_cb)

        self.setLayout(layout)
        self.setFixedWidth(100)

    # 补充唤起历史弹窗的方法
    def open_history_dialog(self):
        dialog = HistoryManagerDialog(self.category_name, self.history_dir, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_path:
            self.image_label.set_image(dialog.selected_path)

    # 获取焦点时高亮边框和底色
    def focusInEvent(self, event):
        self.setStyleSheet("#SlotWidget { background-color: #e3f2fd; border: 2px solid #2196f3; border-radius: 4px; }")
        super().focusInEvent(event)

    # 失去焦点时恢复原样
    def focusOutEvent(self, event):
        self.setStyleSheet("#SlotWidget { background-color: #f5f5f5; border-radius: 4px; border: 2px solid transparent; }")
        super().focusOutEvent(event)

    # 支持 Ctrl+V 粘贴
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_V:
            clipboard = QApplication.clipboard()
            if clipboard.mimeData().hasImage():
                image = clipboard.image()
                os.makedirs("cache/temp", exist_ok=True)
                tmp_path = os.path.join("cache", "temp", f"paste_{int(time.time())}.png")
                image.save(tmp_path, "PNG")
                self.image_label.set_image(tmp_path)
                logging.info(f"✅ 已成功将剪贴板图片粘贴至 [{self.category_name}]")
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # 点击内部任何地方，强制组件获取焦点
        self.setFocus()
        super().mousePressEvent(event)

    def clear(self):
        self.image_label.clear_image()
        self.prompt_input.clear()

    def get_data(self):
        path = self.image_label.current_image_path
        if not path: return None
        
        # 【修复 Bug 3】真正实现图片另存为历史记录的逻辑
        if self.save_cb.isChecked():
            import shutil
            dest_path = os.path.join(self.history_dir, os.path.basename(path))
            if not os.path.exists(dest_path) and path != dest_path:
                try:
                    shutil.copy2(path, dest_path)
                except Exception as e:
                    logging.error(f"[{self.category_name}] 保存历史图片失败: {e}")

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


# ================== 赛博暖暖标签页 ==================
# ================== 赛博暖暖标签页 ==================
class CyberNikiTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.slots = {}
        self.styles_file = "config-styles.json"  # 指向外部的 json 文件
        self.styles_data = {}
        self.initUI()
        self.load_styles_config()

    def initUI(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        
        # 左侧
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_content = QWidget()
        left_grid = QGridLayout(left_content)
        self._build_grid(left_grid, ["角色1", "角色2", "动作", "衣服1", "衣服2", "衣服3"])
        left_scroll.setWidget(left_content)
        left_scroll.setFixedWidth(240)

        # 中间预览
        center_layout = QVBoxLayout()
        self.preview_label = QLabel("产出结果预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff;")
        self.preview_label.setMinimumSize(250, 350)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) 
        self.preview_label.setScaledContents(True)
        center_layout.addWidget(self.preview_label, 1)
        
        self.generate_btn = QPushButton("✨ 开始生成 ✨")
        self.generate_btn.setStyleSheet("padding: 8px; font-size: 12px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 4px;")
        self.generate_btn.clicked.connect(self.start_generation)
        center_layout.addWidget(self.generate_btn)

        # 右侧
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right_grid = QGridLayout(right_content)
        self._build_grid(right_grid, ["背景", "鞋子", "袜子", "手套", "发饰", "手持物"])
        right_scroll.setWidget(right_content)
        right_scroll.setFixedWidth(240)

        top_layout.addWidget(left_scroll)
        top_layout.addLayout(center_layout, 1)
        top_layout.addWidget(right_scroll)

        main_layout.addLayout(top_layout, 2)

        # 底部配置
        config_group = QWidget()
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(0, 5, 0, 5)
        
        inst_layout = QHBoxLayout()
        inst_layout.addWidget(QLabel("<b style='font-size:11px'>画风设定 (Instructions):</b>"))
        
        self.inst_input = QTextEdit()
        self.inst_input.setAcceptRichText(False)
        self.inst_input.setMaximumHeight(50)
        self.inst_input.setStyleSheet("font-size: 11px;")
        self.inst_input.setPlaceholderText("可自定义模型生成画风及约束，会作为 instructions 传给 API...")
        inst_layout.addWidget(self.inst_input, 1)

        inst_btn_layout = QVBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.setMaximumWidth(150)
        self.style_combo.setStyleSheet("font-size: 10px;")
        self.style_combo.currentIndexChanged.connect(self.on_style_selected)
        inst_btn_layout.addWidget(self.style_combo)
        
        save_inst_btn = QPushButton("保存为预设")
        save_inst_btn.setStyleSheet("font-size: 10px;")
        save_inst_btn.clicked.connect(self.save_current_instruction)
        inst_btn_layout.addWidget(save_inst_btn)
        
        inst_layout.addLayout(inst_btn_layout)
        config_layout.addLayout(inst_layout)

        self.extra_prompt_input = QTextEdit()
        self.extra_prompt_input.setMaximumHeight(40)
        self.extra_prompt_input.setStyleSheet("font-size: 11px;")
        self.extra_prompt_input.setPlaceholderText("额外补充的局部 Prompt (如：人物动作补充、场景细节描述)...")
        config_layout.addWidget(self.extra_prompt_input)

        bottom_config_layout = QHBoxLayout()
        bottom_config_layout.addWidget(QLabel("输出比例:"))
        self.ratio_combo = QComboBox()
        # 更新了所有的输出比例选项
        self.ratio_combo.addItems(["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"])
        bottom_config_layout.addWidget(self.ratio_combo)
        bottom_config_layout.addStretch()
        
        edit_config_btn = QPushButton("⚙️ 编辑 API 配置")
        edit_config_btn.setStyleSheet("font-size: 11px;")
        edit_config_btn.clicked.connect(lambda: APIConfigDialog(self).exec_())
        bottom_config_layout.addWidget(edit_config_btn)
        config_layout.addLayout(bottom_config_layout)

        main_layout.addWidget(config_group, 0)
        self.setLayout(main_layout)

    def load_styles_config(self):
        """读取 config-styles.json 文件并初始化下拉菜单"""
        if os.path.exists(self.styles_file):
            try:
                with open(self.styles_file, 'r', encoding='utf-8') as f:
                    self.styles_data = json.load(f)
                    self.style_combo.addItems(self.styles_data.keys())
            except Exception as e:
                logging.error(f"读取 {self.styles_file} 失败: {e}")
        else:
            logging.warning(f"未找到 {self.styles_file}，将使用空预设。")

    def on_style_selected(self, index):
        """下拉框切换时更新文本框内容"""
        selected_style = self.style_combo.currentText()
        if selected_style in self.styles_data:
            self.inst_input.setPlainText(self.styles_data[selected_style])

    def save_current_instruction(self):
        """通过弹窗命名并保存当前文本框内容为新画风预设"""
        current_text = self.inst_input.toPlainText().strip()
        if not current_text: return
        
        name, ok = QInputDialog.getText(self, '保存画风预设', '请输入新预设名称:')
        if ok and name:
            name = name.strip()
            self.styles_data[name] = current_text
            
            # 如果是新名字，添加到下拉列表
            if self.style_combo.findText(name) == -1:
                self.style_combo.addItem(name)
            self.style_combo.setCurrentText(name)
            
            # 写入 json 文件
            try:
                with open(self.styles_file, 'w', encoding='utf-8') as f:
                    json.dump(self.styles_data, f, ensure_ascii=False, indent=4)
                logging.info(f"✅ 成功保存画风预设: [{name}]")
            except Exception as e:
                logging.error(f"保存画风预设失败: {e}")

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

    def update_preview_image(self, image_path):
        self.preview_label.setPixmap(QPixmap(image_path))

    def save_current_state(self):
        state = {
            "instruction": self.inst_input.toPlainText(),
            "extra_prompt": self.extra_prompt_input.toPlainText(),
            "aspect_ratio": self.ratio_combo.currentText(),
            "slots": {cat: {"filepath": slot.image_label.current_image_path or "", "prompt": slot.prompt_input.text()} 
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
                
        if not active_items and not self.inst_input.toPlainText().strip() and not self.extra_prompt_input.toPlainText().strip():
            logging.warning("请至少添加一张参考图或输入提示词！")
            return
            
        instructions = self.inst_input.toPlainText()
        extra_prompt = self.extra_prompt_input.toPlainText()
        aspect_ratio = self.ratio_combo.currentText()

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ 生成中...")
        
        logging.info("=" * 40)
        logging.info("开始构建并发送生成任务...")
        
        threading.Thread(target=self.run_generation_task, args=(active_items, instructions, extra_prompt, aspect_ratio), daemon=True).start()

    def run_generation_task(self, active_items, instructions, extra_prompt, aspect_ratio):
        try:
            PROMPT_DIR = os.path.join("data", "prompts")
            os.makedirs(PROMPT_DIR, exist_ok=True)
            template_path = os.path.join(PROMPT_DIR, "default_template.txt")
            
            if not os.path.exists(template_path):
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write("【生成任务约束】\n1. 严格参考提供的图片元素或文本描述特征。画面要求纯净无水印。\n2. 图片整体生成风格必须是非照片的艺术风格，具体请参考 Instructions。\n")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                text_prompt = f.read()

            text_prompt += "\n【参考图片映射说明】（请严格按照以下序号和对应的部位提取特征）：\n"
            image_paths = []
            
            for idx, item in enumerate(active_items):
                cat = item['category']
                prompt = item['prompt']
                
                if "角色" in cat:
                    desc = f"- 第 {idx + 1} 张传入的图片：用于参考【{cat}】的外貌特征。请务必详细提取并还原该角色的发型、发色、眼睛颜色（瞳色）、五官特点及整体面部轮廓等核心外貌细节。"
                elif "衣服" in cat or cat in ["鞋子", "袜子", "手套", "发饰", "手持物"]:
                    desc = f"- 第 {idx + 1} 张传入的图片：用于参考【{cat}】的设计。请准确提取其款式、材质、颜色搭配及图案纹理等细节。"
                elif cat == "动作":
                    desc = f"- 第 {idx + 1} 张传入的图片：用于参考人物的【{cat}】与姿势。请重点参考并还原其肢体动作、身体朝向和体态重心。"
                elif cat == "背景":
                    desc = f"- 第 {idx + 1} 张传入的图片：用于参考【{cat}】。请提取其场景氛围、环境元素分布、光影关系与整体色调。"
                else:
                    desc = f"- 第 {idx + 1} 张传入的图片：用于参考【{cat}】部位的设计与特征。"
                
                if prompt:
                    desc += f" 补充要求：{prompt}"
                    
                text_prompt += desc + "\n"
                image_paths.append(item['filepath'])
            
            if extra_prompt.strip(): 
                text_prompt += f"\n【用户补充 Prompts】\n{extra_prompt}\n"

            logging.info("--- 当前发送给 API 的 Prompt ---")
            logging.info(f"\n{text_prompt}")
            logging.info("--------------------------------")

            config = load_config(CONFIG_PATH)
            current_api = config.get("current_api", "whatup")
            api_config = config.get("apis", {}).get(current_api, {})
            model_name = api_config.get("model", "nano-banana-2")
            
            # 根据API类型调用相应的生成函数
            if current_api == "aigc2d":
                saved_files = generate_image_aigc2d(
                    prompt=text_prompt,
                    image_paths=image_paths,
                    model=model_name,
                    aspect_ratio=aspect_ratio,
                    instructions=instructions,
                    api_type=current_api
                )
            else:
                saved_files = generate_image_whatai(
                    prompt=text_prompt,
                    image_paths=image_paths,
                    model=model_name,
                    aspect_ratio=aspect_ratio,
                    instructions=instructions,
                    api_type=current_api  
                )

            if saved_files: 
                self.update_preview_image(saved_files[0])
                
        except Exception as e:
            logging.error(f"生成任务发生异常:\n{traceback.format_exc()}")
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("✨ 开始生成 ✨")
            logging.info("生成任务流程结束")
            logging.info("=" * 40)

# ================== 主程序窗口 ==================
class ImageGeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ensure_config_exists()
        self.cyber_tab.load_last_state()

    def initUI(self):
        self.setWindowTitle('多模态图片生成控制台 (Whatai / API)')
        self.resize(850, 600) 
        
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.cyber_tab = CyberNikiTab(self)
        self.tabs.addTab(self.cyber_tab, "👗 赛博暖暖")
        main_layout.addWidget(self.tabs, 3)

        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("<b style='font-size:11px'>执行日志:</b>"))
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 10px;")
        log_layout.addWidget(self.log_output)
        
        main_layout.addLayout(log_layout, 1)
        self.setLayout(main_layout)

        gui_log_handler.textWritten.connect(self.normalOutputWritten)


    def normalOutputWritten(self, text):
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def ensure_config_exists(self):
        if not os.path.exists(CONFIG_PATH):
            default_config = {
                "current_api": "whatup",
                "apis": {
                    "whatup": {
                        "api_key": "",
                        "base_url": "https://api.whatai.cc/v1",
                        "model": "gemini-3.1-flash-image-preview-2k"
                    },
                    "aigc2d": {
                        "api_key": "",
                        "base_url": "",
                        "model": ""
                    }
                }
            }
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)

    def closeEvent(self, event):
        self.cyber_tab.save_current_state()
        super().closeEvent(event)

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    gui = ImageGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())