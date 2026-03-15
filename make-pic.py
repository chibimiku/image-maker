import sys
import json
import os
import time
import traceback
import threading
import shutil
import logging
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QTabWidget, 
                             QScrollArea, QComboBox, QLineEdit, QCheckBox, 
                             QFileDialog, QInputDialog, QMessageBox, QGridLayout,
                             QDialog, QFormLayout, QListWidget, QListWidgetItem, QAbstractItemView)
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
        self.resize(400, 250)
        self.initUI()
        self.load_config()

    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        form_layout.addRow("API Key:", self.api_key_input)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("留空则使用官方默认地址")
        form_layout.addRow("Base URL:", self.base_url_input)

        # 模型选择与获取行
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
                self.api_key_input.setText(data.get("api_key", ""))
                self.base_url_input.setText(data.get("base_url", ""))
                current_model = data.get("model", "gemini-3.1-flash-image")
                self.model_combo.addItem(current_model)
                self.model_combo.setCurrentText(current_model)
            except Exception as e:
                print(f"加载配置失败: {e}")

    def fetch_models(self):
        api_key = self.api_key_input.text().strip()
        base_url = self.base_url_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入 API Key")
            return

        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("获取中...")
        QApplication.processEvents()

        try:
            client_options = {"api_key": api_key}
            if base_url:
                client_options["http_options"] = {"base_url": base_url}
            client = genai.Client(**client_options)
            
            # 获取模型列表
            models = client.models.list()
            self.model_combo.clear()
            model_names = [m.name for m in models]
            if not model_names:
                 model_names = ["gemini-3.1-flash-image", "gemini-2.5-pro", "gemini-2.5-flash"]
            self.model_combo.addItems(model_names)
            
            # 尝试优先选中带 image 的模型
            for name in model_names:
                if "image" in name.lower():
                    self.model_combo.setCurrentText(name)
                    break
                    
            QMessageBox.information(self, "成功", "获取模型列表成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取失败:\n{e}")
        finally:
            self.fetch_btn.setEnabled(True)
            self.fetch_btn.setText("获取列表")

    def save_config(self):
        data = {
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
    """可视化的历史记录管理器"""
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
        if not os.path.exists(self.history_dir):
            return
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
            path = item.data(Qt.UserRole)
            try:
                os.remove(path)
                self.list_widget.takeItem(self.list_widget.row(item))
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除失败: {e}")

    def clear_all(self):
        reply = QMessageBox.question(self, '确认', '确定要清空该分类下所有历史图片吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                for filename in os.listdir(self.history_dir):
                    os.remove(os.path.join(self.history_dir, filename))
                self.list_widget.clear()
            except Exception as e:
                QMessageBox.warning(self, "错误", f"清空失败: {e}")

# ================== UI 核心组件 ==================

class DropImageLabel(QLabel):
    imageChanged = pyqtSignal(str)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px dashed #aaa; background-color: #2d2d2d; color: #aaa; font-size: 11px;")
        # 缩小尺寸以适应两列布局
        self.setMinimumSize(80, 80)
        self.setMaximumSize(80, 80)
        self.setAcceptDrops(True)
        self.setScaledContents(True)
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
            file_path, _ = QFileDialog.getOpenFileName(self, "选择参考图片", "", "Images (*.png *.jpg *.jpeg *.webp)")
            if file_path:
                self.set_image(file_path)

    def set_image(self, file_path):
        if not file_path or not os.path.exists(file_path):
            self.clear_image()
            return
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap)
        self.imageChanged.emit(file_path)

    def clear_image(self):
        self.current_image_path = None
        self.clear()
        self.setText("点击/拖拽\n添加")
        self._reset_style()

    def _reset_style(self):
        if self.current_image_path:
            self.setStyleSheet("border: 1px solid #555;")
        else:
            self.setStyleSheet("border: 1px dashed #aaa; background-color: #2d2d2d; color: #aaa; font-size: 11px;")

class ImageSlotWidget(QWidget):
    def __init__(self, category_name):
        super().__init__()
        self.category_name = category_name
        self.history_dir = os.path.join("cache", "history", category_name)
        os.makedirs(self.history_dir, exist_ok=True)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # 标题与操作栏紧凑布局
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(f"<b style='font-size:12px;'>{category_name}</b>")
        history_btn = QPushButton("历史")
        history_btn.setFixedSize(36, 20)
        history_btn.setStyleSheet("font-size: 10px; padding: 0px;")
        history_btn.clicked.connect(self.open_history_manager)
        clear_btn = QPushButton("x")
        clear_btn.setFixedSize(16, 20)
        clear_btn.setStyleSheet("font-size: 10px; padding: 0px; color: #aaa;")
        clear_btn.clicked.connect(self.clear)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(history_btn)
        title_layout.addWidget(clear_btn)
        layout.addLayout(title_layout)

        # 图片区域
        self.image_label = DropImageLabel(f"点击/拖拽")
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # 描述与保存勾选
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("附加描述")
        self.prompt_input.setStyleSheet("font-size: 11px; padding: 2px;")
        layout.addWidget(self.prompt_input)

        self.save_cb = QCheckBox("存入历史")
        self.save_cb.setChecked(True)
        self.save_cb.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.save_cb)

        self.setLayout(layout)
        # 固定组件宽度以适应双列
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
        if not path:
            return None
            
        if self.save_cb.isChecked():
            filename = os.path.basename(path)
            dest = os.path.join(self.history_dir, filename)
            if path != dest:
                try:
                    shutil.copy2(path, dest)
                except Exception as e:
                    print(f"保存 {self.category_name} 缓存失败: {e}")
            
        return {
            "category": self.category_name,
            "filename": os.path.basename(path),
            "filepath": path,
            "prompt": self.prompt_input.text()
        }
        
    def load_state(self, state_dict):
        """恢复上次状态"""
        if "filepath" in state_dict and os.path.exists(state_dict["filepath"]):
            self.image_label.set_image(state_dict["filepath"])
        if "prompt" in state_dict:
            self.prompt_input.setText(state_dict["prompt"])
        if "save_cb" in state_dict:
            self.save_cb.setChecked(state_dict["save_cb"])

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
        self.reset_drop_area_style()
        layout.addWidget(self.drop_area)

        self.run_btn = QPushButton("执行生成", self)
        self.run_btn.setEnabled(False) 
        self.run_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        self.run_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.run_btn)
        self.setLayout(layout)

    def reset_drop_area_style(self):
        self.drop_area.setStyleSheet("border: 2px dashed #aaa; padding: 30px; font-size: 14px; color: #aaa; background-color: transparent;")
    def highlight_drop_area_style(self):
        self.drop_area.setStyleSheet("border: 2px dashed #4CAF50; padding: 30px; font-size: 14px; color: #4CAF50; background-color: #e8f5e9;")
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.highlight_drop_area_style()
        else:
            event.ignore()
    def dragLeaveEvent(self, event):
        self.reset_drop_area_style()
    def dropEvent(self, event):
        self.reset_drop_area_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith('.json'):
                self.json_file_path = file_path
                self.drop_area.setText(f"已加载: {os.path.basename(file_path)}")
                self.run_btn.setEnabled(True)

    def start_generation(self):
        if not self.json_file_path: return
        self.run_btn.setEnabled(False)
        self.drop_area.setText("正在生成中，请耐心等待...")
        thread = threading.Thread(target=self.run_script, args=(self.json_file_path,))
        thread.daemon = True
        thread.start()

    def run_script(self, json_path):
        try:
            self.main_window.process_image_basic(json_path)
        except Exception as e:
            print(f"运行发生异常: {e}")
            traceback.print_exc()
        finally:
            self.run_btn.setEnabled(True)
            self.drop_area.setText(f"已加载: {os.path.basename(self.json_file_path)}")

class CyberNikiTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.slots = {} # 使用字典方便按名字索引保存状态
        self.instructions_file = os.path.join("cache", "instructions.json")
        self.initUI()
        self.load_instructions_history()

    def initUI(self):
        main_layout = QVBoxLayout()
        
        # 1. 顶部：预览区与插槽 (三分区)
        top_layout = QHBoxLayout()
        
        # 左侧：双列网格
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_content = QWidget()
        left_grid = QGridLayout(left_content)
        left_cats = ["角色", "动作", "衣服1", "衣服2", "衣服3"]
        self._build_grid(left_grid, left_cats)
        left_scroll.setWidget(left_content)
        left_scroll.setFixedWidth(240) # 适当加宽容纳两列
        top_layout.addWidget(left_scroll)

        # 中间：预览区缩小
        center_layout = QVBoxLayout()
        self.preview_label = QLabel("产出结果预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff;")
        self.preview_label.setMinimumSize(250, 350)
        self.preview_label.setScaledContents(True)
        center_layout.addWidget(self.preview_label, 1)
        
        self.generate_btn = QPushButton("✨ 开始生成 ✨")
        self.generate_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.generate_btn.clicked.connect(self.start_generation)
        center_layout.addWidget(self.generate_btn)
        top_layout.addLayout(center_layout, 1)

        # 右侧：双列网格
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right_grid = QGridLayout(right_content)
        right_cats = ["背景", "鞋子", "袜子", "手套", "发饰", "手持物"]
        self._build_grid(right_grid, right_cats)
        right_scroll.setWidget(right_content)
        right_scroll.setFixedWidth(240)
        top_layout.addWidget(right_scroll)

        main_layout.addLayout(top_layout, 2)

        # 2. 中间：配置区
        config_group = QWidget()
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(0, 5, 0, 5)
        
        inst_layout = QHBoxLayout()
        inst_layout.addWidget(QLabel("Instructions:"))
        self.inst_combo = QComboBox()
        self.inst_combo.setEditable(True)
        self.inst_combo.setSizePolicy(self.inst_combo.sizePolicy().Expanding, self.inst_combo.sizePolicy().Fixed)
        inst_layout.addWidget(self.inst_combo, 1)
        save_inst_btn = QPushButton("保存指令")
        save_inst_btn.clicked.connect(self.save_current_instruction)
        inst_layout.addWidget(save_inst_btn)
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
                    history = json.load(f)
                    self.inst_combo.addItems(history)
            except Exception as e:
                print(f"加载指令历史失败: {e}")

    def save_current_instruction(self):
        current_text = self.inst_combo.currentText().strip()
        if not current_text: return
        items = [self.inst_combo.itemText(i) for i in range(self.inst_combo.count())]
        if current_text not in items:
            self.inst_combo.addItem(current_text)
            items.append(current_text)
            try:
                with open(self.instructions_file, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存指令失败: {e}")

    def update_preview_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.preview_label.setPixmap(pixmap)

    def save_current_state(self):
        """保存当前所有输入内容到状态文件"""
        state = {
            "slots": {},
            "instruction": self.inst_combo.currentText(),
            "extra_prompt": self.extra_prompt_input.toPlainText(),
            "aspect_ratio": self.ratio_combo.currentText()
        }
        for cat, slot in self.slots.items():
            path = slot.image_label.current_image_path
            state["slots"][cat] = {
                "filepath": path if path else "",
                "prompt": slot.prompt_input.text(),
                "save_cb": slot.save_cb.isChecked()
            }
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存最后状态失败: {e}")

    def load_last_state(self):
        """加载上次保存的状态"""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.inst_combo.setCurrentText(state.get("instruction", ""))
            self.extra_prompt_input.setText(state.get("extra_prompt", ""))
            self.ratio_combo.setCurrentText(state.get("aspect_ratio", "2:3"))
            
            slots_data = state.get("slots", {})
            for cat, slot_data in slots_data.items():
                if cat in self.slots:
                    self.slots[cat].load_state(slot_data)
        except Exception as e:
            print(f"恢复历史状态失败: {e}")

    def start_generation(self):
        self.save_current_state()
        active_items = []
        for slot in self.slots.values():
            data = slot.get_data()
            if data: active_items.append(data)
                
        if not active_items and not self.inst_combo.currentText().strip() and not self.extra_prompt_input.toPlainText().strip():
            print("请至少添加一张参考图或输入提示词！")
            return

        instruction = self.inst_combo.currentText()
        extra_prompt = self.extra_prompt_input.toPlainText()
        aspect_ratio = self.ratio_combo.currentText()

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ 生成中...")
        print("=" * 40)
        
        thread = threading.Thread(target=self.run_generation_task, args=(active_items, instruction, extra_prompt, aspect_ratio))
        thread.daemon = True
        thread.start()

    def run_generation_task(self, active_items, instruction, extra_prompt, aspect_ratio):
        try:
            output_path = self.main_window.process_image_cyber_niki(active_items, instruction, extra_prompt, aspect_ratio)
            if output_path:
                self.update_preview_image(output_path)
        except Exception as e:
            print(f"生成发生异常: {e}")
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
        # 启动时恢复状态
        self.cyber_tab.load_last_state()

    def initUI(self):
        self.setWindowTitle('Gemini 图片生成控制台')
        self.resize(1100, 750)
        
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
        print("界面初始化完成，准备就绪。")

    def normalOutputWritten(self, text):
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def ensure_config_exists(self):
        if not os.path.exists(CONFIG_PATH):
            default_config = {
                "api_key": "",
                "base_url": "",
                "model": "gemini-3.1-flash-image"
            }
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)

    def open_api_config(self):
        dialog = APIConfigDialog(self)
        dialog.exec_()

    def get_genai_client(self):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as cf:
                config_data = json.load(cf)
                
            api_key = config_data.get("api_key")
            base_url = config_data.get("base_url")
            model_name = config_data.get("model", "gemini-3.1-flash-image")
            
            if not api_key:
                print("错误: 缺失 api_key 参数！请点击[编辑 API 配置]添加。")
                return None, None

            client_options = {"api_key": api_key}
            if base_url:
                client_options["http_options"] = {"base_url": base_url}
                
            client = genai.Client(**client_options)
            return client, model_name
        except Exception as e:
            print(f"读取配置失败: {e}")
            return None, None

    # --- 赛博暖暖生成逻辑 ---
    def process_image_cyber_niki(self, active_items, instruction, extra_prompt, aspect_ratio):
        SAVE_DIR = os.path.join("output", "ootd")
        os.makedirs(SAVE_DIR, exist_ok=True)

        client, model_name = self.get_genai_client()
        if not client: return None

        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        contents = []
        text_prompt = "【生成任务约束】\n"
        text_prompt += "1. 严格参考提供的图片元素。如果描述语言和图片内容发生冲突（例如图片是黑丝而描述是白丝），一律以参考图片内容为准。\n"
        text_prompt += "2. 请完美提取参考图片中的相应特征（如衣服款式、鞋子细节等），并将它们整合到一个人物/场景中。\n"
        text_prompt += "3. 图片整体生成风格必须是非照片的艺术风格，具体请参考下方的 Instructions。如果提供的参考图是真实照片，必须进行艺术化风格转换。\n"
        text_prompt += "4. 画面要求纯净无水印。\n\n"
        
        text_prompt += "【参考元素清单】\n"
        for item in active_items:
            text_prompt += f"- 分类: {item['category']}, 文件名: {item['filename']}, 附加描述: {item['prompt']}\n"
        
        if instruction.strip():
            text_prompt += f"\n【风格 Instructions】\n{instruction}\n"
        if extra_prompt.strip():
            text_prompt += f"\n【用户补充 Prompts】\n{extra_prompt}\n"

        contents.append(text_prompt)
        print("发送的 Prompt:\n" + text_prompt)

        for item in active_items:
            try:
                img_pil = compress_image_if_needed(item['filepath'])
                contents.append(f"[{item['category']} 参考图: {item['filename']}]:")
                contents.append(img_pil)
                print(f"加载图片: {item['filename']}")
            except Exception as e:
                print(f"处理图片 {item['filename']} 失败: {e}")
        
        try:
            result = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        image_size='2K',
                        aspect_ratio=aspect_ratio,
                        output_mime_type="image/png"
                    )
                ),
            )

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"Niki-{timestamp}"
            filepath_base = os.path.join(SAVE_DIR, filename)
            
            output_file = None
            for part in result.parts:
                if part.text is not None:
                    print(f"模型返回: {part.text}")
                elif part.inline_data is not None:
                    image = part.as_image()
                    output_file = filepath_base + (".png" if str(image.mime_type) == 'image/png' else ".jpg")
                    image.save(output_file)
                    print(f"✅ 图片生成成功: {output_file}")
            return output_file

        except Exception as e:
            print("调用模型时出错:")
            traceback.print_exc()
        return None

    # --- 基础版的生成逻辑 ---
    def process_image_basic(self, JSON_FILE_PATH):
        SAVE_DIR = "data"
        client, model_name = self.get_genai_client()
        if not client: return

        base_path = os.path.splitext(JSON_FILE_PATH)[0]
        prompt_file_path = f"{base_path}-prompts.txt"
        if not os.path.exists(prompt_file_path):
            print(f"找不到Prompt文件: {prompt_file_path}")
            return

        with open(prompt_file_path, 'r', encoding='utf-8') as pf:
            prompt = pf.read().strip()
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        aspect_ratio = data.get("aspect_ratio", "2:3") 
        title = data.get("chinese_title", data.get("japanese_title", "untitled"))

        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        print(f"正在使用模型 [{model_name}] 发送生成请求...")
        try:
            result = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        image_size='2K',
                        aspect_ratio=aspect_ratio,
                    )
                ),
            )

            os.makedirs(SAVE_DIR, exist_ok=True)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(SAVE_DIR, f"{safe_title}-{timestamp}")
            
            for part in result.parts:
                if part.text is not None:
                    print(f"模型返回: {part.text}")
                elif part.inline_data is not None:
                    image = part.as_image()
                    s_filename = filepath + (".png" if str(image.mime_type) == 'image/png' else ".jpg")
                    image.save(s_filename)
                    print(f"✅ 图片生成成功: {s_filename}")

        except Exception as e:
            print("调用模型时出错:")
            traceback.print_exc()

    # 关闭窗口时保存状态
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