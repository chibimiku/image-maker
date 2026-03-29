# prompt_generator.py
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QComboBox, 
                             QSpinBox, QLineEdit, QScrollArea, QGridLayout, QFrame, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from openai import OpenAI

# 复用 single_analyzer 中的生图线程
from single_analyzer import ImageGenWorkerThread

class TextPromptGenThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)

    def __init__(self, api_key, base_url, model_name, keywords, length_str, count):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.keywords = keywords
        self.length_str = length_str
        self.count = count

    def run(self):
        self.log_signal.emit("开始生成提示词，请稍候...")
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            system_prompt = "You are an expert prompt engineer for AI image generators. You must respond strictly in JSON format. Return an object with a single key 'prompts' containing an array of strings."
            user_prompt = f"Generate {self.count} unique English prompts for image generation based on the keywords: '{self.keywords}'. The length of each prompt should be '{self.length_str}'. Return JSON format: {{'prompts': ['prompt1', 'prompt2', ...]}}"
            
            response = client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8
            )
            result_json = json.loads(response.choices[0].message.content)
            prompts = result_json.get("prompts", [])
            self.finish_signal.emit(prompts)
        except Exception as e:
            self.log_signal.emit(f"生成提示词失败: {e}")
            self.finish_signal.emit([])

class PromptCellWidget(QFrame):
    def __init__(self, prompt_text, style_getter_func, img_config_getter_func, save_img_cfg_callback, ar_policy_getter_func=None):
        super().__init__()
        self.get_style = style_getter_func
        self.get_img_config = img_config_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.img_thread = None
        self.get_ar_policy = ar_policy_getter_func
        self.initUI(prompt_text)

    def initUI(self, prompt_text):
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 提示词编辑框
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(prompt_text)
        self.text_edit.setMinimumHeight(80)
        self.text_edit.setMaximumHeight(120)
        self.text_edit.setStyleSheet("QTextEdit { border: 1px solid #ddd; background-color: white; }")
        layout.addWidget(self.text_edit)
        
        # 按钮组
        btn_layout = QHBoxLayout()
        self.copy_btn = QPushButton("复制提示词")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.gen_btn = QPushButton("用此提示词生图")
        self.gen_btn.clicked.connect(self.generate_image)
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addWidget(self.gen_btn)
        layout.addLayout(btn_layout)
        
        # 图像展示区
        self.img_label = QLabel("暂无图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumHeight(200)
        self.img_label.setStyleSheet("QLabel { background-color: #eee; border: 1px dashed #aaa; border-radius: 0px; }")
        layout.addWidget(self.img_label)
        
        self.setLayout(layout)

    def _resolve_ar_for_second_stage(self, fallback_ar="1:1") -> str:
        if not self.get_ar_policy:
            return fallback_ar
        policy = self.get_ar_policy() or {}
        override_second = (policy.get("override_second") or "").strip()
        if override_second.startswith("不覆盖"):
            return fallback_ar
        # 【修改前】return default_ar
        # 【修改后】返回用户选择的覆盖比例
        return override_second


    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())

    def generate_image(self):
        self.save_img_cfg()
        img_base_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空！")
            return
            
        current_prompt = self.text_edit.toPlainText().strip()
        active_instructions = self.get_style()
        
        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("正在生成...")
        self.img_label.setText("正在请求生图，请稍候...")
        
        final_ar = self._resolve_ar_for_second_stage("1:1")
        self.img_thread = ImageGenWorkerThread(
            prompt=current_prompt,
            model_name=model_name,
            aspect_ratio=final_ar,
            instructions=active_instructions,
            api_type=api_type
        )

        self.img_thread.finish_signal.connect(self.on_image_finished)
        self.img_thread.start()

    def on_image_finished(self, saved_files):
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("用此提示词生图")
        
        if saved_files and len(saved_files) > 0:
            file_path = saved_files[0]
            pixmap = QPixmap(file_path)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.img_label.setText("生成失败或超时")

class PromptGeneratorWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None):
        super().__init__()
        self.get_text_config = config_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_ar_policy = ar_policy_getter_func
        
        self.initUI()
        
    def initUI(self):
        main_layout = QVBoxLayout()
        
        # 顶部控制面板
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("关键词要求:"))
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("例如: 赛博朋克, 霓虹灯, 侧面半身像")
        control_layout.addWidget(self.keyword_input, stretch=2)
        
        control_layout.addWidget(QLabel("长度:"))
        self.length_combo = QComboBox()
        self.length_combo.addItems(["短 (约10-20词)", "中等 (约30-50词)", "长 (非常详细, 70词以上)"])
        control_layout.addWidget(self.length_combo)
        
        control_layout.addWidget(QLabel("生成个数:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 16)
        self.count_spin.setValue(4)
        control_layout.addWidget(self.count_spin)
        
        main_layout.addLayout(control_layout)

        # 样式选择面板
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("生图画风预设(全Tab通用):"))
        self.main_style_combo = QComboBox()
        style_layout.addWidget(self.main_style_combo, stretch=1)
        
        self.gen_prompts_btn = QPushButton("🚀 开始批量生成提示词")
        self.gen_prompts_btn.clicked.connect(self.generate_prompts)
        style_layout.addWidget(self.gen_prompts_btn)
        main_layout.addLayout(style_layout)

        # 状态提示
        self.status_label = QLabel("输入条件后点击上方按钮...")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        # 滚动显示区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setAlignment(Qt.AlignTop)
        self.scroll_widget.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

    def update_styles(self, style_keys):
        """同步画风列表"""
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)

    def generate_prompts(self):
        base_url, api_key, model_name = self.get_text_config()
        if not api_key:
            QMessageBox.warning(self, "错误", "请先在全局配置中填写文本分析 API Key")
            return
            
        keywords = self.keyword_input.text().strip()
        if not keywords:
            QMessageBox.warning(self, "提示", "请输入关键词要求！")
            return

        self.gen_prompts_btn.setEnabled(False)
        self.clear_grid()
        
        self.gen_thread = TextPromptGenThread(
            api_key=api_key, base_url=base_url, model_name=model_name,
            keywords=keywords, length_str=self.length_combo.currentText(), count=self.count_spin.value()
        )
        self.gen_thread.log_signal.connect(self.status_label.setText)
        self.gen_thread.finish_signal.connect(self.on_prompts_generated)
        self.gen_thread.start()

    def on_prompts_generated(self, prompts):
        self.gen_prompts_btn.setEnabled(True)
        if not prompts:
            self.status_label.setText("生成失败，未能获取有效结果。")
            return
            
        self.status_label.setText(f"✅ 成功生成 {len(prompts)} 组提示词！可以修改文本或直接点击生图。")
        self.populate_grid(prompts)

    def clear_grid(self):
        for i in reversed(range(self.grid_layout.count())): 
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None: 
                widget.setParent(None)

    def get_current_style_instructions(self):
        style_name = self.main_style_combo.currentText()
        return self.get_styles().get(style_name, "")

    def populate_grid(self, prompts):
        # 横向最多4个，纵向滚动
        cols = 4
        for idx, prompt_str in enumerate(prompts):
            row = idx // cols
            col = idx % cols
            cell = PromptCellWidget(
                prompt_text=prompt_str,
                style_getter_func=self.get_current_style_instructions,
                img_config_getter_func=self.get_img_config,
                save_img_cfg_callback=self.save_img_cfg,
                ar_policy_getter_func=self.get_ar_policy
            )

            self.grid_layout.addWidget(cell, row, col)