import os
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QInputDialog,
                             QComboBox, QFormLayout, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt
from openai import OpenAI

# 引入抽离出去的独立组件
from style_analyzer import StyleAnalyzerWidget
from single_analyzer import SingleAnalyzerWidget
# 【新增】引入批量提示词生成组件
from prompt_generator import PromptGeneratorWidget

CONFIG_FILE = "config.json"
CONFIG_IMAGE_FILE = "config-image.json"
CONFIG_STYLES_FILE = "config-styles.json"
DEFAULT_ASPECT_RATIO = "1:1"
ASPECT_RATIO_OPTIONS = ["不覆盖(沿用原逻辑)", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
NO_OVERRIDE_TEXT = "不覆盖(沿用原逻辑)"


DEFAULT_STYLES = {
    "默认(无附加)": ""
}

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.styles_data = DEFAULT_STYLES.copy()
        # 【新增】状态记录器
        self.last_used_style = "默认(无附加)"

        self.initUI()
        self.load_config()
        self.load_styles_config()

    def initUI(self):
        self.setWindowTitle("AI 图像辅助创作工具箱")
        self.resize(1100, 750) # 稍微加宽一点窗口适配新的4列网格
        
        main_layout = QVBoxLayout()
        self.main_tabs = QTabWidget()
        
        # 【Tab 1: 单图内容分析】
        self.single_analyzer_tab = SingleAnalyzerWidget(
            config_getter_func=lambda: (self.url_input.text().strip(), self.key_input.text().strip(), self.model_combo.currentText().strip()),
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip()),
            styles_getter_func=lambda: self.styles_data,
            save_img_cfg_callback=lambda: self.save_image_config(silent=True),
            ar_policy_getter_func=self.get_ar_policy_config   # 新增
        )

        # 【新增】监听画风切换信号以实现多端同步和记忆
        self.single_analyzer_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)
        self.main_tabs.addTab(self.single_analyzer_tab, "单图内容分析")

        # 【Tab 2: 多图画风提取】
        self.style_analyzer_tab = StyleAnalyzerWidget(
            config_getter_func=lambda: (self.url_input.text().strip(), self.key_input.text().strip(), self.model_combo.currentText().strip())
        )
        self.main_tabs.addTab(self.style_analyzer_tab, "多图画风提取")

        # 【新增 Tab 3: 批量提示词与生图】
        self.prompt_generator_tab = PromptGeneratorWidget(
            config_getter_func=lambda: (self.url_input.text().strip(), self.key_input.text().strip(), self.model_combo.currentText().strip()),
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip()),
            styles_getter_func=lambda: self.styles_data,
            save_img_cfg_callback=lambda: self.save_image_config(silent=True),
            ar_policy_getter_func=self.get_ar_policy_config   # 新增
        )

        self.prompt_generator_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)
        self.main_tabs.addTab(self.prompt_generator_tab, "批量提示词与生图")

        # 【Tab 4: 全局配置】
        self.config_tabs = QTabWidget()
        
        # 3.1 文本分析配置
        tab_text = QWidget()
        text_layout = QFormLayout()
        self.url_input = QLineEdit()
        text_layout.addRow("Base URL:", self.url_input)
        
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        text_layout.addRow("API Key:", self.key_input)
        
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True) 
        model_layout.addWidget(self.model_combo, stretch=1)
        self.fetch_btn = QPushButton("获取模型列表")
        self.fetch_btn.clicked.connect(self.fetch_models)
        model_layout.addWidget(self.fetch_btn)
        text_layout.addRow("分析模型:", model_layout)
        
        self.save_text_cfg_btn = QPushButton("保存分析配置")
        self.save_text_cfg_btn.clicked.connect(self.save_text_config)
        text_layout.addRow("", self.save_text_cfg_btn)
        tab_text.setLayout(text_layout)
        
        # 3.2 图片生成配置
        tab_image = QWidget()
        image_layout = QFormLayout()
        self.img_url_input = QLineEdit()
        image_layout.addRow("Base URL:", self.img_url_input)
        
        self.img_key_input = QLineEdit()
        self.img_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        image_layout.addRow("API Key:", self.img_key_input)
        
        self.img_model_combo = QComboBox()
        self.img_model_combo.setEditable(True)
        self.img_model_combo.addItem("nano-banana-2") 
        image_layout.addRow("生图模型:", self.img_model_combo)

        # 默认长宽比（当策略为覆盖时使用）
        self.default_ar_combo = QComboBox()
        self.default_ar_combo.setEditable(True)
        self.default_ar_combo.addItems(ASPECT_RATIO_OPTIONS)
        self.default_ar_combo.setCurrentText(DEFAULT_ASPECT_RATIO)
        image_layout.addRow("默认长宽比:", self.default_ar_combo)

        # 第一次策略：分析后保存 prompt 时是否覆盖
        self.override_ar_first_combo = QComboBox()
        self.override_ar_first_combo.addItems(ASPECT_RATIO_OPTIONS) 
        self.override_ar_first_combo.setCurrentText(NO_OVERRIDE_TEXT)
        image_layout.addRow("第一次长宽比策略:", self.override_ar_first_combo)

        # 第二次策略：真正生图时是否覆盖
        self.override_ar_second_combo = QComboBox()
        self.override_ar_second_combo.addItems(ASPECT_RATIO_OPTIONS)
        self.override_ar_second_combo.setCurrentText(NO_OVERRIDE_TEXT)
        image_layout.addRow("第二次长宽比策略:", self.override_ar_second_combo)

        # ================= 新增：超时与重试配置 =================
        self.img_timeout_spin = QSpinBox()
        self.img_timeout_spin.setRange(10, 600) # 允许设置 10秒 到 600秒
        self.img_timeout_spin.setValue(120)     # 默认 120秒
        self.img_timeout_spin.setSuffix(" 秒")
        image_layout.addRow("请求超时时间:", self.img_timeout_spin)

        self.img_retry_spin = QSpinBox()
        self.img_retry_spin.setRange(0, 10)     # 允许设置 0 到 10 次重试
        self.img_retry_spin.setValue(1)         # 默认重试 1 次
        self.img_retry_spin.setSuffix(" 次")
        image_layout.addRow("失败重试次数:", self.img_retry_spin)
        # =======================================================

        # 变更后自动保存
        self.default_ar_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.override_ar_first_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.override_ar_second_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))


        
        self.save_img_cfg_btn = QPushButton("保存生图配置")
        self.save_img_cfg_btn.clicked.connect(lambda: self.save_image_config(silent=False))
        image_layout.addRow("", self.save_img_cfg_btn)
        tab_image.setLayout(image_layout)

        # 3.3 画风预设管理
        tab_style = QWidget()
        style_layout = QVBoxLayout()
        
        style_top_layout = QHBoxLayout()
        style_top_layout.addWidget(QLabel("选择预设:"))
        self.style_manage_combo = QComboBox()
        self.style_manage_combo.currentTextChanged.connect(self.on_manage_style_changed)
        style_top_layout.addWidget(self.style_manage_combo, stretch=1)
        
        self.add_style_btn = QPushButton("新建预设")
        self.add_style_btn.clicked.connect(self.add_new_style)
        self.del_style_btn = QPushButton("删除预设")
        self.del_style_btn.clicked.connect(self.delete_current_style)
        style_top_layout.addWidget(self.add_style_btn)
        style_top_layout.addWidget(self.del_style_btn)
        style_layout.addLayout(style_top_layout)
        
        self.style_content_edit = QTextEdit()
        style_layout.addWidget(self.style_content_edit)
        
        self.save_style_btn = QPushButton("保存当前预设")
        self.save_style_btn.clicked.connect(self.save_current_style)
        style_layout.addWidget(self.save_style_btn)
        tab_style.setLayout(style_layout)

        self.config_tabs.addTab(tab_text, "文本分析 API")
        self.config_tabs.addTab(tab_image, "图片生成 API")
        self.config_tabs.addTab(tab_style, "画风预设管理")
        
        self.main_tabs.addTab(self.config_tabs, "全局配置")
        self.main_tabs.setCurrentIndex(0)
        main_layout.addWidget(self.main_tabs)
        self.setLayout(main_layout)

    def sync_selected_style(self, style_name):
        """【新增】同步多页面的画风下拉框，并自动保存到硬盘"""
        if not style_name: return
        self.last_used_style = style_name
        
        # 阻断信号避免死循环
        for combo in [self.single_analyzer_tab.main_style_combo, self.prompt_generator_tab.main_style_combo]:
            if combo.currentText() != style_name:
                combo.blockSignals(True)
                combo.setCurrentText(style_name)
                combo.blockSignals(False)
        
        self.save_text_config(silent=True) # 利用基础配置表保存这个状态


    def get_ar_policy_config(self):
        """提供给子组件读取长宽比策略"""
        return {
            "default_aspect_ratio": self.default_ar_combo.currentText().strip() or DEFAULT_ASPECT_RATIO,
            "override_first": self.override_ar_first_combo.currentText().strip() or NO_OVERRIDE_TEXT,
            "override_second": self.override_ar_second_combo.currentText().strip() or NO_OVERRIDE_TEXT,
        }


    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.url_input.setText(config.get("base_url", ""))
                    self.key_input.setText(config.get("api_key", ""))
                    # 【新增】读取上次保存的画风
                    self.last_used_style = config.get("last_used_style", "默认(无附加)")
                    
                    saved_model = config.get("model", "")
                    if saved_model:
                        self.model_combo.clear()
                        self.model_combo.addItem(saved_model)
                        self.model_combo.setCurrentText(saved_model)
            except Exception as e:
                print(f"加载 {CONFIG_FILE} 失败: {e}")
                
        if os.path.exists(CONFIG_IMAGE_FILE):
            try:
                with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.img_url_input.setText(config.get("base_url", "https://api.whatai.cc/v1"))
                    self.img_key_input.setText(config.get("api_key", ""))
                    saved_model = config.get("model", "")
                    if saved_model:
                        if self.img_model_combo.findText(saved_model) == -1:
                            self.img_model_combo.addItem(saved_model)
                        self.img_model_combo.setCurrentText(saved_model)
                    saved_default_ar = config.get("default_aspect_ratio", DEFAULT_ASPECT_RATIO)
                    if self.default_ar_combo.findText(saved_default_ar) == -1:
                        self.default_ar_combo.addItem(saved_default_ar)
                    self.default_ar_combo.setCurrentText(saved_default_ar)

                    saved_first = config.get("override_aspect_ratio_first", "不覆盖(沿用原逻辑)")
                    if self.override_ar_first_combo.findText(saved_first) == -1:
                        self.override_ar_first_combo.addItem(saved_first)
                    self.override_ar_first_combo.setCurrentText(saved_first)

                    saved_second = config.get("override_aspect_ratio_second", "不覆盖(沿用原逻辑)")
                    if self.override_ar_second_combo.findText(saved_second) == -1:
                        self.override_ar_second_combo.addItem(saved_second)
                    self.override_ar_second_combo.setCurrentText(saved_second)

                    # ================= 新增：读取超时与重试 =================
                    saved_timeout = config.get("timeout", 120)
                    self.img_timeout_spin.setValue(saved_timeout)
                    
                    saved_retries = config.get("max_retries", 1)
                    self.img_retry_spin.setValue(saved_retries)
                    # =====================================================
            except Exception as e:
                print(f"加载 {CONFIG_IMAGE_FILE} 失败: {e}")

    def load_styles_config(self):
        if os.path.exists(CONFIG_STYLES_FILE):
            try:
                with open(CONFIG_STYLES_FILE, "r", encoding="utf-8") as f:
                    loaded_styles = json.load(f)
                    if loaded_styles:
                        self.styles_data = loaded_styles
            except Exception as e:
                print(f"加载画风配置失败: {e}")
        self.update_style_combos()

    def update_style_combos(self):
        curr_manage = self.style_manage_combo.currentText()
        
        self.style_manage_combo.blockSignals(True)
        self.style_manage_combo.clear()
        keys = list(self.styles_data.keys())
        self.style_manage_combo.addItems(keys)
        
        if curr_manage in keys: self.style_manage_combo.setCurrentText(curr_manage)
        self.style_manage_combo.blockSignals(False)
        self.on_manage_style_changed(self.style_manage_combo.currentText())
        
        # 同步更新多个组件的画风列表
        if hasattr(self, 'single_analyzer_tab'):
            self.single_analyzer_tab.update_styles(keys)
            # 恢复上次保存的最后使用画风
            if self.last_used_style in keys:
                self.single_analyzer_tab.main_style_combo.setCurrentText(self.last_used_style)
                
        if hasattr(self, 'prompt_generator_tab'):
            self.prompt_generator_tab.update_styles(keys)

    def on_manage_style_changed(self, style_name):
        if style_name in self.styles_data:
            self.style_content_edit.setPlainText(self.styles_data[style_name])

    def save_current_style(self):
        style_name = self.style_manage_combo.currentText()
        if not style_name: return
        self.styles_data[style_name] = self.style_content_edit.toPlainText().strip()
        self.save_styles_to_disk()
        QMessageBox.information(self, "成功", f"画风预设 '{style_name}' 已保存！")

    def save_styles_to_disk(self):
        try:
            with open(CONFIG_STYLES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.styles_data, f, ensure_ascii=False, indent=4)
            self.update_style_combos()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存画风文件失败: {e}")

    def add_new_style(self):
        text, ok = QInputDialog.getText(self, '新建预设', '请输入新画风预设的名称:')
        if ok and text.strip():
            name = text.strip()
            if name in self.styles_data:
                QMessageBox.warning(self, "提示", "预设名称已存在！")
                return
            self.styles_data[name] = ""
            self.update_style_combos()
            self.style_manage_combo.setCurrentText(name)

    def delete_current_style(self):
        style_name = self.style_manage_combo.currentText()
        if not style_name or style_name == "默认(无附加)":
            QMessageBox.warning(self, "提示", "无法删除默认预设！")
            return
        reply = QMessageBox.question(self, '确认删除', f"确定要删除 '{style_name}' 吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.styles_data[style_name]
            self.save_styles_to_disk()

    def save_text_config(self, silent=False):
        config = {
            "base_url": self.url_input.text().strip(),
            "api_key": self.key_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "last_used_style": getattr(self, "last_used_style", "默认(无附加)") # 写入记忆状态
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            if not silent:
                QMessageBox.information(self, "成功", f"配置已保存至 {CONFIG_FILE}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")
            
    def save_image_config(self, silent=False):
        config = {
            "base_url": self.img_url_input.text().strip() or "https://api.whatai.cc/v1",
            "api_key": self.img_key_input.text().strip(),
            "model": self.img_model_combo.currentText().strip(),
            "default_aspect_ratio": self.default_ar_combo.currentText().strip() or DEFAULT_ASPECT_RATIO,
            "override_aspect_ratio_first": self.override_ar_first_combo.currentText().strip() or "不覆盖(沿用原逻辑)",
            "override_aspect_ratio_second": self.override_ar_second_combo.currentText().strip() or "不覆盖(沿用原逻辑)",
            "timeout": self.img_timeout_spin.value(),
            "max_retries": self.img_retry_spin.value(),

        }
        try:
            with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            if not silent:
                QMessageBox.information(self, "成功", f"生图配置已保存至 {CONFIG_IMAGE_FILE}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")

    def fetch_models(self):
        api_key = self.key_input.text().strip()
        base_url = self.url_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "错误", "请先输入文本分析的 API Key")
            return
            
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("获取中...")
        QApplication.processEvents()

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
            model_names = sorted([m.id for m in models.data])
            
            current_text = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(model_names)
            
            index = self.model_combo.findText(current_text)
            if index >= 0: self.model_combo.setCurrentIndex(index)
            elif current_text: self.model_combo.setCurrentText(current_text)
                
            QMessageBox.information(self, "成功", f"成功获取 {len(model_names)} 个可用模型！")
        except Exception as e:
            QMessageBox.warning(self, "获取失败", f"获取模型列表失败，请检查 URL 和 Key 是否正确。\n错误信息: {e}")
        finally:
            self.fetch_btn.setEnabled(True)
            self.fetch_btn.setText("获取模型列表")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())