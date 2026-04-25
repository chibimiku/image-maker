import os
import json
import hashlib
import datetime
import importlib
try:
    importlib.import_module("onnxruntime")
except Exception:
    pass
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QInputDialog,
                             QComboBox, QFormLayout, QMessageBox, QTabWidget, QCheckBox)
from openai import OpenAI

# 引入抽离出去的独立组件
from style_analyzer import StyleAnalyzerWidget
from single_analyzer import SingleAnalyzerWidget
# 【新增】引入批量提示词生成组件
from prompt_generator import PromptGeneratorWidget
# 【新增】引入批量图片分析组件
from batch_analyzer import BatchAnalyzerWidget
# 【新增】引入图片编辑组件
from image_edit import ImageEditWidget
# 【新增】引入角色设计组件
from char_design import CharDesignWidget
from z_image_edit_tab import ZImageEditWidget
from pic_cate_tab import PicCateWidget
from json_dataset_tab import JsonDatasetWidget
from webp_compressor import DragDropCompressor
from flux2_client_tab import Flux2ClientWidget
from upscaler_tab import UpscalerTabWidget
from single_gen_debug_tab import SingleGenDebugWidget
from utils.image_upscale_runtime import normalize_upscale_options

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
CONFIG_IMAGE_FILE = os.path.join(BASE_DIR, "config-image.json")
CONFIG_STYLES_FILE = os.path.join(BASE_DIR, "config-styles.json")
DEFAULT_ASPECT_RATIO = "1:1"
ASPECT_RATIO_OPTIONS = ["不覆盖(沿用原逻辑)", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
NO_OVERRIDE_TEXT = "不覆盖(沿用原逻辑)"
DEFAULT_BOORU_TAG_LIMIT = 30


DEFAULT_STYLES = {
    "默认(无附加)": ""
}

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.styles_data = DEFAULT_STYLES.copy()
        self._style_sync_enabled = False
        self.pic_cate_state = {
            "source_directory": "",
            "target_directory": "",
            "trimmed_directory": "",
            "train_name": ""
        }
        # 【新增】状态记录器
        self.last_used_style = "默认(无附加)"
        self.use_nsfw_single = False
        self.use_nsfw_batch = False
        self.upscale_options = normalize_upscale_options({})

        self.initUI()
        self.load_config()
        self.load_styles_config()
        self._style_sync_enabled = True
        if self.last_used_style not in self.styles_data:
            self.last_used_style = "默认(无附加)"
        self.sync_selected_style(self.last_used_style)

    def initUI(self):
        self.setWindowTitle("AI 图像辅助创作工具箱")
        self.resize(1100, 750) # 稍微加宽一点窗口适配新的4列网格
        
        main_layout = QVBoxLayout()
        self.main_tabs = QTabWidget()
        self.analysis_root_tab = QWidget()
        self.analysis_tabs = QTabWidget()
        self.generation_root_tab = QWidget()
        self.generation_tabs = QTabWidget()
        self.settings_root_tab = QWidget()
        
        # 【Tab 1: 单图内容分析】
        self.single_analyzer_tab = SingleAnalyzerWidget(
            config_getter_func=self.get_text_config,
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data,
            save_img_cfg_callback=lambda: self.save_image_config(silent=True),
            ar_policy_getter_func=self.get_ar_policy_config,
            nsfw_default_getter_func=lambda: self.use_nsfw_single,
            nsfw_changed_callback=self.on_single_nsfw_changed,
            booru_tag_limit_getter_func=self.get_booru_tag_limit,
            timeout_getter_func=self.get_request_timeout_seconds,
            upscale_options_getter_func=self.get_upscale_options,
            upscale_options_changed_callback=self.update_upscale_options
        )

        # 【新增】监听画风切换信号以实现多端同步和记忆
        self.single_analyzer_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)

        # 【新增 Tab 3: 批量提示词与生图】
        self.prompt_generator_tab = PromptGeneratorWidget(
            config_getter_func=self.get_text_config,
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data,
            save_img_cfg_callback=lambda: self.save_image_config(silent=True),
            ar_policy_getter_func=self.get_ar_policy_config,   # 新增
            upscale_options_getter_func=self.get_upscale_options,
            upscale_options_changed_callback=self.update_upscale_options
        )

        self.prompt_generator_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)

        # 【新增 Tab 4: 批量图片分析】
        self.batch_analyzer_tab = BatchAnalyzerWidget(
            config_getter_func=self.get_text_config,
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data,
            save_img_cfg_callback=lambda: self.save_image_config(silent=True),
            ar_policy_getter_func=self.get_ar_policy_config,
            nsfw_default_getter_func=lambda: self.use_nsfw_batch,
            nsfw_changed_callback=self.on_batch_nsfw_changed,
            booru_tag_limit_getter_func=self.get_booru_tag_limit,
            timeout_getter_func=self.get_request_timeout_seconds,
            upscale_options_getter_func=self.get_upscale_options,
            upscale_options_changed_callback=self.update_upscale_options
        )

        self.batch_analyzer_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)
        self.batch_analyzer_tab.quick_export_requested.connect(self.handle_batch_quick_export)

        # 【新增 Tab 5: 批量图片编辑】
        self.image_edit_tab = ImageEditWidget(
            config_getter_func=self.get_text_config,
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data
        )
        self.image_edit_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)

        # 【新增 Tab 6: 角色设计生成】
        self.char_design_tab = CharDesignWidget(
            config_getter_func=self.get_text_config,
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data,
            upscale_options_getter_func=self.get_upscale_options,
            upscale_options_changed_callback=self.update_upscale_options
        )
        self.char_design_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)
        self.single_gen_debug_tab = SingleGenDebugWidget(
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data,
            save_img_cfg_callback=lambda: self.save_image_config(silent=True),
            ar_policy_getter_func=self.get_ar_policy_config
        )
        self.single_gen_debug_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)
        self.z_image_edit_tab = ZImageEditWidget()

        # 【Tab 2: 多图画风提取】
        self.style_analyzer_tab = StyleAnalyzerWidget(
            config_getter_func=self.get_text_config
        )

        self.pic_cate_tab = PicCateWidget(
            save_values_callback=self.save_pic_cate_state
        )

        self.json_dataset_tab = JsonDatasetWidget()
        self.json_dataset_tab.quick_split_requested.connect(self.handle_json_quick_split)

        self.compressor_tab = DragDropCompressor()
        self.compressor_tab.setWindowTitle("PNG/WebP 定体积压缩")
        self.upscaler_tab = UpscalerTabWidget(
            options_getter=self.get_upscale_options,
            options_changed_callback=self.update_upscale_options
        )
        self.flux2_client_tab = Flux2ClientWidget()

        analysis_layout = QVBoxLayout()
        analysis_layout.addWidget(self.analysis_tabs)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_root_tab.setLayout(analysis_layout)

        generation_layout = QVBoxLayout()
        generation_layout.addWidget(self.generation_tabs)
        generation_layout.setContentsMargins(0, 0, 0, 0)
        self.generation_root_tab.setLayout(generation_layout)

        self.analysis_tabs.addTab(self.single_analyzer_tab, "单图内容分析")
        self.analysis_tabs.addTab(self.batch_analyzer_tab, "批量图片分析")
        self.analysis_tabs.addTab(self.json_dataset_tab, "JSON数据集导出")
        self.analysis_tabs.addTab(self.pic_cate_tab, "图片分类切分")

        self.generation_tabs.addTab(self.prompt_generator_tab, "批量提示词与生图")
        self.generation_tabs.addTab(self.image_edit_tab, "批量图片编辑")
        self.generation_tabs.addTab(self.char_design_tab, "角色设计生成")
        self.generation_tabs.addTab(self.single_gen_debug_tab, "单图调试生图")
        self.generation_tabs.addTab(self.style_analyzer_tab, "多图画风提取")
        self.generation_tabs.addTab(self.compressor_tab, "PNG/WebP压缩")
        self.generation_tabs.addTab(self.upscaler_tab, "图片Upscaler")
        self.generation_tabs.addTab(self.flux2_client_tab, "WebUI Img2Img")

        self.main_tabs.addTab(self.analysis_root_tab, "图片分析")
        self.main_tabs.addTab(self.generation_root_tab, "图片生成")

        # 【Tab 8: 全局配置】
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

        self.booru_tag_limit_spin = QSpinBox()
        self.booru_tag_limit_spin.setRange(1, 200)
        self.booru_tag_limit_spin.setValue(DEFAULT_BOORU_TAG_LIMIT)
        self.booru_tag_limit_spin.valueChanged.connect(lambda: self.save_text_config(silent=True))
        text_layout.addRow("booru-tags 数量上限:", self.booru_tag_limit_spin)
        
        self.save_text_cfg_btn = QPushButton("保存分析配置")
        self.save_text_cfg_btn.clicked.connect(self.save_text_config)
        text_layout.addRow("", self.save_text_cfg_btn)
        tab_text.setLayout(text_layout)

        tab_text_nsfw = QWidget()
        text_nsfw_layout = QFormLayout()
        self.nsfw_url_input = QLineEdit()
        self.nsfw_url_input.editingFinished.connect(lambda: self.save_text_config(silent=True))
        text_nsfw_layout.addRow("Base URL:", self.nsfw_url_input)

        self.nsfw_key_input = QLineEdit()
        self.nsfw_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.nsfw_key_input.editingFinished.connect(lambda: self.save_text_config(silent=True))
        text_nsfw_layout.addRow("API Key:", self.nsfw_key_input)

        nsfw_model_layout = QHBoxLayout()
        self.nsfw_model_combo = QComboBox()
        self.nsfw_model_combo.setEditable(True)
        self.nsfw_model_combo.currentTextChanged.connect(lambda: self.save_text_config(silent=True))
        nsfw_model_layout.addWidget(self.nsfw_model_combo, stretch=1)
        self.fetch_nsfw_btn = QPushButton("获取模型列表")
        self.fetch_nsfw_btn.clicked.connect(self.fetch_nsfw_models)
        nsfw_model_layout.addWidget(self.fetch_nsfw_btn)
        text_nsfw_layout.addRow("分析模型:", nsfw_model_layout)

        self.save_nsfw_cfg_btn = QPushButton("保存分析配置")
        self.save_nsfw_cfg_btn.clicked.connect(self.save_text_config)
        text_nsfw_layout.addRow("", self.save_nsfw_cfg_btn)
        tab_text_nsfw.setLayout(text_nsfw_layout)
        
        # 3.2 图片生成配置
        tab_image = QWidget()
        image_layout = QFormLayout()
        
        # API类型选择
        self.api_type_combo = QComboBox()
        self.api_type_combo.addItems(["whatup", "aigc2d", "openai-image", "openrouter-image", "aigc-2d-gpt"])
        # 【优化】动态获取当前下拉框的默认选中值，无论列表怎么变都能保持同步
        self._current_api_type = self.api_type_combo.currentText()
        self.api_type_combo.currentTextChanged.connect(self.on_api_type_changed)
        image_layout.addRow("API类型:", self.api_type_combo)
        
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
        self.img_timeout_spin.setRange(10, 999) # 允许设置 10秒 到 999秒（支持3位数）
        self.img_timeout_spin.setValue(120)     # 默认 120秒
        self.img_timeout_spin.setSuffix(" 秒")
        image_layout.addRow("请求超时时间:", self.img_timeout_spin)

        self.img_retry_spin = QSpinBox()
        self.img_retry_spin.setRange(0, 10)     # 允许设置 0 到 10 次重试
        self.img_retry_spin.setValue(1)         # 默认重试 1 次
        self.img_retry_spin.setSuffix(" 次")
        image_layout.addRow("失败重试次数:", self.img_retry_spin)

        self.img_debug_dump_checkbox = QCheckBox("开启")
        self.img_debug_dump_checkbox.setChecked(False)
        image_layout.addRow("Debug完整HTTP落盘:", self.img_debug_dump_checkbox)
        
        # 生成图片分辨率
        self.img_resolution_combo = QComboBox()
        self.img_resolution_combo.addItems(["1K", "2K", "4K"])
        self.img_resolution_combo.setCurrentText("1K")
        image_layout.addRow("生成图片分辨率:", self.img_resolution_combo)
        # =======================================================

        # 变更后自动保存
        self.default_ar_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.override_ar_first_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.override_ar_second_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.img_resolution_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.img_debug_dump_checkbox.stateChanged.connect(lambda _v: self.save_image_config(silent=True))


        
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
        self.config_tabs.addTab(tab_text_nsfw, "文本分析（NSFW）")
        self.config_tabs.addTab(tab_image, "图片生成 API")
        self.config_tabs.addTab(tab_style, "画风预设管理")

        settings_layout = QVBoxLayout()
        settings_layout.addWidget(self.config_tabs)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_root_tab.setLayout(settings_layout)
        self.main_tabs.addTab(self.settings_root_tab, "设置")

        self.main_tabs.setCurrentIndex(0)
        main_layout.addWidget(self.main_tabs)
        self.setLayout(main_layout)

    def get_text_config(self, use_nsfw=False):
        if use_nsfw:
            return (
                self.nsfw_url_input.text().strip(),
                self.nsfw_key_input.text().strip(),
                self.nsfw_model_combo.currentText().strip()
            )
        return (
            self.url_input.text().strip(),
            self.key_input.text().strip(),
            self.model_combo.currentText().strip()
        )

    def get_booru_tag_limit(self):
        try:
            return int(self.booru_tag_limit_spin.value())
        except Exception:
            return DEFAULT_BOORU_TAG_LIMIT

    def get_request_timeout_seconds(self):
        try:
            return int(self.img_timeout_spin.value())
        except Exception:
            return 120

    def on_single_nsfw_changed(self, checked):
        self.use_nsfw_single = bool(checked)
        self.save_text_config(silent=True)

    def on_batch_nsfw_changed(self, checked):
        self.use_nsfw_batch = bool(checked)
        self.save_text_config(silent=True)

    def get_upscale_options(self):
        return normalize_upscale_options(getattr(self, "upscale_options", {}))

    def update_upscale_options(self, options):
        self.upscale_options = normalize_upscale_options(options)
        if hasattr(self, "url_input") and hasattr(self, "key_input"):
            self.save_text_config(silent=True)

    def sync_selected_style(self, style_name):
        """【新增】同步多页面的画风下拉框，并自动保存到硬盘"""
        if not self._style_sync_enabled:
            return
        if not style_name: return
        self.last_used_style = style_name
        
        # 阻断信号避免死循环
        for combo in [self.single_analyzer_tab.main_style_combo, self.prompt_generator_tab.main_style_combo, self.batch_analyzer_tab.main_style_combo, self.image_edit_tab.main_style_combo, self.char_design_tab.main_style_combo, self.single_gen_debug_tab.main_style_combo]:
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
        self.use_nsfw_single = False
        self.use_nsfw_batch = False
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.url_input.setText(config.get("base_url", ""))
                    self.key_input.setText(config.get("api_key", ""))
                    self.nsfw_url_input.setText(config.get("nsfw_base_url", config.get("base_url", "")))
                    self.nsfw_key_input.setText(config.get("nsfw_api_key", ""))
                    # 【新增】读取上次保存的画风
                    self.last_used_style = config.get("last_used_style", "默认(无附加)")
                    self.use_nsfw_single = bool(config.get("use_nsfw_single", False))
                    self.use_nsfw_batch = bool(config.get("use_nsfw_batch", False))
                    self.upscale_options = normalize_upscale_options(config.get("upscale_options", {}))
                    saved_booru_tag_limit = config.get("booru_tag_limit", DEFAULT_BOORU_TAG_LIMIT)
                    try:
                        saved_booru_tag_limit = int(saved_booru_tag_limit)
                    except Exception:
                        saved_booru_tag_limit = DEFAULT_BOORU_TAG_LIMIT
                    if saved_booru_tag_limit <= 0:
                        saved_booru_tag_limit = DEFAULT_BOORU_TAG_LIMIT
                    self.booru_tag_limit_spin.blockSignals(True)
                    self.booru_tag_limit_spin.setValue(saved_booru_tag_limit)
                    self.booru_tag_limit_spin.blockSignals(False)
                    self.pic_cate_state = config.get("pic_cate", self.pic_cate_state)
                    if hasattr(self, "pic_cate_tab"):
                        self.pic_cate_tab.set_values(self.pic_cate_state)
                    
                    saved_model = config.get("model", "")
                    if saved_model:
                        self.model_combo.clear()
                        self.model_combo.addItem(saved_model)
                        self.model_combo.setCurrentText(saved_model)
                    saved_nsfw_model = config.get("nsfw_model", saved_model)
                    if saved_nsfw_model:
                        self.nsfw_model_combo.clear()
                        self.nsfw_model_combo.addItem(saved_nsfw_model)
                        self.nsfw_model_combo.setCurrentText(saved_nsfw_model)
            except Exception as e:
                print(f"加载 {CONFIG_FILE} 失败: {e}")
        if hasattr(self, "single_analyzer_tab"):
            self.single_analyzer_tab.set_use_nsfw_default(self.use_nsfw_single)
            self.single_analyzer_tab.set_upscale_options_defaults(self.upscale_options)
        if hasattr(self, "batch_analyzer_tab"):
            self.batch_analyzer_tab.set_use_nsfw_default(self.use_nsfw_batch)
            self.batch_analyzer_tab.set_upscale_options_defaults(self.upscale_options)
        if hasattr(self, "prompt_generator_tab"):
            self.prompt_generator_tab.set_upscale_options_defaults(self.upscale_options)
        if hasattr(self, "char_design_tab"):
            self.char_design_tab.set_upscale_options_defaults(self.upscale_options)
        if hasattr(self, "upscaler_tab"):
            self.upscaler_tab.load_saved_options()
                
        if os.path.exists(CONFIG_IMAGE_FILE):
            try:
                with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    # 读取当前API类型
                    current_api = config.get("current_api", "whatup")

                    # 【修改开始】阻断信号，避免初始化加载时触发保存，覆盖原有配置
                    self.api_type_combo.blockSignals(True)
                    self.api_type_combo.setCurrentText(current_api)
                    self._current_api_type = current_api  # 同步状态
                    self.api_type_combo.blockSignals(False)
                    # 【修改结束】
                    
                    # 读取对应API的配置
                    api_config = config.get("apis", {}).get(current_api, {})
                    self.img_url_input.setText(api_config.get("base_url", "https://api.whatai.cc/v1"))
                    self.img_key_input.setText(api_config.get("api_key", ""))
                    saved_model = api_config.get("model", "")
                    if saved_model:
                        if self.img_model_combo.findText(saved_model) == -1:
                            self.img_model_combo.addItem(saved_model)
                        self.img_model_combo.setCurrentText(saved_model)
                    saved_default_ar = api_config.get("default_aspect_ratio", DEFAULT_ASPECT_RATIO)
                    if self.default_ar_combo.findText(saved_default_ar) == -1:
                        self.default_ar_combo.addItem(saved_default_ar)
                    self.default_ar_combo.setCurrentText(saved_default_ar)

                    saved_first = api_config.get("override_aspect_ratio_first", "不覆盖(沿用原逻辑)")
                    if self.override_ar_first_combo.findText(saved_first) == -1:
                        self.override_ar_first_combo.addItem(saved_first)
                    self.override_ar_first_combo.setCurrentText(saved_first)

                    saved_second = api_config.get("override_aspect_ratio_second", "不覆盖(沿用原逻辑)")
                    if self.override_ar_second_combo.findText(saved_second) == -1:
                        self.override_ar_second_combo.addItem(saved_second)
                    self.override_ar_second_combo.setCurrentText(saved_second)

                    # ================= 新增：读取超时与重试 =================
                    saved_timeout = api_config.get("timeout", 120)
                    self.img_timeout_spin.setValue(saved_timeout)
                    
                    saved_retries = api_config.get("max_retries", 1)
                    self.img_retry_spin.setValue(saved_retries)

                    saved_debug_dump = bool(api_config.get("debug_dump_full_http", False))
                    self.img_debug_dump_checkbox.blockSignals(True)
                    self.img_debug_dump_checkbox.setChecked(saved_debug_dump)
                    self.img_debug_dump_checkbox.blockSignals(False)
                    
                    # 读取分辨率配置
                    saved_resolution = api_config.get("resolution", "1K")
                    if self.img_resolution_combo.findText(saved_resolution) == -1:
                        self.img_resolution_combo.addItem(saved_resolution)
                    self.img_resolution_combo.setCurrentText(saved_resolution)
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
            if self.last_used_style in keys:
                self.prompt_generator_tab.main_style_combo.setCurrentText(self.last_used_style)
                
        if hasattr(self, 'batch_analyzer_tab'):
            self.batch_analyzer_tab.update_styles(keys)
            if self.last_used_style in keys:
                self.batch_analyzer_tab.main_style_combo.setCurrentText(self.last_used_style)
                
        if hasattr(self, 'image_edit_tab'):
            self.image_edit_tab.update_styles(keys)
            if self.last_used_style in keys:
                self.image_edit_tab.main_style_combo.setCurrentText(self.last_used_style)
                
        if hasattr(self, 'char_design_tab'):
            self.char_design_tab.update_styles(keys)
            if self.last_used_style in keys:
                self.char_design_tab.main_style_combo.setCurrentText(self.last_used_style)
        if hasattr(self, 'single_gen_debug_tab'):
            self.single_gen_debug_tab.update_styles(keys)
            if self.last_used_style in keys:
                self.single_gen_debug_tab.main_style_combo.setCurrentText(self.last_used_style)

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
        if hasattr(self, "pic_cate_tab"):
            self.pic_cate_state = self.pic_cate_tab.get_values()
        config = {
            "base_url": self.url_input.text().strip(),
            "api_key": self.key_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "nsfw_base_url": self.nsfw_url_input.text().strip(),
            "nsfw_api_key": self.nsfw_key_input.text().strip(),
            "nsfw_model": self.nsfw_model_combo.currentText().strip(),
            "use_nsfw_single": bool(getattr(self, "use_nsfw_single", False)),
            "use_nsfw_batch": bool(getattr(self, "use_nsfw_batch", False)),
            "booru_tag_limit": int(self.get_booru_tag_limit()),
            "last_used_style": getattr(self, "last_used_style", "默认(无附加)"),
            "upscale_options": normalize_upscale_options(getattr(self, "upscale_options", {})),
            "pic_cate": self.pic_cate_state
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            if not silent:
                QMessageBox.information(self, "成功", f"配置已保存至 {CONFIG_FILE}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")

    def save_pic_cate_state(self, values):
        self.pic_cate_state = values or {
            "source_directory": "",
            "target_directory": "",
            "trimmed_directory": "",
            "train_name": ""
        }
        self.save_text_config(silent=True)

    def handle_batch_quick_export(self, json_paths):
        valid_paths = [os.path.abspath(path) for path in (json_paths or []) if os.path.isfile(path) and str(path).lower().endswith(".json")]
        if not valid_paths:
            QMessageBox.warning(self, "提示", "本次批量分析没有可导出的 JSON 文件")
            return
        output_dir = self._build_json_analy_output_dir(valid_paths)
        self.json_dataset_tab.prefill_for_batch(valid_paths, output_dir)
        self.main_tabs.setCurrentWidget(self.analysis_root_tab)
        self.analysis_tabs.setCurrentWidget(self.json_dataset_tab)

    def _build_json_analy_output_dir(self, json_paths):
        today = datetime.datetime.now().strftime("%Y%m%d")
        normalized = "|".join(sorted(set(os.path.abspath(path) for path in json_paths)))
        hash_value = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:10]
        return os.path.join(os.path.dirname(__file__), "data", today, "json-analy", hash_value)

    def handle_json_quick_split(self, source_dir):
        source_dir = os.path.abspath(source_dir or "")
        if not os.path.isdir(source_dir):
            QMessageBox.warning(self, "提示", "导出目录不存在，无法衔接图片分类切分")
            return
        parent_dir = os.path.dirname(source_dir)
        base_name = os.path.basename(source_dir.rstrip("\\/"))
        target_dir = os.path.join(parent_dir, f"{base_name}_cate-copy")
        trimmed_dir = os.path.join(parent_dir, f"{base_name}_trim-train")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(trimmed_dir, exist_ok=True)
        values = {
            "source_directory": source_dir,
            "target_directory": target_dir,
            "trimmed_directory": trimmed_dir,
            "train_name": f"{base_name}_train"
        }
        self.pic_cate_tab.set_values(values)
        self.save_pic_cate_state(values)
        self.main_tabs.setCurrentWidget(self.analysis_root_tab)
        self.analysis_tabs.setCurrentWidget(self.pic_cate_tab)
            
    def on_api_type_changed(self, api_type):
        """当API类型改变时，加载对应API的配置"""
        # 1. 先保存当前界面的配置（此时会安全地存入 self._current_api_type 对应的旧节点）
        self.save_image_config(silent=True)
        
        # 2. 【新增】更新跟踪变量为新的 API 类型
        self._current_api_type = api_type
        
        # 3. 【新增】临时阻断会自动触发保存的控件信号，防止渲染新数据时引发大量错误覆盖
        self.default_ar_combo.blockSignals(True)
        self.override_ar_first_combo.blockSignals(True)
        self.override_ar_second_combo.blockSignals(True)
        self.img_resolution_combo.blockSignals(True)
        
        # 读取配置文件并更新界面
        if os.path.exists(CONFIG_IMAGE_FILE):
            try:
                with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    
                    # 读取对应API的配置
                    api_config = config.get("apis", {}).get(api_type, {})
                    
                    # 根据API类型设置不同的默认base_url
                    if api_type == "aigc2d":
                        self.img_url_input.setText(api_config.get("base_url", ""))
                    elif api_type in ("openai-image", "aigc-2d-gpt"):
                        self.img_url_input.setText(api_config.get("base_url", "https://api.openai.com/v1"))
                    elif api_type == "openrouter-image":
                        self.img_url_input.setText(api_config.get("base_url", "https://openrouter.ai/api"))
                    else:
                        self.img_url_input.setText(api_config.get("base_url", "https://api.whatai.cc/v1"))
                    
                    self.img_key_input.setText(api_config.get("api_key", ""))
                    saved_model = api_config.get("model", "")
                    if saved_model:
                        if self.img_model_combo.findText(saved_model) == -1:
                            self.img_model_combo.addItem(saved_model)
                        self.img_model_combo.setCurrentText(saved_model)
                    else:
                        # 设置默认模型
                        if api_type == "whatup":
                            self.img_model_combo.setCurrentText("nano-banana-2")
                        elif api_type == "aigc2d":
                            self.img_model_combo.setCurrentText("")
                        elif api_type in ("openai-image", "aigc-2d-gpt"):
                            self.img_model_combo.setCurrentText("gpt-image-2")
                        elif api_type == "openrouter-image":
                            self.img_model_combo.setCurrentText("gpt-image-1")
                    
                    saved_default_ar = api_config.get("default_aspect_ratio", DEFAULT_ASPECT_RATIO)
                    if self.default_ar_combo.findText(saved_default_ar) == -1:
                        self.default_ar_combo.addItem(saved_default_ar)
                    self.default_ar_combo.setCurrentText(saved_default_ar)

                    saved_first = api_config.get("override_aspect_ratio_first", "不覆盖(沿用原逻辑)")
                    if self.override_ar_first_combo.findText(saved_first) == -1:
                        self.override_ar_first_combo.addItem(saved_first)
                    self.override_ar_first_combo.setCurrentText(saved_first)

                    saved_second = api_config.get("override_aspect_ratio_second", "不覆盖(沿用原逻辑)")
                    if self.override_ar_second_combo.findText(saved_second) == -1:
                        self.override_ar_second_combo.addItem(saved_second)
                    self.override_ar_second_combo.setCurrentText(saved_second)

                    # 读取超时与重试配置
                    saved_timeout = api_config.get("timeout", 120)
                    self.img_timeout_spin.setValue(saved_timeout)
                    
                    saved_retries = api_config.get("max_retries", 1)
                    self.img_retry_spin.setValue(saved_retries)

                    saved_debug_dump = bool(api_config.get("debug_dump_full_http", False))
                    self.img_debug_dump_checkbox.blockSignals(True)
                    self.img_debug_dump_checkbox.setChecked(saved_debug_dump)
                    self.img_debug_dump_checkbox.blockSignals(False)
                    
                    # 读取分辨率配置
                    saved_resolution = api_config.get("resolution", "1K")
                    if self.img_resolution_combo.findText(saved_resolution) == -1:
                        self.img_resolution_combo.addItem(saved_resolution)
                    self.img_resolution_combo.setCurrentText(saved_resolution)
            except Exception as e:
                print(f"加载 {CONFIG_IMAGE_FILE} 失败: {e}")

        # 4. 【新增】界面数据加载完毕，恢复信号阻断
        self.default_ar_combo.blockSignals(False)
        self.override_ar_first_combo.blockSignals(False)
        self.override_ar_second_combo.blockSignals(False)
        self.img_resolution_combo.blockSignals(False)

    def save_image_config(self, silent=False):
        current_api_global = self.api_type_combo.currentText()
        # 【新增】目标保存的API节点使用跟踪的变量，保障切换时数据存入旧节点
        target_api_node = getattr(self, "_current_api_type", current_api_global)
        
        # 读取现有配置
        existing_config = {}
        if os.path.exists(CONFIG_IMAGE_FILE):
            try:
                with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
            except Exception:
                pass
        
        # 更新配置
        api_config = {
            "base_url": self.img_url_input.text().strip() or "",
            "api_key": self.img_key_input.text().strip(),
            "model": self.img_model_combo.currentText().strip(),
            "default_aspect_ratio": self.default_ar_combo.currentText().strip() or DEFAULT_ASPECT_RATIO,
            "override_aspect_ratio_first": self.override_ar_first_combo.currentText().strip() or "不覆盖(沿用原逻辑)",
            "override_aspect_ratio_second": self.override_ar_second_combo.currentText().strip() or "不覆盖(沿用原逻辑)",
            "timeout": self.img_timeout_spin.value(),
            "max_retries": self.img_retry_spin.value(),
            "debug_dump_full_http": bool(self.img_debug_dump_checkbox.isChecked()),
            "resolution": self.img_resolution_combo.currentText().strip() or "2K",
        }
        
        config = {
            "current_api": current_api_global,
            "apis": existing_config.get("apis", {})
        }
        # 【修改】将数据保存到正确的节点 target_api_node 下
        config["apis"][target_api_node] = api_config
        try:
            with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            if not silent:
                QMessageBox.information(self, "成功", f"生图配置已保存至 {CONFIG_IMAGE_FILE}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")

    def _fetch_models_for(self, api_key, base_url, model_combo, fetch_btn):
        if not api_key:
            QMessageBox.warning(self, "错误", "请先输入文本分析的 API Key")
            return

        fetch_btn.setEnabled(False)
        fetch_btn.setText("获取中...")
        QApplication.processEvents()

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
            model_names = sorted([m.id for m in models.data])
            
            current_text = model_combo.currentText()
            model_combo.clear()
            model_combo.addItems(model_names)
            
            index = model_combo.findText(current_text)
            if index >= 0: model_combo.setCurrentIndex(index)
            elif current_text: model_combo.setCurrentText(current_text)
                
            QMessageBox.information(self, "成功", f"成功获取 {len(model_names)} 个可用模型！")
        except Exception as e:
            QMessageBox.warning(self, "获取失败", f"获取模型列表失败，请检查 URL 和 Key 是否正确。\n错误信息: {e}")
        finally:
            fetch_btn.setEnabled(True)
            fetch_btn.setText("获取模型列表")

    def fetch_models(self):
        self._fetch_models_for(
            api_key=self.key_input.text().strip(),
            base_url=self.url_input.text().strip(),
            model_combo=self.model_combo,
            fetch_btn=self.fetch_btn
        )

    def fetch_nsfw_models(self):
        self._fetch_models_for(
            api_key=self.nsfw_key_input.text().strip(),
            base_url=self.nsfw_url_input.text().strip(),
            model_combo=self.nsfw_model_combo,
            fetch_btn=self.fetch_nsfw_btn
        )

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
