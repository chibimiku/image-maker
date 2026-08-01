import os
import json
import hashlib
import datetime
from utils.gui_entry import redirect_stdio_for_windows_gui_entry, warm_up_optional_module

redirect_stdio_for_windows_gui_entry()
warm_up_optional_module("onnxruntime", skip_env_var="IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD")
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QInputDialog,
                             QComboBox, QFormLayout, QMessageBox, QTabWidget, QCheckBox,
                             QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal
from openai import OpenAI

# 引入抽离出去的独立组件
from modules.image_analysis.style_analyzer import StyleAnalyzerWidget
from modules.image_analysis.single_analyzer import SingleAnalyzerWidget
# 【新增】引入批量提示词生成组件
from modules.image_generation.prompt_generator import PromptGeneratorWidget
# 【新增】引入批量图片分析组件
from modules.image_analysis.batch_analyzer import BatchAnalyzerWidget
# 【新增】引入图片编辑组件
from modules.image_generation.image_edit import ImageEditWidget
# 【新增】引入角色设计组件
from modules.image_generation.char_design import CharDesignWidget
from modules.image_analysis.pic_cate_tab import PicCateWidget
from modules.image_analysis.json_dataset_tab import JsonDatasetWidget
from modules.image_generation.webp_compressor import DragDropCompressor
from modules.image_generation.flux2_client_tab import Flux2ClientWidget
from modules.image_generation.upscaler_tab import UpscalerTabWidget
from modules.image_generation.single_gen_debug_tab import SingleGenDebugWidget, DropLineEdit, CompressPromptThread
from modules.image_generation.sd_workflow_tab import SdWebuiSettingsWidget, SdWorkflowWidget
from modules.others.booru_tag_generator import BooruTagGeneratorWidget
from modules.fashion_collection.collector_tab import FashionCollectorWidget
from modules.image_generation.diff_cg_tab import DiffCgTabWidget
from utils.image_upscale_runtime import normalize_upscale_options
from utils.styles import normalize_style_entry, build_style_entry, ref_image_valid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "conf", "config.json")
CONFIG_IMAGE_FILE = os.path.join(BASE_DIR, "conf", "config-image.json")
CONFIG_STYLES_FILE = os.path.join(BASE_DIR, "conf", "config-styles.json")
DEFAULT_ASPECT_RATIO = "1:1"
ASPECT_RATIO_OPTIONS = ["不覆盖(沿用原逻辑)", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
NO_OVERRIDE_TEXT = "不覆盖(沿用原逻辑)"
DEFAULT_BOORU_TAG_LIMIT = 30


DEFAULT_STYLES = {
    "默认(无附加)": ""
}


class FilterableComboBox(QComboBox):
    """支持输入文字实时筛选下拉列表的 QComboBox"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self._all_items = []
        self._filter_text = ""
        self.lineEdit().textEdited.connect(self._on_text_edited)

    def setItems(self, items):
        """一次性设置所有候选项"""
        self._all_items = list(items) if items else []
        self._rebuild_popup()

    def addItem(self, text, userData=None):
        if text not in self._all_items:
            self._all_items.append(text)
        self._rebuild_popup()

    def addItems(self, items):
        """追加候选项（同时更新全量列表）"""
        if not isinstance(items, (list, tuple)):
            super().addItems(items)
            return
        for item in items:
            if item not in self._all_items:
                self._all_items.append(item)
        self._rebuild_popup()

    def clear(self):
        self._all_items.clear()
        self._filter_text = ""
        super().clear()

    def _rebuild_popup(self):
        """根据当前筛选文字重建下拉列表"""
        self.blockSignals(True)
        super().clear()
        self.blockSignals(False)
        current = self.currentText()
        filter_lower = self._filter_text.lower()
        if filter_lower:
            filtered = [item for item in self._all_items if filter_lower in item.lower()]
        else:
            filtered = list(self._all_items)
        if filtered:
            self.addItems_to_popup(filtered)
            if current and self.findText(current) >= 0:
                pass  # keep current
        elif self._all_items:
            # 无匹配项时显示全部
            self.addItems_to_popup(list(self._all_items))

    def addItems_to_popup(self, items):
        """直接向底层 QComboBox 添加条目（绕过 addItems 重写）"""
        super().addItems(items)

    def _on_text_edited(self, text):
        self._filter_text = text
        self._rebuild_popup()
        self.showPopup()

    def showPopup(self):
        self._rebuild_popup()
        super().showPopup()

    def currentText(self):
        return self.lineEdit().text() if self.lineEdit() else super().currentText()

    def setCurrentText(self, text):
        self.lineEdit().setText(text)

    def findText(self, text, flags=None):
        # 在全量列表中查找
        for i, item in enumerate(self._all_items):
            if item == text:
                return i
        return -1


class ModelFetchThread(QThread):
    success_signal = pyqtSignal(list, str)
    error_signal = pyqtSignal(str, str)

    def __init__(self, api_key, base_url, current_text, request_key):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.current_text = current_text
        self.request_key = request_key

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            models = client.models.list()
            model_names = sorted([m.id for m in models.data])
            self.success_signal.emit(model_names, self.current_text)
        except Exception as e:
            self.error_signal.emit(str(e), self.current_text)

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
        self.enable_outfit_check_single = False
        self.enable_outfit_check_batch = False
        self.remove_photo_style_single = False
        self.style_analyzer_test_gen = True
        self.style_analyzer_test_prompt = ""
        self.outfit_style_override_single = ""
        self.outfit_style_override_batch = ""
        self.outfit_style_override_history = []
        self.upscale_options = normalize_upscale_options({})
        self._model_fetch_threads = {}

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
        self.others_root_tab = QWidget()
        self.others_tabs = QTabWidget()
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
            upscale_options_changed_callback=self.update_upscale_options,
            outfit_check_default_getter_func=lambda: self.enable_outfit_check_single,
            outfit_check_changed_callback=self.on_single_outfit_check_changed,
            remove_photo_style_default_getter_func=lambda: self.remove_photo_style_single,
            remove_photo_style_changed_callback=self.on_single_remove_photo_style_changed,
            outfit_style_history_getter_func=self.get_outfit_style_history,
            outfit_style_default_getter_func=self.get_single_outfit_style_override,
            outfit_style_changed_callback=self.update_single_outfit_style_override,
            outfit_style_delete_callback=self.delete_outfit_style_history_item,
            styles_reload_callback=self.load_styles_config
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
            upscale_options_changed_callback=self.update_upscale_options,
            outfit_check_default_getter_func=lambda: self.enable_outfit_check_batch,
            outfit_check_changed_callback=self.on_batch_outfit_check_changed,
            outfit_style_history_getter_func=self.get_outfit_style_history,
            outfit_style_default_getter_func=self.get_batch_outfit_style_override,
            outfit_style_changed_callback=self.update_batch_outfit_style_override,
            outfit_style_delete_callback=self.delete_outfit_style_history_item
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
            ar_policy_getter_func=self.get_ar_policy_config,
            styles_reload_callback=self.load_styles_config
        )
        self.single_gen_debug_tab.main_style_combo.currentTextChanged.connect(self.sync_selected_style)
        # z-image 当前默认不展示，避免主窗口启动时提前触发重型环境探测
        self.z_image_edit_tab = None

        # 【Tab 2: 多图画风提取】
        self.style_analyzer_tab = StyleAnalyzerWidget(
            config_getter_func=self.get_text_config,
            timeout_getter_func=self.get_request_timeout_seconds,
            img_config_getter_func=lambda: (self.img_url_input.text().strip(), self.img_key_input.text().strip(), self.img_model_combo.currentText().strip(), self.api_type_combo.currentText()),
            styles_getter_func=lambda: self.styles_data,
            test_gen_default_getter_func=lambda: self.style_analyzer_test_gen,
            test_gen_changed_callback=self.on_style_analyzer_test_gen_changed,
            test_prompt_getter_func=lambda: self.style_analyzer_test_prompt,
            test_prompt_changed_callback=self.on_style_analyzer_test_prompt_changed,
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
        self.sd_webui_settings_tab = SdWebuiSettingsWidget()
        self.sd_workflow_tab = SdWorkflowWidget(
            text_config_getter_func=self.get_text_config,
            sd_webui_settings_getter_func=self.get_sd_webui_settings,
            styles_getter_func=lambda: self.styles_data,
            current_style_name_getter_func=lambda: self.last_used_style,
            style_changed_callback=self.sync_selected_style,
        )
        self.diff_cg_tab = DiffCgTabWidget(
            text_config_getter_func=self.get_text_config
        )
        self.booru_tag_generator_tab = BooruTagGeneratorWidget()
        self.fashion_collector_tab = FashionCollectorWidget(project_root=BASE_DIR)

        analysis_layout = QVBoxLayout()
        analysis_layout.addWidget(self.analysis_tabs)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_root_tab.setLayout(analysis_layout)

        generation_layout = QVBoxLayout()
        generation_layout.addWidget(self.generation_tabs)
        generation_layout.setContentsMargins(0, 0, 0, 0)
        self.generation_root_tab.setLayout(generation_layout)

        others_layout = QVBoxLayout()
        others_layout.addWidget(self.others_tabs)
        others_layout.setContentsMargins(0, 0, 0, 0)
        self.others_root_tab.setLayout(others_layout)

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
        self.generation_tabs.addTab(self.sd_workflow_tab, "SD 批量工作流")
        self.generation_tabs.addTab(self.diff_cg_tab, "差分CG生成")

        self.others_tabs.addTab(self.booru_tag_generator_tab, "生成booru-tag")
        self.others_tabs.addTab(self.fashion_collector_tab, "服饰素材采集")

        self.main_tabs.addTab(self.analysis_root_tab, "图片分析")
        self.main_tabs.addTab(self.generation_root_tab, "图片生成")
        self.main_tabs.addTab(self.others_root_tab, "其他")

        # 【Tab 8: 全局配置】
        self.config_tabs = QTabWidget()
        
        # 3.1 文本分析配置
        tab_text = QWidget()
        text_layout = QFormLayout()
        self.url_input = QLineEdit()
        text_layout.addRow("Base URL:", self.url_input)
        
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        text_layout.addRow("API Key:", self.key_input)
        
        model_layout = QHBoxLayout()
        self.model_combo = FilterableComboBox()
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
        self.nsfw_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.nsfw_key_input.editingFinished.connect(lambda: self.save_text_config(silent=True))
        text_nsfw_layout.addRow("API Key:", self.nsfw_key_input)

        nsfw_model_layout = QHBoxLayout()
        self.nsfw_model_combo = FilterableComboBox()
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
        self.img_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        image_layout.addRow("API Key:", self.img_key_input)
        
        self.img_model_combo = FilterableComboBox()
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
        self.img_url_input.editingFinished.connect(lambda: self.save_image_config(silent=True))
        self.img_key_input.editingFinished.connect(lambda: self.save_image_config(silent=True))
        self.img_model_combo.activated.connect(lambda _idx: self.save_image_config(silent=True))
        if self.img_model_combo.lineEdit() is not None:
            self.img_model_combo.lineEdit().editingFinished.connect(lambda: self.save_image_config(silent=True))
        self.default_ar_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.override_ar_first_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.override_ar_second_combo.currentTextChanged.connect(lambda: self.save_image_config(silent=True))
        self.img_timeout_spin.valueChanged.connect(lambda _v: self.save_image_config(silent=True))
        self.img_retry_spin.valueChanged.connect(lambda _v: self.save_image_config(silent=True))
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

        style_comp_header = QHBoxLayout()
        style_comp_header.addWidget(QLabel("压缩版指令(参考优先模式用):"))
        style_comp_header.addStretch()
        self.style_compress_btn = QPushButton("请求 LLM 重新生成")
        self.style_compress_btn.setToolTip("调用文本 API（conf/config.json）把上面的完整指令压缩为精简版，填入本框")
        self.style_compress_btn.clicked.connect(self.regenerate_style_compressed)
        style_comp_header.addWidget(self.style_compress_btn)
        style_layout.addLayout(style_comp_header)
        self.style_compressed_edit = QTextEdit()
        self.style_compressed_edit.setPlaceholderText(
            "参考优先模式下替代完整指令的精简版；可留空（留空时自动用本地启发式压缩）。"
            "可用 tools/compress_styles.py 批量由 LLM 生成。"
        )
        self.style_compressed_edit.setMaximumHeight(130)
        style_layout.addWidget(self.style_compressed_edit)

        style_ref_row = QHBoxLayout()
        style_ref_row.addWidget(QLabel("参考图(仅画风):"))
        self.style_ref_image_edit = DropLineEdit()
        self.style_ref_image_edit.setAcceptDrops(True)
        self.style_ref_image_edit.setPlaceholderText("可留空；样例图路径，生成时仅参考其画风（支持拖拽图片到此处）")
        self.style_ref_image_edit.setToolTip(
            "该画风预设的「艺术风格参考图」。生成时作为附件传给 API，且只参考其画风"
            "（线条/上色/光影/配色/渲染惯例），不参考人物、服装、姿势、场景等内容。"
        )
        style_ref_row.addWidget(self.style_ref_image_edit, stretch=1)
        self.style_ref_browse_btn = QPushButton("浏览...")
        self.style_ref_browse_btn.clicked.connect(self.browse_style_ref_image)
        style_ref_row.addWidget(self.style_ref_browse_btn)
        self.style_ref_clear_btn = QPushButton("清除")
        self.style_ref_clear_btn.clicked.connect(self.clear_style_ref_image)
        style_ref_row.addWidget(self.style_ref_clear_btn)
        style_layout.addLayout(style_ref_row)

        self.save_style_btn = QPushButton("保存当前预设")
        self.save_style_btn.clicked.connect(self.save_current_style)
        style_layout.addWidget(self.save_style_btn)
        tab_style.setLayout(style_layout)

        self.config_tabs.addTab(tab_text, "文本分析 API")
        self.config_tabs.addTab(tab_text_nsfw, "文本分析（NSFW）")
        self.config_tabs.addTab(tab_image, "图片生成 API")
        self.config_tabs.addTab(self.sd_webui_settings_tab, "SD-WebUI接口配置")
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
        if not hasattr(self, "url_input") or not hasattr(self, "model_combo"):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                config = {}
            if use_nsfw:
                return (
                    str(config.get("nsfw_base_url", config.get("base_url", "")) or "").strip(),
                    str(config.get("nsfw_api_key", "") or "").strip(),
                    str(config.get("nsfw_model", config.get("model", "")) or "").strip(),
                )
            return (
                str(config.get("base_url", "") or "").strip(),
                str(config.get("api_key", "") or "").strip(),
                str(config.get("model", "") or "").strip(),
            )
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

    def get_sd_webui_settings(self):
        if hasattr(self, "sd_webui_settings_tab"):
            return self.sd_webui_settings_tab.get_settings()
        return {}

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

    def get_outfit_style_history(self):
        return list(getattr(self, "outfit_style_override_history", []))

    def get_single_outfit_style_override(self):
        return str(getattr(self, "outfit_style_override_single", "") or "").strip()

    def get_batch_outfit_style_override(self):
        return str(getattr(self, "outfit_style_override_batch", "") or "").strip()

    def _normalize_outfit_style_history(self, values):
        normalized = []
        for value in (values or []):
            text = str(value or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized[:100]

    def _refresh_outfit_style_widgets(self):
        history = self.get_outfit_style_history()
        if hasattr(self, "single_analyzer_tab"):
            self.single_analyzer_tab.set_outfit_style_options(history, self.get_single_outfit_style_override())
        if hasattr(self, "batch_analyzer_tab"):
            self.batch_analyzer_tab.set_outfit_style_options(history, self.get_batch_outfit_style_override())

    def _update_outfit_style_override(self, attr_name, text, add_to_history=False):
        value = str(text or "").strip()
        setattr(self, attr_name, value)
        history = self._normalize_outfit_style_history(getattr(self, "outfit_style_override_history", []))
        if add_to_history and value:
            history = [item for item in history if item != value]
            history.insert(0, value)
        self.outfit_style_override_history = self._normalize_outfit_style_history(history)
        self._refresh_outfit_style_widgets()
        self.save_text_config(silent=True)

    def update_single_outfit_style_override(self, text, add_to_history=False):
        self._update_outfit_style_override("outfit_style_override_single", text, add_to_history=add_to_history)

    def update_batch_outfit_style_override(self, text, add_to_history=False):
        self._update_outfit_style_override("outfit_style_override_batch", text, add_to_history=add_to_history)

    def delete_outfit_style_history_item(self, text):
        value = str(text or "").strip()
        if not value:
            return
        history = [item for item in self.get_outfit_style_history() if item != value]
        self.outfit_style_override_history = self._normalize_outfit_style_history(history)
        if self.get_single_outfit_style_override() == value:
            self.outfit_style_override_single = ""
        if self.get_batch_outfit_style_override() == value:
            self.outfit_style_override_batch = ""
        self._refresh_outfit_style_widgets()
        self.save_text_config(silent=True)

    def on_single_outfit_check_changed(self, checked):
        self.enable_outfit_check_single = bool(checked)
        self.save_text_config(silent=True)

    def on_batch_outfit_check_changed(self, checked):
        self.enable_outfit_check_batch = bool(checked)
        self.save_text_config(silent=True)

    def on_single_remove_photo_style_changed(self, checked):
        self.remove_photo_style_single = bool(checked)
        self.save_text_config(silent=True)

    def on_style_analyzer_test_gen_changed(self, checked):
        self.style_analyzer_test_gen = bool(checked)
        self.save_text_config(silent=True)

    def on_style_analyzer_test_prompt_changed(self, text):
        self.style_analyzer_test_prompt = str(text).strip()
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
        for combo in [
            self.single_analyzer_tab.main_style_combo,
            self.prompt_generator_tab.main_style_combo,
            self.batch_analyzer_tab.main_style_combo,
            self.image_edit_tab.main_style_combo,
            self.char_design_tab.main_style_combo,
            self.single_gen_debug_tab.main_style_combo,
            self.sd_workflow_tab.style_combo,
        ]:
            if combo.currentText() != style_name:
                combo.blockSignals(True)
                combo.setCurrentText(style_name)
                combo.blockSignals(False)

        if hasattr(self, 'sd_workflow_tab') and hasattr(self.sd_workflow_tab, 'sync_style_from_external'):
            self.sd_workflow_tab.sync_style_from_external(style_name)

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
        self.enable_outfit_check_single = False
        self.enable_outfit_check_batch = False
        self.remove_photo_style_single = False
        self.outfit_style_override_single = ""
        self.outfit_style_override_batch = ""
        self.outfit_style_override_history = []
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
                    self.enable_outfit_check_single = bool(config.get("enable_outfit_check_single", False))
                    self.enable_outfit_check_batch = bool(config.get("enable_outfit_check_batch", False))
                    self.remove_photo_style_single = bool(config.get("remove_photo_style_single", False))
                    self.style_analyzer_test_gen = bool(config.get("style_analyzer_test_gen", True))
                    self.style_analyzer_test_prompt = str(config.get("style_analyzer_test_prompt", "") or "").strip()
                    self.outfit_style_override_single = str(config.get("outfit_style_override_single", "") or "").strip()
                    self.outfit_style_override_batch = str(config.get("outfit_style_override_batch", "") or "").strip()
                    self.outfit_style_override_history = self._normalize_outfit_style_history(config.get("outfit_style_override_history", []))
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
                    
                    # 恢复缓存的模型列表（先恢复缓存列表，再设置选中值，避免 setItems 覆盖选中）
                    self._cached_models = config.get("cached_models", [])
                    self._cached_nsfw_models = config.get("cached_nsfw_models", [])
                    self._restore_cached_models()

                    saved_model = config.get("model", "")
                    if saved_model:
                        self.model_combo.addItem(saved_model)
                        self.model_combo.setCurrentText(saved_model)
                    saved_nsfw_model = config.get("nsfw_model", saved_model)
                    if saved_nsfw_model:
                        self.nsfw_model_combo.addItem(saved_nsfw_model)
                        self.nsfw_model_combo.setCurrentText(saved_nsfw_model)
            except Exception as e:
                print(f"加载 {CONFIG_FILE} 失败: {e}")
        if hasattr(self, "single_analyzer_tab"):
            self.single_analyzer_tab.set_use_nsfw_default(self.use_nsfw_single)
            self.single_analyzer_tab.set_outfit_check_default(self.enable_outfit_check_single)
            self.single_analyzer_tab.set_remove_photo_style_default(self.remove_photo_style_single)
            self.single_analyzer_tab.set_upscale_options_defaults(self.upscale_options)
        if hasattr(self, "style_analyzer_tab"):
            self.style_analyzer_tab.set_test_gen_default(self.style_analyzer_test_gen)
            self.style_analyzer_tab.set_test_prompt_default(self.style_analyzer_test_prompt)
        if hasattr(self, "batch_analyzer_tab"):
            self.batch_analyzer_tab.set_use_nsfw_default(self.use_nsfw_batch)
            self.batch_analyzer_tab.set_outfit_check_default(self.enable_outfit_check_batch)
            self.batch_analyzer_tab.set_upscale_options_defaults(self.upscale_options)
        self._refresh_outfit_style_widgets()
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
                    if current_api == "aigc2d":
                        self.img_url_input.setText(api_config.get("base_url", "https://next.aigc2d.com/v1beta/models/"))
                    elif current_api in ("openai-image", "aigc-2d-gpt"):
                        self.img_url_input.setText(api_config.get("base_url", "https://api.openai.com/v1"))
                    elif current_api == "openrouter-image":
                        self.img_url_input.setText(api_config.get("base_url", "https://openrouter.ai/api"))
                    else:
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
                    # 恢复缓存的图片生成模型列表
                    self._cached_image_models = config.get("cached_image_models", [])
                    if self._cached_image_models:
                        self.img_model_combo.setItems(self._cached_image_models)
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
        if hasattr(self, 'sd_workflow_tab'):
            self.sd_workflow_tab.update_styles(keys, self.last_used_style)

    def on_manage_style_changed(self, style_name):
        if style_name in self.styles_data:
            entry = normalize_style_entry(self.styles_data[style_name])
            self.style_content_edit.setPlainText(entry["prompt"])
            self.style_compressed_edit.setPlainText(entry["prompt_compressed"])
            self.style_ref_image_edit.setText(entry["ref_image"])

    def browse_style_ref_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图（仅参考画风）", "", "图片文件 (*.png *.jpg *.jpeg *.webp *.gif *.bmp)"
        )
        if file_path:
            self.style_ref_image_edit.setText(file_path)

    def regenerate_style_compressed(self):
        prompt_text = self.style_content_edit.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "提示", "指令文本为空，无法生成压缩版。")
            return
        self.style_compress_btn.setEnabled(False)
        self.style_compress_btn.setText("生成中...")
        self._style_compress_thread = CompressPromptThread(prompt_text, parent=self)
        self._style_compress_thread.finished_ok.connect(self._on_style_compressed_ok)
        self._style_compress_thread.failed.connect(self._on_style_compressed_failed)
        self._style_compress_thread.finished.connect(self._on_style_compress_done)
        self._style_compress_thread.start()

    def _on_style_compressed_ok(self, text):
        self.style_compressed_edit.setPlainText(text)
        QMessageBox.information(self, "完成", f"已生成压缩版指令（{len(text)} 字符）。")

    def _on_style_compressed_failed(self, err_msg):
        QMessageBox.warning(self, "生成失败", err_msg)

    def _on_style_compress_done(self):
        self.style_compress_btn.setEnabled(True)
        self.style_compress_btn.setText("请求 LLM 重新生成")

    def clear_style_ref_image(self):
        self.style_ref_image_edit.clear()

    def save_current_style(self):
        style_name = self.style_manage_combo.currentText()
        if not style_name: return
        ref_image = self.style_ref_image_edit.text().strip()
        if ref_image and not ref_image_valid(ref_image):
            QMessageBox.warning(self, "提示", f"参考图文件不存在，无法保存：\n{ref_image}")
            return
        self.styles_data[style_name] = build_style_entry(
            self.style_content_edit.toPlainText().strip(),
            ref_image,
            self.style_compressed_edit.toPlainText().strip()
        )
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
            self.styles_data[name] = build_style_entry("")
            self.update_style_combos()
            self.style_manage_combo.setCurrentText(name)

    def delete_current_style(self):
        style_name = self.style_manage_combo.currentText()
        if not style_name or style_name == "默认(无附加)":
            QMessageBox.warning(self, "提示", "无法删除默认预设！")
            return
        reply = QMessageBox.question(
            self,
            '确认删除',
            f"确定要删除 '{style_name}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
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
            "enable_outfit_check_single": bool(getattr(self, "enable_outfit_check_single", False)),
            "enable_outfit_check_batch": bool(getattr(self, "enable_outfit_check_batch", False)),
            "remove_photo_style_single": bool(getattr(self, "remove_photo_style_single", False)),
            "style_analyzer_test_gen": bool(getattr(self, "style_analyzer_test_gen", True)),
            "style_analyzer_test_prompt": str(getattr(self, "style_analyzer_test_prompt", "") or "").strip(),
            "outfit_style_override_single": self.get_single_outfit_style_override(),
            "outfit_style_override_batch": self.get_batch_outfit_style_override(),
            "outfit_style_override_history": self._normalize_outfit_style_history(getattr(self, "outfit_style_override_history", [])),
            "booru_tag_limit": int(self.get_booru_tag_limit()),
            "last_used_style": getattr(self, "last_used_style", "默认(无附加)"),
            "upscale_options": normalize_upscale_options(getattr(self, "upscale_options", {})),
            "pic_cate": self.pic_cate_state,
            "cached_models": getattr(self, "_cached_models", []),
            "cached_nsfw_models": getattr(self, "_cached_nsfw_models", []),
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

    def _save_model_cache(self, model_names, request_key):
        """将模型列表缓存在内存并触发配置保存"""
        if request_key == "text":
            self._cached_models = list(model_names or [])
            self.save_text_config(silent=True)
        elif request_key == "nsfw":
            self._cached_nsfw_models = list(model_names or [])
            self.save_text_config(silent=True)
        elif request_key == "image":
            self._cached_image_models = list(model_names or [])
            self.save_image_config(silent=True)

    def _restore_cached_models(self):
        """从配置文件中恢复缓存的模型列表到 FilterableComboBox"""
        # 文本分析模型
        if hasattr(self, "_cached_models") and self._cached_models:
            self.model_combo.setItems(self._cached_models)
        # NSFW 分析模型
        if hasattr(self, "_cached_nsfw_models") and self._cached_nsfw_models:
            self.nsfw_model_combo.setItems(self._cached_nsfw_models)
        # 图片生成模型
        if hasattr(self, "_cached_image_models") and self._cached_image_models:
            self.img_model_combo.setItems(self._cached_image_models)

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
        self.img_timeout_spin.blockSignals(True)
        self.img_retry_spin.blockSignals(True)
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
                        self.img_url_input.setText(api_config.get("base_url", "https://next.aigc2d.com/v1beta/models/"))
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
        self.img_timeout_spin.blockSignals(False)
        self.img_retry_spin.blockSignals(False)
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

        # 保留 config-image.json 其他顶层节点（如 webui_img2img / diff_cg）
        config = existing_config if isinstance(existing_config, dict) else {}
        if not isinstance(config.get("apis"), dict):
            config["apis"] = {}
        config["current_api"] = current_api_global
        # 【修改】将数据保存到正确的节点 target_api_node 下
        config["apis"][target_api_node] = api_config
        config["cached_image_models"] = getattr(self, "_cached_image_models", [])
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

        request_key = "nsfw" if fetch_btn is self.fetch_nsfw_btn else "text"
        if request_key in self._model_fetch_threads:
            QMessageBox.information(self, "提示", "模型列表正在获取中，请稍候。")
            return

        fetch_btn.setEnabled(False)
        fetch_btn.setText("获取中...")
        model_combo.setEnabled(False)
        thread = ModelFetchThread(
            api_key=api_key,
            base_url=base_url,
            current_text=model_combo.currentText(),
            request_key=request_key
        )
        self._model_fetch_threads[request_key] = thread
        thread.success_signal.connect(
            lambda model_names, current_text, combo=model_combo, btn=fetch_btn, key=request_key:
            self._on_fetch_models_success(key, combo, btn, model_names, current_text)
        )
        thread.error_signal.connect(
            lambda error_text, _current_text, combo=model_combo, btn=fetch_btn, key=request_key:
            self._on_fetch_models_error(key, combo, btn, error_text)
        )
        thread.finished.connect(lambda key=request_key: self._cleanup_fetch_models_thread(key))
        thread.start()

    def _on_fetch_models_success(self, request_key, model_combo, fetch_btn, model_names, current_text):
        if request_key not in self._model_fetch_threads:
            return
        model_combo.clear()
        model_combo.addItems(model_names)

        if current_text and any(current_text == m for m in model_names):
            model_combo.setCurrentText(current_text)

        model_combo.setEnabled(True)
        fetch_btn.setEnabled(True)
        fetch_btn.setText("获取模型列表")
        # 保存缓存
        self._save_model_cache(model_names, request_key)
        QMessageBox.information(self, "成功", f"成功获取 {len(model_names)} 个可用模型！（已缓存到本地）")

    def _on_fetch_models_error(self, request_key, model_combo, fetch_btn, error_text):
        if request_key not in self._model_fetch_threads:
            return
        model_combo.setEnabled(True)
        fetch_btn.setEnabled(True)
        fetch_btn.setText("获取模型列表")
        QMessageBox.warning(self, "获取失败", f"获取模型列表失败，请检查 URL 和 Key 是否正确。\n错误信息: {error_text}")

    def _cleanup_fetch_models_thread(self, request_key):
        thread = self._model_fetch_threads.pop(request_key, None)
        if thread is not None:
            thread.deleteLater()

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

    # 支持 --auto-quit N ：启动后 N 秒自动退出，避免测试时卡住终端
    auto_quit_seconds = 0
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--auto-quit':
        try:
            auto_quit_seconds = int(args[1])
        except ValueError:
            auto_quit_seconds = 0
        sys.argv = [sys.argv[0]] + args[2:]  # 清理参数，避免 Qt 解析报错

    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()

    if auto_quit_seconds > 0:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(int(auto_quit_seconds * 1000), app.quit)

    sys.exit(app.exec())
