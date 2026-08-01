"""SD 批量工作流 - 主题批量生成 Tab

按主题 + 模板批量生成图片的工作流，步骤从上到下：
1. 填写主题与风格
2. 编辑提示词模板
3. 设置批量参数
4. 启动生成
5. 查看输出
"""

import json
import os
from utils.styles import style_prompt

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.image_generation.sd_workflow_core import (
    BASE_DIR,
    PROMPTS_DIR,
    NEG_PROMPTS_DIR,
)


# 步骤定义：(标题, 描述)
THEME_BATCH_STEPS = [
    ("步骤 1：填写主题与风格", "选择绘画主题与 Prompt 风格预设"),
    ("步骤 2：编辑提示词模板", "编辑正向 / 反向提示词模板"),
    ("步骤 3：设置批量参数", "设置大模型请求轮数(Y)与单次返回组数(X)"),
    ("步骤 4：启动批量生成", "保存配置并开始批量生成"),
    ("步骤 5：查看输出", "在日志区查看进度，输出文件落到 outputs 目录"),
]


class SdThemeBatchTab(QWidget):
    """主题批量生成 Tab。"""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    style_changed = pyqtSignal(str)
    theme_edited = pyqtSignal(str)
    config_changed = pyqtSignal()

    def __init__(self, styles_getter, current_style_name_getter, tag_manager, parent=None):
        super().__init__(parent)
        self.styles_getter = styles_getter
        self.current_style_name_getter = current_style_name_getter
        self.tag_manager = tag_manager
        self.style_options = {"默认(无附加)": ""}
        self._workflow_done = False
        self._worker_running = False
        self._building_ui = False
        self._syncing = False
        self._config_ref = None
        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        layout = QVBoxLayout(self)

        # 步骤指示器
        self._step_group, self._step_labels = self._build_step_indicator()
        layout.addWidget(self._step_group)

        # 步骤 1：主题与风格
        layout.addWidget(self._build_theme_style_group())

        # 步骤 2：模板
        layout.addWidget(self._build_template_group())

        # 步骤 3：批量参数
        layout.addWidget(self._build_batch_params_group())

        # 步骤 4：启动
        layout.addLayout(self._build_action_layout())

        layout.addStretch()

    def _build_step_indicator(self):
        group = QGroupBox("工作流进度")
        v = QVBoxLayout()
        labels = []
        for title, desc in THEME_BATCH_STEPS:
            lbl = QLabel()
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            labels.append(lbl)
        group.setLayout(v)
        return group, labels

    def _build_theme_style_group(self):
        group = QGroupBox("步骤 1：填写主题与风格")
        form = QFormLayout()

        self.theme_input = QLineEdit()
        self.theme_input.setPlaceholderText("例如：中秋主题少女、赛博朋克城市...")
        self.theme_input.textEdited.connect(self._on_theme_edited)
        form.addRow("绘画主题 (必填):", self.theme_input)

        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(200)
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        form.addRow("Prompt 风格预设:", self.style_combo)

        group.setLayout(form)
        return group

    def _build_template_group(self):
        group = QGroupBox("步骤 2：编辑提示词模板")
        v = QVBoxLayout()

        # 正向模板
        pos_ctrl = QHBoxLayout()
        pos_ctrl.addWidget(QLabel("正向模板 (交由 LLM 扩写):"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(200)
        self.template_combo.currentTextChanged.connect(self.load_template_content)
        pos_ctrl.addWidget(self.template_combo)
        self.save_template_btn = QPushButton("保存当前模板")
        self.save_template_btn.clicked.connect(self.save_current_template)
        pos_ctrl.addWidget(self.save_template_btn)
        self.save_as_template_btn = QPushButton("模板另存为...")
        self.save_as_template_btn.clicked.connect(self.save_as_new_template)
        pos_ctrl.addWidget(self.save_as_template_btn)
        v.addLayout(pos_ctrl)

        self.template_editor = QTextEdit()
        self.template_editor.setPlaceholderText("在这里编辑需要发给大模型进行细节扩写的【正向提示词】基础模板...")
        self.template_editor.setMaximumHeight(80)
        self.template_editor.textChanged.connect(self._on_control_changed)
        v.addWidget(self.template_editor)

        # 反向模板
        neg_ctrl = QHBoxLayout()
        neg_ctrl.addWidget(QLabel("反向模板 (直接发给 SD 引擎):"))
        self.neg_template_combo = QComboBox()
        self.neg_template_combo.setMinimumWidth(200)
        self.neg_template_combo.currentTextChanged.connect(self.load_negative_template_content)
        neg_ctrl.addWidget(self.neg_template_combo)
        self.save_neg_template_btn = QPushButton("保存反向模板")
        self.save_neg_template_btn.clicked.connect(self.save_current_negative_template)
        neg_ctrl.addWidget(self.save_neg_template_btn)
        self.save_as_neg_template_btn = QPushButton("反向模板另存...")
        self.save_as_neg_template_btn.clicked.connect(self.save_as_new_negative_template)
        neg_ctrl.addWidget(self.save_as_neg_template_btn)
        v.addLayout(neg_ctrl)

        self.neg_template_editor = QTextEdit()
        self.neg_template_editor.setPlaceholderText("在这里编辑【反向提示词】内容，这段文本不会经过大模型。")
        self.neg_template_editor.setMaximumHeight(80)
        self.neg_template_editor.textChanged.connect(self._on_control_changed)
        v.addWidget(self.neg_template_editor)

        group.setLayout(v)
        return group

    def _build_batch_params_group(self):
        group = QGroupBox("步骤 3：设置批量参数")
        h = QHBoxLayout()
        h.addWidget(QLabel("大模型请求轮数(Y):"))
        self.loop_count_input = QSpinBox()
        self.loop_count_input.setRange(1, 9999)
        self.loop_count_input.valueChanged.connect(self._on_control_changed)
        h.addWidget(self.loop_count_input)
        h.addWidget(QLabel("单次返回组数(X):"))
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 9999)
        self.count_input.valueChanged.connect(self._on_control_changed)
        h.addWidget(self.count_input)
        h.addStretch()
        group.setLayout(h)
        return group

    def _build_action_layout(self):
        h = QHBoxLayout()
        self.start_btn = QPushButton("步骤 4：保存配置并开始批量生成")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn = QPushButton("停止任务")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-weight: bold; background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        h.addWidget(self.start_btn)
        h.addWidget(self.stop_btn)
        return h

    # ---------------------------------------------------------- 配置同步
    def load_config_to_ui(self, config):
        self._config_ref = config
        self._building_ui = True
        try:
            self.theme_input.setText(str(config.get("last_used_theme", "中秋主题少女")))
            self.loop_count_input.setValue(int(config.get("loop_count", 1)))
            self.count_input.setValue(int(config.get("generate_count", 3)))
        finally:
            self._building_ui = False
        # 模板下拉刷新（依赖 config 中的 last_used_template）
        self.refresh_templates()
        self.refresh_negative_templates()
        self.update_styles()
        self._update_step_indicator()

    def update_config_from_ui(self, config):
        config["last_used_theme"] = self.theme_input.text().strip()
        config["last_used_template"] = self.template_combo.currentText()
        config["last_used_negative_template"] = self.neg_template_combo.currentText()
        selected_style_key = self.style_combo.currentText()
        config["last_used_style_key"] = selected_style_key
        config["last_used_style"] = self.style_options.get(selected_style_key, "")
        config["generate_count"] = self.count_input.value()
        config["loop_count"] = self.loop_count_input.value()

    def _on_control_changed(self, *args):
        if self._building_ui:
            return
        self._update_step_indicator()
        self.config_changed.emit()

    def _on_theme_edited(self, text):
        if self._syncing:
            return
        self._update_step_indicator()
        self.theme_edited.emit(text)
        self.config_changed.emit()

    def _on_style_changed(self, name):
        if self._syncing:
            return
        self.style_changed.emit(str(name or "").strip())

    # ---------------------------------------------------------- 跨 Tab 同步入口
    def set_theme_value(self, text):
        self._syncing = True
        try:
            self.theme_input.setText(text)
        finally:
            self._syncing = False
        self._update_step_indicator()

    def set_style_value(self, name):
        self._syncing = True
        try:
            self.style_combo.blockSignals(True)
            self.style_combo.setCurrentText(name)
            self.style_combo.blockSignals(False)
        finally:
            self._syncing = False

    # ---------------------------------------------------------- 风格下拉
    def _get_style_options(self):
        if callable(self.styles_getter):
            try:
                loaded = self.styles_getter() or {}
                if isinstance(loaded, dict) and loaded:
                    return dict(loaded)
            except Exception:
                pass

        styles_file = os.path.join(BASE_DIR, "conf", "config-styles.json")
        try:
            if os.path.exists(styles_file):
                with open(styles_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded:
                    return loaded
        except Exception as e:
            print(f"加载风格文件失败: {e}")
        return {"默认(无附加)": ""}

    def _get_current_style_name(self):
        if callable(self.current_style_name_getter):
            try:
                value = str(self.current_style_name_getter() or "").strip()
                if value:
                    return value
            except Exception:
                pass
        return "默认(无附加)"

    def _get_selected_style_prompt(self):
        selected_style_key = self.style_combo.currentText().strip()
        return style_prompt(self.style_options, selected_style_key)

    def load_style_options(self):
        self.update_styles()

    def update_styles(self, style_names=None, current_style_name=None):
        self.style_options = self._get_style_options()
        if style_names:
            ordered = []
            for name in style_names:
                key = str(name or "").strip()
                if key and key not in ordered:
                    ordered.append(key)
        else:
            ordered = list(self.style_options.keys())
        if "默认(无附加)" not in ordered:
            ordered.insert(0, "默认(无附加)")
            self.style_options.setdefault("默认(无附加)", "")

        selected_style_name = current_style_name or self._get_current_style_name()
        if selected_style_name not in ordered:
            selected_style_name = "默认(无附加)"

        self._syncing = True
        try:
            self.style_combo.blockSignals(True)
            self.style_combo.clear()
            self.style_combo.addItems(ordered)
            self.style_combo.setCurrentText(selected_style_name)
            self.style_combo.blockSignals(False)
        finally:
            self._syncing = False

    # ---------------------------------------------------------- 模板管理
    def refresh_templates(self):
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        templates = [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
        if templates:
            self.template_combo.addItems(templates)
            last_template = ""
            if hasattr(self, "_config_ref") and self._config_ref:
                last_template = self._config_ref.get("last_used_template", "")
            selected = last_template if last_template in templates else templates[0]
            self.template_combo.setCurrentText(selected)
            self.load_template_content(selected)
        else:
            self.template_combo.addItem("未找到 txt 文件")
            self.template_editor.clear()
        self.template_combo.blockSignals(False)

    def load_template_content(self, filename):
        if not filename or filename == "未找到 txt 文件":
            return
        filepath = os.path.join(PROMPTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.template_editor.setPlainText(f.read())

    def save_current_template(self):
        filename = self.template_combo.currentText()
        if not filename or filename == "未找到 txt 文件":
            QMessageBox.warning(self, "警告", "当前没有选中有效的模板文件。")
            return
        filepath = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.template_editor.toPlainText())
            QMessageBox.information(self, "成功", f"模板 '{filename}' 已保存！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def save_as_new_template(self):
        new_name, ok = QInputDialog.getText(self, "模板另存为", "请输入新模板名称 (无需输入 .txt 后缀):")
        if ok and new_name.strip():
            filename = f"{new_name.strip()}.txt"
            filepath = os.path.join(PROMPTS_DIR, filename)
            if os.path.exists(filepath):
                reply = QMessageBox.question(
                    self,
                    "确认覆盖",
                    f"文件 '{filename}' 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.template_editor.toPlainText())
                self.refresh_templates()
                self.template_combo.setCurrentText(filename)
                QMessageBox.information(self, "成功", f"新模板 '{filename}' 已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def refresh_negative_templates(self):
        self.neg_template_combo.blockSignals(True)
        self.neg_template_combo.clear()
        templates = [f for f in os.listdir(NEG_PROMPTS_DIR) if f.endswith(".txt")]
        if templates:
            self.neg_template_combo.addItems(templates)
            last_template = ""
            if hasattr(self, "_config_ref") and self._config_ref:
                last_template = self._config_ref.get("last_used_negative_template", "")
            selected = last_template if last_template in templates else templates[0]
            self.neg_template_combo.setCurrentText(selected)
            self.load_negative_template_content(selected)
        else:
            self.neg_template_combo.addItem("未找到 txt 文件")
            self.neg_template_editor.clear()
        self.neg_template_combo.blockSignals(False)

    def load_negative_template_content(self, filename):
        if not filename or filename == "未找到 txt 文件":
            return
        filepath = os.path.join(NEG_PROMPTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.neg_template_editor.setPlainText(f.read())

    def save_current_negative_template(self):
        filename = self.neg_template_combo.currentText()
        if not filename or filename == "未找到 txt 文件":
            QMessageBox.warning(self, "警告", "当前没有选中有效的反向模板文件。")
            return
        filepath = os.path.join(NEG_PROMPTS_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.neg_template_editor.toPlainText())
            QMessageBox.information(self, "成功", f"反向模板 '{filename}' 已保存！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def save_as_new_negative_template(self):
        new_name, ok = QInputDialog.getText(self, "反向模板另存为", "请输入新反向模板名称 (无需输入 .txt 后缀):")
        if ok and new_name.strip():
            filename = f"{new_name.strip()}.txt"
            filepath = os.path.join(NEG_PROMPTS_DIR, filename)
            if os.path.exists(filepath):
                reply = QMessageBox.question(
                    self,
                    "确认覆盖",
                    f"文件 '{filename}' 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.neg_template_editor.toPlainText())
                self.refresh_negative_templates()
                self.neg_template_combo.setCurrentText(filename)
                QMessageBox.information(self, "成功", f"新反向模板 '{filename}' 已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    # ---------------------------------------------------------- 运行态 / 步骤
    def set_running(self, running):
        self._worker_running = running
        if running:
            self._workflow_done = False
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.save_template_btn.setEnabled(not running)
        self.save_as_template_btn.setEnabled(not running)
        self.save_neg_template_btn.setEnabled(not running)
        self.save_as_neg_template_btn.setEnabled(not running)
        self._update_step_indicator()

    def mark_workflow_done(self):
        self._workflow_done = True
        self._worker_running = False
        self._update_step_indicator()

    def _update_step_indicator(self):
        theme_ok = bool(self.theme_input.text().strip())
        template_ok = bool(self.template_editor.toPlainText().strip())
        # 步骤 3 / 5 跟进步骤 2 / 4
        step_done = [
            theme_ok,
            template_ok,
            template_ok,  # 步骤 3：参数已设置（默认值即合法）
            self._workflow_done,
            self._workflow_done,  # 步骤 5：输出随步骤 4 完成
        ]
        # 找到第一个未完成步作为 current
        current_idx = -1
        for i, done in enumerate(step_done):
            if not done:
                current_idx = i
                break
        for i, lbl in enumerate(self._step_labels):
            if self._worker_running and i == 3:
                status = "in_progress"
            elif i == current_idx and not self._workflow_done:
                status = "in_progress"
            elif step_done[i]:
                status = "done"
            else:
                status = "pending"
            self._apply_step_label(lbl, i, status)

    def _apply_step_label(self, lbl, idx, status):
        title, desc = THEME_BATCH_STEPS[idx]
        if status == "done":
            icon = "[√]"
            tag = "已完成"
            color = "#2e7d32"
            weight = "normal"
        elif status == "in_progress":
            icon = "[→]"
            tag = "进行中"
            color = "#1565c0"
            weight = "bold"
        else:
            icon = "[ ]"
            tag = "未开始"
            color = "#9e9e9e"
            weight = "normal"
        lbl.setText(
            f"<span style='color:{color}; font-weight:{weight};'>"
            f"{icon} <b>{title}</b>　<span style='font-size:small;'>{desc}</span>　[{tag}]"
            f"</span>"
        )

    # ---------------------------------------------------------- autocomplete
    def setup_autocomplete(self):
        if self.tag_manager is not None:
            self.tag_manager.setup_text_edit(self.template_editor)
