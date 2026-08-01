# -*- coding: utf-8 -*-
"""风格参考图模式选择器（跨 Tab 共享）。

- 4 种模式：关闭 / 头部插入 / 参考优先 / 图文交错（定义见 utils.styles）
- 模式选择持久化到 conf/config-image.json 顶层键 "style_ref_mode"（全局共享）
- 参考图不可用（未配置或文件不存在）时，只允许选择「关闭」

用法（各 Tab）：
    self.style_ref_mode_combo = StyleRefModeCombo(self)          # 建在画风下拉旁边
    # 样式列表加载/切换时校验参考图存在性：
    self.style_ref_mode_combo.set_modes_available(has_ref)
    # 生图时取最终生效模式：
    mode = self.style_ref_mode_combo.effective_mode(has_ref)
"""
import json
import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox

from utils.styles import (
    STYLE_REF_MODES, MODE_OFF, MODE_HEAD, MODE_PRIORITY, MODE_INTERLEAVE,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_IMAGE_FILE = os.path.join(PROJECT_ROOT, "conf", "config-image.json")
STYLE_REF_MODE_KEY = "style_ref_mode"


def load_saved_style_ref_mode(config_file=CONFIG_IMAGE_FILE):
    """从配置文件读取用户上次选择的模式；缺失或非法时返回关闭。"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        mode = str(cfg.get(STYLE_REF_MODE_KEY, "") or "").strip()
        if mode in dict(STYLE_REF_MODES):
            return mode
    except Exception:
        pass
    return MODE_OFF


def save_style_ref_mode(mode, config_file=CONFIG_IMAGE_FILE):
    """把模式写入配置文件（合并保留其他键）。"""
    if mode not in dict(STYLE_REF_MODES):
        mode = MODE_OFF
    try:
        cfg = {}
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg[STYLE_REF_MODE_KEY] = mode
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


class StyleRefModeCombo(QComboBox):
    """4 种模式下拉。参考图不可用时自动禁用除「关闭」外的选项并回退到「关闭」。"""

    mode_changed = pyqtSignal(str)

    def __init__(self, parent=None, config_file=CONFIG_IMAGE_FILE):
        super().__init__(parent)
        self.config_file = config_file
        self.setToolTip("画风参考图模式：参考图仅提取画风，不影响主体与构图（样式参考图为空时不可用）")
        for mode_key, mode_label in STYLE_REF_MODES:
            self.addItem(mode_label, mode_key)
        self.set_mode(load_saved_style_ref_mode(config_file), has_ref=True)
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_index_changed(self):
        mode = self.currentData() or MODE_OFF
        save_style_ref_mode(mode, self.config_file)
        self.mode_changed.emit(mode)

    def set_mode(self, mode, has_ref=True):
        """设置当前模式；has_ref=False 时强制回退到关闭并禁用其余选项。"""
        idx = self.findData(mode)
        self.blockSignals(True)
        self.setCurrentIndex(idx if idx >= 0 else self.findData(MODE_OFF))
        self._apply_availability(bool(has_ref))
        self.blockSignals(False)

    def set_modes_available(self, has_ref):
        """样式列表加载/切换时调用：按参考图是否存在启用/禁用模式项。"""
        self.blockSignals(True)
        self._apply_availability(bool(has_ref))
        self.blockSignals(False)

    def _apply_availability(self, has_ref):
        for i in range(self.count()):
            enabled = has_ref or self.itemData(i) == MODE_OFF
            self.model().item(i).setEnabled(enabled)
        if not has_ref and (self.currentData() or MODE_OFF) != MODE_OFF:
            self.setCurrentIndex(self.findData(MODE_OFF))

    def selected_mode(self):
        """当前下拉选择的模式（可能为参考图不可用时的关闭）。"""
        return self.currentData() or MODE_OFF

    def effective_mode(self, has_ref):
        """生图时实际生效的模式：参考图不可用或为关闭时返回关闭。"""
        mode = self.selected_mode()
        if not has_ref or mode == MODE_OFF:
            return MODE_OFF
        return mode
