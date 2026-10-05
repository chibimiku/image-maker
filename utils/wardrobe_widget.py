"""Compact reusable selector; persistence belongs to its owning Tab."""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QSizePolicy
from PyQt6.QtCore import pyqtSignal

from utils.wardrobe import POLICIES, wardrobe_presets, build_wardrobe_spec


class WardrobeSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("穿衣风格:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("沿用原衣装", "")
        for name, entry in wardrobe_presets().items():
            self.preset_combo.addItem(entry["label"], name)
        self.preset_combo.setMaximumWidth(140)
        self.preset_combo.setToolTip("仅影响生成时的服装设计，可与绘画风格组合；不修改源图分析。")
        row.addWidget(self.preset_combo)
        self.policy_combo = QComboBox()
        for name, label in POLICIES:
            self.policy_combo.addItem(label, name)
        self.policy_combo.setMaximumWidth(90)
        self.policy_combo.setToolTip("完整转化：允许替换原服装类别与剪裁，生成完整目标衣装。\n改款：保留原类别并调整设计。只补全：已有服装不替换。\n明确保留的服装、颜色和配饰优先；绘画风格只控制画法。")
        row.addWidget(self.policy_combo)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.preset_combo.currentIndexChanged.connect(self._changed)
        self.policy_combo.currentIndexChanged.connect(self._changed)
        self._changed()

    def _changed(self, *_args):
        self.policy_combo.setEnabled(bool(self.preset_combo.currentData()))
        self.changed.emit()

    def selection(self):
        return {"name": self.preset_combo.currentData() or "",
                "policy": self.policy_combo.currentData() or "replace"}

    def restore(self, selection):
        selection = selection or {}
        for combo, key, fallback in ((self.preset_combo, "name", ""),
                                     (self.policy_combo, "policy", "replace")):
            idx = combo.findData(selection.get(key, fallback))
            combo.setCurrentIndex(idx if idx >= 0 else 0)

    def snapshot(self):
        return build_wardrobe_spec(**self.selection())
