"""One compact button; optional choices are edited without network calls."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QPushButton, QDialog, QVBoxLayout, QFormLayout, QComboBox, QLabel, QDialogButtonBox
from utils.refine_atmosphere import catalog, freeze


class RefineAtmosphereSelector(QPushButton):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("情绪/季节", parent)
        self.setFixedWidth(92)
        self._selection = {"mood": "", "season": ""}
        self.setToolTip("可选情绪和季节，只加入精修后的生成提示词；默认关闭，不改变固定色板。")
        self.clicked.connect(self._edit)

    def selection(self):
        return dict(self._selection)

    def snapshot(self):
        return freeze(self._selection)

    def restore(self, value):
        value = dict(value or {})
        available = catalog()
        self._selection = {key: value.get(key, "") if value.get(key, "") in
                           {"", *(e["id"] for e in available[group])} else ""
                           for key, group in (("mood", "moods"), ("season", "seasons"))}
        self.setText("氛围 ✓" if any(self._selection.values()) else "情绪/季节")

    def _edit(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("精修生成氛围")
        layout = QVBoxLayout(dialog)
        label = QLabel("可分别关闭。只表达色彩氛围，不添加季节物件、改变人物表情或覆盖手选色板。")
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        combos = {}
        entries = catalog()
        for key, group, name in (("mood", "moods", "情绪"), ("season", "seasons", "季节")):
            combo = QComboBox()
            combo.addItem("关闭", "")
            for entry in entries[group]:
                combo.addItem(entry["label"], entry["id"])
            combo.setCurrentIndex(max(0, combo.findData(self._selection[key])))
            form.addRow(name, combo)
            combos[key] = combo
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.restore({key: combo.currentData() for key, combo in combos.items()})
            self.changed.emit()
