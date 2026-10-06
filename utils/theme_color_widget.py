"""Compact shared manual colour selector; explicit, asynchronous recommendation."""
import json
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton,
                            QDialog, QFormLayout, QLineEdit, QCheckBox, QTextEdit,
                            QDialogButtonBox, QMessageBox, QSizePolicy)
from utils.atomic_io import write_json_atomic
from utils.theme_color import BASE, profiles, direction_catalog, freeze_theme_color, recommend_theme


class RecommendationWorker(QThread):
    result = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, description, parent):
        super().__init__(parent)
        self.description = description

    def run(self):
        try:
            self.result.emit(recommend_theme(self.description))
        except Exception as exc:
            self.failed.emit(str(exc))


class ThemeColorSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None, *, scope="direct", context_getter=None):
        super().__init__(parent)
        self.scope = scope
        self.context_getter = context_getter or (lambda: "")
        self._region_selection = {}
        self._worker = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("配色:"))
        self.combo = QComboBox()
        self.combo.addItem("关闭", "")
        for profile in profiles():
            self.combo.addItem(profile["label"] + "（试用）", profile["id"])
        self.combo.setMaximumWidth(160)
        self.combo.setToolTip("主题色设计，只用于无参考图文字首图。配色方向可控性仍有限，不保证精确色值或后续重绘保留。")
        row.addWidget(self.combo)
        self.details_button = QPushButton("规则")
        self.details_button.setMaximumWidth(46)
        self.details_button.setToolTip("配色落点、独立色调和宽松主次关系；配色关闭时仍可单独选色调。")
        self.details_button.clicked.connect(self.details)
        row.addWidget(self.details_button)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._restore()
        self.combo.currentIndexChanged.connect(self._manual_changed)

    def selection(self):
        return {**self._region_selection, "profile_id": self.combo.currentData() or ""}

    def snapshot(self):
        return freeze_theme_color(self.selection())

    def is_active(self):
        return bool(self.combo.currentData() or self._region_selection.get("tone"))

    def _restore(self):
        path = BASE / "conf/config.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            selection = (data.get("theme_color_ui") or {}).get(self.scope) or {}
            self._region_selection = {k: v for k, v in selection.items() if k != "profile_id"}
            idx = self.combo.findData(selection.get("profile_id", ""))
            self.combo.setCurrentIndex(max(idx, 0))
        except (OSError, ValueError, AttributeError):
            self._region_selection = {}

    def _persist(self):
        path = BASE / "conf/config.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            state = dict(data.get("theme_color_ui") or {})
            state[self.scope] = self.selection()
            data["theme_color_ui"] = state
            write_json_atomic(str(path), data)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "配色记忆未保存", str(exc))

    def _manual_changed(self, *_args):
        self._region_selection.pop("recommendation", None)
        self._persist()
        self.changed.emit()

    def details(self):
        if self._worker and self._worker.isRunning():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("配色规则：色调、主次与落点")
        form = QFormLayout(dialog)
        environment = QLineEdit(self._region_selection.get("environment", ""))
        environment.setPlaceholderText("复制正文短语，例如 sky, river water, distant foliage")
        accents = QLineEdit(self._region_selection.get("accent_regions", ""))
        accents.setPlaceholderText("复制正文短语，例如 waist ribbon sash, shoe bows")
        confirmed = QCheckBox("这些区域已在正文中存在，并允许修改固有色")
        confirmed.setChecked(bool(self._region_selection.get("confirmed")))
        catalog = direction_catalog()
        tone, area = QComboBox(), QComboBox()
        tone.addItem("未指定（保留原色调）", "")
        area.addItem("沿用方案（不新增面积规则）", "")
        for combo, group, key in ((tone, "tones", "tone"), (area, "areas", "area")):
            for entry in catalog[group]:
                combo.addItem(entry["label"], entry["id"])
            combo.setCurrentIndex(max(0, combo.findData(self._region_selection.get(key, ""))))
        auxiliary = QLineEdit(self._region_selection.get("auxiliary_regions", ""))
        auxiliary.setPlaceholderText("三层主次时复制正文已有物体短语，与另两组落点分开")
        form.addRow("独立色调", tone)
        form.addRow("宽松主次", area)
        form.addRow("主色落点", environment)
        form.addRow("第二色／强调落点", accents)
        form.addRow("辅助色落点", auxiliary)
        form.addRow(confirmed)
        palette_active = bool(self.combo.currentData())
        for control in (area, environment, accents, confirmed):
            control.setEnabled(palette_active)
        def update_auxiliary(*_args):
            auxiliary.setEnabled(palette_active and area.currentData() == "three-level")
        area.currentIndexChanged.connect(update_auxiliary)
        update_auxiliary()
        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText("未点名的固有配色、人数、构图和对象保持。\n色调可独立选择；配色关闭时只温和调节原颜色的明度/彩度。主次需要开启配色并指定已有落点，没有通用百分比，不扩大物体来凑面积。辅助色为当前试用方案明确指定的设计色，不是根据书中五色顺序推断。\n三个组合来源于 P1/P2/P3 试验，色板为设计近似值；色调依据 PDF 14–16/110/129，主次依据 18/146/79。新增规则尚未在线验证。\n不能用于源图编辑、带图参考、重绘或后处理。\n\n" + json.dumps(profiles(), ensure_ascii=False, indent=2))
        form.addRow(info)
        recommend = QPushButton("请求一次 LLM 推荐（只发送当前正文）")
        form.addRow(recommend)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        state = {"recommendation": None, "description": ""}
        preview = QPushButton("预览当前生成条款（不联网）")
        form.insertRow(form.rowCount() - 2, preview)

        def selected_fields():
            return {"environment": environment.text().strip(), "accent_regions": accents.text().strip(),
                    "confirmed": confirmed.isChecked(), "tone": tone.currentData(),
                    "area": area.currentData(), "auxiliary_regions": auxiliary.text().strip()}

        def preview_rules():
            try:
                snapshot = freeze_theme_color({"profile_id": self.combo.currentData() or "", **selected_fields()})
                info.append("\n当前条款：\n" + (snapshot.get("prompt", "配色关闭，不附加条款")))
            except ValueError as exc:
                info.append(str(exc))

        preview.clicked.connect(preview_rules)

        def request():
            description = str(self.context_getter() or "").strip()
            if not description:
                info.append("请先填写主题正文。")
                return
            state["description"] = description
            recommend.setEnabled(False)
            buttons.setEnabled(False)
            self._worker = RecommendationWorker(description, self)
            self._worker.result.connect(lambda result: receive(result))
            self._worker.failed.connect(lambda error: info.append("推荐失败，保留手动选择：" + error))
            self._worker.finished.connect(lambda: finished())
            self._worker.start()

        def receive(result):
            state["recommendation"] = result
            info.append("推荐：" + result["profile_id"] + "\n理由：" + result["reason"] + "\n点击确定应用建议，取消保留原选择。")

        def finished():
            recommend.setEnabled(True)
            buttons.setEnabled(True)
            self._worker.deleteLater()
            self._worker = None

        recommend.clicked.connect(request)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._region_selection = selected_fields()
            if state["recommendation"] and str(self.context_getter() or "").strip() == state["description"]:
                result = state["recommendation"]
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(self.combo.findData("" if result["profile_id"] == "off" else result["profile_id"]))
                self.combo.blockSignals(False)
                self._region_selection["recommendation"] = result
            self._persist()
            self.changed.emit()
