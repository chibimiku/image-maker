"""Compact selector with book palettes and reviewable automatic rules."""
import json
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QDialog, QFormLayout, QLineEdit, QCheckBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialogButtonBox, QMessageBox, QSizePolicy)
from utils.atomic_io import write_json_atomic
from utils.theme_color import BASE, profiles, direction_catalog, freeze_theme_color, apply_theme_color
from utils.theme_color_rules import generate_theme_rules


class RecommendationWorker(QThread):
    result = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, description, parent=None, *, profile_id=""):
        super().__init__(parent)
        self.description, self.profile_id = description, profile_id

    def run(self):
        try:
            self.result.emit(generate_theme_rules(self.description, self.profile_id))
        except Exception as exc:
            self.failed.emit(str(exc))


def populate_palettes(combo):
    combo.addItem("关闭", "")
    for book in (True, False):
        combo.insertSeparator(combo.count())
        for p in profiles():
            if (p.get("origin") == "book_case") != book:
                continue
            label = p["label"]
            tooltip = f"书籍案例 · {label}（PDF {p['source']['pdf_page']}）" if book else "实验预设 · " + label
            combo.addItem(label, p["id"])
            combo.setItemData(combo.count()-1, tooltip, Qt.ItemDataRole.ToolTipRole)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    # Keep the popup readable even when the surrounding toolbar is crowded.
    text_width = max(combo.fontMetrics().horizontalAdvance(combo.itemText(i))
                     for i in range(combo.count()))
    combo.view().setMinimumWidth(text_width + 48)
    combo.setMaxVisibleItems(16)


class ColorRulesDialog(QDialog):
    def __init__(self, selector):
        super().__init__(selector)
        self.selector, self.worker, self.auto_description = selector, None, None
        self.setWindowTitle("配色规则 · 自动生成后按需微调")
        self.resize(680, 600)
        form = QFormLayout(self)
        self.palette = QComboBox()
        populate_palettes(self.palette)
        self.palette.setCurrentIndex(max(0, self.palette.findData(selector.combo.currentData() or "")))
        form.addRow("方案", self.palette)
        self.auto_button = QPushButton("自动生成规则（根据当前正文，选定色板不变）")
        self.auto_button.setObjectName("auto_color_rules")
        self.auto_button.setToolTip("点击才请求当前文本API；成功建议缓存。生成已有落点、色调和主次，不生成图片。")
        form.addRow(self.auto_button)
        self.source = QLabel()
        self.source.setWordWrap(True)
        form.addRow(self.source)
        self.tone, self.area = QComboBox(), QComboBox()
        for combo, group, default in ((self.tone, "tones", "未指定（保留原色调）"),
                                      (self.area, "areas", "未指定（不强加主次）")):
            combo.addItem(default, "")
            for entry in direction_catalog()[group]:
                combo.addItem(entry["label"], entry["id"])
        form.addRow("色调", self.tone)
        form.addRow("宽松主次", self.area)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("book_color_bindings")
        self.table.setHorizontalHeaderLabels(["书籍颜色 / 来源", "本次角色", "已有区域（自动填写，可改）"])
        self.table.verticalHeader().hide()
        for col in (0, 1):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(195)
        form.addRow(self.table)
        self.trial_fields = QWidget()
        trial_form = QFormLayout(self.trial_fields)
        trial_form.setContentsMargins(0, 0, 0, 0)
        self.environment, self.accents, self.auxiliary = QLineEdit(), QLineEdit(), QLineEdit()
        for title, field in (("主色落点", self.environment), ("强调落点", self.accents), ("辅助落点", self.auxiliary)):
            field.setPlaceholderText("点自动生成，或复制正文已有区域短语")
            trial_form.addRow(title, field)
        form.addRow(self.trial_fields)
        self.confirmed = QCheckBox("允许将所选色板应用到上面已有区域，其他固有色保持")
        form.addRow(self.confirmed)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(95)
        self.info.setPlainText("选色板后点自动生成；未选方案时自动从31套书籍案例选择。\n建议不会添加物件，不要求精确面积。未注明的角色属于本次建议，不是书中事实。\n仅用于无参考图、无重绘/后处理的文字首图。应用规则不会立即生图。")
        form.addRow(self.info)
        preview = QPushButton("预览条款（不联网）")
        preview.clicked.connect(self.preview)
        form.addRow(preview)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用规则")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.palette.currentIndexChanged.connect(self.update_palette)
        self.area.currentIndexChanged.connect(self.update_area)
        self.auto_button.clicked.connect(self.request_rules)
        self.update_palette()
        self.set_fields(selector._region_selection)

    def profile(self):
        return next((p for p in profiles() if p["id"] == self.palette.currentData()), {})

    def update_palette(self, *_args):
        p = self.profile()
        book = p.get("origin") == "book_case"
        self.table.setVisible(book)
        self.trial_fields.setVisible(bool(p) and not book)
        self.confirmed.setEnabled(bool(p))
        self.area.setEnabled(bool(p))
        self.table.setRowCount(0)
        if book:
            self.table.setRowCount(len(p["colours"]))
            sources = {"scan_measured": "扫描实测", "ncd_nominal": "标称值", "name_only": "仅色名"}
            for row, c in enumerate(p["colours"]):
                item = QTableWidgetItem(c["name"] + " · " + sources[c["hex_source"]])
                item.setData(Qt.ItemDataRole.UserRole, c["order"])
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if c.get("hex"):
                    item.setBackground(QColor(c["hex"]))
                    item.setForeground(QColor("white" if QColor(c["hex"]).lightness() < 125 else "black"))
                item.setToolTip("色值：" + (c.get("hex") or "未测量") + "；不保证精确匹配")
                self.table.setItem(row, 0, item)
                role = QComboBox()
                for label, key in (("未指定", ""), ("主色", "base"), ("辅助", "auxiliary"), ("强调", "accent")):
                    role.addItem(label, key)
                self.table.setCellWidget(row, 1, role)
                self.table.setCellWidget(row, 2, QLineEdit())
            self.source.setText(f"原书案例：{p['label']} · {p['artist']} · PDF {p['source']['pdf_page']}页\n五色保留原目录；未用颜色不强塞到图中。方案尚未实图验证。")
        else:
            self.source.setText("实验预设（非书籍案例）" if p else "配色关闭时可单选色调；自动生成会推荐书籍方案。")
        self.auto_description = None
        self.update_area()

    def update_area(self, *_args):
        self.auxiliary.setEnabled(bool(self.profile()) and self.area.currentData() == "three-level")

    def set_fields(self, fields):
        for combo, key in ((self.tone, "tone"), (self.area, "area")):
            combo.setCurrentIndex(max(0, combo.findData(fields.get(key, ""))))
        for widget, key in ((self.environment, "environment"), (self.accents, "accent_regions"), (self.auxiliary, "auxiliary_regions")):
            widget.setText(fields.get(key, ""))
        bindings = {b["colour_order"]: b for b in fields.get("bindings", [])}
        for row in range(self.table.rowCount()):
            b = bindings.get(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole), {})
            role = self.table.cellWidget(row, 1)
            role.setCurrentIndex(max(0, role.findData(b.get("role", ""))))
            self.table.cellWidget(row, 2).setText(b.get("regions", ""))
        self.confirmed.setChecked(bool(fields.get("confirmed")))

    def selection(self):
        fields = {"profile_id": self.palette.currentData() or "", "tone": self.tone.currentData(),
                  "area": self.area.currentData(), "confirmed": self.confirmed.isChecked()}
        if self.profile().get("origin") == "book_case":
            fields["bindings"] = [{"colour_order": self.table.item(r, 0).data(Qt.ItemDataRole.UserRole),
                "regions": self.table.cellWidget(r, 2).text().strip(), "role": self.table.cellWidget(r, 1).currentData()}
                for r in range(self.table.rowCount()) if self.table.cellWidget(r, 2).text().strip()]
        else:
            fields.update(environment=self.environment.text().strip(), accent_regions=self.accents.text().strip(),
                          auxiliary_regions=self.auxiliary.text().strip())
        return fields

    def preview(self):
        try:
            self.info.setPlainText(freeze_theme_color(self.selection()).get("prompt", "配色关闭"))
        except ValueError as exc:
            self.info.setPlainText(str(exc))

    def request_rules(self):
        if self.worker is not None:
            return
        description = str(self.selector.context_getter() or "").strip()
        if not description:
            self.info.setPlainText("请先填写生图正文，再生成规则。")
            return
        for widget in (self.palette, self.auto_button, self.buttons):
            widget.setEnabled(False)
        self.info.setPlainText("正在根据当前正文生成规则…")
        self.worker = RecommendationWorker(description, self, profile_id=self.palette.currentData() or "")
        self.worker.result.connect(lambda result: self.receive_rules(result, description))
        self.worker.failed.connect(lambda error: self.info.setPlainText("自动生成失败，原手动规则保留：" + error))
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def receive_rules(self, result, description):
        if str(self.selector.context_getter() or "").strip() != description:
            self.info.setPlainText("正文已改变，未应用旧建议。请重新自动生成。")
            return
        self.palette.setCurrentIndex(self.palette.findData(result["profile_id"]))
        self.set_fields(result)
        self.auto_description = description
        self.info.setPlainText("已填写规则，请查看颜色落点后应用。\n" + result["reason"])

    def worker_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        for widget in (self.palette, self.auto_button, self.buttons):
            widget.setEnabled(True)

    def done(self, code):
        if self.worker is None:
            super().done(code)

    def accept(self):
        try:
            description = str(self.selector.context_getter() or "").strip()
            if self.auto_description is not None and description != self.auto_description:
                raise ValueError("正文已改变，请重新生成或调整规则")
            plan = freeze_theme_color(self.selection())
            if description:
                apply_theme_color(description, plan)
        except ValueError as exc:
            self.info.setPlainText(str(exc))
            return
        super().accept()


class ThemeColorSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None, *, scope="direct", context_getter=None):
        super().__init__(parent)
        self.scope, self.context_getter = scope, context_getter or (lambda: "")
        self._region_selection, self._worker = {}, None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("配色:"))
        self.combo = QComboBox()
        populate_palettes(self.combo)
        self.combo.setToolTip("31套书籍案例与3个实验预设。仅用于无参考图文字首图，规则中可自动填写落点。")
        row.addWidget(self.combo)
        self.details_button = QPushButton("规则")
        self.details_button.setMaximumWidth(46)
        self.details_button.setToolTip("自动生成落点、色调和主次建议，也可手动调整")
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
        try:
            data = json.loads((BASE / "conf/config.json").read_text(encoding="utf-8"))
            selection = (data.get("theme_color_ui") or {}).get(self.scope) or {}
            self._region_selection = {k: v for k, v in selection.items() if k != "profile_id"}
            self.combo.setCurrentIndex(max(0, self.combo.findData(selection.get("profile_id", ""))))
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
        self._region_selection.pop("bindings", None)
        self._persist()
        self.changed.emit()

    def details(self):
        dialog = ColorRulesDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            fields = dialog.selection()
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(max(0, self.combo.findData(fields.pop("profile_id"))))
            self.combo.blockSignals(False)
            self._region_selection = fields
            self._persist()
            self.changed.emit()
        dialog.deleteLater()
